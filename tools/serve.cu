// F2K_CUDA persistent inference worker (v2).
//
// Loads the models ONCE and serves generation jobs over a localhost TCP socket
// with a newline-delimited JSON protocol, so each image costs only the ~8s of
// real work instead of the ~16s model load. Qwen3 + tokenizer are resolution-
// independent and stay resident forever; the transformer + VAE (which depend on
// res/precision) are cached and rebuilt only when those change (~6s, since Qwen
// is not reloaded).
//
//   build/serve [--port 8765]
//
// Request  (one JSON line): {"prompt":"...","res":1024,"precision":"fp8",
//                            "steps":4,"seed":777,"out":"/abs/path.png"}
// Response (one JSON line): {"ok":true,"elapsed":8.1,"timing":["...", ...]}
//                       or  {"ok":false,"error":"..."}

#include "backend/cuda/flux_transformer.h"
#include "backend/cuda/qwen_encoder.h"
#include "backend/cuda/sampler.h"
#include "backend/cuda/vae_decoder.h"
#include "backend/cuda/vae_encoder.h"
#include "backend/cuda/kernels/patchify.h"
#include "common/bpe_tokenizer.h"
#include "common/f2k_format.h"
#include "common/f2k_model_loader.h"
#include "common/tensor_router.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STBIR_NO_SIMD          // arm_neon.h intrinsics don't compile under nvcc
#include "stb_image_resize2.h"

#include <arpa/inet.h>
#include <csignal>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;
static inline __nv_bfloat16 f2b(float v) { return __float2bfloat16(v); }
static inline float         b2f(__nv_bfloat16 v) { return __bfloat162float(v); }
static double since(Clock::time_point t){ return std::chrono::duration<double>(Clock::now()-t).count(); }

namespace {
constexpr int PATCH=2, SEQ_TXT=512, IN_CH=128, T5_DIM=12288, TIME_DIM=256;
constexpr int N_HEADS=32, HEAD_DIM=128, FFN_DIM=12288, N_DOUBLE=8, N_SINGLE=24;
const std::string HOME_DIR = f2k::platform::home_dir();   // $HOME | %USERPROFILE%
const char* HOME = HOME_DIR.c_str();

bool write_png(const std::string& path, const float* rgb_chw, int H, int W){
    std::vector<uint8_t> b((size_t)H*W*3);
    for(int h=0;h<H;++h)for(int w=0;w<W;++w)for(int c=0;c<3;++c){
        float v=rgb_chw[(size_t)(c*H+h)*W+w]; v=std::min(1.f,std::max(0.f,(v+1.f)*0.5f));
        b[((size_t)h*W+w)*3+c]=(uint8_t)(v*255.f+0.5f);
    }
    return stbi_write_png(path.c_str(), W, H, 3, b.data(), W*3)!=0;
}

// Base64 (standard alphabet) for streaming preview PNGs inline.
std::string b64encode(const uint8_t* d, size_t n){
    static const char* T="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string o; o.reserve((n+2)/3*4);
    size_t i=0;
    for(; i+3<=n; i+=3){ uint32_t v=(d[i]<<16)|(d[i+1]<<8)|d[i+2];
        o+=T[(v>>18)&63]; o+=T[(v>>12)&63]; o+=T[(v>>6)&63]; o+=T[v&63]; }
    if(n-i==1){ uint32_t v=d[i]<<16; o+=T[(v>>18)&63]; o+=T[(v>>12)&63]; o+="=="; }
    else if(n-i==2){ uint32_t v=(d[i]<<16)|(d[i+1]<<8);
        o+=T[(v>>18)&63]; o+=T[(v>>12)&63]; o+=T[(v>>6)&63]; o+="="; }
    return o;
}
void png_collect(void* ctx, void* data, int size){
    auto* v=static_cast<std::vector<uint8_t>*>(ctx);
    auto* p=static_cast<uint8_t*>(data); v->insert(v->end(), p, p+size);
}

// Load an image, resize to res×res, return BF16 [3,res,res] in model space
// [-1,1] (the inverse of write_png's (v+1)/2 encode). Errors → false.
bool load_image_bf16(const std::string& path, int res,
                     std::vector<__nv_bfloat16>& out, std::string& err){
    int w=0,h=0,n=0;
    uint8_t* px=stbi_load(path.c_str(),&w,&h,&n,3);   // force RGB
    if(!px){ err="load "+path+": "+stbi_failure_reason(); return false; }
    std::vector<uint8_t> rgb;
    const uint8_t* src=px;
    if(w!=res || h!=res){
        rgb.resize((size_t)res*res*3);
        if(!stbir_resize_uint8_linear(px,w,h,0, rgb.data(),res,res,0, STBIR_RGB)){
            stbi_image_free(px); err="resize failed"; return false; }
        src=rgb.data();
    }
    out.resize((size_t)3*res*res);
    for(int y=0;y<res;++y)for(int x=0;x<res;++x)for(int c=0;c<3;++c){
        float v=src[((size_t)y*res+x)*3+c]/255.f*2.f-1.f;     // [0,255]→[-1,1]
        out[((size_t)c*res+y)*res+x]=f2b(v);                  // → CHW
    }
    stbi_image_free(px);
    return true;
}

// Load a grayscale inpaint mask (white=regenerate), resize to res, and average-
// pool to a per-latent-token weight m[SEQ_IMG] in [0,1]. The token grid is
// H_P×W_P; each token covers a (res/H_P)×(res/W_P) image-pixel block — matching
// patchify's token order (token r = hp*W_P + wp).
bool load_mask_tokens(const std::string& path, int res, int H_P, int W_P,
                      std::vector<float>& m, std::string& err){
    int w=0,h=0,n=0;
    uint8_t* px=stbi_load(path.c_str(),&w,&h,&n,1);   // force grayscale
    if(!px){ err="load mask "+path+": "+stbi_failure_reason(); return false; }
    std::vector<uint8_t> g;
    const uint8_t* src=px;
    if(w!=res || h!=res){ g.resize((size_t)res*res);
        if(!stbir_resize_uint8_linear(px,w,h,0, g.data(),res,res,0, STBIR_1CHANNEL)){
            stbi_image_free(px); err="mask resize failed"; return false; }
        src=g.data(); }
    const int bh=res/H_P, bw=res/W_P;
    m.assign((size_t)H_P*W_P, 0.f);
    for(int pr=0;pr<H_P;++pr)for(int pc=0;pc<W_P;++pc){
        double s=0; int cnt=0;
        for(int yy=pr*bh; yy<(pr+1)*bh && yy<res; ++yy)
            for(int xx=pc*bw; xx<(pc+1)*bw && xx<res; ++xx){ s+=src[(size_t)yy*res+xx]; ++cnt; }
        m[(size_t)pr*W_P+pc] = cnt ? (float)(s/cnt/255.0) : 0.f;
    }
    stbi_image_free(px);
    return true;
}

// Resolution/precision-dependent resident pipeline (transformer + VAE + buffers).
struct Pipeline {
    int res=0; std::string precision;
    std::unique_ptr<f2k::F2KModelLoader> ld;
    std::unique_ptr<f2k::TensorRouter> router;
    std::unique_ptr<f2k::cuda::FluxTransformer> model;
    std::unique_ptr<f2k::F2KReader> vae_r;
    std::unique_ptr<f2k::cuda::VAEDecoder> vae;
    std::unique_ptr<f2k::cuda::VAEEncoder> venc;   // img2img: image → latent
    int H_LAT=0,W_LAT=0,SEQ_IMG=0,H_P=0,W_P=0;
    size_t latent_elems=0, token_elems=0, pixel_elems=0, moment_elems=0;
    void *d_latent=nullptr,*d_tokens=nullptr,*d_velocity=nullptr,*d_pixels=nullptr;
    void *d_t_ws=nullptr,*d_v_ws=nullptr;
    void *d_init_pix=nullptr,*d_moments=nullptr,*d_e_ws=nullptr;   // encoder I/O + ws
    std::vector<float> bn_mean, bn_std;   // [IN_CH]

    ~Pipeline(){ for(void* p:{d_latent,d_tokens,d_velocity,d_pixels,d_t_ws,d_v_ws,
                              d_init_pix,d_moments,d_e_ws}) if(p) cudaFree(p); }
};

// Resident, resolution-independent state.
struct Worker {
    std::unique_ptr<f2k::F2KModelLoader> qwen_ld;
    std::unique_ptr<f2k::cuda::QwenEncoder> qwen;
    f2k::BpeTokenizer tok;
    void *d_txt=nullptr,*d_temb=nullptr; int32_t* d_ids=nullptr; void* d_q_ws=nullptr;
    std::unique_ptr<Pipeline> pipe;   // current cached (res,precision)

    bool init(std::string& err){
        // Qwen3 encoder (4 shards) — load once, keep forever.
        qwen_ld=std::make_unique<f2k::F2KModelLoader>();
        for(int i=1;i<=4;++i){ char p[256];
            std::snprintf(p,sizeof(p),"%s/models/flux2-klein-9B/qwen3_f2k/shard-%05d.f2k1",HOME,i);
            if(!qwen_ld->add_shard(p)){ err="qwen loader: "+qwen_ld->last_error(); return false; } }
        f2k::cuda::QwenEncoder::Config qc{}; qc.seq=SEQ_TXT; qc.loader=qwen_ld.get();
        qc.capture_layers={8,17,26};
        auto t=Clock::now(); qwen=std::make_unique<f2k::cuda::QwenEncoder>(qc);
        if(!qwen->ok()){ err=std::string("qwen ctor: ")+qwen->last_error(); return false; }
        std::fprintf(stderr,"[worker] Qwen3 ready in %.1fs\n",since(t));
        if(!tok.load(std::string(HOME)+"/models/flux2-klein-9B/tokenizer")){
            err="tokenizer: "+tok.error(); return false; }
        cudaMalloc(&d_txt, (size_t)SEQ_TXT*T5_DIM*2);
        cudaMalloc(&d_temb, TIME_DIM*2);
        cudaMalloc(&d_ids, SEQ_TXT*sizeof(int32_t));
        cudaMalloc(&d_q_ws, qwen->workspace_size_bytes());
        return true;
    }

    // Build (or rebuild) the transformer+VAE pipeline for (res, precision).
    bool ensure(int res, const std::string& precision, std::string& err){
        if(pipe && pipe->res==res && pipe->precision==precision) return true;
        pipe.reset();   // free old GPU memory first
        auto p=std::make_unique<Pipeline>();
        p->res=res; p->precision=precision;
        p->H_LAT=res/8; p->W_LAT=res/8; p->H_P=p->H_LAT/PATCH; p->W_P=p->W_LAT/PATCH;
        p->SEQ_IMG=p->H_P*p->W_P;
        if(res%16!=0 || p->SEQ_IMG%128!=0){ err="bad res "+std::to_string(res); return false; }
        const bool fp8=(precision=="fp8");
        const char* dir=fp8?"transformer_mxfp8":"transformer_f2k";
        p->ld=std::make_unique<f2k::F2KModelLoader>();
        for(int i=1;i<=2;++i){ char s[256];
            std::snprintf(s,sizeof(s),"%s/models/flux2-klein-9B/%s/shard-%05d.f2k1",HOME,dir,i);
            if(!p->ld->add_shard(s)){ err="tf loader: "+p->ld->last_error(); return false; } }
        p->router=std::make_unique<f2k::TensorRouter>(*p->ld);
        if(!p->router->build()){ err="router: "+p->router->last_error(); return false; }
        f2k::cuda::FluxTransformer::Config tc{};
        tc.batch=1; tc.seq_img=p->SEQ_IMG; tc.seq_txt=SEQ_TXT; tc.H_patches=p->H_P; tc.W_patches=p->W_P;
        tc.in_channels=IN_CH; tc.t5_dim=T5_DIM; tc.time_dim=TIME_DIM; tc.n_heads=N_HEADS;
        tc.head_dim=HEAD_DIM; tc.ffn_dim=FFN_DIM; tc.num_double_blocks=N_DOUBLE;
        tc.num_single_blocks=N_SINGLE; tc.rope_theta=2000.0f; tc.router=p->router.get();
        tc.precision = fp8 ? f2k::cuda::Precision::MXFP8 : f2k::cuda::Precision::NVFP4;
        auto t=Clock::now(); p->model=std::make_unique<f2k::cuda::FluxTransformer>(tc);
        if(!p->model->ok()){ err=std::string("transformer: ")+p->model->last_error(); return false; }
        p->vae_r=std::make_unique<f2k::F2KReader>();
        std::string vp=std::string(HOME)+"/models/flux2-klein-9B/vae_f2k/vae.f2k1";
        if(!p->vae_r->open(vp)){ err="open vae"; return false; }
        f2k::cuda::VAEDecoder::Config vc{}; vc.N=1; vc.H_lat=p->H_LAT; vc.W_lat=p->W_LAT;
        vc.prefix="decoder"; vc.reader=p->vae_r.get();
        p->vae=std::make_unique<f2k::cuda::VAEDecoder>(vc);
        if(!p->vae->ok()){ err=std::string("vae: ")+p->vae->last_error(); return false; }
        // VAE encoder (img2img): same vae.f2k1, "encoder" prefix.
        f2k::cuda::VAEEncoder::Config ec{}; ec.N=1; ec.H_in=res; ec.W_in=res;
        ec.prefix="encoder"; ec.reader=p->vae_r.get();
        p->venc=std::make_unique<f2k::cuda::VAEEncoder>(ec);
        if(!p->venc->ok()){ err=std::string("vae enc: ")+p->venc->last_error(); return false; }
        // bn de-norm stats
        const f2k::TensorView* bm=p->vae_r->find("bn.running_mean");
        const f2k::TensorView* bv=p->vae_r->find("bn.running_var");
        if(!bm||!bv){ err="vae bn missing"; return false; }
        const auto* mbf=reinterpret_cast<const __nv_bfloat16*>(bm->data);
        const auto* vbf=reinterpret_cast<const __nv_bfloat16*>(bv->data);
        p->bn_mean.resize(IN_CH); p->bn_std.resize(IN_CH);
        for(int c=0;c<IN_CH;++c){ p->bn_mean[c]=b2f(mbf[c]); p->bn_std[c]=std::sqrt(b2f(vbf[c])+1e-4f); }
        // buffers
        p->latent_elems=(size_t)32*p->H_LAT*p->W_LAT;
        p->token_elems =(size_t)p->SEQ_IMG*IN_CH;
        p->pixel_elems =(size_t)3*p->vae->output_H()*p->vae->output_W();
        p->moment_elems=(size_t)64*p->H_LAT*p->W_LAT;
        cudaMalloc(&p->d_latent,p->latent_elems*2); cudaMalloc(&p->d_tokens,p->token_elems*2);
        cudaMalloc(&p->d_velocity,p->token_elems*2); cudaMalloc(&p->d_pixels,p->pixel_elems*2);
        cudaMalloc(&p->d_t_ws,p->model->workspace_size_bytes());
        cudaMalloc(&p->d_v_ws,p->vae->workspace_size_bytes());
        cudaMalloc(&p->d_init_pix,(size_t)3*res*res*2);
        cudaMalloc(&p->d_moments,p->moment_elems*2);
        cudaMalloc(&p->d_e_ws,p->venc->workspace_size_bytes());
        std::fprintf(stderr,"[worker] pipeline %dpx/%s built in %.1fs\n",res,precision.c_str(),since(t));
        pipe=std::move(p); return true;
    }

    // Decode the in-progress latent (P.d_tokens) into a small base64 PNG for
    // live preview. Reuses d_velocity/d_latent/d_pixels (free between denoise
    // steps); leaves d_tokens untouched. Returns "" on failure.
    std::string preview_b64(Pipeline& P, int maxdim, int& ow, int& oh){
        std::vector<__nv_bfloat16> ht(P.token_elems);
        cudaMemcpy(ht.data(),P.d_tokens,P.token_elems*2,cudaMemcpyDeviceToHost);
        for(int r=0;r<P.SEQ_IMG;++r)for(int c=0;c<IN_CH;++c){ size_t i=(size_t)r*IN_CH+c;
            ht[i]=f2b(b2f(ht[i])*P.bn_std[c]+P.bn_mean[c]); }
        cudaMemcpy(P.d_velocity,ht.data(),P.token_elems*2,cudaMemcpyHostToDevice);
        if(!f2k::cuda::unpatchify_bf16(P.d_velocity,P.d_latent,1,32,P.H_LAT,P.W_LAT,PATCH)) return "";
        if(!P.vae->forward(P.d_latent,P.d_pixels,P.d_v_ws,P.vae->workspace_size_bytes())) return "";
        cudaDeviceSynchronize();
        const int H=P.vae->output_H(), W=P.vae->output_W();
        std::vector<__nv_bfloat16> hp(P.pixel_elems);
        cudaMemcpy(hp.data(),P.d_pixels,P.pixel_elems*2,cudaMemcpyDeviceToHost);
        std::vector<uint8_t> rgb((size_t)H*W*3);
        for(int h=0;h<H;++h)for(int w=0;w<W;++w)for(int c=0;c<3;++c){
            float v=b2f(hp[(size_t)(c*H+h)*W+w]); v=std::min(1.f,std::max(0.f,(v+1.f)*0.5f));
            rgb[((size_t)h*W+w)*3+c]=(uint8_t)(v*255.f+0.5f); }
        int tw=W, th=H;
        if(std::max(W,H)>maxdim){ float s=(float)maxdim/std::max(W,H);
            tw=std::max(1,(int)(W*s)); th=std::max(1,(int)(H*s)); }
        std::vector<uint8_t> small;
        const uint8_t* src=rgb.data();
        if(tw!=W||th!=H){ small.resize((size_t)tw*th*3);
            stbir_resize_uint8_linear(rgb.data(),W,H,0,small.data(),tw,th,0,STBIR_RGB); src=small.data(); }
        std::vector<uint8_t> jpg; jpg.reserve((size_t)tw*th);
        stbi_write_jpg_to_func(png_collect,&jpg,tw,th,3,src,82);   // JPEG: ~10× smaller than PNG
        ow=tw; oh=th;
        return b64encode(jpg.data(),jpg.size());
    }

    json generate(const json& req, const std::function<void(const json&)>& emit){
        std::string prompt=req.value("prompt",""); int res=req.value("res",1024);
        std::string precision=req.value("precision","fp8"); int steps=req.value("steps",4);
        uint32_t seed=(uint32_t)req.value("seed",(int64_t)0);
        std::string out=req.value("out","");
        std::string init_image=req.value("init_image","");   // img2img source (abs path)
        std::string mask_image=req.value("mask_image","");    // inpaint mask (abs path)
        float strength=req.value("strength",0.6f);           // 0..1; only used w/ init_image
        std::vector<std::string> timing; std::string err;
        auto t_all=Clock::now();
        if(precision!="fp8"&&precision!="nvfp4") precision="fp8";
        steps=std::max(1,std::min(30,steps));
        strength=std::max(0.05f,std::min(1.0f,strength));
        const bool img2img=!init_image.empty();
        const bool inpaint=img2img && !mask_image.empty();
        if(inpaint) strength=std::max(strength,0.6f);   // need enough steps to fill the region
        if(!ensure(res,precision,err)) return json{{"ok",false},{"error",err}};
        Pipeline& P=*pipe;

        // text encode (native tokenizer → Qwen3)
        auto te=Clock::now();
        std::vector<int32_t> ids=tok.encode_for_flux(prompt,SEQ_TXT);
        cudaMemcpy(d_ids,ids.data(),SEQ_TXT*sizeof(int32_t),cudaMemcpyHostToDevice);
        if(!qwen->forward(d_ids,d_txt,d_q_ws,qwen->workspace_size_bytes()))
            return json{{"ok",false},{"error",std::string("qwen: ")+qwen->last_error()}};
        cudaDeviceSynchronize();
        timing.push_back("text encode "+std::to_string((int)(since(te)*1000))+"ms");

        // schedule; for img2img we start partway down it (less noise).
        auto sched=f2k::cuda::FlowMatchScheduler::flux2_dynamic(steps,P.SEQ_IMG);
        int i_start=0;
        if(img2img){
            int n_run=std::max(1,std::min(steps,(int)std::lround((double)steps*strength)));
            i_start=steps-n_run;
        }

        // build the starting latent tokens in P.d_tokens
        std::mt19937 rng(seed); std::normal_distribution<float> dn(0,1);
        std::vector<float> x0v, eps_host, mask;   // inpaint state (transformer space)
        if(!img2img){
            // text-to-image: pure N(0,1) latent → patchify
            std::vector<__nv_bfloat16> hl(P.latent_elems);
            for(auto& v:hl) v=f2b(dn(rng));
            cudaMemcpy(P.d_latent,hl.data(),P.latent_elems*2,cudaMemcpyHostToDevice);
            if(!f2k::cuda::patchify_bf16(P.d_latent,P.d_tokens,1,32,P.H_LAT,P.W_LAT,PATCH))
                return json{{"ok",false},{"error","patchify"}};
        } else {
            // image-to-image: encode init → mean latent → patchify → bn-normalize,
            // then noise to t0 via flow-match interp  x_t = (1-t0)*x0 + t0*noise.
            auto ie=Clock::now();
            std::vector<__nv_bfloat16> hi;
            if(!load_image_bf16(init_image,res,hi,err))
                return json{{"ok",false},{"error",err}};
            cudaMemcpy(P.d_init_pix,hi.data(),hi.size()*2,cudaMemcpyHostToDevice);
            if(!P.venc->forward(P.d_init_pix,P.d_moments,P.d_e_ws,P.venc->workspace_size_bytes()))
                return json{{"ok",false},{"error",std::string("vae enc: ")+P.venc->last_error()}};
            // mean = first 32 channels of the moments tensor → patchify
            if(!f2k::cuda::patchify_bf16(P.d_moments,P.d_tokens,1,32,P.H_LAT,P.W_LAT,PATCH))
                return json{{"ok",false},{"error","enc patchify"}};
            cudaDeviceSynchronize();
            std::vector<__nv_bfloat16> x0(P.token_elems);
            cudaMemcpy(x0.data(),P.d_tokens,P.token_elems*2,cudaMemcpyDeviceToHost);
            // clean latent (transformer space) + fixed noise, retained for inpaint
            x0v.resize(P.token_elems); eps_host.resize(P.token_elems);
            for(int r=0;r<P.SEQ_IMG;++r)for(int c=0;c<IN_CH;++c){ size_t i=(size_t)r*IN_CH+c;
                x0v[i]=(b2f(x0[i])-P.bn_mean[c])/P.bn_std[c]; eps_host[i]=dn(rng); }
            const float t0=sched.t(i_start);
            std::vector<__nv_bfloat16> xt(P.token_elems);
            for(size_t i=0;i<P.token_elems;++i) xt[i]=f2b((1.f-t0)*x0v[i]+t0*eps_host[i]);
            cudaMemcpy(P.d_tokens,xt.data(),P.token_elems*2,cudaMemcpyHostToDevice);
            if(inpaint && !load_mask_tokens(mask_image,res,P.H_P,P.W_P,mask,err))
                return json{{"ok",false},{"error",err}};
            timing.push_back(std::string(inpaint?"inpaint":"img2img")+" encode "+
                             std::to_string((int)(since(ie)*1000))+"ms (str "+
                             std::to_string(strength).substr(0,3)+")");
        }

        // denoise
        auto dl=Clock::now();
        const bool stream=req.value("stream",false);
        const int n_run=steps-i_start;
        const int prev_every=std::max(1,n_run/4);
        std::vector<__nv_bfloat16> cur;   // inpaint host scratch
        if(inpaint) cur.resize(P.token_elems);
        for(int i=i_start;i<steps;++i){
            if(inpaint){
                // lock the kept (unmasked) region to its known noised value at t_i
                // so the model only regenerates the painted region, in context.
                const float t=sched.t(i);
                cudaMemcpy(cur.data(),P.d_tokens,P.token_elems*2,cudaMemcpyDeviceToHost);
                for(int r=0;r<P.SEQ_IMG;++r){ const float mm=mask[r]; if(mm>=0.999f) continue;
                    const float keep=1.f-mm;
                    for(int c=0;c<IN_CH;++c){ size_t idx=(size_t)r*IN_CH+c;
                        float known=(1.f-t)*x0v[idx]+t*eps_host[idx];
                        cur[idx]=f2b(mm*b2f(cur[idx])+keep*known); } }
                cudaMemcpy(P.d_tokens,cur.data(),P.token_elems*2,cudaMemcpyHostToDevice);
            }
            auto ht=f2k::cuda::compute_timestep_embedding(sched.t(i)*1000.0f,TIME_DIM);
            cudaMemcpy(d_temb,ht.data(),TIME_DIM*2,cudaMemcpyHostToDevice);
            if(!P.model->forward(P.d_tokens,d_txt,d_temb,P.d_velocity,P.d_t_ws,P.model->workspace_size_bytes()))
                return json{{"ok",false},{"error",std::string("step: ")+P.model->last_error()}};
            if(!f2k::cuda::axpy_bf16(P.d_tokens,P.d_velocity,sched.dt(i),P.token_elems))
                return json{{"ok",false},{"error","axpy"}};
            if(stream && emit){
                int k=i-i_start;
                if((k+1)%prev_every==0 || k==n_run-1){
                    int ow=0,oh=0; std::string b=preview_b64(P,384,ow,oh);
                    if(!b.empty()) emit(json{{"event","progress"},{"step",k+1},{"total",n_run},
                                            {"w",ow},{"h",oh},{"img_b64",b}});
                }
            }
        }
        if(inpaint){
            // final lock: kept region = the exact clean latent (t=0)
            cudaMemcpy(cur.data(),P.d_tokens,P.token_elems*2,cudaMemcpyDeviceToHost);
            for(int r=0;r<P.SEQ_IMG;++r){ const float mm=mask[r]; if(mm>=0.999f) continue;
                const float keep=1.f-mm;
                for(int c=0;c<IN_CH;++c){ size_t idx=(size_t)r*IN_CH+c;
                    cur[idx]=f2b(mm*b2f(cur[idx])+keep*x0v[idx]); } }
            cudaMemcpy(P.d_tokens,cur.data(),P.token_elems*2,cudaMemcpyHostToDevice);
        }
        cudaDeviceSynchronize();
        timing.push_back("denoise "+std::to_string(since(dl)).substr(0,4)+"s ("+std::to_string(steps-i_start)+" steps)");

        // bn de-norm (host) → unpatchify
        std::vector<__nv_bfloat16> ht(P.token_elems);
        cudaMemcpy(ht.data(),P.d_tokens,P.token_elems*2,cudaMemcpyDeviceToHost);
        for(int r=0;r<P.SEQ_IMG;++r)for(int c=0;c<IN_CH;++c){ size_t i=(size_t)r*IN_CH+c;
            ht[i]=f2b(b2f(ht[i])*P.bn_std[c]+P.bn_mean[c]); }
        cudaMemcpy(P.d_tokens,ht.data(),P.token_elems*2,cudaMemcpyHostToDevice);
        if(!f2k::cuda::unpatchify_bf16(P.d_tokens,P.d_latent,1,32,P.H_LAT,P.W_LAT,PATCH))
            return json{{"ok",false},{"error","unpatchify"}};

        // VAE decode
        auto vd=Clock::now();
        if(!P.vae->forward(P.d_latent,P.d_pixels,P.d_v_ws,P.vae->workspace_size_bytes()))
            return json{{"ok",false},{"error",std::string("vae: ")+P.vae->last_error()}};
        cudaDeviceSynchronize();
        timing.push_back("vae "+std::to_string((int)(since(vd)*1000))+"ms");

        // readback + PNG
        std::vector<__nv_bfloat16> hp(P.pixel_elems);
        cudaMemcpy(hp.data(),P.d_pixels,P.pixel_elems*2,cudaMemcpyDeviceToHost);
        std::vector<float> hf(P.pixel_elems); int nans=0;
        for(size_t i=0;i<P.pixel_elems;++i){ float v=b2f(hp[i]); if(!std::isfinite(v))++nans; hf[i]=v; }
        if(!write_png(out,hf.data(),P.vae->output_H(),P.vae->output_W()))
            return json{{"ok",false},{"error","write png"}};
        return json{{"ok",true},{"elapsed",since(t_all)},{"nans",nans},{"timing",timing}};
    }
};

// read one '\n'-terminated line from fd
bool read_line(int fd, std::string& line){
    line.clear(); char c;
    while(true){ ssize_t n=read(fd,&c,1); if(n<=0) return !line.empty(); if(c=='\n') return true; line+=c; }
}
} // namespace

int main(int argc,char**argv){
    std::signal(SIGPIPE, SIG_IGN);   // a client disconnecting mid-stream must not kill us
    int port=8765;
    for(int i=1;i<argc;++i){ std::string a=argv[i];
        if(a=="--port"&&i+1<argc) port=std::atoi(argv[++i]); }
    Worker w; std::string err;
    std::fprintf(stderr,"[worker] loading resident models...\n");
    if(!w.init(err)){ std::fprintf(stderr,"[worker] init failed: %s\n",err.c_str()); return 1; }

    int srv=socket(AF_INET,SOCK_STREAM,0); int yes=1;
    setsockopt(srv,SOL_SOCKET,SO_REUSEADDR,&yes,sizeof(yes));
    sockaddr_in addr{}; addr.sin_family=AF_INET; addr.sin_port=htons(port);
    addr.sin_addr.s_addr=htonl(INADDR_LOOPBACK);   // 127.0.0.1 only
    if(bind(srv,(sockaddr*)&addr,sizeof(addr))<0){ perror("bind"); return 1; }
    listen(srv,4);
    std::fprintf(stderr,"[worker] ready, listening on 127.0.0.1:%d\n",port);
    while(true){
        int fd=accept(srv,nullptr,nullptr); if(fd<0) continue;
        std::string line; json resp;
        // emit writes one '\n'-delimited JSON line (progress events) to the client.
        auto emit=[&](const json& j){ std::string s=j.dump()+"\n"; (void)!write(fd,s.data(),s.size()); };
        if(read_line(fd,line)){
            try { resp=w.generate(json::parse(line), emit); }
            catch(const std::exception& e){ resp=json{{"ok",false},{"error",std::string("parse/exec: ")+e.what()}}; }
        } else resp=json{{"ok",false},{"error","empty request"}};
        std::string out=resp.dump()+"\n"; (void)!write(fd,out.data(),out.size()); close(fd);
        if(resp.value("ok",false)) std::fprintf(stderr,"[worker] job ok %.1fs\n",resp.value("elapsed",0.0));
        else std::fprintf(stderr,"[worker] job ERR: %s\n",resp.value("error","").c_str());
    }
}
