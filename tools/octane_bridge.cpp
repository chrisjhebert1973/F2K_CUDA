// F2K_CUDA "Roadrunner" bridge — LAN front door for a retro client (e.g. an SGI
// Octane running a ViewKit/Motif app) to drive the resident CUDA worker.
//
// The worker (tools/serve.cu) binds to 127.0.0.1 only and never returns pixels —
// it writes a PNG to a local path and replies with a status JSON, which is fine
// for the Flask UI on the same box but useless to a remote client that can't read
// the Spark's filesystem and REALLY doesn't want to decode PNG on 1990s IRIX.
//
// This bridge sits in between, changing nothing about the worker:
//
//   Octane  ──two-line ASCII request──►  bridge (0.0.0.0:1974)
//                                           │  JSON job (out=/tmp/*.png)
//                                           ▼
//                                     worker 127.0.0.1:8765  (UNCHANGED)
//   Octane  ◄──framed header + raw RGB──  bridge   ◄── reads+decodes that PNG
//
// The remote side therefore needs zero JSON and zero image libraries: it sends a
// prompt + a few ints and gets back width, height, and w*h*3 raw RGB bytes it can
// drop straight into an XImage.
//
//   build/octane_bridge [--port 1974] [--worker-port 8765]
//                       [--model flux2-klein-4B] [--precision fp8]
//
// ---- wire protocol (bridge <-> remote client) ------------------------------
// Request  (ASCII, two '\n'-terminated lines):
//     line 1:  "GEN <res> <steps> <seed> <count> <cfgx100> <seedVar> <varx100>
//                <model>\n"               ints; seed<0=>random; cfgx100=100 => no
//                                          CFG; varx100=0 => no variation; model
//                                          token "-" => bridge default
//     line 2:  "<prompt>\n"               UTF-8 text, no embedded newline
//     line 3:  "<negative>\n"             CFG unconditional prompt (may be empty)
//   ("LIST\n" instead returns newline-separated model names, then EOF.)
//   Remix (img2img): "REMIX <res> <steps> <seed> <count> <strengthx100> <imgW>
//     <imgH> <cfgx100> <seedVar> <varx100> <model>\n" then "<prompt>\n" then
//     "<negative>\n" then imgW*imgH*3 raw RGB bytes. The client pre-crops to
//     square; the bridge stages it as a PNG for the worker.
//
// Response: a stream of tagged messages (a batch => many). Each begins with the
// 4-byte magic 'F','2','K','1' then an int32 type; all int32 are NETWORK byte
// order (the client may be a big-endian MIPS box):
//     PROGRESS (1): u32 imgIndex, u32 imgTotal, u32 permille(0..1000), u32 len, phase[len]
//     IMAGE    (2): u32 imgIndex, u32 w, u32 h, u32 seed, pixels[w*h*3] RGB top-row-first
//     DONE     (3): u32 count                          (batch complete)
//     ERROR    (4): u32 len, message[len] (ASCII)      (aborts the batch)
// A batch is: PROGRESS* (IMAGE PROGRESS*)* DONE  — i.e. interleaved progress and
// images, one IMAGE per requested count, terminated by DONE (or ERROR).

#include <nlohmann/json.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"   // write the remix init image for the worker to read

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <netdb.h>
#include <csignal>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <random>
#include <string>
#include <vector>

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {

constexpr char     MAGIC[4] = {'F', '2', 'K', '1'};
constexpr uint32_t MAX_PROMPT = 4000;          // sanity cap on a request line
const int          RES_OK[] = {256, 512, 768, 1024};   // worker-legal sizes
const std::string  STOCK   = "flux2-klein-9B"; // the worker's built-in default

std::string g_worker_host = "127.0.0.1";
int         g_worker_port = 8765;
std::string g_model       = "flux2-klein-4B"; // default menu pick; "" => stock 9B
std::string g_precision   = "fp8";

std::mt19937 g_rng{std::random_device{}()};

// -- small socket helpers -----------------------------------------------------
bool send_all(int fd, const void* buf, size_t n) {
    const char* p = static_cast<const char*>(buf);
    while (n) {
        ssize_t k = ::write(fd, p, n);
        if (k <= 0) return false;
        p += k; n -= static_cast<size_t>(k);
    }
    return true;
}

// Read one '\n'-terminated line (newline stripped). false on EOF/error with
// nothing buffered, or if the line grows past `cap` (a hostile/oversized client).
bool recv_line(int fd, std::string& line, size_t cap) {
    line.clear();
    char c;
    while (true) {
        ssize_t n = ::read(fd, &c, 1);
        if (n <= 0) return !line.empty();
        if (c == '\n') return true;
        if (c != '\r') line += c;
        if (line.size() > cap) return false;
    }
}

// Read exactly n bytes (for the raw RGB payload that follows a REMIX request).
bool recv_exact(int fd, void* buf, size_t n) {
    char* p = static_cast<char*>(buf);
    while (n) {
        ssize_t k = ::read(fd, p, n);
        if (k <= 0) return false;
        p += k; n -= static_cast<size_t>(k);
    }
    return true;
}

// Connect to the loopback worker. Returns fd or -1.
int worker_connect() {
    addrinfo hints{}; hints.ai_family = AF_INET; hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    const std::string port = std::to_string(g_worker_port);
    if (getaddrinfo(g_worker_host.c_str(), port.c_str(), &hints, &res) != 0 || !res)
        return -1;
    int fd = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd >= 0 && ::connect(fd, res->ai_addr, res->ai_addrlen) != 0) {
        ::close(fd); fd = -1;
    }
    freeaddrinfo(res);
    return fd;
}

// -- tagged message framing (bridge -> client) --------------------------------
// A batch streams many messages: PROGRESS* (IMAGE PROGRESS*)* DONE, or ERROR.
enum { MSG_PROGRESS = 1, MSG_IMAGE = 2, MSG_DONE = 3, MSG_ERROR = 4 };

bool send_u32(int fd, uint32_t v) { uint32_t n = htonl(v); return send_all(fd, &n, 4); }

bool msg_progress(int fd, uint32_t idx, uint32_t total, uint32_t permille,
                  const std::string& phase) {
    return send_all(fd, MAGIC, 4) && send_u32(fd, MSG_PROGRESS) &&
           send_u32(fd, idx) && send_u32(fd, total) && send_u32(fd, permille) &&
           send_u32(fd, (uint32_t)phase.size()) && send_all(fd, phase.data(), phase.size());
}
bool msg_image(int fd, uint32_t idx, int w, int h, uint32_t seed, const uint8_t* rgb) {
    return send_all(fd, MAGIC, 4) && send_u32(fd, MSG_IMAGE) && send_u32(fd, idx) &&
           send_u32(fd, (uint32_t)w) && send_u32(fd, (uint32_t)h) && send_u32(fd, seed) &&
           send_all(fd, rgb, (size_t)w * h * 3);
}
bool msg_done(int fd, uint32_t count) {
    return send_all(fd, MAGIC, 4) && send_u32(fd, MSG_DONE) && send_u32(fd, count);
}
bool msg_error(int fd, const std::string& s) {
    std::fprintf(stderr, "[bridge] -> error: %s\n", s.c_str());
    return send_all(fd, MAGIC, 4) && send_u32(fd, MSG_ERROR) &&
           send_u32(fd, (uint32_t)s.size()) && send_all(fd, s.data(), s.size());
}

// -- worker round-trip (streaming) --------------------------------------------
// Send one job (stream:true, preview:false) and read the worker's newline-
// delimited JSON: progress events go to on_prog, the final object to reply.
bool worker_stream(const json& job, const std::function<void(const json&)>& on_prog,
                   json& reply, std::string& err) {
    int wf = worker_connect();
    if (wf < 0) { err = "worker not reachable on " + g_worker_host + ":" +
                        std::to_string(g_worker_port); return false; }
    const std::string line = job.dump() + "\n";
    if (!send_all(wf, line.data(), line.size())) { ::close(wf); err = "worker send failed"; return false; }

    std::string buf; char chunk[65536]; bool got_final = false;
    for (;;) {
        size_t nl;
        while ((nl = buf.find('\n')) != std::string::npos) {
            std::string l = buf.substr(0, nl); buf.erase(0, nl + 1);
            if (l.empty()) continue;
            json j;
            try { j = json::parse(l); } catch (const std::exception&) { continue; }
            if (j.value("event", std::string()) == "progress") { on_prog(j); }
            else { reply = j; got_final = true; break; }
        }
        if (got_final) break;
        ssize_t n = ::read(wf, chunk, sizeof chunk);
        if (n <= 0) break;
        buf.append(chunk, (size_t)n);
    }
    ::close(wf);
    if (!got_final) { err = "worker sent no final result"; return false; }
    return true;
}

// Map a worker progress event to a within-image fraction [0,1].
float within_fraction(const json& ev) {
    const std::string ph = ev.value("phase", std::string());
    if (ph == "denoise") {
        int s = ev.value("step", 0), t = ev.value("total", 1); if (t < 1) t = 1;
        return 0.10f + 0.80f * ((float)s / (float)t);
    }
    if (ph == "loading")  return 0.03f;
    if (ph == "encoding") return 0.08f;
    if (ph == "decoding") return 0.95f;
    return 0.0f;
}

// -- model discovery / resolution (mirrors the Flask UI's list_models +
//    _model_fields) ----------------------------------------------------------
std::string models_dir() {
    const char* home = std::getenv("HOME");
    return std::string(home ? home : ".") + "/models";
}

bool dir_has_shard(const fs::path& d) {
    std::error_code ec;
    if (!fs::is_directory(d, ec)) return false;
    for (const auto& e : fs::directory_iterator(d, ec))
        if (e.path().extension() == ".f2k1") return true;
    return false;
}

// Every ~/models/<name> holding a transformer_* dir with .f2k1 shards, with the
// configured default (else stock) hoisted to the front so it's the menu default.
std::vector<std::string> list_models() {
    std::vector<std::string> out;
    std::error_code ec;
    const fs::path root = models_dir();
    if (fs::is_directory(root, ec))
        for (const auto& e : fs::directory_iterator(root, ec))
            if (e.is_directory() &&
                (dir_has_shard(e.path() / "transformer_mxfp8") ||
                 dir_has_shard(e.path() / "transformer_f2k")))
                out.push_back(e.path().filename().string());
    std::sort(out.begin(), out.end());
    const std::string first = g_model.empty() ? STOCK : g_model;
    for (const std::string& pick : {first, STOCK}) {
        auto it = std::find(out.begin(), out.end(), pick);
        if (it != out.end()) { out.erase(it); out.insert(out.begin(), pick); break; }
    }
    return out;
}

// Fill in the worker's {model, transformer, precision} fields for a requested
// checkpoint name. "" / "-" => the bridge default; the stock name or an unknown
// checkpoint => leave them unset so the worker uses its built-in 9B.
void resolve_model(std::string name, json& job) {
    if (name.empty() || name == "-") name = g_model;
    if (name.empty() || name == STOCK) return;
    const fs::path root = (name[0] == '/') ? fs::path(name) : fs::path(models_dir()) / name;
    // pick the quant that actually exists, preferring the configured precision.
    std::string chosen = g_precision;
    fs::path tf = root / (g_precision == "fp8" ? "transformer_mxfp8" : "transformer_f2k");
    if (!dir_has_shard(tf)) {
        const bool want_fp8 = (g_precision == "fp8");
        const fs::path alt = root / (want_fp8 ? "transformer_f2k" : "transformer_mxfp8");
        if (!dir_has_shard(alt)) return;             // nothing usable -> stock fallback
        tf = alt; chosen = want_fp8 ? "nvfp4" : "fp8";
    }
    job["precision"] = chosen;
    if (fs::is_directory(root / "qwen3_f2k")) job["model"] = root.string();   // full root
    else                                      job["transformer"] = tf.string(); // 9B overlay
}

// -- one client ---------------------------------------------------------------
void handle_client(int fd) {
    std::string head;
    if (!recv_line(fd, head, 256)) { msg_error(fd, "empty request"); return; }

    // LIST: reply with newline-separated model names, then close (read to EOF).
    if (head == "LIST") {
        std::string out;
        for (const std::string& m : list_models()) { out += m; out += '\n'; }
        send_all(fd, out.data(), out.size());
        std::fprintf(stderr, "[bridge] LIST -> %zu models\n", list_models().size());
        return;
    }

    // GEN <res> <steps> <seed> <count> <model>
    // REMIX <res> <steps> <seed> <count> <strengthx100> <imgW> <imgH> <model>
    //   ...followed (REMIX only) by imgW*imgH*3 raw RGB bytes after the prompt line.
    const bool remix = (std::strncmp(head.c_str(), "REMIX ", 6) == 0);

    // Two text lines follow the header: prompt, then negative (may be empty).
    std::string prompt, negative;
    if (!recv_line(fd, prompt, MAX_PROMPT))   { msg_error(fd, "malformed request"); return; }
    if (!recv_line(fd, negative, MAX_PROMPT)) { msg_error(fd, "malformed request (negative)"); return; }

    int res = 512, steps = 4, count = 1, strength100 = 60, imgW = 0, imgH = 0;
    int cfg100 = 100, var100 = 0; long seed_in = -1, seed_var = 0; char modelbuf[128] = "";
    const char* p = head.c_str();
    if (remix) {
        p += 6;
        std::sscanf(p, "%d %d %ld %d %d %d %d %d %ld %d %127s", &res, &steps, &seed_in, &count,
                    &strength100, &imgW, &imgH, &cfg100, &seed_var, &var100, modelbuf);
    } else {
        if (std::strncmp(p, "GEN ", 4) == 0) p += 4;      // optional keyword
        std::sscanf(p, "%d %d %ld %d %d %ld %d %127s", &res, &steps, &seed_in, &count,
                    &cfg100, &seed_var, &var100, modelbuf);
    }
    if (cfg100 < 0) cfg100 = 100;
    if (var100 < 0) var100 = 0; if (var100 > 100) var100 = 100;

    // Remix: pull the raw RGB payload and stage it as a PNG the worker can read.
    std::string init_png;
    if (remix) {
        if (imgW <= 0 || imgH <= 0 || imgW > 2048 || imgH > 2048) {
            msg_error(fd, "bad init image dimensions"); return;
        }
        std::vector<uint8_t> rgb((size_t)imgW * imgH * 3);
        if (!recv_exact(fd, rgb.data(), rgb.size())) { msg_error(fd, "short init image"); return; }
        char ip[256];
        std::snprintf(ip, sizeof ip, "/tmp/octane_init_%d_%u.png",
                      getpid(), static_cast<unsigned>(g_rng()));
        if (!stbi_write_png(ip, imgW, imgH, 3, rgb.data(), imgW * 3)) {
            msg_error(fd, "stage init png failed"); return;
        }
        init_png = ip;
    }

    // clamp to worker-legal values so a typo can't cost a worker error round-trip
    bool res_ok = false;
    for (int r : RES_OK) if (r == res) res_ok = true;
    if (!res_ok) res = 512;
    if (steps < 1)  steps = 1;
    if (steps > 30) steps = 30;
    if (count < 1)  count = 1;
    if (count > 8)  count = 8;
    if (strength100 < 5)   strength100 = 5;
    if (strength100 > 100) strength100 = 100;
    if (prompt.empty()) { msg_error(fd, "empty prompt"); if (!init_png.empty()) ::remove(init_png.c_str()); return; }

    std::fprintf(stderr, "[bridge] %s res=%d steps=%d seed=%ld count=%d%s model=%s prompt=\"%.50s\"\n",
                 remix ? "remix" : "batch", res, steps, seed_in, count,
                 remix ? (" str=" + std::to_string(strength100)).c_str() : "",
                 modelbuf[0] ? modelbuf : "(default)", prompt.c_str());

    for (int idx = 0; idx < count; ++idx) {
        // reproducible batch (base+i) if a seed was given, else random per image.
        uint32_t seed = (seed_in < 0) ? g_rng() : static_cast<uint32_t>(seed_in + idx);
        msg_progress(fd, (uint32_t)idx, (uint32_t)count,
                     (uint32_t)(1000.0f * idx / count), "starting");

        char out[256];
        std::snprintf(out, sizeof out, "/tmp/octane_bridge_%d_%u.png",
                      getpid(), static_cast<unsigned>(g_rng()));
        json job = {{"prompt", prompt}, {"res", res}, {"precision", g_precision},
                    {"steps", steps}, {"seed", static_cast<int64_t>(seed)}, {"out", out},
                    {"stream", true}, {"preview", false}};   // cheap per-step progress
        resolve_model(modelbuf, job);
        if (!init_png.empty()) {
            job["init_image"] = init_png;
            job["strength"]   = strength100 / 100.0;
        }
        if (cfg100 != 100) { job["cfg"] = cfg100 / 100.0; job["negative"] = negative; }
        if (var100 > 0) {
            job["seed_var"] = static_cast<int64_t>(seed_var);
            job["var_strength"] = var100 / 100.0;
        }

        auto on_prog = [&](const json& ev) {
            float overall = ((float)idx + within_fraction(ev)) / (float)count;
            msg_progress(fd, (uint32_t)idx, (uint32_t)count,
                         (uint32_t)(overall * 1000.0f),
                         ev.value("phase", std::string("denoise")));
        };
        json reply; std::string err;
        if (!worker_stream(job, on_prog, reply, err)) {
            msg_error(fd, err); ::remove(out); if (!init_png.empty()) ::remove(init_png.c_str()); return;
        }
        if (!reply.value("ok", false)) {
            msg_error(fd, "worker: " + reply.value("error", std::string("unknown")));
            ::remove(out); if (!init_png.empty()) ::remove(init_png.c_str()); return;
        }
        int w = 0, h = 0, comp = 0;
        uint8_t* px = stbi_load(out, &w, &h, &comp, 3);   // force RGB
        ::remove(out);
        if (!px) {
            msg_error(fd, std::string("decode png: ") + stbi_failure_reason());
            if (!init_png.empty()) ::remove(init_png.c_str()); return;
        }
        std::fprintf(stderr, "[bridge] -> image %d/%d %dx%d seed=%u (%.1fs)\n",
                     idx + 1, count, w, h, seed, reply.value("elapsed", 0.0));
        msg_image(fd, (uint32_t)idx, w, h, seed, px);
        stbi_image_free(px);
    }
    if (!init_png.empty()) ::remove(init_png.c_str());
    msg_done(fd, (uint32_t)count);
}

} // namespace

int main(int argc, char** argv) {
    int port = 1974;                              // SIGGRAPH's founding year :)
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--port"         && i + 1 < argc) port = std::atoi(argv[++i]);
        else if (a == "--worker-port"  && i + 1 < argc) g_worker_port = std::atoi(argv[++i]);
        else if (a == "--worker-host"  && i + 1 < argc) g_worker_host = argv[++i];
        else if (a == "--model"        && i + 1 < argc) g_model = argv[++i];
        else if (a == "--precision"    && i + 1 < argc) g_precision = argv[++i];
        else { std::fprintf(stderr, "unknown/incomplete arg: %s\n", a.c_str()); return 2; }
    }
    std::signal(SIGPIPE, SIG_IGN);                // a client hangup must not kill us

    int srv = ::socket(AF_INET, SOCK_STREAM, 0);
    int yes = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof yes);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(static_cast<uint16_t>(port));
    addr.sin_addr.s_addr = htonl(INADDR_ANY);     // LAN-facing, unlike the worker
    if (bind(srv, reinterpret_cast<sockaddr*>(&addr), sizeof addr) < 0) {
        std::fprintf(stderr, "[bridge] bind :%d failed: %s\n", port, std::strerror(errno));
        return 1;
    }
    listen(srv, 4);
    std::fprintf(stderr, "[bridge] listening on 0.0.0.0:%d -> worker %s:%d (default=%s, %s)\n",
                 port, g_worker_host.c_str(), g_worker_port,
                 g_model.empty() ? "stock-9B" : g_model.c_str(), g_precision.c_str());
    {
        std::string names; for (const std::string& m : list_models()) { names += ' '; names += m; }
        std::fprintf(stderr, "[bridge] models:%s\n", names.empty() ? " (none found)" : names.c_str());
    }

    while (true) {
        sockaddr_in cli{}; socklen_t clen = sizeof cli;
        int fd = accept(srv, reinterpret_cast<sockaddr*>(&cli), &clen);
        if (fd < 0) continue;
        std::fprintf(stderr, "[bridge] client %s\n", inet_ntoa(cli.sin_addr));
        handle_client(fd);
        ::close(fd);
    }
}
