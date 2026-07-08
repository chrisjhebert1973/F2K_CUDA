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
//     line 1:  "<res> <steps> <seed>\n"   three integers; seed < 0 => random
//     line 2:  "<prompt>\n"               UTF-8 text, no embedded newline
// Response (binary; all int32 in NETWORK byte order — the client may be a
// big-endian MIPS box, so everything goes through htonl/ntohl):
//     magic   : 4 bytes  'F','2','K','1'
//     status  : int32    0 = ok, non-zero = error
//   if ok:
//     width   : int32
//     height  : int32
//     seed    : int32    the seed actually used (echoed so a random one is known)
//     pixels  : width*height*3 bytes, RGB, row-major, top row first
//   if error:
//     msglen  : int32
//     message : msglen bytes (ASCII)
//
// A separate "LIST\n" request returns newline-separated model names as plain
// text (no framing) and the bridge then closes the connection.

#include <nlohmann/json.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

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

// -- response framing ---------------------------------------------------------
bool send_error(int fd, const std::string& msg) {
    std::fprintf(stderr, "[bridge] -> error: %s\n", msg.c_str());
    uint32_t status = htonl(1);
    uint32_t len    = htonl(static_cast<uint32_t>(msg.size()));
    return send_all(fd, MAGIC, 4) && send_all(fd, &status, 4) &&
           send_all(fd, &len, 4) && send_all(fd, msg.data(), msg.size());
}

bool send_image(int fd, int w, int h, uint32_t seed, const uint8_t* rgb) {
    uint32_t status = htonl(0);
    uint32_t nw = htonl(static_cast<uint32_t>(w));
    uint32_t nh = htonl(static_cast<uint32_t>(h));
    uint32_t ns = htonl(seed);                        // echo the seed actually used
    return send_all(fd, MAGIC, 4) && send_all(fd, &status, 4) &&
           send_all(fd, &nw, 4) && send_all(fd, &nh, 4) && send_all(fd, &ns, 4) &&
           send_all(fd, rgb, static_cast<size_t>(w) * h * 3);
}

// -- worker round-trip --------------------------------------------------------
// Send one JSON job to the resident worker, read its one-line JSON reply.
bool worker_generate(const json& job, json& reply, std::string& err) {
    int wf = worker_connect();
    if (wf < 0) { err = "worker not reachable on " + g_worker_host + ":" +
                        std::to_string(g_worker_port); return false; }
    const std::string line = job.dump() + "\n";
    bool ok = send_all(wf, line.data(), line.size());
    std::string resp;
    if (ok) {
        // the worker may take many seconds; read until the terminating newline.
        char buf[65536];
        while (resp.find('\n') == std::string::npos) {
            ssize_t n = ::read(wf, buf, sizeof buf);
            if (n <= 0) break;
            resp.append(buf, static_cast<size_t>(n));
        }
    }
    ::close(wf);
    if (!ok || resp.empty()) { err = "worker sent no response"; return false; }
    try { reply = json::parse(resp); }
    catch (const std::exception& e) { err = std::string("bad worker JSON: ") + e.what(); return false; }
    return true;
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
    if (!recv_line(fd, head, 256)) { send_error(fd, "empty request"); return; }

    // LIST: reply with newline-separated model names, then close (read to EOF).
    if (head == "LIST") {
        std::string out;
        for (const std::string& m : list_models()) { out += m; out += '\n'; }
        send_all(fd, out.data(), out.size());
        std::fprintf(stderr, "[bridge] LIST -> %zu models\n", list_models().size());
        return;
    }

    std::string prompt;
    if (!recv_line(fd, prompt, MAX_PROMPT)) {
        send_error(fd, "malformed request (want: 'GEN <res> <steps> <seed> [model]' then prompt)");
        return;
    }
    int res = 512, steps = 4; long seed_in = -1; char modelbuf[128] = "";
    const char* p = head.c_str();
    if (std::strncmp(p, "GEN ", 4) == 0) p += 4;          // optional keyword
    std::sscanf(p, "%d %d %ld %127s", &res, &steps, &seed_in, modelbuf);

    // clamp to worker-legal values so a typo can't cost a worker error round-trip
    bool res_ok = false;
    for (int r : RES_OK) if (r == res) res_ok = true;
    if (!res_ok) res = 512;
    if (steps < 1)  steps = 1;
    if (steps > 30) steps = 30;
    uint32_t seed = (seed_in < 0) ? g_rng() : static_cast<uint32_t>(seed_in);
    if (prompt.empty()) { send_error(fd, "empty prompt"); return; }

    // stage the PNG where the worker (same host) can write it, then reclaim it.
    char out[256];
    std::snprintf(out, sizeof out, "/tmp/octane_bridge_%d_%u.png",
                  getpid(), static_cast<unsigned>(g_rng()));

    json job = {{"prompt", prompt}, {"res", res}, {"precision", g_precision},
                {"steps", steps}, {"seed", static_cast<int64_t>(seed)}, {"out", out}};
    resolve_model(modelbuf, job);

    std::fprintf(stderr, "[bridge] job res=%d steps=%d seed=%u model=%s prompt=\"%.60s\"\n",
                 res, steps, seed, modelbuf[0] ? modelbuf : "(default)", prompt.c_str());

    json reply; std::string err;
    if (!worker_generate(job, reply, err)) { send_error(fd, err); return; }
    if (!reply.value("ok", false)) {
        send_error(fd, "worker: " + reply.value("error", std::string("unknown")));
        ::remove(out);
        return;
    }

    int w = 0, h = 0, comp = 0;
    uint8_t* px = stbi_load(out, &w, &h, &comp, 3);   // force RGB
    ::remove(out);
    if (!px) { send_error(fd, std::string("decode png: ") + stbi_failure_reason()); return; }

    std::fprintf(stderr, "[bridge] -> image %dx%d seed=%u (%.1fs worker)\n",
                 w, h, seed, reply.value("elapsed", 0.0));
    send_image(fd, w, h, seed, px);
    stbi_image_free(px);
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
