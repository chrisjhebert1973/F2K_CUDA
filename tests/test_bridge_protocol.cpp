// Regression test for the octane_bridge wire protocol.
//
// The bridge is no longer just the Octane's back end: the LLM server on
// spark-65c1 (LLMTest/src/flux_client.cpp) drives GEN as a tool, so the framing
// is a published API with an out-of-tree consumer that cannot be grepped for
// when something here changes. This test is what makes "GEN/LIST framing is
// frozen" a property of the build rather than a promise someone remembered.
//
// It needs no GPU and no worker. LIST, CAPS, the ERROR frames and both the
// clamped:/model: phases are all emitted before the bridge ever opens a socket
// to the worker, so the whole framing contract is reachable with nothing loaded.
// (IMAGE/DONE are not covered here for that reason -- they cost a real
// generation, which belongs in a GPU test, not a protocol one.)
//
// Spawns its own bridge on a free port with a deliberately dead --worker-port,
// so it can never reach the live service or the GPU.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#ifndef OCTANE_BRIDGE_BIN
#error "OCTANE_BRIDGE_BIN must be defined by the build"
#endif

namespace {

int g_fails = 0;

void check(bool ok, const char* what, const std::string& detail = "") {
    if (ok) {
        std::printf("  ok    %s\n", what);
    } else {
        std::printf("  FAIL  %s%s%s\n", what,
                    detail.empty() ? "" : " -- ", detail.c_str());
        ++g_fails;
    }
}

// A port nothing is on right now. Bind, read back, release. There is a race
// between release and the child's bind, but the alternative is hardcoding a
// port and colliding with the live bridge, which is worse.
int free_port() {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    a.sin_port = 0;
    if (::bind(fd, (sockaddr*)&a, sizeof a) != 0) { ::close(fd); return -1; }
    socklen_t len = sizeof a;
    if (::getsockname(fd, (sockaddr*)&a, &len) != 0) { ::close(fd); return -1; }
    const int port = ntohs(a.sin_port);
    ::close(fd);
    return port;
}

int dial(int port) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    a.sin_port = htons(static_cast<uint16_t>(port));
    if (::connect(fd, (sockaddr*)&a, sizeof a) != 0) { ::close(fd); return -1; }
    return fd;
}

bool send_all(int fd, const std::string& s) {
    const char* p = s.data();
    size_t n = s.size();
    while (n) {
        ssize_t k = ::write(fd, p, n);
        if (k <= 0) return false;
        p += k; n -= static_cast<size_t>(k);
    }
    return true;
}

bool read_exact(int fd, void* buf, size_t n) {
    char* p = static_cast<char*>(buf);
    while (n) {
        ssize_t k = ::read(fd, p, n);
        if (k <= 0) return false;
        p += k; n -= static_cast<size_t>(k);
    }
    return true;
}

bool read_u32(int fd, uint32_t& v) {
    uint32_t n;
    if (!read_exact(fd, &n, 4)) return false;
    v = ntohl(n);                       // the contract says network byte order
    return true;
}

// Read whatever the peer sends until EOF (LIST and CAPS both close after).
std::string slurp(int fd) {
    std::string out;
    char buf[4096];
    while (true) {
        ssize_t k = ::read(fd, buf, sizeof buf);
        if (k <= 0) break;
        out.append(buf, static_cast<size_t>(k));
    }
    return out;
}

enum { MSG_PROGRESS = 1, MSG_IMAGE = 2, MSG_DONE = 3, MSG_ERROR = 4 };

struct Msg {
    bool     valid = false;
    uint32_t type  = 0;
    std::string text;               // phase (PROGRESS) or message (ERROR)
};

// One framed message. Verifies the magic and byte order on the way past.
Msg read_msg(int fd) {
    Msg m;
    char magic[4];
    if (!read_exact(fd, magic, 4)) return m;
    if (std::memcmp(magic, "F2K1", 4) != 0) return m;
    if (!read_u32(fd, m.type)) return m;

    if (m.type == MSG_PROGRESS) {
        uint32_t idx, total, permille, len;
        if (!read_u32(fd, idx) || !read_u32(fd, total) ||
            !read_u32(fd, permille) || !read_u32(fd, len)) return m;
        if (len > (1u << 20)) return m;
        std::vector<char> t(len);
        if (len && !read_exact(fd, t.data(), len)) return m;
        m.text.assign(t.data(), len);
    } else if (m.type == MSG_ERROR) {
        uint32_t len;
        if (!read_u32(fd, len)) return m;
        if (len > (1u << 20)) return m;
        std::vector<char> t(len);
        if (len && !read_exact(fd, t.data(), len)) return m;
        m.text.assign(t.data(), len);
    } else if (m.type == MSG_DONE) {
        uint32_t count;
        if (!read_u32(fd, count)) return m;
    }
    m.valid = true;
    return m;
}

// Read frames until one's phase starts with `prefix`, or the stream ends.
// Phase ORDER is deliberately not asserted anywhere in this test: the contract
// tells clients to match on the prefix, so pinning the order here would freeze
// something consumers were told not to depend on.
Msg await_phase(int fd, const std::string& prefix, int budget = 8) {
    for (int i = 0; i < budget; ++i) {
        Msg m = read_msg(fd);
        if (!m.valid) break;
        if (m.type == MSG_PROGRESS && m.text.rfind(prefix, 0) == 0) return m;
        if (m.type == MSG_ERROR) return m;
    }
    return Msg{};
}

std::map<std::string, std::string> parse_caps(const std::string& body) {
    std::map<std::string, std::string> kv;
    size_t pos = 0;
    while (pos < body.size()) {
        size_t nl = body.find('\n', pos);
        if (nl == std::string::npos) nl = body.size();
        const std::string line = body.substr(pos, nl - pos);
        const size_t eq = line.find('=');
        if (eq != std::string::npos) kv[line.substr(0, eq)] = line.substr(eq + 1);
        pos = nl + 1;
    }
    return kv;
}

std::vector<int> parse_csv_ints(const std::string& s) {
    std::vector<int> out;
    size_t pos = 0;
    while (pos <= s.size()) {
        size_t c = s.find(',', pos);
        if (c == std::string::npos) c = s.size();
        if (c > pos) out.push_back(std::atoi(s.substr(pos, c - pos).c_str()));
        pos = c + 1;
    }
    return out;
}

}  // namespace

int main() {
    const int port = free_port();
    if (port < 0) { std::fprintf(stderr, "could not reserve a port\n"); return 1; }

    // --worker-port 1: nothing listens there, so a stray GEN can never reach the
    // real worker or the GPU. Everything this test asserts happens before the
    // bridge would have dialled it.
    const std::string sport = std::to_string(port);
    pid_t child = ::fork();
    if (child < 0) { std::fprintf(stderr, "fork failed\n"); return 1; }
    if (child == 0) {
        // Silence the child; its logs would interleave with the test output.
        // Not fatal if it fails, hence the discarded result.
        if (!::freopen("/dev/null", "w", stdout)) { /* keep going */ }
        if (!::freopen("/dev/null", "w", stderr)) { /* keep going */ }
        ::execl(OCTANE_BRIDGE_BIN, "octane_bridge",
                "--port", sport.c_str(), "--worker-port", "1", (char*)nullptr);
        ::_exit(127);
    }

    // Wait for it to listen.
    int fd = -1;
    for (int i = 0; i < 200; ++i) {
        fd = dial(port);
        if (fd >= 0) break;
        ::usleep(25 * 1000);
    }
    if (fd < 0) {
        std::fprintf(stderr, "bridge did not come up on port %d\n", port);
        ::kill(child, SIGTERM); ::waitpid(child, nullptr, 0);
        return 1;
    }
    ::close(fd);
    std::printf("bridge up on port %d (pid %d)\n", port, (int)child);

    // -- LIST ---------------------------------------------------------------
    // Contract: newline-separated model names, then EOF. Zero models is legal
    // (a machine with no checkpoints), so this asserts shape, not content.
    std::printf("LIST\n");
    {
        fd = dial(port);
        check(fd >= 0, "connect for LIST");
        if (fd >= 0) {
            send_all(fd, "LIST\n");
            const std::string body = slurp(fd);
            ::close(fd);
            bool clean = true;
            for (char c : body) if (c == '\0') clean = false;
            check(clean, "LIST body is plain text");
            check(body.empty() || body.back() == '\n',
                  "LIST is newline-terminated", "got: " + body);
        }
    }

    // -- CAPS ---------------------------------------------------------------
    std::printf("CAPS\n");
    std::map<std::string, std::string> caps;
    {
        fd = dial(port);
        check(fd >= 0, "connect for CAPS");
        if (fd >= 0) {
            send_all(fd, "CAPS\n");
            const std::string body = slurp(fd);
            ::close(fd);
            caps = parse_caps(body);
            // Keys the documented contract promises. Adding keys is allowed and
            // must not break clients; removing one of these is a break.
            for (const char* k : {"protocol", "res", "res_fallback", "steps",
                                  "count", "strength", "verbs"}) {
                check(caps.count(k) != 0, (std::string("CAPS has ") + k).c_str());
            }
            const std::vector<int> res = parse_csv_ints(caps["res"]);
            check(!res.empty(), "CAPS res parses as a list", caps["res"]);
            check(caps["verbs"].find("GEN") != std::string::npos &&
                  caps["verbs"].find("LIST") != std::string::npos,
                  "CAPS verbs advertises GEN and LIST", caps["verbs"]);
        }
    }

    // -- ERROR framing ------------------------------------------------------
    // Terminal errors must still be a well-formed frame; a client that only
    // ever sees valid input would not notice this rotting.
    std::printf("ERROR frames\n");
    {
        fd = dial(port);
        if (fd >= 0) {
            send_all(fd, "NONSENSE 1 2 3\n\n\n");
            const Msg m = read_msg(fd);
            ::close(fd);
            check(m.valid && m.type == MSG_ERROR,
                  "unknown verb produces a framed ERROR");
        }
        fd = dial(port);
        if (fd >= 0) {
            send_all(fd, "GEN 512 4 1 1 100 0 0 -\n\n\n");   // empty prompt
            const Msg m = read_msg(fd);
            ::close(fd);
            check(m.valid && m.type == MSG_ERROR,
                  "empty prompt produces a framed ERROR");
        }
    }

    // -- clamped: phase, and that CAPS is not lying ------------------------
    // The point of the whole exercise. A client is invited to stop hardcoding
    // the table and trust CAPS instead, so CAPS advertising a value the clamp
    // sites do not enforce is the failure that matters. Ask for a resolution
    // CAPS says is illegal, and require the bridge to report landing on exactly
    // the fallback CAPS advertises.
    std::printf("clamped: phase agrees with CAPS\n");
    if (caps.count("res_fallback") && caps.count("res")) {
        const std::vector<int> legal = parse_csv_ints(caps["res"]);
        int bogus = 640;                                  // not a power-of-two step
        for (bool clash = true; clash; ) {                // ...and definitely not legal
            clash = false;
            for (int r : legal) if (r == bogus) { bogus += 7; clash = true; }
        }
        fd = dial(port);
        if (fd >= 0) {
            send_all(fd, "GEN " + std::to_string(bogus) + " 4 1 1 100 0 0 -\nprobe\n\n");
            const Msg m = await_phase(fd, "clamped:");
            ::close(fd);
            check(m.valid && m.type == MSG_PROGRESS, "illegal res reports a clamp");
            const std::string want =
                "res " + std::to_string(bogus) + "->" + caps["res_fallback"];
            check(m.text.find(want) != std::string::npos,
                  "clamp lands on the value CAPS advertises",
                  "wanted \"" + want + "\" in \"" + m.text + "\"");
        }
    }

    // A legal request must NOT report a clamp -- silence has to stay meaningful,
    // or a client cannot read "no clamped: frame" as "ran as sent".
    std::printf("legal request stays silent\n");
    if (caps.count("res") && caps.count("default_steps")) {
        const std::vector<int> legal = parse_csv_ints(caps["res"]);
        if (!legal.empty()) {
            fd = dial(port);
            if (fd >= 0) {
                send_all(fd, "GEN " + std::to_string(legal[0]) + " " +
                             caps["default_steps"] + " 1 1 100 0 0 -\nprobe\n\n");
                bool clamped = false;
                for (int i = 0; i < 4; ++i) {
                    const Msg m = read_msg(fd);
                    if (!m.valid || m.type == MSG_ERROR) break;
                    if (m.type == MSG_PROGRESS && m.text.rfind("clamped:", 0) == 0)
                        clamped = true;
                }
                ::close(fd);
                check(!clamped, "a fully legal request emits no clamp frame");
            }
        }
    }

    // -- model: phase -------------------------------------------------------
    // Asserts shape only. The value depends on which checkpoints are installed,
    // and this test must pass on a machine with none.
    std::printf("model: phase\n");
    {
        fd = dial(port);
        if (fd >= 0) {
            send_all(fd, "GEN 512 4 1 1 100 0 0 -\nprobe\n\n");
            const Msg m = await_phase(fd, "model:");
            ::close(fd);
            check(m.valid && m.type == MSG_PROGRESS, "batch reports a model: phase");
            check(m.text.size() > std::strlen("model: "),
                  "model: phase names something", m.text);
        }
    }

    ::kill(child, SIGTERM);
    ::waitpid(child, nullptr, 0);

    if (g_fails == 0) {
        std::printf("BRIDGE PROTOCOL OK\n");
        return 0;
    }
    std::printf("BRIDGE PROTOCOL FAILED (%d check%s)\n", g_fails, g_fails == 1 ? "" : "s");
    return 1;
}
