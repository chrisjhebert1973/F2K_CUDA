#include "common/platform.h"

#include <cstdlib>
#include <cstring>

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
#else
  #include <cerrno>
  #include <fcntl.h>
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <unistd.h>
#endif

namespace f2k::platform {

#ifdef _WIN32

bool map_readonly(const std::string& path, FileMapping& m, std::string& err) {
    m = FileMapping{};
    HANDLE hFile = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                               OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hFile == INVALID_HANDLE_VALUE) { err = "CreateFile failed: " + path; return false; }
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(hFile, &sz)) { err = "GetFileSizeEx failed"; CloseHandle(hFile); return false; }
    HANDLE hMap = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!hMap) { err = "CreateFileMapping failed"; CloseHandle(hFile); return false; }
    void* p = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    if (!p) { err = "MapViewOfFile failed"; CloseHandle(hMap); CloseHandle(hFile); return false; }
    m.data = p; m.size = static_cast<size_t>(sz.QuadPart); m.h1 = hFile; m.h2 = hMap;
    return true;
}

void unmap(FileMapping& m) {
    if (m.data) UnmapViewOfFile(m.data);
    if (m.h2)   CloseHandle(static_cast<HANDLE>(m.h2));
    if (m.h1)   CloseHandle(static_cast<HANDLE>(m.h1));
    m = FileMapping{};
}

std::string home_dir() {
    const char* p = std::getenv("USERPROFILE");
    return p ? std::string(p) : std::string();
}

#else  // POSIX

bool map_readonly(const std::string& path, FileMapping& m, std::string& err) {
    m = FileMapping{};
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) { err = "open(): " + std::string(std::strerror(errno)) + " (" + path + ")"; return false; }
    struct stat st;
    if (::fstat(fd, &st) != 0) { err = "fstat()"; ::close(fd); return false; }
    const size_t sz = static_cast<size_t>(st.st_size);
    void* p = ::mmap(nullptr, sz, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);                              // the mapping outlives the fd
    if (p == MAP_FAILED) { err = "mmap(): " + std::string(std::strerror(errno)); return false; }
    m.data = p; m.size = sz;
    return true;
}

void unmap(FileMapping& m) {
    if (m.data) ::munmap(m.data, m.size);
    m = FileMapping{};
}

std::string home_dir() {
    const char* p = std::getenv("HOME");
    return p ? std::string(p) : std::string();
}

#endif

} // namespace f2k::platform
