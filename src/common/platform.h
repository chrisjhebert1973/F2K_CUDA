// Thin OS abstraction so the runtime builds on Linux (POSIX) and Windows.
// Keep this surface tiny — it exists only to hide the handful of platform calls
// the engine actually needs: read-only file mapping (how F2K weights are read)
// and the user home directory (how model paths are resolved).

#pragma once

#include <cstddef>
#include <string>

namespace f2k::platform {

// A read-only whole-file memory mapping. `data`/`size` are the view; `h1`/`h2`
// are opaque OS handles the implementation keeps alive until unmap()
// (unused on POSIX, where the fd is closed right after mmap).
struct FileMapping {
    void*  data = nullptr;
    size_t size = 0;
    void*  h1   = nullptr;   // Windows: HANDLE hFile
    void*  h2   = nullptr;   // Windows: HANDLE hMapping
};

// Map `path` read-only. On success fills `m` and returns true; else sets `err`.
// POSIX: mmap(MAP_PRIVATE, PROT_READ). Windows: CreateFileMapping + MapViewOfFile.
bool map_readonly(const std::string& path, FileMapping& m, std::string& err);

// Release a mapping from map_readonly (safe on a zeroed FileMapping).
void unmap(FileMapping& m);

// User home directory: $HOME (POSIX) or %USERPROFILE% (Windows); "" if unset.
std::string home_dir();

} // namespace f2k::platform
