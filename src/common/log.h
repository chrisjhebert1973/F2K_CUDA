#pragma once

#include <chrono>
#include <mutex>
#include <string>
#include <vector>

namespace f2k {

enum class LogLevel { Info, Warn, Error };

struct LogLine {
    LogLevel    level;
    std::string text;
    std::chrono::steady_clock::time_point t;
};

class Log {
public:
    static Log& instance() {
        static Log inst;
        return inst;
    }

    void push(LogLevel lvl, std::string text) {
        std::lock_guard lock(m_);
        lines_.push_back({lvl, std::move(text), std::chrono::steady_clock::now()});
        if (lines_.size() > 4096) {
            lines_.erase(lines_.begin(), lines_.begin() + 1024);
        }
    }

    template <typename Fn>
    void for_each(Fn&& fn) const {
        std::lock_guard lock(m_);
        for (const auto& l : lines_) fn(l);
    }

    void clear() {
        std::lock_guard lock(m_);
        lines_.clear();
    }

private:
    mutable std::mutex   m_;
    std::vector<LogLine> lines_;
};

inline void log_info (std::string s) { Log::instance().push(LogLevel::Info,  std::move(s)); }
inline void log_warn (std::string s) { Log::instance().push(LogLevel::Warn,  std::move(s)); }
inline void log_error(std::string s) { Log::instance().push(LogLevel::Error, std::move(s)); }

} // namespace f2k
