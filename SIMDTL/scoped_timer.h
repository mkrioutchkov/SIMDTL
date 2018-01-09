#pragma once
#include <chrono>

namespace scoped_timer
{
    namespace detail
    {
        struct scoped_timerBase
        {
            auto GetDuration() const
            {
                return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - mCreated);
            }

            const std::chrono::time_point<std::chrono::system_clock> mCreated = std::chrono::system_clock::now();
        };

        template<typename F>
        struct scoped_timer : protected scoped_timerBase
        {
            scoped_timer(const F& func) : mFunc(func) {}
            ~scoped_timer()
            {
                mFunc(GetDuration());
            }
        private:
            const F mFunc;
        };
    }

    template<typename F>
    static auto make_scoped_timer(const F& func)
    {
        return detail::scoped_timer<F>(func);
    }

    static auto make_scoped_timer(std::chrono::milliseconds& output)
    {
        return make_scoped_timer([&](auto& ms) { output = ms; });
    }
}
