#pragma once
#include <type_traits>

namespace mdk
{
    template<typename T, typename U>
    struct preserve_constness
    {
        typedef std::decay_t<U> type;
    };

    template<typename T, typename U>
    struct preserve_constness<const T, U>
    {
        typedef const std::decay_t<U> type;
    };

    template<typename T, typename U>
    using preserve_constness_t = typename preserve_constness<T, U>::type;
}
