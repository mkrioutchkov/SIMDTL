#pragma once

namespace mdk
{
    // This is the type which holds sequences
    template <size_t ... Ns> struct sequence {};

    // First define the template signature
    template <size_t ... Ns> struct seq_gen;

    // Recursion case
    template <size_t I, size_t ... Ns>
    struct seq_gen<I, Ns...>
    {
        // Take front most number of sequence,
        // decrement it, and prepend it twice.
        // First I - 1 goes size_to the counter,
        // Second I - 1 goes size_to the sequence.
        using type = typename seq_gen<
            I - 1, I - 1, Ns...>::type;
    };

    // Recursion abort
    template <size_t ... Ns>
    struct seq_gen<0, Ns...>
    {
        using type = sequence<Ns...>;
    };

    template <size_t ... Ns> struct seq_reverse_gen;

    // Recursion case
    template <size_t I, size_t ... Ns>
    struct seq_reverse_gen<I, Ns...>
    {
        // Take front most number of sequence,
        // decrement it, and prepend it twice.
        // First I - 1 goes size_to the counter,
        // Second I - 1 goes size_to the sequence.
        using type = typename seq_reverse_gen<
            I - 1, Ns..., I - 1>::type;
    };

    // Recursion abort
    template <size_t ... Ns>
    struct seq_reverse_gen<0, Ns...>
    {
        using type = sequence<Ns...>;
    };

    template <size_t N>
    using sequence_t = typename seq_gen<N>::type;

    template <size_t N>
    using sequence_reverse_t = typename seq_reverse_gen<N>::type;

    template<typename T, size_t... Ns>
    static constexpr auto construct_with_reverse_sequence(sequence<Ns...>)
    {
        return T{ Ns... };
    }

    template<typename T, size_t N>
    static constexpr auto construct_with_reverse_sequence()
    {
        return construct_with_reverse_sequence<T>(sequence_reverse_t<N>{});
    }
}
