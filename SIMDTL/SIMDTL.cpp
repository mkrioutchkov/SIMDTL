#include "stdafx.h"
#include "simd_reverse.h"
#include "scoped_timer.h"
#include <iostream>
#include <cstdint> 
#include <random>

enum class enabler_t {};

template<bool B>
using EnableIf = typename std::enable_if<B, enabler_t>::type;

template<typename R, typename F, EnableIf<!std::is_same<R, void>::value >... >
static decltype(auto) make_sure_function_returns_something(F&& func)
{
    return func;
}

template<typename R, typename F, EnableIf<std::is_same<R, void>::value >... >
static decltype(auto) make_sure_function_returns_something(F&& func)
{
    // functions that returned void, now return int(0)
    return WRAP_IN_CLOSURE_EXTENDED(,func, return int());
}

template<typename test_type, size_t N, typename ScalarF, typename VectF, typename... Args>
static void test_generic(const char * test_name, const ScalarF& process_scalar_x, const VectF& process_vectorized_x, Args&&... args)
{   
    std::vector<test_type> testDataSTL;
    typedef std::decay_t<decltype(process_scalar_x(std::begin(testDataSTL), std::end(testDataSTL), std::forward<Args>(args)...))> stl_return_value;
    auto process_scalar = make_sure_function_returns_something<stl_return_value>(process_scalar_x);
    typedef std::decay_t<decltype(process_scalar(std::begin(testDataSTL), std::end(testDataSTL), std::forward<Args>(args)...))> new_stl_return_value;

    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.0, double(std::numeric_limits<test_type>::max())); // the more random the data, usually the slower simd will perform (for most operations)

    testDataSTL.reserve(N);
    for (size_t i = 0; i < N; ++i)
        testDataSTL.push_back(test_type(dis(gen)));
    
    auto testDataSIMD = testDataSTL;

    typedef decltype(process_vectorized_x(testDataSIMD.data(), testDataSIMD.size(), std::forward<Args>(args)...)) simd_return_value;
    auto process_vectorized = make_sure_function_returns_something<simd_return_value>(process_vectorized_x);
    typedef  decltype(process_vectorized(testDataSIMD.data(), testDataSIMD.size(), std::forward<Args>(args)...)) new_simd_return_value;

    std::chrono::milliseconds timeSTL;
    new_stl_return_value resultSTL{};
    std::chrono::milliseconds timeSIMD;
    new_simd_return_value resultSIMD{};
    
    {
        auto timer = scoped_timer::make_scoped_timer(timeSTL);
        resultSTL = process_scalar(std::begin(testDataSTL), std::end(testDataSTL), std::forward<Args>(args)...);
    }
    {
        auto timer = scoped_timer::make_scoped_timer(timeSIMD);
        resultSIMD = process_vectorized(testDataSIMD.data(), testDataSIMD.size(), std::forward<Args>(args)...);
    }

    std::cout << "Test: " << test_name << ' '
              << "Type: " << typeid(test_type).name() << ' '
              << "Iterations: " << N << ' '
              << "STL ms: " <<  timeSTL.count() << ' '
              << "SIMD ms: " << timeSIMD.count() << ' '
              << "Ratio: " << 1.0 * timeSTL.count() / timeSIMD.count() << ' '
              << "STL result: "  << resultSTL << ' '
              << "SIMD result: " << resultSIMD << ' '
              << "Container equivalence: " << (testDataSTL == testDataSIMD) << ' '
              << "Result equivalence: " << (resultSTL == resultSIMD) << std::endl;
}

template<typename test_type, size_t N, typename t_simd = simd::SSE>
static void test_replace()
{
    const auto scalar_replace = WRAP_IN_CLOSURE(std::replace);
    test_generic<test_type, N>("replace", scalar_replace, simd::replace<t_simd, test_type>, test_type(1), test_type(2));
}

static void replace_tests()
{
    constexpr size_t iterations = 40'000'003;
    test_replace<char, iterations>();
    test_replace<short, iterations>();
    test_replace<unsigned int, iterations>();
    test_replace<float, iterations>();
    test_replace<double, iterations>();
    test_replace<long long, iterations>();
}

template<typename test_type, size_t N, typename t_simd = simd::SSE>
static void test_count()
{
    const auto scalar_count = WRAP_IN_CLOSURE(std::count);
    test_generic<test_type, N>("count", scalar_count, simd::count<t_simd, test_type>, test_type(1));
}

static void count_tests()
{
    constexpr size_t iterations = 80'000'001;// '001;
    test_count<char, iterations>();
    test_count<uint16_t, iterations>();
    test_count<unsigned int, iterations>();
    test_count<long long, iterations>();
}

template<typename test_type, size_t N, typename t_simd = simd::AVX>
static void test_add()
{
    constexpr auto value_to_add = test_type(7.17);
    const auto scalar = [&](auto itbegin, auto itend, auto&)
    {
        std::for_each(itbegin, itend, [&](auto& value) { value += value_to_add; });
    };
   
    typedef typename t_simd::basic_t basic_t;
    basic_t value_to_add_broadcast;
    simd::detail::broadcast(value_to_add_broadcast, value_to_add);
    test_generic<test_type, N>("add", scalar, simd::add<test_type, basic_t>, value_to_add_broadcast);
}

static void add_tests()
{
    constexpr size_t iterations = 80'000'001;// '001;
    test_add<float, iterations>();
}

template<typename test_type, size_t N, typename t_simd = simd::AVX>
static void test_reverse()
{
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.0, double(std::numeric_limits<test_type>::max())); // the more random the data, usually the slower simd will perform (for most operations)

    std::vector<test_type> testDataSTL;
    for (size_t i = 0; i < N; ++i)
        testDataSTL.push_back(test_type(dis(gen)));

    std::chrono::milliseconds timeSTL;
    std::chrono::milliseconds timeSIMD;

    auto testDataSimd = testDataSTL;
    {
        auto timer = scoped_timer::make_scoped_timer(timeSTL);
        std::reverse(testDataSTL.begin(), testDataSTL.end());
    }

    {
        auto timer = scoped_timer::make_scoped_timer(timeSIMD);
        simd::reverse(testDataSimd);
    }
    
    std::cout << "Reverse correctness: " << (testDataSTL == testDataSimd) << ' ' 
              << " STL time (ms): " << timeSTL.count() << " SIMD time (ms): " << timeSIMD.count()
              << " Ratio: " << (1.0 * timeSTL.count() / timeSIMD.count()) << std::endl;
}

static void reverse_tests()
{
    constexpr size_t iterations = false ? 100'000 : 120'000'007;
    test_reverse<char, iterations>();
    test_reverse<short, iterations>();
}

int main()
{
    reverse_tests();
    replace_tests();
    add_tests();
    count_tests();
      
    std::cin.get();
}

