#ifndef _COMPILE_TIME_LOOPS_HPP_
#define _COMPILE_TIME_LOOPS_HPP_

#include <utility>

/**
 * @file compile_time_loops.hpp
 * @brief Compile-time loop utilities for tensorial operations
 *
 * Provides constexpr loop constructs that enable compile-time unrolling
 * and optimization of nested loops commonly used in spectral element methods.
 */

/**
 * @brief Implementation helper for compile-time loop using integer sequence
 * @tparam N The upper bound (exclusive) of the loop
 * @tparam F The function/lambda type
 * @tparam Is Parameter pack of integers 0 to N-1
 * @param f Function to call with each std::integral_constant<int, I>
 */
template<int N, typename F, int... Is>
constexpr void for_constexpr_impl(F&& f, std::integer_sequence<int, Is...>) {
    (f(std::integral_constant<int, Is>{}), ...);
}

/**
 * @brief Compile-time for loop from 0 to N-1
 * @tparam N The upper bound (exclusive) of the loop
 * @tparam F The function/lambda type
 * @param f Lambda function accepting std::integral_constant<int, I>
 *
 * Example:
 *   for_constexpr<3>([](auto I) {
 *     constexpr int i = decltype(I)::value;
 *     // Use i at compile time
 *   });
 */
template<int N, typename F>
constexpr void for_constexpr(F&& f) {
    for_constexpr_impl<N>(std::forward<F>(f), std::make_integer_sequence<int, N>{});
}

/**
 * @brief Triple nested compile-time loop
 * @tparam BoundI Upper bound for outer loop
 * @tparam BoundJ Upper bound for middle loop
 * @tparam BoundK Upper bound for inner loop
 * @tparam Lambda The lambda type to execute
 * @param lambda Lambda accepting (std::integral_constant<int, I>,
 *                                  std::integral_constant<int, J>,
 *                                  std::integral_constant<int, K>)
 *
 * Example:
 *   triple_loop<3, 3, 3>([](auto I, auto J, auto K) {
 *     constexpr int i = decltype(I)::value;
 *     constexpr int j = decltype(J)::value;
 *     constexpr int k = decltype(K)::value;
 *     // Use i, j, k at compile time
 *   });
 */
template<int BoundI, int BoundJ, int BoundK, typename Lambda>
constexpr void triple_loop(Lambda&& lambda) {
    for_constexpr<BoundI>([&](auto I) {
        for_constexpr<BoundJ>([&](auto J) {
            for_constexpr<BoundK>([&](auto K) {
                lambda(I, J, K);
            });
        });
    });
}

#endif /* _COMPILE_TIME_LOOPS_HPP_ */
