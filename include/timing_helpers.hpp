#ifndef TIMING_HELPERS_HPP
#define TIMING_HELPERS_HPP

#include <chrono>
#include <functional>
#include <iostream>
#include <string>
#include <type_traits>

namespace timer {

// Version for non-void return types
template <typename Func, typename... Args,
          typename Ret = std::invoke_result_t<Func, Args...>,
          typename = std::enable_if_t<!std::is_void_v<Ret>>>
Ret timeFunction(const std::string& label, Func&& func, Args&&... args) {
    auto start = std::chrono::high_resolution_clock::now();

    Ret result = std::forward<Func>(func)(std::forward<Args>(args)...);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::micro>(end - start);
    std::cout << label << " took " << duration.count() << " µs\n";

    return result;
}

// Version for void return type
template <typename Func, typename... Args,
          typename Ret = std::invoke_result_t<Func, Args...>,
          typename = std::enable_if_t<std::is_void_v<Ret>>>
void timeFunction(const std::string& label, Func&& func, Args&&... args) {
    auto start = std::chrono::high_resolution_clock::now();

    std::forward<Func>(func)(std::forward<Args>(args)...);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::micro>(end - start);
    std::cout << label << " took " << duration.count() << " µs\n";
}

} // namespace timer

#endif // TIMING_HELPERS_HPP
