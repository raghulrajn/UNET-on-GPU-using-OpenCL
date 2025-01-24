#include "Timer.h"

Timer::Timer(const std::string& function_name) : name(function_name) {
    start_time = std::chrono::high_resolution_clock::now();
}

Timer::~Timer() {
    // Destructor does nothing unless explicitly required
}

void Timer::stop() {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << name << " - " << duration.count() << " ms" << std::endl;
}
