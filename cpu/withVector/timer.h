#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <string>
#include <iostream>

class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::string name;

public:
    // Constructor: Initializes the timer and records the start time
    explicit Timer(const std::string& function_name = "Function");

    // Destructor: Outputs the elapsed time (if not explicitly stopped)
    ~Timer();

    // Stops the timer and prints the elapsed time
    void stop();
};

#endif // TIMER_H
