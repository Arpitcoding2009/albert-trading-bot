#include "moving_average.h"

std::vector<double> moving_average(const std::vector<double>& data, int period) {
    std::vector<double> result;
    if (period <= 0 || data.size() < period) {
        return result;
    }

    double sum = 0.0;
    // Calculate initial sum for the first window
    for (int i = 0; i < period; ++i) {
        sum += data[i];
    }

    // First moving average
    result.push_back(sum / period);

    // Slide the window
    for (size_t i = period; i < data.size(); ++i) {
        sum = sum - data[i - period] + data[i];
        result.push_back(sum / period);
    }

    return result;
}
