Traceback (most recent call last):
  File "c:\Users\arpit\CascadeProjects\crypto_trading_bot\test_cpp_module.py", line 24, in <module>
    test_moving_average()
    ~~~~~~~~~~~~~~~~~~~^^
  File "c:\Users\arpit\CascadeProjects\crypto_trading_bot\test_cpp_module.py", line 14, in test_moving_average
    result = moving_average(data, window)
TypeError: moving_average(): incompatible function arguments. The following argument types are supported:
    1. (data: std::vector<double,std::allocator<double> >, window: int) -> std::vector<double,std::allocator<double> >

#ifndef MOVING_AVERAGE_H
#define MOVING_AVERAGE_H

#include <vector>

std::vector<double> moving_average(const std::vector<double>& data, int period);

#endif // MOVING_AVERAGE_H
