import sys
sys.path.append('src/cpp')

from moving_average_module import moving_average
import time

def test_moving_average():
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    window = 3

    start_time = time.time()
    result = moving_average(data, window)
    end_time = time.time()

    print(f"Moving Average Test")
    print(f"Input Data: {data}")
    print(f"Window Size: {window}")
    print(f"Result: {result}")
    print(f"Execution Time: {(end_time - start_time) * 1000:.2f} ms")

if __name__ == "__main__":
    test_moving_average()
