from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "moving_average_module",
        ["moving_average.cpp", "moving_average_bindings.cpp"],
        include_dirs=['./'],
        extra_compile_args=['/O2']  # Optimization flag for MSVC
    )
]

setup(
    name="moving_average_module",
    version="0.1.0",
    author="Albert Trading Bot Team",
    description="High-Performance Moving Average Calculations",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
