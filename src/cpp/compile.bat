@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
set PYTHON_INCLUDE=C:\Users\arpit\AppData\Local\Programs\Python\Python313\include
set PYTHON_LIB=C:\Users\arpit\AppData\Local\Programs\Python\Python313\libs
set PYBIND11_INCLUDE=C:\Users\arpit\AppData\Local\Programs\Python\Python313\Lib\site-packages\pybind11\include

cl /EHsc /LD /I"%PYTHON_INCLUDE%" /I"%PYBIND11_INCLUDE%" moving_average.cpp moving_average_bindings.cpp /link /LIBPATH:"%PYTHON_LIB%" /OUT:moving_average.pyd
