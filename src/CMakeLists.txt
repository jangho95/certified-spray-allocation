cmake_minimum_required(VERSION 3.16)
project(pufoam_stop_spray_solver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(pufoam_solver pufoam_solver.cpp)
target_compile_options(pufoam_solver PRIVATE -O3 -ffp-contract=off -Wall -Wextra -Wpedantic)

add_executable(matrix_exchange matrix_exchange.cpp)
target_compile_options(matrix_exchange PRIVATE -O3 -ffp-contract=off -Wall -Wextra -Wpedantic)
