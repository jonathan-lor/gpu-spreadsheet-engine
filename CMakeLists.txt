cmake_minimum_required(VERSION 3.12)
project(GPU_Spreadsheet LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

set(SOURCES
	src/main.cpp
	src/spreadsheet.cpp
	src/spreadsheet_cuda.cu
)

add_executable(spreadsheet ${SOURCES})
