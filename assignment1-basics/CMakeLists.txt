cmake_minimum_required(VERSION 3.15)
project(cpp_extensions LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 启用 compile_commands.json 生成
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 使用FetchContent获取fmt库
include(FetchContent)

FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG        11.2.0  # 或者使用最新的stable tag
)

FetchContent_MakeAvailable(fmt)

# Python 目录选项
option(PYTHON_DIR "Custom Python directory" "")
if(PYTHON_DIR)
    set(PYTHON_EXECUTABLE "${PYTHON_DIR}/bin/python")
    set(PYTHON_INCLUDE_DIR "${PYTHON_DIR}/include/python3.x")
    set(PYTHON_LIBRARY "${PYTHON_DIR}/lib/libpython3.x.dylib")
    message(STATUS "Using custom Python directory: ${PYTHON_DIR}")
endif()

# AddressSanitizer 配置
option(USE_SANITIZER "Enable AddressSanitizer" OFF)

# 检查环境变量
if(DEFINED ENV{USE_SANITIZER})
    if($ENV{USE_SANITIZER} STREQUAL "ON")
        set(USE_SANITIZER ON)
    endif()
endif()

if(USE_SANITIZER)
    message(STATUS "Sanitizer enabled")
    # 添加编译和链接标志
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread -fno-omit-frame-pointer")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=thread")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -fsanitize=thread")
    
    # 设置环境变量
    set(ENV{ASAN_OPTIONS} "detect_leaks=1:abort_on_error=1:symbolize=1")
    set(ENV{LSAN_OPTIONS} "suppressions=${CMAKE_CURRENT_SOURCE_DIR}/asan_suppressions.txt")
else()
    message(STATUS "Sanitizer disabled")
endif()

# 查找并启用 OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found and enabled")
else()
    message(WARNING "OpenMP not found, parallel processing will be disabled")
endif()

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)

# 创建 C++ 扩展模块
pybind11_add_module(cpp_extensions cs336_basics/cpp/export.cpp)

# 如果找到 OpenMP，链接到目标
if(OpenMP_CXX_FOUND)
    target_link_libraries(cpp_extensions PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(cpp_extensions PRIVATE fmt::fmt)

# 设置输出目录到 Python 包中
set_target_properties(cpp_extensions PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cs336_basics"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cs336_basics"
)

