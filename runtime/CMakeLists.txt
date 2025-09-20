project(Genie_runtime)
cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_STANDARD 17)
# Enable -fPIC globally for all targets
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Use installed pybind11 instead of fetching
find_package(pybind11 REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/onnxruntime)
include_directories(${ORT_PATH})


pybind11_add_module(T2SOnnxCPURuntime src/T2SOnnxCPURuntime.cpp)

# onnxruntime paths passed from setup.py
if(NOT DEFINED ORT_PATH)
    message(FATAL_ERROR "ORT_PATH must be defined")
endif()
if(NOT DEFINED ORT_LIBRARY_PATH)
    message(FATAL_ERROR "ORT_LIBRARY_PATH must be defined")
endif()

message("ORT_PATH: ${ORT_PATH}")
message("ORT_LIBRARY_PATH: ${ORT_LIBRARY_PATH}")

# Extract library name from full path
get_filename_component(ORT_LIBRARY_NAME ${ORT_LIBRARY_PATH} NAME)
get_filename_component(ORT_LIBRARY_DIR ${ORT_LIBRARY_PATH} DIRECTORY)

target_link_libraries(T2SOnnxCPURuntime PRIVATE ${ORT_LIBRARY_PATH})
set_target_properties(T2SOnnxCPURuntime PROPERTIES
    INSTALL_RPATH "${ORT_PATH}"
    BUILD_WITH_INSTALL_RPATH TRUE
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
)

if(UNIX) # Linux specific handling soname since onnxruntime pip package does not provide soft ln to libonnxruntime.so.1
    add_custom_command(
        TARGET T2SOnnxCPURuntime
        POST_BUILD
        COMMAND patchelf --replace-needed libonnxruntime.so.1 ${ORT_LIBRARY_NAME} $<TARGET_FILE:T2SOnnxCPURuntime>
        COMMENT "Replacing dependency libonnxruntime.so.1 → ${ORT_LIBRARY_NAME}"
    )
endif()