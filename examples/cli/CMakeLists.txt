cmake_minimum_required(VERSION 3.14)
project(parakeet_cpp)

# Define the CLI executable
add_executable(parakeet-cli main.cpp)
target_link_libraries(parakeet-cli PRIVATE parakeet)
target_include_directories(parakeet-cli PRIVATE ${CMAKE_SOURCE_DIR}/src)

# Define the shared library for C# Interop / PInvoke
add_library(parakeet_bridge SHARED parakeet_bridge.cpp)
target_link_libraries(parakeet_bridge PRIVATE parakeet)
target_include_directories(parakeet_bridge PRIVATE ${CMAKE_SOURCE_DIR}/src)

# Ensure modern symbol visibility logic across platforms
set_target_properties(parakeet_bridge PROPERTIES
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)