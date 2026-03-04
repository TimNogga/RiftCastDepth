cmake_minimum_required(VERSION 3.27)
list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")

project(RIFTCast LANGUAGES C CXX VERSION 1.0)
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

find_package(OpenCV REQUIRED)

add_subdirectory(RIFTCast/src/riftcast/_C)
add_subdirectory(src)

add_executable(test_depth scripts/test_depths.cpp) 

target_link_libraries(test_depth PRIVATE 
    RIFTCast_lib 
    atcg_lib 
    ${TORCH_LIBRARIES} 
    ${OpenCV_LIBS}
)

target_include_directories(test_depth PRIVATE 
    RIFTCast/include
    ${OpenCV_INCLUDE_DIRS}
)