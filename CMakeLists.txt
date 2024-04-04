cmake_minimum_required(VERSION 3.5 FATAL_ERROR)


project(cppoqss)


set(CMAKE_CXX_STANDARD 17)


if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast")


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(FetchContent)


find_package(MPI REQUIRED)

find_package(PETSc REQUIRED)

FetchContent_Declare(
    indicators 
    GIT_REPOSITORY https://github.com/p-ranav/indicators
    GIT_TAG 26d39ad8fb438f5f50fb16427f554045d8431030 # v2.3
    FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(indicators)

find_package(OpenMP)

find_package(Boost REQUIRED COMPONENTS mpi)

FetchContent_Declare(
    cereal
    GIT_REPOSITORY https://github.com/USCiLab/cereal
    GIT_TAG        ebef1e929807629befafbb2918ea1a08c7194554 # v1.3.2
    FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(argparse)

# For analyzer
find_package(ROOT COMPONENTS Core)
set(ROOT_FOUND EXISTS ${ROOT_INCLUDE_DIRS})


add_subdirectory(${PROJECT_SOURCE_DIR}/document)
add_subdirectory(${PROJECT_SOURCE_DIR}/core)
