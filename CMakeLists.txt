cmake_minimum_required(VERSION 3.5 FATAL_ERROR)


project(cppoqss)


set(CMAKE_CXX_STANDARD 17)


set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

option(BUILD_SHARED_LIBS "Build using shared libraries" OFF)


if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast")


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")


find_package(MPI REQUIRED)

find_package(PETSc REQUIRED)

find_package(indicators REQUIRED)

find_package(OpenMP)

if (DEFINED ENV{BOOST_INCLUDEDIR} AND DEFINED ENV{BOOST_LIBRARYDIR})
  set(BOOST_INCLUDEDIR $ENV{BOOST_INCLUDEDIR})
  set(BOOST_LIBRARYDIR $ENV{BOOST_LIBRARYDIR})
  set(Boost_NO_SYSTEM_PATHS ON)
endif()
find_package(Boost REQUIRED COMPONENTS atomic mpi serialization)

# For analyzer
find_package(ROOT COMPONENTS Core)
set(ROOT_FOUND EXISTS ${ROOT_INCLUDE_DIRS})

find_package(Doxygen)


add_subdirectory(${PROJECT_SOURCE_DIR}/document)
add_subdirectory(${PROJECT_SOURCE_DIR}/core)
