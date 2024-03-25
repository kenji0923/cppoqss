cmake_minimum_required(VERSION 3.5 FATAL_ERROR)


if (DEFINED ENV{PETSC_DIR})
  # set root of location to find PETSc's pkg-config
  set(PETSC $ENV{PETSC_DIR}/$ENV{PETSC_ARCH})
  set(ENV{PKG_CONFIG_PATH} ${PETSC}/lib/pkgconfig)

  # Remove the lines below if you do not wish to have PETSc determine the compilers
  execute_process(COMMAND pkg-config PETSc --variable=ccompiler COMMAND tr -d '\n' OUTPUT_VARIABLE C_COMPILER)
  SET(CMAKE_C_COMPILER ${C_COMPILER})
  execute_process(COMMAND pkg-config PETSc --variable=cxxcompiler COMMAND tr -d '\n' OUTPUT_VARIABLE CXX_COMPILER)
  if (CXX_COMPILER)
    SET(CMAKE_CXX_COMPILER ${CXX_COMPILER})
  endif (CXX_COMPILER)
endif()


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


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")


find_package(PETSc REQUIRED)

find_package(indicators REQUIRED)

find_package(OpenMP)

if (DEFINED ENV{BOOST_INCLUDEDIR} AND DEFINED ENV{BOOST_LIBRARYDIR})
  set(BOOST_INCLUDEDIR $ENV{BOOST_INCLUDEDIR})
  set(BOOST_LIBRARYDIR $ENV{BOOST_LIBRARYDIR})
  set(Boost_NO_SYSTEM_PATHS ON)
endif()
find_package(Boost REQUIRED COMPONENTS filesystem serialization mpi atomic)

# For analyzer
find_package(ROOT COMPONENTS Core)
set(ROOT_FOUND EXISTS ${ROOT_INCLUDE_DIRS})

find_package(Doxygen)


add_subdirectory(${PROJECT_SOURCE_DIR}/document)
add_subdirectory(${PROJECT_SOURCE_DIR}/core)
