cmake_minimum_required(VERSION 3.5.0)

if (DEFINED ENV{PETSC_DIR})
  # set root of location to find PETSc's pkg-config
  set(PETSC $ENV{PETSC_DIR}/$ENV{PETSC_ARCH})
  set(ENV{PKG_CONFIG_PATH} ${PETSC}/lib/pkgconfig)

  find_package(PkgConfig REQUIRED)
  pkg_search_module(PETSC REQUIRED IMPORTED_TARGET PETSc)
  # target PkgConfig::PETSC can be used

  set(PETSc_FOUND TRUE)
  message(STATUS "Found PETSc: $ENV{PETSC_ARCH}")
else()
  set(PETSc_FOUND FALSE)
endif()
