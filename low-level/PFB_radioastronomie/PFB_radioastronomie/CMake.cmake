cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(pfb LANGUAGES CXX CUDA)
     
include(CTest)
     
add_library(pfb STATIC
  kernel.cu
  CriticalPolyphaseFilterbank.cu
  CriticalPolyphaseFilterbank.h
  params.h
  debug.h
  filehelper.h
  reference.c
  resource.h
  Rng.h
  timer.cuh
  utils_cuda.h
  utils_file.h
)
     
# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to 
# particles will also build with -std=c++11
target_compile_features(particles PUBLIC cxx_std_11)
     
# We need to explicitly state that we need all CUDA files in the 
# particle library to be built with -dc as the member functions 
# could be called by other libraries and executables
set_target_properties( particles
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
     
add_executable(pfb kernel.cu CriticalPolyphaseFilterbank.cu)
   
set_property(TARGET pfb 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    
if(APPLE)
  # We need to add the path to the driver (libcuda.dylib) as an rpath, 
  # so that the static cuda runtime can find it at runtime.
  set_property(TARGET particle_test 
               PROPERTY
               BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()