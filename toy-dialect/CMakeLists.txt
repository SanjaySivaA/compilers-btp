cmake_minimum_required(VERSION 3.13)
project(my-opt)

find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH ${MLIR_CMAKE_DIR})
list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIR})
include_directories(${MLIR_INCLUDE_DIR})
link_directories(${LLVM_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# Define our executable
add_executable(my-opt my-opt.cpp)
llvm_update_compile_flags(my-opt)

# Link the necessary MLIR libraries
target_link_libraries(my-opt PRIVATE
  MLIRIR
  MLIRSupport
  MLIRPass
  MLIROptLib
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
)