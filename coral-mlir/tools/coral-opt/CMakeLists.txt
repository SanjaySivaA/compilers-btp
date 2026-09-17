# Fetch all standard MLIR dialects and extensions from the global LLVM registry
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_llvm_executable(coral-opt
  main.cpp
)

# Stitch the LLVM engine, the standard MLIR dialects, and our custom Coral library together
target_link_libraries(coral-opt PRIVATE
  CoralDialect
  MLIROptLib
  ${dialect_libs}
  ${extension_libs}
)
