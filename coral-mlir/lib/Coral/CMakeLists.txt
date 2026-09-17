add_mlir_dialect_library(CoralDialect
  CoralDialect.cpp

  DEPENDS
  CoralOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
