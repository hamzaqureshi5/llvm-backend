add_llvm_component_library(LLVMNPUAsmParser
  NPUAsmParser.cpp

  LINK_COMPONENTS
  MC
  MCParser
  NPUDesc
  NPUInfo
  Support
  TargetParser

  ADD_TO_COMPONENT
  NPU
  )
