add_llvm_component_library(LLVMNPUDesc
  NPUAsmBackend.cpp
  NPUELFObjectWriter.cpp
  NPUInstPrinter.cpp
  NPUMCAsmInfo.cpp
  NPUMCCodeEmitter.cpp
  NPUMCTargetDesc.cpp

  LINK_COMPONENTS
  MC
  NPUInfo
  Support
  TargetParser

  ADD_TO_COMPONENT
  NPU
  )
