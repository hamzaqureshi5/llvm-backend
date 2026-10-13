add_llvm_component_library(LLVMNPUInfo
  NPUTargetInfo.cpp

  LINK_COMPONENTS
  MC
  Support

  ADD_TO_COMPONENT
  NPU
  )
