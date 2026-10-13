add_llvm_component_group(NPU)

set(LLVM_TARGET_DEFINITIONS NPU.td)

# Auto-generated TableGen outputs.
# NPUGenAsmMatcher.inc is required by NPUAsmParser even though the NPU
# assembler parser is hand-written: the generated file supplies the
# ComputeAvailableFeatures() helper and the GET_ASSEMBLER_HEADER boilerplate.
tablegen(LLVM NPUGenAsmMatcher.inc   -gen-asm-matcher)
tablegen(LLVM NPUGenAsmWriter.inc    -gen-asm-writer)
tablegen(LLVM NPUGenCallingConv.inc  -gen-callingconv)
tablegen(LLVM NPUGenDAGISel.inc      -gen-dag-isel)
tablegen(LLVM NPUGenInstrInfo.inc    -gen-instr-info)
tablegen(LLVM NPUGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM NPUGenSDNodeInfo.inc   -gen-sd-node-info)
tablegen(LLVM NPUGenSubtargetInfo.inc -gen-subtarget)

add_public_tablegen_target(NPUCommonTableGen)

add_llvm_target(NPUCodeGen
  NPUAsmPrinter.cpp
  NPUFrameLowering.cpp
  NPUInstrInfo.cpp
  NPUISelDAGToDAG.cpp
  NPUISelLowering.cpp
  NPUMachineFunctionInfo.cpp
  NPUMCInstLower.cpp
  NPURegisterInfo.cpp
  NPUSelectionDAGInfo.cpp
  NPUSubtarget.cpp
  NPUTargetMachine.cpp

  LINK_COMPONENTS
  AsmPrinter
  CodeGen
  CodeGenTypes
  Core
  MC
  NPUAsmParser
  NPUDesc
  NPUInfo
  SelectionDAG
  Support
  Target
  TargetParser

  ADD_TO_COMPONENT
  NPU
  )

add_subdirectory(AsmParser)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
