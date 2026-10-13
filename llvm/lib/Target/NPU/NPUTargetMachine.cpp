//===-- NPUTargetMachine.cpp - Define TargetMachine for NPU --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NPUTargetMachine.h"
#include "NPU.h"
#include "NPUMachineFunctionInfo.h"
#include "TargetInfo/NPUTargetInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Target/TargetOptions.h"
#include <optional>

using namespace llvm;

// Data layout: little-endian, 64-bit pointers and integers. The 32-bit address
// space is enforced physically by the simulator, not by the data layout.
static const char *kDataLayout = "e-m:e-p:64:64-i64:64-n64-S64";

static Reloc::Model getEffectiveRelocModel(std::optional<Reloc::Model> RM) {
  return RM.value_or(Reloc::Static);
}

NPUTargetMachine::NPUTargetMachine(const Target &T, const Triple &TT,
                                   StringRef Cpu, StringRef FeatureString,
                                   const TargetOptions &Options,
                                   std::optional<Reloc::Model> RM,
                                   std::optional<CodeModel::Model> CodeModel,
                                   CodeGenOptLevel OptLevel, bool /*JIT*/)
    : CodeGenTargetMachineImpl(
          T, kDataLayout, TT, Cpu, FeatureString, Options,
          getEffectiveRelocModel(RM),
          getEffectiveCodeModel(CodeModel, CodeModel::Small), OptLevel),
      Subtarget(TT, Cpu, FeatureString, *this),
      TLOF(std::make_unique<TargetLoweringObjectFileELF>()) {
  initAsmInfo();
}

MachineFunctionInfo *NPUTargetMachine::createMachineFunctionInfo(
    BumpPtrAllocator &Allocator, const Function &F,
    const TargetSubtargetInfo *STI) const {
  return NPUMachineFunctionInfo::create<NPUMachineFunctionInfo>(Allocator, F,
                                                                STI);
}

namespace {
class NPUPassConfig : public TargetPassConfig {
public:
  NPUPassConfig(NPUTargetMachine &TM, PassManagerBase *PM)
      : TargetPassConfig(TM, *PM) {}

  NPUTargetMachine &getNPUTargetMachine() const {
    return getTM<NPUTargetMachine>();
  }

  bool addInstSelector() override;
};
} // namespace

TargetPassConfig *NPUTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new NPUPassConfig(*this, &PM);
}

bool NPUPassConfig::addInstSelector() {
  addPass(createNPUISelDag(getNPUTargetMachine()));
  return false;
}

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void LLVMInitializeNPUTarget() {
  RegisterTargetMachine<NPUTargetMachine> X(getTheNPUTarget());
  PassRegistry &PR = *PassRegistry::getPassRegistry();
  initializeNPUDAGToDAGISelLegacyPass(PR);
}
