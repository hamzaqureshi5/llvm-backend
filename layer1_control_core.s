--- |
  ; ModuleID = 'npu-simulator/kernels/vadd.ll'
  source_filename = "npu-simulator/kernels/vadd.ll"
  target datalayout = "e-m:e-p:64:64-i64:64-n64-S64"
  
  define void @vadd(ptr %c, ptr %a, ptr %b, i64 %n) {
  entry:
    br label %loop
  
  loop:                                             ; preds = %loop, %entry
    %lsr.iv = phi i64 [ %lsr.iv.next, %loop ], [ %n, %entry ]
    %a_cur = phi ptr [ %a, %entry ], [ %a_next, %loop ]
    %b_cur = phi ptr [ %b, %entry ], [ %b_next, %loop ]
    %c_cur = phi ptr [ %c, %entry ], [ %c_next, %loop ]
    %av = load i32, ptr %a_cur, align 4
    %bv = load i32, ptr %b_cur, align 4
    %cv = add i32 %av, %bv
    store i32 %cv, ptr %c_cur, align 4
    %a_next = getelementptr i32, ptr %a_cur, i64 1
    %b_next = getelementptr i32, ptr %b_cur, i64 1
    %c_next = getelementptr i32, ptr %c_cur, i64 1
    %lsr.iv.next = add i64 %lsr.iv, -1
    %done = icmp eq i64 %lsr.iv.next, 0
    br i1 %done, label %exit, label %loop
  
  exit:                                             ; preds = %loop
    ret void
  }
...
---
name:            vadd
alignment:       4
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          true
isSSA:           false
noVRegs:         true
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHContTarget: false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: true
registers:       []
liveins:
  - { reg: '$x10', virtual-reg: '' }
  - { reg: '$x11', virtual-reg: '' }
  - { reg: '$x12', virtual-reg: '' }
  - { reg: '$x13', virtual-reg: '' }
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  framePointerPolicy: none
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 0
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: true
  localFrameSize:  0
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo: {}
body:             |
  bb.0.entry:
    successors: %bb.1(0x80000000)
    liveins: $x10, $x11, $x12, $x13
  
    $x14 = LI 0
  
  bb.1.loop:
    successors: %bb.2(0x04000000), %bb.1(0x7c000000)
    liveins: $x10, $x11, $x12, $x13, $x14
  
    $x15 = LW $x12, 0 :: (load (s32) from %ir.b_cur)
    $x16 = LW $x11, 0 :: (load (s32) from %ir.a_cur)
    $x15 = ADD killed $x16, killed $x15
    SW killed $x15, $x10, 0 :: (store (s32) into %ir.c_cur)
    $x10 = ADDI killed $x10, 4
    $x12 = ADDI killed $x12, 4
    $x11 = ADDI killed $x11, 4
    $x13 = ADDI killed $x13, -1
    BNE $x13, $x14, %bb.1
    BR %bb.2
  
  bb.2.exit:
    RET
...
