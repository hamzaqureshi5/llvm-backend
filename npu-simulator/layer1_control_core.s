	.file	"vadd.ll"
	.text
	.globl	vadd                            # -- Begin function vadd
	.p2align	2
	.type	vadd,@function
vadd:                                   # @vadd
# %bb.0:                                # %entry
	LI x14, 0
.LBB0_1:                                # %loop
                                        # =>This Inner Loop Header: Depth=1
	LW x15, 0(x12)
	LW x16, 0(x11)
	ADD x15, x16, x15
	SW x15, 0(x10)
	ADDI x10, x10, 4
	ADDI x12, x12, 4
	ADDI x11, x11, 4
	ADDI x13, x13, -1
	BNE x13, x14, .LBB0_1
	JAL x0, .LBB0_2
.LBB0_2:                                # %exit
	JALR x0, x1, 0
	HALT
.Lfunc_end0:
	.size	vadd, .Lfunc_end0-vadd
                                        # -- End function
	.section	".note.GNU-stack","",@progbits
