# NPU Software Stack Design

## Overview

This project implements a complete software stack for a custom Neural Processing Unit (NPU) consisting of:

* Control Core
* Tensor Engine
* Vector Engine
* DMA Engine
* Memory Controller
* Interrupt Controller

The architecture is inspired by CUDA and modern accelerator runtimes, providing a layered software model that abstracts hardware complexity from applications.

---

# Architecture

```text
+----------------------+
|     Application      |
+----------------------+
           |
+----------------------+
|      Runtime API     |
|   (CUDA-like API)    |
+----------------------+
           |
+----------------------+
|    User-space API    |
+----------------------+
           |
+----------------------+
|     Kernel Driver    |
+----------------------+
           |
+----------------------+
|       Firmware       |
|    (Control Core)    |
+----------------------+
           |
+----------------------+
|        Hardware      |
+----------------------+
```

---

# Hardware Components

## Control Core

Responsible for:

* Command scheduling
* Queue management
* Tensor dispatch
* Vector dispatch
* Synchronization
* Interrupt handling

---

## Tensor Engine

Accelerates:

* Matrix Multiplication
* Convolution
* GEMM
* Deep Learning Operators

Example operation:

```text
C = A × B
```

---

## Vector Engine

Accelerates:

* Vector Add
* Vector Multiply
* Activation Functions
* Element-wise Operations

Example:

```text
C[i] = A[i] + B[i]
```

---

## DMA Engine

Responsible for:

* Host-to-Device transfer
* Device-to-Host transfer
* Device-to-Device transfer

---

# Kernel Driver Design

## Responsibilities

The kernel driver acts as the hardware abstraction layer.

### Device Initialization

```c
int npu_probe();
int npu_remove();
int npu_suspend();
int npu_resume();
```

Responsibilities:

* Discover hardware
* Map MMIO registers
* Initialize interrupts
* Load firmware
* Configure DMA

---

### Memory Management

```c
void* npu_alloc(size_t size);
void npu_free(void* ptr);
```

Internal APIs:

```c
int npu_dma_alloc();
int npu_map_buffer();
int npu_unmap_buffer();
```

Responsibilities:

* Device memory allocation
* DMA buffer creation
* IOMMU mapping

---

### Command Submission

```c
int npu_submit_command_buffer();
```

Example queue:

```c
struct npu_command_queue {
    uint64_t head;
    uint64_t tail;
};
```

Driver workflow:

1. User submits commands
2. Driver places commands in queue
3. Driver rings hardware doorbell
4. Control core executes workload

---

### Synchronization

```c
int npu_create_fence();
int npu_wait_fence();
int npu_signal_fence();
```

Used for:

* Kernel completion
* DMA completion
* Dependency tracking

---

### Interrupt Handling

```c
irqreturn_t npu_irq_handler();
```

Handles:

* Job completion
* DMA completion
* Faults
* Watchdog timeout

---

### Power Management

```c
int npu_clock_enable();
int npu_clock_disable();
int npu_set_frequency();
```

---

# Firmware Design

Firmware executes on the control core.

## Scheduler

```c
void scheduler_loop();
```

Responsibilities:

* Read command queues
* Dispatch work
* Manage dependencies

---

## Tensor Dispatch

```c
void dispatch_tensor_kernel();
```

Programs tensor engine registers and launches execution.

---

## Vector Dispatch

```c
void dispatch_vector_kernel();
```

Programs vector engine registers and launches execution.

---

## DMA Management

```c
void handle_dma();
```

Schedules memory transfers.

---

# Command Buffer Architecture

Commands are encoded into packets.

## Generic Packet

```c
struct Command {
    uint32_t opcode;
    uint64_t arg0;
    uint64_t arg1;
    uint64_t arg2;
};
```

---

## Supported Commands

```text
CMD_DMA_COPY
CMD_VECTOR_ADD
CMD_VECTOR_MUL
CMD_TENSOR_MATMUL
CMD_TENSOR_CONV
CMD_BARRIER
CMD_WAIT
CMD_END
```

---

## Example Command Stream

```text
DMA_COPY_H2D

TENSOR_MATMUL

VECTOR_ADD

DMA_COPY_D2H

END
```

---

# Tensor Engine Interface

## Dispatch Packet

```c
struct TensorDispatch {
    uint64_t inputA;
    uint64_t inputB;
    uint64_t output;

    uint32_t M;
    uint32_t N;
    uint32_t K;
};
```

## Hardware Registers

```text
TENSOR_A_ADDR
TENSOR_B_ADDR
TENSOR_C_ADDR

TENSOR_M
TENSOR_N
TENSOR_K

TENSOR_START
```

---

# Vector Engine Interface

## Dispatch Packet

```c
struct VectorDispatch {
    uint64_t srcA;
    uint64_t srcB;
    uint64_t dst;

    uint32_t length;
    uint32_t opcode;
};
```

Supported operations:

```text
ADD
SUB
MUL
DIV
EXP
RELU
GELU
```

---

# DMA Engine Interface

```c
dma_h2d();
dma_d2h();
dma_d2d();
```

Transfer Types:

* Host → Device
* Device → Host
* Device → Device

---

# Runtime API

Applications interact through a CUDA-like runtime.

## Initialization

```c
npuInit();
```

---

## Memory Allocation

```c
npuMalloc();
npuFree();
```

---

## Memory Copy

```c
npuMemcpy();
```

---

## Kernel Launch

```c
npuLaunchKernel();
```

---

## Synchronization

```c
npuSynchronize();
```

---

# Example Runtime Flow

```c
npuInit();

npuMalloc(&A, size);
npuMalloc(&B, size);
npuMalloc(&C, size);

npuMemcpy(A, hostA);
npuMemcpy(B, hostB);

npuLaunchKernel(matmul_kernel);

npuSynchronize();

npuMemcpy(hostC, C);

npuFree(A);
npuFree(B);
npuFree(C);
```

---

# Future Features

## Streams

```c
npuCreateStream();
```

Allows concurrent execution.

---

## Events

```c
npuEventCreate();
npuEventRecord();
npuEventWait();
```

Provides synchronization primitives.

---

## Graph Execution

```c
npuGraphCreate();
npuGraphLaunch();
```

Optimized execution for AI workloads.

---

## Unified Memory

```c
npuMallocManaged();
```

Automatic memory migration between host and device.

---

# LLVM / MLIR / IREE Integration

Future compiler integration can generate command packets directly for the NPU.

Target flow:

```text
PyTorch
   ↓
Torch-MLIR
   ↓
MLIR
   ↓
IREE HAL
   ↓
NPU Backend
   ↓
Command Buffer
   ↓
Firmware
   ↓
Tensor / Vector Engines
```

This enables end-to-end compilation from machine learning frameworks directly onto the custom NPU hardware.

---

# Initial Milestone

Minimum viable software stack:

* Device initialization
* Firmware loading
* DMA engine support
* Memory allocation
* Command queue
* Tensor dispatch
* Vector dispatch
* Interrupt handling
* Fence synchronization
* Runtime API

With these components implemented, the NPU can execute basic AI workloads such as:

```text
C = MatMul(A, B)
D = ReLU(C)
```

through a complete hardware/software execution pipeline.
