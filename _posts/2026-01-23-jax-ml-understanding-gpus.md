Source: https://jax-ml.github.io/scaling-book/gpus/

## Building Blocks (Single SM) [WIP]

### Subpartition

#### 1. CUDA Cores:

- General arithmetic operations (Floating point, integer)
- Each subpartition has 32 FP32 cores and a smaller number of INT32 and FP64 cores.
- ReLUs and pointwise operations.

#### 2. Tensor Cores:

- Specialized cores for matrix multiplications, accounting for most FLOPs (floating point operations).
- Each Tensor Core of an H100 GPU can perform 1024 FLOPs/TC.cycle.
- There is one Tensor Core per subpartition.

#### 3. Thread:

- Thread loads data from memory and loads into registers.
- Once an instruction arrives from the scheduler, work is executed on the core.
- Basic unit of work.

#### 4. Warp:

- A group of 32 threads.
- All threads execute the same instruction on different data (SIMT).
- Threads are executed on either CUDA cores or Tensor cores.

#### 5. Warp Scheduler:

- Manages multiple warps.
- Decides which warp to send the work to.
- Manages warps to hide latency (if a warp is waiting for memory, it switches to a ready warp).
- Uses the SIMT principle to send work to cores.

#### 6. Titbits
- `warps and threads are logical groupings.`

####  Final Flow
Warp Scheduler -> Warp (32 threads) -> CUDA Cores/Tensor Cores (with data from registers/memory, using L0 cache for instructions)

Note: CUDA cores are very flexible. Each core can perform different operations, which is managed by masking out the cores that do not need to perform a divergent operation. However, if warps diverge too often, performance silently degrades.

### Memory 🧠

- Register File: Private memory accessible to individual threads. The H100 SM has a 256KiB register file. The number of resident warps depends on how much of this memory is used. Let's calculate how many warp's can fit into the registers.
  * Total register capacity: 16384 registers x 4bytes(32 bits) x 4 subparitions = 2,62,144 bytes = 256KiB(KibiByte - Binary Prefix)
  * Each CUDA core can access max 256 registers at a time, even though we can schedule `64 resident warps`.
  * 4 bytes(per register) x 32 (threads per warp) x 256 (max registers a warp can access) = 32768
  * Total Bytes / Max bytes per warp --> 2,62,144 / 32,768 = 8. 8 Warps can be fit into registers when each warp uses it's max registers(256).
- L0 Instruction Cache: Stores instructions to speed up execution.
- L1 Cache(SMEM):
  * Capacity: 256KB.
  * On-chip cache called SMEM.
  * Each SM has a L1 Cache.
  * Either a programmable shared memory or on-chip cache.
  * Used to store TC matmuls, input data and thread block communication(these threads are part of thread block i.e warp groups).
- L2 Cache:
  * Capacity: 50MB
  * On-Chip cache.
  * Accessible to all SM's.
  * Isn't programmery controlled, hence memory access patterns has to be optimized for proper usage.
  * Slower than L1 cache.
  * Bandwidth of 5.5TB/s.
  * If data is not found in register, L1 Cache, thread checks in L2 cache.
- High Bandwidth Memor(HBM):
  * Main GPU memory
  * Capacity: 32GB in Volta to 192GB in Blackwell.
  * Off-chip memory.
  * Bandwidth(HBM to Tensor Core): 3.5TB/s to 9TB/s.
  * Stores model weights, activations, gradients.

### Final Flow:

Subpartion finds data in this hirearchy: Register file --> L1 Cache --> L2 Cache --> HBM.

## Performance Calculation

Performance Calculations (at 1 GHz Clock Speed)

FP32 CUDA Cores:

  - A single FP32 CUDA core can perform 1 GFLOPS (1 billion operations per second).
  - A subpartition has 32 FP32 CUDA cores, so it can perform 32 billion operations per second.
  - A single SM has 4 subpartitions, so it can perform 4 * 32 = 128 billion operations per second.
  - The H100 has 132 SMs, so the total theoretical FP32 performance is 132 * 128 = 16,896 billion operations per second, or 16.896 TeraFLOPS.

Tensor Cores:

  - A single Tensor Core can perform 1024 FLOPS per cycle.
  - At 1 GHz, a single Tensor Core can perform 1024 billion operations per second, or 1.024 TeraFLOPS.
  - A subpartition has 1 Tensor Core.
  - A single SM has 4 subpartitions (and thus 4 Tensor Cores).
  - A single SM can perform 4 * 1.024 = 4.096 TeraFLOPS.
  - The H100 has 132 SMs, so the total theoretical Tensor Core performance is 132 * 4.096 = 540.672 TeraFLOPS.

In Progress...