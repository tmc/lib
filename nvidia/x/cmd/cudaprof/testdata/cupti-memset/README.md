# CUPTI MEMSET activity fault under CUDA graphs

A standalone attempt at the fault that ends every MLX training capture on
GB10 with `CUPTI_ACTIVITY_KIND_MEMSET` enabled: a SIGSEGV inside libcuda in
`cuGraphLaunch`, at the same PC (page offset 0x…1204) each time. cudaprof
works around it with `capture --memsets=false`.

- repro.cu drives the graph lifecycle MLX uses: long-lived executables,
  updated in place with cuGraphExecUpdate, of parent graphs holding kernel
  nodes and stream-captured child graphs, one of which is a 112-byte
  memset followed by a kernel. The memset destination comes from
  cuMemAlloc, cuMemAllocManaged or cuMemAllocAsync (KIND), and is freed
  stream-ordered after each launch so its address is reused. See its
  header comment for the knobs.
- memsetonly.c is a CUPTI injection library that enables only the MEMSET
  activity kind, with knobs to enable it late, disable it early, flush and
  size buffers.

## Build

	nvcc -cubin -arch=sm_121 -o k.cubin repro.cu
	nvcc -O2 -o repro repro.cu -lcuda
	cc -O2 -shared -fPIC -I$CUDA_HOME/extras/CUPTI/include -o memsetonly.so memsetonly.c \
		-L$CUDA_HOME/extras/CUPTI/lib64 -lcupti

## Run

On a shared GPU host, only under its GPU lock wrapper (`$GPU_LOCK`):

	GPU_LOCK_LABEL=cupti-repro $GPU_LOCK \
		env CUDA_INJECTION64_PATH=$PWD/memsetonly.so KIND=2 HOSTFN=1 \
		./repro 20000 k.cubin < /dev/null

## Environment

GB10, driver 580.173.02 (CUDA 13.0 driver API), CUPTI 13.0 through 13.4,
Ubuntu 24.04, kernel 6.17.0-1031-nvidia, aarch64.

## What is known

From the workload (sys1 LoRA training on MLX, MLX_USE_CUDA_GRAPHS=1):

- Every CUPTI from 13.0 to 13.4 faults, at the same libcuda PC.
- MEMSET alone is enough; every other activity kind together is clean.
- The pointer libcuda follows is freed or never-written heap memory:
  under MALLOC_PERTURB_ the fault address is the perturb pattern.
- MEMSET must be enabled at the faulting launch, but not before it:
  enabling it after launch 1000 still faults at 1145, and disabling it
  after launch 50 gives a clean run. The stale state comes from the
  graph lifecycle alone; the MEMSET record path is what reads it.
- Forcing MLX to destroy and re-instantiate executables
  (MLX_CUDA_GRAPH_CACHE_SIZE=8) removes the fault, so long-lived
  executables updated in place are necessary.
- The faulting executable has 14 kernel nodes and 7 child graphs; one child
  holds a 112-byte memset into managed memory and a kernel, and at the
  faulting update the driver reported updates for the kernel nodes only.
- Flushing after every launch and 64 KiB or 8 MiB buffers change nothing;
  256 MiB buffers delay the fault from about launch 1145 to 12539.

## Results

The repro reproduces the fault. Runs on 2026-09-26, driver 580.173.02,
CUPTI 13.0, 20000 launches, under the GPU lock:

	arm                                      result
	no injection, KIND=1, 2 or 3             ok
	MEMSET, KIND=0 (cuMemAlloc)              ok, HOSTFN 0 and 1
	MEMSET, KIND=2 (cuMemAllocAsync)         ok, HOSTFN 0 and 1, SYNC=0, TOPOS=400
	MEMSET, KIND=1 (cuMemAllocManaged)       SIGSEGV at launch 19 (16 with SYNC=1)
	MEMSET, KIND=3 (managed and async)       SIGSEGV, also under MALLOC_PERTURB_
	MEMSET, KIND=1, no launch callback       SIGSEGV
	MEMSET enabled after launch 1000         SIGSEGV at launch 1003
	MEMSET, KIND=1, TOPOS=1 or 2             SIGSEGV at launch 17
	MEMSET, KIND=1, TOPOS=1, SYNC=1          ok

Under gdb the fault is the one MLX hits: libcuda+0x…1204 (`ldr x5,
[x3, #280]` with x3 = 0), called from libcupti, called from libcuda,
called from cuGraphLaunch:

	#0  libcuda.so.1   0x…25b1204
	#1  libcuda.so.1
	#2-#8 libcupti.so.13
	#9-#12 libcuda.so.1
	#13 cuGraphLaunch
	#14 main

So the trigger is a memset node whose destination is managed memory, in an
executable updated in place, launched while CUPTI records MEMSET activity.
Pool and device allocations do not fault. Neither the host function after
each launch nor the launch-counting callback is needed. The smallest
failing arm is one topology, synchronizing and freeing the managed
destinations every 8 launches; it faults on launch 17, the first launch
after the second batch of frees. Freeing after every launch does not fault
with one topology, so the freed managed allocations, still named by
earlier updates of the executable, look like the stale state:

	env CUDA_INJECTION64_PATH=$PWD/memsetonly.so KIND=1 HOSTFN=0 TOPOS=1 \
		./repro 20000 k.cubin

Here the pointer libcuda loads from is NULL rather than the freed heap
memory seen in MLX; either way it is state libcuda failed to set up for a
managed-memory memset node.

For cudaprof this suggests `--memsets=false` is needed only for targets
that memset managed memory inside CUDA graphs, which MLX's CUDA backend
does; other targets were not tested.
