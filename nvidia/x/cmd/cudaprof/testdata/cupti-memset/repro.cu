// repro.cu drives the CUDA graph lifecycle of an MLX training loop, the
// one workload known to fault inside libcuda under CUPTI memset tracing.
//
// Usage: repro [launches]
//
// Each launch commits one of TOPOS parent graph topologies, chosen with a
// skew toward a few hot ones as a training step's shapes are. A commit
// allocates a memset destination, builds a parent graph of kernel nodes and
// stream-captured child graphs, one of which holds a 112-byte memset into
// that destination followed by a kernel, and refreshes the cached exec for
// the topology with cuGraphExecUpdate, instantiating it on a miss. Execs
// are never destroyed. The graph is launched, the parent destroyed, and the
// destination freed stream-ordered without a synchronization, so the pool
// hands its address to a later commit. Every SYNC launches the host waits
// for the stream.
//
// Environment:
//
//	TOPOS=n    distinct topologies (default 48)
//	KIND=k     memset destination: 0 cuMemAlloc, 1 cuMemAllocManaged,
//	           2 cuMemAllocAsync (default), 3 alternate 1 and 2 by launch
//	HOSTFN=1   enqueue a host function after every launch (default 1),
//	           as MLX's completion handler does
//	SYNC=n     synchronize the stream every n launches (default 8)
//	SEED=n     topology choice seed (default 1)
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s_ = "?"; \
	cuGetErrorString(r_, &s_); fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #x, s_); exit(1); } } while (0)

extern "C" __global__ void k(float *p, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) p[i] += 1.0f;
}

static CUfunction fn;
static CUstream stream;
static CUdeviceptr buf;
static int kind;

static int envint(const char *name, int def) {
	const char *v = getenv(name);
	return v != NULL && *v != '\0' ? atoi(v) : def;
}

static void launch_on_stream() {
	int n = 1024;
	void *args[] = {&buf, &n};
	CK(cuLaunchKernel(fn, 4, 1, 1, 256, 1, 1, 0, stream, args, NULL));
}

static void add_kernel(CUgraph g, CUgraphNode *dep, CUgraphNode *out) {
	static int n = 1024;
	void *args[] = {&buf, &n};
	CUDA_KERNEL_NODE_PARAMS p = {};
	p.func = fn;
	p.gridDimX = 4, p.gridDimY = p.gridDimZ = 1;
	p.blockDimX = 256, p.blockDimY = p.blockDimZ = 1;
	p.kernelParams = args;
	CK(cuGraphAddKernelNode(out, g, dep, dep ? 1 : 0, &p));
}

// child captures a one-kernel graph, or a memset into dst and a kernel.
static CUgraph child(CUdeviceptr dst, int salt) {
	CUgraph c;
	CK(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
	if (dst != 0) CK(cuMemsetD8Async(dst, (unsigned char)salt, 112, stream));
	launch_on_stream();
	CK(cuStreamEndCapture(stream, &c));
	return c;
}

// parent builds topology t: kernels alternating with child graphs, the
// fourth node the child holding the memset.
static CUgraph parent(int t, CUdeviceptr dst, int salt) {
	CUgraph g;
	CK(cuGraphCreate(&g, 0));
	CUgraphNode prev = NULL, n;
	int nodes = 14 + t % 7 + 7;
	for (int i = 0; i < nodes; i++) {
		if (i % 3 != 1) {
			add_kernel(g, prev ? &prev : NULL, &n);
		} else {
			CUgraph c = child(i == 4 ? dst : 0, salt);
			CK(cuGraphAddChildGraphNode(&n, g, prev ? &prev : NULL, prev ? 1 : 0, c));
			CK(cuGraphDestroy(c));
		}
		prev = n;
	}
	return g;
}

static CUdeviceptr alloc(int k) {
	CUdeviceptr p;
	switch (k) {
	case 1: CK(cuMemAllocManaged(&p, 256, CU_MEM_ATTACH_GLOBAL)); break;
	case 2: CK(cuMemAllocAsync(&p, 256, stream)); break;
	default: CK(cuMemAlloc(&p, 256));
	}
	return p;
}

// release frees a destination after the launch that uses it, without
// waiting for that launch: stream-ordered for the pool, and after the
// next synchronization otherwise.
static std::vector<CUdeviceptr> deferred;

static void release(CUdeviceptr p, int k) {
	if (k == 2) CK(cuMemFreeAsync(p, stream));
	else deferred.push_back(p);
}

static void done(void *) {}

int main(int argc, char **argv) {
	int launches = argc > 1 ? atoi(argv[1]) : 20000;
	int topos = envint("TOPOS", 48), hostfn = envint("HOSTFN", 1), every = envint("SYNC", 8);
	int mode = envint("KIND", 2);
	srand(envint("SEED", 1));
	CK(cuInit(0));
	CUdevice dev;
	CK(cuDeviceGet(&dev, 0));
	CUcontext ctx;
	CK(cuDevicePrimaryCtxRetain(&ctx, dev));
	CK(cuCtxSetCurrent(ctx));
	CUmodule mod;
	CK(cuModuleLoad(&mod, argc > 2 ? argv[2] : "k.cubin"));
	CK(cuModuleGetFunction(&fn, mod, "k"));
	CK(cuMemAlloc(&buf, 1024 * sizeof(float)));
	CK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
	std::vector<CUgraphExec> cache(topos);
	int inst = 0, upd = 0, failed = 0;
	for (int i = 0; i < launches; i++) {
		// Half the commits go to 4 hot topologies, the rest spread out.
		int t = rand() % 2 ? rand() % (topos < 4 ? topos : 4) : rand() % topos;
		kind = mode == 3 ? 1 + i % 2 : mode;
		CUdeviceptr dst = alloc(kind);
		CUgraph g = parent(t, dst, i);
		CUgraphExec &e = cache[t];
		if (e != NULL) {
			CUgraphExecUpdateResultInfo info = {};
			if (cuGraphExecUpdate(e, g, &info) != CUDA_SUCCESS) {
				failed++;
				CK(cuGraphExecDestroy(e));
				e = NULL;
			} else {
				upd++;
			}
		}
		if (e == NULL) {
			CK(cuGraphInstantiate(&e, g, 0));
			inst++;
		}
		CK(cuGraphLaunch(e, stream));
		if (hostfn) CK(cuLaunchHostFunc(stream, done, NULL));
		CK(cuGraphDestroy(g));
		release(dst, kind);
		if (every > 0 && (i + 1) % every == 0) {
			CK(cuStreamSynchronize(stream));
			for (CUdeviceptr p : deferred) CK(cuMemFree(p));
			deferred.clear();
		}
	}
	CK(cuStreamSynchronize(stream));
	printf("repro ok launches=%d instantiates=%d updates=%d update_failures=%d topos=%d kind=%d hostfn=%d sync=%d\n",
		launches, inst, upd, failed, topos, mode, hostfn, every);
	return 0;
}
