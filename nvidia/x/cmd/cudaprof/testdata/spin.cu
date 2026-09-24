// spin launches a kernel that busy-waits for a fixed number of SM clock
// cycles, so a profiler's report of its duration can be checked against
// two things the profiler does not produce: the cycle count divided by the
// SM clock, and the device's own global timer read at the kernel's first
// and last instruction.
//
// Usage:
//
//	spin stream CYCLES LAUNCHES
//	spin graph CYCLES NODES REPLAYS
//
// stream launches the kernel LAUNCHES times on the default stream. graph
// captures NODES launches into a CUDA graph and replays it REPLAYS times.
// The launches being measured are bracketed by cudaProfilerStart and
// cudaProfilerStop. A warmup kernel, named warm so it is not mistaken for
// the measured one, runs before the range to bring the SM clock up.
//
// The result is one JSON object on stdout: the host wall time of the range
// and, for each launch (for graph mode, each node of the last replay), the
// cycles it spun and the global timer interval it saw.
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct rec {
	long long cycles;
	unsigned long long start, end;
};

__device__ unsigned long long globaltimer() {
	unsigned long long t;
	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
	return t;
}

__device__ void busy(long long n, rec *r) {
	unsigned long long g0 = globaltimer();
	long long c0 = clock64(), c;
	do {
		c = clock64();
	} while (c - c0 < n);
	unsigned long long g1 = globaltimer();
	r->cycles = c - c0;
	r->start = g0;
	r->end = g1;
}

__global__ void spin(long long n, rec *r) { busy(n, r); }
__global__ void warm(long long n, rec *r) { busy(n, r); }

static void check(cudaError_t err, const char *what) {
	if (err != cudaSuccess) {
		fprintf(stderr, "spin: %s: %s\n", what, cudaGetErrorString(err));
		exit(1);
	}
}

static long long now() {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static void usage() {
	fprintf(stderr, "usage: spin stream CYCLES LAUNCHES | spin graph CYCLES NODES REPLAYS\n");
	exit(2);
}

int main(int argc, char **argv) {
	if (argc < 4) {
		usage();
	}
	int graph = strcmp(argv[1], "graph") == 0;
	if (!graph && strcmp(argv[1], "stream") != 0) {
		usage();
	}
	if (graph != (argc == 5) || (!graph && argc != 4)) {
		usage();
	}
	long long cycles = atoll(argv[2]);
	int n = atoi(argv[3]);
	int replays = graph ? atoi(argv[4]) : 0;
	if (cycles <= 0 || n <= 0 || (graph && replays <= 0)) {
		usage();
	}

	rec *d;
	check(cudaMalloc(&d, (n + 1) * sizeof(rec)), "cudaMalloc");
	for (int i = 0; i < 20; i++) {
		warm<<<1, 1>>>(10000000, d + n);
	}
	check(cudaDeviceSynchronize(), "warmup");

	cudaStream_t s = 0;
	cudaGraphExec_t exec = 0;
	if (graph) {
		cudaGraph_t g;
		check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "cudaStreamCreate");
		check(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal), "cudaStreamBeginCapture");
		for (int i = 0; i < n; i++) {
			spin<<<1, 1, 0, s>>>(cycles, d + i);
		}
		check(cudaStreamEndCapture(s, &g), "cudaStreamEndCapture");
		check(cudaGraphInstantiate(&exec, g, 0), "cudaGraphInstantiate");
		check(cudaGraphUpload(exec, s), "cudaGraphUpload");
		check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
	}

	check(cudaProfilerStart(), "cudaProfilerStart");
	long long t0 = now();
	if (graph) {
		for (int i = 0; i < replays; i++) {
			check(cudaGraphLaunch(exec, s), "cudaGraphLaunch");
		}
	} else {
		for (int i = 0; i < n; i++) {
			spin<<<1, 1>>>(cycles, d + i);
		}
	}
	check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
	long long t1 = now();
	check(cudaProfilerStop(), "cudaProfilerStop");
	check(cudaGetLastError(), "launch");

	rec *h = (rec *)malloc(n * sizeof(rec));
	if (h == NULL) {
		fprintf(stderr, "spin: out of memory\n");
		return 1;
	}
	check(cudaMemcpy(h, d, n * sizeof(rec), cudaMemcpyDeviceToHost), "cudaMemcpy");
	printf("{\"mode\":\"%s\",\"cycles\":%lld,\"replays\":%d,\"region_ns\":%lld,\"launches\":[", argv[1], cycles, replays, t1 - t0);
	for (int i = 0; i < n; i++) {
		printf("%s{\"cycles\":%lld,\"start\":%llu,\"end\":%llu}", i ? "," : "", h[i].cycles, h[i].start, h[i].end);
	}
	printf("]}\n");
	return 0;
}
