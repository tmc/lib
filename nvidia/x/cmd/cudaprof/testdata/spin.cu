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
#include <dlfcn.h>
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

// Opt-in ranges require a real provider, never a silent no-op.
static int (*range_push)(const char *);
static int (*range_pop)();
static void (*mark_event)(const char *);
static int delivery_enabled;

static void marker(int token, const char *operation) {
	if (!range_push) return;
	char text[256];
	snprintf(text, sizeof(text), "cudaprof:v1 {\"phase\":\"decode\",\"token\":%d,\"layer\":0,\"operation\":\"%s\",\"dtype\":\"int64\",\"cache_length\":0,\"lookahead\":false,\"materialization\":%s}", token, operation, strcmp(operation,"materialize") == 0 ? "true" : "false");
	range_push(text);
}

// The controlled delivery variant explicitly synchronizes before the event.
// These endpoints describe this workload, not an asynchronous completion guess.
static void delivery_event(int token, const char *event) {
    char text[128];
    snprintf(text,sizeof(text),"cudaprof:v1 {\"phase\":\"decode\",\"token\":%d,\"event\":\"%s\"}",token,event);
    mark_event(text);
}

static void deliver(int token, rec *device, rec *host, int count, long long cycles) {
    if (!delivery_enabled) return;
    marker(token,"materialize");
    check(cudaDeviceSynchronize(),"token materialization");
    check(cudaMemcpy(host,device,count*sizeof(rec),cudaMemcpyDeviceToHost),"token copy");
    for (int i=0;i<count;i++) {
        if (host[i].cycles < cycles || host[i].end < host[i].start) {
            fprintf(stderr,"spin: token materialization invariant failed\n");
            exit(1);
        }
    }
    range_pop();
    delivery_event(token,"delivered");
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
	const char *delivery=getenv("CUDAPROF_SPIN_DELIVERY");
	delivery_enabled=delivery && strcmp(delivery,"1") == 0;
	if (getenv("CUDAPROF_SPIN_MARKERS") || delivery_enabled) {
		void *lib = dlopen("libnvToolsExt.so.1", RTLD_NOW | RTLD_LOCAL);
		if (lib) {
			range_push = (int (*)(const char *))dlsym(lib, "nvtxRangePushA");
			range_pop = (int (*)())dlsym(lib, "nvtxRangePop");
			mark_event = (void (*)(const char *))dlsym(lib,"nvtxMarkA");
		}
		if (!range_push || !range_pop || (delivery_enabled && !mark_event)) {
			fprintf(stderr, "spin: requested NVTX provider unavailable\n");
			return 1;
		}
	}
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
    rec *delivered=delivery_enabled ? (rec *)malloc(n*sizeof(rec)) : NULL;
    if (delivery_enabled && !delivered) { fprintf(stderr,"spin: out of memory\n"); return 1; }
	for (int i = 0; i < 20; i++) {
		warm<<<1, 1>>>(10000000, d + n);
	}
	check(cudaDeviceSynchronize(), "warmup");

	cudaStream_t s = 0;
	cudaGraphExec_t exec = 0;
	if (graph) {
		cudaGraph_t g;
		check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "cudaStreamCreate");
		marker(0, "graph_construction");
		check(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal), "cudaStreamBeginCapture");
		for (int i = 0; i < n; i++) {
			spin<<<1, 1, 0, s>>>(cycles, d + i);
		}
		check(cudaStreamEndCapture(s, &g), "cudaStreamEndCapture");
		if (range_pop) range_pop();
		check(cudaGraphInstantiate(&exec, g, 0), "cudaGraphInstantiate");
		check(cudaGraphUpload(exec, s), "cudaGraphUpload");
		check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
	}

	check(cudaProfilerStart(), "cudaProfilerStart");
	long long t0 = now();
	if (graph) {
		for (int i = 0; i < replays; i++) {
			marker(i, "spin");
			check(cudaGraphLaunch(exec, s), "cudaGraphLaunch");
			if (range_pop) range_pop();
			deliver(i,graph ? d : d+i,delivered,graph ? n : 1,cycles);
		}
	} else {
		for (int i = 0; i < n; i++) {
			marker(i, "spin");
			spin<<<1, 1>>>(cycles, d + i);
			if (range_pop) range_pop();
			deliver(i,graph ? d : d+i,delivered,graph ? n : 1,cycles);
		}
	}
	check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
	long long t1 = now();
	if (delivery_enabled) delivery_event((graph ? replays : n)-1,"token_limit");
	check(cudaProfilerStop(), "cudaProfilerStop");
	check(cudaGetLastError(), "launch");

	rec *h = (rec *)malloc(n * sizeof(rec));
	if (h == NULL) {
		fprintf(stderr, "spin: out of memory\n");
		return 1;
	}
	check(cudaMemcpy(h, d, n * sizeof(rec), cudaMemcpyDeviceToHost), "cudaMemcpy");
	if (getenv("CUDAPROF_SPIN_STABLE_OUTPUT")) {
        for (int i = 0; i < n; i++) {
            if (h[i].cycles < cycles || h[i].end < h[i].start) {
                fprintf(stderr, "spin: kernel invariant failed\n");
                return 1;
            }
        }
        if (delivery_enabled) printf("{\"expected_delivery_events\":%d,\"expected_terminal_events\":1,",graph ? replays : n);
        else printf("{");
        printf("\"mode\":\"%s\",\"cycles\":%lld,\"expected_kernel_records\":%lld,\"expected_submission_ranges\":%d}\n", argv[1], cycles, graph ? (long long)n * replays : n, graph ? replays : n);
        return 0;
    }
    printf("{\"expected_kernel_records\":%lld,\"expected_submission_ranges\":%d,\"markers_enabled\":%s,", graph ? (long long)n * replays : n, graph ? replays : n, range_push ? "true" : "false");
	printf("\"mode\":\"%s\",\"cycles\":%lld,\"replays\":%d,\"region_ns\":%lld,\"launches\":[", argv[1], cycles, replays, t1 - t0);
	for (int i = 0; i < n; i++) {
		printf("%s{\"cycles\":%lld,\"start\":%llu,\"end\":%llu}", i ? "," : "", h[i].cycles, h[i].start, h[i].end);
	}
	printf("]}\n");
	return 0;
}
