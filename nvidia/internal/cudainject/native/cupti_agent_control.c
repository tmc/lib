#define _GNU_SOURCE
/*
 * Deterministic controls for the CUPTI producer. Each scenario drives the
 * real producer path -- buffer_requested, buffer_completed, the bounded ring,
 * the batched drain and the journal -- through the stub CUPTI in testcupti,
 * so no GPU and no CUDA toolkit are required. The harness asserts nothing; it
 * offers a known number of records and reports what it offered. The Go test
 * decodes the journal and owns every assertion.
 *
 * Usage: cupti-agent-control <scenario> [args...]
 */
#include "testcupti/cupti.h"

#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/wait.h>
#include <dirent.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <unistd.h>

/* The producer's external-correlation wrappers, which a target calls directly.
   Declared here rather than included so the control links against the real
   symbols the shipping library exports. */
CUptiResult cudaprof_push_external(uint32_t external_kind, uint64_t external_id);
CUptiResult cudaprof_pop_external(uint32_t external_kind, uint64_t *external_id);
int cudaprof_finalize(void);
int InitializeInjection(void);

/* Counts this process's open descriptors whose /proc/self/fd link matches a
   prefix or suffix, which is how the control observes that a forked child
   really closed the parent's journal and termination pipe rather than merely
   forgetting their numbers. Returns -1 where /proc is unavailable, and the
   control then reports the descriptor question as unanswered rather than
   passed. */
static int count_fds(const char *suffix, const char *prefix) {
	DIR *d = opendir("/proc/self/fd");
	if (d == NULL) return -1;
	int n = 0;
	struct dirent *e;
	while ((e = readdir(d)) != NULL) {
		char path[64], target[PATH_MAX];
		if (e->d_name[0] == '.') continue;
		if (snprintf(path, sizeof(path), "/proc/self/fd/%s", e->d_name) >= (int)sizeof(path)) continue;
		ssize_t len = readlink(path, target, sizeof(target) - 1);
		if (len < 0) continue;
		target[len] = '\0';
		if (prefix != NULL && strncmp(target, prefix, strlen(prefix)) == 0) { n++; continue; }
		if (suffix != NULL) {
			size_t tl = strlen(target), sl = strlen(suffix);
			if (tl >= sl && strcmp(target + tl - sl, suffix) == 0) n++;
		}
	}
	closedir(d);
	return n;
}

/* What the forked child observed about the agent it inherited. The child
   cannot assert; it measures and hands the numbers back through a pipe, and
   the Go control owns every assertion. */
struct child_probe {
	int buffer_granted;   /* buffer_requested handed out a buffer */
	int push_result;      /* cudaprof_push_external return */
	int pop_result;       /* cudaprof_pop_external return */
	int journal_fds;      /* open descriptors on a capture.journal */
	int pipe_fds;         /* open pipe descriptors */
};

struct offer_plan {
	unsigned buffers;       /* buffers this thread offers */
	unsigned per_buffer;    /* records per buffer */
	unsigned markers;       /* NVTX markers per buffer, above per_buffer */
	unsigned memsets;       /* memsets per buffer, above per_buffer */
	unsigned undecoded;     /* records of a kind nobody decodes, above per_buffer */
	unsigned overheads;     /* CUPTI overhead spans per buffer, above per_buffer */
	unsigned graph_traces;  /* graph replay envelopes per buffer, above per_buffer */
	unsigned syncs;         /* synchronization spans per buffer, above per_buffer */
	unsigned pause_micros;  /* pause between buffers, 0 for a burst */
	long offered;           /* out: records accepted into buffers */
	long exhausted;         /* out: offers refused because the pool was busy */
};

static void *offer_thread(void *arg) {
	struct offer_plan *plan = arg;
	unsigned kernels = plan->per_buffer / 2;
	unsigned memcpys = plan->per_buffer / 4;
	unsigned apis = plan->per_buffer - kernels - memcpys;
	for (unsigned i = 0; i < plan->buffers; i++) {
		int placed = cudaprof_test_offer_tiers(kernels, memcpys, apis, 0, plan->markers, plan->memsets, plan->undecoded, plan->overheads, plan->graph_traces, plan->syncs);
		if (placed < 0) break;
		if (placed == 0) { plan->exhausted++; continue; }
		plan->offered += placed;
		if (plan->pause_micros) {
			struct timespec ts = {0, (long)plan->pause_micros * 1000};
			nanosleep(&ts, NULL);
		}
	}
	return NULL;
}

/* Runs threads offer_plans concurrently and returns the totals. */
static void run_offers(struct offer_plan *plans, unsigned threads, long *offered, long *exhausted) {
	pthread_t ids[64];
	if (threads > 64) threads = 64;
	for (unsigned i = 1; i < threads; i++) {
		if (pthread_create(&ids[i], NULL, offer_thread, &plans[i]) != 0) { plans[i].buffers = 0; ids[i] = 0; }
	}
	offer_thread(&plans[0]);
	for (unsigned i = 1; i < threads; i++) if (ids[i]) pthread_join(ids[i], NULL);
	*offered = 0;
	*exhausted = 0;
	for (unsigned i = 0; i < threads; i++) { *offered += plans[i].offered; *exhausted += plans[i].exhausted; }
}

struct trigger {
	unsigned delay_micros;
	int signo;
};

static void *trigger_thread(void *arg) {
	struct trigger *t = arg;
	struct timespec ts = {t->delay_micros / 1000000u, (long)(t->delay_micros % 1000000u) * 1000};
	nanosleep(&ts, NULL);
	kill(getpid(), t->signo);
	return NULL;
}

/* Stands in for an interpreter that installs its own SIGTERM handler after
   the producer's ELF constructor has already run, replacing it. CPython does
   exactly this, so the control exists to show what the producer still
   guarantees once its handler is gone. */
static volatile sig_atomic_t app_terminated;

static void app_sigterm(int signo) { (void)signo; app_terminated = 1; }

static unsigned arg_uint(int argc, char **argv, int index, unsigned fallback) {
	if (index >= argc) return fallback;
	unsigned long v = strtoul(argv[index], NULL, 10);
	return (unsigned)v;
}

int main(int argc, char **argv) {
	if (!cudaprof_test_callbacks_registered()) {
		fprintf(stderr, "producer did not register CUPTI callbacks; CUDAPROF_TRANSPORT/CUDAPROF_SESSION unset?\n");
		return 2;
	}
	const char *scenario = argc > 1 ? argv[1] : "sustained";
	unsigned threads = arg_uint(argc, argv, 2, 1);
	unsigned buffers = arg_uint(argc, argv, 3, 8);
	unsigned per_buffer = arg_uint(argc, argv, 4, 512);
	unsigned pause_micros = arg_uint(argc, argv, 5, 0);
	/* NVTX markers are offered above per_buffer rather than carved out of it,
	   so adding them to a scenario does not change the kernel/memcpy/api mix
	   the scenario was calibrated against. */
	unsigned markers = arg_uint(argc, argv, 7, 0);
	unsigned memsets = arg_uint(argc, argv, 8, 0);
	unsigned undecoded = arg_uint(argc, argv, 9, 0);
	unsigned overheads = arg_uint(argc, argv, 10, 0);
	unsigned graph_traces = arg_uint(argc, argv, 11, 0);
	unsigned syncs = arg_uint(argc, argv, 12, 0);
	if (threads == 0) threads = 1;
	if (threads > 64) threads = 64;

	struct offer_plan plans[64];
	memset(plans, 0, sizeof(plans));
	for (unsigned i = 0; i < threads; i++) {
		plans[i].buffers = buffers;
		plans[i].per_buffer = per_buffer;
		plans[i].markers = markers;
		plans[i].memsets = memsets;
		plans[i].undecoded = undecoded;
		plans[i].overheads = overheads;
		plans[i].graph_traces = graph_traces;
		plans[i].syncs = syncs;
		plans[i].pause_micros = pause_micros;
	}

	/* capture-range drives the gate the producer installs when
	   CUDAPROF_CAPTURE_RANGE names the profiler API. It offers the same buffer
	   three times -- before the range opens, inside it, and after it closes --
	   and reports what the producer accepted at each point along with the
	   number of activity kinds enabled there.

	   Reporting the kind count as well as the record count matters: a run that
	   collects nothing because the gate is shut and a run that collects
	   nothing because every enable failed produce the same zero, and only the
	   kind count separates them. Which cbid to trigger on is an argument so
	   one scenario covers the runtime pair and the driver pair; a target calls
	   one or the other and the gate has to open for either. */
	/* clock reports what the stub observed about the producer's timestamp
	   callback: the order of the registration against the first activity
	   enable, and what the registered callback returns read against this
	   process's own CLOCK_MONOTONIC and CLOCK_REALTIME. Both halves are
	   needed. The order alone would pass a producer that registered a
	   REALTIME callback; the value alone would pass one that registered the
	   right function too late for any record to use it.

	   The readings are taken here, after the producer's constructor has run,
	   so they bracket nothing -- they only have to name the clocks, and the
	   two are separated by boot time, which no scheduling delay approaches. */
	if (strcmp(scenario, "clock") == 0) {
		struct timespec mono, real;
		uint64_t mono_ns = 0, real_ns = 0;
		if (clock_gettime(CLOCK_MONOTONIC, &mono) == 0) mono_ns = (uint64_t)mono.tv_sec * 1000000000ull + (uint64_t)mono.tv_nsec;
		if (clock_gettime(CLOCK_REALTIME, &real) == 0) real_ns = (uint64_t)real.tv_sec * 1000000000ull + (uint64_t)real.tv_nsec;
		printf("{\"scenario\":\"clock\",\"timestamp_registered_at\":%u,\"first_enable_at\":%u,"
			"\"callback_ns\":%llu,\"monotonic_ns\":%llu,\"realtime_ns\":%llu}\n",
			cudaprof_test_timestamp_registered_at(), cudaprof_test_first_enable_at(),
			(unsigned long long)cudaprof_test_timestamp(),
			(unsigned long long)mono_ns, (unsigned long long)real_ns);
		fflush(stdout);
		return 0;
	}

	if (strncmp(scenario, "capture-range", strlen("capture-range")) == 0) {
		unsigned domain = arg_uint(argc, argv, 6, 2);  /* RUNTIME_API */
		unsigned start_cbid = domain == 2 ? 171u : 308u;
		unsigned stop_cbid = domain == 2 ? 172u : 309u;
		/* A target may stop collecting and start again, so the restart
		   variant runs the pair twice and reports the second range. A
		   producer that ends the capture at the first stop has no second
		   range to report and writes two edges where four are due. */
		unsigned cycles = strcmp(scenario, "capture-range-restart") == 0 ? 2u : 1u;
		unsigned before_kinds = cudaprof_test_enabled_kind_count();
		int before = cudaprof_test_offer_tiers(buffers, buffers, buffers, 0, markers, memsets, undecoded, overheads, graph_traces, syncs);
		int fired_start = 0, fired_stop = 0, inside = 0, after = 0;
		unsigned inside_kinds = 0, after_kinds = 0;
		for (unsigned c = 0; c < cycles; c++) {
			fired_start = cudaprof_test_fire_callback(domain, start_cbid, 1 /* API_EXIT */);
			inside_kinds = cudaprof_test_enabled_kind_count();
			inside = cudaprof_test_offer_tiers(buffers, buffers, buffers, 0, markers, memsets, undecoded, overheads, graph_traces, syncs);
			fired_stop = cudaprof_test_fire_callback(domain, stop_cbid, 0 /* API_ENTER */);
			after_kinds = cudaprof_test_enabled_kind_count();
			after = cudaprof_test_offer_tiers(buffers, buffers, buffers, 0, markers, memsets, undecoded, overheads, graph_traces, syncs);
		}
		printf("{\"scenario\":\"%s\",\"subscribed\":%d,\"domain\":%u,"
			"\"start_enabled\":%d,\"stop_enabled\":%d,\"fired_start\":%d,\"fired_stop\":%d,"
			"\"before\":%d,\"inside\":%d,\"after\":%d,"
			"\"before_kinds\":%u,\"inside_kinds\":%u,\"after_kinds\":%u}\n",
			scenario, cudaprof_test_subscribed(), domain,
			cudaprof_test_callback_enabled(domain, start_cbid),
			cudaprof_test_callback_enabled(domain, stop_cbid),
			fired_start, fired_stop, before, inside, after,
			before_kinds, inside_kinds, after_kinds);
		fflush(stdout);
		/* The abrupt variant leaves the way a Go runtime leaves: straight to
		   exit_group, running no ELF destructor and no atexit handler. What
		   the journal holds is what was written before this line. */
		if (strcmp(scenario, "capture-range-abrupt") == 0) _exit(0);
		return 0;
	}

	/* graph-replay offers a graph of <nodes> kernel nodes launched <replays>
	   times, as CUPTI delivers it in whichever graph trace mode the producer
	   chose: node kernel records, or one envelope per replay. */
	if (strcmp(scenario, "graph-replay") == 0) {
		unsigned nodes = arg_uint(argc, argv, 2, 3);
		unsigned replays = arg_uint(argc, argv, 3, 4);
		int offered = cudaprof_test_offer_graph_replays(nodes, replays);
		printf("{\"scenario\":\"graph-replay\",\"offered\":%d}\n", offered);
		fflush(stdout);
		return 0;
	}

	/* injection calls the entry point the CUDA driver calls in a library
	   named by CUDA_INJECTION64_PATH, twice, as a driver initialized twice
	   would, and then offers records. The journal must be the one the
	   constructor opened, with one bootstrap and nothing lost. */
	if (strcmp(scenario, "injection") == 0) {
		int first = InitializeInjection();
		int second = InitializeInjection();
		int offered = cudaprof_test_offer(buffers, buffers, buffers, 0);
		printf("{\"scenario\":\"injection\",\"offered\":%d,\"init_rc\":%d,\"init_rc_again\":%d}\n", offered, first, second);
		fflush(stdout);
		return 0;
	}

	/* modules loads <distinct> module images <repeats> times each, which is
	   how a target that loads one module per context looks to the producer.
	   Past the producer's duplicate table a repeat can no longer be
	   recognised, and the journal has to say so. */
	if (strcmp(scenario, "modules") == 0) {
		unsigned distinct = arg_uint(argc, argv, 2, 4);
		unsigned repeats = arg_uint(argc, argv, 3, 1);
		static const char image[16] = "\x7f" "ELF-test-cubin";
		int offered = 0;
		for (unsigned r = 0; r < repeats; r++)
			for (unsigned i = 1; i <= distinct; i++)
				offered += cudaprof_test_load_module(i, image, sizeof image);
		printf("{\"scenario\":\"modules\",\"offered\":%d}\n", offered);
		fflush(stdout);
		return 0;
	}

	/* module-chains drives each way a launch reaches its module, with fake
	   handles, and completes one kernel record per launch. The correlation
	   id names the case; the Go test holds the module id each must carry:

	     101  CUkernel from an eagerly loaded library, launched directly
	     102  CUfunction from cuKernelGetFunction on that kernel
	     103  CUfunction whose lazy library loads inside cuKernelGetFunction
	     104  the CUkernel behind 103, launched directly
	     105  CUkernel whose lazy library loads inside its first launch
	     106  runtime host stub whose module loads inside its first launch
	     107  the same stub launched again, with no load
	     108  a runtime stub launched after a load in an untracked call
	     109  CUfunction from cuModuleGetFunction */
	if (strcmp(scenario, "module-chains") == 0) {
		enum { D = CUPTI_CB_DOMAIN_DRIVER_API, R = CUPTI_CB_DOMAIN_RUNTIME_API };
		static const char image[16] = "\x7f" "ELF-test-cubin";
		int fired = 0;

		/* An eager library: the module loads inside cuLibraryLoadData. */
		CUlibrary lib1 = (CUlibrary)(uintptr_t)0x1000;
		cuLibraryLoadData_params load = {0};
		load.library = &lib1;
		fired += cudaprof_test_fire_api(D, CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadData, CUPTI_API_ENTER, &load, 1);
		fired += cudaprof_test_load_module(11, image, sizeof image);
		fired += cudaprof_test_fire_api(D, CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadData, CUPTI_API_EXIT, &load, 1);
		CUkernel kern1 = (CUkernel)(uintptr_t)0x2000;
		cuLibraryGetKernel_params get = {&kern1, lib1, "k1"};
		fired += cudaprof_test_fire_api(D, CUPTI_DRIVER_TRACE_CBID_cuLibraryGetKernel, CUPTI_API_ENTER, &get, 2);
		fired += cudaprof_test_fire_api(D, CUPTI_DRIVER_TRACE_CBID_cuLibraryGetKernel, CUPTI_API_EXIT, &get, 2);
		cuLaunchKernel_params launch = {0};
		launch.f = (CUfunction)kern1;
		fired += cudaprof_test_fire_api(D, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel, CUPTI_API_ENTER, &launch, 101);
		fired += cudaprof_test_fire_api(D, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel, CUPTI_API_EXIT, &launch, 101);
		CUfunction fn1 = (CUfunction)(uintptr_t)0x3000;
		cuKernelGetFunction_params kf = {&fn1, kern1};
		fired += cudaprof_test_fire_api(D, CUPTI_DRIVER_TRACE_CBID_cuKernelGetFunction, CUPTI_API_ENTER, &kf, 3);
		fired += cudaprof_test_fire_api(D, CUPTI_DRIVER_TRACE_CBID_cuKernelGetFunction, CUPTI_API_EXIT, &kf, 3);
		launch.f = fn1;
		fired += cudaprof_test_fire_api(D, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel, CUPTI_API_ENTER, &launch, 102);
		fired += cudaprof_test_fire_api(D, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel, CUPTI_API_EXIT, &launch, 102);

		/* A lazy library: nothing loads until cuKernelGetFunction. */
		CUlibrary lib2 = (CUlibrary)(uintptr_t)0x1100;
		load.library = &lib2;
		fired += cudaprof_test_fire_api(D, CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadFromFile, CUPTI_API_ENTER, &load, 4);
		fired += cudaprof_test_fire_api(D, CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadFromFile, CUPTI_API_EXIT, &load, 4);
		CUkernel kern2 = (CUkernel)(uintptr_t)0x2100;
		get.pKernel = &kern2;
		get.library = lib2;
		fired += cudaprof_test_fire_api(D, CUPTI_DRIVER_TRACE_CBID_cuLibraryGetKernel, CUPTI_API_ENTER, &get, 5);
		fired += cudaprof_test_fire_api(D, CUPTI_DRIVER_TRACE_CBID_cuLibraryGetKernel, CUPTI_API_EXIT, &get, 5);
		CUfunction fn2 = (CUfunction)(uintptr_t)0x3100;
		kf.pFunc = &fn2;
		kf.kernel = kern2;
		fired += cudaprof_test_fire_api(D, CUPTI_DRIVER_TRACE_CBID_cuKernelGetFunction, CUPTI_API_ENTER, &kf, 6);
		fired += cudaprof_test_load_module(12, image, sizeof image);
		fired += cudaprof_test_fire_api(D, CUPTI_DRIVER_TRACE_CBID_cuKernelGetFunction, CUPTI_API_EXIT, &kf, 6);
		launch.f = fn2;
		fired += cudaprof_test_fire_api(D, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel, CUPTI_API_ENTER, &launch, 103);
		fired += cudaprof_test_fire_api(D, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel, CUPTI_API_EXIT, &launch, 103);
		cuLaunchKernelEx_params ex = {0};
		ex.f = (CUfunction)kern2;
		fired += cudaprof_test_fire_api(D, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx, CUPTI_API_ENTER, &ex, 104);
		fired += cudaprof_test_fire_api(D, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx, CUPTI_API_EXIT, &ex, 104);

		/* A lazy library whose kernel is launched without a lookup. */
		CUlibrary lib3 = (CUlibrary)(uintptr_t)0x1200;
		load.library = &lib3;
		fired += cudaprof_test_fire_api(D, CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadData, CUPTI_API_ENTER, &load, 7);
		fired += cudaprof_test_fire_api(D, CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadData, CUPTI_API_EXIT, &load, 7);
		CUkernel kern3 = (CUkernel)(uintptr_t)0x2200;
		get.pKernel = &kern3;
		get.library = lib3;
		fired += cudaprof_test_fire_api(D, CUPTI_DRIVER_TRACE_CBID_cuLibraryGetKernel, CUPTI_API_ENTER, &get, 8);
		fired += cudaprof_test_fire_api(D, CUPTI_DRIVER_TRACE_CBID_cuLibraryGetKernel, CUPTI_API_EXIT, &get, 8);
		launch.f = (CUfunction)kern3;
		fired += cudaprof_test_fire_api(D, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel, CUPTI_API_ENTER, &launch, 105);
		fired += cudaprof_test_load_module(13, image, sizeof image);
		fired += cudaprof_test_fire_api(D, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel, CUPTI_API_EXIT, &launch, 105);

		/* The runtime, which launches by host stub. */
		cudaLaunchKernel_v7000_params rt = {0};
		rt.func = (const void *)(uintptr_t)0x4000;
		fired += cudaprof_test_fire_api(R, CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000, CUPTI_API_ENTER, &rt, 106);
		fired += cudaprof_test_load_module(14, image, sizeof image);
		fired += cudaprof_test_fire_api(R, CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000, CUPTI_API_EXIT, &rt, 106);
		cudaLaunchKernelExC_v11060_params rtex = {0};
		rtex.func = rt.func;
		fired += cudaprof_test_fire_api(R, CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060, CUPTI_API_ENTER, &rtex, 107);
		fired += cudaprof_test_fire_api(R, CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060, CUPTI_API_EXIT, &rtex, 107);
		fired += cudaprof_test_load_module(15, image, sizeof image);
		rt.func = (const void *)(uintptr_t)0x5000;
		fired += cudaprof_test_fire_api(R, CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000, CUPTI_API_ENTER, &rt, 108);
		fired += cudaprof_test_fire_api(R, CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000, CUPTI_API_EXIT, &rt, 108);

		/* The module API, unchanged. */
		CUmodule mod = (CUmodule)(uintptr_t)0x6000;
		cuModuleLoad_params ml = {&mod, "m.cubin"};
		fired += cudaprof_test_fire_api(D, CUPTI_DRIVER_TRACE_CBID_cuModuleLoad, CUPTI_API_ENTER, &ml, 9);
		fired += cudaprof_test_load_module(16, image, sizeof image);
		fired += cudaprof_test_fire_api(D, CUPTI_DRIVER_TRACE_CBID_cuModuleLoad, CUPTI_API_EXIT, &ml, 9);
		CUfunction fn4 = (CUfunction)(uintptr_t)0x3300;
		cuModuleGetFunction_params mg = {&fn4, mod, "k4"};
		fired += cudaprof_test_fire_api(D, CUPTI_DRIVER_TRACE_CBID_cuModuleGetFunction, CUPTI_API_ENTER, &mg, 10);
		fired += cudaprof_test_fire_api(D, CUPTI_DRIVER_TRACE_CBID_cuModuleGetFunction, CUPTI_API_EXIT, &mg, 10);
		launch.f = fn4;
		fired += cudaprof_test_fire_api(D, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel, CUPTI_API_ENTER, &launch, 109);
		fired += cudaprof_test_fire_api(D, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel, CUPTI_API_EXIT, &launch, 109);

		int offered = 0;
		for (uint32_t c = 101; c <= 109; c++) offered += cudaprof_test_offer_kernel(c);
		printf("{\"scenario\":\"module-chains\",\"offered\":%d,\"fired\":%d}\n", offered, fired);
		fflush(stdout);
		return 0;
	}

	/* host-sample idles while the producer's own host-state sampler runs.
	   The sampler reads whatever CUDAPROF_HOST_SAMPLE_ROOT names, so this
	   scenario deliberately does no CUPTI work: the thread set under test is
	   the fixture's, not this process's, and offering activity records here
	   would only add noise to the journal the control decodes. */
	if (strcmp(scenario, "host-sample") == 0) {
		unsigned millis = arg_uint(argc, argv, 2, 200);
		struct timespec ts = {millis / 1000u, (long)(millis % 1000u) * 1000000L};
		nanosleep(&ts, NULL);
		printf("{\"scenario\":\"host-sample\",\"slept_millis\":%u}\n", millis);
		fflush(stdout);
		return 0;
	}

	/* finalize-export drives cudaprof_finalize, the entry point a host calls
	   when its runtime will leave by exit_group. The scenario ends that way on
	   purpose, so the journal can only be complete if the call itself
	   completed it. The offers after the call must find every entry point
	   shut, and the second call must report the same finalized capture without
	   writing a second finalized record. */
	if (strcmp(scenario, "finalize-export") == 0) {
		int offered = cudaprof_test_offer_markers(buffers, buffers, buffers, 0, markers);
		int rc = cudaprof_finalize();
		int after = cudaprof_test_offer_markers(buffers, buffers, buffers, 0, markers);
		int again = cudaprof_finalize();
		printf("{\"scenario\":\"finalize-export\",\"offered\":%d,\"after\":%d,"
			"\"finalize_rc\":%d,\"finalize_rc_again\":%d}\n", offered, after, rc, again);
		fflush(stdout);
		_exit(0);
	}

	/* flush-interval is the target that cooperates with nothing: it never
	   fills a CUPTI buffer, never reaches the profiler API, and leaves by
	   exit_group. Its records sit staged inside the stub until something
	   flushes, so the journal holds them only if the producer's own interval
	   flush ran. The wait is several intervals long; the flush count says how
	   many actually happened. */
	if (strcmp(scenario, "flush-interval") == 0) {
		cudaprof_test_stage_for_flush(buffers, buffers, buffers);
		unsigned wait_ms = arg_uint(argc, argv, 6, 400);
		struct timespec ts = {wait_ms / 1000u, (long)(wait_ms % 1000u) * 1000000L};
		nanosleep(&ts, NULL);
		printf("{\"scenario\":\"flush-interval\",\"offered\":%u,\"flushes\":%u}\n",
			cudaprof_test_flush_delivered(), cudaprof_test_flush_count());
		fflush(stdout);
		_exit(0);
	}

	pthread_t trigger_id = 0;
	struct trigger t = {0, 0};
	if (strcmp(scenario, "clobber") == 0 || strcmp(scenario, "fork-inert-clobber") == 0 ||
	    strcmp(scenario, "runtime-exit") == 0 || strcmp(scenario, "runtime-exit-finalize") == 0) {
		struct sigaction sa;
		memset(&sa, 0, sizeof(sa));
		sa.sa_handler = app_sigterm;
		sigemptyset(&sa.sa_mask);
		if (sigaction(SIGTERM, &sa, NULL) != 0) { fprintf(stderr, "install app handler: %s\n", strerror(errno)); return 2; }
		if (strcmp(scenario, "clobber") == 0) {
			t.signo = SIGTERM;
			t.delay_micros = arg_uint(argc, argv, 6, 20000);
		}
	}
	else if (strcmp(scenario, "race") == 0) { t.signo = SIGTERM; t.delay_micros = arg_uint(argc, argv, 6, 20000); }
	else if (strcmp(scenario, "forced-kill") == 0) { t.signo = SIGKILL; t.delay_micros = arg_uint(argc, argv, 6, 20000); }
	if (t.signo != 0 && pthread_create(&trigger_id, NULL, trigger_thread, &t) != 0) {
		fprintf(stderr, "trigger thread: %s\n", strerror(errno));
		return 2;
	}

	long offered = 0, exhausted = 0;
	run_offers(plans, threads, &offered, &exhausted);
	if (trigger_id) pthread_join(trigger_id, NULL);

	/* A forked child inherits the agent but not its drain thread. It must
	   exit on its own rather than hang finalizing, and must leave the
	   parent's journal alone. The child exits through exit(3) so the ELF
	   destructor and atexit handler both run; _exit would skip the path
	   under test. */
	int child_exited = 0, child_status = -1;
	/* runtime-exit-fork: the child leaves the way its parent will, through
	   exit_group, and with a status of its own so the control can see that
	   the status got through. */
	/* runtime-exit-exec: the child execs this harness again as a runtime-exit
	   target, so it loads an agent of its own under the filter the parent
	   installed. Its report goes nowhere; the control reads its journal. */
	if (strcmp(scenario, "fork") == 0 || strcmp(scenario, "runtime-exit-fork") == 0 || strcmp(scenario, "runtime-exit-exec") == 0) {
		pid_t pid = fork();
		if (pid == 0) {
			if (strcmp(scenario, "runtime-exit-fork") == 0) _exit(5);
			if (strcmp(scenario, "runtime-exit-exec") == 0) {
				int null = open("/dev/null", O_WRONLY);
				if (null < 0 || dup2(null, 1) < 0) _exit(6);
				char n[4][16];
				snprintf(n[0], sizeof n[0], "%u", threads);
				snprintf(n[1], sizeof n[1], "%u", buffers);
				snprintf(n[2], sizeof n[2], "%u", per_buffer);
				snprintf(n[3], sizeof n[3], "%u", pause_micros);
				char *args[] = {argv[0], "runtime-exit", n[0], n[1], n[2], n[3], NULL};
				execv("/proc/self/exe", args);
				_exit(6);
			}
			exit(0);
		}
		if (pid < 0) { fprintf(stderr, "fork: %s\n", strerror(errno)); return 2; }
		for (int i = 0; i < 500; i++) {
			pid_t got = waitpid(pid, &child_status, WNOHANG);
			if (got == pid) { child_exited = 1; break; }
			if (got < 0) break;
			struct timespec ts = {0, 10000000L};
			nanosleep(&ts, NULL);
		}
		if (!child_exited) kill(pid, SIGKILL);
	}


	/* Fork inertness, the narrow question 6DE59D5A asked. The child probes
	   every entry point it inherited and hands the numbers back, then raises
	   SIGTERM at itself. Two things are under test at once: the child must be
	   refused a buffer, refused both external wrappers, and hold neither the
	   parent's journal nor either end of its termination pipe; and the child's
	   SIGTERM must kill the child by the disposition the target had before
	   load without waking the parent's terminator. The parent proves the
	   second half by signalling itself afterwards and still finalizing, which
	   it cannot do if the child's signal already consumed its pipe byte. */
	struct child_probe probe = {-1, -1, -1, -1, -1};
	int parent_journal_fds = count_fds("/capture.journal", NULL);
	int parent_pipe_fds = count_fds(NULL, "pipe:");
	int probe_read = 0;
	int child_signaled = 0, child_termsig = -1;
	int parent_push_result = -1, parent_pop_result = -1;
	if (strcmp(scenario, "fork-inert") == 0) {
		/* Probe the wrappers in the parent first. Without this the child's
		   refusal would prove nothing: these wrappers also refuse a run that
		   never had external correlation enabled at all. */
		uint64_t parent_popped = 0;
		parent_push_result = (int)cudaprof_push_external(1, 98);
		parent_pop_result = (int)cudaprof_pop_external(1, &parent_popped);
		int report[2];
		if (pipe(report) != 0) { fprintf(stderr, "pipe: %s\n", strerror(errno)); return 2; }
		pid_t pid = fork();
		if (pid == 0) {
			close(report[0]);
			struct child_probe p;
			p.buffer_granted = cudaprof_test_request_buffer();
			uint64_t popped = 0;
			p.push_result = (int)cudaprof_push_external(1, 99);
			p.pop_result = (int)cudaprof_pop_external(1, &popped);
			p.journal_fds = count_fds("/capture.journal", NULL);
			p.pipe_fds = count_fds(NULL, "pipe:");
			/* The report pipe the child just inherited and still holds is one
			   of the pipe descriptors it counts; discount it so the number
			   means the producer's pipe alone. */
			if (p.pipe_fds > 0) p.pipe_fds--;
			ssize_t w = write(report[1], &p, sizeof(p));
			(void)w;
			close(report[1]);
			raise(SIGTERM);
			/* Only reached if SIGTERM was not restored to a disposition that
			   ends the child; the nonzero exit distinguishes that from a
			   child killed by the signal. */
			_exit(9);
		}
		if (pid < 0) { fprintf(stderr, "fork: %s\n", strerror(errno)); return 2; }
		close(report[1]);
		if (read(report[0], &probe, sizeof(probe)) == (ssize_t)sizeof(probe)) probe_read = 1;
		close(report[0]);
		if (waitpid(pid, &child_status, 0) == pid) {
			child_exited = WIFEXITED(child_status);
			child_signaled = WIFSIGNALED(child_status);
			if (child_signaled) child_termsig = WTERMSIG(child_status);
		}
	}
	/* An application handler installed after the producer loaded belongs to
	   the application. The child must inherit it untouched: restoring the
	   producer's saved disposition there would replace the target's handler
	   with SIG_DFL and change how the child terminates, which is the one thing
	   the producer promises never to do. The child raises SIGTERM at itself
	   and exits 5 only if the application's handler ran. */
	if (strcmp(scenario, "fork-inert-clobber") == 0) {
		uint64_t parent_popped = 0;
		parent_push_result = (int)cudaprof_push_external(1, 96);
		parent_pop_result = (int)cudaprof_pop_external(1, &parent_popped);
		int report[2];
		if (pipe(report) != 0) { fprintf(stderr, "pipe: %s\n", strerror(errno)); return 2; }
		pid_t pid = fork();
		if (pid == 0) {
			close(report[0]);
			struct child_probe p;
			p.buffer_granted = cudaprof_test_request_buffer();
			uint64_t popped = 0;
			p.push_result = (int)cudaprof_push_external(1, 97);
			p.pop_result = (int)cudaprof_pop_external(1, &popped);
			p.journal_fds = count_fds("/capture.journal", NULL);
			p.pipe_fds = count_fds(NULL, "pipe:");
			if (p.pipe_fds > 0) p.pipe_fds--;
			ssize_t w = write(report[1], &p, sizeof(p));
			(void)w;
			close(report[1]);
			raise(SIGTERM);
			_exit(app_terminated ? 5 : 6);
		}
		if (pid < 0) { fprintf(stderr, "fork: %s\n", strerror(errno)); return 2; }
		close(report[1]);
		if (read(report[0], &probe, sizeof(probe)) == (ssize_t)sizeof(probe)) probe_read = 1;
		close(report[0]);
		if (waitpid(pid, &child_status, 0) == pid) {
			child_exited = WIFEXITED(child_status);
			child_signaled = WIFSIGNALED(child_status);
			if (child_signaled) child_termsig = WTERMSIG(child_status);
			if (child_exited) child_status = WEXITSTATUS(child_status);
		}
	}

	printf("{\"scenario\":\"%s\",\"threads\":%u,\"buffers\":%u,\"per_buffer\":%u,\"offered\":%ld,\"pool_exhausted\":%ld,\"app_terminated\":%d,\"child_exited\":%d,\"child_status\":%d,\"child_signaled\":%d,\"child_termsig\":%d,\"probe_read\":%d,\"parent_push_result\":%d,\"parent_pop_result\":%d,\"parent_journal_fds\":%d,\"parent_pipe_fds\":%d,\"child_buffer_granted\":%d,\"child_push_result\":%d,\"child_pop_result\":%d,\"child_journal_fds\":%d,\"child_pipe_fds\":%d}\n",
		scenario, threads, buffers, per_buffer, offered, exhausted, app_terminated ? 1 : 0, child_exited, child_status,
		child_signaled, child_termsig, probe_read, parent_push_result, parent_pop_result, parent_journal_fds, parent_pipe_fds,
		probe.buffer_granted, probe.push_result, probe.pop_result, probe.journal_fds, probe.pipe_fds);
	fflush(stdout);
	if (strcmp(scenario, "fork-inert") == 0) {
		/* The parent's own termination service must have survived the child's
		   signal: this SIGTERM has to reach the producer's handler, wake its
		   terminator, and finalize the journal. */
		raise(SIGTERM);
		for (int i = 0; i < 100; i++) { struct timespec ts = {0, 100000000L}; nanosleep(&ts, NULL); }
	}
	if (strcmp(scenario, "clobber") == 0) {
		/* The target owns SIGTERM now, so it winds down the way it always
		   would and the producer finalizes from atexit. */
		for (int i = 0; i < 100 && !app_terminated; i++) { struct timespec ts = {0, 10000000L}; nanosleep(&ts, NULL); }
		return app_terminated ? 7 : 8;
	}
	/* Stands in for a language runtime: it owns SIGTERM, and it leaves through
	   the exit_group syscall rather than through exit(3), so neither the ELF
	   destructor nor atexit runs. _exit(2) is that shape -- Go reaches it by
	   issuing the syscall itself, which is not something a C control can do
	   portably, but the agent cannot tell the difference: what matters is that
	   libc's exit sequence does not run.

	   The two arms differ by one call. Without it the journal ends unfinalized
	   with no activity, which is the defect. With it the target finalizes for
	   itself, which is the only route left once the other three are gone. */
	if (strcmp(scenario, "runtime-exit") == 0 || strcmp(scenario, "runtime-exit-finalize") == 0) {
		if (strcmp(scenario, "runtime-exit-finalize") == 0) cudaprof_finalize();
		_exit(0);
	}
	/* A status other than 0, so the control can tell the target's own status
	   from one the producer made up. */
	if (strcmp(scenario, "runtime-exit-fork") == 0 || strcmp(scenario, "runtime-exit-exec") == 0) _exit(3);
	if (strcmp(scenario, "race") == 0 || strcmp(scenario, "forced-kill") == 0) {
		/* Stay alive so the signal decides how the process ends. */
		for (int i = 0; i < 100; i++) { struct timespec ts = {0, 100000000L}; nanosleep(&ts, NULL); }
	}
	return 0;
}
