#define _GNU_SOURCE
/*
 * Loadable CUPTI activity producer for Linux. Build this file only against a
 * qualified CUDA toolkit; it is deliberately separate from the portable C11
 * bootstrap and from Go collector/decoder packages.
 */
#include <cupti.h>
#include <cupti_activity.h>
#include <cupti_pcsampling.h>
#include <cupti_version.h>
#if CUPTI_API_VERSION < 26
/* CUpti_ActivityMemcpy6 and CUpti_ActivityUnifiedMemoryCounter3 first
   appear in CUPTI 12.8, whose CUPTI_API_VERSION is 26. */
#error "cupti_agent needs CUPTI 12.8 or newer"
#endif
#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>
#ifdef __linux__
#include <link.h>
#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <sys/ioctl.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/un.h>
#include <unwind.h>
#endif

/* CUDAPROF_AGENT_REVISION is set by the build to the revision of the source
   this agent was compiled from. Empty means the build did not say. */
#ifndef CUDAPROF_AGENT_REVISION
#define CUDAPROF_AGENT_REVISION ""
#endif

/* CUDAPROF_CAPABILITY_BUFFER is the capability record's formatting buffer.
   It is a macro so a test can build the agent with a buffer too small to hold
   the record and exercise the truncation path, which is otherwise reachable
   only by adding fields until it overflows. */
#ifndef CUDAPROF_CAPABILITY_BUFFER
#define CUDAPROF_CAPABILITY_BUFFER 16384
#endif

#define WIRE_HEADER 80
#define MAX_PAYLOAD 4096
#define BUFFER_BYTES (8u << 20)
/* CUPTI asks for a buffer whenever every buffer it holds is full, and it drops
   records outright -- counted in native_dropped, not in this agent's own
   accounting -- for as long as the request goes unanswered. A pool of 4 is
   enough only for a workload that launches slowly. A 2048-token ds4-min
   prefill launches 548,145 activities in 23 s and lost 513,943 of them to
   native_dropped at BUFFER_POOL 4; at 64 the same capture lost none. The ring
   below was never the constraint in either run -- the agent retained 100% of
   what it was offered even while CUPTI was dropping -- so PENDING_RECORDS
   stays where it is. The pool is static BSS and demand-paged: only the buffers
   CUPTI actually holds at once are ever touched. */
#define BUFFER_POOL 64
#define PENDING_RECORDS 2048
/* Records drained per writev. CUPTI hands the buffer-completed callback a
   whole 8 MiB activity buffer in one burst; writing one record per write(2)
   cannot keep up with that and the ring refuses admission. Batching bounds
   the drain at one syscall and one journal-lock acquisition per DRAIN_BATCH
   records. Two iovecs per record must stay under IOV_MAX. */
#define DRAIN_BATCH 256
/* How long admission will wait for the drain to free a slot before charging
   the record to queue_capacity_dropped. Overridable only so a control can set
   it to zero and exercise the real drop path under real pressure.

   This bounds one record's wait, not one callback's: a CUPTI buffer callback
   offers thousands of records and pays this wait per record that finds the
   ring full. Under the burst control, callbacks reach half a second while no
   single admission waits more than a few tens of milliseconds.

   The wait ends when the drain publishes a freed tail, which it does only
   after its batch write returns, so this deadline is really a bound on
   journal write latency. Measured over 2,500 batches on a quiet Darwin/arm64
   host the longest admission wait equals the longest batch write to the
   millisecond -- 16 to 40 ms -- and under concurrent host I/O that pair rises
   together to 99 ms. A write slower than this deadline is what a capacity
   drop is made of; queue_backpressure_max_nanos and drain_write_max_nanos in
   the finalized record are there to say so rather than leave the loss
   unattributed. */
#ifndef ADMIT_WAIT_NANOS
#define ADMIT_WAIT_NANOS (100 * 1000 * 1000L)
#endif
#define KIND_BOOTSTRAP 1
#define KIND_ACTIVITY 2
#define KIND_FINALIZED 3
#define KIND_CAPABILITY 4
#define KIND_FAILURE 5
#define KIND_EXTERNAL 6
#define KIND_API 7
#define KIND_HOST 8
/* A module is not an activity: it is not work the device did and it has no
   duration. Giving it its own kind keeps it out of the activity counts and out
   of the sample projection, where a zero-length record would read as a kernel
   that took no time. */
#define KIND_MODULE 9
/* One folded PC sample, or the run's sampling summary. Gated on
   CUDAPROF_PCSAMPLING for the same reason modules are gated: a consumer that
   meets a kind it cannot decode refuses the whole journal. */
#define KIND_PCSAMPLE 10
/* One cuBLASLt call, or the link from a call to a kernel launch it made.
   Gated on CUDAPROF_GEMMS for the same reason again. */
#define KIND_GEMM 11

/* Host-state sampling. CUPTI can say the device was idle for 8 ms; it cannot
   say what the host was doing during those 8 ms, so every device gap reads the
   same whether the host was computing, blocked on a copy, or asleep on a
   futex. This tier answers that question at the cheapest resolution that
   answers it at all: a thread reads each of this process's /proc task stat
   files on a timer and transcribes the run state and the two CPU-time
   counters. No signals, no stack unwinding, no privilege. It cannot say WHICH
   host code, only which of R/S/D/Z each thread was in, which -- joined against
   the KIND_API stream this agent already collects -- localizes most gaps.

   The alternative, a signal-based sampler with backtraces, is a different
   project: glibc's backtrace() takes the loader lock through dl_iterate_phdr
   and CUDA dlopens lazily throughout a run, so a sample landing inside a
   dlopen can deadlock the target. Nothing here runs in signal context.

   Off unless CUDAPROF_HOST_SAMPLE names a rate, because this agent has no
   measurement of the cost on a Linux host and will not default something on
   against an unmeasured budget. */
#define HOST_SAMPLE_MAX_HZ 1000
#define HOST_SAMPLE_MAX_THREADS 4096
/* Room reserved at the end of a host-sample payload for the fields that follow
   the thread array. Every part is closed with them, so admission of a thread
   entry has to leave space for the close or the record cannot be terminated. */
#define HOST_SAMPLE_SUFFIX_RESERVE 224
/* A root longer than this is refused rather than truncated into the capability
   record, which is a fixed 1024-byte buffer whose overflow would suppress the
   capability record and with it the whole capture. */
#define HOST_SAMPLE_ROOT_MAX 200

/* CUDA 12.x exposes Kernel9; CUDA 13.0 exposes the compatible Kernel10
   extension. Keep the decoder tied to the headers used for this build. */
#if CUPTI_API_VERSION >= 130000
#define CUDAPROF_KERNEL_TYPE CUpti_ActivityKernel10
#define CUDAPROF_KERNEL_LAYOUT "Kernel10"
#define CUDAPROF_GRAPH_ID(k) ((uint32_t)(k)->graphId)
#define CUDAPROF_GRAPH_NODE_ID(k) ((uint64_t)(k)->graphNodeId)

_Static_assert(sizeof(((CUDAPROF_KERNEL_TYPE *)0)->graphId) == 4, "unexpected graph id width");
_Static_assert(sizeof(((CUDAPROF_KERNEL_TYPE *)0)->graphNodeId) == 8, "unexpected graph node id width");
#else
#define CUDAPROF_KERNEL_TYPE CUpti_ActivityKernel9
#define CUDAPROF_KERNEL_LAYOUT "Kernel9"
#define CUDAPROF_GRAPH_ID(k) ((uint32_t)(k)->graphId)
#define CUDAPROF_GRAPH_NODE_ID(k) ((uint64_t)(k)->graphNodeId)
#endif

_Static_assert(sizeof(CUpti_ActivityExternalCorrelation) == 24, "unexpected external correlation size");
_Static_assert(offsetof(CUpti_ActivityExternalCorrelation, kind) == 0, "unexpected external kind offset");
_Static_assert(offsetof(CUpti_ActivityExternalCorrelation, externalKind) == 4, "unexpected external API offset");
_Static_assert(offsetof(CUpti_ActivityExternalCorrelation, externalId) == 8, "unexpected external ID offset");
_Static_assert(offsetof(CUpti_ActivityExternalCorrelation, correlationId) == 16, "unexpected CUDA correlation offset");
_Static_assert(offsetof(CUpti_ActivityExternalCorrelation, reserved) == 20, "unexpected external reserved offset");

/* An NVTX range is two records, not one: CUPTI reports the open and the close
   separately, each carrying the range id and its own timestamp, so the agent
   transcribes them as it receives them and the pairing happens downstream
   where a map is affordable. */
_Static_assert(sizeof(CUpti_ActivityMarker2) == 56, "unexpected marker size");
_Static_assert(offsetof(CUpti_ActivityMarker2, flags) == 4, "unexpected marker flags offset");
_Static_assert(offsetof(CUpti_ActivityMarker2, timestamp) == 8, "unexpected marker timestamp offset");
_Static_assert(offsetof(CUpti_ActivityMarker2, id) == 16, "unexpected marker id offset");
_Static_assert(offsetof(CUpti_ActivityMarker2, name) == 40, "unexpected marker name offset");
_Static_assert(offsetof(CUpti_ActivityMarker2, domain) == 48, "unexpected marker domain offset");

/* Capture-range gating reads callbackSite out of a CUpti_CallbackData the
   library owns, so the two builds must agree on where it sits. The offsets
   below are from CUDA 13.0 cupti_callbacks.h; the enumerator values the gate
   compares against are carried independently by nvidia/cupti/enums.gen.go and
   are unchanged across CUDA 12.6 through 13.1. */
_Static_assert(offsetof(CUpti_CallbackData, callbackSite) == 0, "unexpected callback site offset");
_Static_assert(offsetof(CUpti_CallbackData, functionName) == 8, "unexpected callback name offset");
_Static_assert(offsetof(CUpti_CallbackData, correlationId) == 64, "unexpected callback correlation offset");
_Static_assert(CUPTI_API_ENTER == 0 && CUPTI_API_EXIT == 1, "unexpected callback site values");
_Static_assert(CUPTI_CB_DOMAIN_DRIVER_API == 1 && CUPTI_CB_DOMAIN_RUNTIME_API == 2, "unexpected callback domain values");

static int journal_fd = -1;
static char worker_dir[PATH_MAX];
static unsigned char session_id[16];
static uint64_t generation;
static uint64_t sequence = 1;
static _Atomic uint64_t activity_count;
static _Atomic uint64_t dropped_count;
static _Atomic uint64_t native_dropped_count;
static _Atomic uint64_t queue_dropped_count;
static _Atomic uint64_t queue_oversize_dropped;
static _Atomic uint64_t queue_contention_dropped;
static _Atomic uint64_t queue_capacity_dropped;
static _Atomic uint64_t offered_count;

/* Per-kind admission accounting. offered_count, dropped_count and the three
   drop categories are kind-agnostic -- every record kind shares one ring and
   one admission path -- so a loss figure derived from them describes all kinds
   together. Dividing a kind-specific retained count by an all-kinds dropped
   count compares different populations, which is exactly how a receipt came to
   report an "activity loss" rate that no artifact established. These carry the
   kind identity through admission and out to the finalized record so that a
   per-kind loss statement is derivable rather than inferred. Indexed by the
   KIND_ constants; only kinds that pass through the ring are ever nonzero. */
#define KIND_COUNT 12
static _Atomic uint64_t offered_by_kind[KIND_COUNT];
static _Atomic uint64_t retained_by_kind[KIND_COUNT];
static _Atomic uint64_t dropped_by_kind[KIND_COUNT];

static void count_kind(_Atomic uint64_t *counters, uint16_t kind, uint64_t n) {
	if (kind < KIND_COUNT) atomic_fetch_add(&counters[kind], n);
}
static _Atomic uint64_t queue_backpressure_waits;
static _Atomic uint64_t queue_backpressure_nanos;
/* The longest single admission wait. The totals above say how much latency
   backpressure spent; only the maximum says how close any one record came to
   ADMIT_WAIT_NANOS, which is the boundary a capacity drop crosses. */
static _Atomic uint64_t queue_backpressure_max_nanos;
static _Atomic uint64_t drain_batches;
static _Atomic uint64_t drain_records;
static _Atomic uint64_t drain_max_batch;
static _Atomic uint64_t drain_write_nanos;
/* The longest single batch write. Admission waits on exactly this: the drain
   publishes a freed tail only after its write returns, so a write longer than
   ADMIT_WAIT_NANOS is what a capacity drop is made of. */
static _Atomic uint64_t drain_write_max_nanos;
static _Atomic uint64_t callback_count;
static _Atomic uint64_t callback_total_nanos;
static _Atomic uint64_t callback_max_nanos;
static _Atomic int producer_failed;
static _Atomic unsigned callback_active;
static _Atomic int shutdown_started;
static _Atomic int callbacks_closed;
static pthread_mutex_t journal_mu = PTHREAD_MUTEX_INITIALIZER;
static int initialized;
static int cupti_ready;
static int concurrent_enabled;
static int legacy_enabled;
static int memcpy_enabled;
static int memset_enabled;
static int overhead_enabled;
static int graph_trace_enabled;
/* graph_trace_mode is the graph trace mode the capability record reports:
   "node", "graph" or "requested_unavailable". See graph_trace_requested. */
static const char *graph_trace_mode = "node";
static int sync_enabled;
static int external_enabled;
static int marker_enabled;
static int runtime_enabled;
static int driver_enabled;
static int callback_registered;

/* Capture-range gating. When CUDAPROF_CAPTURE_RANGE names the profiler API,
   the kinds above are enabled at init as usual -- that is how the agent learns
   which of them this CUPTI supports, and a refusal must still be reported at
   init rather than at the first range -- and then collection is switched off
   until the target calls cudaProfilerStart. The *_enabled flags therefore mean
   "this kind is ours to toggle", not "records are arriving right now";
   collecting says which of those two states the agent is in.

   Both the runtime and the driver pair are subscribed: a target that calls
   cuProfilerStart directly never reaches the runtime entry point, and a gate
   that watched only the runtime would silently never open for it. */
static CUpti_SubscriberHandle range_subscriber;
static int graph_lifecycle_enabled;
static int graph_lifecycle_requested;
static unsigned graph_runtime_replay_enabled;
static int range_subscribed;
#define RANGE_NONE 0
#define RANGE_GATED 1
#define RANGE_UNAVAILABLE 2
static int range_gated;
static atomic_int collecting;
static atomic_uint range_opens;
_Static_assert(CUPTI_RUNTIME_TRACE_CBID_cudaProfilerStart_v4000 == 171, "unexpected cudaProfilerStart cbid");
_Static_assert(CUPTI_RUNTIME_TRACE_CBID_cudaProfilerStop_v4000 == 172, "unexpected cudaProfilerStop cbid");
_Static_assert(CUPTI_DRIVER_TRACE_CBID_cuProfilerStart == 308, "unexpected cuProfilerStart cbid");
_Static_assert(CUPTI_DRIVER_TRACE_CBID_cuProfilerStop == 309, "unexpected cuProfilerStop cbid");
/* Finalization policy. The ELF destructor, atexit and the SIGTERM terminator
   all describe a process that winds down through libc. A Go program does not:
   its runtime leaves through exit_group, which runs neither a destructor nor
   an atexit handler, so a Go target that returns from main keeps whatever
   CUPTI was still holding and its journal ends with no finalized record --
   measured as 0 activity records against 51 from the same capture of a C
   target. CUDAPROF_FINALIZE=profiler-stop makes the target's own
   cudaProfilerStop the trigger instead, which is a point the target chooses
   and reaches.

   It is opt-in because a stop is not by itself the end of a capture: a target
   may stop and start collection again, and finalizing on every stop would end
   the journal at the first one. The mode is reported in the capability record,
   including when it was asked for and the stop callback could not be armed, so
   a reader never assumes a trigger that is not there.

   CUDAPROF_FINALIZE=exit-group needs nothing from the target: it catches the
   exit_group syscall itself. See install_exit_group. */
#define FINALIZE_EXIT 0
#define FINALIZE_PROFILER_STOP 1
#define FINALIZE_UNAVAILABLE 2
#define FINALIZE_EXIT_GROUP 3
static int finalize_mode;
static _Atomic int finalize_started;
static _Atomic int finalize_done;
/* device_active is set by the first buffer request, journal_ready by the
   constructor once bootstrap and capability are down, and device_active_written
   by whichever of the two publishes the record. journal_ready also holds the
   interval flusher until the journal can accept an activity record. */
static _Atomic int device_active;
static _Atomic int device_active_written;
static _Atomic int journal_ready;
static const char *finalize_trigger;
static uint64_t self_pid;
static unsigned char buffer_pool[BUFFER_POOL][BUFFER_BYTES] __attribute__((aligned(64)));
static _Atomic int buffer_busy[BUFFER_POOL];
struct pending_record { uint16_t kind; uint64_t timestamp; size_t length; char payload[MAX_PAYLOAD]; };
static struct pending_record pending[PENDING_RECORDS];
static size_t pending_head, pending_tail;
static pthread_mutex_t pending_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t pending_cond = PTHREAD_COND_INITIALIZER;
static pthread_cond_t pending_space_cond = PTHREAD_COND_INITIALIZER;
static int worker_stop;
static pthread_t worker;

/* Optional interval flush. CUPTI hands a completed buffer to buffer_completed
   only when that buffer fills or when something flushes it, so a target whose
   buffers never fill and which never reaches a finalization trigger leaves
   every record inside CUPTI. CUDAPROF_FLUSH_INTERVAL_MS asks for a flush every
   so many milliseconds, which bounds how much of a run is held there. Records
   published this way reach the journal with no finalized record after them,
   which is what they are: a partial capture, and the consumer reads them as
   one.

   The flush runs on its own thread rather than in the drain because
   cuptiActivityFlushAll delivers the buffer on the thread that called it. A
   drain thread that flushed would be inside buffer_completed offering a whole
   buffer to a ring that only it can empty; every record past PENDING_RECORDS
   would wait out ADMIT_WAIT_NANOS and then be dropped for capacity. */
static long flush_interval_ms;
static pthread_t flusher;
static int flush_started;
static int flush_stop;
static pthread_mutex_t flush_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t flush_cond = PTHREAD_COND_INITIALIZER;
/* Drain-owned scratch. Only the single drain thread touches these. */
static unsigned char batch_headers[DRAIN_BATCH][WIRE_HEADER];
static struct iovec batch_iov[DRAIN_BATCH * 2];

/* SIGTERM finalization. The handler does nothing but an atomic store and a
   one-byte self-pipe write; the terminator thread performs the ordered CUPTI
   flush, disable, drain and journal close in a normal execution context. */
static int term_pipe[2] = {-1, -1};
static _Atomic int term_signalled;
static pthread_t terminator;
static struct sigaction prev_sigterm;
static int sigterm_installed;
/* exit_group finalization: the seccomp listener. See install_exit_group. */
static int exit_listener = -1;
/* The rendezvous socket a descendant listens on instead; see
   install_exit_group. */
static int exit_rendezvous = -1;
/* Whether the fork guard is registered; reported in the capability record so a
   run never silently assumes it. */
static int atfork_guard;

/* Which clock CUPTI stamped its records with, and therefore whether a CUPTI
   timestamp and an agent timestamp name the same instant. CUPTI's platform
   default on Linux is clock_gettime(CLOCK_REALTIME) while every timestamp this
   agent takes itself is CLOCK_MONOTONIC, so the two are separated by the
   wall-clock time the host booted: subtracting one from the other yields
   decades, not a skew. Registering a
   timestamp callback replaces CUPTI's source with this agent's, which is the
   only way to make the journal's two timestamp origins comparable. A CUPTI
   that cannot do it is a supported configuration; a capture that silently
   pretends otherwise is not, so the outcome is named in the capability
   record. */
enum {
	AGENT_CLOCK_UNIFIED = 0,    /* CUPTI stamps through cupti_timestamp */
	AGENT_CLOCK_UNSUPPORTED,    /* runtime CUPTI does not export the API */
	AGENT_CLOCK_REFUSED,        /* CUPTI exports it and rejected the callback */
};
static int clock_state = AGENT_CLOCK_UNSUPPORTED;

static uint64_t now_ns(void);
static int write_record(uint16_t, uint64_t, const char *, size_t);
static void cudaprof_cupti_atexit(void);
static void cudaprof_cupti_finalize(const char *trigger, int in_range_callback);

static int write_batch(size_t tail, size_t n);

static int queue_pending(uint16_t kind, uint64_t timestamp, const char *payload, size_t length) {
	atomic_fetch_add(&offered_count, 1);
	count_kind(offered_by_kind, kind, 1);
	if (length > MAX_PAYLOAD) { atomic_fetch_add(&dropped_count, 1); atomic_fetch_add(&queue_dropped_count, 1); atomic_fetch_add(&queue_oversize_dropped, 1); count_kind(dropped_by_kind, kind, 1); atomic_store(&producer_failed, 1); return -1; }
	/* Callback code must not block on transport I/O. It no longer can: every
	   holder of this mutex does a bounded payload copy or a pair of index
	   updates, and the drain releases it before its write. A brief spin
	   absorbs the common contention, then admission waits rather than
	   refusing, because refusing here loses a record that the queue had room
	   for. queue_contention_dropped is retained in the finalized record for
	   receipt continuity across producer versions and is now structurally
	   zero. */
	int locked = 0;
	for (unsigned attempt = 0; attempt < 64; attempt++) {
		if (pthread_mutex_trylock(&pending_mu) == 0) { locked = 1; break; }
		sched_yield();
	}
	if (!locked && pthread_mutex_lock(&pending_mu) != 0) { atomic_fetch_add(&dropped_count, 1); atomic_fetch_add(&queue_dropped_count, 1); atomic_fetch_add(&queue_contention_dropped, 1); count_kind(dropped_by_kind, kind, 1); atomic_store(&producer_failed, 1); return -1; }
	if (pending_head - pending_tail >= PENDING_RECORDS) {
		/* The drain owns the ring and is running, so space becomes available
		   in bounded time. Waiting here spends CUPTI callback latency instead
		   of losing the record. The wait carries an absolute deadline and is
		   abandoned once callbacks are closed or the drain is stopping, so it
		   can never block finalization. */
		uint64_t began = now_ns();
		struct timespec deadline;
		if (clock_gettime(CLOCK_REALTIME, &deadline) == 0) {
			deadline.tv_sec += ADMIT_WAIT_NANOS / 1000000000L;
			deadline.tv_nsec += ADMIT_WAIT_NANOS % 1000000000L;
			if (deadline.tv_nsec >= 1000000000L) { deadline.tv_nsec -= 1000000000L; deadline.tv_sec++; }
			while (pending_head - pending_tail >= PENDING_RECORDS && !worker_stop && !atomic_load(&callbacks_closed)) {
				if (pthread_cond_timedwait(&pending_space_cond, &pending_mu, &deadline) == ETIMEDOUT) break;
			}
		}
		uint64_t waited = now_ns() - began;
		atomic_fetch_add(&queue_backpressure_waits, 1);
		atomic_fetch_add(&queue_backpressure_nanos, waited);
		uint64_t longest = atomic_load(&queue_backpressure_max_nanos);
		while (waited > longest && !atomic_compare_exchange_weak(&queue_backpressure_max_nanos, &longest, waited)) { }
	}
	if (pending_head - pending_tail >= PENDING_RECORDS) { atomic_fetch_add(&dropped_count, 1); atomic_fetch_add(&queue_dropped_count, 1); atomic_fetch_add(&queue_capacity_dropped, 1); count_kind(dropped_by_kind, kind, 1); atomic_store(&producer_failed, 1); pthread_mutex_unlock(&pending_mu); return -1; }
	struct pending_record *r = &pending[pending_head % PENDING_RECORDS];
	r->kind = kind; r->timestamp = timestamp; r->length = length; memcpy(r->payload, payload, length);
	pending_head++; pthread_cond_signal(&pending_cond); pthread_mutex_unlock(&pending_mu); return 0;
}

/* The drain thread is the only consumer, so slots in [tail, head) are its own
   until it publishes a new tail. It may therefore serialize and write them
   without holding the admission mutex, which keeps producers admitting while
   a batch is in flight. */
static void *drain_worker(void *arg) {
	(void)arg;
	for (;;) {
		pthread_mutex_lock(&pending_mu);
		while (pending_head == pending_tail && !worker_stop) pthread_cond_wait(&pending_cond, &pending_mu);
		size_t head = pending_head, tail = pending_tail;
		pthread_mutex_unlock(&pending_mu);
		if (head == tail) return NULL; /* stopping with an empty ring */
		size_t n = head - tail;
		if (n > DRAIN_BATCH) n = DRAIN_BATCH;
		int err = write_batch(tail, n);
		pthread_mutex_lock(&pending_mu);
		pending_tail = tail + n;
		pthread_cond_broadcast(&pending_space_cond);
		pthread_mutex_unlock(&pending_mu);
		if (err != 0) atomic_store(&producer_failed, 1);
	}
}

#ifdef CUDAPROF_TEST_HARNESS
/* cudaprof_test_wait_drain is a test-control acknowledgement. Unlike a sleep,
   it lets a producer scenario submit its next buffer only after this one has
   reached the journal, keeping the bounded-retention test independent of
   scheduler delays. */
int cudaprof_test_wait_drain(uint64_t records, unsigned timeout_ms) {
	struct timespec deadline;
	if (clock_gettime(CLOCK_REALTIME, &deadline) != 0) return -1;
	deadline.tv_sec += timeout_ms / 1000;
	deadline.tv_nsec += (long)(timeout_ms % 1000) * 1000000L;
	if (deadline.tv_nsec >= 1000000000L) {
		deadline.tv_nsec -= 1000000000L;
		deadline.tv_sec++;
	}

	pthread_mutex_lock(&pending_mu);
	while (atomic_load(&drain_records) < records) {
		int err = pthread_cond_timedwait(&pending_space_cond, &pending_mu, &deadline);
		if (err != 0) {
			pthread_mutex_unlock(&pending_mu);
			return -1;
		}
	}
	pthread_mutex_unlock(&pending_mu);
	return 0;
}
#endif

/* Flushes CUPTI every flush_interval_ms until finalization stops it. CUPTI
   delivers the buffers on this thread, so the records take the ordinary
   buffer_completed path and the drain -- a different thread -- keeps emptying
   the ring underneath. A flush is skipped once shutdown has begun: finalize
   owns the ordered CUPTI sequence from that point and a second thread inside
   CUPTI would race it. */
static void *flush_worker(void *arg) {
	(void)arg;
	for (;;) {
		struct timespec deadline;
		if (clock_gettime(CLOCK_REALTIME, &deadline) != 0) return NULL;
		deadline.tv_sec += flush_interval_ms / 1000;
		deadline.tv_nsec += (flush_interval_ms % 1000) * 1000000L;
		if (deadline.tv_nsec >= 1000000000L) { deadline.tv_nsec -= 1000000000L; deadline.tv_sec++; }
		pthread_mutex_lock(&flush_mu);
		while (!flush_stop) {
			if (pthread_cond_timedwait(&flush_cond, &flush_mu, &deadline) == ETIMEDOUT) break;
		}
		int stop = flush_stop;
		pthread_mutex_unlock(&flush_mu);
		if (stop) return NULL;
		if (atomic_load(&shutdown_started)) continue;
		/* The thread is created before the bootstrap and capability records
		   are written, because the capability record reports whether creating
		   it worked. A flush before those two would hand the consumer an
		   activity record ahead of them, and it refuses the whole journal for
		   that. journal_ready is stored once both are down. */
		if (!atomic_load(&journal_ready)) continue;
		(void)cuptiActivityFlushAll(0);
	}
}

/* Stops the interval flusher and waits for it to leave. Finalization calls
   this before its own CUPTI sequence, so that from there on exactly one thread
   is inside CUPTI, and before it stops the drain, because a flush still in
   flight is offering records that the drain has to be alive to accept. */
static void stop_flush_worker(void) {
	if (!flush_started) return;
	pthread_mutex_lock(&flush_mu);
	flush_stop = 1;
	pthread_cond_signal(&flush_cond);
	pthread_mutex_unlock(&flush_mu);
	if (pthread_join(flusher, NULL) != 0) atomic_store(&producer_failed, 1);
	flush_started = 0;
}

static const char *cupti_library_path(void) {
	Dl_info info;
	/* dladdr takes an object pointer, which ISO C does not let a function
	   pointer convert to directly. On every ELF platform this producer builds
	   for the two are the same width and the union carries the value without
	   the diagnostic, which keeps the controls buildable under gcc -pedantic
	   as well as clang. */
	union {
		CUptiResult (*fn)(CUpti_ActivityKind);
		void *obj;
	} symbol = {cuptiActivityEnable};
	if (dladdr(symbol.obj, &info) != 0 && info.dli_fname != NULL) return info.dli_fname;
	return "unknown";
}

static void record_cupti_failure(const char *operation, CUptiResult result) {
	const char *message = "unknown";
	(void)cuptiGetResultString(result, &message);
	char payload[MAX_PAYLOAD];
	int n = snprintf(payload, sizeof(payload), "{\"state\":\"cupti_unavailable\",\"operation\":\"%s\",\"result\":%d,\"message\":\"%s\"}", operation, (int)result, message ? message : "unknown");
	if (n > 0 && (size_t)n < sizeof(payload)) (void)write_record(KIND_FAILURE, now_ns(), payload, (size_t)n);
}

static uint32_t crc32_ieee(const unsigned char *p, size_t n) {
	uint32_t crc = 0xffffffffu;
	while (n--) {
		crc ^= *p++;
		for (int i = 0; i < 8; i++) crc = (crc >> 1) ^ (0xedb88320u & -(crc & 1));
	}
	return ~crc;
}

static void put16(unsigned char *p, uint16_t v) { p[0] = (unsigned char)(v >> 8); p[1] = (unsigned char)v; }
static void put32(unsigned char *p, uint32_t v) { p[0] = (unsigned char)(v >> 24); p[1] = (unsigned char)(v >> 16); p[2] = (unsigned char)(v >> 8); p[3] = (unsigned char)v; }
static void put64(unsigned char *p, uint64_t v) { for (int i = 7; i >= 0; i--) { p[i] = (unsigned char)v; v >>= 8; } }

static int write_all(int fd, const unsigned char *p, size_t n) {
	while (n) {
		ssize_t m = write(fd, p, n);
		if (m > 0) { p += m; n -= (size_t)m; continue; }
		if (m < 0 && errno == EINTR) continue;
		return -1;
	}
	return 0;
}

static int write_all_v(int fd, struct iovec *iov, size_t n) {
	while (n) {
		size_t count = n;
#ifdef IOV_MAX
		if (count > IOV_MAX) count = IOV_MAX;
#endif
		ssize_t m = writev(fd, iov, (int)count);
		if (m < 0) { if (errno == EINTR) continue; return -1; }
		if (m == 0) return -1;
		size_t advance = (size_t)m;
		while (n && advance >= iov->iov_len) { advance -= iov->iov_len; iov++; n--; }
		if (n && advance) { iov->iov_base = (char *)iov->iov_base + advance; iov->iov_len -= advance; }
	}
	return 0;
}

/* Serialize n consumer-owned ring slots into one gathered write. The journal
   mutex is held across serialization because the sequence number is part of
   the header and the header checksum covers it; it is one acquisition and one
   syscall for the whole batch. */
static int write_batch(size_t tail, size_t n) {
	if (journal_fd < 0 || n == 0) return -1;
	uint64_t activity = 0;
	uint64_t retained[KIND_COUNT] = {0};
	size_t iov = 0;
	uint64_t began = now_ns();
	pthread_mutex_lock(&journal_mu);
	/* Read the cumulative drop count under the journal lock, as the
	   single-record path does. Sampling it outside would let a concurrent
	   write_record publish a higher count first and leave this batch
	   reporting a decreasing one, which the decoder rejects. */
	uint64_t dropped = atomic_load(&dropped_count);
	for (size_t i = 0; i < n; i++) {
		const struct pending_record *r = &pending[(tail + i) % PENDING_RECORDS];
		if (r->length > MAX_PAYLOAD) { pthread_mutex_unlock(&journal_mu); return -1; }
		unsigned char *h = batch_headers[i];
		memcpy(h, "CDW1", 4);
		put16(h + 4, 1);
		put16(h + 6, r->kind);
		put32(h + 8, (uint32_t)r->length);
		memcpy(h + 12, session_id, sizeof(session_id));
		put64(h + 28, self_pid);
		put64(h + 36, generation);
		put64(h + 44, sequence++);
		put64(h + 52, dropped);
		put64(h + 60, r->timestamp);
		put32(h + 72, crc32_ieee((const unsigned char *)r->payload, r->length));
		memset(h + 76, 0, WIRE_HEADER - 76);
		put32(h + 68, crc32_ieee(h, 68));
		batch_iov[iov].iov_base = h;
		batch_iov[iov].iov_len = WIRE_HEADER;
		iov++;
		if (r->length) {
			batch_iov[iov].iov_base = (void *)(uintptr_t)r->payload;
			batch_iov[iov].iov_len = r->length;
			iov++;
		}
		if (r->kind == KIND_ACTIVITY) activity++;
		if (r->kind < KIND_COUNT) retained[r->kind]++;
	}
	int result = write_all_v(journal_fd, batch_iov, iov);
	pthread_mutex_unlock(&journal_mu);
	if (result != 0) { atomic_store(&producer_failed, 1); return -1; }
	uint64_t wrote = now_ns() - began;
	atomic_fetch_add(&drain_write_nanos, wrote);
	uint64_t slowest = atomic_load(&drain_write_max_nanos);
	while (wrote > slowest && !atomic_compare_exchange_weak(&drain_write_max_nanos, &slowest, wrote)) { }
	atomic_fetch_add(&drain_batches, 1);
	atomic_fetch_add(&drain_records, (uint64_t)n);
	uint64_t observed = atomic_load(&drain_max_batch);
	while (n > observed && !atomic_compare_exchange_weak(&drain_max_batch, &observed, (uint64_t)n)) { }
	if (activity) atomic_fetch_add(&activity_count, activity);
	/* Published only after the batch reached the journal, so retained means
	   written rather than merely drained. */
	for (uint16_t kind = 0; kind < KIND_COUNT; kind++) if (retained[kind]) count_kind(retained_by_kind, kind, retained[kind]);
	return 0;
}

static int write_record(uint16_t kind, uint64_t timestamp, const char *payload, size_t payload_len) {
	if (journal_fd < 0 || payload_len > MAX_PAYLOAD) return -1;
	unsigned char frame[WIRE_HEADER + MAX_PAYLOAD];
	/* Only WIRE_HEADER + payload_len bytes are written. Every header byte
	   below 76 is assigned before the write, so bytes 76-79 are the only
	   padding that needs zeroing; the rest of the buffer never reaches the
	   journal. Zeroing the whole frame costs 4176 bytes per record in the
	   drain path. */
	memset(frame + 76, 0, WIRE_HEADER - 76);
	memcpy(frame, "CDW1", 4);
	put16(frame + 4, 1);
	put16(frame + 6, kind);
	put32(frame + 8, (uint32_t)payload_len);
	memcpy(frame + 12, session_id, sizeof(session_id));
	put64(frame + 28, self_pid);
	put64(frame + 36, generation);
	put64(frame + 60, timestamp);
	memcpy(frame + WIRE_HEADER, payload, payload_len);
	put32(frame + 68, crc32_ieee(frame, 68));
	put32(frame + 72, crc32_ieee(frame + WIRE_HEADER, payload_len));
	pthread_mutex_lock(&journal_mu);
	put64(frame + 44, sequence++);
	put64(frame + 52, atomic_load(&dropped_count));
	/* Recompute the header checksum after assigning serialized fields. */
	put32(frame + 68, crc32_ieee(frame, 68));
	int result = write_all(journal_fd, frame, WIRE_HEADER + payload_len);
	pthread_mutex_unlock(&journal_mu);
	if (result != 0) atomic_store(&producer_failed, 1);
	return result;
}

static uint64_t now_ns(void) {
	struct timespec ts;
	if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0;
	return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/* The timestamp source handed to CUPTI. It must be now_ns itself and not
   merely another monotonic reading: the point of the callback is that a
   record's CUPTI timestamp and the record header this agent stamps around it
   come from one clock, and any second source reintroduces the offset the
   callback exists to remove. CUPTI calls this from its own threads and, for
   GPU-side activities, during post-processing while it interpolates device
   timestamps into the CPU base. It is safe in both: clock_gettime is
   reentrant and this function takes no lock, allocates nothing and writes no
   journal. */
static uint64_t CUPTIAPI cupti_timestamp(void) {
	return now_ns();
}

static int parse_session(const char *s) {
	if (s == NULL || strlen(s) != 32) return -1;
	for (size_t i = 0; i < sizeof(session_id); i++) {
		unsigned int x;
		if (sscanf(s + i * 2, "%2x", &x) != 1) return -1;
		session_id[i] = (unsigned char)x;
	}
	return 0;
}

static void write_handshake(const char *failure) {
	if (worker_dir[0] == '\0' || journal_fd < 0) return;
	char path[PATH_MAX];
	if (snprintf(path, sizeof(path), "%s/handshake.json", worker_dir) >= (int)sizeof(path)) return;
	FILE *f = fopen(path, "w");
	if (f == NULL) return;
	if (failure != NULL)
		fprintf(f, "{\"protocol\":1,\"session\":\"%s\",\"process\":%" PRIu64 ",\"generation\":%" PRIu64 ",\"injection_failure\":\"%s\"}\n", getenv("CUDAPROF_SESSION"), (uint64_t)getpid(), generation, failure);
	else
		fprintf(f, "{\"protocol\":1,\"session\":\"%s\",\"process\":%" PRIu64 ",\"generation\":%" PRIu64 ",\"transport_ready\":true}\n", getenv("CUDAPROF_SESSION"), (uint64_t)getpid(), generation);
	fclose(f);
}

/* Module capture. CUPTI's resource callback hands over the cubin the driver
   loaded, which is the only place device code is available to an
   out-of-process profiler: the workload compiles or embeds its kernels and the
   driver loads them, and nothing in the activity stream says what the code was.

   This is deliberately outside the capture-range gate. Modules are
   instantiated during warm-up, before a target opens its profiler range, so
   collecting them the way activity collection is collected -- off at
   bootstrap, on at cuProfilerStart -- would miss exactly the workloads that
   build their code early, and would miss them silently. A module is a fact
   about the process, not about the range.

   Images are written beside the journal rather than through it. The journal is
   a bounded ring of small fixed records sized for activity; a multi-megabyte
   cubin pushed through it would evict real records and read downstream as
   missing kernels. The record carries the file name, so the journal stays the
   index and the bytes stay out of it. */
#define MODULE_SEEN_MAX 256

static pthread_mutex_t module_mu = PTHREAD_MUTEX_INITIALIZER;
static int modules_enabled;
static unsigned module_seq;
static unsigned module_omitted;
static uint64_t module_total_bytes;
static uint64_t module_file_limit = 16ull * 1024 * 1024;
static uint64_t module_total_limit = 128ull * 1024 * 1024;
static struct { uint32_t id; uint64_t size; } module_seen[MODULE_SEEN_MAX];
static unsigned module_seen_count;
/* module_unchecked counts the loads that arrived once module_seen was full.
   Each was written without a duplicate check, so a capture past the bound
   may hold one image several times; the count is what says so. */
static unsigned module_unchecked;

/* env_bytes reads a byte budget from the environment, keeping the default when
   the variable is absent or does not parse. A budget that silently became zero
   would omit every module and leave a capture indistinguishable from a target
   that loaded none. */
static uint64_t env_bytes(const char *name, uint64_t fallback) {
	const char *v = getenv(name);
	if (v == NULL || *v == '\0') return fallback;
	char *end = NULL;
	unsigned long long n = strtoull(v, &end, 10);
	if (end == v || *end != '\0' || n == 0) return fallback;
	return (uint64_t)n;
}

/* module_first_load reports whether an (id, size) pair has not been written
   yet, recording it when it has not. A module loaded once per context arrives
   once per load with identical bytes, and writing every copy would multiply
   the capture by the context count for nothing. Past the table's bound every
   load counts as new: writing a duplicate costs size, where refusing a
   distinct module would cost correctness. *unchecked is set for such a load,
   so its record can say it may repeat an earlier one. */
static int module_first_load(uint32_t id, uint64_t size, int *unchecked) {
	*unchecked = 0;
	for (unsigned i = 0; i < module_seen_count; i++)
		if (module_seen[i].id == id && module_seen[i].size == size) return 0;
	if (module_seen_count < MODULE_SEEN_MAX) {
		module_seen[module_seen_count].id = id;
		module_seen[module_seen_count].size = size;
		module_seen_count++;
	} else {
		*unchecked = 1;
		module_unchecked++;
	}
	return 1;
}

/* write_module writes one cubin beside the journal and records what happened to
   it. A module the budget refuses is recorded with its size rather than
   dropped: "the target loaded no such module" and "the agent declined to keep
   one this large" are different facts, and silence reads as the first. */
static void write_module(uint32_t module_id, const char *cubin, size_t size) {
	if (worker_dir[0] == '\0' || journal_fd < 0 || cubin == NULL || size == 0) return;
	const char *reason = "";
	int omit = 0, unchecked;
	unsigned seq;
	pthread_mutex_lock(&module_mu);
	if (!module_first_load(module_id, (uint64_t)size, &unchecked)) { pthread_mutex_unlock(&module_mu); return; }
	if ((uint64_t)size > module_file_limit) { omit = 1; reason = "over per-module budget"; }
	else if (module_total_bytes + (uint64_t)size > module_total_limit) { omit = 1; reason = "over total module budget"; }
	seq = module_seq++;
	if (omit) module_omitted++;
	else module_total_bytes += (uint64_t)size;
	pthread_mutex_unlock(&module_mu);

	char name[64];
	if (snprintf(name, sizeof(name), "module-%04u-id%" PRIu32 ".cubin", seq, module_id) >= (int)sizeof(name)) return;
	if (!omit) {
		char dir[PATH_MAX], path[PATH_MAX];
		if (snprintf(dir, sizeof(dir), "%s/modules", worker_dir) >= (int)sizeof(dir)) return;
		if (mkdir(dir, 0700) != 0 && errno != EEXIST) { omit = 1; reason = "module directory unavailable"; }
		else if (snprintf(path, sizeof(path), "%s/%s", dir, name) >= (int)sizeof(path)) return;
		else {
			int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
			if (fd < 0) { omit = 1; reason = "module file unavailable"; }
			else {
				size_t off = 0;
				while (off < size) {
					ssize_t w = write(fd, cubin + off, size - off);
					if (w <= 0) { omit = 1; reason = "module write failed"; break; }
					off += (size_t)w;
				}
				if (close(fd) != 0 && !omit) { omit = 1; reason = "module close failed"; }
			}
		}
		/* The budget counted these bytes before the write was attempted. An
		   I/O failure has to give them back, or a run that cannot write at all
		   reports a total it never stored. */
		if (omit) {
			pthread_mutex_lock(&module_mu);
			module_total_bytes -= (uint64_t)size;
			module_omitted++;
			pthread_mutex_unlock(&module_mu);
		}
	}
	/* The field names are module-prefixed because the consumer decodes every
	   record kind into one flat structure: an unprefixed "retained" or "size"
	   would land on a field another kind already owns, and a type that
	   disagrees fails the whole journal rather than this record. */
	char payload[MAX_PAYLOAD];
	int n = snprintf(payload, sizeof(payload),
		"{\"kind\":\"module\",\"module_id\":%" PRIu32 ",\"module_size\":%zu,\"module_file\":\"%s\",\"module_stored\":%s,\"module_reason\":\"%s\",\"module_unchecked\":%s}",
		module_id, size, omit ? "" : name, omit ? "false" : "true", reason, unchecked ? "true" : "false");
	if (n > 0 && (size_t)n < sizeof(payload)) (void)write_record(KIND_MODULE, now_ns(), payload, (size_t)n);
}

/* Which module a kernel came from. A CUPTI kernel activity record names the
   kernel and nothing else: two modules holding a kernel of the same name --
   a default and a fast-math build of one graph, say -- produce records a
   consumer cannot tell apart, so `cudaprof lines` has no way to charge a
   sample to the build that ran.

   The driver does say, in pieces, and the pieces arrive on the thread that
   loads the code. CUPTI's module-loaded resource callback fires inside the
   load call with the module id; the load's own exit callback hands back the
   CUmodule that id belongs to; cuModuleGetFunction pairs a CUfunction with
   that module; and a launch callback names the CUfunction alongside the
   correlation id the kernel record will carry. Following that chain leaves a
   correlation-to-module answer waiting by the time the activity record is
   decoded.

   A target that launches through the CUDA runtime's own loader never calls
   cuModuleGetFunction, so its kernels keep a zero module id, which the
   consumer reads as unknown rather than as module zero. The tables are
   split only for a capture that holds more than one module, so the cost of
   not knowing is the same note the consumer printed before any of this. */
#define MODULE_HANDLE_SLOTS 256
#define FUNCTION_HANDLE_SLOTS 1024
#define LAUNCH_SLOTS 8192

static pthread_mutex_t handle_mu = PTHREAD_MUTEX_INITIALIZER;
/* Images are CUmodule and CUlibrary handles; code is CUfunction and CUkernel
   handles and the host stubs the runtime launches by. The driver accepts a
   CUkernel wherever it takes a CUfunction, so a launch looks both up in one
   table. Handles are opaque and only ever compared, never dereferenced. */
static struct { const void *handle; uint32_t id; } module_handles[MODULE_HANDLE_SLOTS];
static int module_handle_count;
static struct { const void *handle; uint32_t id; } function_handles[FUNCTION_HANDLE_SLOTS];
static int function_handle_count;
/* Correlation ids rise monotonically, so the slot is the id itself modulo the
   table size. A launch whose slot is taken before its record is decoded is
   one the reader loses the module of, which is the same outcome as never
   having recorded it. */
static struct { _Atomic uint32_t correlation; _Atomic uint32_t id; } launch_modules[LAUNCH_SLOTS];
/* The module id of a load running on this thread, published by the resource
   callback and taken by the exit callback of the call it fired inside. The
   driver loads lazily, so that call can be a library load, a function lookup
   or a launch; every tracked call clears it on entry, so an id published
   inside an untracked call is never charged to a later one. */
static _Thread_local uint32_t loading_module_id;

static uint32_t take_loaded_module(void) {
	uint32_t id = loading_module_id;
	loading_module_id = 0;
	return id;
}

static void bind_module_handle(const void *handle, uint32_t id) {
	if (handle == NULL || id == 0) return;
	pthread_mutex_lock(&handle_mu);
	for (int i = 0; i < module_handle_count; i++) {
		if (module_handles[i].handle == handle) { module_handles[i].id = id; pthread_mutex_unlock(&handle_mu); return; }
	}
	if (module_handle_count < MODULE_HANDLE_SLOTS) {
		module_handles[module_handle_count].handle = handle;
		module_handles[module_handle_count].id = id;
		module_handle_count++;
	}
	pthread_mutex_unlock(&handle_mu);
}

/* bind_function_handle records that code came from the image with handle, or
   from module id when the image is not known; id wins when both are given,
   because it is what the driver just loaded for this very call. */
static void bind_function_handle(const void *fn, const void *handle, uint32_t id) {
	if (fn == NULL) return;
	pthread_mutex_lock(&handle_mu);
	for (int i = 0; id == 0 && handle != NULL && i < module_handle_count; i++) {
		if (module_handles[i].handle == handle) id = module_handles[i].id;
	}
	if (id != 0) {
		int found = 0;
		for (int i = 0; i < function_handle_count; i++) {
			if (function_handles[i].handle == fn) { function_handles[i].id = id; found = 1; break; }
		}
		if (!found && function_handle_count < FUNCTION_HANDLE_SLOTS) {
			function_handles[function_handle_count].handle = fn;
			function_handles[function_handle_count].id = id;
			function_handle_count++;
		}
	}
	pthread_mutex_unlock(&handle_mu);
}

/* function_module returns the module id bound to fn, or zero. */
static uint32_t function_module(const void *fn) {
	uint32_t id = 0;
	pthread_mutex_lock(&handle_mu);
	for (int i = 0; i < function_handle_count; i++) {
		if (function_handles[i].handle == fn) { id = function_handles[i].id; break; }
	}
	pthread_mutex_unlock(&handle_mu);
	return id;
}

static void note_launch_module(const void *fn, uint32_t correlation) {
	if (fn == NULL || correlation == 0) return;
	uint32_t id = function_module(fn);
	if (id == 0) return;
	size_t slot = correlation % LAUNCH_SLOTS;
	atomic_store(&launch_modules[slot].id, id);
	atomic_store(&launch_modules[slot].correlation, correlation);
}

/* launch_module answers for one kernel record, or zero when the chain above
   was never completed for it. */
static uint32_t launch_module(uint32_t correlation) {
	if (correlation == 0) return 0;
	size_t slot = correlation % LAUNCH_SLOTS;
	if (atomic_load(&launch_modules[slot].correlation) != correlation) return 0;
	return atomic_load(&launch_modules[slot].id);
}

/* Graph replay correlations name cuGraphLaunch, not a CUfunction. Retain
   the CUPTI node identity when an explicit kernel node is added, and copy it
   when instantiation clones the node. Never infer an image from a symbol.
   Updates invalidate this bounded table: delayed activity may describe either
   version, so resolving it with the latest function would be incorrect. */
#define GRAPH_MODULE_SLOTS 4096
static struct { uint64_t node; uint32_t module; } graph_modules[GRAPH_MODULE_SLOTS];
static unsigned graph_module_count;

static uint32_t graph_module(uint64_t node) {
	uint32_t module = 0;
	pthread_mutex_lock(&handle_mu);
	for (unsigned i = 0; i < graph_module_count; i++) {
		if (graph_modules[i].node == node) { module = graph_modules[i].module; break; }
	}
	pthread_mutex_unlock(&handle_mu);
	return module;
}

static void bind_graph_module(CUgraphNode node, uint32_t module) {
	uint64_t id = 0;
	if (module == 0 || node == NULL || cuptiGetGraphNodeId(node, &id) != CUPTI_SUCCESS || id == 0) return;
	pthread_mutex_lock(&handle_mu);
	for (unsigned i = 0; i < graph_module_count; i++) {
		if (graph_modules[i].node == id) {
			if (graph_modules[i].module != module) graph_modules[i].module = 0;
			pthread_mutex_unlock(&handle_mu);
			return;
		}
	}
	if (graph_module_count < GRAPH_MODULE_SLOTS) {
		graph_modules[graph_module_count].node = id;
		graph_modules[graph_module_count++].module = module;
	}
	pthread_mutex_unlock(&handle_mu);
}

static void note_graph_function(CUgraphNode node, CUfunction fn) {
	uint32_t module = 0;
	pthread_mutex_lock(&handle_mu);
	for (int i = 0; i < function_handle_count; i++) {
		if (function_handles[i].handle == fn) { module = function_handles[i].id; break; }
	}
	pthread_mutex_unlock(&handle_mu);
	bind_graph_module(node, module);
}

static int invalidate_runtime_graph(CUpti_CallbackDomain domain, CUpti_CallbackId cbid) {
	if (domain != CUPTI_CB_DOMAIN_RUNTIME_API) return 0;
	switch (cbid) {
	case CUPTI_RUNTIME_TRACE_CBID_cudaGraphKernelNodeSetParams_v10000:
	case CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecKernelNodeSetParams_v10010:
	case CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecUpdate_v10020:
	case CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecChildGraphNodeSetParams_v11010:
	case CUPTI_RUNTIME_TRACE_CBID_cudaGraphNodeSetParams_v12020:
	case CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecNodeSetParams_v12020:
		pthread_mutex_lock(&handle_mu);
		graph_module_count = 0;
		pthread_mutex_unlock(&handle_mu);
		return 1;
	default:
		return 0;
	}
}

/* A record that did not fit its buffer is a record nothing downstream will
   ever hear about: queue_pending never sees it, so it is not counted as
   dropped either, and the journal reads as if the work never happened. Say
   so, on the same rule the name guards apply -- exact or a failure, never a
   quiet loss. what names the record that overflowed. */
static int payload_overflow(const char *what) {
	char failure[192];
	int n = snprintf(failure, sizeof(failure), "{\"state\":\"producer_failure\",\"error\":\"%s record does not fit its buffer\"}", what);
	if (n > 0 && (size_t)n < sizeof(failure)) (void)queue_pending(KIND_FAILURE, now_ns(), failure, (size_t)n);
	return -1;
}

static int append_activity(const char *kind, const char *name, uint64_t start, uint64_t end, uint64_t bytes,
	uint64_t queued, uint64_t submitted, uint8_t copy_kind, uint32_t memset_value,
	uint32_t runtime_correlation, uint32_t device, uint32_t context, uint32_t stream, uint32_t correlation,
	uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
	uint32_t block_x, uint32_t block_y, uint32_t block_z,
	uint32_t registers_per_thread, uint32_t dynamic_shared_memory,
	uint32_t graph_id, uint64_t graph_node_id, uint32_t module_id) {
	char payload[MAX_PAYLOAD];
	int n = snprintf(payload, sizeof(payload), "{\"kind\":\"activity\",\"subtype\":\"%s\",\"name\":\"", kind);
	if (n < 0 || n >= (int)sizeof(payload)) return -1;
	for (const unsigned char *p = (const unsigned char *)(name ? name : ""); *p; p++) {
		if (n + 8 >= (int)sizeof(payload)) {
			const char *failure = "{\"state\":\"producer_failure\",\"error\":\"activity name too long\"}";
			(void)queue_pending(KIND_FAILURE, now_ns(), failure, strlen(failure));
			return -1;
		}
		if (*p == '"' || *p == '\\') payload[n++] = '\\';
		if (*p < 0x20) { const char *failure = "{\"state\":\"producer_failure\",\"error\":\"activity name contains control byte\"}"; (void)queue_pending(KIND_FAILURE, now_ns(), failure, strlen(failure)); return -1; }
		payload[n++] = (char)*p;
	}
	int valid = start != CUPTI_TIMESTAMP_UNKNOWN && end != CUPTI_TIMESTAMP_UNKNOWN && end >= start;
	n += snprintf(payload + n, sizeof(payload) - (size_t)n,
		"\",\"valid\":%s,\"start\":%" PRIu64 ",\"end\":%" PRIu64 ",\"bytes\":%" PRIu64 ",\"queued\":%" PRIu64 ",\"submitted\":%" PRIu64 ",\"copy_kind\":%u,\"value\":%u,\"runtime_correlation\":%u,\"device\":%u,\"context\":%u,\"stream\":%u,\"correlation\":%u,\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],\"grid_x\":%u,\"grid_y\":%u,\"grid_z\":%u,\"block_x\":%u,\"block_y\":%u,\"block_z\":%u,\"registers_per_thread\":%u,\"dynamic_shared_memory\":%u,\"graph_id\":%u,\"graph_node_id\":%" PRIu64 ",\"module_id\":%u}",
		valid ? "true" : "false", start, end, bytes, queued, submitted, copy_kind, memset_value, runtime_correlation, device, context, stream, correlation,
		grid_x, grid_y, grid_z, block_x, block_y, block_z,
		grid_x, grid_y, grid_z, block_x, block_y, block_z,
		registers_per_thread, dynamic_shared_memory, graph_id, graph_node_id, module_id);
	/* The name guard above reserves one escaped byte, not the several hundred
	   the fields below take, so a name within a tail's length of the buffer
	   passes it and overflows here. */
	if (n < 0 || n >= (int)sizeof(payload)) return payload_overflow("activity");
	return queue_pending(KIND_ACTIVITY, start, payload, (size_t)n);
}

/* CUPTI's own cost, as a span. This is the one tier here that is not the
   target's work at all: a buffer flush, an instrumentation pause, a lazily
   loaded module. Until it was collected, every one of those intervals was
   indistinguishable from the device being idle, so the tool's headline idle
   number carried the tool.

   An overhead record names an object, not a device, and the object is usually
   a host thread. The device/context/stream fields are filled only when CUPTI
   actually scoped the overhead that way; otherwise they stay zero and
   object_kind is what says so. Writing a zero device as though it meant
   "device 0" is the failure this tier exists to avoid, not one to repeat. */
static int append_overhead(const CUpti_ActivityOverhead *o) {
	uint32_t process_id = 0, thread_id = 0, device = 0, context = 0, stream = 0;
	switch (o->objectKind) {
	case CUPTI_ACTIVITY_OBJECT_PROCESS:
		process_id = o->objectId.pt.processId;
		break;
	case CUPTI_ACTIVITY_OBJECT_THREAD:
		process_id = o->objectId.pt.processId;
		thread_id = o->objectId.pt.threadId;
		break;
	case CUPTI_ACTIVITY_OBJECT_STREAM:
		stream = o->objectId.dcs.streamId;
		/* fallthrough */
	case CUPTI_ACTIVITY_OBJECT_CONTEXT:
		context = o->objectId.dcs.contextId;
		/* fallthrough */
	case CUPTI_ACTIVITY_OBJECT_DEVICE:
		device = o->objectId.dcs.deviceId;
		break;
	default:
		break;
	}
	int valid = o->start != CUPTI_TIMESTAMP_UNKNOWN && o->end != CUPTI_TIMESTAMP_UNKNOWN && o->end >= o->start;
	char payload[MAX_PAYLOAD];
	int n = snprintf(payload, sizeof(payload),
		"{\"kind\":\"activity\",\"subtype\":\"overhead\",\"name\":\"\",\"valid\":%s,\"start\":%" PRIu64 ",\"end\":%" PRIu64 ",\"overhead_kind\":%u,\"object_kind\":%u,\"device\":%u,\"context\":%u,\"stream\":%u,\"process_id\":%u,\"thread_id\":%u}",
		valid ? "true" : "false", o->start, o->end, (unsigned)o->overheadKind, (unsigned)o->objectKind,
		device, context, stream, process_id, thread_id);
	if (n < 0 || (size_t)n >= sizeof(payload)) return -1;
	return queue_pending(KIND_ACTIVITY, o->start, payload, (size_t)n);
}

/* graph_trace_requested reports whether CUDAPROF_GRAPH_TRACE asks for graph
   trace mode. Only "graph" does; unset, "node" or anything else is node mode. */
static int graph_trace_requested(void) {
	const char *v = getenv("CUDAPROF_GRAPH_TRACE");
	return v != NULL && strcmp(v, "graph") == 0;
}

/* The graph replay envelope: one record per graph launch, spanning the whole
   replay. It is recorded only in graph trace mode (CUDAPROF_GRAPH_TRACE=graph).

   It is not a complement to per-node records. cupti_activity.h describes
   CUpti_ActivityGraphTrace as execution "for a graph without giving
   visibility about the execution of its nodes. This is intended to reduce
   overheads in tracing each node." (CUDA 13.4 cupti_activity.h:9897-9903;
   the same text in 12.8, 13.0 and 13.2). With the kind enabled, the kernels
   inside a replay were not delivered as kernel records: a ds4 prefill capture
   had 86 envelopes and 0 of 8,686 kernel records carrying a graph node id.
   Node mode, the default, leaves the kind off so each node kernel arrives as
   a kernel record carrying graphId and graphNodeId.

   Unlike an overhead or a synchronization, a graph trace does carry a device
   id, so nothing downstream has to resolve one for it. */
static int append_graph_trace(const CUpti_ActivityGraphTrace *g) {
	int valid = g->start != CUPTI_TIMESTAMP_UNKNOWN && g->end != CUPTI_TIMESTAMP_UNKNOWN && g->end >= g->start;
	char payload[MAX_PAYLOAD];
	int n = snprintf(payload, sizeof(payload),
		"{\"kind\":\"activity\",\"subtype\":\"graph_trace\",\"name\":\"\",\"valid\":%s,\"start\":%" PRIu64 ",\"end\":%" PRIu64 ",\"device\":%u,\"context\":%u,\"stream\":%u,\"correlation\":%u,\"graph_id\":%u}",
		valid ? "true" : "false", g->start, g->end, g->deviceId, g->contextId, g->streamId, g->correlationId, g->graphId);
	if (n < 0 || (size_t)n >= sizeof(payload)) return -1;
	return queue_pending(KIND_ACTIVITY, g->start, payload, (size_t)n);
}

/* A synchronization span: the window a cudaStreamSynchronize, an event wait or
   a context synchronize was open. It is host time, not device work, and its
   value is that it explains device idle rather than adding to device busy.

   The record carries a context and a stream and no device at all. The device
   is left at zero here deliberately -- the producer has no map from contexts
   to devices, and a guess written into the journal is indistinguishable from a
   measurement once it is there. Resolving it is the consumer's job, and so is
   saying when it could not be resolved. */
static int append_sync(const CUpti_ActivitySynchronization *y) {
	int valid = y->start != CUPTI_TIMESTAMP_UNKNOWN && y->end != CUPTI_TIMESTAMP_UNKNOWN && y->end >= y->start;
	char payload[MAX_PAYLOAD];
	int n = snprintf(payload, sizeof(payload),
		"{\"kind\":\"activity\",\"subtype\":\"sync\",\"name\":\"\",\"valid\":%s,\"start\":%" PRIu64 ",\"end\":%" PRIu64 ",\"sync_type\":%u,\"cuda_event_id\":%u,\"context\":%u,\"stream\":%u,\"correlation\":%u}",
		valid ? "true" : "false", y->start, y->end, (unsigned)y->type, y->cudaEventId, y->contextId, y->streamId, y->correlationId);
	if (n < 0 || (size_t)n >= sizeof(payload)) return -1;
	return queue_pending(KIND_ACTIVITY, y->start, payload, (size_t)n);
}

static int append_external(uint32_t external_kind, uint64_t external_id, uint32_t correlation) {
	char payload[MAX_PAYLOAD];
	int n = snprintf(payload, sizeof(payload), "{\"kind\":\"external_correlation\",\"external_kind\":%u,\"external_id\":%" PRIu64 ",\"correlation\":%u}", external_kind, external_id, correlation);
	if (n < 0 || (size_t)n >= sizeof(payload)) return -1;
	return queue_pending(KIND_EXTERNAL, 0, payload, (size_t)n);
}

/* Appends text at payload[n] as the body of a JSON string, quotes excluded,
   and returns the new length. A name that cannot be transcribed exactly is a
   producer failure rather than a silently truncated one, on the same rule
   append_activity applies to a kernel name: a truncated symbol resolves to a
   different symbol, and nothing downstream can tell that happened. what names
   the field in the failure record.
*/
static int append_escaped(char *payload, size_t cap, int n, const char *text, const char *what) {
	for (const unsigned char *p = (const unsigned char *)(text ? text : ""); *p; p++) {
		if (n < 0 || (size_t)(n + 8) >= cap) {
			char failure[192];
			int fn = snprintf(failure, sizeof(failure), "{\"state\":\"producer_failure\",\"error\":\"%s too long\"}", what);
			if (fn > 0 && (size_t)fn < sizeof(failure)) (void)queue_pending(KIND_FAILURE, now_ns(), failure, (size_t)fn);
			return -1;
		}
		if (*p < 0x20) {
			char failure[192];
			int fn = snprintf(failure, sizeof(failure), "{\"state\":\"producer_failure\",\"error\":\"%s contains control byte\"}", what);
			if (fn > 0 && (size_t)fn < sizeof(failure)) (void)queue_pending(KIND_FAILURE, now_ns(), failure, (size_t)fn);
			return -1;
		}
		if (*p == '"' || *p == '\\') payload[n++] = '\\';
		payload[n++] = (char)*p;
	}
	return n;
}

/* Transcribes one NVTX marker. CUPTI reports a range as two records -- a
   START and an END sharing an id -- so this emits exactly what it received
   and leaves the pairing to a consumer that can afford a map. start and end
   both carry the single timestamp the record reported: it is a point in
   time, and "valid" says that timestamp is usable, not that a duration was
   measured. Reading a duration off one marker reads zero, which is the
   truth about one marker and a lie about the range it belongs to.

   The object identity is recorded under the keys the union arm actually
   names. A marker attributed to a thread has no stream, and filling one in
   from the wrong arm of the union would put a host range on a device track. */
static int append_marker(const CUpti_ActivityMarker2 *m) {
	char payload[MAX_PAYLOAD];
	int n = snprintf(payload, sizeof(payload), "{\"kind\":\"activity\",\"subtype\":\"marker\",\"name\":\"");
	if (n < 0 || n >= (int)sizeof(payload)) return -1;
	if ((n = append_escaped(payload, sizeof(payload), n, m->name, "marker name")) < 0) return -1;
	int w = snprintf(payload + n, sizeof(payload) - (size_t)n, "\",\"marker_domain\":\"");
	if (w < 0 || (size_t)(n + w) >= sizeof(payload)) return payload_overflow("marker");
	n += w;
	if ((n = append_escaped(payload, sizeof(payload), n, m->domain, "marker domain")) < 0) return -1;
	uint32_t process_id = 0, thread_id = 0, device = 0, context = 0, stream = 0;
	switch (m->objectKind) {
	case CUPTI_ACTIVITY_OBJECT_PROCESS:
		process_id = m->objectId.pt.processId;
		break;
	case CUPTI_ACTIVITY_OBJECT_THREAD:
		process_id = m->objectId.pt.processId;
		thread_id = m->objectId.pt.threadId;
		break;
	case CUPTI_ACTIVITY_OBJECT_STREAM:
		stream = m->objectId.dcs.streamId;
		/* fallthrough */
	case CUPTI_ACTIVITY_OBJECT_CONTEXT:
		context = m->objectId.dcs.contextId;
		/* fallthrough */
	case CUPTI_ACTIVITY_OBJECT_DEVICE:
		device = m->objectId.dcs.deviceId;
		break;
	default:
		break;
	}
	w = snprintf(payload + n, sizeof(payload) - (size_t)n,
		"\",\"valid\":%s,\"start\":%" PRIu64 ",\"end\":%" PRIu64 ",\"marker_id\":%u,\"marker_flags\":%u,\"object_kind\":%u,\"process_id\":%u,\"thread_id\":%u,\"device\":%u,\"context\":%u,\"stream\":%u}",
		m->timestamp != CUPTI_TIMESTAMP_UNKNOWN ? "true" : "false", m->timestamp, m->timestamp,
		(unsigned)m->id, (unsigned)m->flags, (unsigned)m->objectKind,
		process_id, thread_id, device, context, stream);
	if (w < 0 || (size_t)(n + w) >= sizeof(payload)) return payload_overflow("marker");
	n += w;
	return queue_pending(KIND_ACTIVITY, m->timestamp, payload, (size_t)n);
}

static int append_api(const char *domain, const CUpti_ActivityAPI *api) {
	char payload[MAX_PAYLOAD];
	int n = snprintf(payload, sizeof(payload),
		"{\"kind\":\"api\",\"domain\":\"%s\",\"callback_id\":%u,\"start\":%" PRIu64 ",\"end\":%" PRIu64 ",\"process_id\":%u,\"thread_id\":%u,\"correlation\":%u,\"return_value\":%u}",
		domain, (unsigned)api->cbid, api->start, api->end, api->processId, api->threadId, api->correlationId, api->returnValue);
	if (n < 0 || (size_t)n >= sizeof(payload)) return -1;
	return queue_pending(KIND_API, api->start, payload, (size_t)n);
}

/* ------------------------------------------------------------------------
   cuBLASLt calls.

   A kernel record names the kernel cuBLASLt chose and nothing about the call
   that chose it: not the shape, the types, beta or the algorithm. No CUPTI
   domain covers cuBLAS, and interposing its entry points misses a target that
   resolves them with dlsym on its own handle -- a purego binding does exactly
   that, and a lookup through a handle never reaches a preloaded library.
   cuBLASLt's logger reaches every caller, because it is a setting of the
   library itself: at mask 2 ("performance trace") it hands the callback one
   line per matmul, on the calling thread and before the launch, naming the
   layouts, the compute descriptor, the algorithm and beta. cuBLAS's own GEMMs
   arrive there too, as the internal Lt entry they are routed through.

   The line is forwarded as written and parsed by the reader. cublasLt.h calls
   the logger experimental, and a reader can be fixed after the fact where an
   agent that dropped a field cannot.

   The call is joined to its kernels here, because only here is the call still
   on the stack. The line arrives before the launch and nothing marks the
   call's return, so the callback leaves the call pending on its thread and
   each driver launch on that thread asks whether it is still inside
   libcublasLt: a return address inside the library's code means the launch is
   the call's (a split-K GEMM launches twice), and none means the call has
   returned. The walk follows frame records, which AArch64 code keeps. The
   x86-64 driver and libraries omit them, so a walk that misses the library
   is retried with the unwinder, which reads the libraries' .eh_frame tables
   instead. A stack neither can walk leaves the call with no launches, which
   the reader reports rather than guesses at.

   The library is loaded when the target decides, so it is looked for at
   start and then on each callback until it appears. dl_iterate_phdr's
   load counter keeps the look to one object when nothing new was loaded. */
static int gemms_requested;
static _Atomic int gemm_hooked;
static pthread_mutex_t gemm_mu = PTHREAD_MUTEX_INITIALIZER;
static unsigned long long gemm_adds;
static uintptr_t gemm_text_lo, gemm_text_hi;
static char gemm_library[PATH_MAX];
static _Atomic uint64_t gemm_next_id;
static _Atomic uint64_t gemm_calls;
static _Atomic uint64_t gemm_launches;
static _Atomic uint64_t gemm_oversize;
/* The call this thread is inside, or zero. */
static _Thread_local uint64_t gemm_pending;
/* This thread's stack, which bounds the frame walk: 1 known, -1 unavailable. */
static _Thread_local int gemm_stack_state;
static _Thread_local uintptr_t gemm_stack_lo, gemm_stack_hi;

typedef void (*gemm_logger_fn)(int, const char *, const char *);

struct gemm_scan {
	unsigned long long adds;
	int first;
	uintptr_t lo, hi;
	char path[PATH_MAX];
};

#ifdef __linux__
static int gemm_scan_object(struct dl_phdr_info *info, size_t size, void *arg) {
	(void)size;
	struct gemm_scan *s = arg;
	if (s->first) {
		s->first = 0;
		if (info->dlpi_adds == s->adds) return 1;
		s->adds = info->dlpi_adds;
	}
	const char *name = info->dlpi_name ? info->dlpi_name : "";
	const char *base = strrchr(name, '/');
	base = base ? base + 1 : name;
	if (strncmp(base, "libcublasLt.so", 14) != 0) return 0;
	uintptr_t lo = UINTPTR_MAX, hi = 0;
	for (int i = 0; i < info->dlpi_phnum; i++) {
		const ElfW(Phdr) *ph = &info->dlpi_phdr[i];
		if (ph->p_type != PT_LOAD || !(ph->p_flags & PF_X)) continue;
		uintptr_t a = (uintptr_t)(info->dlpi_addr + ph->p_vaddr);
		if (a < lo) lo = a;
		if (a + ph->p_memsz > hi) hi = a + ph->p_memsz;
	}
	if (hi <= lo || strlen(name) >= sizeof(s->path)) return 0;
	s->lo = lo;
	s->hi = hi;
	strcpy(s->path, name);
	return 1;
}
#endif

static void gemm_log(int level, const char *function, const char *message);

/* gemm_probe installs the logger once libcublasLt is mapped. */
static void gemm_probe(void) {
	if (!gemms_requested || atomic_load(&gemm_hooked)) return;
	/* A thread that finds another probing leaves: the other will finish. */
	if (pthread_mutex_trylock(&gemm_mu) != 0) return;
	if (atomic_load(&gemm_hooked)) { pthread_mutex_unlock(&gemm_mu); return; }
	struct gemm_scan s;
	memset(&s, 0, sizeof(s));
	s.adds = gemm_adds;
	s.first = 1;
#ifdef __linux__
	(void)dl_iterate_phdr(gemm_scan_object, &s);
#endif
	gemm_adds = s.adds;
	if (s.path[0] != '\0') {
		/* The handle is kept: it holds the library, and with it the logger
		   setting, for the rest of the process. */
		void *h = dlopen(s.path, RTLD_NOW | RTLD_NOLOAD);
		int (*set_callback)(gemm_logger_fn) = NULL;
		int (*set_file)(FILE *) = NULL;
		int (*set_mask)(int) = NULL;
		if (h != NULL) {
			*(void **)(&set_callback) = dlsym(h, "cublasLtLoggerSetCallback");
			*(void **)(&set_file) = dlsym(h, "cublasLtLoggerSetFile");
			*(void **)(&set_mask) = dlsym(h, "cublasLtLoggerSetMask");
		}
		if (set_callback != NULL && set_mask != NULL) {
			gemm_text_lo = s.lo;
			gemm_text_hi = s.hi;
			strcpy(gemm_library, s.path);
			/* The callback replaces the log file only once the file is
			   cleared; left alone, every line also goes to the target's
			   stdout. */
			if (set_file != NULL && set_file(NULL) != 0) {
				FILE *null = fopen("/dev/null", "w");
				if (null != NULL) set_file(null);
			}
			if (set_callback(gemm_log) == 0 && set_mask(2) == 0) {
				atomic_store(&gemm_hooked, 1);
			} else {
				record_cupti_failure("cublasLtLogger", CUPTI_ERROR_UNKNOWN);
				gemms_requested = 0;
			}
		}
	}
	pthread_mutex_unlock(&gemm_mu);
}

/* gemm_append copies a log line into a JSON string body. The logger's text
   is not a symbol, so unlike append_escaped a control byte becomes a space
   rather than a failure; a line that does not fit is counted and skipped. */
static int gemm_append(char *payload, size_t cap, int n, const char *text) {
	for (const unsigned char *p = (const unsigned char *)text; *p; p++) {
		if (n < 0 || (size_t)n + 8 >= cap) return -1;
		unsigned char c = *p < 0x20 ? ' ' : *p;
		if (c == '"' || c == '\\') payload[n++] = '\\';
		payload[n++] = (char)c;
	}
	return n;
}

static void gemm_log(int level, const char *function, const char *message) {
	(void)level;
	if (function == NULL || message == NULL) return;
	size_t fl = strlen(function);
	if (fl < 6 || strcmp(function + fl - 6, "Matmul") != 0) return;
	atomic_fetch_add(&callback_active, 1);
	gemm_pending = 0;
	if (atomic_load(&shutdown_started) || !atomic_load(&collecting)) {
		atomic_fetch_sub(&callback_active, 1);
		return;
	}
	uint64_t id = atomic_fetch_add(&gemm_next_id, 1) + 1;
	uint64_t ts = now_ns();
#ifdef __linux__
	long thread = (long)syscall(SYS_gettid);
#else
	long thread = (long)(uintptr_t)pthread_self();
#endif
	char payload[MAX_PAYLOAD];
	int n = snprintf(payload, sizeof(payload), "{\"kind\":\"gemm\",\"id\":%" PRIu64 ",\"thread\":%ld,\"api\":\"", id, thread);
	n = gemm_append(payload, sizeof(payload), n, function);
	if (n >= 0) {
		int w = snprintf(payload + n, sizeof(payload) - (size_t)n, "\",\"log\":\"");
		n = w < 0 || (size_t)(n + w) >= sizeof(payload) ? -1 : n + w;
	}
	n = n < 0 ? -1 : gemm_append(payload, sizeof(payload), n, message);
	if (n < 0 || (size_t)n + 2 >= sizeof(payload)) {
		atomic_fetch_add(&gemm_oversize, 1);
	} else {
		payload[n++] = '"';
		payload[n++] = '}';
		if (queue_pending(KIND_GEMM, ts, payload, (size_t)n) == 0) {
			atomic_fetch_add(&gemm_calls, 1);
			gemm_pending = id;
		}
	}
	atomic_fetch_sub(&callback_active, 1);
}

#ifdef __linux__
struct gemm_unwind {
	int found;
	int depth;
};

static _Unwind_Reason_Code gemm_unwind_frame(struct _Unwind_Context *ctx, void *arg) {
	struct gemm_unwind *u = arg;
	uintptr_t pc = (uintptr_t)_Unwind_GetIP(ctx);
	if (pc >= gemm_text_lo && pc < gemm_text_hi) {
		u->found = 1;
		return _URC_END_OF_STACK;
	}
	return ++u->depth < 64 ? _URC_NO_REASON : _URC_END_OF_STACK;
}
#endif

/* gemm_frames_in_library reports whether a caller of gemm_in_library is
   code in libcublasLt, by the return addresses in this thread's frame
   records. */
__attribute__((always_inline)) static inline int gemm_frames_in_library(void) {
	if (gemm_stack_state == 0) {
		gemm_stack_state = -1;
#ifdef __linux__
		pthread_attr_t attr;
		if (pthread_getattr_np(pthread_self(), &attr) == 0) {
			void *addr;
			size_t size;
			if (pthread_attr_getstack(&attr, &addr, &size) == 0) {
				gemm_stack_lo = (uintptr_t)addr;
				gemm_stack_hi = (uintptr_t)addr + size;
				gemm_stack_state = 1;
			}
			pthread_attr_destroy(&attr);
		}
#endif
	}
	if (gemm_stack_state < 0) return 0;
	const uintptr_t *fp = __builtin_frame_address(0);
	for (int depth = 0; depth < 64; depth++) {
		uintptr_t p = (uintptr_t)fp;
		if (p < gemm_stack_lo || p > gemm_stack_hi - 2 * sizeof(uintptr_t) || (p & (sizeof(uintptr_t) - 1)) != 0) return 0;
		if (fp[1] >= gemm_text_lo && fp[1] < gemm_text_hi) return 1;
		if (fp[0] <= p) return 0;
		fp = (const uintptr_t *)fp[0];
	}
	return 0;
}

/* gemm_in_library reports whether a caller of this function is code in
   libcublasLt: by frame records, or else by the unwinder. */
__attribute__((noinline)) static int gemm_in_library(void) {
	if (gemm_frames_in_library()) return 1;
#ifdef __linux__
	struct gemm_unwind u = {0, 0};
	_Unwind_Backtrace(gemm_unwind_frame, &u);
	return u.found;
#else
	return 0;
#endif
}

static int gemm_launch_cbid(CUpti_CallbackId cbid) {
	switch (cbid) {
	case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel:
	case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz:
	case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx:
	case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz:
	case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel:
	case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz:
		return 1;
	}
	return 0;
}

/* gemm_launch links a driver launch to the call pending on this thread, or
   ends that call when the launch is not the library's. */
static void gemm_launch(uint32_t correlation) {
	uint64_t id = gemm_pending;
	if (id == 0 || correlation == 0) return;
	if (!gemm_in_library()) {
		gemm_pending = 0;
		return;
	}
	atomic_fetch_add(&callback_active, 1);
	if (!atomic_load(&shutdown_started)) {
		char payload[128];
		int n = snprintf(payload, sizeof(payload), "{\"kind\":\"gemm_launch\",\"id\":%" PRIu64 ",\"correlation\":%u}", id, correlation);
		if (n > 0 && (size_t)n < sizeof(payload) && queue_pending(KIND_GEMM, now_ns(), payload, (size_t)n) == 0) atomic_fetch_add(&gemm_launches, 1);
	}
	atomic_fetch_sub(&callback_active, 1);
}

/* ------------------------------------------------------------------------
   Host run-state sampling.
   ------------------------------------------------------------------------ */

/* One thread's host run state, as its /proc stat file reported it. The CPU
   counters are in clock ticks, which is the unit the kernel writes and the
   unit this agent forwards: converting here would bake sysconf(_SC_CLK_TCK)
   from the producing host into a number the consumer cannot re-derive, so the
   tick rate travels in the capability record instead. */
struct host_thread {
	uint64_t tid;
	uint64_t utime_ticks;
	uint64_t stime_ticks;
	char state;
};

static int host_sample_hz;
static char host_sample_root[PATH_MAX];
static int host_sample_root_override;
/* "off", "sampling", "root_unavailable", "root_rejected" or "thread_failed".
   A reader must never have to infer from an absent record whether sampling was
   asked for, was refused, or simply found nothing. */
static const char *host_sample_disposition = "off";
static long host_sample_clk_tck = 100;
static int host_sample_running;
static pthread_t host_sampler;
static pthread_mutex_t host_sample_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t host_sample_cond = PTHREAD_COND_INITIALIZER;
static int host_sample_stop;
static _Atomic uint64_t host_samples_taken;
static _Atomic uint64_t host_sample_records;
static struct host_thread host_sample_threads[HOST_SAMPLE_MAX_THREADS];

/*
 * Parses the four fields of a /proc/<pid>/stat line this sampler reports: the
 * thread id, the run state, and the utime and stime counters.
 *
 * The comm field is the trap in this format. It is the thread's name, the
 * kernel does not escape it, and it can hold spaces and parentheses -- systemd
 * ships a process literally named "(sd-pam)", and any program may prctl its
 * own name to "a ) b". So scanning "%d (%s) %c", splitting the line on
 * whitespace, and searching for the first ')' all read the wrong fields for a
 * whole class of ordinary names, and read them as plausible numbers rather
 * than failing. The one guarantee the kernel does make is that comm is
 * bracketed and that the LAST ')' on the line closes it; that is what is used
 * here.
 *
 * Returns 0 on success and -1 on any malformed line, which the caller counts
 * rather than guesses at.
 */
static int parse_proc_stat(const char *buf, size_t len, struct host_thread *out) {
	if (buf == NULL || out == NULL || len == 0) return -1;
	size_t i = 0;
	while (i < len && buf[i] == ' ') i++;
	if (i >= len || buf[i] < '0' || buf[i] > '9') return -1;
	uint64_t tid = 0;
	for (; i < len && buf[i] >= '0' && buf[i] <= '9'; i++) {
		if (tid > (UINT64_MAX - (uint64_t)(buf[i] - '0')) / 10) return -1;
		tid = tid * 10 + (uint64_t)(buf[i] - '0');
	}
	size_t close = len;
	for (size_t j = len; j-- > 0;) if (buf[j] == ')') { close = j; break; }
	if (close == len) return -1;
	i = close + 1;
	while (i < len && buf[i] == ' ') i++;
	if (i >= len) return -1;
	char state = buf[i];
	if (!((state >= 'A' && state <= 'Z') || (state >= 'a' && state <= 'z'))) return -1;
	i++;
	/* utime and stime are the 11th and 12th fields after the state: ppid,
	   pgrp, session, tty_nr, tpgid, flags, minflt, cminflt, majflt and
	   cmajflt come between. tty_nr and tpgid are signed. */
	const int skip = 10;
	uint64_t times[2] = {0, 0};
	for (int k = 0; k < skip + 2; k++) {
		while (i < len && (buf[i] == ' ' || buf[i] == '\t')) i++;
		int negative = 0;
		if (i < len && buf[i] == '-') { negative = 1; i++; }
		if (i >= len || buf[i] < '0' || buf[i] > '9') return -1;
		uint64_t v = 0;
		for (; i < len && buf[i] >= '0' && buf[i] <= '9'; i++) {
			if (v > (UINT64_MAX - (uint64_t)(buf[i] - '0')) / 10) return -1;
			v = v * 10 + (uint64_t)(buf[i] - '0');
		}
		if (k >= skip) {
			if (negative) return -1;
			times[k - skip] = v;
		}
	}
	out->tid = tid;
	out->state = state;
	out->utime_ticks = times[0];
	out->stime_ticks = times[1];
	return 0;
}

/* Reads one task's stat file. Returns 0 on success, -1 if it could not be read
   or did not parse -- a thread that exits between the directory scan and this
   open is the ordinary case and is counted, not reported as a failure. */
static int read_task_stat(const char *dir, const char *name, struct host_thread *out) {
	char path[PATH_MAX];
	if (snprintf(path, sizeof(path), "%s/%s/stat", dir, name) >= (int)sizeof(path)) return -1;
	int fd = open(path, O_RDONLY | O_CLOEXEC);
	if (fd < 0) return -1;
	char buf[1024];
	size_t n = 0;
	for (;;) {
		ssize_t m = read(fd, buf + n, sizeof(buf) - n);
		if (m < 0 && errno == EINTR) continue;
		if (m <= 0) break;
		n += (size_t)m;
		if (n == sizeof(buf)) break;
	}
	(void)close(fd);
	if (n == 0) return -1;
	return parse_proc_stat(buf, n, out);
}

/* Emits one sample as one or more KIND_HOST records. A payload is capped at
   MAX_PAYLOAD and a process can have more threads than fit, so the thread
   array is split across parts rather than truncated: every part carries its
   sample index and its part index, the last part says so, and the totals ride
   on every part so a consumer that receives a partial set can tell. Truncating
   silently would report a subset of threads as the whole process, which is the
   shape of error this stream exists to remove. */
static void emit_host_sample(uint64_t index, uint64_t wall, size_t n,
	uint64_t unreadable, uint64_t overflowed) {
	size_t i = 0;
	unsigned part = 0;
	do {
		char payload[MAX_PAYLOAD];
		int p = snprintf(payload, sizeof(payload),
			"{\"kind\":\"host_sample\",\"sample\":%" PRIu64 ",\"part\":%u,\"wall\":%" PRIu64 ",\"threads\":[",
			index, part, wall);
		if (p < 0 || (size_t)p >= sizeof(payload)) return;
		size_t first = i;
		while (i < n) {
			char entry[128];
			int e = snprintf(entry, sizeof(entry), "%s{\"tid\":%" PRIu64 ",\"state\":\"%c\",\"utime_ticks\":%" PRIu64 ",\"stime_ticks\":%" PRIu64 "}",
				i > first ? "," : "", host_sample_threads[i].tid, host_sample_threads[i].state,
				host_sample_threads[i].utime_ticks, host_sample_threads[i].stime_ticks);
			if (e < 0 || (size_t)e >= sizeof(entry)) { i++; overflowed++; continue; }
			if ((size_t)p + (size_t)e + HOST_SAMPLE_SUFFIX_RESERVE >= sizeof(payload)) break;
			memcpy(payload + p, entry, (size_t)e);
			p += e;
			i++;
		}
		/* A part that admitted nothing while threads remain cannot make
		   progress; charge the entry that does not fit and move on rather
		   than spin. Unreachable at the current entry and reserve sizes, and
		   kept so a future widening of either cannot hang the sampler. */
		if (i == first && i < n) { i++; overflowed++; }
		int w = snprintf(payload + p, sizeof(payload) - (size_t)p,
			"],\"last\":%s,\"in_part\":%zu,\"threads_total\":%zu,\"unreadable\":%" PRIu64 ",\"overflowed\":%" PRIu64 "}",
			i >= n ? "true" : "false", i - first, n, unreadable, overflowed);
		if (w < 0 || (size_t)(p + w) >= sizeof(payload)) return;
		p += w;
		if (queue_pending(KIND_HOST, wall, payload, (size_t)p) == 0) atomic_fetch_add(&host_sample_records, 1);
		part++;
	} while (i < n);
}

/* Takes one sample of every task under the configured root. The sampler thread
   is itself one of those tasks and is reported like any other: it is a real
   thread of this process consuming real time, and omitting it would make the
   sample disagree with the process it describes. */
static int take_host_sample(uint64_t index) {
	DIR *d = opendir(host_sample_root);
	if (d == NULL) return -1;
	size_t n = 0;
	uint64_t unreadable = 0, overflowed = 0;
	struct dirent *e;
	while ((e = readdir(d)) != NULL) {
		if (e->d_name[0] == '.') continue;
		if (n == HOST_SAMPLE_MAX_THREADS) { overflowed++; continue; }
		if (read_task_stat(host_sample_root, e->d_name, &host_sample_threads[n]) != 0) { unreadable++; continue; }
		n++;
	}
	(void)closedir(d);
	emit_host_sample(index, now_ns(), n, unreadable, overflowed);
	atomic_fetch_add(&host_samples_taken, 1);
	return 0;
}

static void *host_sample_worker(void *arg) {
	(void)arg;
	long period = 1000000000L / host_sample_hz;
	for (uint64_t index = 0;; index++) {
		if (take_host_sample(index) != 0) {
			/* The root went away mid-run. Say so once and stop, rather than
			   fail every interval for the rest of the process. */
			const char *failure = "{\"state\":\"producer_failure\",\"error\":\"host sample root became unreadable\"}";
			(void)queue_pending(KIND_FAILURE, now_ns(), failure, strlen(failure));
			return NULL;
		}
		pthread_mutex_lock(&host_sample_mu);
		if (!host_sample_stop) {
			struct timespec deadline;
			if (clock_gettime(CLOCK_REALTIME, &deadline) == 0) {
				deadline.tv_nsec += period;
				if (deadline.tv_nsec >= 1000000000L) { deadline.tv_nsec -= 1000000000L; deadline.tv_sec++; }
				while (!host_sample_stop) {
					if (pthread_cond_timedwait(&host_sample_cond, &host_sample_mu, &deadline) == ETIMEDOUT) break;
				}
			}
		}
		int stop = host_sample_stop;
		pthread_mutex_unlock(&host_sample_mu);
		if (stop) return NULL;
	}
}

/* Reads the sampler's configuration and settles its disposition. Called before
   the capability record is built so that record states the outcome. */
static void configure_host_sampler(void) {
	const char *rate = getenv("CUDAPROF_HOST_SAMPLE");
	if (rate == NULL || *rate == '\0') return;
	char *end = NULL;
	long hz = strtol(rate, &end, 10);
	if (end == rate || (end != NULL && *end != '\0') || hz <= 0) return;
	if (hz > HOST_SAMPLE_MAX_HZ) hz = HOST_SAMPLE_MAX_HZ;
	host_sample_hz = (int)hz;
#if defined(_SC_CLK_TCK)
	long tck = sysconf(_SC_CLK_TCK);
	if (tck > 0) host_sample_clk_tck = tck;
#endif
	/* The root is overridable so the controls can drive the shipping sampler
	   against a fixture tree on a host with no /proc. An overridden root is
	   reported in the capability record: a fixture run and a real one must
	   never be indistinguishable to a reader. */
	const char *root = getenv("CUDAPROF_HOST_SAMPLE_ROOT");
	if (root != NULL && *root != '\0') {
		size_t len = strlen(root);
		int printable = len <= HOST_SAMPLE_ROOT_MAX;
		for (size_t i = 0; printable && i < len; i++) {
			unsigned char c = (unsigned char)root[i];
			if (c < 0x20 || c == '"' || c == '\\') printable = 0;
		}
		if (!printable) { host_sample_hz = 0; host_sample_disposition = "root_rejected"; return; }
		memcpy(host_sample_root, root, len + 1);
		host_sample_root_override = 1;
	} else {
		/* /proc is where this tier's data lives and it exists only on Linux.
		   Elsewhere the sampler compiles and is inert, and the capability
		   record says which. */
#ifdef __linux__
		memcpy(host_sample_root, "/proc/self/task", sizeof("/proc/self/task"));
#else
		host_sample_hz = 0;
		host_sample_disposition = "root_unavailable";
		return;
#endif
	}
	DIR *probe = opendir(host_sample_root);
	if (probe == NULL) { host_sample_hz = 0; host_sample_disposition = "root_unavailable"; return; }
	(void)closedir(probe);
	host_sample_disposition = "sampling";
}

static void start_host_sampler(void) {
	if (host_sample_hz <= 0) return;
	if (pthread_create(&host_sampler, NULL, host_sample_worker, NULL) != 0) {
		host_sample_disposition = "thread_failed";
		const char *failure = "{\"state\":\"producer_failure\",\"error\":\"host sampler thread creation failed\"}";
		(void)write_record(KIND_FAILURE, now_ns(), failure, strlen(failure));
		return;
	}
	host_sample_running = 1;
}

/* Stopped and joined before the drain, so its last record is in the ring
   before worker_stop and reaches the journal ahead of the finalized record. */
static void stop_host_sampler(void) {
	if (!host_sample_running) return;
	pthread_mutex_lock(&host_sample_mu);
	host_sample_stop = 1;
	pthread_cond_signal(&host_sample_cond);
	pthread_mutex_unlock(&host_sample_mu);
	if (pthread_join(host_sampler, NULL) != 0) atomic_store(&producer_failed, 1);
	host_sample_running = 0;
}

/* device_active records that CUPTI asked this process for an activity buffer,
   which it does only when there is device work to store. It is the one piece
   of evidence a journal that never finalizes still carries: activity and API
   records stay inside CUPTI until a flush, so a target that exits through
   exit_group leaves a journal holding bootstrap and capability and nothing
   else -- byte for byte what a process that never touched CUDA leaves. A
   consumer reading that cannot tell a lost capture from a wrapper script, and
   refusing both costs a shell-wrapped run every record its children collected.

   Written once, directly rather than through the queue, because the drain that
   empties the queue is exactly what the exiting target does not reach. The
   constructor publishes it if the request arrives before the capability record
   is down, since a consumer refuses an activity record that precedes one. */
static void publish_device_active(void) {
	int expected = 0;
	if (!atomic_compare_exchange_strong(&device_active_written, &expected, 1)) return;
	static const char record[] = "{\"kind\":\"activity\",\"subtype\":\"device_active\"}";
	(void)write_record(KIND_ACTIVITY, now_ns(), record, sizeof(record) - 1);
}

static void note_device_active(void) {
	if (atomic_exchange(&device_active, 1)) return;
	if (atomic_load(&journal_ready)) publish_device_active();
}

static void CUPTIAPI buffer_requested(uint8_t **buffer, size_t *size, size_t *max_records) {
	*buffer = NULL; *size = 0;
	if (atomic_load(&shutdown_started)) { *max_records = 0; return; }
	note_device_active();
	for (int i = 0; i < BUFFER_POOL; i++) {
		int expected = 0;
		if (atomic_compare_exchange_strong(&buffer_busy[i], &expected, 1)) { *buffer = buffer_pool[i]; *size = BUFFER_BYTES; break; }
	}
	*max_records = 0;
}

static void release_buffer(uint8_t *buffer) {
	for (int i = 0; i < BUFFER_POOL; i++) if (buffer == buffer_pool[i]) { atomic_store(&buffer_busy[i], 0); return; }
}

static int buffers_in_use(void) {
	for (int i = 0; i < BUFFER_POOL; i++) if (atomic_load(&buffer_busy[i])) return 1;
	return 0;
}

/* Records how long one buffer-completed callback owned the CUPTI buffer.
   Callback latency is the cost backpressure trades loss for, so it is
   measured on every exit path rather than inferred. */
static void note_callback_latency(uint64_t began) {
	uint64_t elapsed = now_ns() - began;
	atomic_fetch_add(&callback_count, 1);
	atomic_fetch_add(&callback_total_nanos, elapsed);
	uint64_t observed = atomic_load(&callback_max_nanos);
	while (elapsed > observed && !atomic_compare_exchange_weak(&callback_max_nanos, &observed, elapsed)) { }
}

#include "cupti_memory.h"

static void CUPTIAPI buffer_completed(CUcontext context, uint32_t stream_id, uint8_t *buffer, size_t size, size_t valid_size) {
	uint64_t callback_began = now_ns();
	atomic_fetch_add(&callback_active, 1);
	if (atomic_load(&callbacks_closed)) {
		release_buffer(buffer);
		atomic_fetch_sub(&callback_active, 1);
		note_callback_latency(callback_began);
		return;
	}
	uintptr_t begin = (uintptr_t)buffer, end, rp;
	int owned = 0;
	for (int i = 0; i < BUFFER_POOL; i++) if (buffer == buffer_pool[i]) { owned = 1; break; }
	if (!owned || buffer == NULL || valid_size > size || size > BUFFER_BYTES || begin > UINTPTR_MAX - valid_size) {
		const char *failure = "{\"state\":\"producer_failure\",\"error\":\"invalid CUPTI callback buffer span\"}";
		(void)queue_pending(KIND_FAILURE, now_ns(), failure, strlen(failure));
		if (owned) release_buffer(buffer);
		atomic_fetch_sub(&callback_active, 1);
		note_callback_latency(callback_began);
		return;
	}
	if (atomic_load(&callbacks_closed)) {
		release_buffer(buffer);
		atomic_fetch_sub(&callback_active, 1);
		note_callback_latency(callback_began);
		return;
	}
	end = begin + valid_size;
	CUpti_Activity *record = NULL;
	CUptiResult next;
	while ((next = cuptiActivityGetNextRecord(buffer, valid_size, &record)) == CUPTI_SUCCESS && record != NULL) {
		rp = (uintptr_t)record;
		if (rp < begin || rp + sizeof(CUpti_Activity) < rp || rp + sizeof(CUpti_Activity) > end) {
			const char *failure = "{\"state\":\"producer_failure\",\"error\":\"CUPTI record header outside returned buffer\"}";
			(void)queue_pending(KIND_FAILURE, now_ns(), failure, strlen(failure));
			break;
		}
		if (record->kind == CUPTI_ACTIVITY_KIND_KERNEL || record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
			if (rp + sizeof(CUDAPROF_KERNEL_TYPE) < rp || rp + sizeof(CUDAPROF_KERNEL_TYPE) > end) { const char *failure = "{\"state\":\"producer_failure\",\"error\":\"short kernel record\"}"; (void)queue_pending(KIND_FAILURE, now_ns(), failure, strlen(failure)); break; }
			const CUDAPROF_KERNEL_TYPE *k = (const CUDAPROF_KERNEL_TYPE *)record;
			(void)append_activity("kernel", k->name, k->start, k->end, 0, k->queued, k->submitted, 0, 0, 0, k->deviceId, k->contextId, k->streamId, k->correlationId,
				(uint32_t)(k->gridX > 0 ? k->gridX : 0),
				(uint32_t)(k->gridY > 0 ? k->gridY : 0),
				(uint32_t)(k->gridZ > 0 ? k->gridZ : 0),
				(uint32_t)(k->blockX > 0 ? k->blockX : 0),
				(uint32_t)(k->blockY > 0 ? k->blockY : 0),
				(uint32_t)(k->blockZ > 0 ? k->blockZ : 0),
				(uint32_t)k->registersPerThread,
				(uint32_t)(k->dynamicSharedMemory > 0 ? k->dynamicSharedMemory : 0),
				CUDAPROF_GRAPH_ID(k), CUDAPROF_GRAPH_NODE_ID(k), CUDAPROF_GRAPH_NODE_ID(k) ? graph_module(CUDAPROF_GRAPH_NODE_ID(k)) : launch_module(k->correlationId));
		} else if (record->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
			if (rp + sizeof(CUpti_ActivityMemcpy6) < rp || rp + sizeof(CUpti_ActivityMemcpy6) > end) { const char *failure = "{\"state\":\"producer_failure\",\"error\":\"short Memcpy6 record\"}"; (void)queue_pending(KIND_FAILURE, now_ns(), failure, strlen(failure)); break; }
			const CUpti_ActivityMemcpy6 *m6 = (const CUpti_ActivityMemcpy6 *)record;
			(void)append_activity("memcpy", "", m6->start, m6->end, m6->bytes, 0, 0, m6->copyKind, 0, m6->runtimeCorrelationId, m6->deviceId, m6->contextId, m6->streamId, m6->correlationId,
				0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
		} else if (record->kind == CUPTI_ACTIVITY_KIND_MEMSET) {
			if (rp + sizeof(CUpti_ActivityMemset4) < rp || rp + sizeof(CUpti_ActivityMemset4) > end) { const char *failure = "{\"state\":\"producer_failure\",\"error\":\"short Memset4 record\"}"; (void)queue_pending(KIND_FAILURE, now_ns(), failure, strlen(failure)); break; }
			const CUpti_ActivityMemset4 *ms = (const CUpti_ActivityMemset4 *)record;
			(void)append_activity("memset", "", ms->start, ms->end, ms->bytes, 0, 0, 0, ms->value, 0, ms->deviceId, ms->contextId, ms->streamId, ms->correlationId,
				0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
		} else if (record->kind == CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER) {
			if (rp + sizeof(CUpti_ActivityUnifiedMemoryCounter3) < rp || rp + sizeof(CUpti_ActivityUnifiedMemoryCounter3) > end) { const char *failure = "{\"state\":\"producer_failure\",\"error\":\"short UnifiedMemoryCounter3 record\"}"; (void)queue_pending(KIND_FAILURE, now_ns(), failure, strlen(failure)); break; }
			(void)append_unified_memory((const CUpti_ActivityUnifiedMemoryCounter3 *)record);
		} else if (record->kind == CUPTI_ACTIVITY_KIND_DRIVER || record->kind == CUPTI_ACTIVITY_KIND_RUNTIME) {
			if (rp + sizeof(CUpti_ActivityAPI) < rp || rp + sizeof(CUpti_ActivityAPI) > end) { const char *failure = "{\"state\":\"producer_failure\",\"error\":\"short API record\"}"; (void)queue_pending(KIND_FAILURE, now_ns(), failure, strlen(failure)); break; }
			const CUpti_ActivityAPI *api = (const CUpti_ActivityAPI *)record;
			(void)append_api(record->kind == CUPTI_ACTIVITY_KIND_DRIVER ? "driver" : "runtime", api);
		} else if (record->kind == CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION) {
			if (rp + sizeof(CUpti_ActivityExternalCorrelation) < rp || rp + sizeof(CUpti_ActivityExternalCorrelation) > end) { const char *failure = "{\"state\":\"producer_failure\",\"error\":\"short external correlation record\"}"; (void)queue_pending(KIND_FAILURE, now_ns(), failure, strlen(failure)); break; }
			const CUpti_ActivityExternalCorrelation *x = (const CUpti_ActivityExternalCorrelation *)record;
			(void)append_external((uint32_t)x->externalKind, x->externalId, x->correlationId);
		} else if (record->kind == CUPTI_ACTIVITY_KIND_OVERHEAD) {
			if (rp + sizeof(CUpti_ActivityOverhead) < rp || rp + sizeof(CUpti_ActivityOverhead) > end) { const char *failure = "{\"state\":\"producer_failure\",\"error\":\"short Overhead record\"}"; (void)queue_pending(KIND_FAILURE, now_ns(), failure, strlen(failure)); break; }
			(void)append_overhead((const CUpti_ActivityOverhead *)record);
		} else if (record->kind == CUPTI_ACTIVITY_KIND_GRAPH_TRACE) {
			if (rp + sizeof(CUpti_ActivityGraphTrace) < rp || rp + sizeof(CUpti_ActivityGraphTrace) > end) { const char *failure = "{\"state\":\"producer_failure\",\"error\":\"short GraphTrace record\"}"; (void)queue_pending(KIND_FAILURE, now_ns(), failure, strlen(failure)); break; }
			(void)append_graph_trace((const CUpti_ActivityGraphTrace *)record);
		} else if (record->kind == CUPTI_ACTIVITY_KIND_SYNCHRONIZATION) {
			if (rp + sizeof(CUpti_ActivitySynchronization) < rp || rp + sizeof(CUpti_ActivitySynchronization) > end) { const char *failure = "{\"state\":\"producer_failure\",\"error\":\"short Synchronization record\"}"; (void)queue_pending(KIND_FAILURE, now_ns(), failure, strlen(failure)); break; }
			(void)append_sync((const CUpti_ActivitySynchronization *)record);
		} else if (record->kind == CUPTI_ACTIVITY_KIND_MARKER) {
			if (rp + sizeof(CUpti_ActivityMarker2) < rp || rp + sizeof(CUpti_ActivityMarker2) > end) { const char *failure = "{\"state\":\"producer_failure\",\"error\":\"short Marker2 record\"}"; (void)queue_pending(KIND_FAILURE, now_ns(), failure, strlen(failure)); break; }
			(void)append_marker((const CUpti_ActivityMarker2 *)record);
		} else {
			/* Every other anomaly in this loop is recorded; a kind nobody
			   decodes was not. That silence is what let an unenabled MEMSET
			   read as idle device time for as long as it did: the consumer
			   cannot tell a kind that produced no records from a kind whose
			   records were dropped here. The record is skipped either way --
			   there is nothing to decode -- but the reader is told. */
			char failure[128];
			int fn = snprintf(failure, sizeof(failure), "{\"state\":\"producer_failure\",\"error\":\"undecoded CUPTI activity kind %u\"}", (unsigned)record->kind);
			if (fn > 0 && fn < (int)sizeof(failure)) (void)queue_pending(KIND_FAILURE, now_ns(), failure, (size_t)fn);
		}
	}
	if (next != CUPTI_ERROR_MAX_LIMIT_REACHED && next != CUPTI_SUCCESS) {
		const char *failure = "{\"state\":\"producer_failure\",\"error\":\"malformed CUPTI activity record\"}";
		(void)queue_pending(KIND_FAILURE, now_ns(), failure, strlen(failure));
	}
	size_t dropped = 0;
	if (cuptiActivityGetNumDroppedRecords(context, stream_id, &dropped) == CUPTI_SUCCESS) { atomic_fetch_add(&dropped_count, (uint64_t)dropped); atomic_fetch_add(&native_dropped_count, (uint64_t)dropped); }
	else { const char *failure = "{\"state\":\"producer_failure\",\"error\":\"CUPTI dropped-record query failed\"}"; (void)queue_pending(KIND_FAILURE, now_ns(), failure, strlen(failure)); }
	release_buffer(buffer);
	(void)size;
	atomic_fetch_sub(&callback_active, 1);
	note_callback_latency(callback_began);
}

/* Async-signal-safe by construction: an atomic store and one byte on the
   self-pipe, nothing else. No CUPTI call, no malloc, no lock, and no journal
   I/O happens in signal context; the terminator thread does all of that in a
   normal execution context. */
static void cudaprof_sigterm_handler(int signo, siginfo_t *info, void *context) {
	(void)signo;
	(void)info;
	(void)context;
	int saved = errno;
	atomic_store(&term_signalled, 1);
	if (term_pipe[1] >= 0) {
		const unsigned char byte = 1;
		ssize_t n;
		do { n = write(term_pipe[1], &byte, 1); } while (n < 0 && errno == EINTR);
		(void)n;
	}
	errno = saved;
	/* The target's own disposition is NOT run here. It used to be, inline,
	   and that made SIGTERM useless as a finalization route for any target
	   that installs a handler of its own -- which is every Go program, since
	   the Go runtime always installs one, and most C++ servers. The target's
	   handler would terminate the process while the terminator thread was
	   still waiting to be scheduled, and the journal would end unfinalized.
	   The terminator restores the previous disposition and re-raises once the
	   journal is closed, which is what the default-disposition case always
	   did; this is only the same thing for the rest. */
}

static void *terminator_thread(void *arg) {
	(void)arg;
	for (;;) {
		unsigned char byte;
		ssize_t n = read(term_pipe[0], &byte, 1);
		if (n > 0) break;
		if (n == 0) return NULL; /* write end closed: nothing to finalize */
		if (errno == EINTR) continue;
		return NULL;
	}
	if (!atomic_load(&term_signalled)) return NULL;
	cudaprof_cupti_finalize("signal", 0);
	/* Finalization is complete, so put back whatever disposition the target
	   had and deliver the signal again. One path covers both cases because
	   prev_sigterm IS the previous disposition: SIG_DFL for a target with
	   none, and the target's own handler otherwise. install_sigterm refuses to
	   install over SIG_IGN, so it is never that.

	   kill rather than raise: raise delivers to the calling thread, and this
	   thread belongs to the agent rather than to the target's runtime. A Go
	   runtime handles a process-directed SIGTERM through its normal path and
	   exits; a signal raised on a thread it does not know takes its
	   foreign-thread path instead.

	   The cost of doing this after finalization rather than before is that the
	   target's termination is delayed by the flush, and a finalize that hangs
	   leaves the target alive where it would previously have died. That
	   exposure is not new -- it is what the default-disposition case has
	   always had -- and the alternative is a route that silently collects
	   nothing. */
	(void)sigaction(SIGTERM, &prev_sigterm, NULL);
	(void)kill(getpid(), SIGTERM);
	return NULL;
}

/* pipe2 is the Linux path this producer ships on; the portable fallback keeps
   the controls runnable on a development host. */
static int open_term_pipe(void) {
#ifdef __linux__
	return pipe2(term_pipe, O_CLOEXEC);
#else
	if (pipe(term_pipe) != 0) return -1;
	for (int i = 0; i < 2; i++) (void)fcntl(term_pipe[i], F_SETFD, FD_CLOEXEC);
	return 0;
#endif
}

static void install_sigterm(void) {
	struct sigaction current;
	if (sigaction(SIGTERM, NULL, &current) != 0) return;
	/* A target that ignores SIGTERM is not terminating; do not resurrect the
	   signal just to finalize. */
	if (current.sa_handler == SIG_IGN) return;
	if (open_term_pipe() != 0) { term_pipe[0] = term_pipe[1] = -1; return; }
	if (pthread_create(&terminator, NULL, terminator_thread, NULL) != 0) {
		close(term_pipe[0]); close(term_pipe[1]); term_pipe[0] = term_pipe[1] = -1;
		return;
	}
	prev_sigterm = current;
	struct sigaction sa;
	memset(&sa, 0, sizeof(sa));
	sa.sa_sigaction = cudaprof_sigterm_handler;
	sa.sa_flags = SA_SIGINFO | SA_RESTART;
	sigfillset(&sa.sa_mask);
	if (sigaction(SIGTERM, &sa, NULL) == 0) sigterm_installed = 1;
}

/* Crash trace. With CUDAPROF_CRASH_TRACE=N the agent keeps the last API
   callbacks the target made in a ring and, when the target dies of a fault,
   writes the last N of them to standard error before the process goes. A
   capture that ends in a crash otherwise ends before any finalization: the
   records still inside CUPTI are lost, and which call the target was in when
   it died takes a debugger to find out.

   The ring holds what a callback can record without allocating or locking:
   the time, the thread, the callback id and site, the correlation id and
   CUPTI's own name for the function, which is a string CUPTI keeps for the
   life of the process. A slot is claimed with one fetch-add and published by
   storing its sequence number last, so the handler can tell a slot that was
   being written when the fault arrived from a finished one. */
#define CRASH_RING 4096
struct crash_slot {
	_Atomic uint64_t seq; /* claim index + 1 once written, 0 while empty */
	uint64_t ns;
	uint64_t tid;
	const char *name;
	uint32_t cbid;
	uint32_t correlation;
	uint16_t domain;
	uint16_t site;
};
static struct crash_slot crash_ring[CRASH_RING];
static _Atomic uint64_t crash_next;
static unsigned crash_depth; /* 0: crash trace off */
static int crash_installed;
static _Atomic int crash_dumped;
static const int crash_signals[] = {SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGABRT};
#define NCRASH_SIGNALS (sizeof(crash_signals) / sizeof(crash_signals[0]))
static struct sigaction prev_crash[NCRASH_SIGNALS];

static uint64_t crash_tid(void) {
#ifdef __linux__
	return (uint64_t)syscall(SYS_gettid);
#else
	return (uint64_t)(uintptr_t)pthread_self();
#endif
}

static void crash_note(CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const CUpti_CallbackData *data) {
	uint64_t i = atomic_fetch_add_explicit(&crash_next, 1, memory_order_relaxed);
	struct crash_slot *s = &crash_ring[i % CRASH_RING];
	atomic_store_explicit(&s->seq, 0, memory_order_relaxed);
	s->ns = now_ns();
	s->tid = crash_tid();
	s->domain = (uint16_t)domain;
	s->cbid = (uint32_t)cbid;
	s->site = data != NULL ? (uint16_t)data->callbackSite : 0;
	s->correlation = data != NULL ? data->correlationId : 0;
	s->name = data != NULL ? data->functionName : NULL;
	atomic_store_explicit(&s->seq, i + 1, memory_order_release);
}

/* The dump runs in a signal handler, so it formats by hand and writes with
   write(2): printf and its relatives are not async-signal-safe. */
struct crash_line {
	char b[512];
	size_t n;
};

static void crash_str(struct crash_line *l, const char *s) {
	while (s != NULL && *s != '\0' && l->n < sizeof(l->b) - 1) l->b[l->n++] = *s++;
}

static void crash_uint(struct crash_line *l, uint64_t v, unsigned base) {
	char tmp[24];
	int n = 0;
	do {
		tmp[n++] = "0123456789abcdef"[v % base];
		v /= base;
	} while (v != 0);
	while (n > 0 && l->n < sizeof(l->b) - 1) l->b[l->n++] = tmp[--n];
}

static void crash_flush(struct crash_line *l) {
	l->b[l->n++] = '\n';
	for (size_t off = 0; off < l->n;) {
		ssize_t w = write(STDERR_FILENO, l->b + off, l->n - off);
		if (w > 0) off += (size_t)w;
		else if (w < 0 && errno == EINTR) continue;
		else break;
	}
	l->n = 0;
}

static const char *crash_domain_name(uint16_t domain) {
	switch (domain) {
	case CUPTI_CB_DOMAIN_DRIVER_API: return "driver";
	case CUPTI_CB_DOMAIN_RUNTIME_API: return "runtime";
	case CUPTI_CB_DOMAIN_RESOURCE: return "resource";
	}
	return "domain?";
}

static void crash_dump(int sig, const siginfo_t *si) {
	int saved = errno;
	uint64_t end = atomic_load(&crash_next);
	uint64_t n = end < crash_depth ? end : crash_depth;
	uint64_t now = now_ns();
	struct crash_line l = {.n = 0};
	crash_str(&l, "cudaprof: signal ");
	crash_uint(&l, (uint64_t)sig, 10);
	/* si_addr means something only for a fault the kernel raised. */
	if (si != NULL && si->si_code > 0 && (sig == SIGSEGV || sig == SIGBUS)) {
		crash_str(&l, " at 0x");
		crash_uint(&l, (uint64_t)(uintptr_t)si->si_addr, 16);
	}
	crash_str(&l, " on thread ");
	crash_uint(&l, crash_tid(), 10);
	crash_str(&l, "; last ");
	crash_uint(&l, n, 10);
	crash_str(&l, " of ");
	crash_uint(&l, end, 10);
	crash_str(&l, " CUDA API callbacks, oldest first:");
	crash_flush(&l);
	for (uint64_t i = end - n; i < end; i++) {
		const struct crash_slot *s = &crash_ring[i % CRASH_RING];
		crash_str(&l, "cudaprof:   ");
		if (atomic_load_explicit(&s->seq, memory_order_acquire) != i + 1) {
			crash_str(&l, "(overwritten or being written)");
			crash_flush(&l);
			continue;
		}
		crash_uint(&l, now > s->ns ? (now - s->ns) / 1000 : 0, 10);
		crash_str(&l, "us before, tid ");
		crash_uint(&l, s->tid, 10);
		crash_str(&l, " ");
		crash_str(&l, crash_domain_name(s->domain));
		crash_str(&l, " cbid ");
		crash_uint(&l, s->cbid, 10);
		crash_str(&l, s->site == CUPTI_API_ENTER ? " enter" : " exit");
		crash_str(&l, " correlation ");
		crash_uint(&l, s->correlation, 10);
		if (s->name != NULL) {
			crash_str(&l, " ");
			crash_str(&l, s->name);
		}
		crash_flush(&l);
	}
	errno = saved;
}

/* crash_handler dumps the ring once and hands the signal back to whatever
   disposition the target had before the agent loaded -- for most targets the
   default, which ends the process with the signal. A fault raised by the
   instruction that caused it is simply returned into: the instruction runs
   again and faults under the restored disposition. A signal someone sent
   (kill, raise, abort) would not recur, so it is raised again.

   A Go runtime installs its own handlers after this one and forwards to it a
   fault raised outside Go code, which is where a CUDA crash happens; SA_ONSTACK
   is what the runtime requires of a handler it forwards to. */
static void crash_handler(int sig, siginfo_t *si, void *uc) {
	(void)uc;
	if (atomic_exchange(&crash_dumped, 1) == 0) crash_dump(sig, si);
	for (size_t i = 0; i < NCRASH_SIGNALS; i++) {
		if (crash_signals[i] == sig) (void)sigaction(sig, &prev_crash[i], NULL);
	}
	if (sig == SIGABRT || si == NULL || si->si_code <= 0) (void)raise(sig);
}

static void install_crash_trace(void) {
	for (size_t i = 0; i < NCRASH_SIGNALS; i++) {
		struct sigaction sa;
		memset(&sa, 0, sizeof(sa));
		sa.sa_sigaction = crash_handler;
		sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
		sigemptyset(&sa.sa_mask);
		if (sigaction(crash_signals[i], &sa, &prev_crash[i]) != 0) continue;
		/* A target that ignores the signal keeps ignoring it. */
		if (prev_crash[i].sa_handler == SIG_IGN) {
			(void)sigaction(crash_signals[i], &prev_crash[i], NULL);
			continue;
		}
		crash_installed = 1;
	}
}

/* exit_group finalization, for a target whose runtime leaves by issuing the
   exit_group syscall itself -- every Go program -- and so never reaches the
   destructor, atexit or cudaprof_finalize. A seccomp filter hands each
   exit_group to a user-space listener instead of running it. The calling
   thread waits in the kernel while every other thread, this agent's among
   them, keeps running; exit_watcher finalizes and then tells the kernel to
   continue the syscall, so the process exits exactly as it would have, with
   its own status, once the finalized record is on disk.

   The filter is the one piece of this agent that outlives it, which is what
   makes the mode opt-in:

   - Installing a filter without CAP_SYS_ADMIN needs no_new_privs, which this
     sets for the process: from here on an exec cannot gain privileges, so a
     setuid program the target runs runs as the target's user.
   - Filters survive fork and exec, so every descendant's exit_group reaches
     this listener too, and the kernel allows no second listener in the chain.
     A descendant that loads its own agent therefore listens on a unix socket
     named for the session and its pid instead (exit_rendezvous_main), and
     exit_watcher, holding that descendant's exit_group, asks it to finalize
     there before continuing the syscall. A descendant without an agent is
     continued at once. A descendant that outlives this process finds the
     listener gone, and its exit_group fails with ENOSYS; a C child then falls
     back to exit, a Go child crashes.

   A finalize that hangs holds the exit, as it does for SIGTERM. */
#ifdef __linux__
#if defined(__aarch64__)
#define EXIT_GROUP_AUDIT_ARCH AUDIT_ARCH_AARCH64
#elif defined(__x86_64__)
#define EXIT_GROUP_AUDIT_ARCH AUDIT_ARCH_X86_64
#endif
#ifndef SECCOMP_FILTER_FLAG_TSYNC_ESRCH
#define SECCOMP_FILTER_FLAG_TSYNC_ESRCH (1UL << 4)
#endif
#ifndef SECCOMP_USER_NOTIF_FLAG_CONTINUE
#define SECCOMP_USER_NOTIF_FLAG_CONTINUE (1UL << 0)
#endif
#endif

#if defined(__linux__) && defined(EXIT_GROUP_AUDIT_ARCH) && defined(SECCOMP_IOCTL_NOTIF_RECV)
/* owns_thread reports whether a notification's thread id names a thread of
   this process rather than of a child that inherited the filter. */
static int owns_thread(uint32_t tid) {
	char path[64];
	if ((uint64_t)tid == self_pid) return 1;
	if (snprintf(path, sizeof(path), "/proc/self/task/%u", tid) >= (int)sizeof(path)) return 0;
	return access(path, F_OK) == 0;
}

/* rendezvous_address fills addr with the abstract socket name of process
   pid's rendezvous and returns its length, or 0 if there is no session. */
static socklen_t rendezvous_address(struct sockaddr_un *addr, uint64_t pid) {
	const char *session = getenv("CUDAPROF_SESSION");
	if (session == NULL || session[0] == '\0') return 0;
	memset(addr, 0, sizeof(*addr));
	addr->sun_family = AF_UNIX;
	/* sun_path[0] stays 0: an abstract name, which leaves nothing behind. */
	int n = snprintf(addr->sun_path + 1, sizeof(addr->sun_path) - 1, "cudaprof-exit-%s-%" PRIu64, session, pid);
	if (n <= 0 || (size_t)n >= sizeof(addr->sun_path) - 1) return 0;
	return (socklen_t)(offsetof(struct sockaddr_un, sun_path) + 1 + (size_t)n);
}

/* thread_group returns the process that thread tid belongs to, or 0. */
static uint64_t thread_group(uint32_t tid) {
	char path[64], line[128];
	if (snprintf(path, sizeof(path), "/proc/%u/status", tid) >= (int)sizeof(path)) return 0;
	FILE *f = fopen(path, "re");
	if (f == NULL) return 0;
	uint64_t tgid = 0;
	while (fgets(line, sizeof(line), f) != NULL) {
		if (sscanf(line, "Tgid: %" SCNu64, &tgid) == 1) break;
	}
	fclose(f);
	return tgid;
}

/* finalize_descendant asks the process containing thread tid to finalize,
   and waits for it to answer or give up. A process with no rendezvous has no
   agent of its own to finalize. */
static void finalize_descendant(uint32_t tid) {
	struct sockaddr_un addr;
	uint64_t tgid = thread_group(tid);
	socklen_t len = tgid != 0 ? rendezvous_address(&addr, tgid) : 0;
	if (len == 0) return;
	int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
	if (fd < 0) return;
	/* Longer than finalize's own 10 s wait for a concurrent finalize, so a
	   descendant that answers at all is heard. */
	struct timeval tv = {30, 0};
	(void)setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
	(void)setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
	char b = 'f';
	if (connect(fd, (struct sockaddr *)&addr, len) == 0 && write(fd, &b, 1) == 1) {
		ssize_t n = read(fd, &b, 1);
		(void)n;
	}
	(void)close(fd);
}

static void *exit_rendezvous_main(void *arg) {
	(void)arg;
	for (;;) {
		int fd = accept4(exit_rendezvous, NULL, NULL, SOCK_CLOEXEC);
		if (fd < 0) {
			if (errno == EINTR || errno == ECONNABORTED) continue;
			return NULL;
		}
		/* Only this user's processes may end the capture. */
		struct ucred cred;
		socklen_t cl = sizeof(cred);
		char b;
		if (getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &cred, &cl) == 0 && cred.uid == getuid() && read(fd, &b, 1) == 1) {
			cudaprof_cupti_finalize("exit_group", 0);
			/* The answer only ends the watcher's wait; a watcher that gave
			   up has nothing to be told. */
			ssize_t n = write(fd, &b, 1);
			(void)n;
		}
		(void)close(fd);
	}
}

/* install_exit_rendezvous starts the rendezvous a descendant uses when an
   ancestor already holds the listener. It returns NULL or the step that
   failed with errno set. */
static const char *install_exit_rendezvous(void) {
	struct sockaddr_un addr;
	socklen_t len = rendezvous_address(&addr, self_pid);
	if (len == 0) { errno = EINVAL; return "rendezvous name"; }
	int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
	if (fd < 0) return "socket(rendezvous)";
	if (bind(fd, (struct sockaddr *)&addr, len) != 0 || listen(fd, 4) != 0) {
		int e = errno;
		(void)close(fd);
		errno = e;
		return "bind(rendezvous)";
	}
	exit_rendezvous = fd;
	pthread_t id;
	int rc = pthread_create(&id, NULL, exit_rendezvous_main, NULL);
	if (rc != 0) {
		(void)close(exit_rendezvous);
		exit_rendezvous = -1;
		errno = rc;
		return "pthread_create(exit rendezvous)";
	}
	(void)pthread_detach(id);
	return NULL;
}

/* exit_watcher serves exit_listener. */
static pthread_t exit_watcher;

static void *exit_watcher_main(void *arg) {
	(void)arg;
	for (;;) {
		struct seccomp_notif req;
		memset(&req, 0, sizeof(req));
		if (ioctl(exit_listener, SECCOMP_IOCTL_NOTIF_RECV, &req) != 0) {
			/* ENOENT: the caller died before its notification was read. */
			if (errno == EINTR || errno == ENOENT) continue;
			return NULL;
		}
		if (owns_thread(req.pid)) cudaprof_cupti_finalize("exit_group", 0);
		else finalize_descendant(req.pid);
		struct seccomp_notif_resp resp;
		memset(&resp, 0, sizeof(resp));
		resp.id = req.id;
		resp.flags = SECCOMP_USER_NOTIF_FLAG_CONTINUE;
		/* ENOENT again means the caller is already gone; nothing to answer. */
		(void)ioctl(exit_listener, SECCOMP_IOCTL_NOTIF_SEND, &resp);
	}
}

/* install_exit_group returns NULL on success, or the step that failed with
   errno set. */
static const char *install_exit_group(void) {
	struct sock_filter code[] = {
		BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, arch)),
		BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, EXIT_GROUP_AUDIT_ARCH, 0, 2),
		BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, nr)),
		BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_exit_group, 1, 0),
		BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
		BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_USER_NOTIF),
	};
	struct sock_fprog prog = {.len = sizeof(code) / sizeof(code[0]), .filter = code};
	if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0) return "prctl(PR_SET_NO_NEW_PRIVS)";
	/* TSYNC because a CUDA_INJECTION64_PATH load runs this constructor inside
	   cuInit, on whichever thread got there, with the rest of the target's
	   threads already running. TSYNC_ESRCH lets the call return the listener
	   rather than the id of a thread it could not synchronize. */
	long fd = syscall(SYS_seccomp, SECCOMP_SET_MODE_FILTER, SECCOMP_FILTER_FLAG_NEW_LISTENER | SECCOMP_FILTER_FLAG_TSYNC | SECCOMP_FILTER_FLAG_TSYNC_ESRCH, &prog);
	/* EBUSY: an ancestor's filter already has the listener, and its watcher
	   will serve this process's exit_group through the rendezvous. */
	if (fd < 0 && errno == EBUSY) return install_exit_rendezvous();
	if (fd < 0) return "seccomp(SECCOMP_FILTER_FLAG_NEW_LISTENER)";
	exit_listener = (int)fd;
	/* The filter is installed and cannot be removed. Without a watcher every
	   exit_group would wait on a listener nobody reads, so closing it is the
	   fallback: the kernel then fails those calls with ENOSYS, and a libc exit
	   falls back to exit. */
	int rc = pthread_create(&exit_watcher, NULL, exit_watcher_main, NULL);
	if (rc != 0) {
		(void)close(exit_listener);
		exit_listener = -1;
		errno = rc;
		return "pthread_create(exit watcher)";
	}
	return NULL;
}
#else
static const char *install_exit_group(void) { errno = ENOSYS; return "unsupported platform"; }
#endif

/* fork(2) copies only the calling thread, so a child inherits the producer's
   initialized state, its descriptors and its signal disposition, but none of
   its threads. Every remaining entry point is harmful there. Finalizing joins a
   pthread_t that names no thread in the child and hangs. A record it wrote would
   interleave with the parent's into one journal and break the decoder's
   per-stream sequence. A CUPTI call would reach a subsystem the child does not
   own. Worst, the inherited SIGTERM handler writes the parent's self-pipe: the
   parent's terminator wakes with its own term_signalled still clear and returns,
   so the parent loses its termination service for the rest of its life. The
   child therefore disowns the agent outright rather than merely declining to
   finalize.

   This is inertness for fork-without-exec, not fork capture: such a child is
   not profiled. A child that execs runs the constructor again and gets a real
   agent of its own. */
static void cudaprof_after_fork_child(void) {
	/* Close every entry point first, so a signal or callback arriving during
	   the rest of this function already finds the agent shut. shutdown_started
	   is what buffer_requested and the external wrappers test. */
	atomic_store(&shutdown_started, 1);
	atomic_store(&callbacks_closed, 1);
	atomic_store(&finalize_started, 1);
	atomic_store(&finalize_done, 1);
	external_enabled = 0;
	marker_enabled = 0;
	callback_registered = 0;
	flush_started = 0;
	initialized = 0;
	/* fork(2) copies only the calling thread, so the sampler does not exist
	   here. Clearing this keeps any later finalize from joining a pthread_t
	   that names no thread in the child. */
	host_sample_running = 0;

	/* Give the child back the disposition the target had before the producer
	   installed its handler, so the child terminates on the target's own terms
	   rather than through a terminator that does not exist here. prev_sigterm
	   is SIG_DFL exactly when the producer installed over a default.

	   Restore only if the producer's handler is still the one installed. A
	   target may replace it after load -- CPython does -- and that handler is
	   the application's, not the producer's to undo; overwriting it in the
	   child would change how the child terminates, which is the one thing the
	   producer promises never to do. In that case the child already has no
	   inherited path into the parent's pipe, so there is nothing to restore. */
	if (sigterm_installed) {
#ifdef CUDAPROF_TEST_FORCE_UNCONDITIONAL_SIGTERM_RESTORE
		(void)sigaction(SIGTERM, &prev_sigterm, NULL);
#else
		struct sigaction current;
		if (sigaction(SIGTERM, NULL, &current) == 0 && current.sa_sigaction == cudaprof_sigterm_handler) {
			(void)sigaction(SIGTERM, &prev_sigterm, NULL);
		}
#endif
		sigterm_installed = 0;
	}

	/* Close these, do not merely forget them: they are the parent's. The write
	   end matters most -- while it stays open here, a signal in the child can
	   still reach the parent's terminator. O_CLOEXEC covers a later exec, not a
	   child that lives on without one. */
	if (term_pipe[1] >= 0) { (void)close(term_pipe[1]); term_pipe[1] = -1; }
	if (term_pipe[0] >= 0) { (void)close(term_pipe[0]); term_pipe[0] = -1; }
	if (journal_fd >= 0) { (void)close(journal_fd); journal_fd = -1; }
	/* The parent's watcher still serves this child's exit_group. A child that
	   kept the listener open would hold it open past the parent, and its own
	   exit_group would then wait on it forever. */
	if (exit_listener >= 0) { (void)close(exit_listener); exit_listener = -1; }
	/* The rendezvous names the parent's pid, and its thread stayed there. */
	if (exit_rendezvous >= 0) { (void)close(exit_rendezvous); exit_rendezvous = -1; }
	worker_dir[0] = '\0';
}

/* set_activity_collection turns every kind this agent owns on or off in one
   step and reports how many CUPTI refused. The set is taken from the *_enabled
   flags rather than from a fixed list so a kind that was never enabled -- an
   optional one like MARKER, or one whose enable failed -- is never disabled,
   and so this stays correct when a kind is added above.

   Teardown and capture-range gating share it because they want exactly the
   same operation; keeping one copy is what stops a kind from being added to
   the range path and forgotten in the teardown path. */
static int set_activity_collection(int on) {
	static const CUpti_ActivityKind owned[] = {
		CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
		CUPTI_ACTIVITY_KIND_KERNEL,
		CUPTI_ACTIVITY_KIND_MEMCPY,
		CUPTI_ACTIVITY_KIND_MEMSET,
		CUPTI_ACTIVITY_KIND_OVERHEAD,
		CUPTI_ACTIVITY_KIND_GRAPH_TRACE,
		CUPTI_ACTIVITY_KIND_SYNCHRONIZATION,
		CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION,
		CUPTI_ACTIVITY_KIND_MARKER,
		CUPTI_ACTIVITY_KIND_RUNTIME,
		CUPTI_ACTIVITY_KIND_DRIVER,
		CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER,
	};
	const int *flags[] = {
		&concurrent_enabled, &legacy_enabled, &memcpy_enabled, &memset_enabled,
		&overhead_enabled, &graph_trace_enabled, &sync_enabled, &external_enabled,
		&marker_enabled, &runtime_enabled, &driver_enabled, &um_enabled,
	};
	int failures = 0;
	for (size_t i = 0; i < sizeof(owned) / sizeof(owned[0]); i++) {
		if (!*flags[i]) continue;
		CUptiResult r = on ? cuptiActivityEnable(owned[i]) : cuptiActivityDisable(owned[i]);
		if (r != CUPTI_SUCCESS) failures++;
	}
	atomic_store(&collecting, on ? 1 : 0);
	return failures;
}

/* Records a capture-range edge. The consumer needs these to state what the
   journal covers: without them an empty journal from a target that never
   called cudaProfilerStart is indistinguishable from one whose work produced
   no activity, which is the "complete means nothing dropped" trap this
   producer already has elsewhere. */
static void write_range_edge(const char *edge, const char *api, unsigned index, int failures, uint64_t ts, uint64_t flush_ns) {
	char payload[MAX_PAYLOAD];
	int n = snprintf(payload, sizeof(payload),
		"{\"kind\":\"activity\",\"subtype\":\"capture_range\",\"edge\":\"%s\",\"api\":\"%s\",\"index\":%u,\"toggle_failures\":%d,\"flush_nanos\":%llu}",
		edge, api, index, failures, (unsigned long long)flush_ns);
	if (n > 0 && (size_t)n < sizeof(payload)) (void)write_record(KIND_ACTIVITY, ts, payload, (size_t)n);
}

/* track_module_callback follows the driver's own words about where a kernel
   came from and reports whether cbid was one of them. Nothing here calls into
   CUDA: every value it needs is already in the callback's parameter block, so
   the callback stays a read of memory the driver handed over.

   Three chains reach a launch. The module API binds a CUmodule at load and a
   CUfunction at cuModuleGetFunction. The library API binds a CUlibrary at
   load, a CUkernel at cuLibraryGetKernel and a CUfunction at
   cuKernelGetFunction. The runtime launches by host stub and exposes no
   handle at all, so a stub is bound only when the module it needs is loaded
   inside its launch -- the first launch from each lazily loaded module. A
   module loaded eagerly, before the stub's first launch, leaves that stub's
   kernels with no module id, which the reader then attributes by symbol. */
static int track_module_callback(CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const CUpti_CallbackData *data) {
	if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
		const void *func;
		switch (cbid) {
		case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000:
		case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000:
		case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_v9000:
		case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_ptsz_v9000:
			/* The host stub is the first parameter of all four. */
			if (data->functionParams == NULL) return 1;
			func = ((const cudaLaunchKernel_v7000_params *)data->functionParams)->func;
			break;
		case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060:
		case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_ptsz_v11060:
			if (data->functionParams == NULL) return 1;
			func = ((const cudaLaunchKernelExC_v11060_params *)data->functionParams)->func;
			break;
		default:
			return 0;
		}
		if (data->callbackSite == CUPTI_API_ENTER) {
			loading_module_id = 0;
			note_launch_module(func, data->correlationId);
			return 1;
		}
		uint32_t loaded = take_loaded_module();
		if (loaded != 0 && function_module(func) == 0) {
			bind_function_handle(func, NULL, loaded);
			note_launch_module(func, data->correlationId);
		}
		return 1;
	}
	if (domain != CUPTI_CB_DOMAIN_DRIVER_API) return 0;
	switch (cbid) {
	case CUPTI_DRIVER_TRACE_CBID_cuGraphAddKernelNode: {
		if (data->callbackSite != CUPTI_API_EXIT || data->functionParams == NULL || data->functionReturnValue == NULL || *(CUresult *)data->functionReturnValue != CUDA_SUCCESS) return 1;
		const cuGraphAddKernelNode_params *p = data->functionParams;
		if (p->phGraphNode != NULL && p->nodeParams != NULL) note_graph_function(*p->phGraphNode, p->nodeParams->func);
		return 1;
	}
	case CUPTI_DRIVER_TRACE_CBID_cuGraphAddKernelNode_v2: {
		if (data->callbackSite != CUPTI_API_EXIT || data->functionParams == NULL || data->functionReturnValue == NULL || *(CUresult *)data->functionReturnValue != CUDA_SUCCESS) return 1;
		const cuGraphAddKernelNode_v2_params *p = data->functionParams;
		if (p->phGraphNode != NULL && p->nodeParams != NULL) note_graph_function(*p->phGraphNode, p->nodeParams->func);
		return 1;
	}
	case CUPTI_DRIVER_TRACE_CBID_cuGraphKernelNodeSetParams:
	case CUPTI_DRIVER_TRACE_CBID_cuGraphKernelNodeSetParams_v2:
	case CUPTI_DRIVER_TRACE_CBID_cuGraphExecKernelNodeSetParams:
	case CUPTI_DRIVER_TRACE_CBID_cuGraphExecKernelNodeSetParams_v2:
	case CUPTI_DRIVER_TRACE_CBID_cuGraphNodeSetParams:
	case CUPTI_DRIVER_TRACE_CBID_cuGraphExecNodeSetParams:
	case CUPTI_DRIVER_TRACE_CBID_cuGraphExecChildGraphNodeSetParams:
	case CUPTI_DRIVER_TRACE_CBID_cuGraphExecUpdate:
	case CUPTI_DRIVER_TRACE_CBID_cuGraphExecUpdate_v2:
		pthread_mutex_lock(&handle_mu);
		graph_module_count = 0;
		pthread_mutex_unlock(&handle_mu);
		return 1;
	case CUPTI_DRIVER_TRACE_CBID_cuModuleLoad:
	case CUPTI_DRIVER_TRACE_CBID_cuModuleLoadData:
	case CUPTI_DRIVER_TRACE_CBID_cuModuleLoadDataEx:
	case CUPTI_DRIVER_TRACE_CBID_cuModuleLoadFatBinary: {
		if (data->callbackSite == CUPTI_API_ENTER) { loading_module_id = 0; return 1; }
		uint32_t loaded = take_loaded_module();
		if (data->functionParams == NULL) return 1;
		/* Every load takes CUmodule* first, so one layout reads all four. */
		const cuModuleLoad_params *p = (const cuModuleLoad_params *)data->functionParams;
		if (p->module != NULL) bind_module_handle(*p->module, loaded);
		return 1;
	}
	case CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadData:
	case CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadFromFile: {
		/* A library loaded lazily announces no module here; its kernels
		   are bound when a later call loads it. */
		if (data->callbackSite == CUPTI_API_ENTER) { loading_module_id = 0; return 1; }
		uint32_t loaded = take_loaded_module();
		if (data->functionParams == NULL) return 1;
		/* Both take CUlibrary* first. */
		const cuLibraryLoadData_params *p = (const cuLibraryLoadData_params *)data->functionParams;
		if (p->library != NULL) bind_module_handle(*p->library, loaded);
		return 1;
	}
	case CUPTI_DRIVER_TRACE_CBID_cuModuleGetFunction: {
		if (data->callbackSite == CUPTI_API_ENTER) { loading_module_id = 0; return 1; }
		uint32_t loaded = take_loaded_module();
		if (data->functionParams == NULL) return 1;
		const cuModuleGetFunction_params *p = (const cuModuleGetFunction_params *)data->functionParams;
		if (p->hfunc != NULL) bind_function_handle(*p->hfunc, p->hmod, loaded);
		return 1;
	}
	case CUPTI_DRIVER_TRACE_CBID_cuLibraryGetKernel: {
		if (data->callbackSite == CUPTI_API_ENTER) { loading_module_id = 0; return 1; }
		uint32_t loaded = take_loaded_module();
		if (data->functionParams == NULL) return 1;
		const cuLibraryGetKernel_params *p = (const cuLibraryGetKernel_params *)data->functionParams;
		if (p->pKernel != NULL) bind_function_handle(*p->pKernel, p->library, loaded);
		return 1;
	}
	case CUPTI_DRIVER_TRACE_CBID_cuKernelGetFunction: {
		if (data->callbackSite == CUPTI_API_ENTER) { loading_module_id = 0; return 1; }
		uint32_t loaded = take_loaded_module();
		if (data->functionParams == NULL) return 1;
		const cuKernelGetFunction_params *p = (const cuKernelGetFunction_params *)data->functionParams;
		/* The kernel's module is known from cuLibraryGetKernel, or from
		   the load this call just caused; either binds both handles. */
		if (loaded == 0) loaded = function_module(p->kernel);
		else bind_function_handle(p->kernel, NULL, loaded);
		if (p->pFunc != NULL) bind_function_handle(*p->pFunc, NULL, loaded);
		return 1;
	}
	case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel:
	case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz:
	case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel:
	case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz:
	case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx:
	case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz: {
		if (data->functionParams == NULL) return 1;
		/* CUfunction is the first parameter of the first four and the
		   second of the Ex pair. It may be a CUkernel. */
		const void *f = cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx || cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz
			? (const void *)((const cuLaunchKernelEx_params *)data->functionParams)->f
			: (const void *)((const cuLaunchKernel_params *)data->functionParams)->f;
		if (data->callbackSite == CUPTI_API_ENTER) {
			loading_module_id = 0;
			note_launch_module(f, data->correlationId);
			return 1;
		}
		/* A CUkernel launched before any cuKernelGetFunction loads its
		   module here, inside the launch. */
		uint32_t loaded = take_loaded_module();
		if (loaded != 0 && function_module(f) == 0) {
			bind_function_handle(f, NULL, loaded);
			note_launch_module(f, data->correlationId);
		}
		return 1;
	}
	default:
		return 0;
	}
}

/* PC sampling. The activity records above say how long a kernel ran; nothing
   in them says which of its instructions the warps were sitting on. CUPTI's
   continuous PC sampler does, and it is off unless CUDAPROF_PCSAMPLING asks
   for it, because it is not free: measured on a GB10 it costs a roughly
   constant 25 to 27 microseconds of host drain per launch, 3% of a kernel
   that runs for 0.8 ms and triple the cost of one that runs for 14
   microseconds.

   A sample arrives as a cubin CRC, a function name and a function-relative
   byte offset. The CRC is what joins it to a stored module: cuptiGetCubinCrc
   over the image the module-loaded callback hands us computes the same number
   CUPTI stamps on the samples, so the two meet without matching names, which
   two builds of one graph would make ambiguous -- the same reason the kernel
   records above follow handles rather than names.

   Samples fold into a fixed table keyed by module, function and offset, and
   are written once at finalize. A record per drain would be one record per PC
   per wakeup for the whole run, all of them saying the same thing about the
   same instruction. */

#define PCSAMPLE_NAME_BYTES 256
#define PCSAMPLE_SLOTS 8192
#define MODULE_CRC_SLOTS 256

static int pcsampling_requested;
static int pcsampling_running;
static uint32_t pcsampling_period = 10;
static uint32_t pcsampling_sleep_ms = 10;
static uint64_t pcsampling_scratch_bytes = 4ull << 20;
static uint64_t pcsampling_hardware_bytes = 16ull << 20;
static uint64_t pcsampling_hardware_used;
static size_t pcsampling_pcs = 1024;
static CUcontext pcsampling_context;
static CUpti_PCSamplingData *pcsampling_output;
static CUpti_PCSamplingData *pcsampling_drain;
static uint32_t *pcsampling_reasons;
static size_t pcsampling_reason_count;
static pthread_t pcsampling_thread;
static int pcsampling_thread_started;
static _Atomic int pcsampling_stop;
static _Atomic uint64_t pcsampling_folded;
static _Atomic uint64_t pcsampling_total;
static _Atomic uint64_t pcsampling_hw_dropped;
static _Atomic uint64_t pcsampling_unkeyed;
static _Atomic uint64_t pcsampling_overflows;
static pthread_mutex_t pcsampling_mu = PTHREAD_MUTEX_INITIALIZER;
static struct pc_row {
	char name[PCSAMPLE_NAME_BYTES];
	uint64_t offset;
	uint64_t samples;
	uint32_t module_id;
} pc_table[PCSAMPLE_SLOTS];
static unsigned pc_table_used;
static struct { uint64_t crc; uint32_t id; } module_crcs[MODULE_CRC_SLOTS];
static unsigned module_crc_count;

/* bind_module_crc records the CRC CUPTI will stamp on this module's samples
   against the number the agent stored the image under, so a sample can name
   the file a reader can disassemble. A module whose CRC will not compute
   still samples; its PCs name module 0, which reads downstream as kernel work
   belonging to no stored image. */
static void bind_module_crc(const void *cubin, size_t size, uint32_t id) {
	if (!pcsampling_requested || cubin == NULL || size == 0) return;
	CUpti_GetCubinCrcParams p;
	memset(&p, 0, sizeof(p));
	p.size = CUpti_GetCubinCrcParamsSize;
	p.cubinSize = size;
	p.cubin = cubin;
	CUptiResult r = cuptiGetCubinCrc(&p);
	if (r != CUPTI_SUCCESS) { record_cupti_failure("cuptiGetCubinCrc", r); return; }
	pthread_mutex_lock(&pcsampling_mu);
	for (unsigned i = 0; i < module_crc_count; i++) {
		if (module_crcs[i].crc == p.cubinCrc) { pthread_mutex_unlock(&pcsampling_mu); return; }
	}
	if (module_crc_count < MODULE_CRC_SLOTS) {
		module_crcs[module_crc_count].crc = p.cubinCrc;
		module_crcs[module_crc_count].id = id;
		module_crc_count++;
	}
	pthread_mutex_unlock(&pcsampling_mu);
}

/* pcsampling_fold adds one PC's samples to the table. A name too long for a
   row is counted as unkeyed rather than truncated: a truncated symbol reads
   as a different symbol, and two kernels sharing a prefix would silently
   merge, which is the rule the kernel records follow too. */
static void pcsampling_fold(uint64_t crc, const char *name, uint64_t offset, uint64_t samples) {
	if (samples == 0 || name == NULL || *name == '\0') return;
	if (strnlen(name, PCSAMPLE_NAME_BYTES) >= PCSAMPLE_NAME_BYTES) {
		atomic_fetch_add(&pcsampling_unkeyed, samples);
		return;
	}
	uint64_t h = 1469598103934665603ull;
	for (const unsigned char *p = (const unsigned char *)name; *p; p++) { h ^= *p; h *= 1099511628211ull; }
	h ^= offset; h *= 1099511628211ull;
	h ^= crc; h *= 1099511628211ull;
	pthread_mutex_lock(&pcsampling_mu);
	uint32_t id = 0;
	for (unsigned i = 0; i < module_crc_count; i++) {
		if (module_crcs[i].crc == crc) { id = module_crcs[i].id; break; }
	}
	for (unsigned probe = 0; probe < PCSAMPLE_SLOTS; probe++) {
		struct pc_row *row = &pc_table[(h + probe) % PCSAMPLE_SLOTS];
		if (row->name[0] == '\0') {
			memcpy(row->name, name, strlen(name) + 1);
			row->offset = offset;
			row->module_id = id;
			row->samples = samples;
			pc_table_used++;
			pthread_mutex_unlock(&pcsampling_mu);
			return;
		}
		if (row->offset == offset && row->module_id == id && strcmp(row->name, name) == 0) {
			row->samples += samples;
			pthread_mutex_unlock(&pcsampling_mu);
			return;
		}
	}
	pthread_mutex_unlock(&pcsampling_mu);
	atomic_fetch_add(&pcsampling_unkeyed, samples);
}

/* pcsampling_collect folds a filled buffer and empties it for reuse. Only the
   selected reasons are summed, and the selection leaves out the _not_issued
   companions, each of which counts a subset of the cycles its base reason
   already counted: including them doubles the total against CUPTI's own. */
static void pcsampling_collect(CUpti_PCSamplingData *d) {
	if (d == NULL) return;
	atomic_fetch_add(&pcsampling_total, d->totalSamples);
	atomic_fetch_add(&pcsampling_hw_dropped, d->droppedSamples);
	size_t n = d->totalNumPcs < d->collectNumPcs ? d->totalNumPcs : d->collectNumPcs;
	for (size_t i = 0; i < n; i++) {
		CUpti_PCSamplingPCData *pc = &d->pPcData[i];
		uint64_t samples = 0;
		for (size_t j = 0; j < pc->stallReasonCount; j++) samples += pc->stallReason[j].samples;
		pcsampling_fold(pc->cubinCrc, pc->functionName, pc->pcOffset, samples);
		atomic_fetch_add(&pcsampling_folded, samples);
	}
	d->totalNumPcs = 0;
	d->totalSamples = 0;
	d->droppedSamples = 0;
	d->remainingNumPcs = 0;
}

/* pcsampling_drain_once reads one buffer's worth and reports whether CUPTI
   still holds PCs, so a caller can drain to empty. A full hardware buffer is
   counted rather than failed: the samples in it are lost, the ones after it
   are not, and the count is what tells a reader the period was too short for
   the run. */
static int pcsampling_drain_once(void) {
	if (!pcsampling_running || pcsampling_drain == NULL) return 0;
	CUpti_PCSamplingGetDataParams p;
	memset(&p, 0, sizeof(p));
	p.size = CUpti_PCSamplingGetDataParamsSize;
	p.ctx = pcsampling_context;
	p.pcSamplingData = pcsampling_drain;
	CUptiResult r = cuptiPCSamplingGetData(&p);
	if (r == CUPTI_ERROR_OUT_OF_MEMORY) {
		atomic_fetch_add(&pcsampling_overflows, 1);
	} else if (r != CUPTI_SUCCESS) {
		record_cupti_failure("cuptiPCSamplingGetData", r);
		return 0;
	}
	int more = pcsampling_drain->remainingNumPcs > 0;
	pcsampling_collect(pcsampling_drain);
	return more;
}

/* The drain thread. CUPTI parses the hardware buffer on its own workers and
   holds the result until it is read; nothing on the target's threads reads
   it, so this one wakes on a fixed span and empties whatever is there.
   Finalize joins it before it touches CUPTI itself. */
static void *pcsampling_worker_main(void *arg) {
	(void)arg;
	while (!atomic_load(&pcsampling_stop)) {
		struct timespec ts = {pcsampling_sleep_ms / 1000, (long)(pcsampling_sleep_ms % 1000) * 1000000L};
		nanosleep(&ts, NULL);
		for (int i = 0; i < 256 && !atomic_load(&pcsampling_stop); i++) {
			if (!pcsampling_drain_once()) break;
		}
	}
	return NULL;
}

/* pcsampling_alloc_buffer builds one parsed-data buffer: the header, the PC
   array, and one stall-reason array per PC. CUPTI writes into all three. */
static CUpti_PCSamplingData *pcsampling_alloc_buffer(size_t pcs, size_t reasons) {
	CUpti_PCSamplingData *d = calloc(1, sizeof(*d));
	if (d == NULL) return NULL;
	d->size = sizeof(CUpti_PCSamplingData);
	d->collectNumPcs = pcs;
	d->pPcData = calloc(pcs, sizeof(CUpti_PCSamplingPCData));
	if (d->pPcData == NULL) { free(d); return NULL; }
	for (size_t i = 0; i < pcs; i++) {
		d->pPcData[i].size = sizeof(CUpti_PCSamplingPCData);
		d->pPcData[i].stallReason = calloc(reasons, sizeof(CUpti_PCSamplingStallReason));
		if (d->pPcData[i].stallReason == NULL) return NULL;
	}
	return d;
}

/* pcsampling_apply writes the whole configuration for one hardware buffer
   size. It is a function because that size is retried: CUPTI allocates the
   hardware buffer on the device, and a device with a resident model on it
   refuses the default. */
static CUptiResult pcsampling_apply(CUcontext ctx, uint64_t hardware_bytes) {
	CUpti_PCSamplingConfigurationInfo info[9];
	memset(info, 0, sizeof(info));
	info[0].attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_PERIOD;
	info[0].attributeData.samplingPeriodData.samplingPeriod = pcsampling_period;
	info[1].attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_STALL_REASON;
	info[1].attributeData.stallReasonData.stallReasonCount = pcsampling_reason_count;
	info[1].attributeData.stallReasonData.pStallReasonIndex = pcsampling_reasons;
	info[2].attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SCRATCH_BUFFER_SIZE;
	info[2].attributeData.scratchBufferSizeData.scratchBufferSize = pcsampling_scratch_bytes;
	info[3].attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_HARDWARE_BUFFER_SIZE;
	info[3].attributeData.hardwareBufferSizeData.hardwareBufferSize = hardware_bytes;
	info[4].attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_COLLECTION_MODE;
	info[4].attributeData.collectionModeData.collectionMode = CUPTI_PC_SAMPLING_COLLECTION_MODE_CONTINUOUS;
	info[5].attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_ENABLE_START_STOP_CONTROL;
	info[5].attributeData.enableStartStopControlData.enableStartStopControl = 1;
	info[6].attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_OUTPUT_DATA_FORMAT;
	info[6].attributeData.outputDataFormatData.outputDataFormat = CUPTI_PC_SAMPLING_OUTPUT_DATA_FORMAT_PARSED;
	info[7].attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_DATA_BUFFER;
	info[7].attributeData.samplingDataBufferData.samplingDataBuffer = pcsampling_output;
	info[8].attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_WORKER_THREAD_PERIODIC_SLEEP_SPAN;
	info[8].attributeData.workerThreadPeriodicSleepSpanData.workerThreadPeriodicSleepSpan = pcsampling_sleep_ms;
	CUpti_PCSamplingConfigurationInfoParams params;
	memset(&params, 0, sizeof(params));
	params.size = CUpti_PCSamplingConfigurationInfoParamsSize;
	params.ctx = ctx;
	params.numAttributes = sizeof(info) / sizeof(info[0]);
	params.pPCSamplingConfigurationInfo = info;
	return cuptiPCSamplingSetConfigurationAttribute(&params);
}

/* pcsampling_begin configures and starts sampling on the first context the
   target creates. A second context is left alone and recorded: CUPTI
   configures per context, and sampling two would need two buffers, two drains
   and a way for a record to say which context a PC came from, none of which
   the record shape carries.

   The reason selection drops the _not_issued companions, so the buffer holds
   what pcsampling_collect is allowed to sum. */
static void pcsampling_begin(CUcontext ctx) {
	if (!pcsampling_requested || ctx == NULL) return;
	if (pcsampling_running || pcsampling_context != NULL) {
		if (ctx != pcsampling_context) record_cupti_failure("pc sampling covers the first context only", CUPTI_ERROR_NOT_SUPPORTED);
		return;
	}
	pcsampling_context = ctx;
	CUpti_PCSamplingEnableParams enable;
	memset(&enable, 0, sizeof(enable));
	enable.size = CUpti_PCSamplingEnableParamsSize;
	enable.ctx = ctx;
	CUptiResult r = cuptiPCSamplingEnable(&enable);
	if (r != CUPTI_SUCCESS) { record_cupti_failure("cuptiPCSamplingEnable", r); return; }

	size_t count = 0;
	CUpti_PCSamplingGetNumStallReasonsParams num;
	memset(&num, 0, sizeof(num));
	num.size = CUpti_PCSamplingGetNumStallReasonsParamsSize;
	num.ctx = ctx;
	num.numStallReasons = &count;
	r = cuptiPCSamplingGetNumStallReasons(&num);
	if (r != CUPTI_SUCCESS || count == 0) { record_cupti_failure("cuptiPCSamplingGetNumStallReasons", r); return; }

	uint32_t *indexes = calloc(count, sizeof(*indexes));
	char **names = calloc(count, sizeof(*names));
	if (indexes == NULL || names == NULL) { free(indexes); free(names); return; }
	for (size_t i = 0; i < count; i++) {
		names[i] = calloc(CUPTI_STALL_REASON_STRING_SIZE, 1);
		if (names[i] == NULL) return;
	}
	CUpti_PCSamplingGetStallReasonsParams reasons;
	memset(&reasons, 0, sizeof(reasons));
	reasons.size = CUpti_PCSamplingGetStallReasonsParamsSize;
	reasons.ctx = ctx;
	reasons.numStallReasons = count;
	reasons.stallReasonIndex = indexes;
	reasons.stallReasons = names;
	r = cuptiPCSamplingGetStallReasons(&reasons);
	if (r != CUPTI_SUCCESS) { record_cupti_failure("cuptiPCSamplingGetStallReasons", r); return; }
	pcsampling_reasons = calloc(count, sizeof(*pcsampling_reasons));
	if (pcsampling_reasons == NULL) return;
	for (size_t i = 0; i < count; i++) {
		const char *companion = "_not_issued";
		size_t n = strlen(names[i]), c = strlen(companion);
		if (n > c && strcmp(names[i] + n - c, companion) == 0) continue;
		pcsampling_reasons[pcsampling_reason_count++] = indexes[i];
	}
	for (size_t i = 0; i < count; i++) free(names[i]);
	free(names);
	free(indexes);
	if (pcsampling_reason_count == 0) return;

	pcsampling_output = pcsampling_alloc_buffer(pcsampling_pcs, pcsampling_reason_count);
	pcsampling_drain = pcsampling_alloc_buffer(pcsampling_pcs, pcsampling_reason_count);
	if (pcsampling_output == NULL || pcsampling_drain == NULL) return;

	uint64_t bytes = pcsampling_hardware_bytes;
	for (;;) {
		r = pcsampling_apply(ctx, bytes);
		if (r != CUPTI_ERROR_OUT_OF_MEMORY || bytes <= (1ull << 20)) break;
		record_cupti_failure("cuptiPCSamplingSetConfigurationAttribute(hardware buffer)", r);
		bytes /= 2;
	}
	if (r != CUPTI_SUCCESS) { record_cupti_failure("cuptiPCSamplingSetConfigurationAttribute", r); return; }
	pcsampling_hardware_used = bytes;

	CUpti_PCSamplingStartParams start;
	memset(&start, 0, sizeof(start));
	start.size = CUpti_PCSamplingStartParamsSize;
	start.ctx = ctx;
	r = cuptiPCSamplingStart(&start);
	if (r != CUPTI_SUCCESS) { record_cupti_failure("cuptiPCSamplingStart", r); return; }
	pcsampling_running = 1;
	if (pthread_create(&pcsampling_thread, NULL, pcsampling_worker_main, NULL) == 0) pcsampling_thread_started = 1;
}

static void stop_pcsampling_worker(void) {
	if (!pcsampling_thread_started) return;
	atomic_store(&pcsampling_stop, 1);
	if (pthread_join(pcsampling_thread, NULL) != 0) atomic_store(&producer_failed, 1);
	pcsampling_thread_started = 0;
}

/* flush_pcsampling ends collection and writes the table. Stop closes the
   range, the drain loop takes what CUPTI has parsed, and Disable copies what
   is left into the configured buffer -- which is why that buffer is collected
   after the disable and not before. */
static void flush_pcsampling(void) {
	if (!pcsampling_running) return;
	CUpti_PCSamplingStopParams stop;
	memset(&stop, 0, sizeof(stop));
	stop.size = CUpti_PCSamplingStopParamsSize;
	stop.ctx = pcsampling_context;
	CUptiResult r = cuptiPCSamplingStop(&stop);
	if (r != CUPTI_SUCCESS) record_cupti_failure("cuptiPCSamplingStop", r);
	for (int i = 0; i < 4096; i++) {
		if (!pcsampling_drain_once()) break;
	}
	CUpti_PCSamplingDisableParams disable;
	memset(&disable, 0, sizeof(disable));
	disable.size = CUpti_PCSamplingDisableParamsSize;
	disable.ctx = pcsampling_context;
	r = cuptiPCSamplingDisable(&disable);
	if (r != CUPTI_SUCCESS) record_cupti_failure("cuptiPCSamplingDisable", r);
	pcsampling_running = 0;
	pcsampling_collect(pcsampling_output);

	unsigned written = 0;
	for (unsigned i = 0; i < PCSAMPLE_SLOTS; i++) {
		struct pc_row *row = &pc_table[i];
		if (row->name[0] == '\0' || row->samples == 0) continue;
		char payload[MAX_PAYLOAD];
		int n = snprintf(payload, sizeof(payload), "{\"kind\":\"pcsample\",\"module_id\":%" PRIu32 ",\"pc_offset\":%" PRIu64 ",\"pc_samples\":%" PRIu64 ",\"pc_function\":\"", row->module_id, row->offset, row->samples);
		if (n < 0 || (size_t)n >= sizeof(payload)) continue;
		if ((n = append_escaped(payload, sizeof(payload), n, row->name, "pc sample function")) < 0) continue;
		if ((size_t)n + 2 >= sizeof(payload)) continue;
		payload[n++] = '"';
		payload[n++] = '}';
		if (queue_pending(KIND_PCSAMPLE, now_ns(), payload, (size_t)n) == 0) written++;
	}
	char payload[MAX_PAYLOAD];
	int n = snprintf(payload, sizeof(payload),
		"{\"kind\":\"pcsampling\",\"pc_period\":%" PRIu32 ",\"pc_hardware_bytes\":%" PRIu64 ",\"pc_distinct\":%u,\"pc_written\":%u,\"pc_folded\":%" PRIu64 ",\"pc_total\":%" PRIu64 ",\"pc_hw_dropped\":%" PRIu64 ",\"pc_unkeyed\":%" PRIu64 ",\"pc_overflows\":%" PRIu64 "}",
		pcsampling_period, pcsampling_hardware_used, pc_table_used, written,
		atomic_load(&pcsampling_folded), atomic_load(&pcsampling_total), atomic_load(&pcsampling_hw_dropped),
		atomic_load(&pcsampling_unkeyed), atomic_load(&pcsampling_overflows));
	if (n > 0 && (size_t)n < sizeof(payload)) (void)queue_pending(KIND_PCSAMPLE, now_ns(), payload, (size_t)n);
}



#include "cupti_dependency.h"

static void graph_lifecycle_resource(CUpti_CallbackId cbid, const void *cbdata) {
 if (!graph_lifecycle_enabled || !((cbid >= 9 && cbid <= 20) || cbid == 22 || cbid == 23)) return;
 const CUpti_ResourceData *r=cbdata;
 const CUpti_GraphData *g=r->resourceDescriptor;
 if (!g) return;
 uint64_t node_id=0, source_node_id=0;
 if(g->node) (void)cuptiGetGraphNodeId(g->node,&node_id);
 if(g->originalNode) (void)cuptiGetGraphNodeId(g->originalNode,&source_node_id);
 uint64_t ts=now_ns();
 char payload[MAX_PAYLOAD];
 int n=snprintf(payload,sizeof(payload),"{\"kind\":\"graph_lifecycle\",\"timestamp_domain\":\"%s\",\"domain\":\"resource\",\"lifecycle_event\":\"resource\",\"callback_id\":%u,\"graph_handle\":%" PRIu64 ",\"executable_handle\":%" PRIu64 ",\"node_handle\":%" PRIu64 ",\"source_graph_handle\":%" PRIu64 ",\"source_node_handle\":%" PRIu64 ",\"node_id\":%" PRIu64 ",\"source_node_id\":%" PRIu64 ",\"start\":%" PRIu64 ",\"end\":%" PRIu64 "}",clock_state == AGENT_CLOCK_UNIFIED ? "monotonic" : "unknown",(unsigned)cbid,(uint64_t)(uintptr_t)g->graph,(uint64_t)(uintptr_t)g->graphExec,(uint64_t)(uintptr_t)g->node,(uint64_t)(uintptr_t)g->originalGraph,(uint64_t)(uintptr_t)g->originalNode,node_id,source_node_id,ts,ts);
 if (n>0 && (size_t)n<sizeof(payload)) (void)queue_pending(KIND_API,ts,payload,(size_t)n);
}


/* driver_types.h defines cudaGraphExec_t as CUgraphExec_st*, exactly the
   driver's CUgraphExec opaque handle type. Retain that observed handle; do
   not map a membership id or choose a nearby driver launch. */
_Static_assert(_Generic((cudaGraphExec_t)0, CUgraphExec: 1, default: 0), "runtime graph exec handle differs from driver handle");
static void graph_lifecycle_runtime_replay(CUpti_CallbackId cbid, const CUpti_CallbackData *data) {
 if (!graph_lifecycle_enabled || data->callbackSite != CUPTI_API_EXIT) return;
 uint64_t exec=0;
 if (!data->functionParams) return;
 if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_v10000) {
  const cudaGraphLaunch_v10000_params *p=data->functionParams; exec=(uintptr_t)p->graphExec;
 } else if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_ptsz_v10000) {
  const cudaGraphLaunch_ptsz_v10000_params *p=data->functionParams; exec=(uintptr_t)p->graphExec;
 } else return;
 uint64_t ts=now_ns();
 const char *success=data->functionReturnValue ? (*(const int *)data->functionReturnValue == 0 ? "true" : "false") : "null";
 char payload[MAX_PAYLOAD];
 int n=snprintf(payload,sizeof(payload),"{\"kind\":\"graph_lifecycle\",\"timestamp_domain\":\"%s\",\"domain\":\"runtime\",\"lifecycle_event\":\"replay\",\"executable_handle\":%" PRIu64 ",\"correlation\":%u,\"callback_id\":%u,\"context\":%u,\"start\":%" PRIu64 ",\"end\":%" PRIu64 ",\"success\":%s}",clock_state == AGENT_CLOCK_UNIFIED ? "monotonic" : "unknown",exec,data->correlationId,(unsigned)cbid,data->contextUid,ts,ts,success);
 if(n>0 && (size_t)n<sizeof(payload)) (void)queue_pending(KIND_API,ts,payload,(size_t)n);
}

/* Opt-in driver graph handle observations. These do not infer an executable
   from a graph membership id. Unhandled graph APIs are retained explicitly. */
static void graph_lifecycle_callback(CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const CUpti_CallbackData *data) {
 if (!graph_lifecycle_enabled || domain != CUPTI_CB_DOMAIN_DRIVER_API || data->callbackSite != CUPTI_API_EXIT) return;
 const char *name = data->functionName;
 if (name == NULL || (strncmp(name, "cuGraph", 7) != 0 && strncmp(name, "cuStreamEndCapture", 18) != 0)) return;
 const char *event = "unsupported";
 if (strstr(name,"Get") || strstr(name,"Find") || strstr(name,"Debug") || strstr(name,"Upload")) event="observe";
 uint64_t graph = 0, exec = 0, source = 0, node = 0;
 int known = data->functionReturnValue != NULL;
 int success = known && *(const CUresult *)data->functionReturnValue == CUDA_SUCCESS;
 const void *params = data->functionParams;
 if (params != NULL) {
  switch (cbid) {
  case 529: { const cuGraphChildGraphNodeGetGraph_params *p=params; event="child_copy"; node=(uintptr_t)p->hNode; if(success && p->phGraph) graph=(uintptr_t)*p->phGraph; break; }
  case 523: { const cuGraphClone_params *p=params; event="clone"; source=(uintptr_t)p->originalGraph; if(success && p->phGraphClone) graph=(uintptr_t)*p->phGraphClone; break; }
  case 525: { const cuGraphAddChildGraphNode_params *p=params; event="child"; graph=(uintptr_t)p->hGraph; source=(uintptr_t)p->childGraph; if(success && p->phGraphNode) node=(uintptr_t)*p->phGraphNode; break; }
  case CUPTI_DRIVER_TRACE_CBID_cuGraphCreate: { const cuGraphCreate_params *p=params; event="create"; if (success && p->phGraph) graph=(uintptr_t)*p->phGraph; break; }
  case CUPTI_DRIVER_TRACE_CBID_cuStreamEndCapture: { const cuStreamEndCapture_params *p=params; event="create"; if (success && p->phGraph) graph=(uintptr_t)*p->phGraph; break; }
  case CUPTI_DRIVER_TRACE_CBID_cuStreamEndCapture_ptsz: { const cuStreamEndCapture_ptsz_params *p=params; event="create"; if (success && p->phGraph) graph=(uintptr_t)*p->phGraph; break; }
  case CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiate: { const cuGraphInstantiate_params *p=params; event="instantiate"; graph=(uintptr_t)p->hGraph; if (success && p->phGraphExec) exec=(uintptr_t)*p->phGraphExec; break; }
  case CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiate_v2: { const cuGraphInstantiate_v2_params *p=params; event="instantiate"; graph=(uintptr_t)p->hGraph; if (success && p->phGraphExec) exec=(uintptr_t)*p->phGraphExec; break; }
  case CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiateWithFlags: { const cuGraphInstantiateWithFlags_params *p=params; event="instantiate"; graph=(uintptr_t)p->hGraph; if (success && p->phGraphExec) exec=(uintptr_t)*p->phGraphExec; break; }
  case CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiateWithParams: { const cuGraphInstantiateWithParams_params *p=params; event="instantiate"; graph=(uintptr_t)p->hGraph; if (success && p->phGraphExec) exec=(uintptr_t)*p->phGraphExec; break; }
  case CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiateWithParams_ptsz: { const cuGraphInstantiateWithParams_ptsz_params *p=params; event="instantiate"; graph=(uintptr_t)p->hGraph; if (success && p->phGraphExec) exec=(uintptr_t)*p->phGraphExec; break; }
  case CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch: { const cuGraphLaunch_params *p=params; event="replay"; exec=(uintptr_t)p->hGraph; break; }
  case CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz: { const cuGraphLaunch_ptsz_params *p=params; event="replay"; exec=(uintptr_t)p->hGraphExec; break; }
  case CUPTI_DRIVER_TRACE_CBID_cuGraphExecDestroy: { const cuGraphExecDestroy_params *p=params; event="destroy_exec"; exec=(uintptr_t)p->hGraphExec; break; }
  case CUPTI_DRIVER_TRACE_CBID_cuGraphDestroy: { const cuGraphDestroy_params *p=params; event="destroy_graph"; graph=(uintptr_t)p->hGraph; break; }
  case CUPTI_DRIVER_TRACE_CBID_cuGraphExecUpdate: { const cuGraphExecUpdate_params *p=params; event="update"; graph=(uintptr_t)p->hGraph; exec=(uintptr_t)p->hGraphExec; if (success) { if (!p->updateResult_out) known=0; else success=*p->updateResult_out == 0; } break; }
  case CUPTI_DRIVER_TRACE_CBID_cuGraphExecUpdate_v2: { const cuGraphExecUpdate_v2_params *p=params; event="update"; graph=(uintptr_t)p->hGraph; exec=(uintptr_t)p->hGraphExec; if (success) { if (!p->resultInfo) known=0; else success=p->resultInfo->result == 0; } break; }
  case CUPTI_DRIVER_TRACE_CBID_cuGraphExecKernelNodeSetParams: { const cuGraphExecKernelNodeSetParams_params *p=params; event="update_exec"; exec=(uintptr_t)p->hGraphExec; break; }
  case CUPTI_DRIVER_TRACE_CBID_cuGraphExecKernelNodeSetParams_v2: { const cuGraphExecKernelNodeSetParams_v2_params *p=params; event="update_exec"; exec=(uintptr_t)p->hGraphExec; break; }
  case CUPTI_DRIVER_TRACE_CBID_cuGraphAddKernelNode: case CUPTI_DRIVER_TRACE_CBID_cuGraphAddKernelNode_v2: { const cuGraphAddKernelNode_params *p=params; event="update_graph"; graph=(uintptr_t)p->hGraph; break; }
  }
 }
 /* These observation callbacks add timestamped evidence, not an API timing
    estimate. The activity API record supplies the full interval. */
 uint64_t ts=now_ns();
 char payload[MAX_PAYLOAD];
 int n=snprintf(payload,sizeof(payload), "{\"kind\":\"graph_lifecycle\",\"timestamp_domain\":\"%s\",\"domain\":\"driver\",\"lifecycle_event\":\"%s\",\"source_graph_handle\":%" PRIu64 ",\"node_handle\":%" PRIu64 ",\"graph_handle\":%" PRIu64 ",\"executable_handle\":%" PRIu64 ",\"correlation\":%u,\"callback_id\":%u,\"context\":%u,\"start\":%" PRIu64 ",\"end\":%" PRIu64 ",\"success\":%s}",clock_state == AGENT_CLOCK_UNIFIED ? "monotonic" : "unknown",event,source,node,graph,exec,data->correlationId,(unsigned)cbid,data->contextUid,ts,ts,known ? (success ? "true" : "false") : "null");
 if (n>0 && (size_t)n<sizeof(payload)) (void)queue_pending(KIND_API,ts,payload,(size_t)n);
}

static void CUPTIAPI range_callback(void *userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void *cbdata) {
	(void)userdata;
	if (cbdata == NULL || atomic_load(&shutdown_started)) return;
	/* The resource domain passes a different structure than the API domains do,
	   so the domain is tested before the pointer is interpreted. Reading a
	   CUpti_ResourceData as a CUpti_CallbackData is not an error the compiler
	   can see and not one the fields would announce: they are live memory
	   either way. Measured on CUPTI 13.0, the module payload arrives through
	   resourceDescriptor -- reading cbdata itself as the module data yields a
	   plausible-looking module id and a zero size. */
	if (domain == CUPTI_CB_DOMAIN_RESOURCE) {
		if (graph_lifecycle_enabled) {
			atomic_fetch_add(&callback_active,1);
			if (!atomic_load(&shutdown_started)) graph_lifecycle_resource(cbid,cbdata);
			atomic_fetch_sub(&callback_active,1);
		}
		if (cbid == CUPTI_CBID_RESOURCE_CONTEXT_CREATED) {
			/* The first point at which there is a context to configure. PC
			   sampling is per context and the target's first kernel may
			   follow immediately, so it is set up here rather than on a
			   later launch. */
			const CUpti_ResourceData *res = (const CUpti_ResourceData *)cbdata;
			atomic_fetch_add(&callback_active, 1);
			if (!atomic_load(&shutdown_started)) pcsampling_begin(res->context);
			atomic_fetch_sub(&callback_active, 1);
			return;
		}
		/* A module load is often the first callback after the target
		   loads libcublasLt, and so the first chance to find it. */
		if (gemms_requested && !atomic_load(&gemm_hooked)) gemm_probe();
		if (cbid == CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED && modules_enabled) {
			const CUpti_ResourceData *res = cbdata;
			const CUpti_GraphData *g = res->resourceDescriptor;
			if (g == NULL) return;
			uint64_t original = 0;
			if (cuptiGetGraphNodeId(g->originalNode, &original) == CUPTI_SUCCESS) bind_graph_module(g->node, graph_module(original));
			return;
		}
		if (cbid != CUPTI_CBID_RESOURCE_MODULE_LOADED || !modules_enabled) return;
		const CUpti_ResourceData *res = (const CUpti_ResourceData *)cbdata;
		const CUpti_ModuleResourceData *mod = (const CUpti_ModuleResourceData *)res->resourceDescriptor;
		if (mod == NULL) return;
		/* Counted like the range edges below so finalize can wait for it: this
		   writes the journal, and teardown closes that descriptor. */
		atomic_fetch_add(&callback_active, 1);
		if (!atomic_load(&shutdown_started)) {
			write_module(mod->moduleId, mod->pCubin, mod->cubinSize);
			/* While the image is still mapped: the CRC is what a PC sample
			   carries in place of a module id. */
			bind_module_crc(mod->pCubin, mod->cubinSize, mod->moduleId);
		}
		atomic_fetch_sub(&callback_active, 1);
		/* The load that caused this callback is still on the stack; its exit
		   callback below is what turns the id into a handle. */
		loading_module_id = mod->moduleId;
		return;
	}
	if (dependency_enabled && (domain == CUPTI_CB_DOMAIN_DRIVER_API || domain == CUPTI_CB_DOMAIN_RUNTIME_API)) {
		atomic_fetch_add(&callback_active, 1);
		if (!atomic_load(&shutdown_started)) dependency_callback(domain, cbid, cbdata);
		atomic_fetch_sub(&callback_active, 1);
	}
	if (modules_enabled && invalidate_runtime_graph(domain, cbid)) return;
	const CUpti_CallbackData *data = (const CUpti_CallbackData *)cbdata;
	if (memory_policy_enabled) {
		atomic_fetch_add(&callback_active, 1);
		if (!atomic_load(&shutdown_started)) memory_policy_callback(domain, cbid, data);
		atomic_fetch_sub(&callback_active, 1);
	}
	if (graph_lifecycle_enabled) {
		atomic_fetch_add(&callback_active, 1);
		if (!atomic_load(&shutdown_started)) {
			graph_lifecycle_callback(domain, cbid, data);
			if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) graph_lifecycle_runtime_replay(cbid,data);
		}
		atomic_fetch_sub(&callback_active, 1);
	}
	if (crash_depth > 0) crash_note(domain, cbid, data);
	if (gemms_requested && domain == CUPTI_CB_DOMAIN_DRIVER_API) {
		if (!atomic_load(&gemm_hooked)) gemm_probe();
		if (data->callbackSite == CUPTI_API_ENTER && gemm_launch_cbid(cbid)) gemm_launch(data->correlationId);
	}
	if (modules_enabled && track_module_callback(domain, cbid, data)) return;
	int start;
	const char *api;
	if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
		if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaProfilerStart_v4000) { start = 1; api = "cudaProfilerStart"; }
		else if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaProfilerStop_v4000) { start = 0; api = "cudaProfilerStop"; }
		else return;
	} else if (domain == CUPTI_CB_DOMAIN_DRIVER_API) {
		if (cbid == CUPTI_DRIVER_TRACE_CBID_cuProfilerStart) { start = 1; api = "cuProfilerStart"; }
		else if (cbid == CUPTI_DRIVER_TRACE_CBID_cuProfilerStop) { start = 0; api = "cuProfilerStop"; }
		else return;
	} else {
		return;
	}
	/* Open on the way out of Start and close on the way in to Stop, so the
	   range covers the work between the two calls and not the calls
	   themselves. Taking both edges at the same site would put one of the two
	   profiler API calls inside every range. */
	if (data->callbackSite != (start ? CUPTI_API_EXIT : CUPTI_API_ENTER)) return;
	/* Past this point the callback mutates agent state and writes the journal,
	   so finalize has to be able to wait for it. buffer_completed carries the
	   same accounting; range_callback did not, which left finalize's two
	   quiescence loops structurally blind to it -- a range edge descheduled
	   after the shutdown check above could resume once teardown had flushed,
	   disabled and unsubscribed, re-enable every activity kind, and write into
	   a journal_fd that close(2) had already released to another thread's
	   open(2). Re-reading the flag after the increment is what closes that
	   window: finalize stores shutdown_started before it waits, so a callback
	   that gets counted either sees the flag or is seen by the loop. */
	atomic_fetch_add(&callback_active, 1);
	if (atomic_load(&shutdown_started)) {
		atomic_fetch_sub(&callback_active, 1);
		return;
	}
	/* Gating and finalize-on-stop subscribe to the same four callbacks for
	   different reasons, so an ungated run that only wants the stop trigger
	   toggles nothing and writes no edges: there is no range for an edge to
	   bound, and a journal that reported one would claim a scope it never had.

	   A target may call Start twice, or Stop without Start. Collection is a
	   level, not a counter: a redundant edge is recorded and otherwise
	   ignored, because toggling on an already-open range would make the
	   journal claim a boundary the hardware never saw.

	   Claiming the level and testing it must be one step. set_activity_collection
	   does not publish `collecting` until after its enable loop -- seven CUPTI
	   calls on a real library -- so a load-then-act leaves that whole window
	   open: a Stop arriving inside a concurrent Start's loop reads the old
	   level, calls itself redundant, and returns, after which the Start
	   publishes 1 and collection stays on for the rest of the process. The
	   journal then blames the target for a Stop it did issue. The mirror
	   holds too, leaving the gate shut for the rest of a run. The exchange
	   below is what makes the level a gate rather than a hint; on failure the
	   edge really is redundant, because some other thread owns the transition. */
	if (range_gated == RANGE_GATED) {
		int expected = start ? 0 : 1;
		if (atomic_compare_exchange_strong(&collecting, &expected, start)) {
			unsigned index = start ? atomic_fetch_add(&range_opens, 1) : atomic_load(&range_opens) - 1;
			/* A stop edge is stamped before collection is disabled and the
			   buffers flushed, not after: the flush delivers every record the
			   range produced, so its cost grows with the record count, and a
			   stamp taken after it put that cost inside the range. A target
			   with three times the launches read a tail three times as long.
			   A start edge is stamped once collection is on, so the range
			   opens no earlier than the gate. */
			uint64_t ts = start ? 0 : now_ns();
			int failures = set_activity_collection(start);
			uint64_t flush_ns = 0;
			if (!start) {
				uint64_t t0 = now_ns();
				(void)cuptiActivityFlushAll(0);
				flush_ns = now_ns() - t0;
			} else {
				ts = now_ns();
			}
			write_range_edge(start ? "start" : "stop", api, index, failures, ts, flush_ns);
		} else {
			/* A redundant edge belongs to no range of its own, so it carries the
			   index of the range it was talking about: the open one for a second
			   Start, and for a Stop with nothing open the index the next Start
			   will take. Reporting the raw count instead invents a range one past
			   the end, which a consumer grouping edges by index reads as a range
			   that opened and never closed. */
			unsigned opens = atomic_load(&range_opens);
			write_range_edge(start ? "redundant_start" : "redundant_stop", api, start && opens > 0 ? opens - 1 : opens, 0, now_ns(), 0);
		}
	}
	/* Every stop finalizes in this mode, including one the gate called
	   redundant: the target asked to stop collecting, and the mode says that
	   is where this capture ends. Leave the callback first -- finalize waits
	   for callback_active to reach zero and this callback is counted in it, so
	   staying here would make finalize wait for itself. Nothing below re-enters
	   the agent, and finalize stores shutdown_started before it waits, which
	   shuts every entry point a concurrent caller could arrive through. */
	if (!start && finalize_mode == FINALIZE_PROFILER_STOP) {
		atomic_fetch_sub(&callback_active, 1);
		cudaprof_cupti_finalize("profiler-stop", 1);
		return;
	}
	atomic_fetch_sub(&callback_active, 1);
}

__attribute__((constructor)) static void cudaprof_cupti_init(void) {
	const char *dir = getenv("CUDAPROF_TRANSPORT");
	const char *session = getenv("CUDAPROF_SESSION");
	if (dir == NULL || parse_session(session) != 0) return;
	char path[PATH_MAX];
	self_pid = (uint64_t)getpid();
	generation = now_ns() ^ self_pid;
	if (generation == 0) generation = 1;
	if (snprintf(worker_dir, sizeof(worker_dir), "%s/process-%" PRIu64 "-%" PRIu64, dir, self_pid, generation) >= (int)sizeof(worker_dir)) return;
	if (mkdir(worker_dir, 0700) != 0 && errno != EEXIST) { worker_dir[0] = '\0'; return; }
	if (snprintf(path, sizeof(path), "%s/capture.journal", worker_dir) >= (int)sizeof(path)) return;
	journal_fd = open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
	if (journal_fd < 0) { worker_dir[0] = '\0'; return; }
	initialized = 1;
	(void)atexit(cudaprof_cupti_atexit);
#ifdef CUDAPROF_TEST_FORCE_ATFORK_FAILURE
	/* Fault injection for the control that asks what a run looks like when the
	   guard cannot be registered. Never defined in a shipping build. */
	const int atfork_rc = 1;
	(void)cudaprof_after_fork_child;
#else
	const int atfork_rc = pthread_atfork(NULL, NULL, cudaprof_after_fork_child);
#endif
	if (atfork_rc != 0) {
		/* Without this handler a forked child keeps a live-looking agent. That is
		   a property of the run, so it is recorded rather than assumed: the
		   capability record carries atfork_guard and a failure record names it. */
		atfork_guard = 0;
		const char *failure = "{\"state\":\"producer_failure\",\"error\":\"pthread_atfork registration failed\"}";
		(void)write_record(KIND_FAILURE, now_ns(), failure, strlen(failure));
	} else {
		atfork_guard = 1;
	}
	if (pthread_create(&worker, NULL, drain_worker, NULL) != 0) { close(journal_fd); journal_fd = -1; return; }
	install_sigterm();
	/* Unify the clocks before the first cuptiActivityEnable below. NVIDIA's
	   documentation makes that an ordering requirement, not a preference:
	   records produced before the callback is installed keep CUPTI's previous
	   timestamp method, so a late registration yields a journal straddling two
	   bases -- worse than never registering, because nothing in the record
	   says which base it is on. This is placed ahead of every CUPTI activity
	   call rather than immediately before the first enable so that an enable
	   added later cannot land in front of it.

	   The API is resolved through dlsym rather than called directly. A direct
	   call makes the symbol a load-time dependency of this library, so a
	   runtime libcupti that does not export it would fail to load the agent
	   into the target process at all -- losing the whole capture to gain a
	   clock. Same reason and same shape as the cuptiActivityGetStructSize
	   probe further down. The typedef is local rather than CUPTI's own
	   because a toolkit whose headers predate the API would not compile a
	   reference to it. */
	typedef uint64_t (CUPTIAPI *pfn_timestamp_callback)(void);
	typedef CUptiResult (*pfn_register_timestamp)(pfn_timestamp_callback);
	pfn_register_timestamp fn_register_timestamp = NULL;
	*(void **)(&fn_register_timestamp) = dlsym(RTLD_DEFAULT, "cuptiActivityRegisterTimestampCallback");
	if (fn_register_timestamp == NULL) {
		clock_state = AGENT_CLOCK_UNSUPPORTED;
	} else {
		CUptiResult clock_result = fn_register_timestamp(cupti_timestamp);
		if (clock_result == CUPTI_SUCCESS) {
			clock_state = AGENT_CLOCK_UNIFIED;
		} else {
			/* A refusal is CUPTI answering a call this agent made, so it is
			   recorded the way every other refused CUPTI call is. Absence is
			   not: there is no call and no result, only a runtime older than
			   the API, which the capability record states. */
			clock_state = AGENT_CLOCK_REFUSED;
			record_cupti_failure("cuptiActivityRegisterTimestampCallback", clock_result);
		}
	}
	CUptiResult result = cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed);
	if (result != CUPTI_SUCCESS) {
		record_cupti_failure("cuptiActivityRegisterCallbacks", result);
		write_handshake(NULL);
		return;
	}
	callback_registered = 1;
	uint32_t runtime_version = 0;
	if (cuptiGetVersion(&runtime_version) != CUPTI_SUCCESS) {
		const char *failure = "{\"state\":\"cupti_unavailable\",\"error\":\"cuptiGetVersion failed\"}";
		(void)write_record(KIND_FAILURE, now_ns(), failure, strlen(failure));
		write_handshake(NULL);
		return;
	}
#ifdef CUDAPROF_TEST_OVERRIDE_RUNTIME_VERSION
	runtime_version = (uint32_t)(CUDAPROF_TEST_OVERRIDE_RUNTIME_VERSION);
#endif
	/* Major layout family must match:
	   - CUDA 13+ uses version format xxyyzz (major = version / 10000).
	   - CUDA 12 uses legacy versions 19..28.
	   Cross-major execution (CUDA 12 Kernel9 vs CUDA 13 Kernel10) is rejected. */
	int compile_is_cu13 = (CUPTI_API_VERSION >= 130000);
	int runtime_is_cu13 = (runtime_version >= 130000);
	if (compile_is_cu13 != runtime_is_cu13) {
		char failure[256];
		snprintf(failure, sizeof(failure),
			"{\"state\":\"cupti_unavailable\",\"error\":\"CUPTI major version mismatch (compile %u, runtime %u)\"}",
			CUPTI_API_VERSION, runtime_version);
		(void)write_record(KIND_FAILURE, now_ns(), failure, strlen(failure));
		write_handshake(NULL);
		return;
	}
	if (compile_is_cu13 && (CUPTI_API_VERSION / 10000 != runtime_version / 10000)) {
		char failure[256];
		snprintf(failure, sizeof(failure),
			"{\"state\":\"cupti_unavailable\",\"error\":\"CUPTI major version mismatch (compile %u, runtime %u)\"}",
			CUPTI_API_VERSION / 10000, runtime_version / 10000);
		(void)write_record(KIND_FAILURE, now_ns(), failure, strlen(failure));
		write_handshake(NULL);
		return;
	}
	/* Feature-detect cuptiActivityGetStructSize (exported in CUDA 13.2+) to verify
	   that runtime activity record sizes satisfy the compiled layout requirements. */
	typedef CUptiResult (*pfn_get_struct_size)(CUpti_ActivityKind, uint32_t, size_t *);
#ifdef CUDAPROF_TEST_STRUCT_SIZE_FN
	extern CUptiResult CUDAPROF_TEST_STRUCT_SIZE_FN(CUpti_ActivityKind, uint32_t, size_t *);
	pfn_get_struct_size fn_get_struct_size = CUDAPROF_TEST_STRUCT_SIZE_FN;
#else
	/* ISO C forbids a direct object-to-function pointer cast, which -pedantic
	   rejects. POSIX sanctions assigning through an object pointer instead. */
	pfn_get_struct_size fn_get_struct_size = NULL;
	*(void **)(&fn_get_struct_size) = dlsym(RTLD_DEFAULT, "cuptiActivityGetStructSize");
#ifdef CUDAPROF_REQUIRE_STRUCT_SIZE_FN
	/* A build that links a CUPTI exporting the symbol is a different case from
	   a runtime too old to have it: there the resolution failing says the
	   build is wrong, not that the runtime is old, and a configuration that
	   knows the symbol is present refuses rather than skipping every check
	   below. Production keeps the degrading path, where an old runtime is
	   real. */
	if (fn_get_struct_size == NULL) {
		const char *failure = "{\"state\":\"cupti_unavailable\",\"error\":\"cuptiActivityGetStructSize did not resolve in a build that requires it\"}";
		(void)write_record(KIND_FAILURE, now_ns(), failure, strlen(failure));
		write_handshake(NULL);
		return;
	}
#endif
#endif
	/* Whether the runtime layouts were checked at all is a separate fact from
	   what they were checked against: without the symbol every check below is
	   skipped, and a capability record that reported only the compiled layout
	   names would read identically whether they were verified or assumed. */
	const char *layout_check = fn_get_struct_size != NULL ? "cuptiActivityGetStructSize" : "unavailable";
	/* CUDAPROF_AGENT_REVISION names the source this agent was built from. The
	   build recipe in README.md defines it; a build that does not leaves it
	   empty, and an empty string is written as an absent field rather than as
	   "unknown", because a reader will treat any value as an answer.

	   The agent and the cudaprof that reads it are separate artifacts resolved
	   separately -- the .so is found next to the executable -- so either can be
	   older than the other, and a capability record identifying only one of
	   them still cannot say what it describes. cudaprof stamps the other half
	   under "producer". */
	const char *agent_revision = CUDAPROF_AGENT_REVISION;
	char agent_revision_buf[128];
	const char *agent_revision_field = "";
	if (agent_revision[0] != '\0') {
		int rev_len = snprintf(agent_revision_buf, sizeof(agent_revision_buf), "\"agent_revision\":\"%s\",", agent_revision);
		/* A truncated revision is the failure this field exists to prevent: it
		   would name a different commit and still look like an answer. Drop the
		   field instead, so the record says it does not know. */
		if (rev_len > 0 && (size_t)rev_len < sizeof(agent_revision_buf)) {
			agent_revision_field = agent_revision_buf;
		}
	}

	/* A runtime whose memset record is shorter than the layout this agent
	   reads would be read past its end. That is a reason not to collect the
	   kind, not a reason to fail the capture, so unlike the kernel and memcpy
	   checks below it degrades: the capability record says memsets are off. */
	int memset_layout_ok = 1;
	if (fn_get_struct_size != NULL) {
		size_t memset_sz = 0;
		if (fn_get_struct_size(CUPTI_ACTIVITY_KIND_MEMSET, 0, &memset_sz) != CUPTI_SUCCESS || memset_sz < sizeof(CUpti_ActivityMemset4)) memset_layout_ok = 0;
	}
	/* The same degrade-rather-than-fail policy for the three tiers below. Each
	   is read as a prefix of the record the runtime delivers, so a runtime
	   whose record is shorter than the prefix would be read past its end. */
	int overhead_layout_ok = 1, graph_trace_layout_ok = 1, sync_layout_ok = 1;
	if (fn_get_struct_size != NULL) {
		size_t sz = 0;
		if (fn_get_struct_size(CUPTI_ACTIVITY_KIND_OVERHEAD, 0, &sz) != CUPTI_SUCCESS || sz < sizeof(CUpti_ActivityOverhead)) overhead_layout_ok = 0;
		sz = 0;
		if (fn_get_struct_size(CUPTI_ACTIVITY_KIND_GRAPH_TRACE, 0, &sz) != CUPTI_SUCCESS || sz < sizeof(CUpti_ActivityGraphTrace)) graph_trace_layout_ok = 0;
		sz = 0;
		if (fn_get_struct_size(CUPTI_ACTIVITY_KIND_SYNCHRONIZATION, 0, &sz) != CUPTI_SUCCESS || sz < sizeof(CUpti_ActivitySynchronization)) sync_layout_ok = 0;
	}
	if (fn_get_struct_size != NULL) {
		size_t kernel_sz = 0, memcpy_sz = 0;
		if (fn_get_struct_size(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL, 0, &kernel_sz) != CUPTI_SUCCESS ||
		    kernel_sz < sizeof(CUDAPROF_KERNEL_TYPE)) {
			const char *failure = "{\"state\":\"cupti_unavailable\",\"error\":\"CUPTI runtime kernel activity layout incompatible\"}";
			(void)write_record(KIND_FAILURE, now_ns(), failure, strlen(failure));
			write_handshake(NULL);
			return;
		}
		if (fn_get_struct_size(CUPTI_ACTIVITY_KIND_MEMCPY, 0, &memcpy_sz) != CUPTI_SUCCESS ||
		    memcpy_sz < sizeof(CUpti_ActivityMemcpy6)) {
			const char *failure = "{\"state\":\"cupti_unavailable\",\"error\":\"CUPTI runtime memcpy activity layout incompatible\"}";
			(void)write_record(KIND_FAILURE, now_ns(), failure, strlen(failure));
			write_handshake(NULL);
			return;
		}
	}
	configure_memory_activity(runtime_version, fn_get_struct_size);
	const char *kernel_mode = "concurrent_kernel";
	result = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
	if (result != CUPTI_SUCCESS) {
		CUptiResult legacy = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
		if (legacy != CUPTI_SUCCESS) {
			record_cupti_failure("cuptiActivityEnable(CONCURRENT_KERNEL/KERNEL)", result);
			write_handshake(NULL);
			return;
		}
		kernel_mode = "legacy_kernel";
		legacy_enabled = 1;
	} else {
		concurrent_enabled = 1;
	}
	result = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
	if (result != CUPTI_SUCCESS) {
		record_cupti_failure("cuptiActivityEnable(MEMCPY)", result);
		write_handshake(NULL);
		return;
	}
	memcpy_enabled = 1;
	/* A memset is device work with a duration, and until it was collected the
	   time a cudaMemsetAsync spent on the device was reported as idle by every
	   consumer that reads this journal. It is enabled on the MARKER policy
	   rather than the MEMCPY one: a CUPTI too old to offer the kind is a
	   supported configuration recorded in the capability record, whereas
	   failing the capture would lose the kernels and copies as well.

	   The kind is off unless CUDAPROF_MEMSETS=1. With it enabled, CUDA 13
	   cuGraphLaunch on a GB10 faulted inside libcuda, called from CUPTI, for an
	   MLX LoRA run that relaunched a cached exec updated with a captured
	   memset node (5 of 5 runs; 0 of 3 with every other kind this agent
	   enables). The fault is in the driver, and the kind is the only switch
	   this agent holds over it, so a target must ask for it. */
	const char *memset_setting = getenv("CUDAPROF_MEMSETS");
	int memset_requested = memset_setting != NULL && strcmp(memset_setting, "1") == 0;
	if (memset_requested && memset_layout_ok) {
		result = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET);
		if (result == CUPTI_SUCCESS) memset_enabled = 1;
	}
	/* CUPTI's own cost, the graph replay envelope and the synchronization
	   windows. None of the three is device work, and all three exist to
	   explain device idle rather than to add to device busy -- an overhead
	   interval and a sync window were previously indistinguishable from the
	   device having nothing to do. They take the MARKER policy: a CUPTI that
	   refuses one is a supported configuration recorded in the capability
	   record, not a reason to lose the kernels and copies with it. */
	if (overhead_layout_ok) {
		result = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OVERHEAD);
		if (result == CUPTI_SUCCESS) overhead_enabled = 1;
	}
	/* Graph replays are traced one of two ways, and the reader chooses. In
	   node mode, the default, GRAPH_TRACE stays off and every kernel node of
	   a replay arrives as a kernel record carrying graphId and graphNodeId.
	   In graph mode GRAPH_TRACE is enabled and CUPTI reports one envelope per
	   replay instead, without the nodes inside it (see append_graph_trace):
	   cheaper to trace, but the kernels in a replay are then invisible. A
	   graph request CUPTI refuses leaves node mode in effect, and the
	   capability record says the request went unmet. */
	if (graph_trace_requested()) {
		graph_trace_mode = "requested_unavailable";
		if (graph_trace_layout_ok) {
			result = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_GRAPH_TRACE);
			if (result == CUPTI_SUCCESS) {
				graph_trace_enabled = 1;
				graph_trace_mode = "graph";
			}
		}
	}
	if (sync_layout_ok) {
		result = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_SYNCHRONIZATION);
		if (result == CUPTI_SUCCESS) sync_enabled = 1;
	}
	result = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION);
	if (result != CUPTI_SUCCESS) {
		record_cupti_failure("cuptiActivityEnable(EXTERNAL_CORRELATION)", result);
		write_handshake(NULL);
		return;
	}
	external_enabled = 1;
	/* NVTX is the only tier here the target supplies rather than the driver.
	   A target with no NVTX instrumentation is the ordinary case, and a CUPTI
	   without marker support is a supported configuration, so a refusal here
	   is recorded in the capability record and the capture continues. Every
	   other enable above is a hard failure because losing it means losing
	   activity the target did not have to ask for. */
	result = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MARKER);
	if (result == CUPTI_SUCCESS) marker_enabled = 1;
	/* Runtime and driver API attribution dominates journal volume: it was 92%
	   of the records in the measured Dynamo worker journals. Now that the
	   drain retains everything offered, that volume is real bytes on disk, so
	   API attribution is a configured choice and an activity-only run is a
	   separate, explicitly recorded configuration rather than a silent
	   degradation. */
	const char *api_setting = getenv("CUDAPROF_API_ACTIVITY");
	int api_requested = !(api_setting != NULL && strcmp(api_setting, "0") == 0);
	if (api_requested) {
		result = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME);
		if (result != CUPTI_SUCCESS) {
			record_cupti_failure("cuptiActivityEnable(RUNTIME)", result);
			write_handshake(NULL);
			return;
		}
		runtime_enabled = 1;
	}
	cupti_ready = 1;
	const char *bootstrap = "{\"kind\":\"bootstrap\"}";
	// 4096 was outgrown once already: the record gained agent_revision, and
	// every field added to it costs bytes that nothing was counting. Truncation
	// used to drop the record with no trace, which reaches the reader as
	// "untyped or out-of-order activity" -- a complaint about the next record,
	// naming neither this buffer nor the field that overflowed it.
	char capability[CUDAPROF_CAPABILITY_BUFFER];
	if (api_requested) {
		result = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER);
		if (result != CUPTI_SUCCESS) {
			record_cupti_failure("cuptiActivityEnable(DRIVER)", result);
			write_handshake(NULL);
			return;
		}
		driver_enabled = 1;
	}
	/* Capture-range gating is established last, after every enable above has
	   reported whether its kind is available. Doing it here rather than before
	   the enables keeps one meaning for an enable failure -- the kind is
	   unavailable -- instead of two.

	   If gating was asked for and cannot be established, collection stays ON
	   and the capability record says so. The alternative, failing closed,
	   would produce an empty journal that looks exactly like a target that
	   never opened a range; the alternative of staying silent would hand the
	   consumer an unscoped journal it believes is scoped. Neither is a state
	   the reader can tell apart without being told, so it is told.

	   Gating is one of two reasons to watch the profiler API. Finalizing on a
	   stop is the other, and it is a separate request: a target that calls
	   only cudaProfilerStop, to hand the agent the flush its exit path will
	   never run, has no range to gate and gating it would collect nothing. One
	   subscription serves both, because CUPTI allows one subscriber. */
	const char *range_env = getenv("CUDAPROF_CAPTURE_RANGE");
	const char *graph_env = getenv("CUDAPROF_GRAPH_LIFECYCLE");
	graph_lifecycle_requested = graph_env != NULL && strcmp(graph_env,"1") == 0;
	int range_requested = range_env != NULL && strcmp(range_env, "cudaProfilerApi") == 0;
	const char *finalize_env = getenv("CUDAPROF_FINALIZE");
	int stop_requested = finalize_env != NULL && strcmp(finalize_env, "profiler-stop") == 0;
	int exit_group_requested = finalize_env != NULL && strcmp(finalize_env, "exit-group") == 0;
	/* Module capture rides the same subscription, because CUPTI allows one
	   subscriber per process and the profiler-API triggers already hold it.

	   It is off unless the reader asks for it. The journal's record kinds are
	   closed -- a consumer that meets a kind it does not know refuses the whole
	   journal, not the record -- so an agent that emitted module records by
	   default would break every cudaprof built before this kind existed, on a
	   capture that otherwise had nothing wrong with it. The component that
	   knows it can decode the kind is the one that turns it on, and cudaprof
	   capture exports it, so a user still gets modules without asking. */
	const char *modules_env = getenv("CUDAPROF_MODULES");
	modules_enabled = modules_env != NULL && strcmp(modules_env, "0") != 0 && strcmp(modules_env, "off") != 0;
	module_file_limit = env_bytes("CUDAPROF_MODULE_FILE_BYTES", module_file_limit);
	module_total_limit = env_bytes("CUDAPROF_MODULE_TOTAL_BYTES", module_total_limit);
	/* PC sampling is gated the same way and for the same two reasons: the
	   record kind is new, and the collector is not free. The period is the
	   knob that decides both how fine the profile is and whether the hardware
	   buffer survives the run. */
	const char *pcsampling_env = getenv("CUDAPROF_PCSAMPLING");
	pcsampling_requested = pcsampling_env != NULL && strcmp(pcsampling_env, "0") != 0 && strcmp(pcsampling_env, "off") != 0;
	uint64_t period = env_bytes("CUDAPROF_PCSAMPLING_PERIOD", pcsampling_period);
	if (period >= 5 && period <= 31) pcsampling_period = (uint32_t)period;
	pcsampling_hardware_bytes = env_bytes("CUDAPROF_PCSAMPLING_HARDWARE_BYTES", pcsampling_hardware_bytes);
	pcsampling_scratch_bytes = env_bytes("CUDAPROF_PCSAMPLING_SCRATCH_BYTES", pcsampling_scratch_bytes);
	/* cuBLASLt calls: gated for the same reason as modules, and not free --
	   the library formats a line per call. */
	const char *gemms_env = getenv("CUDAPROF_GEMMS");
	gemms_requested = gemms_env != NULL && strcmp(gemms_env, "0") != 0 && strcmp(gemms_env, "off") != 0;
	int gemms_requested_at_start = gemms_requested;
	int modules_requested = modules_enabled;
	int pcsampling_requested_at_start = pcsampling_requested;
	uint64_t depth = env_bytes("CUDAPROF_CRASH_TRACE", 0);
	crash_depth = depth > CRASH_RING ? CRASH_RING : (unsigned)depth;
	unsigned crash_requested = crash_depth;
	const char *dependency_env = getenv("CUDAPROF_DEPENDENCIES");
	dependency_requested = dependency_env != NULL && strcmp(dependency_env, "1") == 0;
	if (range_requested || stop_requested || modules_enabled || pcsampling_requested || crash_depth > 0 || gemms_requested || graph_lifecycle_requested || dependency_requested || memory_policy_requested) {
		CUptiResult sub = cuptiSubscribe(&range_subscriber, range_callback, NULL);
		if (sub != CUPTI_SUCCESS) {
			record_cupti_failure("cuptiSubscribe(profiler API)", sub);
			if (range_requested) range_gated = RANGE_UNAVAILABLE;
			if (stop_requested) finalize_mode = FINALIZE_UNAVAILABLE;
			modules_enabled = 0;
			pcsampling_requested = 0;
			crash_depth = 0;
			gemms_requested = 0;
		} else {
			range_subscribed = 1;
			if (memory_policy_requested) enable_memory_policy(range_subscriber);
			if (dependency_requested) {
				CUptiResult dr = cuptiEnableDomain(1, range_subscriber, CUPTI_CB_DOMAIN_DRIVER_API);
				if (dr == CUPTI_SUCCESS) dependency_enabled = 1;
				else record_cupti_failure("cuptiEnableDomain(dependencies)", dr);
			}
			if (graph_lifecycle_requested) {
				CUptiResult gr = cuptiEnableDomain(1,range_subscriber,CUPTI_CB_DOMAIN_DRIVER_API);
				if (gr == CUPTI_SUCCESS) {
					graph_lifecycle_enabled=1;
					CUptiResult rr=cuptiEnableCallback(1,range_subscriber,CUPTI_CB_DOMAIN_RUNTIME_API,CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_v10000);
                    if (rr != CUPTI_SUCCESS) record_cupti_failure("cuptiEnableCallback(runtime graph replay)",rr);
                    else graph_runtime_replay_enabled |= 1;
                    rr=cuptiEnableCallback(1,range_subscriber,CUPTI_CB_DOMAIN_RUNTIME_API,CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_ptsz_v10000);
                    if (rr != CUPTI_SUCCESS) record_cupti_failure("cuptiEnableCallback(runtime graph replay ptsz)",rr);
                    else graph_runtime_replay_enabled |= 2;
                    gr=cuptiEnableDomain(1,range_subscriber,CUPTI_CB_DOMAIN_RESOURCE);
					if (gr != CUPTI_SUCCESS) record_cupti_failure("cuptiEnableDomain(graph resource)",gr);
				}
				else record_cupti_failure("cuptiEnableDomain(graph lifecycle)",gr);
			}
			/* Every API call rather than a chosen few: which call a target
			   died in is not known in advance. */
			if (crash_depth > 0) {
				CUptiResult dr = cuptiEnableDomain(1, range_subscriber, CUPTI_CB_DOMAIN_DRIVER_API);
				if (dr == CUPTI_SUCCESS) dr = cuptiEnableDomain(1, range_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
				if (dr != CUPTI_SUCCESS) {
					record_cupti_failure("cuptiEnableDomain(crash trace)", dr);
					(void)cuptiEnableDomain(0, range_subscriber, CUPTI_CB_DOMAIN_DRIVER_API);
					crash_depth = 0;
				} else {
					install_crash_trace();
					if (!crash_installed) crash_depth = 0;
				}
			}
			/* The launches a cuBLASLt call makes, which join it to its
			   kernels; the same callbacks give the library a chance to be
			   found once the target loads it. */
			if (gemms_requested) {
				static const CUpti_CallbackId gemm_trace[] = {
					CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel,
					CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz,
					CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx,
					CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz,
					CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel,
					CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz,
				};
				for (size_t i = 0; i < sizeof(gemm_trace) / sizeof(gemm_trace[0]); i++) {
					CUptiResult r = cuptiEnableCallback(1, range_subscriber, CUPTI_CB_DOMAIN_DRIVER_API, gemm_trace[i]);
					if (r != CUPTI_SUCCESS) record_cupti_failure("cuptiEnableCallback(gemm launch)", r);
				}
			}
			if (pcsampling_requested) {
				CUptiResult cr = cuptiEnableCallback(1, range_subscriber, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_CONTEXT_CREATED);
				if (cr != CUPTI_SUCCESS) {
					record_cupti_failure("cuptiEnableCallback(context created)", cr);
					pcsampling_requested = 0;
				}
			}
			if (modules_enabled) {
				CUptiResult mr = cuptiEnableCallback(1, range_subscriber, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_MODULE_LOADED);
				if (mr != CUPTI_SUCCESS) {
					record_cupti_failure("cuptiEnableCallback(module loaded)", mr);
					modules_enabled = 0;
				}
			}
			/* The chain that tells a kernel record which module it came
			   from. A refusal here costs the module id on the affected
			   records and nothing else, so each is recorded and the capture
			   continues: the consumer already says how many records named
			   no module. */
			if (modules_enabled) {
				static const CUpti_CallbackId runtime_graph_updates[] = {
					CUPTI_RUNTIME_TRACE_CBID_cudaGraphKernelNodeSetParams_v10000,
					CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecKernelNodeSetParams_v10010,
					CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecUpdate_v10020,
					CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecChildGraphNodeSetParams_v11010,
					CUPTI_RUNTIME_TRACE_CBID_cudaGraphNodeSetParams_v12020,
					CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecNodeSetParams_v12020,
				};
				for (size_t i = 0; i < sizeof(runtime_graph_updates)/sizeof(runtime_graph_updates[0]); i++) {
					CUptiResult r = cuptiEnableCallback(1, range_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, runtime_graph_updates[i]);
					if (r != CUPTI_SUCCESS) record_cupti_failure("cuptiEnableCallback(graph update)", r);
				}
				CUptiResult gr = cuptiEnableCallback(1, range_subscriber, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED);
				if (gr != CUPTI_SUCCESS) record_cupti_failure("cuptiEnableCallback(graph node clone)", gr);
				static const struct { CUpti_CallbackDomain domain; CUpti_CallbackId cbid; } module_trace[] = {
					{CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuGraphAddKernelNode},
					{CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuGraphAddKernelNode_v2},
					{CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuGraphKernelNodeSetParams},
					{CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuGraphKernelNodeSetParams_v2},
					{CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuGraphExecKernelNodeSetParams},
					{CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuGraphExecKernelNodeSetParams_v2},
					{CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuGraphNodeSetParams},
					{CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuGraphExecNodeSetParams},
					{CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuGraphExecChildGraphNodeSetParams},
					{CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuGraphExecUpdate},
					{CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuGraphExecUpdate_v2},

					{CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuModuleLoad},
					{CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuModuleLoadData},
					{CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuModuleLoadDataEx},
					{CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuModuleLoadFatBinary},
					{CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuModuleGetFunction},
					{CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadData},
					{CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadFromFile},
					{CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLibraryGetKernel},
					{CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuKernelGetFunction},
					{CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel},
					{CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz},
					{CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx},
					{CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz},
					{CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel},
					{CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz},
					{CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000},
					{CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000},
					{CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_v9000},
					{CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_ptsz_v9000},
					{CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060},
					{CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_ptsz_v11060},
				};
				for (size_t i = 0; i < sizeof(module_trace) / sizeof(module_trace[0]); i++) {
					CUptiResult r = cuptiEnableCallback(1, range_subscriber, module_trace[i].domain, module_trace[i].cbid);
					if (r != CUPTI_SUCCESS) record_cupti_failure("cuptiEnableCallback(module attribution)", r);
				}
			}
			static const struct { CUpti_CallbackDomain domain; CUpti_CallbackId cbid; int opens; const char *what; } triggers[] = {
				{CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaProfilerStart_v4000, 1, "cuptiEnableCallback(cudaProfilerStart)"},
				{CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaProfilerStop_v4000, 0, "cuptiEnableCallback(cudaProfilerStop)"},
				{CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuProfilerStart, 1, "cuptiEnableCallback(cuProfilerStart)"},
				{CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuProfilerStop, 0, "cuptiEnableCallback(cuProfilerStop)"},
			};
			int opens_armed = 0, closes_armed = 0;
			for (size_t i = 0; i < sizeof(triggers) / sizeof(triggers[0]); i++) {
				CUptiResult r = cuptiEnableCallback(1, range_subscriber, triggers[i].domain, triggers[i].cbid);
				if (r == CUPTI_SUCCESS) { if (triggers[i].opens) opens_armed++; else closes_armed++; continue; }
				record_cupti_failure(triggers[i].what, r);
			}
			/* One surviving trigger of each kind is enough to gate: a target
			   calls either the runtime pair or the driver pair, not both.
			   Counting triggers without regard to direction is not the same
			   test. If both Start cbids fail and both Stop cbids arm, the
			   count is 2 and the gate closes around a range nothing can ever
			   open -- a journal byte-identical to a correctly gated run whose
			   target simply never profiled, which is exactly the
			   indistinguishable-empty-journal state this gating exists to
			   refuse. Treat an unpaired arm as no gate at all.

			   Finalizing on a stop asks less of the same triggers: it opens no
			   range, so a surviving Stop is the whole requirement and an
			   unarmable Start costs it nothing. The two requests are therefore
			   answered separately, and each says in the capability record
			   whether it was met. */
			if (range_requested) range_gated = opens_armed > 0 && closes_armed > 0 ? RANGE_GATED : RANGE_UNAVAILABLE;
			if (stop_requested) finalize_mode = closes_armed > 0 ? FINALIZE_PROFILER_STOP : FINALIZE_UNAVAILABLE;
			if (range_gated == RANGE_GATED) (void)set_activity_collection(0);
			else if (finalize_mode != FINALIZE_PROFILER_STOP && !modules_enabled && crash_depth == 0 && !gemms_requested && !graph_lifecycle_enabled && !dependency_enabled && !memory_policy_enabled) {
				(void)cuptiUnsubscribe(range_subscriber);
				range_subscribed = 0;
			}
		}
	}
	if (range_gated != RANGE_GATED) atomic_store(&collecting, 1);
	gemm_probe();
	configure_host_sampler();
	/* An interval flush is the last resort for a target that cooperates with
	   nothing: it neither fills a CUPTI buffer nor reaches a trigger, so
	   without one its records are still inside CUPTI when the process ends.
	   Default off, because a flush the run does not need costs CUPTI callback
	   latency on the target's own threads for nothing. The thread starts here,
	   after every kind that can deliver a record has been enabled. */
	const char *flush_env = getenv("CUDAPROF_FLUSH_INTERVAL_MS");
	if (flush_env != NULL) {
		char *end = NULL;
		long ms = strtol(flush_env, &end, 10);
		if (end != flush_env && *end == '\0' && ms > 0) flush_interval_ms = ms;
	}
	if (flush_interval_ms > 0) {
		if (pthread_create(&flusher, NULL, flush_worker, NULL) == 0) {
			flush_started = 1;
		} else {
			/* The capture still runs; it just holds its records until a
			   trigger. Reporting zero here is what keeps the capability record
			   a statement about this process rather than about the request. */
			flush_interval_ms = 0;
			const char *failure = "{\"state\":\"producer_failure\",\"error\":\"flush thread creation failed\"}";
			(void)write_record(KIND_FAILURE, now_ns(), failure, strlen(failure));
		}
	}
	/* Last, so the capture is collecting before the target's exits are held
	   for it. */
	if (exit_group_requested) {
		const char *failed = install_exit_group();
		if (failed == NULL) {
			finalize_mode = FINALIZE_EXIT_GROUP;
		} else {
			finalize_mode = FINALIZE_UNAVAILABLE;
			char failure[256];
			int n = snprintf(failure, sizeof(failure), "{\"state\":\"producer_failure\",\"error\":\"exit-group finalize: %s: %s\"}", failed, strerror(errno));
			if (n > 0 && (size_t)n < sizeof(failure)) (void)write_record(KIND_FAILURE, now_ns(), failure, (size_t)n);
		}
	}
	int capability_len = snprintf(capability, sizeof(capability), "{\"kind\":\"capability\",\"activity\":true,\"api_records\":%s,\"driver_api\":%s,\"runtime_api\":%s,\"external_correlation\":true,\"nvtx_markers\":%s,\"capture_range\":\"%s\",\"compile_cupti\":%d,\"runtime_cupti\":%u,%s\"layout_check\":\"%s\",\"kernel_mode\":\"%s\",\"kernel_layout\":\"%s\",\"memcpy_layout\":\"Memcpy6\",\"memsets\":%s,\"memsets_requested\":%s,\"overheads\":%s,\"graph_traces\":%s,\"graph_trace_mode\":\"%s\",\"graph_lifecycle\":\"%s\",\"graph_lifecycle_coverage\":\"driver_partial\",\"graph_runtime_replay\":%s,\"synchronization\":%s,\"clocks_unified\":%s,\"cupti_clock\":\"%s\",\"device_active_note\":true,\"sigterm_finalize\":%s,\"finalize\":\"%s\",\"flush_interval_ms\":%ld,\"atfork_guard\":%s,\"drain_batch\":%d,\"queue_capacity\":%d,\"admit_wait_nanos\":%ld,\"host_sample\":\"%s\",\"host_sample_hz\":%d,\"host_sample_root\":\"%s\",\"host_sample_root_override\":%s,\"clock_ticks_per_second\":%ld,\"modules\":\"%s\",\"pc_sampling\":\"%s\",\"pc_sampling_period\":%u,\"crash_trace\":\"%s\",\"crash_trace_depth\":%u,\"gemms\":\"%s\",\"library\":\"%s\"}", api_requested ? "true" : "false", driver_enabled ? "true" : "false", runtime_enabled ? "true" : "false", marker_enabled ? "true" : "false", range_gated == RANGE_GATED ? "cudaProfilerApi" : range_gated == RANGE_UNAVAILABLE ? "requested_unavailable" : "none", CUPTI_API_VERSION, runtime_version, agent_revision_field, layout_check, kernel_mode, CUDAPROF_KERNEL_LAYOUT, memset_enabled ? "true" : "false", memset_requested ? "true" : "false", overhead_enabled ? "true" : "false", graph_trace_enabled ? "true" : "false", graph_trace_mode, graph_lifecycle_enabled ? "collected" : graph_lifecycle_requested ? "requested_unavailable" : "off", graph_runtime_replay_enabled == 3 ? "true" : "false", sync_enabled ? "true" : "false", clock_state == AGENT_CLOCK_UNIFIED ? "true" : "false", clock_state == AGENT_CLOCK_UNIFIED ? "monotonic" : clock_state == AGENT_CLOCK_REFUSED ? "platform_default_callback_refused" : "platform_default_callback_unsupported", sigterm_installed ? "true" : "false", finalize_mode == FINALIZE_PROFILER_STOP ? "profiler-stop" : finalize_mode == FINALIZE_EXIT_GROUP ? "exit-group" : finalize_mode == FINALIZE_UNAVAILABLE ? "requested_unavailable" : "exit", flush_interval_ms, atfork_guard ? "true" : "false", DRAIN_BATCH, PENDING_RECORDS, (long)ADMIT_WAIT_NANOS, host_sample_disposition, host_sample_hz, host_sample_root, host_sample_root_override ? "true" : "false", host_sample_clk_tck, modules_enabled ? "collected" : modules_requested ? "requested_unavailable" : "off", pcsampling_requested ? "requested" : pcsampling_requested_at_start ? "requested_unavailable" : "off", pcsampling_period, crash_depth > 0 ? "armed" : crash_requested > 0 ? "requested_unavailable" : "off", crash_depth, gemms_requested ? "requested" : gemms_requested_at_start ? "requested_unavailable" : "off", cupti_library_path());
	(void)write_record(KIND_BOOTSTRAP, now_ns(), bootstrap, strlen(bootstrap));
	capability_len = append_memory_capability(capability, sizeof(capability), capability_len);
	capability_len = append_dependency_capability(capability, sizeof(capability), capability_len);
	int capability_written = 0;
	if (capability_len > 0 && (size_t)capability_len < sizeof(capability))
		capability_written = write_record(KIND_CAPABILITY, now_ns(), capability, (size_t)capability_len) == 0;
	else {
		/* Say so. Without this the capture fails several records later, in
		   words that describe the wrong record: the reader refuses an activity
		   for arriving before a capability that was never written. snprintf
		   returns the length it WANTED, so the message can name the size the
		   record needs rather than leaving it to be guessed. */
		char failure[256];
		int fn = capability_len < 0
			? snprintf(failure, sizeof(failure), "{\"state\":\"producer_failure\",\"error\":\"capability record could not be formatted\"}")
			: snprintf(failure, sizeof(failure), "{\"state\":\"producer_failure\",\"error\":\"capability record needs %d bytes, buffer holds %zu\"}", capability_len + 1, sizeof(capability));
		if (fn > 0 && (size_t)fn < sizeof(failure)) (void)write_record(KIND_FAILURE, now_ns(), failure, (size_t)fn);
	}
	/* The sampler starts here and nowhere earlier. A consumer rejects any
	   record that arrives before bootstrap and capability, and both of those
	   took their sequence numbers under the journal lock above, so a sampler
	   created after this point cannot get ahead of them. Starting it at
	   configure time instead would make the ordering a question of thread
	   scheduling, and the failure would be a rejected capture rather than a
	   missing sample. */
	if (capability_written) {
		atomic_store(&journal_ready, 1);
		if (atomic_load(&device_active)) publish_device_active();
		start_host_sampler();
	}
	write_handshake(NULL);
}

/* Finalization may be triggered by the ELF destructor, by atexit, by the
   terminator thread after SIGTERM, by a capture-range stop under
   CUDAPROF_FINALIZE=profiler-stop, or by a host program calling
   cudaprof_finalize. Exactly one caller runs the ordered sequence and names
   itself in the finalized record; the others wait for it to publish that
   record, so the process cannot exit over a half-written journal.

   in_range_callback says the caller is range_callback, which reaches here from
   inside a callback the range subscriber is dispatching. */
static void cudaprof_cupti_finalize(const char *trigger, int in_range_callback) {
	if (!initialized || journal_fd < 0) return;
	int expected = 0;
	if (!atomic_compare_exchange_strong(&finalize_started, &expected, 1)) {
		for (int i = 0; i < 10000 && !atomic_load(&finalize_done); i++) { struct timespec ts = {0, 1000000}; nanosleep(&ts, NULL); }
		return;
	}
	finalize_trigger = trigger;
	/* Before any of the sequence below, so that from here on this thread is
	   the only one inside CUPTI, and while the drain is still running to take
	   whatever a flush in flight is delivering. */
	stop_flush_worker();
	/* Joined here for the same reason: from this point this thread is the
	   only one inside CUPTI. */
	stop_pcsampling_worker();
	int flush_error = 0;
	if (callback_registered && (concurrent_enabled || legacy_enabled || memcpy_enabled || memset_enabled || overhead_enabled || graph_trace_enabled || sync_enabled || external_enabled || marker_enabled || runtime_enabled || driver_enabled)) {
		if (cuptiActivityFlushAll(1) != CUPTI_SUCCESS) flush_error = 1;
		/* A gated run whose range is still open is disabled here exactly as an
		   ungated one is; a gated run already closed disables nothing, which
		   CUPTI accepts and which set_activity_collection reports as zero
		   failures because it skips kinds that were never enabled. */
		if (set_activity_collection(0) != 0) flush_error = 1;
		/* Unsubscribing from inside a callback that this subscriber is
		   dispatching asks CUPTI to retire a callback that has not returned,
		   so a finalize on the stop edge leaves the subscription in place. It
		   is inert either way: shutdown_started is stored below and
		   range_callback tests it before it touches anything. Disabling the
		   activity kinds from inside the same callback is already how a gated
		   stop closes its range, so only the unsubscribe waits. */
		if (range_subscribed && !in_range_callback) {
			if (cuptiUnsubscribe(range_subscriber) != CUPTI_SUCCESS) flush_error = 1;
			range_subscribed = 0;
		}
		atomic_store(&shutdown_started, 1);
		/* Disabling may release another CUPTI buffer; drain it before the
		 * callback quiescence check. */
		if (cuptiActivityFlushAll(1) != CUPTI_SUCCESS) flush_error = 1;
	}
	atomic_store(&shutdown_started, 1);
	for (int i = 0; i < 1000 && (atomic_load(&callback_active) != 0 || buffers_in_use()); i++) { struct timespec ts = {0, 1000000}; nanosleep(&ts, NULL); }
	if (atomic_load(&callback_active) != 0 || buffers_in_use()) flush_error = 1;
	atomic_store(&callbacks_closed, 1);
	for (int i = 0; i < 1000 && (atomic_load(&callback_active) != 0 || buffers_in_use()); i++) { struct timespec ts = {0, 1000000}; nanosleep(&ts, NULL); }
	if (atomic_load(&callback_active) != 0 || buffers_in_use()) flush_error = 1;
	if (flush_error) atomic_store(&producer_failed, 1);
	stop_host_sampler();
	/* Before the ring is closed: the folded table is written as ordinary
	   records and needs the drain to still be taking them. */
	flush_pcsampling();
	pthread_mutex_lock(&pending_mu); worker_stop = 1; pthread_cond_signal(&pending_cond); pthread_cond_broadcast(&pending_space_cond); pthread_mutex_unlock(&pending_mu);
	if (pthread_join(worker, NULL) != 0) atomic_store(&producer_failed, 1);
	char payload[MAX_PAYLOAD];
	const char *state = atomic_load(&producer_failed) ? "producer_failure" : (atomic_load(&activity_count) ? "collected" : "initialized_no_activity");
	int n = snprintf(payload, sizeof(payload), "{\"state\":\"%s\",\"trigger\":\"%s\",\"activity\":%" PRIu64 ",\"dropped\":%" PRIu64 ",\"native_dropped\":%" PRIu64 ",\"queue_dropped\":%" PRIu64 ",\"queue_oversize_dropped\":%" PRIu64 ",\"queue_contention_dropped\":%" PRIu64 ",\"queue_capacity_dropped\":%" PRIu64 ",\"offered\":%" PRIu64 ",\"retained\":%" PRIu64 ",\"backpressure_waits\":%" PRIu64 ",\"backpressure_nanos\":%" PRIu64 ",\"backpressure_max_nanos\":%" PRIu64 ",\"drain_batches\":%" PRIu64 ",\"drain_max_batch\":%" PRIu64 ",\"drain_write_nanos\":%" PRIu64 ",\"drain_write_max_nanos\":%" PRIu64 ",\"callbacks\":%" PRIu64 ",\"callback_nanos\":%" PRIu64 ",\"callback_max_nanos\":%" PRIu64 ",\"by_kind\":{\"activity\":{\"offered\":%" PRIu64 ",\"retained\":%" PRIu64 ",\"dropped\":%" PRIu64 "},\"api\":{\"offered\":%" PRIu64 ",\"retained\":%" PRIu64 ",\"dropped\":%" PRIu64 "},\"external\":{\"offered\":%" PRIu64 ",\"retained\":%" PRIu64 ",\"dropped\":%" PRIu64 "},\"failure\":{\"offered\":%" PRIu64 ",\"retained\":%" PRIu64 ",\"dropped\":%" PRIu64 "},\"host\":{\"offered\":%" PRIu64 ",\"retained\":%" PRIu64 ",\"dropped\":%" PRIu64 "},\"module\":{\"offered\":%" PRIu64 ",\"retained\":%" PRIu64 ",\"dropped\":%" PRIu64 "},\"pcsample\":{\"offered\":%" PRIu64 ",\"retained\":%" PRIu64 ",\"dropped\":%" PRIu64 "},\"gemm\":{\"offered\":%" PRIu64 ",\"retained\":%" PRIu64 ",\"dropped\":%" PRIu64 "}},\"modules_written\":%u,\"modules_omitted\":%u,\"module_bytes\":%" PRIu64 ",\"module_seen_max\":%u,\"modules_unchecked\":%u,\"host_samples\":%" PRIu64 ",\"host_sample_records\":%" PRIu64 ",\"gemm_calls\":%" PRIu64 ",\"gemm_launches\":%" PRIu64 ",\"gemm_oversize\":%" PRIu64 ",\"gemm_library\":\"%s\"}", state, finalize_trigger, atomic_load(&activity_count), atomic_load(&dropped_count), atomic_load(&native_dropped_count), atomic_load(&queue_dropped_count), atomic_load(&queue_oversize_dropped), atomic_load(&queue_contention_dropped), atomic_load(&queue_capacity_dropped), atomic_load(&offered_count), atomic_load(&drain_records), atomic_load(&queue_backpressure_waits), atomic_load(&queue_backpressure_nanos), atomic_load(&queue_backpressure_max_nanos), atomic_load(&drain_batches), atomic_load(&drain_max_batch), atomic_load(&drain_write_nanos), atomic_load(&drain_write_max_nanos), atomic_load(&callback_count), atomic_load(&callback_total_nanos), atomic_load(&callback_max_nanos),
		atomic_load(&offered_by_kind[KIND_ACTIVITY]), atomic_load(&retained_by_kind[KIND_ACTIVITY]), atomic_load(&dropped_by_kind[KIND_ACTIVITY]),
		atomic_load(&offered_by_kind[KIND_API]), atomic_load(&retained_by_kind[KIND_API]), atomic_load(&dropped_by_kind[KIND_API]),
		atomic_load(&offered_by_kind[KIND_EXTERNAL]), atomic_load(&retained_by_kind[KIND_EXTERNAL]), atomic_load(&dropped_by_kind[KIND_EXTERNAL]),
		atomic_load(&offered_by_kind[KIND_FAILURE]), atomic_load(&retained_by_kind[KIND_FAILURE]), atomic_load(&dropped_by_kind[KIND_FAILURE]),
		atomic_load(&offered_by_kind[KIND_HOST]), atomic_load(&retained_by_kind[KIND_HOST]), atomic_load(&dropped_by_kind[KIND_HOST]),
		atomic_load(&offered_by_kind[KIND_MODULE]), atomic_load(&retained_by_kind[KIND_MODULE]), atomic_load(&dropped_by_kind[KIND_MODULE]),
		atomic_load(&offered_by_kind[KIND_PCSAMPLE]), atomic_load(&retained_by_kind[KIND_PCSAMPLE]), atomic_load(&dropped_by_kind[KIND_PCSAMPLE]),
		atomic_load(&offered_by_kind[KIND_GEMM]), atomic_load(&retained_by_kind[KIND_GEMM]), atomic_load(&dropped_by_kind[KIND_GEMM]),
		module_seq - module_omitted, module_omitted, module_total_bytes, (unsigned)MODULE_SEEN_MAX, module_unchecked,
		atomic_load(&host_samples_taken), atomic_load(&host_sample_records),
		atomic_load(&gemm_calls), atomic_load(&gemm_launches), atomic_load(&gemm_oversize), gemm_library);
	if (n < 0 || (size_t)n >= sizeof(payload) || write_record(KIND_FINALIZED, now_ns(), payload, (size_t)n) != 0) atomic_store(&producer_failed, 1);
	if (fsync(journal_fd) != 0) atomic_store(&producer_failed, 1);
	if (close(journal_fd) != 0) atomic_store(&producer_failed, 1);
	journal_fd = -1;
	atomic_store(&finalize_done, 1);
}

__attribute__((destructor)) static void cudaprof_cupti_fini(void) { cudaprof_cupti_finalize("exit", 0); }

/* cudaprof_finalize is the route a target takes for itself, and it exists
   because the other three are all reachable only through libc.

   The ELF destructor and atexit both run from libc's exit(3), and the SIGTERM
   terminator needs a signal nobody sends. A runtime that issues the exit_group
   syscall directly reaches none of them: Go does exactly this -- runtime.exit
   is SYS_exit_group in sys_linux_arm64.s, no libc on the path -- and a runtime
   that installs its own SIGTERM disposition after this agent's constructor has
   run has taken the third route away as well. Such a target exits 0, reports
   its own work as done, and leaves a journal with no finalized record and no
   activity, because CUPTI's buffers are flushed at finalize and die with the
   process.

   Interposing exit(3) would not help: the syscall is issued without it. So the
   target has to ask, and this is what it asks. Idempotent by the same
   compare-and-swap as every other route, so calling it and then exiting
   normally finalizes once. It returns only once the finalized record is on
   disk, whether this call ran the sequence or found another trigger already
   running it, so a caller that returns from this and then calls exit_group
   loses nothing. It reports no CUptiResult because what it finalizes is the
   journal, not a CUPTI operation: 0 means the capture is finalized, -1 that
   this process has no initialized agent to finalize -- an unconfigured run,
   or a forked child, which owns none of what it inherited. */
__attribute__((visibility("default"))) int cudaprof_finalize(void);
int cudaprof_finalize(void) {
	if (!initialized) return -1;
	cudaprof_cupti_finalize("host", 0);
	return atomic_load(&finalize_done) ? 0 : -1;
}

/* InitializeInjection is the entry point the CUDA driver calls in a library
   named by CUDA_INJECTION64_PATH, which it loads during cuInit. The ELF
   constructor has already run by then -- the driver's dlopen runs it -- so
   there is nothing left to set up, and the agent is the same one LD_PRELOAD
   loads. What differs is who loads it: only a process that initializes CUDA,
   so a shell or launcher in front of the target writes no journal at all
   rather than one that says it saw no CUDA. It returns 1, as the driver
   expects of an injection library that loaded; whether the agent is
   collecting is what its journal says, not this. */
__attribute__((visibility("default"))) int InitializeInjection(void);
int InitializeInjection(void) {
	return 1;
}

/* Thin opt-in wrappers for Python/operator instrumentation. The caller must
   push, dispatch synchronously, and pop on the same OS thread. */
CUptiResult cudaprof_push_external(uint32_t external_kind, uint64_t external_id) {
	if (!external_enabled || atomic_load(&shutdown_started)) return CUPTI_ERROR_NOT_SUPPORTED;
	return cuptiActivityPushExternalCorrelationId((CUpti_ExternalCorrelationKind)external_kind, external_id);
}

CUptiResult cudaprof_pop_external(uint32_t external_kind, uint64_t *external_id) {
	if (!external_enabled || external_id == NULL || atomic_load(&shutdown_started)) return CUPTI_ERROR_NOT_SUPPORTED;
	return cuptiActivityPopExternalCorrelationId((CUpti_ExternalCorrelationKind)external_kind, external_id);
}

static void cudaprof_cupti_atexit(void) { cudaprof_cupti_finalize("exit", 0); }
