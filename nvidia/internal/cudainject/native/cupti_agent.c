#define _GNU_SOURCE
/*
 * Loadable CUPTI activity producer for Linux. Build this file only against a
 * qualified CUDA toolkit; it is deliberately separate from the portable C11
 * bootstrap and from Go collector/decoder packages.
 */
#include <cupti.h>
#include <cupti_activity.h>
#include <cupti_version.h>
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
#define CUDAPROF_GRAPH_ID(k) ((uint32_t)0)
#define CUDAPROF_GRAPH_NODE_ID(k) ((uint64_t)0)
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
#define KIND_COUNT 9
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
static _Atomic int finalize_started;
static _Atomic int finalize_done;
static const char *finalize_trigger = "exit";
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
static void cudaprof_cupti_finalize(void);

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

static int append_activity(const char *kind, const char *name, uint64_t start, uint64_t end, uint64_t bytes,
	uint64_t queued, uint64_t submitted, uint8_t copy_kind, uint32_t memset_value,
	uint32_t runtime_correlation, uint32_t device, uint32_t context, uint32_t stream, uint32_t correlation,
	uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
	uint32_t block_x, uint32_t block_y, uint32_t block_z,
	uint32_t registers_per_thread, uint32_t dynamic_shared_memory,
	uint32_t graph_id, uint64_t graph_node_id) {
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
		"\",\"valid\":%s,\"start\":%" PRIu64 ",\"end\":%" PRIu64 ",\"bytes\":%" PRIu64 ",\"queued\":%" PRIu64 ",\"submitted\":%" PRIu64 ",\"copy_kind\":%u,\"value\":%u,\"runtime_correlation\":%u,\"device\":%u,\"context\":%u,\"stream\":%u,\"correlation\":%u,\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],\"grid_x\":%u,\"grid_y\":%u,\"grid_z\":%u,\"block_x\":%u,\"block_y\":%u,\"block_z\":%u,\"registers_per_thread\":%u,\"dynamic_shared_memory\":%u,\"graph_id\":%u,\"graph_node_id\":%" PRIu64 "}",
		valid ? "true" : "false", start, end, bytes, queued, submitted, copy_kind, memset_value, runtime_correlation, device, context, stream, correlation,
		grid_x, grid_y, grid_z, block_x, block_y, block_z,
		grid_x, grid_y, grid_z, block_x, block_y, block_z,
		registers_per_thread, dynamic_shared_memory, graph_id, graph_node_id);
	if (n < 0 || n >= (int)sizeof(payload)) return -1;
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

/* The graph replay envelope. Per-node visibility already exists -- kernel
   records carry graphId and graphNodeId -- so what this adds is the complement:
   the span the whole replay occupied, against which the nodes inside it can be
   measured. The difference between the two is graph launch and scheduling
   cost, and no per-node record holds it.

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
	if (w < 0 || (size_t)(n + w) >= sizeof(payload)) return -1;
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
	if (w < 0 || (size_t)(n + w) >= sizeof(payload)) return -1;
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

static void CUPTIAPI buffer_requested(uint8_t **buffer, size_t *size, size_t *max_records) {
	*buffer = NULL; *size = 0;
	if (atomic_load(&shutdown_started)) { *max_records = 0; return; }
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
				CUDAPROF_GRAPH_ID(k), CUDAPROF_GRAPH_NODE_ID(k));
		} else if (record->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
			if (rp + sizeof(CUpti_ActivityMemcpy6) < rp || rp + sizeof(CUpti_ActivityMemcpy6) > end) { const char *failure = "{\"state\":\"producer_failure\",\"error\":\"short Memcpy6 record\"}"; (void)queue_pending(KIND_FAILURE, now_ns(), failure, strlen(failure)); break; }
			const CUpti_ActivityMemcpy6 *m6 = (const CUpti_ActivityMemcpy6 *)record;
			(void)append_activity("memcpy", "", m6->start, m6->end, m6->bytes, 0, 0, m6->copyKind, 0, m6->runtimeCorrelationId, m6->deviceId, m6->contextId, m6->streamId, m6->correlationId,
				0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
		} else if (record->kind == CUPTI_ACTIVITY_KIND_MEMSET) {
			if (rp + sizeof(CUpti_ActivityMemset4) < rp || rp + sizeof(CUpti_ActivityMemset4) > end) { const char *failure = "{\"state\":\"producer_failure\",\"error\":\"short Memset4 record\"}"; (void)queue_pending(KIND_FAILURE, now_ns(), failure, strlen(failure)); break; }
			const CUpti_ActivityMemset4 *ms = (const CUpti_ActivityMemset4 *)record;
			(void)append_activity("memset", "", ms->start, ms->end, ms->bytes, 0, 0, 0, ms->value, 0, ms->deviceId, ms->contextId, ms->streamId, ms->correlationId,
				0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
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
	finalize_trigger = "signal";
	cudaprof_cupti_finalize();
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
	};
	const int *flags[] = {
		&concurrent_enabled, &legacy_enabled, &memcpy_enabled, &memset_enabled,
		&overhead_enabled, &graph_trace_enabled, &sync_enabled, &external_enabled,
		&marker_enabled, &runtime_enabled, &driver_enabled,
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
static void write_range_edge(const char *edge, const char *api, unsigned index, int failures) {
	char payload[MAX_PAYLOAD];
	int n = snprintf(payload, sizeof(payload),
		"{\"kind\":\"activity\",\"subtype\":\"capture_range\",\"edge\":\"%s\",\"api\":\"%s\",\"index\":%u,\"toggle_failures\":%d}",
		edge, api, index, failures);
	if (n > 0 && (size_t)n < sizeof(payload)) (void)write_record(KIND_ACTIVITY, now_ns(), payload, (size_t)n);
}

static void CUPTIAPI range_callback(void *userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void *cbdata) {
	(void)userdata;
	const CUpti_CallbackData *data = (const CUpti_CallbackData *)cbdata;
	if (data == NULL || atomic_load(&shutdown_started)) return;
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
	/* A target may call Start twice, or Stop without Start. Collection is a
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
	int expected = start ? 0 : 1;
	if (!atomic_compare_exchange_strong(&collecting, &expected, start)) {
		/* A redundant edge belongs to no range of its own, so it carries the
		   index of the range it was talking about: the open one for a second
		   Start, and for a Stop with nothing open the index the next Start
		   will take. Reporting the raw count instead invents a range one past
		   the end, which a consumer grouping edges by index reads as a range
		   that opened and never closed. */
		unsigned opens = atomic_load(&range_opens);
		write_range_edge(start ? "redundant_start" : "redundant_stop", api, start && opens > 0 ? opens - 1 : opens, 0);
		atomic_fetch_sub(&callback_active, 1);
		return;
	}
	unsigned index = start ? atomic_fetch_add(&range_opens, 1) : atomic_load(&range_opens) - 1;
	int failures = set_activity_collection(start);
	if (!start) (void)cuptiActivityFlushAll(0);
	write_range_edge(start ? "start" : "stop", api, index, failures);
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
	   failing the capture would lose the kernels and copies as well. */
	if (memset_layout_ok) {
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
	if (graph_trace_layout_ok) {
		result = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_GRAPH_TRACE);
		if (result == CUPTI_SUCCESS) graph_trace_enabled = 1;
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
	   the reader can tell apart without being told, so it is told. */
	const char *range_env = getenv("CUDAPROF_CAPTURE_RANGE");
	if (range_env != NULL && strcmp(range_env, "cudaProfilerApi") == 0) {
		CUptiResult sub = cuptiSubscribe(&range_subscriber, range_callback, NULL);
		if (sub != CUPTI_SUCCESS) {
			record_cupti_failure("cuptiSubscribe(capture range)", sub);
			range_gated = RANGE_UNAVAILABLE;
		} else {
			range_subscribed = 1;
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
			   refuse. Treat an unpaired arm as no gate at all. */
			if (opens_armed == 0 || closes_armed == 0) {
				(void)cuptiUnsubscribe(range_subscriber);
				range_subscribed = 0;
				range_gated = RANGE_UNAVAILABLE;
			} else {
				range_gated = RANGE_GATED;
				(void)set_activity_collection(0);
			}
		}
	}
	if (range_gated != RANGE_GATED) atomic_store(&collecting, 1);
	configure_host_sampler();
	int capability_len = snprintf(capability, sizeof(capability), "{\"kind\":\"capability\",\"activity\":true,\"api_records\":%s,\"driver_api\":%s,\"runtime_api\":%s,\"external_correlation\":true,\"nvtx_markers\":%s,\"capture_range\":\"%s\",\"compile_cupti\":%d,\"runtime_cupti\":%u,%s\"layout_check\":\"%s\",\"kernel_mode\":\"%s\",\"kernel_layout\":\"%s\",\"memcpy_layout\":\"Memcpy6\",\"memsets\":%s,\"overheads\":%s,\"graph_traces\":%s,\"synchronization\":%s,\"clocks_unified\":%s,\"cupti_clock\":\"%s\",\"sigterm_finalize\":%s,\"atfork_guard\":%s,\"drain_batch\":%d,\"queue_capacity\":%d,\"admit_wait_nanos\":%ld,\"host_sample\":\"%s\",\"host_sample_hz\":%d,\"host_sample_root\":\"%s\",\"host_sample_root_override\":%s,\"clock_ticks_per_second\":%ld,\"library\":\"%s\"}", api_requested ? "true" : "false", driver_enabled ? "true" : "false", runtime_enabled ? "true" : "false", marker_enabled ? "true" : "false", range_gated == RANGE_GATED ? "cudaProfilerApi" : range_gated == RANGE_UNAVAILABLE ? "requested_unavailable" : "none", CUPTI_API_VERSION, runtime_version, agent_revision_field, layout_check, kernel_mode, CUDAPROF_KERNEL_LAYOUT, memset_enabled ? "true" : "false", overhead_enabled ? "true" : "false", graph_trace_enabled ? "true" : "false", sync_enabled ? "true" : "false", clock_state == AGENT_CLOCK_UNIFIED ? "true" : "false", clock_state == AGENT_CLOCK_UNIFIED ? "monotonic" : clock_state == AGENT_CLOCK_REFUSED ? "platform_default_callback_refused" : "platform_default_callback_unsupported", sigterm_installed ? "true" : "false", atfork_guard ? "true" : "false", DRAIN_BATCH, PENDING_RECORDS, (long)ADMIT_WAIT_NANOS, host_sample_disposition, host_sample_hz, host_sample_root, host_sample_root_override ? "true" : "false", host_sample_clk_tck, cupti_library_path());
	(void)write_record(KIND_BOOTSTRAP, now_ns(), bootstrap, strlen(bootstrap));
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
	if (capability_written) start_host_sampler();
	write_handshake(NULL);
}

/* Finalization may be triggered by the ELF destructor, by atexit, or by the
   terminator thread after SIGTERM. Exactly one caller runs the ordered
   sequence; the others wait for it to publish the finalized record so the
   process cannot exit over a half-written journal. */
static void cudaprof_cupti_finalize(void) {
	if (!initialized || journal_fd < 0) return;
	int expected = 0;
	if (!atomic_compare_exchange_strong(&finalize_started, &expected, 1)) {
		for (int i = 0; i < 10000 && !atomic_load(&finalize_done); i++) { struct timespec ts = {0, 1000000}; nanosleep(&ts, NULL); }
		return;
	}
	int flush_error = 0;
	if (callback_registered && (concurrent_enabled || memcpy_enabled || external_enabled || marker_enabled || runtime_enabled || driver_enabled)) {
		if (cuptiActivityFlushAll(1) != CUPTI_SUCCESS) flush_error = 1;
		/* A gated run whose range is still open is disabled here exactly as an
		   ungated one is; a gated run already closed disables nothing, which
		   CUPTI accepts and which set_activity_collection reports as zero
		   failures because it skips kinds that were never enabled. */
		if (set_activity_collection(0) != 0) flush_error = 1;
		if (range_subscribed) {
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
	pthread_mutex_lock(&pending_mu); worker_stop = 1; pthread_cond_signal(&pending_cond); pthread_cond_broadcast(&pending_space_cond); pthread_mutex_unlock(&pending_mu);
	if (pthread_join(worker, NULL) != 0) atomic_store(&producer_failed, 1);
	char payload[MAX_PAYLOAD];
	const char *state = atomic_load(&producer_failed) ? "producer_failure" : (atomic_load(&activity_count) ? "collected" : "initialized_no_activity");
	int n = snprintf(payload, sizeof(payload), "{\"state\":\"%s\",\"trigger\":\"%s\",\"activity\":%" PRIu64 ",\"dropped\":%" PRIu64 ",\"native_dropped\":%" PRIu64 ",\"queue_dropped\":%" PRIu64 ",\"queue_oversize_dropped\":%" PRIu64 ",\"queue_contention_dropped\":%" PRIu64 ",\"queue_capacity_dropped\":%" PRIu64 ",\"offered\":%" PRIu64 ",\"retained\":%" PRIu64 ",\"backpressure_waits\":%" PRIu64 ",\"backpressure_nanos\":%" PRIu64 ",\"backpressure_max_nanos\":%" PRIu64 ",\"drain_batches\":%" PRIu64 ",\"drain_max_batch\":%" PRIu64 ",\"drain_write_nanos\":%" PRIu64 ",\"drain_write_max_nanos\":%" PRIu64 ",\"callbacks\":%" PRIu64 ",\"callback_nanos\":%" PRIu64 ",\"callback_max_nanos\":%" PRIu64 ",\"by_kind\":{\"activity\":{\"offered\":%" PRIu64 ",\"retained\":%" PRIu64 ",\"dropped\":%" PRIu64 "},\"api\":{\"offered\":%" PRIu64 ",\"retained\":%" PRIu64 ",\"dropped\":%" PRIu64 "},\"external\":{\"offered\":%" PRIu64 ",\"retained\":%" PRIu64 ",\"dropped\":%" PRIu64 "},\"failure\":{\"offered\":%" PRIu64 ",\"retained\":%" PRIu64 ",\"dropped\":%" PRIu64 "},\"host\":{\"offered\":%" PRIu64 ",\"retained\":%" PRIu64 ",\"dropped\":%" PRIu64 "}},\"host_samples\":%" PRIu64 ",\"host_sample_records\":%" PRIu64 "}", state, finalize_trigger, atomic_load(&activity_count), atomic_load(&dropped_count), atomic_load(&native_dropped_count), atomic_load(&queue_dropped_count), atomic_load(&queue_oversize_dropped), atomic_load(&queue_contention_dropped), atomic_load(&queue_capacity_dropped), atomic_load(&offered_count), atomic_load(&drain_records), atomic_load(&queue_backpressure_waits), atomic_load(&queue_backpressure_nanos), atomic_load(&queue_backpressure_max_nanos), atomic_load(&drain_batches), atomic_load(&drain_max_batch), atomic_load(&drain_write_nanos), atomic_load(&drain_write_max_nanos), atomic_load(&callback_count), atomic_load(&callback_total_nanos), atomic_load(&callback_max_nanos),
		atomic_load(&offered_by_kind[KIND_ACTIVITY]), atomic_load(&retained_by_kind[KIND_ACTIVITY]), atomic_load(&dropped_by_kind[KIND_ACTIVITY]),
		atomic_load(&offered_by_kind[KIND_API]), atomic_load(&retained_by_kind[KIND_API]), atomic_load(&dropped_by_kind[KIND_API]),
		atomic_load(&offered_by_kind[KIND_EXTERNAL]), atomic_load(&retained_by_kind[KIND_EXTERNAL]), atomic_load(&dropped_by_kind[KIND_EXTERNAL]),
		atomic_load(&offered_by_kind[KIND_FAILURE]), atomic_load(&retained_by_kind[KIND_FAILURE]), atomic_load(&dropped_by_kind[KIND_FAILURE]),
		atomic_load(&offered_by_kind[KIND_HOST]), atomic_load(&retained_by_kind[KIND_HOST]), atomic_load(&dropped_by_kind[KIND_HOST]),
		atomic_load(&host_samples_taken), atomic_load(&host_sample_records));
	if (n < 0 || (size_t)n >= sizeof(payload) || write_record(KIND_FINALIZED, now_ns(), payload, (size_t)n) != 0) atomic_store(&producer_failed, 1);
	if (fsync(journal_fd) != 0) atomic_store(&producer_failed, 1);
	if (close(journal_fd) != 0) atomic_store(&producer_failed, 1);
	journal_fd = -1;
	atomic_store(&finalize_done, 1);
}

__attribute__((destructor)) static void cudaprof_cupti_fini(void) { cudaprof_cupti_finalize(); }

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
   normally finalizes once. */
__attribute__((visibility("default"))) void cudaprof_finalize(void);
void cudaprof_finalize(void) { cudaprof_cupti_finalize(); }

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

static void cudaprof_cupti_atexit(void) { cudaprof_cupti_finalize(); }
