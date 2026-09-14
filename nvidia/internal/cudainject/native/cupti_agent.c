#define _GNU_SOURCE
/*
 * Loadable CUPTI activity producer for Linux. Build this file only against a
 * qualified CUDA toolkit; it is deliberately separate from the portable C11
 * bootstrap and from Go collector/decoder packages.
 */
#include <cupti.h>
#include <cupti_activity.h>
#include <cupti_version.h>
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

#define WIRE_HEADER 80
#define MAX_PAYLOAD 4096
#define BUFFER_BYTES (8u << 20)
#define BUFFER_POOL 4
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

/* CUDA 12.9 exposes Kernel9; CUDA 13 exposes the compatible Kernel10
   extension. Keep the decoder tied to the headers used for this build. */
#if CUPTI_API_VERSION >= 28
#define CUDAPROF_KERNEL_TYPE CUpti_ActivityKernel10
#define CUDAPROF_KERNEL_LAYOUT "Kernel10"
#else
#define CUDAPROF_KERNEL_TYPE CUpti_ActivityKernel9
#define CUDAPROF_KERNEL_LAYOUT "Kernel9"
#endif

_Static_assert(sizeof(CUpti_ActivityExternalCorrelation) == 24, "unexpected external correlation size");
_Static_assert(offsetof(CUpti_ActivityExternalCorrelation, kind) == 0, "unexpected external kind offset");
_Static_assert(offsetof(CUpti_ActivityExternalCorrelation, externalKind) == 4, "unexpected external API offset");
_Static_assert(offsetof(CUpti_ActivityExternalCorrelation, externalId) == 8, "unexpected external ID offset");
_Static_assert(offsetof(CUpti_ActivityExternalCorrelation, correlationId) == 16, "unexpected CUDA correlation offset");
_Static_assert(offsetof(CUpti_ActivityExternalCorrelation, reserved) == 20, "unexpected external reserved offset");

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
#define KIND_COUNT 8
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
static int external_enabled;
static int runtime_enabled;
static int driver_enabled;
static int callback_registered;
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
static int sigterm_chain_default;
static int sigterm_installed;
/* Whether the fork guard is registered; reported in the capability record so a
   run never silently assumes it. */
static int atfork_guard;

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
	uint64_t queued, uint64_t submitted, uint8_t copy_kind,
	uint32_t runtime_correlation, uint32_t device, uint32_t context, uint32_t stream, uint32_t correlation,
	uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
	uint32_t block_x, uint32_t block_y, uint32_t block_z,
	uint32_t registers_per_thread, uint32_t dynamic_shared_memory) {
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
		"\",\"valid\":%s,\"start\":%" PRIu64 ",\"end\":%" PRIu64 ",\"bytes\":%" PRIu64 ",\"queued\":%" PRIu64 ",\"submitted\":%" PRIu64 ",\"copy_kind\":%u,\"runtime_correlation\":%u,\"device\":%u,\"context\":%u,\"stream\":%u,\"correlation\":%u,\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],\"grid_x\":%u,\"grid_y\":%u,\"grid_z\":%u,\"block_x\":%u,\"block_y\":%u,\"block_z\":%u,\"registers_per_thread\":%u,\"dynamic_shared_memory\":%u}",
		valid ? "true" : "false", start, end, bytes, queued, submitted, copy_kind, runtime_correlation, device, context, stream, correlation,
		grid_x, grid_y, grid_z, block_x, block_y, block_z,
		grid_x, grid_y, grid_z, block_x, block_y, block_z,
		registers_per_thread, dynamic_shared_memory);
	if (n < 0 || n >= (int)sizeof(payload)) return -1;
	return queue_pending(KIND_ACTIVITY, start, payload, (size_t)n);
}

static int append_external(uint32_t external_kind, uint64_t external_id, uint32_t correlation) {
	char payload[MAX_PAYLOAD];
	int n = snprintf(payload, sizeof(payload), "{\"kind\":\"external_correlation\",\"external_kind\":%u,\"external_id\":%" PRIu64 ",\"correlation\":%u}", external_kind, external_id, correlation);
	if (n < 0 || (size_t)n >= sizeof(payload)) return -1;
	return queue_pending(KIND_EXTERNAL, 0, payload, (size_t)n);
}

static int append_api(const char *domain, const CUpti_ActivityAPI *api) {
	char payload[MAX_PAYLOAD];
	int n = snprintf(payload, sizeof(payload),
		"{\"kind\":\"api\",\"domain\":\"%s\",\"callback_id\":%u,\"start\":%" PRIu64 ",\"end\":%" PRIu64 ",\"process_id\":%u,\"thread_id\":%u,\"correlation\":%u,\"return_value\":%u}",
		domain, (unsigned)api->cbid, api->start, api->end, api->processId, api->threadId, api->correlationId, api->returnValue);
	if (n < 0 || (size_t)n >= sizeof(payload)) return -1;
	return queue_pending(KIND_API, api->start, payload, (size_t)n);
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
			(void)append_activity("kernel", k->name, k->start, k->end, 0, k->queued, k->submitted, 0, 0, k->deviceId, k->contextId, k->streamId, k->correlationId,
				(uint32_t)(k->gridX > 0 ? k->gridX : 0),
				(uint32_t)(k->gridY > 0 ? k->gridY : 0),
				(uint32_t)(k->gridZ > 0 ? k->gridZ : 0),
				(uint32_t)(k->blockX > 0 ? k->blockX : 0),
				(uint32_t)(k->blockY > 0 ? k->blockY : 0),
				(uint32_t)(k->blockZ > 0 ? k->blockZ : 0),
				(uint32_t)k->registersPerThread,
				(uint32_t)(k->dynamicSharedMemory > 0 ? k->dynamicSharedMemory : 0));
		} else if (record->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
			if (rp + sizeof(CUpti_ActivityMemcpy6) < rp || rp + sizeof(CUpti_ActivityMemcpy6) > end) { const char *failure = "{\"state\":\"producer_failure\",\"error\":\"short Memcpy6 record\"}"; (void)queue_pending(KIND_FAILURE, now_ns(), failure, strlen(failure)); break; }
			const CUpti_ActivityMemcpy6 *m6 = (const CUpti_ActivityMemcpy6 *)record;
			(void)append_activity("memcpy", "", m6->start, m6->end, m6->bytes, 0, 0, m6->copyKind, m6->runtimeCorrelationId, m6->deviceId, m6->contextId, m6->streamId, m6->correlationId,
				0, 0, 0, 0, 0, 0, 0, 0);
		} else if (record->kind == CUPTI_ACTIVITY_KIND_DRIVER || record->kind == CUPTI_ACTIVITY_KIND_RUNTIME) {
			if (rp + sizeof(CUpti_ActivityAPI) < rp || rp + sizeof(CUpti_ActivityAPI) > end) { const char *failure = "{\"state\":\"producer_failure\",\"error\":\"short API record\"}"; (void)queue_pending(KIND_FAILURE, now_ns(), failure, strlen(failure)); break; }
			const CUpti_ActivityAPI *api = (const CUpti_ActivityAPI *)record;
			(void)append_api(record->kind == CUPTI_ACTIVITY_KIND_DRIVER ? "driver" : "runtime", api);
		} else if (record->kind == CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION) {
			if (rp + sizeof(CUpti_ActivityExternalCorrelation) < rp || rp + sizeof(CUpti_ActivityExternalCorrelation) > end) { const char *failure = "{\"state\":\"producer_failure\",\"error\":\"short external correlation record\"}"; (void)queue_pending(KIND_FAILURE, now_ns(), failure, strlen(failure)); break; }
			const CUpti_ActivityExternalCorrelation *x = (const CUpti_ActivityExternalCorrelation *)record;
			(void)append_external((uint32_t)x->externalKind, x->externalId, x->correlationId);
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
	int saved = errno;
	atomic_store(&term_signalled, 1);
	if (term_pipe[1] >= 0) {
		const unsigned char byte = 1;
		ssize_t n;
		do { n = write(term_pipe[1], &byte, 1); } while (n < 0 && errno == EINTR);
		(void)n;
	}
	errno = saved;
	/* Chain to the disposition the target installed so its own termination
	   semantics are unchanged. A default disposition is not re-raised here:
	   the terminator restores and re-raises it once the journal is closed. */
	if (!sigterm_chain_default) {
		/* sa_handler and sa_sigaction alias, so SIG_DFL and SIG_IGN are
		   recognized by value whatever SA_SIGINFO says; only a real function
		   pointer is dispatched, and only then does SA_SIGINFO pick the
		   signature. */
		if (prev_sigterm.sa_handler != SIG_DFL && prev_sigterm.sa_handler != SIG_IGN) {
			if (prev_sigterm.sa_flags & SA_SIGINFO) prev_sigterm.sa_sigaction(signo, info, context);
			else prev_sigterm.sa_handler(signo);
		}
	}
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
	if (sigterm_chain_default) {
		/* The target had no SIGTERM disposition of its own. Finalization is
		   complete, so restore the default and re-raise to terminate with the
		   status the sender asked for. */
		struct sigaction dfl;
		memset(&dfl, 0, sizeof(dfl));
		dfl.sa_handler = SIG_DFL;
		sigemptyset(&dfl.sa_mask);
		(void)sigaction(SIGTERM, &dfl, NULL);
		raise(SIGTERM);
	}
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
	/* Classify by the handler value, not by SA_SIGINFO: a default
	   disposition can be reported with that flag already set, and treating it
	   as a handler would leave the signal unhandled and the target alive. */
	sigterm_chain_default = current.sa_handler == SIG_DFL;
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
	callback_registered = 0;
	initialized = 0;

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
	pfn_get_struct_size fn_get_struct_size = (pfn_get_struct_size)dlsym(RTLD_DEFAULT, "cuptiActivityGetStructSize");
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
	result = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION);
	if (result != CUPTI_SUCCESS) {
		record_cupti_failure("cuptiActivityEnable(EXTERNAL_CORRELATION)", result);
		write_handshake(NULL);
		return;
	}
	external_enabled = 1;
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
	char capability[1024];
	if (api_requested) {
		result = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER);
		if (result != CUPTI_SUCCESS) {
			record_cupti_failure("cuptiActivityEnable(DRIVER)", result);
			write_handshake(NULL);
			return;
		}
		driver_enabled = 1;
	}
	int capability_len = snprintf(capability, sizeof(capability), "{\"kind\":\"capability\",\"activity\":true,\"api_records\":%s,\"driver_api\":%s,\"runtime_api\":%s,\"external_correlation\":true,\"compile_cupti\":%d,\"runtime_cupti\":%u,\"kernel_mode\":\"%s\",\"kernel_layout\":\"%s\",\"memcpy_layout\":\"Memcpy6\",\"sigterm_finalize\":%s,\"atfork_guard\":%s,\"drain_batch\":%d,\"queue_capacity\":%d,\"library\":\"%s\"}", api_requested ? "true" : "false", driver_enabled ? "true" : "false", runtime_enabled ? "true" : "false", CUPTI_API_VERSION, runtime_version, kernel_mode, CUDAPROF_KERNEL_LAYOUT, sigterm_installed ? "true" : "false", atfork_guard ? "true" : "false", DRAIN_BATCH, PENDING_RECORDS, cupti_library_path());
	(void)write_record(KIND_BOOTSTRAP, now_ns(), bootstrap, strlen(bootstrap));
	if (capability_len > 0 && (size_t)capability_len < sizeof(capability)) (void)write_record(KIND_CAPABILITY, now_ns(), capability, (size_t)capability_len);
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
	if (callback_registered && (concurrent_enabled || memcpy_enabled || external_enabled || runtime_enabled || driver_enabled)) {
		if (cuptiActivityFlushAll(1) != CUPTI_SUCCESS) flush_error = 1;
		if (concurrent_enabled) { if (cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) != CUPTI_SUCCESS) flush_error = 1; }
		if (legacy_enabled) { if (cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL) != CUPTI_SUCCESS) flush_error = 1; }
		if (memcpy_enabled && cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY) != CUPTI_SUCCESS) flush_error = 1;
		if (external_enabled && cuptiActivityDisable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION) != CUPTI_SUCCESS) flush_error = 1;
		if (runtime_enabled && cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME) != CUPTI_SUCCESS) flush_error = 1;
		if (driver_enabled && cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DRIVER) != CUPTI_SUCCESS) flush_error = 1;
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
	pthread_mutex_lock(&pending_mu); worker_stop = 1; pthread_cond_signal(&pending_cond); pthread_cond_broadcast(&pending_space_cond); pthread_mutex_unlock(&pending_mu);
	if (pthread_join(worker, NULL) != 0) atomic_store(&producer_failed, 1);
	char payload[MAX_PAYLOAD];
	const char *state = atomic_load(&producer_failed) ? "producer_failure" : (atomic_load(&activity_count) ? "collected" : "initialized_no_activity");
	int n = snprintf(payload, sizeof(payload), "{\"state\":\"%s\",\"trigger\":\"%s\",\"activity\":%" PRIu64 ",\"dropped\":%" PRIu64 ",\"native_dropped\":%" PRIu64 ",\"queue_dropped\":%" PRIu64 ",\"queue_oversize_dropped\":%" PRIu64 ",\"queue_contention_dropped\":%" PRIu64 ",\"queue_capacity_dropped\":%" PRIu64 ",\"offered\":%" PRIu64 ",\"retained\":%" PRIu64 ",\"backpressure_waits\":%" PRIu64 ",\"backpressure_nanos\":%" PRIu64 ",\"backpressure_max_nanos\":%" PRIu64 ",\"drain_batches\":%" PRIu64 ",\"drain_max_batch\":%" PRIu64 ",\"drain_write_nanos\":%" PRIu64 ",\"drain_write_max_nanos\":%" PRIu64 ",\"callbacks\":%" PRIu64 ",\"callback_nanos\":%" PRIu64 ",\"callback_max_nanos\":%" PRIu64 ",\"by_kind\":{\"activity\":{\"offered\":%" PRIu64 ",\"retained\":%" PRIu64 ",\"dropped\":%" PRIu64 "},\"api\":{\"offered\":%" PRIu64 ",\"retained\":%" PRIu64 ",\"dropped\":%" PRIu64 "},\"external\":{\"offered\":%" PRIu64 ",\"retained\":%" PRIu64 ",\"dropped\":%" PRIu64 "},\"failure\":{\"offered\":%" PRIu64 ",\"retained\":%" PRIu64 ",\"dropped\":%" PRIu64 "}}}", state, finalize_trigger, atomic_load(&activity_count), atomic_load(&dropped_count), atomic_load(&native_dropped_count), atomic_load(&queue_dropped_count), atomic_load(&queue_oversize_dropped), atomic_load(&queue_contention_dropped), atomic_load(&queue_capacity_dropped), atomic_load(&offered_count), atomic_load(&drain_records), atomic_load(&queue_backpressure_waits), atomic_load(&queue_backpressure_nanos), atomic_load(&queue_backpressure_max_nanos), atomic_load(&drain_batches), atomic_load(&drain_max_batch), atomic_load(&drain_write_nanos), atomic_load(&drain_write_max_nanos), atomic_load(&callback_count), atomic_load(&callback_total_nanos), atomic_load(&callback_max_nanos),
		atomic_load(&offered_by_kind[KIND_ACTIVITY]), atomic_load(&retained_by_kind[KIND_ACTIVITY]), atomic_load(&dropped_by_kind[KIND_ACTIVITY]),
		atomic_load(&offered_by_kind[KIND_API]), atomic_load(&retained_by_kind[KIND_API]), atomic_load(&dropped_by_kind[KIND_API]),
		atomic_load(&offered_by_kind[KIND_EXTERNAL]), atomic_load(&retained_by_kind[KIND_EXTERNAL]), atomic_load(&dropped_by_kind[KIND_EXTERNAL]),
		atomic_load(&offered_by_kind[KIND_FAILURE]), atomic_load(&retained_by_kind[KIND_FAILURE]), atomic_load(&dropped_by_kind[KIND_FAILURE]));
	if (n < 0 || (size_t)n >= sizeof(payload) || write_record(KIND_FINALIZED, now_ns(), payload, (size_t)n) != 0) atomic_store(&producer_failed, 1);
	if (fsync(journal_fd) != 0) atomic_store(&producer_failed, 1);
	if (close(journal_fd) != 0) atomic_store(&producer_failed, 1);
	journal_fd = -1;
	atomic_store(&finalize_done, 1);
}

__attribute__((destructor)) static void cudaprof_cupti_fini(void) { cudaprof_cupti_finalize(); }

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
