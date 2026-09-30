#define _GNU_SOURCE
/*
 * Stub CUPTI used by the producer control harness. It delivers synthetic
 * activity buffers through the producer's own registered request/complete
 * callbacks, so the path under test is the real one: buffer_requested,
 * buffer_completed, the bounded ring, the batched drain, and the journal.
 */
#include "cupti.h"

#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if CUPTI_API_VERSION >= 130000
#define TEST_KERNEL_TYPE CUpti_ActivityKernel10
#else
#define TEST_KERNEL_TYPE CUpti_ActivityKernel9
#endif

static CUpti_BuffersCallbackRequestFunc request_cb;
static CUpti_BuffersCallbackCompleteFunc complete_cb;
static _Atomic size_t native_dropped;
static _Atomic uint32_t next_correlation;

static size_t record_size(CUpti_ActivityKind kind) {
	size_t n;
	switch (kind) {
	case CUPTI_ACTIVITY_KIND_KERNEL:
	case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: n = sizeof(TEST_KERNEL_TYPE); break;
	case CUPTI_ACTIVITY_KIND_MEMCPY: n = sizeof(CUpti_ActivityMemcpy6); break;
	case CUPTI_ACTIVITY_KIND_MEMSET: n = sizeof(CUpti_ActivityMemset4); break;
	case CUPTI_ACTIVITY_KIND_OVERHEAD: n = sizeof(CUpti_ActivityOverhead); break;
	case CUPTI_ACTIVITY_KIND_SYNCHRONIZATION: n = sizeof(CUpti_ActivitySynchronization); break;
	case CUPTI_ACTIVITY_KIND_GRAPH_TRACE: n = sizeof(CUpti_ActivityGraphTrace); break;
	case CUPTI_ACTIVITY_KIND_DRIVER:
	case CUPTI_ACTIVITY_KIND_RUNTIME: n = sizeof(CUpti_ActivityAPI); break;
	case CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION: n = sizeof(CUpti_ActivityExternalCorrelation); break;
	case CUPTI_ACTIVITY_KIND_MARKER: n = sizeof(CUpti_ActivityMarker2); break;
	/* Only the kind prefix: a record nothing decodes is read for its kind and
	   nothing else, so inventing a body for it would assert a layout no code
	   on either side looks at. */
	case CUPTI_ACTIVITY_KIND_TEST_UNDECODED: n = sizeof(CUpti_Activity); break;
	default: return 0;
	}
	return (n + 7u) & ~(size_t)7u;
}

CUptiResult cuptiActivityRegisterCallbacks(CUpti_BuffersCallbackRequestFunc request, CUpti_BuffersCallbackCompleteFunc complete) {
	if (request == NULL || complete == NULL) return CUPTI_ERROR_INVALID_PARAMETER;
	request_cb = request;
	complete_cb = complete;
	return CUPTI_SUCCESS;
}

/* Call ordering. CUPTI requires the timestamp callback to be registered before
   any activity kind is enabled, because records produced earlier keep the
   previous timestamp method. That is a property of the order of two calls and
   nothing else -- a journal from a late registration is byte-indistinguishable
   from a correct one on this stub, and on a real host it differs only by an
   offset no field names. So the stub numbers the calls as they arrive and the
   control reads the two numbers back. A control that only asked "was it
   registered" would pass on the broken order. */
static _Atomic unsigned call_sequence;
static _Atomic unsigned timestamp_registered_at;
static _Atomic unsigned first_enable_at;

static unsigned next_call_sequence(void) { return atomic_fetch_add(&call_sequence, 1u) + 1u; }

/* Tracks which kinds the producer enabled, so an activity-only configuration
   really does stop API records from being delivered, as CUPTI would.

   The bound covers CUPTI's own dense kind enum, which reaches the low nineties
   in CUDA 13, rather than the handful of kinds the agent happened to enable
   when this was written: a kind past the end was refused with
   CUPTI_ERROR_INVALID_PARAMETER, and because every optional kind here degrades
   on a refusal, adding one whose value exceeded the bound would have produced
   a producer that enabled nothing, a stub that delivered nothing, and a suite
   that passed. */
#define TEST_MAX_ACTIVITY_KINDS 128
static _Atomic int enabled_kinds[TEST_MAX_ACTIVITY_KINDS];

static int kind_enabled(CUpti_ActivityKind kind) {
	if ((unsigned)kind >= TEST_MAX_ACTIVITY_KINDS) return 0;
	return atomic_load(&enabled_kinds[(unsigned)kind]);
}

CUptiResult cuptiActivityEnable(CUpti_ActivityKind kind) {
	unsigned at = next_call_sequence();
	unsigned unset = 0;
	/* Only the FIRST enable matters: the requirement is that no record can be
	   produced before the callback is installed, and the first enable is when
	   that becomes possible. */
	(void)atomic_compare_exchange_strong(&first_enable_at, &unset, at);
	if ((unsigned)kind >= TEST_MAX_ACTIVITY_KINDS) return CUPTI_ERROR_INVALID_PARAMETER;
	atomic_store(&enabled_kinds[(unsigned)kind], 1);
	return CUPTI_SUCCESS;
}

/* The timestamp source the producer registered, retained so the control can
   read what it returns. Registering a callback is not the property under test;
   registering one that reads the same clock as the producer's own record
   headers is, and only calling it can tell those apart. */
static _Atomic(CUpti_TimestampCallbackFunc) timestamp_cb;

#ifndef CUDAPROF_TEST_NO_TIMESTAMP_CALLBACK
/* Defining this symbol is what makes the producer's dlsym probe succeed; a
   build with CUDAPROF_TEST_NO_TIMESTAMP_CALLBACK omits the definition and
   stands for a CUPTI predating the API. CUDAPROF_TEST_REFUSE_TIMESTAMP stands
   for one that exports it and rejects the callback. */
CUptiResult cuptiActivityRegisterTimestampCallback(CUpti_TimestampCallbackFunc callback) {
	unsigned at = next_call_sequence();
	if (callback == NULL) return CUPTI_ERROR_INVALID_PARAMETER;
	const char *refuse = getenv("CUDAPROF_TEST_REFUSE_TIMESTAMP");
	if (refuse != NULL && *refuse != '\0' && strcmp(refuse, "0") != 0) return CUPTI_ERROR_NOT_SUPPORTED;
	atomic_store(&timestamp_cb, callback);
	atomic_store(&timestamp_registered_at, at);
	return CUPTI_SUCCESS;
}
#endif

unsigned cudaprof_test_timestamp_registered_at(void) { return atomic_load(&timestamp_registered_at); }
unsigned cudaprof_test_first_enable_at(void) { return atomic_load(&first_enable_at); }

uint64_t cudaprof_test_timestamp(void) {
	CUpti_TimestampCallbackFunc fn = atomic_load(&timestamp_cb);
	if (fn == NULL) return 0;
	return fn();
}

CUptiResult cuptiActivityDisable(CUpti_ActivityKind kind) {
	if ((unsigned)kind >= TEST_MAX_ACTIVITY_KINDS) return CUPTI_ERROR_INVALID_PARAMETER;
	atomic_store(&enabled_kinds[(unsigned)kind], 0);
	return CUPTI_SUCCESS;
}
/* Callback API, the surface capture-range gating subscribes to. CUPTI allows
   one subscriber per process and returns CUPTI_ERROR_MAX_LIMIT_REACHED for a
   second; the stub reproduces that so a producer that subscribes twice fails
   here rather than silently in the field. */
static CUpti_CallbackFunc subscriber_cb;
static void *subscriber_data;
static int subscriber_live;

#define TEST_MAX_ENABLED_CBIDS 64
static struct { unsigned domain, cbid; } enabled_cbids[TEST_MAX_ENABLED_CBIDS];
static unsigned enabled_cbid_count;
/* Domains enabled whole by cuptiEnableDomain, indexed by domain. */
#define TEST_MAX_DOMAINS 8
static int enabled_domains[TEST_MAX_DOMAINS];

CUptiResult cuptiSubscribe(CUpti_SubscriberHandle *subscriber, CUpti_CallbackFunc callback, void *userdata) {
	if (subscriber == NULL || callback == NULL) return CUPTI_ERROR_INVALID_PARAMETER;
	if (subscriber_live) return CUPTI_ERROR_MAX_LIMIT_REACHED;
	subscriber_cb = callback;
	subscriber_data = userdata;
	subscriber_live = 1;
	/* The handle is opaque to the producer and is only ever handed back to
	   this stub, so a non-null distinguished value is the whole contract. */
	*subscriber = (CUpti_SubscriberHandle)&subscriber_live;
	return CUPTI_SUCCESS;
}

CUptiResult cuptiUnsubscribe(CUpti_SubscriberHandle subscriber) {
	if (!subscriber_live || subscriber != (CUpti_SubscriberHandle)&subscriber_live) return CUPTI_ERROR_INVALID_PARAMETER;
	subscriber_cb = NULL;
	subscriber_data = NULL;
	subscriber_live = 0;
	enabled_cbid_count = 0;
	memset(enabled_domains, 0, sizeof enabled_domains);
	return CUPTI_SUCCESS;
}

/* refuses_cbid reports whether the test asked this stub to refuse arming a
   particular callback. CUDAPROF_TEST_REFUSE_CBIDS is a comma-separated list of
   decimal cbids, matched in any domain.

   A partial arming failure is not otherwise reachable: real CUPTI refuses a
   cbid only on a version or permission mismatch, and a producer that treats
   "some triggers armed" as "the gate is established" can close a gate around a
   range nothing is able to open. That state is invisible from outside -- the
   journal is byte-identical to a correctly gated run whose target never
   profiled -- so it needs an injected refusal to be tested at all. */
static int refuses_cbid(unsigned cbid) {
	const char *list = getenv("CUDAPROF_TEST_REFUSE_CBIDS");
	if (list == NULL || *list == '\0') return 0;
	for (const char *p = list; *p != '\0';) {
		char *end = NULL;
		unsigned long v = strtoul(p, &end, 10);
		if (end == p) break;
		if (v == cbid) return 1;
		p = (*end == ',') ? end + 1 : end;
		if (*end != ',' && *end != '\0') break;
	}
	return 0;
}

CUptiResult cuptiEnableCallback(uint32_t enable, CUpti_SubscriberHandle subscriber, CUpti_CallbackDomain domain, CUpti_CallbackId cbid) {
	if (!subscriber_live || subscriber != (CUpti_SubscriberHandle)&subscriber_live) return CUPTI_ERROR_INVALID_PARAMETER;
	if (enable && refuses_cbid((unsigned)cbid)) return CUPTI_ERROR_INVALID_PARAMETER;
	for (unsigned i = 0; i < enabled_cbid_count; i++) {
		if (enabled_cbids[i].domain != (unsigned)domain || enabled_cbids[i].cbid != (unsigned)cbid) continue;
		if (enable) return CUPTI_SUCCESS;
		enabled_cbids[i] = enabled_cbids[--enabled_cbid_count];
		return CUPTI_SUCCESS;
	}
	if (!enable) return CUPTI_SUCCESS;
	if (enabled_cbid_count == TEST_MAX_ENABLED_CBIDS) return CUPTI_ERROR_MAX_LIMIT_REACHED;
	enabled_cbids[enabled_cbid_count].domain = (unsigned)domain;
	enabled_cbids[enabled_cbid_count].cbid = (unsigned)cbid;
	enabled_cbid_count++;
	return CUPTI_SUCCESS;
}

CUptiResult cuptiEnableDomain(uint32_t enable, CUpti_SubscriberHandle subscriber, CUpti_CallbackDomain domain) {
	if (!subscriber_live || subscriber != (CUpti_SubscriberHandle)&subscriber_live) return CUPTI_ERROR_INVALID_PARAMETER;
	if ((unsigned)domain >= TEST_MAX_DOMAINS) return CUPTI_ERROR_INVALID_PARAMETER;
	if (enable && getenv("CUDAPROF_TEST_REFUSE_DOMAINS") != NULL) return CUPTI_ERROR_INVALID_PARAMETER;
	enabled_domains[domain] = enable != 0;
	return CUPTI_SUCCESS;
}

int cudaprof_test_subscribed(void) { return subscriber_live; }

int cudaprof_test_callback_enabled(unsigned domain, unsigned cbid) {
	if (domain < TEST_MAX_DOMAINS && enabled_domains[domain]) return 1;
	for (unsigned i = 0; i < enabled_cbid_count; i++) {
		if (enabled_cbids[i].domain == domain && enabled_cbids[i].cbid == cbid) return 1;
	}
	return 0;
}

int cudaprof_test_fire_callback(unsigned domain, unsigned cbid, unsigned site) {
	if (!subscriber_live || subscriber_cb == NULL) return 0;
	if (!cudaprof_test_callback_enabled(domain, cbid)) return 0;
	CUpti_CallbackData data;
	memset(&data, 0, sizeof data);
	data.callbackSite = (CUpti_ApiCallbackSite)site;
	data.functionName = "cudaprof_control_profiler_api";
	data.correlationId = 1;
	subscriber_cb(subscriber_data, (CUpti_CallbackDomain)domain, (CUpti_CallbackId)cbid, &data);
	return 1;
}

int cudaprof_test_load_module(uint32_t id, const void *cubin, size_t size) {
	if (!subscriber_live || subscriber_cb == NULL) return 0;
	if (!cudaprof_test_callback_enabled(CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_MODULE_LOADED)) return 0;
	CUpti_ModuleResourceData mod = {id, size, (const char *)cubin};
	CUpti_ResourceData res;
	memset(&res, 0, sizeof res);
	res.resourceDescriptor = &mod;
	subscriber_cb(subscriber_data, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_MODULE_LOADED, &res);
	return 1;
}

int cudaprof_test_fire_api(unsigned domain, unsigned cbid, unsigned site, const void *params, uint32_t correlation) {
	if (!subscriber_live || subscriber_cb == NULL) return 0;
	if (!cudaprof_test_callback_enabled(domain, cbid)) return 0;
	CUpti_CallbackData data;
	memset(&data, 0, sizeof data);
	data.callbackSite = (CUpti_ApiCallbackSite)site;
	data.functionName = "cudaprof_control_api";
	data.functionParams = params;
	data.correlationId = correlation;
	subscriber_cb(subscriber_data, (CUpti_CallbackDomain)domain, (CUpti_CallbackId)cbid, &data);
	return 1;
}

int cudaprof_test_offer_kernel(uint32_t correlation) {
	if (request_cb == NULL || complete_cb == NULL) return -1;
	if (!kind_enabled(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)) return 0;
	uint8_t *buffer = NULL;
	size_t size = 0, max_records = 0;
	request_cb(&buffer, &size, &max_records);
	size_t n = record_size(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
	if (buffer == NULL || size < n) return 0;
	TEST_KERNEL_TYPE *k = (TEST_KERNEL_TYPE *)buffer;
	memset(k, 0, n);
	k->kind = CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL;
	k->name = "cudaprof_module_kernel";
	k->start = 50000 + (uint64_t)correlation * 1000;
	k->end = k->start + 100;
	k->queued = k->start - 10;
	k->submitted = k->start - 5;
	k->contextId = 1;
	k->streamId = 7;
	k->correlationId = correlation;
	k->gridX = k->gridY = k->gridZ = 1;
	k->blockX = 32;
	k->blockY = k->blockZ = 1;
	complete_cb(NULL, 7, buffer, size, n);
	return 1;
}

unsigned cudaprof_test_enabled_kind_count(void) {
	unsigned n = 0;
	for (unsigned i = 0; i < TEST_MAX_ACTIVITY_KINDS; i++) {
		if (atomic_load(&enabled_kinds[i])) n++;
	}
	return n;
}

/* Records held back until something flushes. Real CUPTI hands a buffer to
   buffer_completed when that buffer fills or when a flush asks for it, and a
   run whose buffers never fill sees its records only through the second path.
   Staging a buffer here is what lets a control ask what the flush path alone
   delivers: nothing at all reaches the journal until a flush happens. */
static _Atomic unsigned flush_calls;
static _Atomic unsigned flush_delivered;
static _Atomic unsigned staged_kernels, staged_memcpys, staged_apis;

void cudaprof_test_stage_for_flush(unsigned kernels, unsigned memcpys, unsigned apis) {
	atomic_store(&staged_kernels, kernels);
	atomic_store(&staged_memcpys, memcpys);
	atomic_store(&staged_apis, apis);
}

unsigned cudaprof_test_flush_count(void) { return atomic_load(&flush_calls); }
unsigned cudaprof_test_flush_delivered(void) { return atomic_load(&flush_delivered); }

CUptiResult cuptiActivityFlushAll(uint32_t flag) {
	atomic_fetch_add(&flush_calls, 1);
	/* A real flush costs time in proportion to the records it delivers. A
	   delay on the non-forced flush, the one a range stop issues, lets a test
	   tell an edge stamped before the flush from one stamped after it. */
	const char *delay = getenv("CUDAPROF_TEST_FLUSH_DELAY_MS");
	if (flag == 0 && delay != NULL) {
		long ms = strtol(delay, NULL, 10);
		struct timespec ts = {ms / 1000, (ms % 1000) * 1000000L};
		while (nanosleep(&ts, &ts) != 0) {
		}
	}
	unsigned kernels = atomic_exchange(&staged_kernels, 0);
	unsigned memcpys = atomic_exchange(&staged_memcpys, 0);
	unsigned apis = atomic_exchange(&staged_apis, 0);
	if (kernels == 0 && memcpys == 0 && apis == 0) return CUPTI_SUCCESS;
	/* Delivered on the caller's thread, as CUPTI delivers a flushed buffer,
	   so the producer's flusher really does take the buffer_completed path. */
	int placed = cudaprof_test_offer_markers(kernels, memcpys, apis, 0, 0);
	if (placed > 0) atomic_fetch_add(&flush_delivered, (unsigned)placed);
	return CUPTI_SUCCESS;
}

CUptiResult cuptiActivityGetNextRecord(uint8_t *buffer, size_t valid_size, CUpti_Activity **record) {
	if (buffer == NULL || record == NULL) return CUPTI_ERROR_INVALID_PARAMETER;
	uint8_t *next;
	if (*record == NULL) {
		next = buffer;
	} else {
		size_t n = record_size((*record)->kind);
		if (n == 0) return CUPTI_ERROR_UNKNOWN;
		next = (uint8_t *)*record + n;
	}
	if ((size_t)(next - buffer) + sizeof(CUpti_Activity) > valid_size) return CUPTI_ERROR_MAX_LIMIT_REACHED;
	CUpti_Activity *candidate = (CUpti_Activity *)next;
	size_t n = record_size(candidate->kind);
	if (n == 0 || (size_t)(next - buffer) + n > valid_size) return CUPTI_ERROR_MAX_LIMIT_REACHED;
	*record = candidate;
	return CUPTI_SUCCESS;
}

CUptiResult cuptiActivityGetNumDroppedRecords(CUcontext context, uint32_t stream_id, size_t *dropped) {
	(void)context;
	(void)stream_id;
	if (dropped == NULL) return CUPTI_ERROR_INVALID_PARAMETER;
	*dropped = atomic_exchange(&native_dropped, 0);
	return CUPTI_SUCCESS;
}

CUptiResult cuptiActivityPushExternalCorrelationId(CUpti_ExternalCorrelationKind kind, uint64_t id) { (void)kind; (void)id; return CUPTI_SUCCESS; }
CUptiResult cuptiActivityPopExternalCorrelationId(CUpti_ExternalCorrelationKind kind, uint64_t *id) { (void)kind; if (id) *id = 0; return CUPTI_SUCCESS; }
CUptiResult cuptiGetVersion(uint32_t *version) { if (version == NULL) return CUPTI_ERROR_INVALID_PARAMETER; *version = CUPTI_API_VERSION; return CUPTI_SUCCESS; }
CUptiResult cuptiGetResultString(CUptiResult result, const char **message) { (void)result; if (message) *message = "stub"; return CUPTI_SUCCESS; }

void cudaprof_test_set_native_dropped(size_t dropped) { atomic_store(&native_dropped, dropped); }
int cudaprof_test_callbacks_registered(void) { return request_cb != NULL && complete_cb != NULL; }

int cudaprof_test_request_buffer(void) {
	if (request_cb == NULL) return -1;
	uint8_t *buffer = NULL;
	size_t size = 0, max_records = 0;
	request_cb(&buffer, &size, &max_records);
	return buffer != NULL && size != 0;
}

/* Fills one buffer with the requested record mix and delivers it. Returns the
   number of records actually placed, which is the offered count for that
   buffer; a return of zero means the producer's buffer pool was exhausted and
   nothing was offered. */
int cudaprof_test_offer(unsigned kernels, unsigned memcpys, unsigned apis, unsigned externals) {
	return cudaprof_test_offer_markers(kernels, memcpys, apis, externals, 0);
}

/* Markers are offered in START/END pairs sharing an id, because that is the
   only shape CUPTI delivers an NVTX range in. A caller asking for an odd
   count gets a trailing START with no END, which is the open-range case a
   consumer has to survive rather than a malformed input. */
int cudaprof_test_offer_markers(unsigned kernels, unsigned memcpys, unsigned apis, unsigned externals, unsigned markers) {
	return cudaprof_test_offer_mix(kernels, memcpys, apis, externals, markers, 0);
}

/* Memsets are offered above the other counts for the same reason markers are:
   a scenario calibrated on a kernel/memcpy/api mix keeps that mix when a
   memset count is added to it. */
int cudaprof_test_offer_mix(unsigned kernels, unsigned memcpys, unsigned apis, unsigned externals, unsigned markers, unsigned memsets) {
	return cudaprof_test_offer_undecoded(kernels, memcpys, apis, externals, markers, memsets, 0);
}

/* undecoded records carry a kind the producer has no branch for. CUPTI
   delivers only kinds that were enabled, so this shape does not arise from a
   cuptiActivityEnable the agent made; it is what a runtime newer than the
   build delivers, and what the decode loop's final else has to answer for. */
int cudaprof_test_offer_undecoded(unsigned kernels, unsigned memcpys, unsigned apis, unsigned externals, unsigned markers, unsigned memsets, unsigned undecoded) {
	return cudaprof_test_offer_tiers(kernels, memcpys, apis, externals, markers, memsets, undecoded, 0, 0, 0);
}

/* cudaprof_test_offer_graph_replays offers what CUPTI delivers for a graph of
   nodes kernel nodes launched replays times, in one buffer.

   It models the trade the graph trace kind makes. With
   CUPTI_ACTIVITY_KIND_GRAPH_TRACE enabled, each replay is one envelope and
   the kernels inside it are not delivered (cupti_activity.h: a graph trace
   "represents execution for a graph without giving visibility about the
   execution of its nodes"). With it off, each replay delivers one kernel
   record per node, carrying the graph id and the node's id. The kernels of
   one replay share the replay's correlation id, as the launch's do. Graph and
   node ids exist only in the CUDA 13 kernel layout, so a CUDA 12 build
   delivers the node kernels with no ids, as that CUPTI would.

   Returns the number of records placed, or -1 without buffer callbacks. */
int cudaprof_test_offer_graph_replays(unsigned nodes, unsigned replays) {
	if (request_cb == NULL || complete_cb == NULL) return -1;
	uint8_t *buffer = NULL;
	size_t size = 0, max_records = 0;
	request_cb(&buffer, &size, &max_records);
	if (buffer == NULL || size == 0) return 0;
	int envelopes = kind_enabled(CUPTI_ACTIVITY_KIND_GRAPH_TRACE);
	int kernels = kind_enabled(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
	size_t offset = 0;
	int placed = 0;
	for (unsigned r = 0; r < replays; r++) {
		uint32_t correlation = atomic_fetch_add(&next_correlation, 1) + 1;
		uint64_t start = 20000 + (uint64_t)r * 1000;
		if (envelopes) {
			size_t n = record_size(CUPTI_ACTIVITY_KIND_GRAPH_TRACE);
			if (offset + n > size) break;
			CUpti_ActivityGraphTrace *g = (CUpti_ActivityGraphTrace *)(buffer + offset);
			memset(g, 0, n);
			g->kind = CUPTI_ACTIVITY_KIND_GRAPH_TRACE;
			g->correlationId = correlation;
			g->start = start;
			g->end = start + 100 * (uint64_t)nodes + 50;
			g->graphId = 5;
			g->contextId = 1;
			g->streamId = 7;
			offset += n;
			placed++;
			continue;
		}
		if (!kernels) continue;
		for (unsigned i = 0; i < nodes; i++) {
			size_t n = record_size(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
			if (offset + n > size) break;
			TEST_KERNEL_TYPE *k = (TEST_KERNEL_TYPE *)(buffer + offset);
			memset(k, 0, n);
			k->kind = CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL;
			k->name = "cudaprof_graph_node";
			k->start = start + 25 + (uint64_t)i * 100;
			k->end = k->start + 80;
			k->queued = k->start - 10;
			k->submitted = k->start - 5;
			k->contextId = 1;
			k->streamId = 7;
			k->correlationId = correlation;
			k->gridX = k->gridY = k->gridZ = 1;
			k->blockX = 32;
			k->blockY = k->blockZ = 1;
			k->graphId = 5;
			k->graphNodeId = (uint64_t)i + 1;
			offset += n;
			placed++;
		}
	}
	complete_cb(NULL, 7, buffer, size, offset);
	return placed;
}

/* Overheads, graph traces and syncs are offered above the other counts on the
   same rule as markers and memsets: a scenario calibrated on a kernel/memcpy/
   api mix keeps that mix when one of them is added to it.

   The three are the kinds a consumer must not count as device work. The stub
   gives each a span that overlaps the hole between the kernels it writes, so a
   consumer that does count them erases idle time rather than merely reporting
   an extra record. */
int cudaprof_test_offer_tiers(unsigned kernels, unsigned memcpys, unsigned apis, unsigned externals, unsigned markers, unsigned memsets, unsigned undecoded, unsigned overheads, unsigned graph_traces, unsigned syncs) {
	if (request_cb == NULL || complete_cb == NULL) return -1;
	uint8_t *buffer = NULL;
	size_t size = 0, max_records = 0;
	request_cb(&buffer, &size, &max_records);
	if (buffer == NULL || size == 0) return 0;
	size_t offset = 0;
	int placed = 0;
	unsigned markers_placed = 0;
	for (;;) {
		CUpti_ActivityKind kind;
		if (kernels) { kind = CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL; kernels--; }
		else if (memcpys) { kind = CUPTI_ACTIVITY_KIND_MEMCPY; memcpys--; }
		else if (apis) { kind = CUPTI_ACTIVITY_KIND_RUNTIME; apis--; }
		else if (externals) { kind = CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION; externals--; }
		else if (markers) { kind = CUPTI_ACTIVITY_KIND_MARKER; markers--; }
		else if (memsets) { kind = CUPTI_ACTIVITY_KIND_MEMSET; memsets--; }
		else if (undecoded) { kind = CUPTI_ACTIVITY_KIND_TEST_UNDECODED; undecoded--; }
		else if (overheads) { kind = CUPTI_ACTIVITY_KIND_OVERHEAD; overheads--; }
		else if (graph_traces) { kind = CUPTI_ACTIVITY_KIND_GRAPH_TRACE; graph_traces--; }
		else if (syncs) { kind = CUPTI_ACTIVITY_KIND_SYNCHRONIZATION; syncs--; }
		else break;
		if (kind != CUPTI_ACTIVITY_KIND_TEST_UNDECODED && !kind_enabled(kind)) continue;
		size_t n = record_size(kind);
		if (offset + n > size) break;
		uint8_t *slot = buffer + offset;
		memset(slot, 0, n);
		uint32_t correlation = atomic_fetch_add(&next_correlation, 1) + 1;
		switch (kind) {
		case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
			TEST_KERNEL_TYPE *k = (TEST_KERNEL_TYPE *)slot;
			k->kind = kind;
			k->name = "cudaprof_control_kernel";
			k->start = 1000 + correlation;
			k->end = k->start + 100;
			k->queued = k->start - 10;
			k->submitted = k->start - 5;
			k->deviceId = correlation & 1u;
			k->contextId = 1;
			k->streamId = 7;
			k->correlationId = correlation;
			k->gridX = 1;
			k->gridY = 1;
			k->gridZ = 1;
			k->blockX = 32;
			k->blockY = 1;
			k->blockZ = 1;
			k->registersPerThread = 32;
			k->dynamicSharedMemory = 1024;
#if CUPTI_API_VERSION >= 130000
			k->graphId = 0;
			k->graphNodeId = 0;
#endif
			break;
		}
		case CUPTI_ACTIVITY_KIND_MEMCPY: {
			CUpti_ActivityMemcpy6 *m = (CUpti_ActivityMemcpy6 *)slot;
			m->kind = kind;
			m->start = 2000 + correlation;
			m->end = m->start + 20;
			m->bytes = 4096;
			m->copyKind = 1;
			m->runtimeCorrelationId = correlation;
			m->deviceId = correlation & 1u;
			m->contextId = 1;
			m->streamId = 7;
			m->correlationId = correlation;
			break;
		}
		case CUPTI_ACTIVITY_KIND_TEST_UNDECODED:
			((CUpti_Activity *)slot)->kind = kind;
			break;
		case CUPTI_ACTIVITY_KIND_OVERHEAD: {
			CUpti_ActivityOverhead *o = (CUpti_ActivityOverhead *)slot;
			o->kind = kind;
			o->overheadKind = CUPTI_ACTIVITY_OVERHEAD_CUPTI_BUFFER_FLUSH;
			/* A thread, not a device: this is where CUPTI's own cost is
			   actually charged, and a stub that handed the agent a device id
			   would hide the resolution problem the consumer has to solve. */
			o->objectKind = CUPTI_ACTIVITY_OBJECT_THREAD;
			o->objectId.pt.processId = 4242;
			o->objectId.pt.threadId = 99;
			o->start = 6000 + correlation;
			o->end = o->start + 70;
			break;
		}
		case CUPTI_ACTIVITY_KIND_GRAPH_TRACE: {
			CUpti_ActivityGraphTrace *g = (CUpti_ActivityGraphTrace *)slot;
			g->kind = kind;
			g->correlationId = correlation;
			g->start = 7000 + correlation;
			g->end = g->start + 90;
			g->deviceId = correlation & 1u;
			g->graphId = 5;
			g->contextId = 1;
			g->streamId = 7;
			break;
		}
		case CUPTI_ACTIVITY_KIND_SYNCHRONIZATION: {
			CUpti_ActivitySynchronization *y = (CUpti_ActivitySynchronization *)slot;
			y->kind = kind;
			y->type = CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_SYNCHRONIZE;
			y->start = 8000 + correlation;
			y->end = y->start + 60;
			y->correlationId = correlation;
			y->contextId = 1;
			y->streamId = 7;
			y->cudaEventId = 3;
			break;
		}
		case CUPTI_ACTIVITY_KIND_MEMSET: {
			CUpti_ActivityMemset4 *ms = (CUpti_ActivityMemset4 *)slot;
			ms->kind = kind;
			ms->start = 5000 + correlation;
			ms->end = ms->start + 40;
			ms->bytes = 8192;
			ms->value = 0;
			ms->deviceId = correlation & 1u;
			ms->contextId = 1;
			ms->streamId = 7;
			ms->correlationId = correlation;
			break;
		}
		case CUPTI_ACTIVITY_KIND_RUNTIME: {
			CUpti_ActivityAPI *a = (CUpti_ActivityAPI *)slot;
			a->kind = kind;
			a->cbid = 211;
			a->start = 3000 + correlation;
			a->end = a->start + 5;
			a->processId = 1;
			a->threadId = 2;
			a->correlationId = correlation;
			a->returnValue = 0;
			break;
		}
		case CUPTI_ACTIVITY_KIND_MARKER: {
			CUpti_ActivityMarker2 *m = (CUpti_ActivityMarker2 *)slot;
			m->kind = kind;
			m->flags = (markers_placed % 2u) == 0u ? CUPTI_ACTIVITY_FLAG_MARKER_START : CUPTI_ACTIVITY_FLAG_MARKER_END;
			m->id = (markers_placed / 2u) + 1u;
			markers_placed++;
			m->timestamp = 4000 + correlation;
			m->objectKind = CUPTI_ACTIVITY_OBJECT_THREAD;
			m->objectId.pt.processId = 1;
			m->objectId.pt.threadId = 2;
			m->name = "cudaprof_control_range";
			m->domain = "cudaprof";
			break;
		}
		default: {
			CUpti_ActivityExternalCorrelation *x = (CUpti_ActivityExternalCorrelation *)slot;
			x->kind = kind;
			x->externalKind = 1;
			x->externalId = correlation;
			x->correlationId = correlation;
			break;
		}
		}
		offset += n;
		placed++;
	}
	complete_cb(NULL, 7, buffer, size, offset);
	return placed;
}

/* The real CUPTI exports this from 13.2 on, and the producer resolves it with
   dlsym(RTLD_DEFAULT, ...). Without a definition here that lookup found
   nothing, so every layout check the producer makes was skipped in every
   scenario and only the CUDAPROF_TEST_STRUCT_SIZE_FN builds reached them.
   Reporting the compiled sizes is the compatible case. On its own that arm
   is tautological: these sizeof expressions and the producer's come from the
   same headers in the same build, so the check compares the layout against
   itself and cannot reject. The short-size mocks below are the other half of
   the control and the only reason the comparison discriminates -- delete them
   and what is left is an assertion that cannot fail. */
#ifndef CUDAPROF_TEST_NO_STRUCT_SIZE_SYMBOL
CUptiResult cuptiActivityGetStructSize(CUpti_ActivityKind kind, uint32_t flags, size_t *size) {
	(void)flags;
	if (size == NULL) return CUPTI_ERROR_INVALID_PARAMETER;
	switch (kind) {
	case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
	case CUPTI_ACTIVITY_KIND_KERNEL:
		*size = sizeof(TEST_KERNEL_TYPE);
		return CUPTI_SUCCESS;
	case CUPTI_ACTIVITY_KIND_MEMCPY:
		*size = sizeof(CUpti_ActivityMemcpy6);
		return CUPTI_SUCCESS;
	case CUPTI_ACTIVITY_KIND_MEMSET:
		*size = sizeof(CUpti_ActivityMemset4);
		return CUPTI_SUCCESS;
	case CUPTI_ACTIVITY_KIND_OVERHEAD:
		*size = sizeof(CUpti_ActivityOverhead);
		return CUPTI_SUCCESS;
	case CUPTI_ACTIVITY_KIND_GRAPH_TRACE:
		*size = sizeof(CUpti_ActivityGraphTrace);
		return CUPTI_SUCCESS;
	case CUPTI_ACTIVITY_KIND_SYNCHRONIZATION:
		*size = sizeof(CUpti_ActivitySynchronization);
		return CUPTI_SUCCESS;
	default:
		return CUPTI_ERROR_NOT_SUPPORTED;
	}
}

#endif /* CUDAPROF_TEST_NO_STRUCT_SIZE_SYMBOL */
CUptiResult cudaprof_test_mock_short_kernel_struct_size(CUpti_ActivityKind kind, uint32_t flags, size_t *size) {
	(void)flags;
	if (size == NULL) return CUPTI_ERROR_INVALID_PARAMETER;
	if (kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL || kind == CUPTI_ACTIVITY_KIND_KERNEL) {
		*size = sizeof(CUpti_ActivityKernel9) > 8 ? sizeof(CUpti_ActivityKernel9) - 8 : 1;
		return CUPTI_SUCCESS;
	}
	if (kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
		*size = sizeof(CUpti_ActivityMemcpy6);
		return CUPTI_SUCCESS;
	}
	if (kind == CUPTI_ACTIVITY_KIND_MEMSET) {
		*size = sizeof(CUpti_ActivityMemset4);
		return CUPTI_SUCCESS;
	}
	if (kind == CUPTI_ACTIVITY_KIND_OVERHEAD) {
		*size = sizeof(CUpti_ActivityOverhead);
		return CUPTI_SUCCESS;
	}
	if (kind == CUPTI_ACTIVITY_KIND_SYNCHRONIZATION) {
		*size = sizeof(CUpti_ActivitySynchronization);
		return CUPTI_SUCCESS;
	}
	if (kind == CUPTI_ACTIVITY_KIND_GRAPH_TRACE) {
		*size = sizeof(CUpti_ActivityGraphTrace);
		return CUPTI_SUCCESS;
	}
	return CUPTI_ERROR_NOT_SUPPORTED;
}

CUptiResult cudaprof_test_mock_short_memcpy_struct_size(CUpti_ActivityKind kind, uint32_t flags, size_t *size) {
	(void)flags;
	if (size == NULL) return CUPTI_ERROR_INVALID_PARAMETER;
	if (kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL || kind == CUPTI_ACTIVITY_KIND_KERNEL) {
		*size = sizeof(CUpti_ActivityKernel9);
		return CUPTI_SUCCESS;
	}
	if (kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
		*size = sizeof(CUpti_ActivityMemcpy6) > 8 ? sizeof(CUpti_ActivityMemcpy6) - 8 : 1;
		return CUPTI_SUCCESS;
	}
	if (kind == CUPTI_ACTIVITY_KIND_MEMSET) {
		*size = sizeof(CUpti_ActivityMemset4);
		return CUPTI_SUCCESS;
	}
	if (kind == CUPTI_ACTIVITY_KIND_OVERHEAD) {
		*size = sizeof(CUpti_ActivityOverhead);
		return CUPTI_SUCCESS;
	}
	if (kind == CUPTI_ACTIVITY_KIND_SYNCHRONIZATION) {
		*size = sizeof(CUpti_ActivitySynchronization);
		return CUPTI_SUCCESS;
	}
	if (kind == CUPTI_ACTIVITY_KIND_GRAPH_TRACE) {
		*size = sizeof(CUpti_ActivityGraphTrace);
		return CUPTI_SUCCESS;
	}
	return CUPTI_ERROR_NOT_SUPPORTED;
}

/* cudaprof_test_mock_short_graph_trace_struct_size reports every layout at its
   compiled size except the graph trace record, which it reports short. It is
   the runtime on which a graph trace request cannot be met. */
CUptiResult cudaprof_test_mock_short_graph_trace_struct_size(CUpti_ActivityKind kind, uint32_t flags, size_t *size) {
	(void)flags;
	if (size == NULL) return CUPTI_ERROR_INVALID_PARAMETER;
	switch (kind) {
	case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
	case CUPTI_ACTIVITY_KIND_KERNEL:
		*size = sizeof(TEST_KERNEL_TYPE);
		return CUPTI_SUCCESS;
	case CUPTI_ACTIVITY_KIND_MEMCPY:
		*size = sizeof(CUpti_ActivityMemcpy6);
		return CUPTI_SUCCESS;
	case CUPTI_ACTIVITY_KIND_MEMSET:
		*size = sizeof(CUpti_ActivityMemset4);
		return CUPTI_SUCCESS;
	case CUPTI_ACTIVITY_KIND_OVERHEAD:
		*size = sizeof(CUpti_ActivityOverhead);
		return CUPTI_SUCCESS;
	case CUPTI_ACTIVITY_KIND_GRAPH_TRACE:
		*size = sizeof(CUpti_ActivityGraphTrace) - 8;
		return CUPTI_SUCCESS;
	case CUPTI_ACTIVITY_KIND_SYNCHRONIZATION:
		*size = sizeof(CUpti_ActivitySynchronization);
		return CUPTI_SUCCESS;
	default:
		return CUPTI_ERROR_NOT_SUPPORTED;
	}
}

/* PC sampling. The harness has no hardware sampler, so the surface reports
   itself unavailable: a producer that treats an unavailable sampler as a
   failure of the whole capture would fail here, which is the point. */
CUptiResult cuptiPCSamplingEnable(CUpti_PCSamplingEnableParams *pParams) {
	(void)pParams;
	return CUPTI_ERROR_NOT_SUPPORTED;
}

CUptiResult cuptiPCSamplingDisable(CUpti_PCSamplingDisableParams *pParams) {
	(void)pParams;
	return CUPTI_ERROR_NOT_SUPPORTED;
}

CUptiResult cuptiPCSamplingStart(CUpti_PCSamplingStartParams *pParams) {
	(void)pParams;
	return CUPTI_ERROR_NOT_SUPPORTED;
}

CUptiResult cuptiPCSamplingStop(CUpti_PCSamplingStopParams *pParams) {
	(void)pParams;
	return CUPTI_ERROR_NOT_SUPPORTED;
}

CUptiResult cuptiPCSamplingGetData(CUpti_PCSamplingGetDataParams *pParams) {
	(void)pParams;
	return CUPTI_ERROR_NOT_SUPPORTED;
}

CUptiResult cuptiPCSamplingSetConfigurationAttribute(CUpti_PCSamplingConfigurationInfoParams *pParams) {
	(void)pParams;
	return CUPTI_ERROR_NOT_SUPPORTED;
}

CUptiResult cuptiPCSamplingGetNumStallReasons(CUpti_PCSamplingGetNumStallReasonsParams *pParams) {
	(void)pParams;
	return CUPTI_ERROR_NOT_SUPPORTED;
}

CUptiResult cuptiPCSamplingGetStallReasons(CUpti_PCSamplingGetStallReasonsParams *pParams) {
	(void)pParams;
	return CUPTI_ERROR_NOT_SUPPORTED;
}

CUptiResult cuptiGetCubinCrc(CUpti_GetCubinCrcParams *pParams) {
	(void)pParams;
	return CUPTI_ERROR_NOT_SUPPORTED;
}

CUptiResult cuptiGetGraphNodeId(CUgraphNode node, uint64_t *id) {
	if (node == NULL || id == NULL) return CUPTI_ERROR_INVALID_PARAMETER;
	*id = (uint64_t)(uintptr_t)node;
	return CUPTI_SUCCESS;
}

/* Announce two same-named functions and clone their explicit kernel nodes.
			Node ids 1 and 2 are the identities emitted by offer_graph_replays. */
void cudaprof_test_graph_modules(int mutate) {
	for (uintptr_t i = 1; i <= 2; i++) {
		CUmodule module = (CUmodule)(100 + i);
		CUfunction fn = (CUfunction)(200 + i);
		CUgraphNode node = (CUgraphNode)(300 + i);
		CUpti_ModuleResourceData image = {(uint32_t)(20 + i), 0, NULL};
		CUpti_ResourceData resource = {0}; resource.resourceDescriptor = &image;
		subscriber_cb(subscriber_data, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_MODULE_LOADED, &resource);
		cuModuleLoad_params load = {&module, NULL};
		CUresult success = CUDA_SUCCESS;
		CUpti_CallbackData call = {0}; call.callbackSite = CUPTI_API_EXIT; call.functionReturnValue = &success;
		call.functionParams = &load;
		subscriber_cb(subscriber_data, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuModuleLoad, &call);
		cuModuleGetFunction_params function = {&fn, module, "same"}; call.functionParams = &function;
		subscriber_cb(subscriber_data, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuModuleGetFunction, &call);
		test_kernel_node_params params = {fn};
		cuGraphAddKernelNode_params add = {&node, NULL, NULL, 0, &params}; call.functionParams = &add;
		subscriber_cb(subscriber_data, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuGraphAddKernelNode, &call);
		CUpti_GraphData clone = {0}; clone.node = (CUgraphNode)i; clone.originalNode = node;
		resource.resourceDescriptor = &clone;
		subscriber_cb(subscriber_data, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED, &resource);
	}
	if (mutate) {
		CUpti_CallbackData call = {0}; call.callbackSite = CUPTI_API_ENTER;
		subscriber_cb(subscriber_data, mutate >= 1000 ? CUPTI_CB_DOMAIN_RUNTIME_API : CUPTI_CB_DOMAIN_DRIVER_API, (CUpti_CallbackId)(mutate % 1000), &call);
	}
}


static void graph_call(unsigned cbid, const char *name, const void *params, unsigned correlation) {
 if (!subscriber_live || !subscriber_cb || !cudaprof_test_callback_enabled(CUPTI_CB_DOMAIN_DRIVER_API,cbid)) return;
 CUresult success=CUDA_SUCCESS;
 CUpti_CallbackData call={0}; call.callbackSite=CUPTI_API_EXIT; call.functionName=name; call.functionParams=params; call.functionReturnValue=&success; call.correlationId=correlation; call.contextUid=3;
 subscriber_cb(subscriber_data,CUPTI_CB_DOMAIN_DRIVER_API,cbid,&call);
}

void cudaprof_test_graph_lifecycle(void) {
 CUgraph graph=(CUgraph)(uintptr_t)0x101;
 CUgraphExec exec=(CUgraphExec)(uintptr_t)0x202;
 cuGraphCreate_params create={&graph,0}; graph_call(501,"cuGraphCreate",&create,1);
 cuGraphInstantiateWithFlags_params instantiate={&exec,graph,0}; graph_call(643,"cuGraphInstantiateWithFlags",&instantiate,2);
 cuGraphLaunch_params launch={exec,0}; graph_call(514,"cuGraphLaunch",&launch,3);
 cuGraphExecDestroy_params destroy={exec}; graph_call(516,"cuGraphExecDestroy",&destroy,4);
 graph_call(643,"cuGraphInstantiateWithFlags",&instantiate,5);
 graph_call(514,"cuGraphLaunch",&launch,6);
 graph_call(516,"cuGraphExecDestroy",&destroy,7);
 cuGraphDestroy_params destroy_graph={graph}; graph_call(517,"cuGraphDestroy",&destroy_graph,8);
}


void cudaprof_test_graph_updates(void) {
 CUgraph graph=(CUgraph)(uintptr_t)0x101;
 CUgraphExec exec=(CUgraphExec)(uintptr_t)0x202;
 cuGraphCreate_params create={&graph,0}; graph_call(501,"cuGraphCreate",&create,1);
 cuGraphInstantiateWithFlags_params instantiate={&exec,graph,0}; graph_call(643,"cuGraphInstantiateWithFlags",&instantiate,2);
 CUgraphNode node=(CUgraphNode)(uintptr_t)0x303;
 cuGraphAddKernelNode_params add={&node,graph,NULL,0,NULL}; graph_call(502,"cuGraphAddKernelNode",&add,3);
 CUgraphExecUpdateResult result=0;
 cuGraphExecUpdate_params update={exec,graph,NULL,&result}; graph_call(561,"cuGraphExecUpdate",&update,4);
 cuGraphLaunch_params launch={exec,0}; graph_call(514,"cuGraphLaunch",&launch,5);
 cuGraphExecKernelNodeSetParams_params kernel={exec,node,NULL}; graph_call(538,"cuGraphExecKernelNodeSetParams",&kernel,6);
 graph_call(514,"cuGraphLaunch",&launch,7);
 graph_call(586,"cuGraphExecChildGraphNodeSetParams",&kernel,8);
 graph_call(514,"cuGraphLaunch",&launch,9);
}


void cudaprof_test_graph_runtime_replay(void) {
 CUgraph graph=(CUgraph)(uintptr_t)0x101;
 CUgraphExec exec=(CUgraphExec)(uintptr_t)0x202;
 cuGraphCreate_params create={&graph,0}; graph_call(501,"cuGraphCreate",&create,1);
 cuGraphInstantiateWithFlags_params instantiate={&exec,graph,0}; graph_call(643,"cuGraphInstantiateWithFlags",&instantiate,2);
 cuGraphLaunch_params launch={exec,0}; graph_call(514,"cuGraphLaunch",&launch,3);
 if (!subscriber_live || !subscriber_cb || !cudaprof_test_callback_enabled(CUPTI_CB_DOMAIN_RUNTIME_API,311)) return;
 cudaGraphLaunch_v10000_params runtime_launch={exec,0};
 int success=0;
 CUpti_CallbackData call={0};call.callbackSite=CUPTI_API_EXIT;call.functionName="cudaGraphLaunch";call.functionParams=&runtime_launch;call.functionReturnValue=&success;call.correlationId=3;call.contextUid=3;
 subscriber_cb(subscriber_data,CUPTI_CB_DOMAIN_RUNTIME_API,CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_v10000,&call);
}
