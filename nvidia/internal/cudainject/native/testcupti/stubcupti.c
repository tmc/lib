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

/* Tracks which kinds the producer enabled, so an activity-only configuration
   really does stop API records from being delivered, as CUPTI would. */
static _Atomic int enabled_kinds[16];

static int kind_enabled(CUpti_ActivityKind kind) {
	if ((unsigned)kind >= 16) return 0;
	return atomic_load(&enabled_kinds[(unsigned)kind]);
}

CUptiResult cuptiActivityEnable(CUpti_ActivityKind kind) {
	if ((unsigned)kind >= 16) return CUPTI_ERROR_INVALID_PARAMETER;
	atomic_store(&enabled_kinds[(unsigned)kind], 1);
	return CUPTI_SUCCESS;
}

CUptiResult cuptiActivityDisable(CUpti_ActivityKind kind) {
	if ((unsigned)kind >= 16) return CUPTI_ERROR_INVALID_PARAMETER;
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

#define TEST_MAX_ENABLED_CBIDS 16
static struct { unsigned domain, cbid; } enabled_cbids[TEST_MAX_ENABLED_CBIDS];
static unsigned enabled_cbid_count;

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

int cudaprof_test_subscribed(void) { return subscriber_live; }

int cudaprof_test_callback_enabled(unsigned domain, unsigned cbid) {
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

unsigned cudaprof_test_enabled_kind_count(void) {
	unsigned n = 0;
	for (unsigned i = 0; i < 16; i++) {
		if (atomic_load(&enabled_kinds[i])) n++;
	}
	return n;
}

CUptiResult cuptiActivityFlushAll(uint32_t flag) { (void)flag; return CUPTI_SUCCESS; }

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
   Reporting the compiled sizes is the compatible case; the mocks below are
   the incompatible ones. */
CUptiResult cuptiActivityGetStructSize(CUpti_ActivityKind kind, uint32_t flags, size_t *size) {
	(void)flags;
	if (size == NULL) return CUPTI_ERROR_INVALID_PARAMETER;
	switch (kind) {
	case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
	case CUPTI_ACTIVITY_KIND_KERNEL:
		*size = sizeof(CUpti_ActivityKernel9);
		return CUPTI_SUCCESS;
	case CUPTI_ACTIVITY_KIND_MEMCPY:
		*size = sizeof(CUpti_ActivityMemcpy6);
		return CUPTI_SUCCESS;
	case CUPTI_ACTIVITY_KIND_MEMSET:
		*size = sizeof(CUpti_ActivityMemset4);
		return CUPTI_SUCCESS;
	default:
		return CUPTI_ERROR_NOT_SUPPORTED;
	}
}

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
	return CUPTI_ERROR_NOT_SUPPORTED;
}
