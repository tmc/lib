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
#include <string.h>

static CUpti_BuffersCallbackRequestFunc request_cb;
static CUpti_BuffersCallbackCompleteFunc complete_cb;
static _Atomic size_t native_dropped;
static _Atomic uint32_t next_correlation;

static size_t record_size(CUpti_ActivityKind kind) {
	size_t n;
	switch (kind) {
	case CUPTI_ACTIVITY_KIND_KERNEL:
	case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: n = sizeof(CUpti_ActivityKernel9); break;
	case CUPTI_ACTIVITY_KIND_MEMCPY: n = sizeof(CUpti_ActivityMemcpy6); break;
	case CUPTI_ACTIVITY_KIND_DRIVER:
	case CUPTI_ACTIVITY_KIND_RUNTIME: n = sizeof(CUpti_ActivityAPI); break;
	case CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION: n = sizeof(CUpti_ActivityExternalCorrelation); break;
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
	if (request_cb == NULL || complete_cb == NULL) return -1;
	uint8_t *buffer = NULL;
	size_t size = 0, max_records = 0;
	request_cb(&buffer, &size, &max_records);
	if (buffer == NULL || size == 0) return 0;
	size_t offset = 0;
	int placed = 0;
	for (;;) {
		CUpti_ActivityKind kind;
		if (kernels) { kind = CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL; kernels--; }
		else if (memcpys) { kind = CUPTI_ACTIVITY_KIND_MEMCPY; memcpys--; }
		else if (apis) { kind = CUPTI_ACTIVITY_KIND_RUNTIME; apis--; }
		else if (externals) { kind = CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION; externals--; }
		else break;
		if (!kind_enabled(kind)) continue;
		size_t n = record_size(kind);
		if (offset + n > size) break;
		uint8_t *slot = buffer + offset;
		memset(slot, 0, n);
		uint32_t correlation = atomic_fetch_add(&next_correlation, 1) + 1;
		switch (kind) {
		case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
			CUpti_ActivityKernel9 *k = (CUpti_ActivityKernel9 *)slot;
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
	return CUPTI_ERROR_NOT_SUPPORTED;
}
