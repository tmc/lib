#ifndef CUDAPROF_TESTCUPTI_ACTIVITY_H
#define CUDAPROF_TESTCUPTI_ACTIVITY_H

#include <stddef.h>
#include <stdint.h>

#define CUPTIAPI
#define CUPTI_TIMESTAMP_UNKNOWN ((uint64_t)0)

typedef struct CUctx_st *CUcontext;

typedef enum {
	CUPTI_SUCCESS = 0,
	CUPTI_ERROR_NOT_SUPPORTED = 1,
	CUPTI_ERROR_INVALID_PARAMETER = 2,
	CUPTI_ERROR_MAX_LIMIT_REACHED = 3,
	CUPTI_ERROR_UNKNOWN = 999
} CUptiResult;

typedef enum {
	CUPTI_ACTIVITY_KIND_INVALID = 0,
	CUPTI_ACTIVITY_KIND_MEMCPY = 1,
	CUPTI_ACTIVITY_KIND_KERNEL = 3,
	CUPTI_ACTIVITY_KIND_DRIVER = 4,
	CUPTI_ACTIVITY_KIND_RUNTIME = 5,
	CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL = 6,
	CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION = 7
} CUpti_ActivityKind;

typedef enum {
	CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN = 0,
	CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0 = 1
} CUpti_ExternalCorrelationKind;

typedef struct { CUpti_ActivityKind kind; } CUpti_Activity;

typedef struct {
	CUpti_ActivityKind kind;
	const char *name;
	uint64_t start, end, queued, submitted;
	uint32_t deviceId, contextId, streamId, correlationId;
	int32_t gridX, gridY, gridZ;
	int32_t blockX, blockY, blockZ;
	uint16_t registersPerThread;
	int32_t dynamicSharedMemory;
} CUpti_ActivityKernel9;

typedef CUpti_ActivityKernel9 CUpti_ActivityKernel10;

typedef struct {
	CUpti_ActivityKind kind;
	uint64_t start, end, bytes;
	uint8_t copyKind;
	uint32_t runtimeCorrelationId, deviceId, contextId, streamId, correlationId;
} CUpti_ActivityMemcpy6;

typedef struct {
	CUpti_ActivityKind kind;
	uint32_t cbid;
	uint64_t start, end;
	uint32_t processId, threadId, correlationId, returnValue;
} CUpti_ActivityAPI;

/* cupti_agent.c static-asserts this layout, so the stub must match the real
   one byte for byte. */
typedef struct {
	CUpti_ActivityKind kind;
	uint32_t externalKind;
	uint64_t externalId;
	uint32_t correlationId;
	uint32_t reserved;
} CUpti_ActivityExternalCorrelation;

typedef void (CUPTIAPI *CUpti_BuffersCallbackRequestFunc)(uint8_t **buffer, size_t *size, size_t *maxNumRecords);
typedef void (CUPTIAPI *CUpti_BuffersCallbackCompleteFunc)(CUcontext context, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize);

CUptiResult cuptiActivityRegisterCallbacks(CUpti_BuffersCallbackRequestFunc, CUpti_BuffersCallbackCompleteFunc);
CUptiResult cuptiActivityEnable(CUpti_ActivityKind);
CUptiResult cuptiActivityDisable(CUpti_ActivityKind);
CUptiResult cuptiActivityFlushAll(uint32_t flag);
CUptiResult cuptiActivityGetNextRecord(uint8_t *buffer, size_t validSize, CUpti_Activity **record);
CUptiResult cuptiActivityGetNumDroppedRecords(CUcontext, uint32_t streamId, size_t *dropped);
CUptiResult cuptiActivityPushExternalCorrelationId(CUpti_ExternalCorrelationKind, uint64_t id);
CUptiResult cuptiActivityPopExternalCorrelationId(CUpti_ExternalCorrelationKind, uint64_t *id);

#endif
