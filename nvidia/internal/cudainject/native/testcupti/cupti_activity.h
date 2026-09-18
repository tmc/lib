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
	CUPTI_ACTIVITY_KIND_MEMSET = 2,
	CUPTI_ACTIVITY_KIND_KERNEL = 3,
	CUPTI_ACTIVITY_KIND_DRIVER = 4,
	CUPTI_ACTIVITY_KIND_RUNTIME = 5,
	CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL = 6,
	CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION = 7,
	CUPTI_ACTIVITY_KIND_MARKER = 12,
	/* A kind the producer has no decode branch for, and never will. CUPTI's
	   own kinds are a dense enum -- CUDA 13 reaches the low nineties -- so a
	   value five orders of magnitude above the top of that range cannot
	   collide with a real kind this agent later learns to decode. That
	   non-collision is the point: the control has to stay a control after
	   MEMSET, SYNCHRONIZATION and whatever follows them are all handled. */
	CUPTI_ACTIVITY_KIND_TEST_UNDECODED = 9000000
} CUpti_ActivityKind;

/* NVTX reaches CUPTI as three separate marker records rather than as one
   range: a range opens with a START record and closes with an END record
   carrying the same id, and a point event arrives alone as INSTANTANEOUS.
   Nothing in a single record says how long anything took. */
typedef enum {
	CUPTI_ACTIVITY_FLAG_NONE = 0,
	CUPTI_ACTIVITY_FLAG_MARKER_INSTANTANEOUS = 1 << 0,
	CUPTI_ACTIVITY_FLAG_MARKER_START = 1 << 1,
	CUPTI_ACTIVITY_FLAG_MARKER_END = 1 << 2
} CUpti_ActivityFlag;

typedef enum {
	CUPTI_ACTIVITY_OBJECT_UNKNOWN = 0,
	CUPTI_ACTIVITY_OBJECT_PROCESS = 1,
	CUPTI_ACTIVITY_OBJECT_THREAD = 2,
	CUPTI_ACTIVITY_OBJECT_DEVICE = 3,
	CUPTI_ACTIVITY_OBJECT_CONTEXT = 4,
	CUPTI_ACTIVITY_OBJECT_STREAM = 5
} CUpti_ActivityObjectKind;

typedef union {
	struct { uint32_t processId, threadId; } pt;
	struct { uint32_t deviceId, contextId, streamId; } dcs;
} CUpti_ActivityObjectKindId;

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

typedef struct {
	CUpti_ActivityKind kind;
	const char *name;
	uint64_t start, end, queued, submitted;
	uint32_t deviceId, contextId, streamId, correlationId;
	int32_t gridX, gridY, gridZ;
	int32_t blockX, blockY, blockZ;
	uint16_t registersPerThread;
	int32_t dynamicSharedMemory;
	uint32_t graphId;
	uint64_t graphNodeId;
} CUpti_ActivityKernel10;

typedef struct {
	CUpti_ActivityKind kind;
	uint64_t start, end, bytes;
	uint8_t copyKind;
	uint32_t runtimeCorrelationId, deviceId, contextId, streamId, correlationId;
} CUpti_ActivityMemcpy6;

/* Only the prefix cupti_agent.c reads. The real CUpti_ActivityMemset4 carries
   flags, memoryKind, graph attribution and channel fields after these; the
   agent reads none of them, and a stub that invented their offsets would be
   asserting a layout nothing here checks. */
typedef struct {
	CUpti_ActivityKind kind;
	uint32_t value;
	uint64_t bytes, start, end;
	uint32_t deviceId, contextId, streamId, correlationId;
} CUpti_ActivityMemset4;

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

/* cupti_agent.c static-asserts this layout, so the stub must match the real
   one byte for byte. */
typedef struct {
	CUpti_ActivityKind kind;
	CUpti_ActivityFlag flags;
	uint64_t timestamp;
	uint32_t id;
	CUpti_ActivityObjectKind objectKind;
	CUpti_ActivityObjectKindId objectId;
	uint32_t pad;
	const char *name;
	const char *domain;
} CUpti_ActivityMarker2;

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
