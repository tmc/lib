#ifndef CUDAPROF_TESTCUPTI_PCSAMPLING_H
#define CUDAPROF_TESTCUPTI_PCSAMPLING_H

/* The PC sampling surface the producer compiles against, in the layout CUPTI
   publishes it. The harness does not sample: every entry point below reports
   the API unavailable, which is the state the producer must survive, because
   it is also the state of every machine whose driver declines PC sampling. */

#include <stddef.h>
#include <stdint.h>

#include "cupti_activity.h"

#define CUPTI_STALL_REASON_STRING_SIZE 128

#define CUPTI_PCSAMPLING_STRUCT_SIZE(type_, lastfield_) (offsetof(type_, lastfield_) + sizeof(((type_ *)0)->lastfield_))

typedef enum {
	CUPTI_PC_SAMPLING_COLLECTION_MODE_INVALID = 0,
	CUPTI_PC_SAMPLING_COLLECTION_MODE_CONTINUOUS = 1,
	CUPTI_PC_SAMPLING_COLLECTION_MODE_KERNEL_SERIALIZED = 2
} CUpti_PCSamplingCollectionMode;

typedef enum {
	CUPTI_PC_SAMPLING_OUTPUT_DATA_FORMAT_INVALID = 0,
	CUPTI_PC_SAMPLING_OUTPUT_DATA_FORMAT_PARSED = 1
} CUpti_PCSamplingOutputDataFormat;

typedef struct {
	uint32_t pcSamplingStallReasonIndex;
	uint32_t samples;
} CUpti_PCSamplingStallReason;

typedef struct {
	size_t size;
	uint64_t cubinCrc;
	uint64_t pcOffset;
	uint32_t functionIndex;
	uint32_t pad;
	char *functionName;
	size_t stallReasonCount;
	CUpti_PCSamplingStallReason *stallReason;
	uint32_t correlationId;
} CUpti_PCSamplingPCData;

typedef struct {
	size_t size;
	size_t collectNumPcs;
	uint64_t totalSamples;
	uint64_t droppedSamples;
	size_t totalNumPcs;
	size_t remainingNumPcs;
	uint64_t rangeId;
	CUpti_PCSamplingPCData *pPcData;
	uint64_t nonUsrKernelsTotalSamples;
	uint8_t hardwareBufferFull;
} CUpti_PCSamplingData;

typedef enum {
	CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_INVALID = 0,
	CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_PERIOD = 1,
	CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_STALL_REASON = 2,
	CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SCRATCH_BUFFER_SIZE = 3,
	CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_HARDWARE_BUFFER_SIZE = 4,
	CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_COLLECTION_MODE = 5,
	CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_ENABLE_START_STOP_CONTROL = 6,
	CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_OUTPUT_DATA_FORMAT = 7,
	CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_DATA_BUFFER = 8,
	CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_WORKER_THREAD_PERIODIC_SLEEP_SPAN = 9
} CUpti_PCSamplingConfigurationAttributeType;

typedef struct {
	CUpti_PCSamplingConfigurationAttributeType attributeType;
	CUptiResult attributeStatus;
	union {
		struct {
			uint64_t data[3];
		} invalidData;
		struct {
			uint32_t samplingPeriod;
		} samplingPeriodData;
		struct {
			size_t stallReasonCount;
			uint32_t *pStallReasonIndex;
		} stallReasonData;
		struct {
			size_t scratchBufferSize;
		} scratchBufferSizeData;
		struct {
			size_t hardwareBufferSize;
		} hardwareBufferSizeData;
		struct {
			CUpti_PCSamplingCollectionMode collectionMode;
		} collectionModeData;
		struct {
			uint32_t enableStartStopControl;
		} enableStartStopControlData;
		struct {
			CUpti_PCSamplingOutputDataFormat outputDataFormat;
		} outputDataFormatData;
		struct {
			void *samplingDataBuffer;
		} samplingDataBufferData;
		struct {
			uint32_t workerThreadPeriodicSleepSpan;
		} workerThreadPeriodicSleepSpanData;
	} attributeData;
} CUpti_PCSamplingConfigurationInfo;

typedef struct {
	size_t size;
	void *pPriv;
	CUcontext ctx;
	size_t numAttributes;
	CUpti_PCSamplingConfigurationInfo *pPCSamplingConfigurationInfo;
} CUpti_PCSamplingConfigurationInfoParams;
#define CUpti_PCSamplingConfigurationInfoParamsSize CUPTI_PCSAMPLING_STRUCT_SIZE(CUpti_PCSamplingConfigurationInfoParams, pPCSamplingConfigurationInfo)

typedef struct {
	size_t size;
	void *pPriv;
	CUcontext ctx;
	void *pcSamplingData;
} CUpti_PCSamplingGetDataParams;
#define CUpti_PCSamplingGetDataParamsSize CUPTI_PCSAMPLING_STRUCT_SIZE(CUpti_PCSamplingGetDataParams, pcSamplingData)

typedef struct {
	size_t size;
	void *pPriv;
	CUcontext ctx;
} CUpti_PCSamplingEnableParams;
#define CUpti_PCSamplingEnableParamsSize CUPTI_PCSAMPLING_STRUCT_SIZE(CUpti_PCSamplingEnableParams, ctx)

typedef struct {
	size_t size;
	void *pPriv;
	CUcontext ctx;
} CUpti_PCSamplingDisableParams;
#define CUpti_PCSamplingDisableParamsSize CUPTI_PCSAMPLING_STRUCT_SIZE(CUpti_PCSamplingDisableParams, ctx)

typedef struct {
	size_t size;
	void *pPriv;
	CUcontext ctx;
} CUpti_PCSamplingStartParams;
#define CUpti_PCSamplingStartParamsSize CUPTI_PCSAMPLING_STRUCT_SIZE(CUpti_PCSamplingStartParams, ctx)

typedef struct {
	size_t size;
	void *pPriv;
	CUcontext ctx;
} CUpti_PCSamplingStopParams;
#define CUpti_PCSamplingStopParamsSize CUPTI_PCSAMPLING_STRUCT_SIZE(CUpti_PCSamplingStopParams, ctx)

typedef struct {
	size_t size;
	void *pPriv;
	CUcontext ctx;
	size_t *numStallReasons;
} CUpti_PCSamplingGetNumStallReasonsParams;
#define CUpti_PCSamplingGetNumStallReasonsParamsSize CUPTI_PCSAMPLING_STRUCT_SIZE(CUpti_PCSamplingGetNumStallReasonsParams, numStallReasons)

typedef struct {
	size_t size;
	void *pPriv;
	CUcontext ctx;
	size_t numStallReasons;
	uint32_t *stallReasonIndex;
	char **stallReasons;
} CUpti_PCSamplingGetStallReasonsParams;
#define CUpti_PCSamplingGetStallReasonsParamsSize CUPTI_PCSAMPLING_STRUCT_SIZE(CUpti_PCSamplingGetStallReasonsParams, stallReasons)

typedef struct {
	size_t size;
	size_t cubinSize;
	const void *cubin;
	uint64_t cubinCrc;
} CUpti_GetCubinCrcParams;
#define CUpti_GetCubinCrcParamsSize CUPTI_PCSAMPLING_STRUCT_SIZE(CUpti_GetCubinCrcParams, cubinCrc)

CUptiResult CUPTIAPI cuptiPCSamplingEnable(CUpti_PCSamplingEnableParams *pParams);
CUptiResult CUPTIAPI cuptiPCSamplingDisable(CUpti_PCSamplingDisableParams *pParams);
CUptiResult CUPTIAPI cuptiPCSamplingStart(CUpti_PCSamplingStartParams *pParams);
CUptiResult CUPTIAPI cuptiPCSamplingStop(CUpti_PCSamplingStopParams *pParams);
CUptiResult CUPTIAPI cuptiPCSamplingGetData(CUpti_PCSamplingGetDataParams *pParams);
CUptiResult CUPTIAPI cuptiPCSamplingSetConfigurationAttribute(CUpti_PCSamplingConfigurationInfoParams *pParams);
CUptiResult CUPTIAPI cuptiPCSamplingGetNumStallReasons(CUpti_PCSamplingGetNumStallReasonsParams *pParams);
CUptiResult CUPTIAPI cuptiPCSamplingGetStallReasons(CUpti_PCSamplingGetStallReasonsParams *pParams);
CUptiResult CUPTIAPI cuptiGetCubinCrc(CUpti_GetCubinCrcParams *pParams);

#endif
