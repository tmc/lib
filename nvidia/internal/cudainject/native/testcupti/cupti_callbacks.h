#ifndef CUDAPROF_TESTCUPTI_CALLBACKS_H
#define CUDAPROF_TESTCUPTI_CALLBACKS_H

#include "cupti_activity.h"

/* Declarations transcribed from CUDA 13.0
   targets/sbsa-linux/include/cupti_callbacks.h. cupti_agent.c static-asserts
   the CUpti_CallbackData layout, so this must match the real header byte for
   byte; the enumerator values below are additionally checked against the
   generated Go bindings in nvidia/cupti/enums.gen.go, which carry the same
   numbers, and are unchanged across CUDA 12.6 through 13.1. */

typedef enum {
	CUPTI_API_ENTER = 0,
	CUPTI_API_EXIT = 1,
	CUPTI_API_CBSITE_FORCE_INT = 0x7fffffff
} CUpti_ApiCallbackSite;

typedef enum {
	CUPTI_CB_DOMAIN_INVALID = 0,
	CUPTI_CB_DOMAIN_DRIVER_API = 1,
	CUPTI_CB_DOMAIN_RUNTIME_API = 2,
	CUPTI_CB_DOMAIN_RESOURCE = 3,
	CUPTI_CB_DOMAIN_SYNCHRONIZE = 4,
	CUPTI_CB_DOMAIN_NVTX = 5,
	CUPTI_CB_DOMAIN_STATE = 6,
	CUPTI_CB_DOMAIN_SIZE,
	CUPTI_CB_DOMAIN_FORCE_INT = 0x7fffffff
} CUpti_CallbackDomain;

/* The four callback ids capture-range gating triggers on. The runtime pair
   covers cudaProfilerStart/Stop; the driver pair covers a target that calls
   cuProfilerStart/Stop directly, which the runtime pair never sees. */
typedef enum {
	CUPTI_RUNTIME_TRACE_CBID_cudaProfilerStart_v4000 = 171,
	CUPTI_RUNTIME_TRACE_CBID_cudaProfilerStop_v4000 = 172
} CUpti_runtime_api_trace_cbid_subset;

typedef enum {
	CUPTI_DRIVER_TRACE_CBID_cuProfilerStart = 308,
	CUPTI_DRIVER_TRACE_CBID_cuProfilerStop = 309
} CUpti_driver_api_trace_cbid_subset;

typedef uint32_t CUpti_CallbackId;
typedef struct CUpti_Subscriber_st *CUpti_SubscriberHandle;

typedef void (*CUpti_CallbackFunc)(void *userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void *cbdata);

typedef struct {
	CUpti_ApiCallbackSite callbackSite;
	const char *functionName;
	const void *functionParams;
	void *functionReturnValue;
	const char *symbolName;
	CUcontext context;
	uint32_t contextUid;
	uint64_t *correlationData;
	uint32_t correlationId;
} CUpti_CallbackData;

CUptiResult cuptiSubscribe(CUpti_SubscriberHandle *subscriber, CUpti_CallbackFunc callback, void *userdata);
CUptiResult cuptiUnsubscribe(CUpti_SubscriberHandle subscriber);
CUptiResult cuptiEnableCallback(uint32_t enable, CUpti_SubscriberHandle subscriber, CUpti_CallbackDomain domain, CUpti_CallbackId cbid);

#endif
