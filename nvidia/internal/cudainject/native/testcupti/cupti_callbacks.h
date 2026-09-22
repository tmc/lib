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
	CUPTI_DRIVER_TRACE_CBID_cuModuleLoad = 18,
	CUPTI_DRIVER_TRACE_CBID_cuModuleLoadData = 19,
	CUPTI_DRIVER_TRACE_CBID_cuModuleLoadDataEx = 20,
	CUPTI_DRIVER_TRACE_CBID_cuModuleLoadFatBinary = 21,
	CUPTI_DRIVER_TRACE_CBID_cuModuleGetFunction = 23,
	CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel = 307,
	CUPTI_DRIVER_TRACE_CBID_cuProfilerStart = 308,
	CUPTI_DRIVER_TRACE_CBID_cuProfilerStop = 309,
	CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz = 442,
	CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel = 477,
	CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz = 478,
	CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx = 652,
	CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz = 653
} CUpti_driver_api_trace_cbid_subset;

/* The resource callback ids module attribution follows. A module is announced
   here and nowhere else: the activity record of a kernel names the kernel and
   not the image it was loaded from. */
typedef enum {
	CUPTI_CBID_RESOURCE_CONTEXT_CREATED = 1,
	CUPTI_CBID_RESOURCE_MODULE_LOADED = 6
} CUpti_CallbackIdResource;

/* The driver handles module attribution threads together. They are opaque
   pointers to the agent, which never dereferences one: a handle is only ever
   compared against another handle. */
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUstream_st *CUstream;
typedef struct CUlaunchConfig_st CUlaunchConfig;

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

typedef struct {
	CUcontext context;
	union {
		CUstream stream;
	} resourceHandle;
	void *resourceDescriptor;
} CUpti_ResourceData;

typedef struct {
	uint32_t moduleId;
	size_t cubinSize;
	const char *pCubin;
} CUpti_ModuleResourceData;

/* The driver parameter blocks the agent reads, transcribed from CUDA 13.0
   targets/sbsa-linux/include/generated_cuda_meta.h. Only the leading fields
   the agent reads have to match, but the whole block is carried so the next
   field it needs is already right. */
typedef struct {
	CUmodule *module;
	const char *fname;
} cuModuleLoad_params;

typedef struct {
	CUfunction *hfunc;
	CUmodule hmod;
	const char *name;
} cuModuleGetFunction_params;

typedef struct {
	CUfunction f;
	unsigned int gridDimX;
	unsigned int gridDimY;
	unsigned int gridDimZ;
	unsigned int blockDimX;
	unsigned int blockDimY;
	unsigned int blockDimZ;
	unsigned int sharedMemBytes;
	CUstream hStream;
	void **kernelParams;
	void **extra;
} cuLaunchKernel_params;

typedef struct {
	const CUlaunchConfig *config;
	CUfunction f;
	void **kernelParams;
	void **extra;
} cuLaunchKernelEx_params;

CUptiResult cuptiSubscribe(CUpti_SubscriberHandle *subscriber, CUpti_CallbackFunc callback, void *userdata);
CUptiResult cuptiUnsubscribe(CUpti_SubscriberHandle subscriber);
CUptiResult cuptiEnableCallback(uint32_t enable, CUpti_SubscriberHandle subscriber, CUpti_CallbackDomain domain, CUpti_CallbackId cbid);

#endif
