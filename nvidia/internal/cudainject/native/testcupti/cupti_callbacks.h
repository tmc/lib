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
	CUPTI_RUNTIME_TRACE_CBID_cudaGraphKernelNodeSetParams_v10000 = 288,
	CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecKernelNodeSetParams_v10010 = 326,
	CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecUpdate_v10020 = 335,
	CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecChildGraphNodeSetParams_v11010 = 361,
	CUPTI_RUNTIME_TRACE_CBID_cudaGraphNodeSetParams_v12020 = 446,
	CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecNodeSetParams_v12020 = 447,
	CUPTI_RUNTIME_TRACE_CBID_cudaProfilerStart_v4000 = 171,
	CUPTI_RUNTIME_TRACE_CBID_cudaProfilerStop_v4000 = 172,
	CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000 = 211,
	CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000 = 214,
	CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_v9000 = 269,
	CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_ptsz_v9000 = 270,
	CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060 = 430,
	CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_ptsz_v11060 = 431
} CUpti_runtime_api_trace_cbid_subset;

typedef enum {
	CUPTI_DRIVER_TRACE_CBID_cuGraphAddKernelNode = 502,
	CUPTI_DRIVER_TRACE_CBID_cuGraphAddKernelNode_v2 = 689,
	CUPTI_DRIVER_TRACE_CBID_cuGraphExecChildGraphNodeSetParams = 586,
	CUPTI_DRIVER_TRACE_CBID_cuGraphExecKernelNodeSetParams = 538,
	CUPTI_DRIVER_TRACE_CBID_cuGraphExecKernelNodeSetParams_v2 = 692,
	CUPTI_DRIVER_TRACE_CBID_cuGraphExecNodeSetParams = 714,
	CUPTI_DRIVER_TRACE_CBID_cuGraphExecUpdate = 561,
	CUPTI_DRIVER_TRACE_CBID_cuGraphExecUpdate_v2 = 696,
	CUPTI_DRIVER_TRACE_CBID_cuGraphKernelNodeSetParams = 521,
	CUPTI_DRIVER_TRACE_CBID_cuGraphKernelNodeSetParams_v2 = 691,
	CUPTI_DRIVER_TRACE_CBID_cuGraphNodeSetParams = 713,
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
	CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz = 653,
	CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadData = 678,
	CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadFromFile = 679,
	CUPTI_DRIVER_TRACE_CBID_cuLibraryGetKernel = 681,
	CUPTI_DRIVER_TRACE_CBID_cuKernelGetFunction = 683
} CUpti_driver_api_trace_cbid_subset;

/* The resource callback ids module attribution follows. A module is announced
   here and nowhere else: the activity record of a kernel names the kernel and
   not the image it was loaded from. */
typedef enum {
	CUPTI_CBID_RESOURCE_CONTEXT_CREATED = 1,
	CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED = 20,
	CUPTI_CBID_RESOURCE_MODULE_LOADED = 6
} CUpti_CallbackIdResource;

/* The driver handles module attribution threads together. They are opaque
   pointers to the agent, which never dereferences one: a handle is only ever
   compared against another handle. */
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUlib_st *CUlibrary;
typedef struct CUkern_st *CUkernel;
typedef struct CUstream_st *CUstream;
typedef struct CUlaunchConfig_st CUlaunchConfig;
typedef struct cudaLaunchConfig_st cudaLaunchConfig_t;
typedef struct { unsigned int x, y, z; } dim3;
typedef int CUjit_option;
typedef int CUlibraryOption;

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

typedef struct {
	CUlibrary *library;
	const void *code;
	CUjit_option *jitOptions;
	void **jitOptionsValues;
	unsigned int numJitOptions;
	CUlibraryOption *libraryOptions;
	void **libraryOptionValues;
	unsigned int numLibraryOptions;
} cuLibraryLoadData_params;

typedef struct {
	CUkernel *pKernel;
	CUlibrary library;
	const char *name;
} cuLibraryGetKernel_params;

typedef struct {
	CUfunction *pFunc;
	CUkernel kernel;
} cuKernelGetFunction_params;

/* The runtime parameter blocks, transcribed from CUDA 13.4
   include/generated_cuda_runtime_api_meta.h. */
typedef struct {
	const void *func;
	dim3 gridDim;
	dim3 blockDim;
	void **args;
	size_t sharedMem;
	CUstream stream;
} cudaLaunchKernel_v7000_params;

typedef struct {
	const cudaLaunchConfig_t *config;
	const void *func;
	void **args;
} cudaLaunchKernelExC_v11060_params;

CUptiResult cuptiSubscribe(CUpti_SubscriberHandle *subscriber, CUpti_CallbackFunc callback, void *userdata);
CUptiResult cuptiUnsubscribe(CUpti_SubscriberHandle subscriber);
CUptiResult cuptiEnableCallback(uint32_t enable, CUpti_SubscriberHandle subscriber, CUpti_CallbackDomain domain, CUpti_CallbackId cbid);
CUptiResult cuptiEnableDomain(uint32_t enable, CUpti_SubscriberHandle subscriber, CUpti_CallbackDomain domain);

/* Graph identity declarations read by the producer. The stub treats a
   non-null node handle as its stable CUPTI id. */
typedef struct CUgraph_st *CUgraph;
typedef struct CUgraphNode_st *CUgraphNode;
typedef struct CUgraphExec_st *CUgraphExec;
typedef int CUresult;
#define CUDA_SUCCESS 0
typedef struct {
	CUgraph graph;
	CUgraph originalGraph;
	CUgraphNode node;
	CUgraphNode originalNode;
	int nodeType;
	CUgraphNode dependency;
	CUgraphExec graphExec;
} CUpti_GraphData;
typedef struct { CUfunction func; } test_kernel_node_params;
typedef struct {
	CUgraphNode *phGraphNode;
	CUgraph hGraph;
	const CUgraphNode *dependencies;
	size_t numDependencies;
	const test_kernel_node_params *nodeParams;
} cuGraphAddKernelNode_params;
typedef cuGraphAddKernelNode_params cuGraphAddKernelNode_v2_params;
CUptiResult cuptiGetGraphNodeId(CUgraphNode node, uint64_t *id);
#endif
