/* Controlled driver-only delivery graph: two graph nodes, an independent
   second-stream kernel, an event wait joining both branches, a combine kernel,
   asynchronous host copy, successful stream synchronization and delivery.
   NVTX markers run on the actual submitting thread. Go/Python subprocess
   wrappers do not measure differences between inference engines. */
#include <cuda.h>
#include <cudaProfiler.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int (*push)(const char *);
static int (*pop)(void);
static void (*mark)(const char *);

static void check(CUresult result,const char *what) {
 if(result==CUDA_SUCCESS) return;
 const char *text=NULL;cuGetErrorString(result,&text);
 fprintf(stderr,"dependency: %s: %s\n",what,text ? text : "unknown CUDA error");exit(1);
}

static void range(const char *operation,int materialization) {
 char message[256];
 snprintf(message,sizeof(message),"cudaprof:v1 {\"phase\":\"decode\",\"token\":0,\"layer\":0,\"operation\":\"%s\",\"dtype\":\"int64\",\"cache_length\":0,\"lookahead\":false,\"materialization\":%s}",operation,materialization ? "true" : "false");
 push(message);
}

int main(int argc,char **argv) {
 if(argc!=2) {fprintf(stderr,"usage: dependency dependency_kernel.ptx\n");return 2;}
 void *nvtx=dlopen("libnvToolsExt.so.1",RTLD_NOW|RTLD_LOCAL);
 if(!nvtx) {fprintf(stderr,"dependency: requested NVTX provider unavailable: %s\n",dlerror());return 1;}
 *(void **)(&push)=dlsym(nvtx,"nvtxRangePushA");*(void **)(&pop)=dlsym(nvtx,"nvtxRangePop");*(void **)(&mark)=dlsym(nvtx,"nvtxMarkA");
 if(!push || !pop || !mark) {fprintf(stderr,"dependency: NVTX range or mark entry point unavailable\n");return 1;}
 check(cuInit(0),"initialize");
 CUdevice device;check(cuDeviceGet(&device,0),"device");CUcontext context;
#if CUDA_VERSION >= 13000
 check(cuCtxCreate(&context,NULL,0,device),"context");
#else
 check(cuCtxCreate(&context,0,device),"context");
#endif
 CUmodule module;check(cuModuleLoad(&module,argv[1]),"load kernel module");
 CUfunction write,doubled,combine;check(cuModuleGetFunction(&write,module,"dependency_write"),"write function");check(cuModuleGetFunction(&doubled,module,"dependency_double"),"double function");check(cuModuleGetFunction(&combine,module,"dependency_combine"),"combine function");
 CUdeviceptr output;check(cuMemAlloc(&output,4*sizeof(long long)),"device output");
 long long *host;check(cuMemAllocHost((void **)&host,4*sizeof(long long)),"host output");
 CUstream a,b;check(cuStreamCreate(&a,CU_STREAM_NON_BLOCKING),"stream a");check(cuStreamCreate(&b,CU_STREAM_NON_BLOCKING),"stream b");CUevent event;check(cuEventCreate(&event,CU_EVENT_DISABLE_TIMING),"event");
 CUgraph graph;CUgraphExec executable;CUgraphNode first,second;
 check(cuGraphCreate(&graph,0),"graph");int index=0;long long value=11;void *write_args[]={&output,&index,&value};void *output_args[]={&output};
 CUDA_KERNEL_NODE_PARAMS node;memset(&node,0,sizeof(node));node.func=write;node.gridDimX=node.gridDimY=node.gridDimZ=1;node.blockDimX=node.blockDimY=node.blockDimZ=1;node.kernelParams=write_args;
 check(cuGraphAddKernelNode(&first,graph,NULL,0,&node),"graph first node");node.func=doubled;node.kernelParams=output_args;check(cuGraphAddKernelNode(&second,graph,&first,1,&node),"graph dependent node");check(cuGraphInstantiateWithFlags(&executable,graph,0),"instantiate graph");
 check(cuProfilerStart(),"profiler start");range("token",0);
 range("graph_replay",0);check(cuGraphLaunch(executable,a),"replay graph");pop();
 index=2;value=33;range("parallel",0);check(cuLaunchKernel(write,1,1,1,1,1,1,0,b,write_args,NULL),"parallel kernel");pop();
 check(cuEventRecord(event,a),"record graph completion");check(cuStreamWaitEvent(b,event,0),"join graph branch");
 range("combine",0);check(cuLaunchKernel(combine,1,1,1,1,1,1,0,b,output_args,NULL),"combine kernel");pop();
 range("materialize",1);check(cuMemcpyDtoHAsync(host,output,4*sizeof(long long),b),"host output copy");check(cuStreamSynchronize(b),"materialization synchronization");
 if(host[0]!=11 || host[1]!=22 || host[2]!=33 || host[3]!=66) {fprintf(stderr,"dependency: independent output invariant failed\n");return 1;}
 pop();pop();mark("cudaprof:v1 {\"phase\":\"decode\",\"token\":0,\"event\":\"delivered\"}");mark("cudaprof:v1 {\"phase\":\"decode\",\"token\":0,\"event\":\"token_limit\"}");check(cuProfilerStop(),"profiler stop");
 printf("{\"workload\":\"controlled_driver_dependency_v1\",\"expected_kernel_records\":4,\"expected_memcpy_records\":1,\"expected_device_records\":5,\"expected_graph_nodes\":2,\"expected_delivered_events\":1,\"expected_terminal_events\":1,\"output\":[11,22,33,66]}\n");
 check(cuGraphExecDestroy(executable),"destroy executable");check(cuGraphDestroy(graph),"destroy graph");check(cuEventDestroy(event),"destroy event");check(cuStreamDestroy(a),"destroy stream a");check(cuStreamDestroy(b),"destroy stream b");check(cuMemFree(output),"free device output");check(cuMemFreeHost(host),"free host output");check(cuModuleUnload(module),"unload module");check(cuCtxDestroy(context),"destroy context");return 0;
}
