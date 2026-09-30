/* Observed driver callback receipts. Numeric callback IDs below are stable
   CUPTI ABI IDs from cupti_driver_cbid.h, checked by the production headers.
   No CUDA API is called from the callback. Concurrent submissions are retained
   as intervals of observation ordinals, not ordered by timestamp proximity. */
static int dependency_requested;
static int dependency_enabled;
static _Atomic uint64_t dependency_ordinal;

#ifndef CUDAPROF_TESTCUPTI_CALLBACKS_H
_Static_assert(CUPTI_DRIVER_TRACE_CBID_cuCtxSynchronize == 17 &&
 CUPTI_DRIVER_TRACE_CBID_cuEventCreate == 118 &&
 CUPTI_DRIVER_TRACE_CBID_cuEventRecord == 119 &&
 CUPTI_DRIVER_TRACE_CBID_cuStreamCreate == 124 &&
 CUPTI_DRIVER_TRACE_CBID_cuStreamWaitEvent == 295 &&
 CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel == 307 &&
 CUPTI_DRIVER_TRACE_CBID_cuGraphAddKernelNode == 502 &&
 CUPTI_DRIVER_TRACE_CBID_cuGraphAddKernelNode_v2 == 689,
 "dependency callback ABI differs from supported CUDA headers");
_Static_assert(sizeof(cuGraphAddKernelNode_params) == sizeof(cuGraphAddKernelNode_v2_params),
 "graph-node callback parameter layout differs");
#endif

static int dependency_graph_nodes(char *out, size_t capacity, CUgraphNode handle, const CUgraphNode *deps, size_t count, uint64_t *node) {
 if(!handle || cuptiGetGraphNodeId(handle,node)!=CUPTI_SUCCESS || *node==0 || count>128 || (count && !deps)) return -1;
 size_t used=1;out[0]='[';
 for(size_t i=0;i<count;i++) {
  uint64_t id=0;
  if(cuptiGetGraphNodeId(deps[i],&id)!=CUPTI_SUCCESS || id==0) return -1;
  int n=snprintf(out+used,capacity-used,"%s%" PRIu64,i ? "," : "",id);
  if(n<0 || (size_t)n>=capacity-used-2) return -1;
  used+=(size_t)n;
 }
 out[used++]=']';out[used]=0;return 0;
}

static void dependency_callback(CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const CUpti_CallbackData *data) {
 if (!dependency_enabled || domain != CUPTI_CB_DOMAIN_DRIVER_API || !data || (!data->functionParams && cbid != 17 && cbid != 800)) return;
 const void *params=data->functionParams;
 const char *event=NULL;
 uint64_t stream=0, handle=0, graph=0, exec=0, node=0, node_handle=0, child=0;
 char dependencies[2800]="[]";
 unsigned flags=0;
 int exit=data->callbackSite == CUPTI_API_EXIT;
 int known=exit && data->functionReturnValue != NULL;
 int success=known && *(const CUresult *)data->functionReturnValue == 0;
 switch (cbid) {
 case 502: case 689: {
  const cuGraphAddKernelNode_params *p=params;event="graph_node";graph=(uintptr_t)p->hGraph;
  if(success) {
   if(p->phGraphNode) node_handle=(uintptr_t)*p->phGraphNode;
   if(!p->phGraphNode || dependency_graph_nodes(dependencies,sizeof(dependencies),*p->phGraphNode,p->dependencies,p->numDependencies,&node)!=0) {event="unsupported";strcpy(dependencies,"[]");}
  }
  break;
 }
 case 525: {
  const cuGraphAddChildGraphNode_params *p=params;event="graph_child";graph=(uintptr_t)p->hGraph;child=(uintptr_t)p->childGraph;
  if(success) {
   if(p->phGraphNode) node_handle=(uintptr_t)*p->phGraphNode;
   if(!p->phGraphNode || dependency_graph_nodes(dependencies,sizeof(dependencies),*p->phGraphNode,p->dependencies,p->numDependencies,&node)!=0) {event="unsupported";strcpy(dependencies,"[]");}
  }
  break;
 }
 case 501: case 513: case 578: case 643: case 656: case 657: case 516: case 517: case 523: case 529: event="graph_metadata"; break;
 case 118: { const cuEventCreate_params *p=params; event="create_event"; flags=p->Flags; if(success && p->phEvent) handle=(uintptr_t)*p->phEvent; break; }
 case 119: case 441: { const cuEventRecord_params *p=params; event="event_record"; handle=(uintptr_t)p->hEvent; stream=(uintptr_t)p->hStream; break; }
 case 587: case 588: { const cuEventRecordWithFlags_params *p=params; event="event_record"; handle=(uintptr_t)p->hEvent; stream=(uintptr_t)p->hStream; flags=p->flags; break; }
 case 121: { const cuEventSynchronize_params *p=params; event="event_sync"; handle=(uintptr_t)p->hEvent; break; }
 case 325: { const cuEventDestroy_v2_params *p=params; event="destroy_event"; handle=(uintptr_t)p->hEvent; break; }
 case 124: { const cuStreamCreate_params *p=params; event="create_stream"; flags=p->Flags; if(success && p->phStream) stream=(uintptr_t)*p->phStream; break; }
 case 367: { const cuStreamCreateWithPriority_params *p=params; event="create_stream"; flags=p->flags; if(success && p->phStream) stream=(uintptr_t)*p->phStream; break; }
 case 126: case 440: { const cuStreamSynchronize_params *p=params; event="stream_sync"; stream=(uintptr_t)p->hStream; break; }
 case 326: { const cuStreamDestroy_v2_params *p=params; event="destroy_stream"; stream=(uintptr_t)p->hStream; break; }
 case 295: case 436: { const cuStreamWaitEvent_params *p=params; event="event_wait"; stream=(uintptr_t)p->hStream; handle=(uintptr_t)p->hEvent; flags=p->Flags; break; }
 case 307: case 442: { const cuLaunchKernel_params *p=params; event="launch"; stream=(uintptr_t)p->hStream; break; }
 case 514: { const cuGraphLaunch_params *p=params; event="graph_launch"; stream=(uintptr_t)p->hStream; exec=(uintptr_t)p->hGraph; break; }
 case 515: { const cuGraphLaunch_ptsz_params *p=params; event="graph_launch"; stream=(uintptr_t)p->hStream; exec=(uintptr_t)p->hGraphExec; break; }
 case 279: case 422: { const cuMemcpyDtoHAsync_v2_params *p=params; event="copy_to_host"; stream=(uintptr_t)p->hStream; break; }
 case 17: case 800: event="context_sync"; break;
 default:
  /* Unknown work-affecting calls are evidence of missing coverage, including
     capture, graph mutation, launch variants and other copy/memset forms. */
  if (data->functionName && (strncmp(data->functionName,"cuLaunch",8)==0 || strncmp(data->functionName,"cuMemcpy",8)==0 || strncmp(data->functionName,"cuMemset",8)==0 || strncmp(data->functionName,"cuGraph",7)==0 || strncmp(data->functionName,"cuStream",8)==0 || strncmp(data->functionName,"cuEvent",7)==0)) event="unsupported";
  else return;
 }
 uint64_t ordinal=atomic_fetch_add(&dependency_ordinal,1)+1;
 uint64_t ts=now_ns();
 char payload[MAX_PAYLOAD];
 int n=snprintf(payload,sizeof(payload),"{\"kind\":\"dependency\",\"domain\":\"driver\",\"dependency_event\":\"%s\",\"site\":\"%s\",\"ordinal\":%" PRIu64 ",\"timestamp_domain\":\"%s\",\"callback_id\":%u,\"correlation\":%u,\"context\":%u,\"stream_handle\":%" PRIu64 ",\"event_handle\":%" PRIu64 ",\"flags\":%u,\"graph_handle\":%" PRIu64 ",\"executable_handle\":%" PRIu64 ",\"graph_node_id\":%" PRIu64 ",\"node_handle\":%" PRIu64 ",\"child_graph_handle\":%" PRIu64 ",\"dependencies\":%s,\"start\":%" PRIu64 ",\"end\":%" PRIu64 ",\"success\":%s}",event,exit ? "exit" : "enter",ordinal,clock_state == AGENT_CLOCK_UNIFIED ? "monotonic" : "unknown",(unsigned)cbid,data->correlationId,data->contextUid,stream,handle,flags,graph,exec,node,node_handle,child,dependencies,ts,ts,known ? (success ? "true" : "false") : "null");
 if(n>0 && (size_t)n<sizeof(payload)) (void)queue_pending(KIND_API,ts,payload,(size_t)n);
}

static int append_dependency_capability(char *dst, size_t cap, int len) {
 if (len<=0 || (size_t)len>=cap || dst[len-1]!='}') return len;
 int n=snprintf(dst+len-1,cap-(size_t)len+1,",\"dependencies\":\"%s\",\"dependency_coverage\":\"driver_partial\"}",dependency_enabled ? "collected" : dependency_requested ? "requested_unavailable" : "off");
 return n<0 ? n : len-1+n;
}
