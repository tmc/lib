/* Optional Unified Memory tracing. UM records describe migration/fault service,
   not device occupancy or current page residency. The documented record layout
   is CUpti_ActivityUnifiedMemoryCounter3 in qualified CUDA 12.9/13.x headers. */
#ifndef CUDAPROF_CUPTI_MEMORY_H
#define CUDAPROF_CUPTI_MEMORY_H

_Static_assert(offsetof(CUpti_ActivityUnifiedMemoryCounter3, value) == 8, "UM value layout");
_Static_assert(offsetof(CUpti_ActivityUnifiedMemoryCounter3, address) == 32, "UM address layout");
_Static_assert(offsetof(CUpti_ActivityUnifiedMemoryCounter3, processors) == 64, "UM processors layout");
_Static_assert(sizeof(CUpti_ActivityUnifiedMemoryCounter3) == 104, "UM record layout");
static int um_requested, um_enabled;
static int memory_policy_requested;
static unsigned memory_policy_enabled;
static int memory_policy_results[4] = {-1,-1,-1,-1};
static int um_enable_result = -1;
static int um_results[4] = {-1, -1, -1, -1};
static const char *um_state = "off";
#if defined(__aarch64__) || defined(_M_ARM64)
static const char *um_architecture_policy = "unsupported_arm";
#else
static const char *um_architecture_policy = "permitted";
#endif
static const char *um_layout_check = "unprobed";
static const CUpti_ActivityUnifiedMemoryCounterKind um_kinds[4] = {
 CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD,
 CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH,
 CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT,
 CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT
};
typedef CUptiResult (*memory_size_query)(CUpti_ActivityKind, uint32_t, size_t *);

static void configure_memory_activity(uint32_t runtime_version, memory_size_query query) {
 const char *policy = getenv("CUDAPROF_MEMORY_POLICY");
 memory_policy_requested = policy != NULL && strcmp(policy,"1")==0;
 const char *setting = getenv("CUDAPROF_UNIFIED_MEMORY");
 um_requested = setting != NULL && strcmp(setting, "1") == 0;
 if (!um_requested) return;
 if (!(runtime_version == 28 || (runtime_version >= 130000 && runtime_version < 130500))) { um_state = "unsupported_version"; return; }
 um_layout_check = "qualified_version_and_buffer_bounds";
 if (query != NULL) {
  size_t size = 0;
  if (query(CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER, 0, &size) != CUPTI_SUCCESS || size < sizeof(CUpti_ActivityUnifiedMemoryCounter3)) { um_state = "unsupported_layout"; return; }
  um_layout_check = "cuptiActivityGetStructSize";
 }
 typedef CUptiResult (*configure_fn)(CUpti_ActivityUnifiedMemoryCounterConfig *, uint32_t);
 configure_fn configure = NULL;
 *(void **)(&configure) = dlsym(RTLD_DEFAULT, "cuptiActivityConfigureUnifiedMemoryCounter");
 if (configure == NULL) { um_state = "configuration_symbol_unavailable"; return; }
 unsigned supported = 0;
 for (unsigned i = 0; i < 4; i++) {
  CUpti_ActivityUnifiedMemoryCounterConfig config = {0};
  config.scope = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_ALL_DEVICES;
  config.kind = um_kinds[i]; config.enable = 1;
  CUptiResult result = configure(&config, 1); um_results[i] = (int)result;
  if (result == CUPTI_SUCCESS) supported++;
 }
 if (supported == 0) { um_state = "configuration_refused"; return; }
#if (defined(__aarch64__) || defined(_M_ARM64)) && !defined(CUDAPROF_TEST_ALLOW_UM)
 /* Record measured configuration results separately from the ARM policy. */
 for (unsigned i = 0; i < 4; i++) if (um_results[i] == 0) {
  CUpti_ActivityUnifiedMemoryCounterConfig config = {0};
  config.scope = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_ALL_DEVICES;
  config.kind = um_kinds[i]; config.enable = 0; (void)configure(&config, 1);
 }
 um_state = "architecture_unsupported"; return;
#endif
 CUptiResult enabled = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER);
 um_enable_result = (int)enabled;
 if (enabled != CUPTI_SUCCESS) { um_state = "activity_enable_refused"; return; }
 um_enabled = 1; um_state = supported == 4 ? "collected" : "collected_partial";
}


static void enable_memory_policy(CUpti_SubscriberHandle subscriber) {
 if (!memory_policy_requested) return;
 const CUpti_CallbackId ids[4] = {CUPTI_DRIVER_TRACE_CBID_cuMemRangeGetAttribute,CUPTI_DRIVER_TRACE_CBID_cuMemAdvise_v2,CUPTI_DRIVER_TRACE_CBID_cuMemPrefetchAsync_v2,CUPTI_DRIVER_TRACE_CBID_cuMemPrefetchAsync_v2_ptsz};
 for (unsigned i=0;i<4;i++) { CUptiResult result=cuptiEnableCallback(1,subscriber,CUPTI_CB_DOMAIN_DRIVER_API,ids[i]); memory_policy_results[i]=(int)result; if(result==CUPTI_SUCCESS)memory_policy_enabled|=1u<<i; }
}

static void memory_policy_callback(CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const CUpti_CallbackData *data) {
 if (!memory_policy_enabled || domain != CUPTI_CB_DOMAIN_DRIVER_API || data->callbackSite != CUPTI_API_EXIT || data->functionParams==NULL) return;
 uint64_t pointer=0,bytes=0;int attribute=0,advice=0,location_type=0,location_id=0;
 const char *event=NULL;int32_t value=0;int have_value=0;
 int known=data->functionReturnValue!=NULL;
 int result=known ? (int)*(const CUresult *)data->functionReturnValue : -1;
 if(cbid==CUPTI_DRIVER_TRACE_CBID_cuMemRangeGetAttribute){
  const cuMemRangeGetAttribute_params *p=data->functionParams;event="attribute";pointer=p->devPtr;bytes=p->count;attribute=(int)p->attribute;
  if(known && result==0 && p->data!=NULL && p->dataSize==sizeof(int32_t) && (attribute==1||attribute==2||attribute==4||attribute==5||attribute==6||attribute==7||attribute==8)){memcpy(&value,p->data,sizeof(value));have_value=1;}
 }else if(cbid==CUPTI_DRIVER_TRACE_CBID_cuMemAdvise_v2){
  const cuMemAdvise_v2_params *p=data->functionParams;event="advice";pointer=p->devPtr;bytes=p->count;advice=(int)p->advice;location_type=(int)p->location.type;location_id=p->location.id;
 }else if(cbid==CUPTI_DRIVER_TRACE_CBID_cuMemPrefetchAsync_v2||cbid==CUPTI_DRIVER_TRACE_CBID_cuMemPrefetchAsync_v2_ptsz){
  const cuMemPrefetchAsync_v2_params *p=data->functionParams;event="prefetch_request";pointer=p->devPtr;bytes=p->count;location_type=(int)p->location.type;location_id=p->location.id;
 }else return;
 char output_value[48];if(have_value)snprintf(output_value,sizeof(output_value),"%" PRId32,value);else snprintf(output_value,sizeof(output_value),"null");
 uint64_t timestamp=now_ns();char payload[MAX_PAYLOAD];
 uint32_t thread=0;
#ifdef __linux__
 thread=(uint32_t)syscall(SYS_gettid);
#endif
 int length=snprintf(payload,sizeof(payload),"{\"kind\":\"memory_policy\",\"policy_event\":\"%s\",\"timestamp_domain\":\"%s\",\"process_id\":%u,\"thread_id\":%u,\"correlation\":%u,\"callback_id\":%u,\"context\":%u,\"start\":%" PRIu64 ",\"end\":%" PRIu64 ",\"policy_pointer\":%" PRIu64 ",\"policy_bytes\":%" PRIu64 ",\"policy_attribute\":%d,\"policy_advice\":%d,\"policy_location_type\":%d,\"policy_location_id\":%d,\"policy_value\":%s,\"policy_result\":%d,\"policy_result_known\":%s}",event,clock_state==AGENT_CLOCK_UNIFIED?"monotonic":"unknown",(unsigned)getpid(),thread,data->correlationId,(unsigned)cbid,data->contextUid,timestamp,timestamp,pointer,bytes,attribute,advice,location_type,location_id,output_value,result,known?"true":"false");
 if(length>0 && (size_t)length<sizeof(payload))(void)queue_pending(KIND_API,timestamp,payload,(size_t)length);
}

/* Extend the existing capability object, preserving the truncation gate. */
static int append_memory_capability(char *buffer, size_t capacity, int length) {
 if (length <= 0 || (size_t)length >= capacity || buffer[length-1] != '}') return length;
 int wrote = snprintf(buffer + length-1, capacity-(size_t)length+1,
  ",\"unified_memory\":{\"requested\":%s,\"state\":\"%s\",\"layout\":\"UnifiedMemoryCounter3\",\"layout_check\":\"%s\",\"architecture_policy\":\"%s\",\"counter_kinds\":[1,2,3,4],\"native_configuration_results\":[%d,%d,%d,%d],\"native_enable_result\":%d,\"residency_observed\":false},\"memory_policy\":{\"requested\":%s,\"enabled_mask\":%u,\"native_callback_results\":[%d,%d,%d,%d],\"residency_observed\":false}}",
  um_requested ? "true" : "false", um_state, um_layout_check, um_architecture_policy, um_results[0], um_results[1], um_results[2], um_results[3],um_enable_result,memory_policy_requested?"true":"false",memory_policy_enabled,memory_policy_results[0],memory_policy_results[1],memory_policy_results[2],memory_policy_results[3]);
 return wrote < 0 ? -1 : length-1+wrote;
}

static int append_unified_memory(const CUpti_ActivityUnifiedMemoryCounter3 *record) {
 unsigned kind = (unsigned)record->counterKind;
 int end_valid = kind == 1 || kind == 2 || kind == 4 || kind == 6 || kind == 8;
 int source_valid = kind == 1 || kind == 2 || kind == 4 || kind == 5 || kind == 6 || kind == 7 || kind == 8;
 int destination_valid = kind == 1 || kind == 2 || kind == 7 || kind == 8;
 uint64_t end = end_valid ? record->end : record->start;
 int valid = record->start != CUPTI_TIMESTAMP_UNKNOWN && (!end_valid || (record->end != CUPTI_TIMESTAMP_UNKNOWN && record->end >= record->start));
 char payload[MAX_PAYLOAD];
 int length = snprintf(payload, sizeof(payload),
  "{\"kind\":\"activity\",\"subtype\":\"unified_memory\",\"valid\":%s,\"start\":%" PRIu64 ",\"end\":%" PRIu64 ",\"um_raw_end\":%" PRIu64 ",\"um_end_valid\":%s,\"um_counter_kind\":%u,\"um_value\":%" PRIu64 ",\"um_address\":%" PRIu64 ",\"um_src_id\":%u,\"um_dst_id\":%u,\"um_src_valid\":%s,\"um_dst_valid\":%s,\"um_raw_stream\":%u,\"um_flags\":%u,\"process_id\":%u}",
  valid ? "true" : "false",record->start,end,record->end,end_valid ? "true" : "false",kind,record->value,record->address,record->srcId,record->dstId,source_valid ? "true" : "false",destination_valid ? "true" : "false",record->streamId,record->flags,record->processId);
 if (length < 0 || (size_t)length >= sizeof(payload)) return payload_overflow("unified memory");
 return queue_pending(KIND_ACTIVITY, record->start, payload, (size_t)length);
}
#endif
