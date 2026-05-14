// Package cupti provides low-level Go bindings for CUPTI, the CUDA Profiling
// Tools Interface.
//
// CUPTI is the library NVIDIA profilers are built on. It offers several
// largely independent APIs:
//
//   - Activity: asynchronous records of GPU and CPU activity, delivered
//     into buffers the caller supplies ([ActivityEnable],
//     [ActivityRegisterCallbacks], [ActivityFlushAll]).
//   - Callback: synchronous callbacks at driver, runtime, resource and
//     synchronization events ([Subscribe], [Subscriber.EnableDomain]).
//   - Event and Metric: hardware counter collection on pre-Volta style
//     interfaces ([NewEventGroup], [MetricGetValue]).
//   - Profiler and Range Profiler: the current counter collection path
//     ([ProfilerInitialize], [ProfilerBeginPass]).
//   - PC Sampling, PM Sampling and SASS Metrics: instruction-level and
//     time-series sampling.
//
// CUPTI runs in the process being profiled and requires elevated GPU
// performance counter permissions for most counter APIs.
//
// This package maps every function from the cupti.h header family to a Go
// function using [github.com/ebitengine/purego] for cgo-free dynamic
// loading of libcupti. All functions return errors wrapping CUptiResult
// with descriptive messages from cuptiGetResultString.
//
// # Activity API
//
//   - [ActivityConfigurePCSampling]: Set PC sampling configuration.
//   - [ActivityConfigureUnifiedMemoryCounter]: Set Unified Memory Counter configuration.
//   - [ActivityDisable]: Disable collection of a specific kind of activity record.
//   - [ActivityDisableContext]: Disable collection of a specific kind of activity record for a context.
//   - [Subscriber.ActivityDisableV2]: Disable collection of a specific kind of activity record for the subscriber.
//   - [ActivityEnable]: Enable collection of a specific kind of activity record.
//   - [ActivityEnableAllSyncRecords]: Enables collecting records for all synchronization operations.
//   - [ActivityEnableAllocationSource]: Enables tracking the source library for memory allocation requests.
//   - [ActivityEnableAndDump]: Enable collection of a specific kind of activity record.
//   - [Subscriber.ActivityEnableAndDumpV2]: Enable collection of a specific kind of activity record.
//   - [ActivityEnableContext]: Enable collection of a specific kind of activity record for a context.
//   - [ActivityEnableCudaEventDeviceTimestamps]: Enable/Disable collecting device timestamp for [ActivityKindCudaEvent] record.
//   - [ActivityEnableDeviceGraph]: Controls the collection of records for device launched graphs.
//   - [ActivityEnableDriverApi]: Controls the collection of activity records for specific CUDA Driver APIs.
//   - [ActivityEnableHWTrace]: Enables CUDA kernel timestamp collection via Hardware Event System (HES).
//   - [ActivityEnableLatencyTimestamps]: Controls the collection of queued and submitted timestamps for kernels.
//   - [ActivityEnableLaunchAttributes]: Controls the collection of launch attributes for kernels.
//   - [ActivityEnableRuntimeApi]: Controls the collection of activity records for specific CUDA Runtime APIs.
//   - [Subscriber.ActivityEnableV2]: Enable collection of a specific kind of activity record for the subscriber.
//   - [ActivityFlush]: Wait for all activity records to be delivered via the completion callback.
//   - [ActivityFlushAll]: Request to deliver activity records via the buffer completion callback.
//   - [ActivityFlushPeriod]: Sets the flush period for the worker thread.
//   - [ActivityGetAttribute]: Read an activity API attribute.
//   - [Subscriber.ActivityGetAttributeV2]: Read an activity API attribute.
//   - [Subscriber.ActivityGetEnabledKinds]: Get the enabled activity kinds for a subscriber.
//   - [ActivityGetNextRecord]: Iterate over the activity records in a buffer.
//   - [ActivityGetNumDroppedRecords]: Get the number of activity records that were dropped of insufficient buffer space.
//   - [ActivityGetStructSize]: Get the size of the activity struct for a given CUPTI version.
//   - [ActivityPopExternalCorrelationId]: Pop an external correlation id for the calling thread.
//   - [ActivityPushExternalCorrelationId]: Push an external correlation id for the calling thread.
//   - [ActivityRegisterCallbacks]: Registers callback functions with CUPTI for activity buffer handling.
//   - [Subscriber.ActivityRegisterCallbacksV2]: Registers callback functions with CUPTI for activity buffer handling for the subscriber.
//   - [ActivityRegisterTimestampCallback]: Registers callback function with CUPTI for providing timestamp.
//   - [ActivitySetAttribute]: Write an activity API attribute.
//   - [Subscriber.ActivitySetAttributeV2]: Write an activity API attribute.
//
// # Checkpoint API
//
//   - [CheckpointFree]: Free the backing data for a checkpoint.
//   - [CheckpointRestore]: Restore a checkpoint to the device associated with its context.
//   - [CheckpointSave]: Initialize and save a checkpoint of the device state associated with the handle context.
//
// # Device Queries
//
//   - [DeviceEnumEventDomains]: Get the event domains for a device.
//   - [DeviceEnumMetrics]: Get the metrics for a device.
//   - [DeviceGetAttribute]: Read a device attribute.
//   - [DeviceGetEventDomainAttribute]: Read an event domain attribute.
//   - [DeviceGetNumEventDomains]: Get the number of domains for a device.
//   - [DeviceGetNumMetrics]: Get the number of metrics for a device.
//   - [DeviceSupported]: Check support for a compute device.
//   - [DeviceVirtualizationMode]: Query the virtualization mode of the device.
//
// # Callback API
//
//   - [Subscriber.EnableAllDomains]: Enable or disable all callbacks in all domains.
//   - [Subscriber.EnableCallback]: Enable or disabled callbacks for a specific domain and callback ID.
//   - [Subscriber.EnableDomain]: Enable or disabled all callbacks for a specific domain.
//   - [EnableKernelReplayMode]: Enable kernel replay mode.
//   - [GetCallbackName]: Get the name of a callback for a specific domain and callback ID.
//   - [Subscriber.GetCallbackState]: Get the current enabled/disabled state of a callback for a specific domain and function ID.
//   - [Subscriber.GetEnabledCallbacks]: Get the enabled callbacks for a subscriber.
//   - [Subscribe]: Initialize a callback subscriber with a callback function and user data.
//   - [SubscribeV2]: Initialize a callback subscriber with a callback function and user data.
//   - [SupportedDomains]: Get the available callback domains.
//   - [Subscriber.Unsubscribe]: Unregister a callback subscriber.
//
// # Event API
//
//   - [EventDomainEnumEvents]: Get the events in a domain.
//   - [EventDomainGetAttribute]: Read an event domain attribute.
//   - [EventDomainGetNumEvents]: Get number of events in a domain.
//   - [EventGetAttribute]: Get an event attribute.
//   - [EventGetIdFromName]: Find an event by name.
//   - [EventGroup.AddEvent]: Add an event to an event group.
//   - [NewEventGroup]: Create a new event group for a context.
//   - [EventGroup.Destroy]: Destroy an event group.
//   - [EventGroup.Disable]: Disable an event group.
//   - [EventGroup.Enable]: Enable an event group.
//   - [EventGroup.GetAttribute]: Read an event group attribute.
//   - [EventGroup.ReadAllEvents]: Read the values for all the events in an event group.
//   - [EventGroup.ReadEvent]: Read the value for an event in an event group.
//   - [EventGroup.RemoveAllEvents]: Remove all events from an event group.
//   - [EventGroup.RemoveEvent]: Remove an event from an event group.
//   - [EventGroup.ResetAllEvents]: Zero all the event counts in an event group.
//   - [EventGroup.SetAttribute]: Write an event group attribute.
//   - [EventGroupSetDisable]: Disable an event group set.
//   - [EventGroupSetEnable]: Enable an event group set.
//   - [EventGroupSetsCreate]: For a set of events, get the grouping that indicates the number of passes and the event groups necessary to collect the events.
//   - [EventGroupSetsDestroy]: Destroy a event group sets object.
//
// # Lifecycle
//
//   - [Finalize]: Detach CUPTI from the running process.
//
// # Identifiers
//
//   - [GetContextId]: Get the ID of a context.
//   - [GetDeviceId]: Get the ID of a device.
//   - [GetGraphExecId]: Get the unique ID of executable graph.
//   - [GetGraphId]: Get the unique ID of graph.
//   - [GetGraphNodeId]: Get the unique ID of a graph node.
//   - [GetStreamId]: Get the ID of a stream.
//   - [GetStreamIdEx]: Get the ID of a stream.
//
// # Timestamps
//
//   - [GetTimestamp]: Get the CUPTI timestamp.
//
// # Version Queries
//
//   - [GetVersion]: Get the CUPTI API version.
//
// # Kernel Replay
//
//   - [KernelReplaySubscribeUpdate]: Subscribe to kernel replay updates.
//
// # Metric API
//
//   - [MetricCreateEventGroupSets]: For a set of metrics, get the grouping that indicates the number of passes and the event groups necessary to collect the events required for those metrics.
//   - [MetricEnumEvents]: Get the events required to calculating a metric.
//   - [MetricEnumProperties]: Get the properties required to calculating a metric.
//   - [MetricGetAttribute]: Get a metric attribute.
//   - [MetricGetIdFromName]: Find an metric by name.
//   - [MetricGetNumEvents]: Get number of events required to calculate a metric.
//   - [MetricGetNumProperties]: Get number of properties required to calculate a metric.
//   - [MetricGetRequiredEventGroupSets]: For a metric get the groups of events that must be collected in the same pass.
//   - [MetricGetValue]: Calculate the value for a metric.
//   - [MetricGetValue2]: Calculate the value for a metric.
//
// # PC Sampling API
//
//   - [PCSamplingDisable]: Disable PC sampling.
//   - [PCSamplingEnable]: Enable PC sampling.
//   - [PCSamplingGetConfigurationAttribute]: Read PC Sampling configuration attribute.
//   - [PCSamplingGetData]: Flush GPU PC sampling data periodically.
//   - [PCSamplingGetNumStallReasons]: Get PC sampling stall reason count.
//   - [PCSamplingGetStallReasons]: Get PC sampling stall reasons.
//   - [PCSamplingSetConfigurationAttribute]: Write PC Sampling configuration attribute.
//   - [PCSamplingStart]: Start PC sampling.
//   - [PCSamplingStop]: Stop PC sampling.
//
// # PM Sampling API
//
//   - [PmSamplingCounterDataGetSampleInfo]: Get the sample info (start and end time stamp) for the given sample index.
//   - [PmSamplingCounterDataImageInitialize]: Initialize the counter data to CUPTI record format for storing the metric data.
//   - [PmSamplingDecodeData]: Decode the metrics data stored in the hardware buffer to the counter data image.
//   - [PmSamplingDisable]: Disable PM sampling on the CUDA device and destroy the PM sampling object.
//   - [PmSamplingEnable]: Create a PM sampling object and enable PM sampling on the CUDA device.
//   - [PmSamplingGetCounterAvailability]: Query counter availibility information in a buffer which can be used to filter unavailable raw metrics on host.
//   - [PmSamplingGetCounterDataInfo]: Get the counter data info like number of samples, number of populated samples and number of completed samples in a counter data image.
//   - [PmSamplingGetCounterDataSize]: Query the size of the counter data image which will be used to store the metrics data.
//   - [PmSamplingSetConfig]: Set the configuration for PM sampling like sampling interval, HW buffer size, trigger mode and the config image which has scheduling info for metric collection.
//   - [PmSamplingStart]: Start the PM sampling.
//   - [PmSamplingStop]: Stop the PM sampling.
//
// # Profiling API
//
//   - [ProfilerBeginPass]: Replay API: used for multipass collection.
//   - [ProfilerBeginSession]: Begin profiling session sets up the profiling on the device.
//   - [ProfilerCounterDataImageCalculateScratchBufferSize]: A temporary storage for CounterData image needed for internal operations.
//   - [ProfilerCounterDataImageCalculateSize]: A CounterData image allocates space for values for each counter for each range.
//   - [ProfilerDeInitialize]: DeInitializes the profiler interface.
//   - [ProfilerDeviceSupported]: Query device compatibility with Profiling API.
//   - [ProfilerDisableProfiling]: Disable Profiling.
//   - [ProfilerEnableProfiling]: Enables Profiling.
//   - [ProfilerEndPass]: Replay API: used for multipass collection.
//   - [ProfilerEndSession]: Ends profiling session.
//   - [ProfilerFlushCounterData]: Decode all the submitted passes.
//   - [ProfilerGetCounterAvailability]: Query counter availibility.
//   - [ProfilerHostConfigAddMetrics]: Add the metrics to the profiler host object for generating the config image.
//   - [ProfilerHostDeinitialize]: Deinitialize and destroy the profiler host object ([ProfilerHost]).
//   - [ProfilerHostEvaluateToGpuValues]: Evaluate the metric values for the range index stored in the counter data.
//   - [ProfilerHostGetBaseMetrics]: Get the list of supported base metrics for the chip.
//   - [ProfilerHostGetConfigImage]: Get the config image for the metrics added to the profiler host object.
//   - [ProfilerHostGetConfigImageSize]: Get the size of the config image for the metrics added to the profiler host object.
//   - [ProfilerHostGetMaxNumHardwareMetricsPerPass]: Get the maximum number of hardware metrics (metric names which doesn’t include sass keyword) that can be scheduled in a single pass for a chip.
//   - [ProfilerHostGetMetricProperties]: Get the properties of the metric.
//   - [ProfilerHostGetMetricsInSinglePassSet]: Get all the metrics defined in the single pass metric set.
//   - [ProfilerHostGetNumOfPasses]: Get the number of passes required for profiling the scheduled metrics in the config image.
//   - [ProfilerHostGetRangeName]: Get the range name for the range index stored in the counter data.
//   - [ProfilerHostGetSinglePassSets]: Get the single pass metric sets defined in the metric config file.
//   - [ProfilerHostGetSubMetrics]: Get the list of supported sub-metrics for the metric.
//   - [ProfilerHostGetSupportedChips]: Get the list of supported chips.
//   - [ProfilerHostInitialize]: Create and initialize the profiler host object ([ProfilerHost]).
//   - [ProfilerInitialize]: Initializes the profiler interface.
//   - [ProfilerIsPassCollected]: Asynchronous call to query if the submitted pass to GPU is collected.
//   - [ProfilerPopRange]: Range API’s : Pop user range.
//   - [ProfilerPushRange]: Range API’s : Push user range.
//   - [ProfilerSetConfig]: Set metrics configuration to be profiled.
//   - [ProfilerUnsetConfig]: Unset metrics configuration profiled.
//
// # Range Profiler API
//
//   - [RangeProfilerCounterDataGetRangeInfo]: Get the range name for the given range index.
//   - [RangeProfilerCounterDataImageInitialize]: Initialize the counter data image with the profiling data for the ranges profiled.
//   - [RangeProfilerDecodeData]: Decode the profiling data stored in the hardware to the counter data image passed in the SetConfig API.
//   - [RangeProfilerDisable]: Disable the range profiler on the CUDA context and destroy the range profiler object.
//   - [RangeProfilerEnable]: Create a range profiler object and enable range profiling on the CUDA context.
//   - [RangeProfilerGetCounterDataInfo]: Get the number of ranges stored in the counter data image.
//   - [RangeProfilerGetCounterDataSize]: Get the size of the counter data image required to store the profiling data for the ranges profiled.
//   - [RangeProfilerPopRange]: pop the current range to the Range Profiler.
//   - [RangeProfilerPushRange]: Add a new range to the Range Profiler with a given range name.
//   - [RangeProfilerSetConfig]: Set the configuration for range profiler like maximum number of ranges per pass, number of nesting levels, range and replay mode and the config image which has scheduling info for metric collection.
//   - [RangeProfilerStart]: Start the range profiler.
//   - [RangeProfilerStop]: Stop the range profiler.
//
// # SASS Metrics API
//
//   - [SassMetricsDisable]: SASS metric data collection disable API will mark the end of a range, any kernel launched after this API call will not be profiled for the SASS metrics.
//   - [SassMetricsEnable]: Sass metric data collection enable API will mark the start of a range, between which kernel will be profiled for SASS metrics.
//   - [SassMetricsFlushData]: Flush SASS metrics data from CUPTI internal buffer to the user buffer.
//   - [SassMetricsGetDataProperties]: SASS metric data properties API will give the data regarding number of instances of a metric value and number of SASS instruction data has been collected.
//   - [SassMetricsGetMetrics]: Get the list of all supported SASS metrics for the chip.
//   - [SassMetricsGetNumOfMetrics]: Get the number of supported SASS metrics for the chip.
//   - [SassMetricsGetProperties]: Get metric properties for the queried metric.
//   - [SassMetricsSetConfig]: Set config for the SASS metric data collection for a device.
//   - [SassMetricsUnsetConfig]: Unset config API will reset the SASS metric data collection configuration for the device.
//
// # Other
//
//   - [ComputeCapabilitySupported]: Check support for a compute capability.
//   - [DisableKernelReplayMode]: Disable kernel replay mode.
//   - [EnumEventDomains]: Get the event domains available on any device.
//   - [EnumMetrics]: Get all the metrics available on any device.
//   - [GetAutoBoostState]: Get auto boost state.
//   - [GetCubinCrc]: Get the CRC of cubin.
//   - [GetLastError]: Returns the last error from a cupti call or callback.
//   - [GetNumEventDomains]: Get the number of event domains available on any device.
//   - [GetNumMetrics]: Get the total number of metrics available on any device.
//   - [GetSassToSourceCorrelation]: SASS to Source correlation.
//   - [GetThreadIdType]: Get the thread-id type.
//   - [IsTracingSessionRunning]: Check whether a CUPTI tracing session is still running.
//   - [RegisterComputeCrcCallback]: Register callback function with CUPTI to use your own algorithm to compute cubin crc.
//   - [SetEventCollectionMode]: Set the event collection mode.
//   - [SetThreadIdType]: Set the thread-id type.
//
// [CUPTI Documentation]: https://docs.nvidia.com/cupti/
package cupti

// Code generated by gen-nvidia from cupti.h. DO NOT EDIT.
