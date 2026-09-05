// Package cuptirange configures, collects and evaluates CUPTI Range Profiling
// metrics with explicit memory and replay limits.
//
// NewPlan creates a host configuration for a caller-identified chip. Plan.Capture
// uses automatic kernel ranges and caller-controlled user replay, retaining a
// CUDA context through cleanup. The replay callback must restore identical
// workload state on every pass. Capture rejects incomplete replay, dropped
// ranges and empty results. Call Close on any nonnil result returned with an
// error to retry native cleanup.
//
// Plan.Evaluate accepts trusted counter images from the matching configuration;
// it does not validate replay completeness or parse untrusted input. WriteMetrics
// exports named values to Perfetto without inferring units or measurement times.
// Profiling can change execution; its results are not an uninstrumented timing
// baseline. Hardware support and permissions are checked by the loaded CUPTI
// library. No library installation or privilege changes are performed.
package cuptirange
