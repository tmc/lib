// Nsysgaps reports the idle device time between kernel launches in an
// Nsight Systems trace and models it on a faster device.
//
// Usage:
//
//	nsysgaps [-end NAME] [-skip N] [-floor DUR] [-top N] [-api API.csv] GPU.csv
//
// GPU.csv is the output of
//
//	nsys stats --report cuda_gpu_trace --format csv REPORT.nsys-rep
//
// and API.csv, if given, that of the cuda_api_trace report for the same
// capture. Nsysgaps splits the kernels into steps, each ending with a
// kernel whose name contains -end (default Argmax), drops the first -skip
// steps (default 1, the prefill), and prints for each step its launches,
// span, busy and idle time, and the mean idle time per launch. It then
// prints the median step and the kernels with the most idle time before
// them in that step.
//
// For each -floor duration (repeatable; default 2ms), it prints
// floor + median idle: the step time on a device that runs the step's
// kernels in floor but keeps this trace's launch gaps. That is a model,
// not a measurement. With -api it also prints the host time spent in
// kernel and graph launch calls.
package main
