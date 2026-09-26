// Package perfscript reads host CPU samples from the text perf script
// writes, so they can be placed on a capture's timeline.
//
// The capture agent stamps every record with CLOCK_MONOTONIC, the clock
// perf uses with -k CLOCK_MONOTONIC. Record and convert with:
//
//	perf record -k CLOCK_MONOTONIC --call-graph fp -e cpu-clock --all-user -o perf.data -- cudaprof capture -o run.pprof -- ./target
//	perf script -i perf.data --ns -F comm,tid,time,ip,sym > host.txt
package perfscript
