package cudatracehost_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudatracehost"
)

func ExampleParse() {
	o, err := cudatracehost.Parse(" app-42 [000] .... 1.000001: cuda0_in: a0=0x123")
	fmt.Println(o.TID, o.Timestamp, o.Args[0], err)
	// Output: 42 1000001000 291 <nil>
}

func ExampleParseScheduler() {
	e, err := cudatracehost.ParseScheduler(" app-42 [003] .... 1.25: sched_wakeup: comm=app pid=42 prio=120 target_cpu=003")
	fmt.Println(e.Kind, e.WakeTID, e.CPU, e.Clock, err)
	// Output: sched_wakeup 42 3 linux_monotonic <nil>
}
