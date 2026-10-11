package bpfload_test

import (
	"fmt"
	"github.com/tmc/lib/ebpfhost/bpfload"
)

func ExampleLoad() {
	// Invalid programs are rejected before attempting a privileged syscall.
	_, _, err := bpfload.Load(bpfload.ProgramSpec{})
	fmt.Println(err)
	// Output: bpf: missing program type
}
