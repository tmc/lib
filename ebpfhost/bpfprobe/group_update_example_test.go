package bpfprobe_test

import (
	"fmt"
	"github.com/tmc/lib/ebpfhost/bpfprobe"
)

func ExampleGroup_Add() {
	// Establish a group with AttachGroup first, and supply a verifier checking
	// process and task birth identities against retained /proc evidence.
	var group *bpfprobe.Group
	err := group.Add([]bpfprobe.Config{{PID: 42, Path: "/proc/42/exe"}}, nil)
	fmt.Println(err)
	// Output: bpf probe: nil group
}
