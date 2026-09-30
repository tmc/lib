// spin-go launches the native control with the same arguments and environment.
// NVTX ranges are emitted by the native submitting thread, not the Go wrapper.
package main

import (
	"fmt"
	"os"
	"os/exec"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: spin-go NATIVE_SPIN ARGS...")
		os.Exit(2)
	}
	cmd := exec.Command(os.Args[1], os.Args[2:]...)
	cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
	if err := cmd.Run(); err != nil {
		if err, ok := err.(*exec.ExitError); ok {
			os.Exit(err.ExitCode())
		}
		fmt.Fprintf(os.Stderr, "spin-go: %v\n", err)
		os.Exit(1)
	}
}
