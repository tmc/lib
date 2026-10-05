package main

import (
	"context"
	"fmt"
	"os"

	"github.com/tmc/lib/nvidia/x/cudaschedule"
)

func run(args []string) error { return cudaschedule.RunRecordedHelper(context.Background(), args) }
func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
