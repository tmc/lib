// This consumer deliberately imports no internal package.
package main

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/tmc/lib/nvidia/x/cudainject"
)

func main() {
	if len(os.Args) != 4 {
		panic("compiler toolkit build-directory required")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	agent, err := cudainject.BuildAgent(ctx, cudainject.BuildOptions{Compiler: os.Args[1], CUDAHome: os.Args[2], Directory: os.Args[3], Target: "sbsa-linux"})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println(len(agent.Sources))
}
