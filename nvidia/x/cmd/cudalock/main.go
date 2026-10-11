// Command cudalock provides exclusive GPU locking via an atomic directory mutex.
//
// Usage:
//
//	cudalock [flags] [--] <command> [args...]
//	cudalock -status
//
// Flags:
//
//	-dir string
//	    Lock directory path (default $GPU_LOCKDIR or $HOME/ml-explore/.gpu-lock/lock)
//	-label string
//	    Descriptive label recorded in lock (default $GPU_LOCK_LABEL or current PID)
//	-timeout duration
//	    Maximum duration to wait for lock acquisition; 0 waits indefinitely
//	-poll duration
//	    Interval between lock acquisition attempts (default 250ms)
//	-status
//	    Inspect and print filesystem mutex and device compute application status
//	-quiet
//	    Suppress lock acquisition and release log messages to stderr
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/tmc/lib/nvidia/x/cudalock"
)

func main() {
	os.Exit(run())
}

func run() int {
	var (
		dirFlag     = flag.String("dir", "", "Lock directory path (overrides $GPU_LOCKDIR)")
		labelFlag   = flag.String("label", "", "Label recorded in lock owner file")
		timeoutFlag = flag.Duration("timeout", 0, "Maximum time to wait for lock acquisition (0 = wait forever)")
		pollFlag    = flag.Duration("poll", 250*time.Millisecond, "Interval between acquire polling attempts")
		statusFlag  = flag.Bool("status", false, "Print current lock and device compute status and exit")
		quietFlag   = flag.Bool("quiet", false, "Suppress acquire and release messages on stderr")
	)

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: cudalock [flags] [--] <command> [args...]\n")
		fmt.Fprintf(os.Stderr, "       cudalock -status\n\n")
		flag.PrintDefaults()
	}
	flag.Parse()

	if *statusFlag {
		targetDir := *dirFlag
		if targetDir == "" {
			targetDir = cudalock.DefaultDir()
		}
		rep, err := cudalock.Status(targetDir)
		if err != nil {
			fmt.Fprintf(os.Stderr, "cudalock: status check failed: %v\n", err)
			return 1
		}
		fmt.Print(rep.String())
		return 0
	}

	args := flag.Args()
	if len(args) > 0 && args[0] == "--" {
		args = args[1:]
	}
	if len(args) == 0 {
		flag.Usage()
		return 2
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	if *timeoutFlag > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, *timeoutFlag)
		defer cancel()
	}

	opts := []cudalock.Option{
		cudalock.WithPollInterval(*pollFlag),
	}
	if *dirFlag != "" {
		opts = append(opts, cudalock.WithDir(*dirFlag))
	}
	if *labelFlag != "" {
		opts = append(opts, cudalock.WithLabel(*labelFlag))
	}

	if !*quietFlag {
		fmt.Fprintf(os.Stderr, "cudalock: waiting for GPU lock... (%s)\n", args[0])
	}

	handle, err := cudalock.Acquire(ctx, opts...)
	if err != nil {
		fmt.Fprintf(os.Stderr, "cudalock: %v\n", err)
		return 1
	}

	if !*quietFlag {
		fmt.Fprintf(os.Stderr, "cudalock: acquired (label=%s, pid=%d) — running: %v\n",
			handle.Owner().Label, handle.Owner().PID, args)
	}

	exitCode, err := cudalock.Run(ctx, args, opts...)
	if err != nil {
		fmt.Fprintf(os.Stderr, "cudalock: command error: %v\n", err)
	}
	if releaseErr := handle.Release(); releaseErr != nil {
		fmt.Fprintf(os.Stderr, "cudalock: %v\n", releaseErr)
		if exitCode == 0 {
			exitCode = 1
		}
	} else if !*quietFlag {
		fmt.Fprintf(os.Stderr, "cudalock: released (rc=%d)\n", exitCode)
	}
	return exitCode
}
