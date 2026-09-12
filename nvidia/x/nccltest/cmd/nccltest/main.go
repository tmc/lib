// Command nccltest runs correctness-checked NCCL cases and writes JSON.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"

	"github.com/tmc/lib/nvidia/x/nccltest"
)

func main() {
	spread := flag.Float64("spread", 1, "generated alltoallv spread in [0, 1]")
	datatype := flag.String("datatype", "float", "int8, uint8, int32, uint32, int64, uint64, half, bfloat16, f8e4m3, f8e5m2, float, double")
	op := flag.String("op", "all_reduce", "collective name")
	devices := flag.String("devices", "0", "comma-separated CUDA device ordinals")
	sizesFlag := flag.String("sizes", "", "comma-separated byte sizes, sharing resources across the sweep")
	bytes := flag.Int("bytes", 1024, "requested message bytes")
	root := flag.Int("root", 0, "root rank")
	reduction := flag.String("reduction", "sum", "sum, prod, min, max, avg, mulsum")
	inPlace := flag.Bool("in-place", false, "use operation-specific aliasing")
	warmup := flag.Int("warmup", 5, "warmup iterations")
	iterations := flag.Int("iterations", 20, "timed iterations")
	aggregation := flag.Int("aggregation", 1, "collectives per timed iteration")
	cycles := flag.Int("cycles", 1, "cycles reusing communicators and buffers; zero repeats until interrupted")
	checks := flag.Int("checks", 2, "correctness passes")
	timing := flag.String("timing", "mean", "rank timing selection: mean, rank0, min, max")
	defaultStream := flag.Bool("default-stream", false, "use CUDA stream zero")
	offset := flag.Int("offset", 0, "leading elements before send and receive buffers (reference -u)")
	syncIterations := flag.Bool("sync-iterations", false, "synchronize after each aggregated timed iteration")
	graphLaunches := flag.Int("graph-launches", 0, "timed CUDA graph replays; zero disables capture")
	register := flag.Bool("register", false, "register buffers with NCCL")
	flag.Parse()
	bytesSet, sizesSet, spreadSet := false, false, false
	flag.Visit(func(f *flag.Flag) {
		if f.Name == "spread" {
			spreadSet = true
		}
		if f.Name == "bytes" {
			bytesSet = true
		}
		if f.Name == "sizes" {
			sizesSet = true
		}
	})
	if sizesSet && (bytesSet || *sizesFlag == "") {
		fmt.Fprintln(os.Stderr, "-sizes must be nonempty and cannot be combined with -bytes")
		os.Exit(2)
	}
	if *cycles < 0 {
		fmt.Fprintln(os.Stderr, "cycles must not be negative")
		os.Exit(2)
	}
	var ordinals []int32
	for _, value := range strings.Split(*devices, ",") {
		n, err := strconv.ParseInt(value, 10, 32)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
		ordinals = append(ordinals, int32(n))
	}
	c := nccltest.Case{DataType: nccltest.DataType(*datatype), Operation: nccltest.Operation(*op), Bytes: *bytes, Root: *root, Reduction: nccltest.Reduction(*reduction), InPlace: *inPlace}
	if spreadSet {
		c.Spread = spread
	}
	options := nccltest.Options{SyncIterations: *syncIterations, Offset: *offset, GraphLaunches: *graphLaunches, Register: *register, DefaultStream: *defaultStream, Timing: nccltest.Timing(*timing), Devices: ordinals, Warmup: *warmup, Iterations: *iterations, Checks: *checks, Aggregation: *aggregation}
	if *sizesFlag != "" || *cycles == 0 {
		var sizes []int
		sizeText := *sizesFlag
		if sizeText == "" {
			sizeText = strconv.Itoa(*bytes)
		}
		for _, text := range strings.Split(sizeText, ",") {
			n, err := strconv.Atoi(text)
			if err != nil || n <= 0 {
				fmt.Fprintln(os.Stderr, "invalid sweep size", text)
				os.Exit(2)
			}
			sizes = append(sizes, n)
		}
		if *cycles > int(^uint(0)>>1)/len(sizes) {
			fmt.Fprintln(os.Stderr, "overflowing sweep count")
			os.Exit(2)
		}
		ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
		err := writeSweep(os.Stdout, sizes, *cycles, func(yield func(nccltest.Result) error) error {
			return nccltest.RunSweepContext(ctx, c, options, sizes, *cycles, yield)
		})
		stop()
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}
	if *cycles != 1 {
		err := writeCycles(os.Stdout, *cycles, func(yield func(nccltest.Result) error) error { return nccltest.RunCycles(c, options, *cycles, yield) })
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}
	result, err := nccltest.Run(c, options)
	receipt := struct {
		Result nccltest.Result
		Error  string
	}{Result: result}
	if err != nil {
		receipt.Error = err.Error()
	}
	if e := json.NewEncoder(os.Stdout).Encode(receipt); e != nil {
		fmt.Fprintln(os.Stderr, e)
		os.Exit(1)
	}
	if err != nil {
		os.Exit(1)
	}
}

func writeCycles(w io.Writer, count int, run func(func(nccltest.Result) error) error) error {
	encoder := json.NewEncoder(w)
	completed := 0
	err := run(func(r nccltest.Result) error {
		if completed >= count {
			return fmt.Errorf("too many cycle results")
		}
		if err := encoder.Encode(struct {
			Kind   string
			Cycle  int
			Result nccltest.Result
		}{"cycle", completed + 1, r}); err != nil {
			return err
		}
		completed++
		return nil
	})
	if err == nil && completed != count {
		err = fmt.Errorf("completed %d cycles, want %d", completed, count)
	}
	message := ""
	if err != nil {
		message = err.Error()
	}
	finalErr := encoder.Encode(struct {
		Kind                 string
		Requested, Completed int
		Complete             bool
		Error                string
	}{"summary", count, completed, err == nil, message})
	return errors.Join(err, finalErr)
}

func writeSweep(w io.Writer, sizes []int, cycles int, run func(func(nccltest.Result) error) error) error {
	if len(sizes) == 0 || cycles < 0 || cycles > int(^uint(0)>>1)/len(sizes) {
		return fmt.Errorf("invalid sweep dimensions")
	}
	encoder := json.NewEncoder(w)
	completed, total := 0, cycles*len(sizes)
	err := run(func(r nccltest.Result) error {
		if completed == int(^uint(0)>>1) || (cycles > 0 && completed >= total) || r.Case.Bytes != sizes[completed%len(sizes)] {
			return fmt.Errorf("unexpected sweep result")
		}
		row := struct {
			Kind             string
			Cycle, SizeIndex int
			Result           nccltest.Result
		}{"sample", completed/len(sizes) + 1, completed%len(sizes) + 1, r}
		if err := encoder.Encode(row); err != nil {
			return err
		}
		completed++
		return nil
	})
	if err == nil && cycles == 0 {
		err = fmt.Errorf("continuous sweep returned without a stop reason")
	}
	if err == nil && completed != total {
		err = fmt.Errorf("completed %d samples, want %d", completed, total)
	}
	message := ""
	if err != nil {
		message = err.Error()
	}
	finalErr := encoder.Encode(struct {
		Kind                                                string
		RequestedCycles, RequestedSamples, CompletedSamples int
		Complete                                            bool
		Error                                               string
	}{"summary", cycles, total, completed, err == nil, message})
	return errors.Join(err, finalErr)
}
