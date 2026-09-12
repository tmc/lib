// Command ncclcompare retains matched upstream and Go NCCL test receipts.
package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/tmc/lib/nvidia/x/nccltest"
)

type attempt struct {
	Environment []string // Explicit child overrides only; ambient environment is not recorded.
	Started     time.Time
	Finished    time.Time
	Command     []string
	ExitError   string
	Output      string
	SHA256      string
}
type comparison struct {
	Cycle           int
	Reduction       nccltest.Reduction
	Operation       nccltest.Operation
	Bytes           int
	InPlace         bool
	Reference       nccltest.Measurement
	ReferenceConfig nccltest.ReferenceConfig
	Go              nccltest.Result
	TimeRatio       float64
	Error           string
}
type receipt struct {
	Spread             *float64
	SharedSweep        bool
	Cycles             int // Requested cycles per operation/reduction/size workload.
	RequestedWorkloads int // Operation/reduction/size combinations, before in-place expansion.
	FinishedWorkloads  int
	Complete           bool // True only after every requested workload passed.
	Started            time.Time
	Ranks              int
	Attempts           []attempt
	Comparisons        []comparison
	Error              string
}

func main() {
	spreadFlag := flag.Float64("spread", 1, "generated alltoallv spread in [0, 1]; requires -operations alltoallv")
	datatype := flag.String("datatype", "float", "int8, uint8, int32, uint32, int64, uint64, half, bfloat16, f8e4m3, f8e5m2, float, double")
	native := flag.String("reference", "", "directory with pinned nccl-tests executables")
	candidate := flag.String("go-test", "", "path to nccltest executable")
	output := flag.String("output", "", "new receipt directory")
	operations := flag.String("operations", "all_reduce,all_gather,broadcast,reduce_scatter,reduce,alltoall,alltoallv,scatter,gather,sendrecv,hypercube", "comma-separated operations")
	reductions := flag.String("reductions", "sum,prod,min,max,avg,mulsum", "reductions for arithmetic collectives")
	sizes := flag.String("sizes", "16,1024,65536", "comma-separated requested byte sizes")
	first := flag.Int("first", 0, "first byte size in a sweep (requires -last and -step or -factor)")
	last := flag.Int("last", 0, "last byte size in a sweep")
	step := flag.Int("step", 0, "linear byte increment")
	factor := flag.Int("factor", 0, "geometric byte multiplier")
	aggregation := flag.Int("aggregation", 1, "collectives per timed iteration (reference -m)")
	ranks := flag.Int("ranks", 1, "number of distinct local devices, starting at zero")
	root := flag.Int("root", 0, "root rank for rooted collectives")
	timing := flag.String("timing", "mean", "rank timing selection: mean, rank0, min, max")
	defaultStream := flag.Bool("default-stream", false, "use CUDA stream zero")
	offset := flag.Int("offset", 0, "leading buffer elements (reference -u)")
	sharedSweep := flag.Bool("shared-sweep", false, "reuse resources across each complete size sweep")
	cycles := flag.Int("cycles", 1, "finite cycles per case (reference -N)")
	syncIterations := flag.Bool("sync-iterations", false, "synchronize timed iterations (reference -z 3)")
	graphLaunches := flag.Int("graph-launches", 0, "timed CUDA graph replays (reference -G)")
	register := flag.Bool("register", false, "register buffers with NCCL")
	flag.Parse()
	var spread *float64
	flag.Visit(func(f *flag.Flag) {
		if f.Name == "spread" {
			spread = spreadFlag
		}
	})
	if spread != nil && (*operations != "alltoallv" || math.IsNaN(*spread) || math.IsInf(*spread, 0) || *spread < 0 || *spread > 1) {
		fmt.Fprintln(os.Stderr, "spread requires -operations alltoallv and a finite value in [0, 1]")
		os.Exit(2)
	}
	if *cycles <= 0 || *cycles > 1<<31-1 {
		fmt.Fprintln(os.Stderr, "cycles outside reference range")
		os.Exit(2)
	}
	if err := checkEnvironment(os.Environ()); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	if *datatype == "" || nccltest.DataType(*datatype).Size() == 0 {
		fmt.Fprintf(os.Stderr, "unsupported datatype %q\n", *datatype)
		os.Exit(2)
	}
	if *native == "" || *candidate == "" || *output == "" || *ranks < 1 || *aggregation < 1 || *aggregation > int(^uint(0)>>1)/20 {
		fmt.Fprintln(os.Stderr, "-reference, -go-test, -output, and positive -ranks are required")
		os.Exit(2)
	}
	if *syncIterations && *graphLaunches != 0 {
		fmt.Fprintln(os.Stderr, "synchronized iterations cannot use graphs")
		os.Exit(2)
	}
	timedOperations := 20 * *aggregation
	if *graphLaunches < 0 || (*graphLaunches > 0 && (*defaultStream || timedOperations > int(^uint(0)>>1) / *graphLaunches)) {
		fmt.Fprintln(os.Stderr, "invalid graph configuration or overflowing operation count")
		os.Exit(2)
	}
	if *graphLaunches > 0 {
		timedOperations *= *graphLaunches
	}
	// Upstream pads by 8*unalign using a C int before allocating.
	if *offset < 0 || *offset > (1<<31-1)/8 {
		fmt.Fprintln(os.Stderr, "offset exceeds reference allocation range")
		os.Exit(2)
	}
	if *root < 0 || *root >= *ranks {
		fmt.Fprintln(os.Stderr, "-root must be a rank in the communicator")
		os.Exit(2)
	}
	explicitSizes := false
	flag.Visit(func(f *flag.Flag) {
		if f.Name == "sizes" {
			explicitSizes = true
		}
	})
	sweep, err := selectSizes(*sizes, explicitSizes, *first, *last, *step, *factor)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	if err := validateOffsetSizes(sweep, *offset); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	batches := make([][]int, len(sweep))
	for i, size := range sweep {
		batches[i] = []int{size}
	}
	if *sharedSweep {
		if _, err := referenceSweepArgs(sweep); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
		batches = [][]int{sweep}
	}
	if *cycles > int(^uint(0)>>1)/len(sweep) {
		fmt.Fprintln(os.Stderr, "overflowing sweep count")
		os.Exit(2)
	}
	workloads, err := prepareWorkloads(*operations, *reductions, sweep, nccltest.DataType(*datatype), *ranks, *root)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	if *ranks > 1 && *timing != "mean" {
		fmt.Fprintln(os.Stderr, "non-mean timing comparison requires one rank: upstream aggregates host threads, Go aggregates device events")
		os.Exit(2)
	}
	average, err := referenceTiming(*timing)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	if err := os.Mkdir(*output, 0755); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	r := receipt{Spread: spread, SharedSweep: *sharedSweep, Cycles: *cycles, Started: time.Now().UTC(), Ranks: *ranks, RequestedWorkloads: len(workloads) * len(sweep)}
	if err := writeReceipt(*output, r); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	var devices []string
	for i := 0; i < *ranks; i++ {
		devices = append(devices, strconv.Itoa(i))
	}
	run := func(name string, environment []string, args ...string) ([]byte, error) {
		binary, err := os.ReadFile(args[0])
		if err != nil {
			return nil, err
		}
		ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
		defer cancel()
		cmd := exec.CommandContext(ctx, args[0], args[1:]...)
		cmd.Env = append(os.Environ(), environment...)
		started := time.Now().UTC()
		out, err := cmd.CombinedOutput()
		finished := time.Now().UTC()
		a := attempt{Environment: append([]string(nil), environment...), Started: started, Finished: finished, Command: args, Output: name, SHA256: fmt.Sprintf("%x", sha256.Sum256(binary))}
		if err != nil {
			a.ExitError = err.Error()
		}
		r.Attempts = append(r.Attempts, a)
		if e := os.WriteFile(filepath.Join(*output, name), out, 0644); e != nil {
			return nil, e
		}
		return out, err
	}
	blockingMode := "0"
	if *syncIterations {
		blockingMode = "3"
	}
	registrationMode := "0"
	if *register {
		registrationMode = "1"
	}
	streamMode := "0"
	if *defaultStream {
		streamMode = "1"
	}
	for _, work := range workloads {
		before := len(r.Comparisons)
		opName, reduction := work.operation, work.reduction
		for _, batch := range batches {
			size := batch[0]
			sizeName := strconv.Itoa(size)
			op := nccltest.Operation(opName)
			base := opName + "-" + reduction + "-" + sizeName
			nativeArgs := []string{filepath.Join(*native, opName+"_perf"), "-g", strconv.Itoa(*ranks), "-d", *datatype, "-o", reduction, "-r", strconv.Itoa(*root), "-w", "5", "-n", "20", "-m", strconv.Itoa(*aggregation), "-c", "2", "-a", average, "-y", streamMode, "-R", registrationMode, "-G", strconv.Itoa(*graphLaunches), "-u", strconv.Itoa(*offset), "-N", strconv.Itoa(*cycles), "-z", blockingMode}
			sweepArgs, _ := referenceSweepArgs(batch)
			nativeArgs = append(nativeArgs, sweepArgs...)
			var environment []string
			if spread != nil {
				environment = []string{"NCCL_TESTS_ALLTOALLV_SPREAD=" + strconv.FormatFloat(*spread, 'g', -1, 64)}
			}
			nativeOut, err := run(base+"-native.log", environment, nativeArgs...)
			if err != nil {
				r.Error = fmt.Sprintf("%s upstream failed: %v", base, err)
				break
			}
			rows, err := nccltest.ParseReference(bytes.NewReader(nativeOut))
			if err != nil || len(rows) != *cycles*len(batch) {
				r.Error = fmt.Sprintf("%s reference parse: rows=%d error=%v", base, len(rows), err)
				break
			}
			wantConfig := nccltest.ReferenceConfig{Threads: 1, GPUs: *ranks, Warmup: 5, Iterations: 20, Aggregation: *aggregation, Checks: 2, GraphLaunches: *graphLaunches, Offset: *offset}
			for i, row := range rows {
				c := nccltest.Case{Spread: spread, Operation: op, DataType: nccltest.DataType(*datatype), Reduction: nccltest.Reduction(reduction), Bytes: batch[i%len(batch)], Root: *root}
				if err := verifyReferenceRow(row, c, *ranks, wantConfig); err != nil {
					r.Error = base + ": " + err.Error()
					break
				}
				if row.NCCLVersion != rows[0].NCCLVersion || row.InPlace.Checked != rows[0].InPlace.Checked || row.OutOfPlace.Checked != rows[0].OutOfPlace.Checked {
					r.Error = base + ": cycle applicability mismatch"
					break
				}
			}
			if r.Error != "" {
				break
			}
			for _, inPlace := range []bool{false, true} {
				ref := rows[0].OutOfPlace
				if inPlace {
					ref = rows[0].InPlace
				}
				if !ref.Checked {
					continue
				}
				c := nccltest.Case{Spread: spread, DataType: nccltest.DataType(*datatype), Operation: op, Bytes: size, Root: *root, Reduction: nccltest.Reduction(reduction), InPlace: inPlace}
				l, err := c.Layout(*ranks)
				if err != nil {
					r.Error = fmt.Sprintf("%s Go layout: %v", base, err)
					break
				}
				args := []string{*candidate, "-sync-iterations=" + strconv.FormatBool(*syncIterations), "-cycles", strconv.Itoa(*cycles), "-offset", strconv.Itoa(*offset), "-graph-launches", strconv.Itoa(*graphLaunches), "-register=" + strconv.FormatBool(*register), "-default-stream=" + strconv.FormatBool(*defaultStream), "-timing", *timing, "-datatype", *datatype, "-op", opName, "-reduction", reduction, "-root", strconv.Itoa(*root), "-devices", strings.Join(devices, ","), "-warmup", "5", "-iterations", "20", "-aggregation", strconv.Itoa(*aggregation), "-checks", "2"}
				if spread != nil {
					args = append(args, "-spread", strconv.FormatFloat(*spread, 'g', -1, 64))
				}
				if *sharedSweep {
					values := make([]string, len(batch))
					for i, n := range batch {
						values[i] = strconv.Itoa(n)
					}
					args = append(args, "-sizes", strings.Join(values, ","))
				} else {
					args = append(args, "-bytes", sizeName)
				}
				if inPlace {
					args = append(args, "-in-place")
				}
				mode := "out"
				if inPlace {
					mode = "in"
				}
				out, runErr := run(base+"-"+mode+"-go.json", nil, args...)
				var results []nccltest.Result
				var decodeErr error
				if *sharedSweep {
					results, decodeErr = readSweepResults(bytes.NewReader(out), batch, *cycles)
				} else if *cycles == 1 {
					var got struct {
						Result nccltest.Result
						Error  string
					}
					decodeErr = json.Unmarshal(out, &got)
					if decodeErr == nil && got.Error != "" {
						decodeErr = fmt.Errorf("%s", got.Error)
					}
					results = []nccltest.Result{got.Result}
				} else {
					results, decodeErr = readCycleResults(bytes.NewReader(out), *cycles)
				}
				if runErr != nil || decodeErr != nil {
					r.Error = fmt.Sprintf("%s Go run: %v; decode: %v", base, runErr, decodeErr)
					break
				}
				for cycle, result := range results {
					c.Bytes = batch[cycle%len(batch)]
					l, err = c.Layout(*ranks)
					if err != nil {
						r.Error = err.Error()
						break
					}
					ref = rows[cycle].OutOfPlace
					if inPlace {
						ref = rows[cycle].InPlace
					}
					item := comparison{Cycle: cycle/len(batch) + 1, Reduction: nccltest.Reduction(reduction), Operation: op, Bytes: c.Bytes, InPlace: inPlace, Reference: ref, ReferenceConfig: rows[0].Config, Go: result}
					if result.SyncIterations != *syncIterations || result.Offset != *offset || !validTiming(result) || !validComparisonCount(result) || !validGraphResult(result, *graphLaunches) || result.Register != *register || result.DefaultStream != *defaultStream || result.Timing != nccltest.Timing(*timing) || !sameCase(result.Case, c) || ref.Wrong != 0 || result.Wrong != 0 || result.Comparisons <= 0 || result.Ranks != *ranks || result.Layout != l || result.Aggregation != *aggregation || result.Iterations != 20 || result.Warmup != 5 || result.Checks != 2 || result.TimedOperations != timedOperations || result.NCCLVersion != rows[0].NCCLVersion {
						item.Error = "correctness or effective case mismatch"
					}
					if ref.Microseconds > 0 {
						item.TimeRatio = float64(result.GPUTime) / float64(time.Microsecond) / ref.Microseconds
					}
					r.Comparisons = append(r.Comparisons, item)
					if err := writeReceipt(*output, r); err != nil {
						r.Error = err.Error()
						break
					}
					if item.Error != "" {
						r.Error = base + ": " + item.Error
						break
					}
				}
				if r.Error != "" {
					break
				}
			}
			if r.Error != "" {
				break
			}
			r.FinishedWorkloads += len(batch)
			if err := writeReceipt(*output, r); err != nil {
				r.Error = err.Error()
				break
			}
		}
		if r.Error != "" {
			break
		}
		fmt.Fprintf(os.Stderr, "%s %s %s: %d checked cases\n", *datatype, opName, reduction, len(r.Comparisons)-before)
	}
	if len(r.Comparisons) == 0 && r.Error == "" {
		r.Error = "no checked comparisons"
	}
	r.Complete = r.Error == "" && r.FinishedWorkloads == r.RequestedWorkloads
	err = writeReceipt(*output, r)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if r.Error != "" {
		fmt.Fprintln(os.Stderr, r.Error)
		os.Exit(1)
	}
	fmt.Printf("%d checked cases agree on correctness; timing ratios retained in %s\n", len(r.Comparisons), filepath.Join(*output, "comparison.json"))
}

// Reference-only overrides can change the workload without changing the
// printed controls. Require explicit support before allowing one in a comparison.
func checkEnvironment(env []string) error {
	for _, entry := range env {
		name, _, _ := strings.Cut(entry, "=")
		if strings.HasPrefix(name, "NCCL_TESTS_") {
			return fmt.Errorf("unset %s: reference-only overrides are not supported in matched comparisons", name)
		}
	}
	return nil
}

func selectSizes(list string, explicit bool, first, last, step, factor int) ([]int, error) {
	if first != 0 || last != 0 || step != 0 || factor != 0 {
		if explicit || (step == 0) == (factor == 0) {
			return nil, fmt.Errorf("choose -sizes or a sweep with exactly one of -step and -factor")
		}
		if step != 0 {
			return nccltest.LinearSizes(first, last, step)
		}
		return nccltest.Sizes(first, last, factor)
	}
	var sizes []int
	seen := make(map[int]bool)
	for _, word := range strings.Split(list, ",") {
		n, err := strconv.Atoi(word)
		if err != nil || n <= 0 || seen[n] {
			return nil, fmt.Errorf("invalid or duplicate byte size %q", word)
		}
		seen[n] = true
		sizes = append(sizes, n)
	}
	return sizes, nil
}

type workload struct {
	operation string
	reduction string
}

// Validate every case before creating receipts or starting either executable.
func prepareWorkloads(operations, reductions string, sizes []int, datatype nccltest.DataType, ranks, root int) ([]workload, error) {
	var result []workload
	seen := make(map[workload]bool)
	for _, op := range strings.Split(operations, ",") {
		ops := []string{"sum"}
		if op == "all_reduce" || op == "reduce" || op == "reduce_scatter" {
			ops = strings.Split(reductions, ",")
		}
		for _, reduction := range ops {
			w := workload{op, reduction}
			if seen[w] {
				return nil, fmt.Errorf("duplicate workload %s/%s", op, reduction)
			}
			seen[w] = true
			for _, size := range sizes {
				c := nccltest.Case{Operation: nccltest.Operation(op), Reduction: nccltest.Reduction(reduction), DataType: datatype, Bytes: size, Root: root}
				if _, err := c.Layout(ranks); err != nil {
					return nil, fmt.Errorf("%s/%s bytes=%d: %w", op, reduction, size, err)
				}
			}
			result = append(result, w)
		}
	}
	return result, nil
}

// nccl-tests common.cu: average 0=rank0, 1=mean, 2=min, 3=max.
func referenceTiming(mode string) (string, error) {
	switch nccltest.Timing(mode) {
	case nccltest.TimingRankZero:
		return "0", nil
	case nccltest.TimingMean:
		return "1", nil
	case nccltest.TimingMinimum:
		return "2", nil
	case nccltest.TimingMaximum:
		return "3", nil
	}
	return "", fmt.Errorf("unknown timing selection %q", mode)
}

// Rename within the output directory keeps the last complete JSON checkpoint
// readable if the process is terminated during a later write.
func writeReceipt(directory string, r receipt) error {
	data, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	temporary := filepath.Join(directory, "comparison.json.next")
	if err := os.WriteFile(temporary, append(data, '\n'), 0644); err != nil {
		return err
	}
	return os.Rename(temporary, filepath.Join(directory, "comparison.json"))
}

func validTiming(r nccltest.Result) bool {
	if r.GPUTime <= 0 {
		return false
	}
	for _, value := range []float64{r.AlgorithmGBs, r.BusGBs} {
		if value < 0 || math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

// Replay checks reset and inspect the same receive elements as ordinary checks.
func validGraphResult(r nccltest.Result, launches int) bool {
	if r.GraphLaunches != launches || r.GraphWrong != 0 {
		return false
	}
	if launches == 0 {
		return r.GraphComparisons == 0
	}
	return r.GraphComparisons > 0 && r.Comparisons%2 == 0 && r.GraphComparisons == r.Comparisons/2
}

// CheckBytes counts only the root's output for Reduce and Gather.
func validComparisonCount(r nccltest.Result) bool {
	if r.Comparisons <= 0 || r.Ranks <= 0 || r.Checks <= 0 || r.Layout.RecvCount <= 0 || r.GraphLaunches < 0 {
		return false
	}
	ranks := r.Ranks
	if r.Case.Operation == nccltest.Reduce || r.Case.Operation == nccltest.Gather {
		ranks = 1
	}
	passes := 1
	if r.GraphLaunches > 0 {
		passes = 2
	}
	// Divide instead of multiplying untrusted receipt fields, avoiding overflow.
	n := r.Comparisons
	for _, factor := range []int{ranks, r.Checks, passes, r.Layout.RecvCount} {
		if n%factor != 0 {
			return false
		}
		n /= factor
	}
	return n == 1
}

func validateOffsetSizes(sizes []int, offset int) error {
	// Reference allocation reserves eight bytes per leading element even
	// when the selected datatype is narrower.
	limit := int(^uint(0) >> 1)
	if offset < 0 || offset > (1<<31-1)/8 {
		return fmt.Errorf("offset exceeds reference allocation range")
	}
	for _, size := range sizes {
		if size <= 0 || size > limit-8*offset {
			return fmt.Errorf("size %d with offset %d overflows allocation", size, offset)
		}
	}
	return nil
}

func sameCase(a, b nccltest.Case) bool {
	if (a.Spread == nil) != (b.Spread == nil) {
		return false
	}
	if a.Spread != nil && *a.Spread != *b.Spread {
		return false
	}
	a.Spread, b.Spread = nil, nil
	return a == b
}
