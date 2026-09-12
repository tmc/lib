package nccltest

import (
	"context"
	"errors"
	"os"
	"testing"
)

func requireCUDA(t testing.TB) {
	t.Helper()
	if os.Getenv("NCCLTEST_CUDA") != "1" {
		t.Skip("set NCCLTEST_CUDA=1 to require real CUDA/NCCL execution")
	}
}

func TestCollectives(t *testing.T) {
	requireCUDA(t)
	for _, datatype := range []DataType{Int8, Uint8, Int32, Uint32, Int64, Uint64, Float16, BFloat16, Float8E4M3, Float8E5M2, Float32, Float64} {
		t.Run(string(datatype), func(t *testing.T) { testCollectives(t, datatype) })
	}
}

func testCollectives(t *testing.T, datatype DataType) {
	for _, op := range []Operation{AllReduce, AllGather, Broadcast, ReduceScatter, Reduce, AllToAll, AllToAllV, Scatter, Gather, SendRecv, Hypercube} {
		t.Run(string(op), func(t *testing.T) {
			for _, inPlace := range []bool{false, true} {
				if inPlace && (op == AllToAll || op == AllToAllV || op == SendRecv) {
					continue
				}
				name := "out_of_place"
				if inPlace {
					name = "in_place"
				}
				t.Run(name, func(t *testing.T) {
					result, err := Run(Case{Operation: op, DataType: datatype, Bytes: 65536, InPlace: inPlace}, Options{Devices: []int32{0}, Checks: 2, Warmup: 5, Iterations: 20})
					if err != nil {
						t.Fatal(err)
					}
					if result.Comparisons <= 0 || result.Wrong != 0 {
						t.Fatalf("no passing comparisons: %+v", result)
					}
					t.Logf("ranks=%d comparisons=%d GPU=%s algbw=%g busbw=%g", result.Ranks, result.Comparisons, result.GPUTime, result.AlgorithmGBs, result.BusGBs)
				})
			}
		})
	}
}

// BenchmarkCollectives reports CUDA-event metrics rather than the testing
// framework's host ns/op, which includes initialization and correctness checks.
// Use -benchtime=20x for a fixed count matching nccl-tests -n 20.
func BenchmarkCollectives(b *testing.B) {
	requireCUDA(b)
	for _, datatype := range []DataType{Int8, Uint8, Int32, Uint32, Int64, Uint64, Float16, BFloat16, Float8E4M3, Float8E5M2, Float32, Float64} {
		b.Run(string(datatype), func(b *testing.B) { benchmarkCollectives(b, datatype) })
	}
}

func benchmarkCollectives(b *testing.B, datatype DataType) {
	for _, op := range []Operation{AllReduce, AllGather, Broadcast, ReduceScatter, Reduce, AllToAll, AllToAllV, Scatter, Gather, SendRecv, Hypercube} {
		b.Run(string(op), func(b *testing.B) {
			result, err := Run(Case{Operation: op, DataType: datatype, Bytes: 65536}, Options{Devices: []int32{0}, Checks: 2, Warmup: 5, Iterations: b.N})
			if err != nil {
				b.Fatal(err)
			}
			b.ReportMetric(0, "ns/op")
			b.ReportMetric(float64(result.GPUTime), "gpu-ns/op")
			b.ReportMetric(result.AlgorithmGBs, "alg-GB/s")
			b.ReportMetric(result.BusGBs, "bus-GB/s")
		})
	}
}

func TestMissingCollective(t *testing.T) {
	requireCUDA(t)
	for _, datatype := range []DataType{Int8, Uint8, Int32, Uint32, Int64, Uint64, Float16, BFloat16, Float8E4M3, Float8E5M2, Float32, Float64} {
		t.Run(string(datatype), func(t *testing.T) {
			result, err := run(Case{Operation: AllReduce, DataType: datatype, Bytes: 65536}, Options{Devices: []int32{0}, Checks: 1, Iterations: 1}, true)
			want := 65536 / datatype.Size()
			if err == nil || result.Comparisons != want || result.Wrong != want {
				t.Fatalf("omitted GPU operation was not rejected: result=%+v err=%v", result, err)
			}
			t.Logf("omitted collective rejected: %d/%d wrong", result.Wrong, result.Comparisons)
		})
	}
}

func TestAggregation(t *testing.T) {
	requireCUDA(t)
	result, err := Run(Case{Operation: AllReduce, Bytes: 65536}, Options{Devices: []int32{0}, Warmup: 5, Iterations: 20, Aggregation: 4, Checks: 2})
	if err != nil {
		t.Fatal(err)
	}
	if result.Aggregation != 4 || result.TimedOperations != 80 || result.Comparisons != 32768 || result.Wrong != 0 || result.GPUTime <= 0 {
		t.Fatalf("invalid aggregated result: %+v", result)
	}
	t.Logf("aggregation=%d operations=%d comparisons=%d GPU/op=%s", result.Aggregation, result.TimedOperations, result.Comparisons, result.GPUTime)
}

func TestPreMulSum(t *testing.T) {
	requireCUDA(t)
	for _, datatype := range []DataType{Int8, Uint8, Int32, Uint32, Int64, Uint64, Float16, BFloat16, Float8E4M3, Float8E5M2, Float32, Float64} {
		t.Run(string(datatype), func(t *testing.T) { testPreMulSum(t, datatype) })
	}
}

func testPreMulSum(t *testing.T, datatype DataType) {
	for _, op := range []Operation{AllReduce, Reduce, ReduceScatter} {
		t.Run(string(op), func(t *testing.T) {
			result, err := Run(Case{Operation: op, DataType: datatype, Reduction: PreMulSum, Bytes: 65536}, Options{Devices: []int32{0}, Warmup: 5, Iterations: 20, Aggregation: 4, Checks: 2})
			if err != nil {
				t.Fatal(err)
			}
			if result.Comparisons != 2*65536/datatype.Size() || result.Wrong != 0 || result.TimedOperations != 80 {
				t.Fatalf("invalid premul result: %+v", result)
			}
			t.Logf("comparisons=%d timed operations=%d", result.Comparisons, result.TimedOperations)
		})
	}
}

func TestDefaultStream(t *testing.T) {
	requireCUDA(t)
	for _, op := range []Operation{AllReduce, Hypercube} {
		c := Case{Operation: op, Bytes: 65536}
		o := Options{DefaultStream: true, Devices: []int32{0}, Checks: 2, Warmup: 2, Iterations: 10}
		result, err := Run(c, o)
		if err != nil {
			t.Fatal(err)
		}
		if !result.DefaultStream || result.Comparisons <= 0 || result.Wrong != 0 {
			t.Fatalf("invalid result: %+v", result)
		}
		if _, err := run(c, o, true); err == nil {
			t.Fatal("omitted default-stream collective accepted")
		}
	}
}

func TestRegisteredBuffers(t *testing.T) {
	requireCUDA(t)
	for _, inPlace := range []bool{false, true} {
		c := Case{Operation: AllReduce, Bytes: 65536, InPlace: inPlace}
		o := Options{Register: true, Devices: []int32{0}, Checks: 2, Warmup: 2, Iterations: 10}
		result, err := Run(c, o)
		if err != nil {
			t.Fatal(err)
		}
		if !result.Register || result.Comparisons <= 0 || result.Wrong != 0 {
			t.Fatalf("invalid result: %+v", result)
		}
		// A one-rank in-place sum is the identity: the initial input already
		// equals the expected output. Only disjoint receive poison detects omission.
		if !inPlace {
			omitted, err := run(c, o, true)
			if err == nil || omitted.Comparisons != 32768 || omitted.Wrong != omitted.Comparisons {
				t.Fatalf("registered omission control: result=%+v err=%v", omitted, err)
			}
		}
	}
}

func TestGraphCollectives(t *testing.T) {
	requireCUDA(t)
	for _, op := range []Operation{AllReduce, AllGather, Broadcast, ReduceScatter, Reduce, AllToAll, AllToAllV, Scatter, Gather, SendRecv, Hypercube} {
		t.Run(string(op), func(t *testing.T) {
			c := Case{Operation: op, Bytes: 1024}
			r, err := Run(c, Options{Devices: []int32{0}, Warmup: 1, Iterations: 2, Aggregation: 2, Checks: 1, GraphLaunches: 3})
			if err != nil {
				t.Fatal(err)
			}
			if r.TimedOperations != 12 || r.GraphLaunches != 3 || r.GraphComparisons <= 0 || r.Comparisons != 2*r.GraphComparisons || r.Wrong != 0 || r.GraphWrong != 0 {
				t.Fatalf("incomplete graph evidence: %+v", r)
			}
		})
	}
}

func TestGraphMissingCollective(t *testing.T) {
	requireCUDA(t)
	// Disjoint buffers make omission observable even for a one-rank identity.
	c := Case{Operation: AllReduce, Bytes: 1024}
	r, err := run(c, Options{Devices: []int32{0}, Iterations: 1, Checks: 1, GraphLaunches: 1}, true)
	if err == nil {
		t.Fatal("omitted graph collective accepted")
	}
	// A capture failure is not evidence that the checker rejected omission.
	if r.GraphComparisons != 256 || r.GraphWrong != 256 || r.Comparisons != 512 || r.Wrong != 512 {
		t.Fatalf("omission did not reach both checkers: %+v; error: %v", r, err)
	}
}

func TestBufferOffsets(t *testing.T) {
	requireCUDA(t)
	for _, datatype := range []DataType{Uint8, Float16, Float32, Float64} {
		t.Run(string(datatype), func(t *testing.T) {
			for _, inPlace := range []bool{false, true} {
				c := Case{Operation: AllReduce, DataType: datatype, Bytes: 1024, InPlace: inPlace}
				o := Options{Devices: []int32{0}, Offset: 3, Iterations: 2, Warmup: 1, Checks: 2}
				r, err := Run(c, o)
				want := 2 * 1024 / datatype.Size()
				if err != nil || r.Offset != 3 || r.Comparisons != want || r.Wrong != 0 {
					t.Fatalf("inPlace=%v: result=%+v error=%v", inPlace, r, err)
				}
				if !inPlace {
					r, err = run(c, o, true)
					if err == nil || r.Comparisons != want || r.Wrong != want {
						t.Fatalf("shifted omission: result=%+v error=%v", r, err)
					}
				}
			}
		})
	}
}

func TestCycles(t *testing.T) {
	requireCUDA(t)
	for _, mode := range []string{"ordinary", "graph", "registered_offset"} {
		t.Run(mode, func(t *testing.T) {
			o := Options{Devices: []int32{0}, Iterations: 2, Warmup: 1, Checks: 2}
			if mode == "graph" {
				o.GraphLaunches = 2
			}
			if mode == "registered_offset" {
				o.Register = true
				o.Offset = 3
			}
			c := Case{Operation: AllReduce, Bytes: 1024}
			seen := 0
			err := RunCycles(c, o, 3, func(r Result) error {
				seen++
				want := 512
				if o.GraphLaunches > 0 {
					want *= 2
				}
				if r.Comparisons != want || r.Wrong != 0 || r.Offset != o.Offset || r.Register != o.Register || r.GraphLaunches != o.GraphLaunches {
					t.Fatalf("cycle %d: %+v", seen, r)
				}
				return nil
			})
			if err != nil || seen != 3 {
				t.Fatalf("cycles=%d error=%v", seen, err)
			}
		})
	}
}

func TestCyclesCallbackStops(t *testing.T) {
	requireCUDA(t)
	stop := errors.New("stop after second cycle")
	seen := 0
	err := RunCycles(Case{Operation: AllReduce, Bytes: 1024}, Options{Devices: []int32{0}, Iterations: 1, Checks: 1}, 3, func(r Result) error {
		seen++
		if r.Comparisons != 256 || r.Wrong != 0 {
			t.Fatalf("invalid cycle %+v", r)
		}
		if seen == 2 {
			return stop
		}
		return nil
	})
	if !errors.Is(err, stop) || seen != 2 {
		t.Fatalf("callbacks=%d error=%v", seen, err)
	}
}

func TestSweep(t *testing.T) {
	requireCUDA(t)
	sizes := []int{1024, 16, 64}
	for _, op := range []Operation{AllReduce, AllGather, Broadcast, ReduceScatter, Reduce, AllToAll, AllToAllV, Scatter, Gather, SendRecv, Hypercube} {
		t.Run(string(op), func(t *testing.T) {
			for _, inPlace := range []bool{false, true} {
				if inPlace && (op == AllToAll || op == AllToAllV || op == SendRecv) {
					continue
				}
				seen := 0
				err := RunSweep(Case{Operation: op, InPlace: inPlace}, Options{Devices: []int32{0}, Offset: 3, Iterations: 2, Warmup: 1, Checks: 1}, sizes, 2, func(r Result) error {
					if r.Case.Bytes != sizes[seen%len(sizes)] || r.Comparisons != r.Layout.RecvCount || r.Wrong != 0 || r.Offset != 3 {
						t.Fatalf("sample %d: %+v", seen, r)
					}
					seen++
					return nil
				})
				if err != nil || seen != 6 {
					t.Fatalf("inPlace=%v samples=%d err=%v", inPlace, seen, err)
				}
			}
		})
	}
}

func TestSynchronizedIterations(t *testing.T) {
	requireCUDA(t)
	for _, aggregation := range []int{1, 4} {
		r, err := Run(Case{Operation: AllReduce, Bytes: 1024}, Options{Devices: []int32{0}, Iterations: 3, Aggregation: aggregation, Checks: 2, Warmup: 1, SyncIterations: true})
		if err != nil || !r.SyncIterations || r.TimedOperations != 3*aggregation || r.Comparisons != 512 || r.Wrong != 0 {
			t.Fatalf("aggregation=%d result=%+v error=%v", aggregation, r, err)
		}
	}
}

func TestSweepContextCancellationCUDA(t *testing.T) {
	requireCUDA(t)
	for _, graph := range []int{0, 2} {
		ctx, cancel := context.WithCancel(context.Background())
		seen := 0
		o := Options{Devices: []int32{0}, Iterations: 2, Checks: 1, GraphLaunches: graph}
		err := RunSweepContext(ctx, Case{Operation: AllReduce}, o, []int{16, 64}, 0, func(r Result) error {
			if r.Case.Bytes != []int{16, 64}[seen%2] || r.Wrong != 0 || r.Comparisons <= 0 {
				t.Errorf("invalid sample: %+v", r)
			}
			seen++
			if seen == 3 {
				cancel()
			}
			return nil
		})
		cancel()
		// Cleanup wraps cancellation with errors.Join even when it succeeds.
		// Require cancellation alone, without an appended cleanup failure.
		if !errors.Is(err, context.Canceled) || err.Error() != context.Canceled.Error() || seen != 3 {
			t.Fatalf("graphs=%d samples=%d error=%v", graph, seen, err)
		}
		if _, err := Run(Case{Operation: AllReduce, Bytes: 16}, o); err != nil {
			t.Fatalf("run after cancellation: %v", err)
		}
	}
}

func TestGeneratedSpreadCUDA(t *testing.T) {
	requireCUDA(t)
	for _, value := range []float64{0, 0.5, 1} {
		spread := value
		c := Case{Operation: AllToAllV, Bytes: 256, Spread: &spread}
		o := Options{Devices: []int32{0}, Iterations: 2, Checks: 1}
		seen := 0
		err := RunSweep(c, o, []int{256, 1024}, 2, func(r Result) error {
			if r.Case.Spread == nil || *r.Case.Spread != value || r.Comparisons != r.Case.Bytes/4 || r.Wrong != 0 {
				t.Errorf("spread=%g result=%+v", value, r)
			}
			seen++
			// Neither the input nor a yielded result may change later samples.
			spread = 1 - value
			if r.Case.Spread != nil {
				*r.Case.Spread = 1 - value
			}
			return nil
		})
		if err != nil || seen != 4 {
			t.Fatalf("spread=%g samples=%d error=%v", value, seen, err)
		}
		spread = value
		r, err := run(c, o, true)
		if err == nil || r.Comparisons != 64 || r.Wrong == 0 {
			t.Fatalf("omitted spread=%g result=%+v error=%v", value, r, err)
		}
	}
}
