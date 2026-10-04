//go:build native

package cudafuse

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

type nativeExecutionFixture struct {
	dev         *cuda.Device
	stream      *cuda.Stream
	plan        *MaterializedPlan
	nodes       []*LazyNode
	bindings    map[string]ExecutionBinding
	want        []float32
	inputDigest string
}

func nativeLazyFixture(t *testing.T) *nativeExecutionFixture {
	t.Helper()
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Skip("native lazy execution requires gpu-lock")
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Fatal(err)
	}
	dev := cudatest.RequireDevice(t)
	t.Cleanup(func() {
		if err := dev.Close(); err != nil {
			t.Error(err)
		}
	})
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := stream.Close(); err != nil {
			t.Error(err)
		}
	})
	plan, nodes := executionChain(t, 1003)
	a, b, c, want := executionValues(1003)
	values := [][]float32{a, b, c}
	data := make([]byte, 0, 3*1003*4)
	bindings := map[string]ExecutionBinding{}
	// Names are creation IDs, not topological encounter order (c is encountered
	// after both relus). Keep the three distinct inputs bound to their own names.
	for _, node := range nodes {
		if node.Op != "input" {
			continue
		}
		v := values[node.ID-1]
		storage, err := cudamem.Alloc[float32](dev, 1005)
		t.Cleanup(func() {
			if err := storage.Free(); err != nil {
				t.Error(err)
			}
		})
		if err != nil {
			t.Fatal(err)
		}
		host := make([]float32, 1005)
		copy(host[1:], v)
		if err := storage.CopyFromHost(host); err != nil {
			t.Fatal(err)
		}
		bindings[node.Name()] = ExecutionBinding{Buffer: storage.Slice(1, 1004), Shape: []int{1003}, DType: cudair.Float32}
	}
	for _, v := range values {
		for _, x := range v {
			var bits [4]byte
			binary.LittleEndian.PutUint32(bits[:], math.Float32bits(x))
			data = append(data, bits[:]...)
		}
	}
	return &nativeExecutionFixture{dev: dev, stream: stream, plan: plan, nodes: nodes, bindings: bindings, want: want, inputDigest: fmt.Sprintf("%x", sha256.Sum256(data))}
}

func nativeExecutionPolicy(f *nativeExecutionFixture, e *Execution) cudatest.CUDAOutputPolicy {
	receipts := e.Receipts()
	hash := sha256.New()
	for _, r := range receipts {
		hash.Write([]byte(r.SourceDigest))
	}
	target := ""
	if len(receipts) > 0 {
		target = receipts[0].Target
	}
	return cudatest.CUDAOutputPolicy{Target: target, DType: "fp32", MathMode: "--fmad=false; no fast math", SourceDigest: fmt.Sprintf("%x", hash.Sum(nil)), InputDigest: f.inputDigest, Elements: 1003, NaN: "reject", Infinity: "reject", Zero: "equal"}
}

func retainNativeExecutionPacket(t *testing.T, e *Execution, out, want []float32, policy cudatest.CUDAOutputPolicy) {
	t.Helper()
	t.Logf("actual calls: %+v; compiled/loaded receipts: %+v; oracle: %+v", e.Stats(), e.Receipts(), policy)
	if dir := os.Getenv("CUDAFUSE_EXECUTION_ARTIFACT_DIR"); dir != "" {
		if !filepath.IsAbs(dir) {
			t.Fatal("artifact directory must be absolute")
		}
		if err := os.MkdirAll(dir, 0700); err != nil {
			t.Fatal(err)
		}
		path := filepath.Join(dir, strings.ReplaceAll(t.Name(), "/", "-")+".json")
		file, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
		if err != nil {
			t.Fatal(err)
		}
		err = json.NewEncoder(file).Encode(struct {
			Stats            ExecutionStats
			Receipts         []KernelReceipt
			Policy           cudatest.CUDAOutputPolicy
			Actual, Expected []float32
		}{e.Stats(), e.Receipts(), policy, out, want})
		closeErr := file.Close()
		if err != nil || closeErr != nil {
			t.Fatal(errors.Join(err, closeErr))
		}
	}
}

func TestLazyExecution_ChainMatchesReference1003(t *testing.T) {
	f := nativeLazyFixture(t)
	e, err := f.plan.Execute(context.Background(), f.stream, f.bindings)
	if e != nil {
		t.Cleanup(func() {
			if err := e.Close(); err != nil {
				t.Error(err)
			}
		})
	}
	if err != nil {
		t.Fatal(err)
	}
	out, err := e.Readback(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	policy := nativeExecutionPolicy(f, e)
	if err := cudatest.CompareCUDAOutputFP32(out, f.want, policy); err != nil {
		t.Fatal(err)
	}
	receipts := e.Receipts()
	if len(receipts) != 1 || len(receipts[0].PTXDigest) != 64 || len(receipts[0].CompilerKey) != 64 {
		t.Fatal("missing actual load identity", receipts)
	}
	source, err := executionSource(f.nodes, f.plan.RootNode, receipts[0].Name, false)
	if err != nil {
		t.Fatal(err)
	}
	if receipts[0].SourceDigest != fmt.Sprintf("%x", sha256.Sum256([]byte(source.Source))) {
		t.Fatal("receipt differs from actual graph source")
	}
	if e.Stats() != (ExecutionStats{Allocations: 1, Launches: 1}) {
		t.Fatal("unexpected successful device calls", e.Stats())
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	if e.Stats().Frees != 1 {
		t.Fatal("actual output free not counted")
	}
	retainNativeExecutionPacket(t, e, out, f.want, policy)
}

func TestLazyExecution_FusedAndEagerCounts(t *testing.T) {
	f := nativeLazyFixture(t)
	for _, eager := range []bool{false, true} {
		t.Run(fmt.Sprint(eager), func(t *testing.T) {
			var e *Execution
			var err error
			if eager {
				e, err = f.plan.ExecuteEager(context.Background(), f.stream, f.bindings)
			} else {
				e, err = f.plan.Execute(context.Background(), f.stream, f.bindings)
			}
			if e != nil {
				defer func() {
					if err := e.Close(); err != nil {
						t.Error(err)
					}
				}()
			}
			if err != nil {
				t.Fatal(err)
			}
			out, err := e.Readback(context.Background())
			if err != nil {
				t.Fatal(err)
			}
			policy := nativeExecutionPolicy(f, e)
			if err := cudatest.CompareCUDAOutputFP32(out, f.want, policy); err != nil {
				t.Fatal(err)
			}
			want := 1
			if eager {
				want = 4
			}
			if e.Stats() != (ExecutionStats{Allocations: want, Launches: want}) {
				t.Fatal("actual calls differ", e.Stats())
			}
			if err := e.Close(); err != nil {
				t.Fatal(err)
			}
			if e.Stats().Frees != want {
				t.Fatal("free calls differ", e.Stats())
			}
			retainNativeExecutionPacket(t, e, out, f.want, policy)
		})
	}
}

func TestLazyExecution_NativeOutputControls(t *testing.T) {
	f := nativeLazyFixture(t)
	for _, control := range []string{"wrong-second-operand", "omit-last-write", "write-past-tail", "omit-output-write"} {
		t.Run(control, func(t *testing.T) {
			ops := nativeExecutionOps(f.dev, f.stream)
			compile := ops.compile
			ops.compile = func(source FusedKernel) (*executionKernel, error) {
				old := source.Source
				switch control {
				case "wrong-second-operand":
					source.Source = strings.ReplaceAll(source.Source, "node_2_input[idx]", "node_1_input[idx]")
				case "omit-last-write":
					source.Source = strings.Replace(source.Source, "idx < (unsigned int)n;", "idx < (unsigned int)(n - 1);", 1)
				case "write-past-tail":
					source.Source = strings.Replace(source.Source, "idx < (unsigned int)n;", "idx < (unsigned int)(n + 1);", 1)
				case "omit-output-write":
					lines := strings.Split(source.Source, "\n")
					for i, line := range lines {
						if strings.Contains(line, source.Outputs[0]+"[idx] =") {
							lines[i] = "        // intentionally omit output write"
						}
					}
					source.Source = strings.Join(lines, "\n")
				}
				if source.Source == old {
					return nil, errors.New("control did not change source")
				}
				return compile(source)
			}
			pointers := map[*LazyNode]driver.DevicePtr{}
			var releases []func()
			for _, node := range f.nodes {
				if node.Op == "input" {
					b := f.bindings[node.Name()].Buffer
					release, err := b.Borrow(f.dev)
					if err != nil {
						t.Fatal(err)
					}
					releases = append(releases, release)
					pointers[node] = b.Ptr()
				}
			}
			e, err := runExecution(context.Background(), f.nodes, 1003, pointers, false, ops, releases)
			if e != nil {
				defer func() {
					if err := e.Close(); err != nil {
						t.Error(err)
					}
				}()
			}
			if err != nil {
				t.Fatal("control failed before output oracle", err)
			}
			out, err := e.Readback(context.Background())
			if control == "wrong-second-operand" {
				if err != nil {
					t.Fatal(err)
				}
				var mismatch *cudatest.CUDAOutputMismatch
				if err := cudatest.CompareCUDAOutputFP32(out, f.want, nativeExecutionPolicy(f, e)); !errors.As(err, &mismatch) {
					t.Fatal("wrong operand accepted", err)
				}
			} else {
				expected := ErrOutputUnwritten
				if control == "write-past-tail" {
					expected = ErrOutputGuard
				}
				if !errors.Is(err, expected) {
					t.Fatal("control did not reach the declared output oracle", err)
				}
			}
		})
	}
}
