//go:build native

package cudafuse

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

// indexedNativeDevice refuses an absent lease or noncanonical UUID before
// loading the driver. Enumeration does not create contexts on other devices.
func indexedNativeDevice(t *testing.T) *cuda.Device {
	t.Helper()
	if os.Getenv("CUDAFUSE_TRANSPOSE_NATIVE") != "1" {
		t.Skip("transpose native fixture requires explicit opt-in")
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Fatal(err)
	}
	selected := os.Getenv("CUDAFUSE_TRANSPOSE_UUID")
	raw := strings.ReplaceAll(strings.TrimPrefix(selected, "GPU-"), "-", "")
	bytes, err := hex.DecodeString(raw)
	if err != nil || len(bytes) != 16 {
		t.Fatal("transpose native fixture requires a canonical GPU UUID")
	}
	canonical := fmt.Sprintf("GPU-%x-%x-%x-%x-%x", bytes[:4], bytes[4:6], bytes[6:8], bytes[8:10], bytes[10:])
	if selected != canonical {
		t.Fatal("transpose native fixture requires a canonical GPU UUID")
	}
	if err := driver.Load(""); err != nil {
		t.Fatal(err)
	}
	if err := driver.Init(0); err != nil {
		t.Fatal(err)
	}
	count, err := driver.DeviceGetCount()
	if err != nil {
		t.Fatal(err)
	}
	ordinal := -1
	for i := 0; i < count; i++ {
		d, err := driver.DeviceGet(i)
		if err != nil {
			t.Fatal(err)
		}
		uuid, err := d.Uuid()
		if err != nil {
			t.Fatal(err)
		}
		b := uuid.Bytes[:]
		if selected == fmt.Sprintf("GPU-%x-%x-%x-%x-%x", b[:4], b[4:6], b[6:8], b[8:10], b[10:]) {
			if ordinal != -1 {
				t.Fatal("duplicate selected UUID")
			}
			ordinal = i
		}
	}
	if ordinal == -1 {
		t.Fatal("selected GPU UUID not found")
	}
	t.Logf("selected exact device UUID=%s ordinal=%d", selected, ordinal)
	dev, err := cuda.NewDevice(ordinal)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := dev.Close(); err != nil {
			t.Error(err)
		}
	})
	return dev
}

func TestIndexedExecutionNativeTranspose(t *testing.T) {
	dev := indexedNativeDevice(t)
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := stream.Close(); err != nil {
			t.Error(err)
		}
	})
	ctx := NewLazyContext()
	x := NewLazyArray(ctx, []int{17, 59}, cudair.Float32)
	y := NewLazyArray(ctx, []int{59, 17}, cudair.Float32)
	root := x.Transpose([]int{1, 0}).Map("neg").Add(y)
	plan, err := root.Materialize()
	if err != nil {
		t.Fatal(err)
	}
	nodes, count, err := executionNodes(plan)
	if err != nil {
		t.Fatal(err)
	}
	a, b := make([]float32, count), make([]float32, count)
	for i := range a {
		a[i], b[i] = float32(i%31-15)/4, float32(i%19+1)/2
	}
	want := transposeOracle(a, x.Shape, []int{1, 0})
	for i := range want {
		want[i] = -want[i] + b[i]
	}
	bindings := make(map[string]ExecutionBinding)
	var storages []cudamem.Slice[float32]
	var initial [][]float32
	inputHash := sha256.New()
	for i, input := range []*LazyArray{x, y} {
		values := [][]float32{a, b}[i]
		storage, err := cudamem.Alloc[float32](dev, count+2)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := storage.Free(); err != nil {
				t.Error(err)
			}
		})
		host := make([]float32, count+2)
		host[0], host[count+1] = -9876, 9876
		copy(host[1:], values)
		if err := storage.CopyFromHost(host); err != nil {
			t.Fatal(err)
		}
		storages = append(storages, storage)
		initial = append(initial, slices.Clone(host))
		bindings[input.Node.Name()] = ExecutionBinding{Buffer: storage.Slice(1, count+1), Shape: input.Shape, DType: cudair.Float32}
		for _, value := range values {
			var bits [4]byte
			binary.LittleEndian.PutUint32(bits[:], math.Float32bits(value))
			inputHash.Write(bits[:])
		}
	}
	for _, eager := range []bool{false, true} {
		t.Run(fmt.Sprintf("eager=%v", eager), func(t *testing.T) {
			var execution *Execution
			var err error
			if eager {
				execution, err = plan.ExecuteEager(context.Background(), stream, bindings)
			} else {
				execution, err = plan.Execute(context.Background(), stream, bindings)
			}
			if execution != nil {
				t.Cleanup(func() {
					if err := execution.Close(); err != nil {
						t.Error(err)
					}
				})
			}
			if err != nil {
				t.Fatal(err)
			}
			out, err := execution.Readback(context.Background())
			if err != nil {
				t.Fatal(err)
			}
			calls := 1
			if eager {
				calls = 3
			}
			receipts := execution.Receipts()
			if len(receipts) != calls || execution.Stats() != (ExecutionStats{Allocations: calls, Launches: calls}) {
				t.Fatal("native indexed dispatch count differs")
			}
			sourceHash := sha256.New()
			for i, receipt := range receipts {
				output := root.Node
				if eager {
					var operations []*LazyNode
					for _, node := range nodes {
						if node.Op != "input" {
							operations = append(operations, node)
						}
					}
					output = operations[i]
				}
				kernel, err := executionSource(nodes, output, receipt.Name, eager)
				if err != nil {
					t.Fatal(err)
				}
				if receipt.SourceDigest != fmt.Sprintf("%x", sha256.Sum256([]byte(kernel.Source))) || len(receipt.PTXDigest) != 64 || len(receipt.CompilerKey) != 64 {
					t.Fatal("native indexed load receipt differs from source")
				}
				sourceHash.Write([]byte(receipt.SourceDigest))
			}
			policy := cudatest.CUDAOutputPolicy{Target: receipts[0].Target, DType: "fp32", MathMode: "--fmad=false; no fast math", SourceDigest: fmt.Sprintf("%x", sourceHash.Sum(nil)), InputDigest: fmt.Sprintf("%x", inputHash.Sum(nil)), Elements: count, NaN: "reject", Infinity: "reject", Zero: "equal"}
			if err := cudatest.CompareCUDAOutputFP32(out, want, policy); err != nil {
				t.Fatal(err)
			}
			if err := execution.Close(); err != nil {
				t.Fatal(err)
			}
			if execution.Stats().Frees != calls {
				t.Fatal("native indexed output ownership not retired")
			}
			for i, storage := range storages {
				got := make([]float32, count+2)
				if err := storage.CopyToHost(got); err != nil {
					t.Fatal(err)
				}
				if !slices.Equal(got, initial[i]) {
					t.Fatal("native indexed input or guard modified")
				}
			}
			retainNativeExecutionPacket(t, execution, out, want, policy)
		})
	}
}
