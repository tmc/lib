//go:build linux

package main

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func shapeNativeUpload[T any](t *testing.T, device *cuda.Device, owner *shapeStateNativeOwner, values []T) cudamem.Slice[T] {
	t.Helper()
	var zero T
	switch any(zero).(type) {
	case float32, int32, uint32:
	default:
		t.Fatal("native fixture requires32bit elements")
	}
	if len(values) == 0 || len(values) > 1<<20 {
		t.Fatal("native fixture allocation envelope")
	}
	buffer, err := cudamem.Alloc[T](device, len(values))
	owner.retainAllocation(buffer.Ptr() != 0, buffer.Free, err)
	if err != nil {
		t.Fatal(err)
	}
	if err := buffer.CopyFromHost(values); err != nil {
		t.Fatal(err)
	}
	return buffer
}

func TestShapeRouterNativeValues(t *testing.T) {
	if os.Getenv("DS4_SHAPE_NATIVE") != "1" {
		t.Skip("native gate not requested")
	}
	if os.Getenv("DS4_SHAPE_ADMISSION") == "" {
		t.Fatal("missing reviewed admission")
	}
	dir := os.Getenv("DS4_SHAPE_RECEIPT")
	if dir == "" {
		t.Fatal("missing sealed output directory")
	}
	owner := new(shapeStateNativeOwner)
	device := shapeNativeDevice(t, dir, "router", owner)
	source := cudaShapeRouterSource
	if p := os.Getenv("DS4_SHAPE_ROUTER_NATIVE_SOURCE"); p != "" {
		b, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		source = string(b)
	}
	program, err := nvrtcx.Compile(source, nvrtcx.WithName("shape_router_value.cu"), nvrtcx.WithDeviceArch(device), nvrtcx.WithNoCache())
	if err != nil {
		t.Fatal(err)
	}
	if len(program.CUBIN()) == 0 {
		t.Fatal("missing actual CUBIN")
	}
	shapeNativeMappedCompiler(t, dir, "router")
	if err := os.WriteFile(filepath.Join(dir, "router.cubin"), program.CUBIN(), 0600); err != nil {
		t.Fatal(err)
	}
	t.Log("ROUTER_COMPILE_PASS")
	module, _, err := program.LoadOutputWithLog(device, nvrtcx.OutputCUBIN)
	if module != 0 {
		owner.children = append(owner.children, shapeStateNativeCloseFunc(func() error {
			if module == 0 {
				return nil
			}
			return device.WithContext(func() error {
				if err := module.Close(); err != nil {
					return err
				}
				module = 0
				return nil
			})
		}))
	}
	if err != nil {
		owner.retainError(err)
		t.Fatal(err)
	}
	var entry driver.Function
	if err := device.WithContext(func() error { var err error; entry, err = module.Function("ds4_shape_router_select"); return err }); err != nil {
		t.Fatal(err)
	}
	if entry == 0 {
		t.Fatal("missing entry")
	}
	t.Log("ROUTER_LOAD_PASS")
	for _, experts := range []int{256, 384} {
		for _, mode := range []string{"topk", "hash", "nonfinite"} {
			t.Run(fmt.Sprintf("%d-%s", experts, mode), func(t *testing.T) {
				scale := float32(1.5)
				if experts == 384 {
					scale = 2.5
				}
				logits, bias, probabilities := make([]float32, experts), make([]float32, experts), make([]float32, experts)
				reference := make([]float64, experts)
				ids := make([]int, experts)
				for i := range logits {
					logits[i] = float32(i-128) / 32
					bias[i] = float32(i%5-2) / 64
					probabilities[i] = float32(math.NaN())
					reference[i] = math.Sqrt(math.Log1p(math.Exp(float64(logits[i]))))
					ids[i] = i
				}
				sort.SliceStable(ids, func(i, j int) bool {
					return reference[ids[i]]+float64(bias[ids[i]]) > reference[ids[j]]+float64(bias[ids[j]])
				})
				hash := []int32{0, int32(experts - 1), 13, 31, 127, 128}
				hashMode := 0
				if mode == "hash" {
					hashMode = 1
					for i, id := range hash {
						ids[i] = int(id)
					}
				}
				ids = ids[:6]
				wantWeights := make([]float64, 6)
				sum := 0.0
				for _, id := range ids {
					sum += reference[id]
				}
				for i, id := range ids {
					wantWeights[i] = reference[id] / math.Max(sum, 6.103515625e-5) * float64(scale)
				}
				if mode == "nonfinite" {
					logits[experts-1] = float32(math.NaN())
				}
				selected := []int32{-1, -1, -1, -1, -1, -1}
				weights := make([]float32, 6)
				for i := range weights {
					weights[i] = float32(math.NaN())
				}
				l, b, p, h, selectedGPU, w, invalid := shapeNativeUpload(t, device, owner, logits), shapeNativeUpload(t, device, owner, bias), shapeNativeUpload(t, device, owner, probabilities), shapeNativeUpload(t, device, owner, hash), shapeNativeUpload(t, device, owner, selected), shapeNativeUpload(t, device, owner, weights), shapeNativeUpload(t, device, owner, []uint32{0})
				if err := device.WithContext(func() error {
					if err := cudalaunch.Kernel(entry).Grid(1, 1, 1).Block(32, 1, 1).Launch(selectedGPU, w, p, b, h, l, driver.DevicePtr(0), int32(0), uint32(1), uint32(1), int32(1), int32(hashMode), uint32(experts), scale, invalid); err != nil {
						return err
					}
					return driver.CtxSynchronize()
				}); err != nil {
					t.Fatal(err)
				}
				var flag [1]uint32
				for _, read := range []func() error{func() error { return p.CopyToHost(probabilities) }, func() error { return selectedGPU.CopyToHost(selected) }, func() error { return w.CopyToHost(weights) }, func() error { return invalid.CopyToHost(flag[:]) }} {
					if err := read(); err != nil {
						t.Fatal(err)
					}
				}
				bits := func(v []float32) []uint32 {
					out := make([]uint32, len(v))
					for i, x := range v {
						out[i] = math.Float32bits(x)
					}
					return out
				}
				receipt := struct {
					SourceSHA256, CUBINSHA256, Mode          string
					Experts                                  int
					Scale                                    float32
					Logits, Bias, Probabilities, Weights     []uint32
					Hash, Selected                           []int32
					Invalid                                  uint32
					ReferenceProbabilities, ReferenceWeights []float64
					ReferenceSelected                        []int
					AbsLimit, RelativeLimit                  float64
				}{fmt.Sprintf("%x", sha256.Sum256([]byte(source))), fmt.Sprintf("%x", sha256.Sum256(program.CUBIN())), mode, experts, scale, bits(logits), bits(bias), bits(probabilities), bits(weights), hash, selected, flag[0], reference, wantWeights, ids, 2e-6, 2e-6}
				encoded, err := json.MarshalIndent(receipt, "", "  ")
				if err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filepath.Join(dir, fmt.Sprintf("%d-%s-router-values.json", experts, mode)), encoded, 0600); err != nil {
					t.Fatal(err)
				}
				if mode == "nonfinite" {
					if flag[0] != 1 {
						t.Fatalf("ROUTER_FINITE_REFUSAL got=%d", flag[0])
					}
					return
				}
				if flag[0] != 0 {
					t.Fatalf("ROUTER_INVALID got=%d", flag[0])
				}
				for i, want := range reference {
					got := float64(probabilities[i])
					if math.IsNaN(got) || math.IsInf(got, 0) || math.Abs(got-want) > 2e-6+2e-6*math.Abs(want) {
						t.Fatalf("ROUTER_PROBABILITY_MISMATCH expert=%d got=%g want=%g", i, got, want)
					}
				}
				for i, want := range ids {
					if int(selected[i]) != want {
						t.Fatalf("ROUTER_ID_MISMATCH slot=%d got=%d want=%d", i, selected[i], want)
					}
					got := float64(weights[i])
					if math.IsNaN(got) || math.IsInf(got, 0) || math.Abs(got-wantWeights[i]) > 2e-6+2e-6*math.Abs(wantWeights[i]) {
						t.Fatalf("ROUTER_WEIGHT_MISMATCH slot=%d got=%g want=%g", i, got, wantWeights[i])
					}
				}
				t.Logf("ROUTER_VALUE_PASS experts=%d probabilities=%d selected=6 weights=6", experts, len(probabilities))
			})
		}
	}
}
