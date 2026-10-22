//go:build linux

package main

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// This model-free gate is opt-in. A sealed outer admission driver must verify
// the live lease, tools, sources and common context receipt before enabling it.
// GPU_LOCK_HELD by itself is not proof of a lease and does not qualify results.
func TestShapeAttentionNativeValues(t *testing.T) {
	if os.Getenv("DS4_SHAPE_NATIVE") != "1" {
		t.Skip("native gate not requested")
	}
	if os.Getenv("DS4_SHAPE_ADMISSION") == "" {
		t.Fatal("missing reviewed admission")
	}
	receipts := os.Getenv("DS4_SHAPE_RECEIPT")
	if receipts == "" {
		t.Fatal("missing sealed output directory")
	}
	owner := new(shapeStateNativeOwner)
	device := shapeNativeDevice(t, receipts, "attention", owner)
	var free uint
	if err := device.WithContext(func() error { var err error; free, _, err = driver.MemGetInfo(); return err }); err != nil {
		t.Fatal(err)
	}
	if free < 64<<20 {
		t.Fatal("native buffer guard below64MiB")
	}
	source := cudaShapeAttentionSource
	if p := os.Getenv("DS4_SHAPE_ATTENTION_NATIVE_SOURCE"); p != "" {
		b, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		source = string(b)
	}
	program, err := nvrtcx.Compile(source, nvrtcx.WithName("shape_attention_value.cu"), nvrtcx.WithDeviceArch(device), nvrtcx.WithNoCache())
	if err != nil {
		t.Fatal(err)
	}
	if len(program.CUBIN()) == 0 {
		t.Fatal("missing actual CUBIN")
	}
	shapeNativeMappedCompiler(t, receipts, "attention")
	t.Log("ATTENTION_COMPILE_PASS")
	if err := os.WriteFile(filepath.Join(receipts, "attention.cubin"), program.CUBIN(), 0600); err != nil {
		t.Fatal(err)
	}
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
	if err := device.WithContext(func() error {
		var err error
		entry, err = module.Function("ds4_shape_attention_decode_mixed")
		return err
	}); err != nil {
		t.Fatal(err)
	}
	if entry == 0 {
		t.Fatal("missing entry")
	}
	t.Log("ATTENTION_LOAD_PASS")
	upload := func(t *testing.T, values []float32) cudamem.Slice[float32] {
		t.Helper()
		buffer, err := cudamem.Alloc[float32](device, len(values))
		owner.retainAllocation(buffer.Ptr() != 0, buffer.Free, err)
		if err != nil {
			t.Fatal(err)
		}
		if err := buffer.CopyFromHost(values); err != nil {
			t.Fatal(err)
		}
		return buffer
	}
	for _, tc := range []struct {
		name                          string
		heads, ratio, position, lanes int
	}{
		{"flash-lanes0", 64, 4, 7, 0}, {"flash-lanes4", 64, 4, 7, 4}, {"flash-lanes8", 64, 4, 7, 8},
		{"pro-lanes0", 128, 128, 255, 0}, {"pro-lanes4", 128, 128, 255, 4}, {"pro-lanes8", 128, 128, 255, 8},
	} {
		t.Run(tc.name, func(t *testing.T) {
			q, raw, comp, sinks, mask := shapeAttentionFixture(tc.heads)
			want := shapeAttentionReference(q, raw, comp, sinks, mask, tc.heads, 512)
			query, rawKV, compKV, sink, add := upload(t, q), upload(t, raw), upload(t, comp), upload(t, sinks), upload(t, mask)
			values := make([]float32, tc.heads*512)
			for i := range values {
				values[i] = float32(math.NaN())
			}
			out := upload(t, values)
			if err := device.WithContext(func() error {
				if err := cudalaunch.Kernel(entry).Grid(1, uint32(tc.heads), 1).Block(512, 1, 1).Launch(out, sink, query, rawKV, compKV, add, uint32(1), uint32(1), uint32(tc.position), uint32(3), uint32(3), uint32(0), uint32(2), uint32(128), uint32(tc.ratio), uint32(tc.heads), uint32(512), uint32(tc.lanes)); err != nil {
					return err
				}
				return driver.CtxSynchronize()
			}); err != nil {
				t.Fatal(err)
			}
			if err := out.CopyToHost(values); err != nil {
				t.Fatal(err)
			}
			bits := func(values []float32) []uint32 {
				out := make([]uint32, len(values))
				for i, v := range values {
					out[i] = math.Float32bits(v)
				}
				return out
			}
			receipt := struct {
				SourceSHA256, CUBINSHA256               string
				Heads, Width, Ratio, Position, Lanes    int
				Q, Raw, Compressed, Sinks, Mask, Output []uint32
				Reference                               []float64
				AbsLimit, RelativeLimit                 float64
			}{fmt.Sprintf("%x", sha256.Sum256([]byte(source))), fmt.Sprintf("%x", sha256.Sum256(program.CUBIN())), tc.heads, 512, tc.ratio, tc.position, tc.lanes, bits(q), bits(raw), bits(comp), bits(sinks), bits(mask), bits(values), want, 2e-6, 2e-5}
			encoded, err := json.MarshalIndent(receipt, "", "  ")
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(receipts, tc.name+"-attention-values.json"), encoded, 0600); err != nil {
				t.Fatal(err)
			}
			if len(values) != len(want) {
				t.Fatal("ATTENTION_OUTPUT_COUNT")
			}
			for i, v := range values {
				limit := 2e-6 + 2e-5*math.Abs(want[i])
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.Abs(float64(v)-want[i]) > limit {
					t.Fatalf("ATTENTION_VALUE_MISMATCH coordinate=%d got=%g want=%g limit=%g", i, v, want[i], limit)
				}
			}
			t.Logf("ATTENTION_VALUE_PASS heads=%d lanes=%d width=512 output_count=%d", tc.heads, tc.lanes, len(values))
		})
	}
}
