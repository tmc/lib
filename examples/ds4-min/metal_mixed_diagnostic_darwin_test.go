//go:build darwin && arm64

package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/apple/objectivec"
	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

// This diagnostic retains numerical failures; its PASS means that stage
// extraction preserves production output, not that the numeric gate passed.
func TestMetalMixedCapacityStages(t *testing.T) {
	dir := os.Getenv("DS4_METAL_MIXED_DIAGNOSTIC_DIR")
	if dir == "" {
		t.Skip("opt-in diagnostic output directory required")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("gpu-lock required")
	}
	if err := mixedDiagnosticDirectory(dir); err != nil {
		t.Fatal(err)
	}
	dev, err := libmetal.NewDevice()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	ops, err := newMetalPinnedOps(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer ops.Close()
	const noop = `
kernel void capacity_no_unrotate(device float *out [[buffer(0)]],
  constant uint &position [[buffer(1)]], uint head [[thread_position_in_threadgroup]]) {
  (void)out; (void)position; (void)head;
}
`
	library, err := dev.CompileLibrary(metalPinnedMSL + noop)
	if err != nil {
		t.Fatal(err)
	}
	defer objectivec.ObjectFromID(library.GetID()).Release()
	pipeline, err := dev.NewComputePipeline(library, "capacity_no_unrotate")
	if err != nil {
		t.Fatal(err)
	}
	defer objectivec.ObjectFromID(pipeline.GetID()).Release()
	t.Logf("production_source_sha256=%x noop_source_sha256=%x", sha256.Sum256([]byte(metalPinnedMSL)), sha256.Sum256([]byte(metalPinnedMSL+noop)))
	for _, fixture := range []struct{ position, count int }{{127, 1}, {2048, 0}, {2048, 1}} {
		if !t.Run(fmt.Sprintf("position_%d/compressed_%d", fixture.position, fixture.count), func(t *testing.T) {
			arena := metalmem.NewArena(dev)
			defer arena.Free()
			position, count := fixture.position, fixture.count
			query, current, sinks := mixedCapacityInputs(position)
			rotated := append([]float32(nil), query...)
			for head := range 64 {
				mixedCapacityRotate(rotated[head*512:(head+1)*512], position, false)
			}
			currentRotated := append([]float32(nil), current...)
			mixedCapacityRotate(currentRotated, position, false)
			stored := rawCapacityStoredRow(currentRotated)
			history := rawCapacityHistory(position, 128)
			cache := make([]float32, 128*512)
			for i := range cache {
				cache[i] = 13
			}
			for i, row := range history {
				copy(cache[((position-len(history)+i)%128)*512:], row)
			}
			wantCache := append([]float32(nil), cache...)
			copy(wantCache[(position%128)*512:], stored)
			history = append(history, stored)
			compressed, mask := mixedCapacityRows(count)
			flat := make([]float32, max(1, count)*512)
			for i, row := range compressed {
				copy(flat[i*512:], row)
			}
			if count == 0 {
				mask = []uint32{0}
			}
			wantFinal, wantScores, wantPre := mixedCapacityAttentionStages(rotated, history, compressed, mask, sinks, position)
			q := metalmem.UploadIn(arena, "query", query)
			kv := metalmem.UploadIn(arena, "current", current)
			comp := metalmem.UploadIn(arena, "compressed", flat)
			allowed := metalmem.UploadIn(arena, "allowed", mask)
			sink := testBatchF32Tensor(arena, "sinks", sinks)
			original := ops.kernels.RopeOutUnrotate
			defer func() { ops.kernels.RopeOutUnrotate = original }()
			var finalProduction []float32
			for _, split := range []bool{false, true} {
				raw := metalmem.UploadIn(arena, "raw", cache)
				scores := metalmem.AllocIn[float32](arena, "scores", 64*(128+count))
				out := metalmem.AllocIn[float32](arena, "out", 32768+512+32768)
				for _, values := range [][]float32{scores.HostSlice(), out.HostSlice()} {
					for i := range values {
						values[i] = float32(math.NaN())
					}
				}
				ops.kernels.RopeOutUnrotate = original
				if split {
					ops.kernels.RopeOutUnrotate = pipeline
				}
				batch, err := newMetalBatch(ops)
				if err != nil {
					t.Fatal(err)
				}
				if err := batch.EncodeRopeAttentionMixed(q, kv, sink, raw, comp, count, allowed, scores, out, position, 2); err != nil {
					t.Fatal(err)
				}
				if err := batch.Wait(t.Context()); err != nil {
					t.Fatal(err)
				}
				got := out.HostSlice()
				if !split {
					finalProduction = append([]float32(nil), got...)
					continue
				}
				ops.kernels.RopeOutUnrotate = original
				for i, x := range got[:32768] {
					if math.Float32bits(rawCapacityRound(x, 10, -14)) != math.Float32bits(rawCapacityRound(rotated[i], 10, -14)) {
						t.Fatalf("halfquery mismatch %d", i)
					}
				}
				if err := rawCapacityExact(raw.HostSlice(), wantCache); err != nil {
					t.Fatal(err)
				}
				if err := rawCapacityExact(comp.HostSlice(), flat); err != nil {
					t.Fatal(err)
				}
				prefix := fmt.Sprintf("pos%d-comp%d", position, count)
				mixedDiagnosticDump(t, dir, prefix+"-query", got[:32768], rotated)
				mixedDiagnosticDump(t, dir, prefix+"-scores", scores.HostSlice(), wantScores)
				mixedDiagnosticDump(t, dir, prefix+"-pre", got[33280:], wantPre)
				batch, err = newMetalBatch(ops)
				if err != nil {
					t.Fatal(err)
				}
				enc := batch.encoder()
				enc.SetComputePipelineState(original)
				enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset(), 0)
				libmetal.SetScalar(enc, uint32(position), 1)
				enc.DispatchThreadgroupsThreadsPerThreadgroup(libmetal.Threads(1), libmetal.Threads(64))
				batch.encoded++
				if err := batch.Wait(t.Context()); err != nil {
					t.Fatal(err)
				}
				if err := rawCapacityExact(out.HostSlice(), finalProduction); err != nil {
					t.Fatalf("split/full endpoint identity: %v", err)
				}
				mixedDiagnosticDump(t, dir, prefix+"-final", out.HostSlice()[33280:], wantFinal)
				t.Logf("split/full endpoint bit-identical; numeric verdict=%v", mixedCapacityClose(out.HostSlice()[33280:], wantFinal))
			}
		}) {
			return
		}
	}
	if err := mixedDiagnosticArtifacts(dir); err != nil {
		t.Fatal(err)
	}
}

func TestMetalMixedDiagnosticAdmission(t *testing.T) {
	binary, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	run := func(dir string, locked bool) ([]byte, error) {
		cmd := exec.Command(binary, "-test.run=^TestMetalMixedCapacityStages$", "-test.v", "-test.count=1")
		for _, env := range os.Environ() {
			if strings.HasPrefix(env, "DS4_METAL_MIXED_DIAGNOSTIC_DIR=") || !locked && strings.HasPrefix(env, "GPU_LOCK_HELD=") {
				continue
			}
			cmd.Env = append(cmd.Env, env)
		}
		cmd.Env = append(cmd.Env, "DS4_METAL_MIXED_DIAGNOSTIC_DIR="+dir)
		return cmd.CombinedOutput()
	}
	output, err := run("", false)
	if err != nil || !bytes.Contains(output, []byte("--- SKIP: TestMetalMixedCapacityStages")) {
		t.Fatalf("absent gate must skip before lock: %v\n%s", err, output)
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("gpu-lock required for supplied-invalid gate control")
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "existing"), []byte("do not overwrite"), 0600); err != nil {
		t.Fatal(err)
	}
	output, err = run(dir, true)
	if err == nil || !bytes.Contains(output, []byte("create fresh diagnostic directory:")) || bytes.Contains(output, []byte("--- SKIP:")) {
		t.Fatalf("supplied-invalid gate must fail: %v\n%s", err, output)
	}
}

func mixedDiagnosticDirectory(dir string) error {
	if dir == "" {
		return fmt.Errorf("diagnostic output directory required")
	}
	if err := os.Mkdir(dir, 0700); err != nil {
		return fmt.Errorf("create fresh diagnostic directory: %w", err)
	}
	return nil
}

func mixedDiagnosticArtifacts(dir string) error {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return err
	}
	if len(entries) != 24 {
		return fmt.Errorf("diagnostic artifacts %d, want 24", len(entries))
	}
	for _, fixture := range []struct{ position, count int }{{127, 1}, {2048, 0}, {2048, 1}} {
		for _, stage := range []struct {
			name   string
			values int
		}{{"query", 32768}, {"scores", 64 * (128 + fixture.count)}, {"pre", 32768}, {"final", 32768}} {
			for _, side := range []string{"device", "reference"} {
				name := fmt.Sprintf("pos%d-comp%d-%s-%s.f32", fixture.position, fixture.count, stage.name, side)
				info, err := os.Stat(filepath.Join(dir, name))
				if err != nil {
					return err
				}
				if !info.Mode().IsRegular() || info.Size() != int64(stage.values*4) {
					return fmt.Errorf("invalid diagnostic artifact %s", name)
				}
			}
		}
	}
	return nil
}

func TestMixedDiagnosticOutputDirectory(t *testing.T) {
	if mixedDiagnosticDirectory("") == nil {
		t.Fatal("accepted missing directory")
	}
	dir := filepath.Join(t.TempDir(), "new")
	if err := mixedDiagnosticDirectory(dir); err != nil {
		t.Fatal(err)
	}
	if mixedDiagnosticDirectory(dir) == nil {
		t.Fatal("accepted existing directory")
	}
	if mixedDiagnosticArtifacts(dir) == nil {
		t.Fatal("accepted empty artifacts")
	}
}

func mixedDiagnosticDump(t *testing.T, dir, name string, got, want []float32) {
	t.Helper()
	var maximum, sum float64
	for i, value := range got {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("%s nonfinite %d", name, i)
		}
		d := float64(value) - float64(want[i])
		maximum = math.Max(maximum, math.Abs(d))
		sum += d * d
	}
	for _, data := range []struct {
		suffix string
		values []float32
	}{{"device", got}, {"reference", want}} {
		bytes := make([]byte, len(data.values)*4)
		for i, value := range data.values {
			binary.LittleEndian.PutUint32(bytes[i*4:], math.Float32bits(value))
		}
		file, err := os.OpenFile(filepath.Join(dir, name+"-"+data.suffix+".f32"), os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
		if err != nil {
			t.Fatal(err)
		}
		_, writeErr := file.Write(bytes)
		closeErr := file.Close()
		if writeErr != nil {
			t.Fatal(writeErr)
		}
		if closeErr != nil {
			t.Fatal(closeErr)
		}
		t.Logf("artifact=%s-%s values=%d sha256=%x", name, data.suffix, len(data.values), sha256.Sum256(bytes))
	}
	t.Logf("stage=%s max_abs=%g rms=%g verdict=%v", name, maximum, math.Sqrt(sum/float64(len(got))), mixedCapacityClose(got, want))
}
