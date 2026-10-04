//go:build native

package cudareduce

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudablock"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

func TestBuildSparseTeacherPacket(t *testing.T) {
	if os.Getenv("DISTILL_BUILD_ONLY") != "1" {
		t.Skip("offline compiler preparation not requested")
	}
	dir, path, builtins := os.Getenv("DISTILL_PACKET_DIR"), os.Getenv("DISTILL_NVRTC_PATH"), os.Getenv("DISTILL_BUILTINS_PATH")
	if !filepath.IsAbs(dir) || !filepath.IsAbs(path) || !filepath.IsAbs(builtins) {
		t.Fatal("explicit absolute compiler/packet paths required")
	}
	if err := os.MkdirAll(dir, 0700); err != nil {
		t.Fatal(err)
	}
	if err := nvrtc.Load(path); err != nil {
		t.Fatal(err)
	}
	major, minor, err := nvrtc.Version()
	if err != nil || major != 13 || minor != 4 {
		t.Fatalf("compiler13.4 required: %d.%d %v", major, minor, err)
	}
	compiler, err := distillReadBounded(path, 512<<20)
	if err != nil {
		t.Fatal(err)
	}
	builtin, err := distillReadBounded(builtins, 512<<20)
	if err != nil {
		t.Fatal(err)
	}
	sources := map[string]string{"baseline": SparseTeacherSource}
	for _, m := range distillMutants() {
		sources[m.Name] = distillMutantSource(m)
	}
	for name, source := range sources {
		a, err := distillCompileSource("sm_121", source)
		if err != nil {
			t.Fatalf("compile %s: %v", name, err)
		}
		if err := distillAdmitArtifact(a, source); err != nil {
			t.Fatalf("%v; retained dependencies=%+v", err, a.Dependencies)
		}
		p := distillPacket{Version: 1, SourceDigest: distillDigest([]byte(source)), HeaderDigest: distillDigest([]byte(cudablock.Header)), CompilerPath: nvrtc.Path(), CompilerVersion: "13.4.59", CompilerDigest: distillDigest(compiler), BuiltinsPath: builtins, BuiltinsDigest: distillDigest(builtin), Artifact: *a}
		fragment, err := distillBlockHeader()
		if err != nil {
			t.Fatal(err)
		}
		p.FragmentDigest = distillDigest([]byte(fragment))
		raw, err := json.MarshalIndent(p, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		for suffix, data := range map[string][]byte{".json": raw, ".cu": []byte(source), ".cubin": a.Raw} {
			if err := os.WriteFile(filepath.Join(dir, name+suffix), data, 0600); err != nil {
				t.Fatal(err)
			}
		}
		t.Logf("compiled %s source=%s image=%s compiler=%s", name, p.SourceDigest, a.Digest, p.CompilerDigest)
	}
	if err := os.WriteFile(filepath.Join(dir, "cudablock.cuh"), []byte(cudablock.Header), 0600); err != nil {
		t.Fatal(err)
	}
	fragment, err := distillBlockHeader()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "cudablock-used.cuh"), []byte(fragment), 0600); err != nil {
		t.Fatal(err)
	}
}
func TestSparseTeacher_PreparedPacket(t *testing.T) {
	dir := os.Getenv("DISTILL_HOST_PACKET_DIR")
	if dir == "" {
		t.Skip("retained packet host audit not requested")
	}
	sources := map[string]string{"baseline": SparseTeacherSource}
	for _, m := range distillMutants() {
		sources[m.Name] = distillMutantSource(m)
	}
	for name, source := range sources {
		if _, err := distillReadPacket(dir, name, source); err != nil {
			t.Fatal(err)
		}
	}
}

const distillGuard uint32 = 0x5e31ab29

type distillNativeMemory struct {
	Buffers SparseTeacherBuffers
	check   []func() error
}

func distillGuarded[T ~float32 | ~int32](t *testing.T, dev *cuda.Device, n int, guards *[]func() error) cudamem.Slice[T] {
	t.Helper()
	whole, err := cudamem.Alloc[T](dev, n+2)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := whole.Free(); err != nil {
			t.Error(err)
		}
	})
	seed := make([]T, n+2)
	var guard T
	switch any(guard).(type) {
	case float32:
		guard = T(math.Float32frombits(distillGuard))
	case int32:
		guard = T(int32(distillGuard))
	}
	for i := range seed {
		seed[i] = guard
	}
	if err := whole.CopyFromHost(seed); err != nil {
		t.Fatal(err)
	}
	*guards = append(*guards, func() error {
		read := make([]T, n+2)
		if err := whole.CopyToHost(read); err != nil {
			return err
		}
		if read[0] != guard || read[n+1] != guard {
			return fmt.Errorf("distill redzone changed")
		}
		return nil
	})
	return whole.Slice(1, n+1)
}
func distillNativeBuffers(t *testing.T, dev *cuda.Device, c SparseTeacherConfig) distillNativeMemory {
	t.Helper()
	e, err := c.Extents()
	if err != nil {
		t.Fatal(err)
	}
	var m distillNativeMemory
	b := &m.Buffers
	b.Logits = distillGuarded[float32](t, dev, e.Logits, &m.check)
	b.Weights = distillGuarded[float32](t, dev, e.Targets, &m.check)
	b.IDs = distillGuarded[int32](t, dev, e.Targets, &m.check)
	b.Masks = distillGuarded[int32](t, dev, e.Rows, &m.check)
	b.Status = distillGuarded[int32](t, dev, e.Rows, &m.check)
	b.Stats = distillGuarded[float32](t, dev, e.Stats, &m.check)
	b.ScratchLosses = distillGuarded[float32](t, dev, e.Rows, &m.check)
	b.ScratchGradients = distillGuarded[float32](t, dev, e.Logits, &m.check)
	b.ScratchTotal = distillGuarded[float32](t, dev, 1, &m.check)
	b.Losses = distillGuarded[float32](t, dev, e.Rows, &m.check)
	b.Gradients = distillGuarded[float32](t, dev, e.Logits, &m.check)
	b.Total = distillGuarded[float32](t, dev, 1, &m.check)
	return m
}
func distillUpload(f distillFixture, b SparseTeacherBuffers) error {
	return errors.Join(b.Logits.CopyFromHost(f.Z), b.Weights.CopyFromHost(f.W), b.IDs.CopyFromHost(f.IDs), b.Masks.CopyFromHost(f.Masks))
}
func distillCheckNativeInputs(f distillFixture, b SparseTeacherBuffers) error {
	z, w := make([]float32, len(f.Z)), make([]float32, len(f.W))
	ids, masks := make([]int32, len(f.IDs)), make([]int32, len(f.Masks))
	if err := errors.Join(b.Logits.CopyToHost(z), b.Weights.CopyToHost(w), b.IDs.CopyToHost(ids), b.Masks.CopyToHost(masks)); err != nil {
		return err
	}
	for i, x := range z {
		if math.Float32bits(x) != math.Float32bits(f.Z[i]) {
			return fmt.Errorf("distill input logit changed")
		}
	}
	for i, x := range w {
		if math.Float32bits(x) != math.Float32bits(f.W[i]) {
			return fmt.Errorf("distill teacher weight changed")
		}
	}
	for i, x := range ids {
		if x != f.IDs[i] {
			return fmt.Errorf("distill teacher ID changed")
		}
	}
	for i, x := range masks {
		if x != f.Masks[i] {
			return fmt.Errorf("distill mask changed")
		}
	}
	return nil
}
func distillReadNative(c SparseTeacherConfig, m distillNativeMemory) (distillHostResult, error) {
	for _, check := range m.check {
		if err := check(); err != nil {
			return distillHostResult{}, err
		}
	}
	r := distillHostResult{Status: make([]int32, c.Rows), Losses: make([]float32, c.Rows), Gradients: make([]float32, c.Rows*c.Vocabulary), Stats: make([]float32, c.Rows*4)}
	total := make([]float32, 1)
	err := errors.Join(m.Buffers.Status.CopyToHost(r.Status), m.Buffers.Losses.CopyToHost(r.Losses), m.Buffers.Gradients.CopyToHost(r.Gradients), m.Buffers.Stats.CopyToHost(r.Stats), m.Buffers.Total.CopyToHost(total))
	r.Total = total[0]
	return r, err
}
func distillRecord(t *testing.T, name string, value any) {
	t.Helper()
	dir := os.Getenv("DISTILL_RESULT_DIR")
	if !filepath.IsAbs(dir) {
		t.Fatal("absolute result directory required")
	}
	if err := os.MkdirAll(dir, 0700); err != nil {
		t.Fatal(err)
	}
	raw, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, name+".json"), raw, 0600); err != nil {
		t.Fatal(err)
	}
}
func distillBits(s []float32) []uint32 {
	out := make([]uint32, len(s))
	for i, x := range s {
		out[i] = math.Float32bits(x)
	}
	return out
}
func distillNativeOwner(t *testing.T, dev *cuda.Device, c SparseTeacherConfig, name, source string) *SparseTeacher {
	t.Helper()
	packet, err := distillReadPacket(os.Getenv("DISTILL_PACKET_DIR"), name, source)
	if err != nil {
		t.Fatal(err)
	}
	owner, err := distillLoadSource(dev, c, &packet.Artifact, source)
	if owner != nil {
		t.Cleanup(func() {
			if err := owner.Close(); err != nil {
				t.Error(err)
			}
		})
	}
	if err != nil {
		t.Fatal(err)
	}
	attrs := map[string]int{}
	for name, attr := range map[string]driver.FunctionAttribute{"registers": driver.FuncAttributeNumRegs, "shared_bytes": driver.FuncAttributeSharedSizeBytes, "local_bytes": driver.FuncAttributeLocalSizeBytes, "max_threads": driver.FuncAttributeMaxThreadsPerBlock} {
		value, err := owner.kernel.Attribute(attr)
		if err != nil {
			t.Fatal(err)
		}
		attrs[name] = value
	}
	t.Logf("loaded %s image=%s attributes=%v", name, packet.Artifact.Digest, attrs)
	return owner
}

// TestSparseTeacher_Native is the fixed 24-operation correctness/lifetime packet:
// nine healthy fixtures (one runs two changed-target epochs), five mutants,
// eight admission refusals and one queued cancellation. There are 23 subtests.
// Images are precompiled; no NVRTC work or timing experiment runs under lock.
func TestSparseTeacher_Native(t *testing.T) {
	if os.Getenv("DISTILL_NATIVE") != "1" {
		t.Skip("coordinator native dispatch not requested")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" || os.Getenv("GPU_LOCK_DISABLED") != "" || os.Getenv("CUDATEST_NO_LOCK") != "" {
		t.Fatal("actual inherited gpu-lock required")
	}
	if err := driver.Load(""); err != nil {
		t.Fatal(err)
	}
	if err := cuda.Init(); err != nil {
		t.Fatal(err)
	}
	count, err := cuda.DeviceCount()
	if err != nil || count < 1 {
		t.Fatal("no actual device")
	}
	dev := cudatest.Require(t)
	major, minor, err := dev.ComputeCapability()
	if err != nil || major != 12 || minor != 1 {
		t.Fatal("packet requires sm121")
	}
	name, err := dev.Name()
	if err != nil {
		t.Fatal(err)
	}
	uuid, err := dev.Raw().Uuid()
	if err != nil {
		t.Fatal(err)
	}
	version, err := driver.DriverGetVersion()
	if err != nil {
		t.Fatal(err)
	}
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	binary, err := distillReadBounded(exe, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("actual producer go=%s platform=%s/%s device=%s uuid=%x driver=%s API=%d binary=%s", runtime.Version(), runtime.GOOS, runtime.GOARCH, name, uuid, driver.Path(), version, distillDigest(binary))
	for _, f := range distillFixtures() {
		if f.Name == "normalized_full" {
			continue
		} // Native full support is singleton and tail_full; larger normalized full is host-only.
		t.Run(f.Name, func(t *testing.T) {
			stream, err := dev.NewStream()
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := stream.Close(); err != nil {
					t.Error(err)
				}
			})
			memory := distillNativeBuffers(t, dev, f.Config)
			owner := distillNativeOwner(t, dev, f.Config, "baseline", SparseTeacherSource)
			epochs := 1
			if f.Name == "tail_irregular" {
				epochs = 2
			}
			for epoch := 0; epoch < epochs; epoch++ {
				if epoch == 1 {
					for i := range f.W {
						f.W[i] *= 0.5
					}
					f.Masks[0] = 0
					f.Masks[1] = 1
				}
				if err := distillUpload(f, memory.Buffers); err != nil {
					t.Fatal(err)
				}
				if err := owner.Execute(context.Background(), stream, memory.Buffers); err != nil {
					t.Fatal(err)
				}
				result, err := distillReadNative(f.Config, memory)
				if err != nil {
					t.Fatal(err)
				}
				if err := distillCheckNativeInputs(f, memory.Buffers); err != nil {
					t.Fatal(err)
				}
				for _, x := range result.Stats {
					if !distillFinite(x) || math.Float32bits(x) == distillGuard {
						t.Fatal("scratch stats missing or nonfinite write")
					}
				}
				distillCheckReference(t, f, result)
				loss, grad, total := distillReference(f)
				distillRecord(t, fmt.Sprintf("%s_epoch%d", f.Name, epoch), map[string]any{"input_digest": distillInputDigest(f), "source_digest": distillDigest([]byte(SparseTeacherSource)), "status": result.Status, "loss_bits": distillBits(result.Losses), "gradient_bits": distillBits(result.Gradients), "total_bits": math.Float32bits(result.Total), "reference_loss": loss, "reference_gradient": grad, "reference_total": total, "input_and_output_guards": true, "scratch_stats_bits": distillBits(result.Stats)})
			}
		})
	}
	for _, mutant := range distillMutants() {
		t.Run("mutant_"+mutant.Name, func(t *testing.T) {
			f := distillFixtureByName(mutant.Fixture)
			stream, err := dev.NewStream()
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { stream.Close() })
			memory := distillNativeBuffers(t, dev, f.Config)
			source := distillMutantSource(mutant)
			owner := distillNativeOwner(t, dev, f.Config, mutant.Name, source)
			if err := distillUpload(f, memory.Buffers); err != nil {
				t.Fatal(err)
			}
			if err := owner.Execute(context.Background(), stream, memory.Buffers); err != nil {
				t.Fatalf("mutant operational failure: %v", err)
			}
			r, err := distillReadNative(f.Config, memory)
			if err != nil {
				t.Fatal(err)
			}
			if err := distillCheckNativeInputs(f, memory.Buffers); err != nil {
				t.Fatal(err)
			}
			loss, grad, total := distillReference(f)
			a := distillCompare(f, "native mutant", r.Losses, loss)
			b := distillCompare(f, "native mutant", r.Gradients, grad)
			c := distillCompare(f, "native mutant", []float32{r.Total}, []float64{total})
			var mismatch *cudatest.CUDAOutputMismatch
			if !errors.As(a, &mismatch) && !errors.As(b, &mismatch) && !errors.As(c, &mismatch) {
				t.Fatal("mutant did not produce typed numerical mismatch")
			}
			distillRecord(t, "mutant_"+mutant.Name, map[string]any{"source_digest": distillDigest([]byte(source)), "input_digest": distillInputDigest(f), "status": r.Status, "loss_bits": distillBits(r.Losses), "gradient_bits": distillBits(r.Gradients), "total_bits": math.Float32bits(r.Total), "reference_loss": loss, "reference_gradient": grad, "reference_total": total, "input_and_output_guards": true, "typed_numerical_mismatch": true})
		})
	}
	invalid := distillInvalidFixtures()
	full := distillFixtureByName("tail_full")
	full.Name = "absent_tail_mass"
	full.W[0] = 0
	for _, f := range []distillFixture{invalid[0], invalid[1], invalid[3], invalid[4], invalid[5], invalid[8], invalid[9], full} {
		t.Run("resident_invalid_"+f.Name, func(t *testing.T) {
			stream, err := dev.NewStream()
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { stream.Close() })
			memory := distillNativeBuffers(t, dev, f.Config)
			owner := distillNativeOwner(t, dev, f.Config, "baseline", SparseTeacherSource)
			if err := distillUpload(f, memory.Buffers); err != nil {
				t.Fatal(err)
			}
			err = owner.Execute(context.Background(), stream, memory.Buffers)
			if !errors.Is(err, errDistillStatus) {
				t.Fatalf("invalid device data lacks checked status refusal: %v", err)
			}
			r, readErr := distillReadNative(f.Config, memory)
			if readErr != nil {
				t.Fatal(readErr)
			}
			if CheckSparseTeacherStatus(r.Status, f.Config.Rows) == nil {
				t.Fatal("invalid refusal lacks device status")
			}
			for _, s := range [][]float32{r.Losses, r.Gradients, []float32{r.Total}} {
				for _, x := range s {
					if math.Float32bits(x) != distillGuard {
						t.Fatal("invalid resident input published output")
					}
				}
			}
			distillRecord(t, "invalid_"+f.Name, map[string]any{"device_status": r.Status, "publication_unchanged": true, "input_and_output_guards": true, "error": err.Error()})
		})
	}
	t.Run("queued_cancellation_lifetime", func(t *testing.T) {
		f := distillFixtureByName("normalized_outside")
		stream, err := dev.NewStream()
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { stream.Close() })
		memory := distillNativeBuffers(t, dev, f.Config)
		owner := distillNativeOwner(t, dev, f.Config, "baseline", SparseTeacherSource)
		if err := distillUpload(f, memory.Buffers); err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		owner.beforeWait = cancel
		err = owner.Execute(ctx, stream, memory.Buffers)
		if !distillOnlyCanceled(err) || owner.pending == nil || owner.pending.Done() {
			t.Fatalf("cancel lifetime: %v", err)
		}
		if !errors.Is(memory.Buffers.Logits.Free(), cudamem.ErrInUse) || !errors.Is(memory.Buffers.Stats.Free(), cudamem.ErrInUse) {
			t.Fatal("queued input/scratch not pinned")
		}
		if owner.Execute(context.Background(), stream, memory.Buffers) == nil {
			t.Fatal("incomplete invocation reused")
		}
		owner.beforeWait = nil
		if err := owner.Close(); err != nil {
			t.Fatal(err)
		}
		r, err := distillReadNative(f.Config, memory)
		if err != nil {
			t.Fatal(err)
		}
		if err := CheckSparseTeacherStatus(r.Status, f.Config.Rows); err != nil {
			t.Fatal("queued validation did not complete", err)
		}
		for _, x := range r.Gradients {
			if math.Float32bits(x) != distillGuard {
				t.Fatal("canceled invocation published")
			}
		}
		if err := memory.Buffers.Stats.Free(); err != nil {
			t.Fatal(err)
		}
		distillRecord(t, "queued_cancellation_lifetime", map[string]any{"borrow_refusal": true, "reuse_refusal": true, "real_validation_drain": true, "publication_unchanged": true, "scratch_release_after_drain": true})
	})
}
