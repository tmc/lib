// SPDX-License-Identifier: Apache-2.0

package tileir_test

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"math"
	"os"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaerrors"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudaliveness"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
	"github.com/tmc/lib/nvidia/x/tileir"
)

const tileIRNativeEnv = "TILEIR_NATIVE"

var errTileIRNativeGateClosed error = tileIRNativeGateClosedError{}

type tileIRNativeGateClosedError struct{}

func (tileIRNativeGateClosedError) Error() string {
	return "native qualification pending: TILEIR_NATIVE=1 required"
}

func (tileIRNativeGateClosedError) Unwrap() error {
	return cudaerrors.ErrHardwarePending
}

func admitTileIRNativeGate(getenv func(string) string) (cudalock.Owner, int, error) {
	if getenv == nil || getenv(tileIRNativeEnv) != "1" {
		return cudalock.Owner{}, 0, errTileIRNativeGateClosed
	}
	owner, err := cudalock.Inherited(getenv)
	if err != nil {
		return cudalock.Owner{}, 0, fmt.Errorf("tileir: native gate enabled but %w: run under gpu-lock", err)
	}
	ordinal := 0
	if raw := strings.TrimSpace(getenv("TILEIR_DEVICE")); raw != "" {
		parsed, err := strconv.Atoi(raw)
		if err != nil || parsed < 0 {
			return cudalock.Owner{}, 0, fmt.Errorf("tileir: invalid TILEIR_DEVICE=%q", raw)
		}
		ordinal = parsed
	}
	return owner, ordinal, nil
}

// tileIRNativeCleanupOwners retains io.Closer cleanup handles returned inside
// nvrtcx.Compile or module load errors as well as non-zero driver.Module
// handles returned by cudakernel.Artifact.Load (including when Load returns an
// error after loading the module), closing them in LIFO order before dependent
// device resources are torn down.
type tileIRNativeCleanupOwners struct {
	owners []io.Closer
	mods   []driver.Module
}

func (r *tileIRNativeCleanupOwners) adoptModule(mod driver.Module) {
	if mod != 0 {
		r.mods = append(r.mods, mod)
	}
}

func (r *tileIRNativeCleanupOwners) adoptError(err error) error {
	var owner io.Closer
	if errors.As(err, &owner) && owner != nil {
		r.owners = append(r.owners, owner)
	}
	return err
}

func (r *tileIRNativeCleanupOwners) closeBefore(dependents func() error) error {
	return r.closeWithModules(nil, dependents)
}

func (r *tileIRNativeCleanupOwners) closeWithModules(unloadMod func(driver.Module) error, dependents func() error) error {
	for len(r.owners) > 0 {
		last := len(r.owners) - 1
		if err := r.owners[last].Close(); err != nil {
			return fmt.Errorf("tileir: close retained compiler/module owner: %w", err)
		}
		r.owners = r.owners[:last]
	}
	for len(r.mods) > 0 {
		last := len(r.mods) - 1
		m := r.mods[last]
		if m != 0 {
			if unloadMod == nil {
				return errors.New("tileir: module unloader unavailable; retaining module")
			}
			if err := unloadMod(m); err != nil {
				return fmt.Errorf("tileir: close module: %w", err)
			}
		}
		r.mods = r.mods[:last]
	}
	if dependents == nil {
		return nil
	}
	return dependents()
}

func validateTileIRArtifact(art *cudakernel.Artifact, entryPoint string, wantSchema cudakernel.Schema) error {
	if art == nil {
		return errors.New("tileir: nil artifact")
	}
	if art.Kind != cudakernel.KindTileIR {
		return fmt.Errorf("tileir: refusing non-TileIR artifact kind %s (want %s)", art.Kind, cudakernel.KindTileIR)
	}
	if err := art.Validate(); err != nil {
		return fmt.Errorf("tileir: invalid artifact: %w", err)
	}
	if !tileir.IsTileIRBytecode(art.Raw) {
		return errors.New("tileir: artifact raw payload lacks TileIR bytecode magic header")
	}
	if art.EntryPoint != entryPoint {
		return fmt.Errorf("tileir: artifact entry point = %q, want %q", art.EntryPoint, entryPoint)
	}
	if len(art.Schema.Params) != len(wantSchema.Params) || art.Schema.TotalParamBytes != wantSchema.TotalParamBytes {
		return fmt.Errorf("tileir: artifact schema mismatch: got %d params (%d B), want %d params (%d B)",
			len(art.Schema.Params), art.Schema.TotalParamBytes, len(wantSchema.Params), wantSchema.TotalParamBytes)
	}
	for i := range wantSchema.Params {
		if art.Schema.Params[i] != wantSchema.Params[i] {
			return fmt.Errorf("tileir: artifact schema param[%d] = %+v, want %+v",
				i, art.Schema.Params[i], wantSchema.Params[i])
		}
	}
	return nil
}

// loadTileIRArtifact validates that art is a KindTileIR bytecode artifact with
// the expected entry point and ABI schema, invokes loadFn (typically art.Load(dev)),
// adopts any non-zero returned driver.Module into retained before checking err
// (preserving cudakernel.Artifact.Load's error-carried module ownership contract),
// retains any io.Closer error owner into retained, and looks up entryPoint.
func loadTileIRArtifact(
	art *cudakernel.Artifact,
	entryPoint string,
	wantSchema cudakernel.Schema,
	retained *tileIRNativeCleanupOwners,
	loadFn func(*cudakernel.Artifact) (driver.Module, error),
	lookupFn func(driver.Module, string) (driver.Function, error),
) (driver.Function, error) {
	if retained == nil || loadFn == nil || lookupFn == nil {
		return 0, errors.New("tileir: missing artifact load custody or callbacks")
	}
	if err := validateTileIRArtifact(art, entryPoint, wantSchema); err != nil {
		return 0, err
	}
	mod, err := loadFn(art)
	retained.adoptModule(mod)
	if err != nil {
		_ = retained.adoptError(err)
		return 0, err
	}
	if mod == 0 {
		return 0, errors.New("tileir: artifact loader returned zero module without error")
	}
	fn, err := lookupFn(mod, art.EntryPoint)
	if err != nil {
		return 0, fmt.Errorf("tileir: lookup function %q: %w", art.EntryPoint, err)
	}
	return fn, nil
}

func tileIRCoreCompileOptions(kernelName string, major, minor int) []nvrtcx.Option {
	opts := []nvrtcx.Option{
		nvrtcx.WithArch(major, minor),
		nvrtcx.WithFlag("--std=c++20"),
		nvrtcx.WithOutput(nvrtcx.OutputTileIR),
		nvrtcx.WithNoCache(),
	}
	if kernelName != "" {
		opts = append([]nvrtcx.Option{nvrtcx.WithName(kernelName + ".cu")}, opts...)
	}
	return opts
}

func tileIRCompileOptions(kernelName string, major, minor int) []nvrtcx.Option {
	return append(tileIRCoreCompileOptions(kernelName, major, minor), nvrtcx.WithToolkitIncludes())
}

func compileTileIRSourceArtifact(
	ctx context.Context,
	source string,
	kernelName string,
	schema cudakernel.Schema,
	major, minor int,
	rtcMajor, rtcMinor int32,
	retained *tileIRNativeCleanupOwners,
) (*cudakernel.Artifact, error) {
	if err := tileir.ValidateTileCPPSource(source); err != nil {
		return nil, err
	}
	options := tileIRCompileOptions(kernelName, major, minor)
	compiled, err := nvrtcx.CompileContext(ctx, source, options...)
	if err != nil {
		if retained != nil {
			_ = retained.adoptError(err)
		}
		return nil, fmt.Errorf("tileir: compile TileIR source (%s): %w", kernelName, err)
	}
	if compiled.Output() != nvrtcx.OutputTileIR {
		return nil, fmt.Errorf("tileir: compiled output = %v, want OutputTileIR", compiled.Output())
	}
	art, err := compiled.ToArtifact(kernelName, schema, fmt.Sprintf("sm_%d%d", major, minor))
	if err != nil {
		return nil, err
	}
	art.Toolchain = fmt.Sprintf("nvrtc %d.%d", rtcMajor, rtcMinor)
	if err := validateTileIRArtifact(art, kernelName, schema); err != nil {
		return nil, err
	}
	return art, nil
}

type tileIRSyntheticCleanupError struct {
	cause error
	calls int
	fail  bool
}

func (e *tileIRSyntheticCleanupError) Error() string { return e.cause.Error() }
func (e *tileIRSyntheticCleanupError) Unwrap() error { return e.cause }
func (e *tileIRSyntheticCleanupError) Close() error {
	e.calls++
	if e.fail {
		return errors.New("synthetic owner cleanup still failing")
	}
	return nil
}

func finiteFloat32WithinTol(got, want, tol float32) bool {
	if math.IsNaN(float64(got)) || math.IsInf(float64(got), 0) ||
		math.IsNaN(float64(want)) || math.IsInf(float64(want), 0) ||
		math.IsNaN(float64(tol)) || math.IsInf(float64(tol), 0) || tol < 0 {
		return false
	}
	return math.Abs(float64(got)-float64(want)) <= float64(tol)
}

func float32SliceDigest(vals []float32) string {
	h := sha256.New()
	var buf [4]byte
	for _, v := range vals {
		binary.LittleEndian.PutUint32(buf[:], math.Float32bits(v))
		_, _ = h.Write(buf[:])
	}
	return hex.EncodeToString(h.Sum(nil))
}

// tileIRNumericalFixture holds a deterministic semantic TileIR region
// (relu(a * b - c) + d) lowered to TileIR AST, pointer ABI schema, and
// executable Tile C++ source, paired with signed non-trivial inputs and an
// independent CPU scalar reference oracle.
type tileIRNumericalFixture struct {
	kernelName string
	length     int
	snapshot   *cudair.Snapshot
	lowering   *tileir.LoweringResult
	a, b, c, d []float32
	want       []float32
	mutWant    []float32
}

func newTileIRNumericalFixture(t *testing.T) tileIRNumericalFixture {
	t.Helper()
	const (
		kernelName = "tileir_fused_affine_relu"
		length     = 16
	)
	g := cudair.NewGraph()
	aNode := g.AddInput("A", []int{length}, cudair.Float32)
	bNode := g.AddInput("B", []int{length}, cudair.Float32)
	cNode := g.AddInput("C", []int{length}, cudair.Float32)
	dNode := g.AddInput("D", []int{length}, cudair.Float32)
	prod := g.AddBinary("mul", aNode, bNode)
	diff := g.AddBinary("sub", prod, cNode)
	act := g.AddUnary("relu", diff)
	out := g.AddBinary("add", act, dNode)
	if err := g.SetOutputs(out); err != nil {
		t.Fatalf("SetOutputs: %v", err)
	}
	snap, err := g.Snapshot()
	if err != nil {
		t.Fatalf("Snapshot: %v", err)
	}

	lowering, err := tileir.LowerSemanticRegion(snap, tileir.LowerOptions{
		KernelName: kernelName,
		TileM:      length,
		TileN:      1,
		TileK:      1,
	})
	if err != nil {
		t.Fatalf("LowerSemanticRegion: %v", err)
	}

	// Signed non-trivial inputs exercising both active (> 0) and clamped (<= 0)
	// ReLU branches as well as positive and negative bias offsets in D.
	a := []float32{
		1.75, -2.5, 3.25, -1.125,
		0.875, 2.125, -3.5, -1.5,
		4.0, -0.625, 1.375, -2.75,
		-0.5, 2.625, -1.875, 3.5,
	}
	b := []float32{
		1.5, 1.25, -0.75, -2.0,
		-3.0, 1.5, -0.5, 2.25,
		0.625, 3.0, -1.5, -1.25,
		-4.5, -0.75, 1.5, 0.875,
	}
	c := []float32{
		0.5, -1.0, -1.5, 0.75,
		-1.25, 4.0, 0.5, -2.0,
		1.125, -0.5, -1.0, 1.5,
		3.0, -1.0, -0.75, 1.25,
	}
	d := []float32{
		-0.375, 1.125, -2.25, 0.625,
		1.75, -0.875, 2.5, -1.25,
		0.5, -1.625, 0.875, -0.5,
		1.25, -0.75, 2.125, -1.375,
	}

	want := cpuScalarTileIROracle(a, b, c, d)
	// Planted mutant oracle: replace subtraction of C with addition of C
	// (relu(a*b + c) + d), which changes elements where a*b - c != a*b + c.
	negC := make([]float32, len(c))
	for i, v := range c {
		negC[i] = -v
	}
	mutWant := cpuScalarTileIROracle(a, b, negC, d)

	return tileIRNumericalFixture{
		kernelName: kernelName,
		length:     length,
		snapshot:   snap,
		lowering:   lowering,
		a:          a,
		b:          b,
		c:          c,
		d:          d,
		want:       want,
		mutWant:    mutWant,
	}
}

// cpuScalarTileIROracle computes out[i] = max(a[i]*b[i] - c[i], 0) + d[i]
// directly in scalar FP32 without consulting cudair or tileir evaluators.
func cpuScalarTileIROracle(a, b, c, d []float32) []float32 {
	out := make([]float32, len(a))
	for i := range a {
		prod := a[i] * b[i]
		diff := prod - c[i]
		relu := diff
		if relu < 0 {
			relu = 0
		}
		out[i] = relu + d[i]
	}
	return out
}

func (f tileIRNumericalFixture) compareOutput(got []float32, tol float32) error {
	if len(got) != len(f.want) {
		return fmt.Errorf("tileir: output length = %d, want %d", len(got), len(f.want))
	}
	if math.IsNaN(float64(tol)) || math.IsInf(float64(tol), 0) || tol < 0 {
		return fmt.Errorf("tileir: invalid tolerance %v", tol)
	}
	// Explicitly reject any NaN or Inf in either buffer before tolerance comparison.
	for i, v := range got {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("tileir: non-finite device output[%d] = %v (0x%08x)", i, v, math.Float32bits(v))
		}
	}
	for i, w := range f.want {
		if math.IsNaN(float64(w)) || math.IsInf(float64(w), 0) {
			return fmt.Errorf("tileir: non-finite reference output[%d] = %v (0x%08x)", i, w, math.Float32bits(w))
		}
		if !finiteFloat32WithinTol(got[i], w, tol) {
			return fmt.Errorf("tileir: output[%d] = %v (0x%08x), want %v (0x%08x), tol=%v",
				i, got[i], math.Float32bits(got[i]), w, math.Float32bits(w), tol)
		}
	}
	return nil
}

func (f tileIRNumericalFixture) countMismatches(got []float32, tol float32) int {
	if len(got) != len(f.want) {
		return len(f.want) + 1
	}
	mismatches := 0
	for i, w := range f.want {
		if !finiteFloat32WithinTol(got[i], w, tol) {
			mismatches++
		}
	}
	return mismatches
}

func (f tileIRNumericalFixture) verifyBindings() error {
	if f.lowering == nil || f.lowering.Module == nil {
		return errors.New("tileir: missing lowered module")
	}
	if err := tileir.ValidateTileCPPSource(f.lowering.Source); err != nil {
		return fmt.Errorf("tileir: invalid Tile C++ source: %w", err)
	}
	fn := f.lowering.Module.FindFunction(f.kernelName)
	if fn == nil || !fn.Entry {
		return fmt.Errorf("tileir: missing entry function %q", f.kernelName)
	}
	hazards, shapes := fn.Verify()
	if len(hazards) != 0 || len(shapes) != 0 {
		return fmt.Errorf("tileir: module verification failed: hazards=%v shapes=%v", hazards, shapes)
	}
	wantRoles := []string{"A", "B", "C", "D", f.snapshot.Outputs()[0].Name}
	if len(f.lowering.Schema.Params) != len(wantRoles) || len(fn.Params) != len(wantRoles) {
		return fmt.Errorf("tileir: schema params = %d, fn params = %d, want %d",
			len(f.lowering.Schema.Params), len(fn.Params), len(wantRoles))
	}
	if f.lowering.Schema.TotalParamBytes != len(wantRoles)*8 {
		return fmt.Errorf("tileir: total param bytes = %d, want %d", f.lowering.Schema.TotalParamBytes, len(wantRoles)*8)
	}
	for i, wantName := range wantRoles {
		p := f.lowering.Schema.Params[i]
		if p.Name != wantName || p.Type != "f32" || !p.IsPointer || p.ByteWidth != 8 || p.Alignment != 8 || p.Offset != i*8 {
			return fmt.Errorf("tileir: schema param[%d] = %+v, want pointer %q at offset %d", i, p, wantName, i*8)
		}
	}
	packArgs := []any{
		driver.DevicePtr(0x1000),
		driver.DevicePtr(0x2000),
		driver.DevicePtr(0x3000),
		driver.DevicePtr(0x4000),
		driver.DevicePtr(0x5000),
	}
	packed, err := f.lowering.Schema.Pack(packArgs...)
	if err != nil {
		return fmt.Errorf("tileir: pack pointer ABI: %w", err)
	}
	if len(packed) != len(packArgs)*8 {
		return fmt.Errorf("tileir: packed ABI bytes = %d, want %d", len(packed), len(packArgs)*8)
	}
	for _, wantSubstr := range []string{
		`#include "cuda_tile.h"`,
		`extern "C" __tile_global__ void ` + f.kernelName,
		"ct::mul(arg0[idx], arg1[idx])",
		"ct::sub(ct::mul(arg0[idx], arg1[idx]), arg2[idx])",
		"ct::max(ct::sub(ct::mul(arg0[idx], arg1[idx]), arg2[idx]), 0.0f)",
		"if (idx < 16) arg4[idx] = ct::add(ct::max(ct::sub(ct::mul(arg0[idx], arg1[idx]), arg2[idx]), 0.0f), arg3[idx]);",
	} {
		if !strings.Contains(f.lowering.Source, wantSubstr) {
			return fmt.Errorf("tileir: generated source missing %q in:\n%s", wantSubstr, f.lowering.Source)
		}
	}
	return nil
}

func (f tileIRNumericalFixture) mutatedSource() (string, error) {
	if f.lowering == nil {
		return "", errors.New("tileir: missing lowering result")
	}
	if !strings.Contains(f.lowering.Source, "ct::sub(") {
		return "", errors.New("tileir: baseline source missing ct::sub(")
	}
	mut := strings.Replace(f.lowering.Source, "ct::sub(", "ct::add(", 1)
	if mut == f.lowering.Source || strings.Contains(mut, "ct::sub(") {
		return "", errors.New("tileir: failed to replace ct::sub( with ct::add( in mutant source")
	}
	if !strings.Contains(mut, "ct::add(ct::mul(arg0[idx], arg1[idx]), arg2[idx])") {
		return "", errors.New("tileir: mutated source missing ct::add(ct::mul(arg0[idx], arg1[idx]), arg2[idx])")
	}
	if err := tileir.ValidateTileCPPSource(mut); err != nil {
		return "", fmt.Errorf("tileir: mutated source violated Tile C++ contract: %w", err)
	}
	return mut, nil
}

// TestTileIRNumericalFixtureContract runs on CPU with 0 skips and verifies
// closed/open gate admission, cleanup owner retention (including non-zero
// driver.Module handles returned alongside Artifact.Load errors), refusal of
// non-KindTileIR artifacts before driver load, the explicit OutputTileIR mutant
// source/option contract, the independent CPU scalar reference oracle, emitted
// TileIR kernel bindings, NaN/Inf poison rejection, perturbed-output rejection,
// and cudaliveness negative controls.
func TestTileIRNumericalFixtureContract(t *testing.T) {
	checkUnloaded := func() {
		t.Helper()
		if driver.Loaded() || nvrtc.Loaded() {
			t.Fatal("CPU fixture contract test loaded an NVIDIA library")
		}
	}
	checkUnloaded()
	defer checkUnloaded()

	f := newTileIRNumericalFixture(t)
	if err := f.verifyBindings(); err != nil {
		t.Fatalf("verifyBindings: %v", err)
	}
	if err := f.compareOutput(f.want, 1e-6); err != nil {
		t.Fatalf("baseline oracle self-check failed: %v", err)
	}

	// Verify that both positive and negative (clamped) ReLU pre-activations
	// are exercised by the signed input fixture.
	activeCount, clampedCount := 0, 0
	for i := 0; i < f.length; i++ {
		if f.a[i]*f.b[i]-f.c[i] > 0 {
			activeCount++
		} else {
			clampedCount++
		}
	}
	if activeCount == 0 || clampedCount == 0 {
		t.Fatalf("fixture did not exercise both ReLU branches: active=%d clamped=%d", activeCount, clampedCount)
	}

	t.Run("gate_admission", func(t *testing.T) {
		if _, _, err := admitTileIRNativeGate(nil); !errors.Is(err, errTileIRNativeGateClosed) || !errors.Is(err, cudaerrors.ErrHardwarePending) {
			t.Fatalf("nil getenv err = %v, want errTileIRNativeGateClosed wrapping ErrHardwarePending", err)
		}
		if _, _, err := admitTileIRNativeGate(func(string) string { return "" }); !errors.Is(err, errTileIRNativeGateClosed) || !errors.Is(err, cudaerrors.ErrHardwarePending) {
			t.Fatalf("closed gate err = %v, want errTileIRNativeGateClosed wrapping ErrHardwarePending", err)
		}
		if _, _, err := admitTileIRNativeGate(func(k string) string {
			if k == tileIRNativeEnv {
				return "1"
			}
			return ""
		}); !errors.Is(err, cudalock.ErrNotInherited) {
			t.Fatalf("open gate without GPU_LOCK_HELD err = %v, want cudalock.ErrNotInherited", err)
		}
		if _, _, err := admitTileIRNativeGate(func(k string) string {
			switch k {
			case tileIRNativeEnv:
				return "1"
			case "GPU_LOCK_HELD":
				return "1"
			}
			return ""
		}); !errors.Is(err, cudalock.ErrNotInherited) {
			t.Fatalf("open gate with literal GPU_LOCK_HELD=1 err = %v, want cudalock.ErrNotInherited", err)
		}

		// Verify open gate admission with a real host lock lease in a temp dir.
		lockDir := t.TempDir() + "/lock"
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		handle, err := cudalock.Acquire(ctx, cudalock.WithDir(lockDir), cudalock.WithLabel("tileir-fixture-contract"))
		if err != nil {
			t.Fatalf("cudalock.Acquire: %v", err)
		}
		defer func() {
			if err := handle.Release(); err != nil {
				t.Errorf("handle.Release: %v", err)
			}
		}()

		env := map[string]string{
			tileIRNativeEnv: "1",
			"GPU_LOCKDIR":   lockDir,
			"GPU_LOCK_HELD": strconv.Itoa(handle.Owner().PID),
			"TILEIR_DEVICE": "2",
		}
		owner, ordinal, err := admitTileIRNativeGate(func(k string) string { return env[k] })
		if err != nil || owner.PID != handle.Owner().PID || ordinal != 2 {
			t.Fatalf("admitTileIRNativeGate(valid) = (%+v, %d, %v), want PID=%d ordinal=2 nil",
				owner, ordinal, err, handle.Owner().PID)
		}

		env["TILEIR_DEVICE"] = "-1"
		if _, _, err := admitTileIRNativeGate(func(k string) string { return env[k] }); err == nil {
			t.Fatal("negative TILEIR_DEVICE was accepted")
		}
	})

	t.Run("cleanup_owner_retention", func(t *testing.T) {
		for _, stage := range []string{"compile-program", "module-load"} {
			cause := errors.New(stage)
			owner := &tileIRSyntheticCleanupError{cause: cause, fail: true}
			wrapped := fmt.Errorf("tileir native %s: %w", stage, owner)
			var retained tileIRNativeCleanupOwners
			if got := retained.adoptError(wrapped); got != wrapped || !errors.Is(got, cause) {
				t.Fatalf("%s: adoptError replaced original error: %v", stage, got)
			}
			dependentReleased := false
			dependents := func() error {
				dependentReleased = true
				return nil
			}
			if err := retained.closeBefore(dependents); err == nil || owner.calls != 1 || dependentReleased {
				t.Fatalf("%s: failed owner close did not block dependent teardown: err=%v calls=%d released=%v",
					stage, err, owner.calls, dependentReleased)
			}
			owner.fail = false
			if err := retained.closeBefore(dependents); err != nil || owner.calls != 2 || !dependentReleased {
				t.Fatalf("%s: retry owner close failed: err=%v calls=%d released=%v",
					stage, err, owner.calls, dependentReleased)
			}
		}

		// Verify that loadTileIRArtifact preserves a non-zero driver.Module
		// returned alongside an Artifact.Load error (as well as any io.Closer
		// error owner) and unloads the module before dependent teardown.
		rawTileBC := append(slices.Clone(tileir.Magic[:]), 13, 4, 0, 0, 0)
		validArt, err := cudakernel.NewArtifact(
			cudakernel.KindTileIR,
			rawTileBC,
			f.kernelName,
			f.lowering.Schema,
			"sm_121",
		)
		if err != nil {
			t.Fatalf("NewArtifact(KindTileIR): %v", err)
		}

		const errCarriedMod = driver.Module(0xdeadbeef)
		loadCause := errors.New("restore context failed after cuModuleLoadData")
		loadOwner := &tileIRSyntheticCleanupError{cause: loadCause, fail: true}
		wrappedLoadErr := fmt.Errorf("artifact load failed: %w", loadOwner)

		var retained tileIRNativeCleanupOwners
		lookupCalled := false
		_, gotErr := loadTileIRArtifact(
			validArt,
			f.kernelName,
			f.lowering.Schema,
			&retained,
			func(*cudakernel.Artifact) (driver.Module, error) {
				return errCarriedMod, wrappedLoadErr
			},
			func(driver.Module, string) (driver.Function, error) {
				lookupCalled = true
				return driver.Function(0x1234), nil
			},
		)
		if !errors.Is(gotErr, loadCause) || lookupCalled {
			t.Fatalf("loadTileIRArtifact on error-carried module = %v (lookupCalled=%v), want %v",
				gotErr, lookupCalled, loadCause)
		}
		if len(retained.mods) != 1 || retained.mods[0] != errCarriedMod {
			t.Fatalf("retained.mods = %v, want [%#x]", retained.mods, errCarriedMod)
		}
		if len(retained.owners) != 1 {
			t.Fatalf("len(retained.owners) = %d, want 1", len(retained.owners))
		}

		var unloaded []driver.Module
		unloadFail := true
		dependentReleased := false
		unloadMod := func(m driver.Module) error {
			if unloadFail {
				return errors.New("module unload failed")
			}
			unloaded = append(unloaded, m)
			return nil
		}
		dependents := func() error {
			dependentReleased = true
			return nil
		}

		// An absent unloader must never discard retained module ownership.
		missingUnloader := tileIRNativeCleanupOwners{mods: []driver.Module{errCarriedMod}}
		if err := missingUnloader.closeWithModules(nil, nil); err == nil || len(missingUnloader.mods) != 1 {
			t.Fatalf("missing unloader discarded module: err=%v mods=%v", err, missingUnloader.mods)
		}

		// 1. Failing io.Closer owner blocks module unload and dependent teardown.
		if err := retained.closeWithModules(unloadMod, dependents); err == nil || loadOwner.calls != 1 || len(unloaded) != 0 || dependentReleased {
			t.Fatalf("owner failure did not block module/dependent teardown: err=%v calls=%d unloaded=%v released=%v",
				err, loadOwner.calls, unloaded, dependentReleased)
		}
		// 2. Owner succeeds, but failing module unload still blocks dependent teardown.
		loadOwner.fail = false
		if err := retained.closeWithModules(unloadMod, dependents); err == nil || loadOwner.calls != 2 || len(unloaded) != 0 || dependentReleased {
			t.Fatalf("module unload failure did not block dependent teardown: err=%v calls=%d unloaded=%v released=%v",
				err, loadOwner.calls, unloaded, dependentReleased)
		}
		// 3. Module unload retry succeeds and releases dependents.
		unloadFail = false
		if err := retained.closeWithModules(unloadMod, dependents); err != nil || !slices.Equal(unloaded, []driver.Module{errCarriedMod}) || !dependentReleased {
			t.Fatalf("retry module unload failed: err=%v unloaded=%v released=%v",
				err, unloaded, dependentReleased)
		}

		// Verify LIFO unload order across baseline and mutant modules, including
		// when function lookup fails after module load succeeds.
		var lifoRetained tileIRNativeCleanupOwners
		fn1, err := loadTileIRArtifact(
			validArt,
			f.kernelName,
			f.lowering.Schema,
			&lifoRetained,
			func(*cudakernel.Artifact) (driver.Module, error) { return driver.Module(0x1001), nil },
			func(m driver.Module, name string) (driver.Function, error) { return driver.Function(0x2001), nil },
		)
		if err != nil || fn1 != driver.Function(0x2001) {
			t.Fatalf("baseline loadTileIRArtifact = (%#x, %v), want (0x2001, nil)", fn1, err)
		}
		_, err = loadTileIRArtifact(
			validArt,
			f.kernelName,
			f.lowering.Schema,
			&lifoRetained,
			func(*cudakernel.Artifact) (driver.Module, error) { return driver.Module(0x1002), nil },
			func(m driver.Module, name string) (driver.Function, error) {
				return 0, errors.New("missing entry symbol")
			},
		)
		if err == nil {
			t.Fatal("expected lookup error on second module")
		}
		var lifoUnloaded []driver.Module
		if err := lifoRetained.closeWithModules(func(m driver.Module) error {
			lifoUnloaded = append(lifoUnloaded, m)
			return nil
		}, nil); err != nil {
			t.Fatalf("lifo closeWithModules: %v", err)
		}
		if want := []driver.Module{0x1002, 0x1001}; !slices.Equal(lifoUnloaded, want) {
			t.Fatalf("lifoUnloaded = %v, want %v", lifoUnloaded, want)
		}
	})

	t.Run("tileir_artifact_routing_and_mutant_contract", func(t *testing.T) {
		rawTileBC := append(slices.Clone(tileir.Magic[:]), 13, 4, 0, 0, 0)
		ptxArt, err := cudakernel.NewArtifact(cudakernel.KindPTX, rawTileBC, f.kernelName, f.lowering.Schema, "sm_121")
		if err != nil {
			t.Fatalf("NewArtifact(KindPTX): %v", err)
		}
		ltoArt, err := cudakernel.NewArtifact(cudakernel.KindLTOIR, []byte("ltobc"), f.kernelName, f.lowering.Schema, "sm_121")
		if err != nil {
			t.Fatalf("NewArtifact(KindLTOIR): %v", err)
		}
		cubinArt := &cudakernel.Artifact{
			Kind:       cudakernel.KindCUBIN,
			Raw:        rawTileBC,
			EntryPoint: f.kernelName,
			Schema:     f.lowering.Schema,
			Target:     "sm_121",
		}
		nonMagicTileArt, err := cudakernel.NewArtifact(cudakernel.KindTileIR, []byte("not-tileir-bytecode"), f.kernelName, f.lowering.Schema, "sm_121")
		if err != nil {
			t.Fatalf("NewArtifact(nonMagicTileArt): %v", err)
		}
		wrongEntryTileArt, err := cudakernel.NewArtifact(cudakernel.KindTileIR, rawTileBC, "other_entry", f.lowering.Schema, "sm_121")
		if err != nil {
			t.Fatalf("NewArtifact(wrongEntryTileArt): %v", err)
		}
		shortSchema, err := cudakernel.NewSchema(f.lowering.Schema.Params[:4]...)
		if err != nil {
			t.Fatalf("NewSchema(shortSchema): %v", err)
		}
		wrongSchemaTileArt, err := cudakernel.NewArtifact(cudakernel.KindTileIR, rawTileBC, f.kernelName, shortSchema, "sm_121")
		if err != nil {
			t.Fatalf("NewArtifact(wrongSchemaTileArt): %v", err)
		}

		for _, tc := range []struct {
			name string
			art  *cudakernel.Artifact
		}{
			{"nil_artifact", nil},
			{"kind_cubin_refused", cubinArt},
			{"kind_ptx_refused", ptxArt},
			{"kind_ltoir_refused", ltoArt},
			{"missing_tileir_magic_refused", nonMagicTileArt},
			{"wrong_entry_point_refused", wrongEntryTileArt},
			{"wrong_schema_refused", wrongSchemaTileArt},
		} {
			loadCalled := false
			var retained tileIRNativeCleanupOwners
			_, err := loadTileIRArtifact(
				tc.art,
				f.kernelName,
				f.lowering.Schema,
				&retained,
				func(*cudakernel.Artifact) (driver.Module, error) {
					loadCalled = true
					return driver.Module(0x9999), nil
				},
				func(driver.Module, string) (driver.Function, error) {
					t.Fatal("rejected artifact reached lookup")
					return 0, nil
				},
			)
			if err == nil || loadCalled || len(retained.mods) != 0 {
				t.Fatalf("%s: loadTileIRArtifact = %v (loadCalled=%v mods=%v), want pre-load refusal",
					tc.name, err, loadCalled, retained.mods)
			}
		}

		// Verify mutated source construction preserves the Tile C++ contract and
		// changes only the targeted ct::sub -> ct::add operation.
		mutSource, err := f.mutatedSource()
		if err != nil {
			t.Fatalf("mutatedSource: %v", err)
		}
		if mutSource == f.lowering.Source {
			t.Fatal("mutatedSource matched baseline source")
		}

		// Verify explicit OutputTileIR option routing vs rejected WithPhysicalArch/OutputAuto.
		flags, err := nvrtcx.CompileFlags(tileIRCoreCompileOptions(f.kernelName, 12, 1)...)
		if err != nil {
			t.Fatalf("CompileFlags(tileIRCoreCompileOptions): %v", err)
		}
		for _, wantFlag := range []string{"--gpu-architecture=compute_121", "--std=c++20", "--no-cache", "--generate-line-info"} {
			if !slices.Contains(flags, wantFlag) {
				t.Fatalf("tileIRCoreCompileOptions flags %v missing %q", flags, wantFlag)
			}
		}
		// CompileFlags exposes resolved source flags, not the output selector.
		// The source-routing test below binds the explicit output selector to
		// this actual recipe; do not manufacture offline flags as NVRTC evidence.
		if _, err := nvrtcx.CompileFlags(
			nvrtcx.WithPhysicalArch(12, 1),
			nvrtcx.WithOutput(nvrtcx.OutputTileIR),
		); err == nil {
			t.Fatal("WithPhysicalArch + WithOutput(OutputTileIR) was accepted")
		}
		if _, err := nvrtcx.CompileFlags(
			nvrtcx.WithOutput(nvrtcx.OutputTileIR),
			nvrtcx.WithPhysicalArch(12, 1),
		); err == nil {
			t.Fatal("WithOutput(OutputTileIR) + WithPhysicalArch was accepted")
		}
	})

	t.Run("nan_and_inf_poison_rejected", func(t *testing.T) {
		allNaN := make([]float32, f.length)
		for i := range allNaN {
			allNaN[i] = float32(math.NaN())
		}
		if err := f.compareOutput(allNaN, 1e-5); err == nil {
			t.Fatal("unchanged all-NaN poison buffer was accepted")
		}
		for _, tc := range []struct {
			name string
			idx  int
			val  float32
		}{
			{"head_quiet_nan", 0, float32(math.NaN())},
			{"middle_signaling_nan", 7, math.Float32frombits(0x7fa00001)},
			{"tail_neg_nan", f.length - 1, math.Float32frombits(0xffc00123)},
			{"head_pos_inf", 0, float32(math.Inf(1))},
			{"middle_neg_inf", 5, float32(math.Inf(-1))},
			{"tail_pos_inf", f.length - 1, float32(math.Inf(1))},
		} {
			poisoned := slices.Clone(f.want)
			poisoned[tc.idx] = tc.val
			if err := f.compareOutput(poisoned, 1e-5); err == nil {
				t.Fatalf("%s at index %d (%v) was accepted", tc.name, tc.idx, tc.val)
			}
		}
		badRef := f
		badRef.want = slices.Clone(f.want)
		badRef.want[0] = float32(math.NaN())
		if err := badRef.compareOutput(f.want, 1e-5); err == nil {
			t.Fatal("NaN reference value in compareOutput was accepted")
		}
	})

	t.Run("perturbed_and_cudaliveness_controls", func(t *testing.T) {
		for idx := 0; idx < f.length; idx++ {
			perturbed := slices.Clone(f.want)
			perturbed[idx] += 1e-3
			if err := f.compareOutput(perturbed, 1e-5); err == nil {
				t.Fatalf("perturbed output at index %d was accepted", idx)
			}
		}
		if err := f.compareOutput(f.want[:f.length-1], 1e-5); err == nil {
			t.Fatal("truncated output slice was accepted")
		}

		// Verify the planted subtraction->addition mutant changes a known positive
		// number of elements and satisfies cudaliveness.Assert controls.
		mismatches := f.countMismatches(f.mutWant, 1e-5)
		if mismatches == 0 {
			t.Fatal("planted sub->add mutant produced 0 mismatches against baseline")
		}
		if err := cudaliveness.Assert(
			cudaliveness.Planted{
				Site:     "tileir_fused_affine_relu sub->add source mutation",
				Reported: mismatches,
				Ran:      true,
			},
			cudaliveness.Adjacent{
				Site:     "tileir_fused_affine_relu unmutated baseline",
				Reported: f.countMismatches(f.want, 1e-5),
				Ran:      true,
			},
			cudaliveness.Executed{
				Site:      "tileir_fused_affine_relu sub->add kernel image",
				Predicted: float32SliceDigest(f.mutWant),
				Why:       "replacing ct::sub(..., arg2[idx]) with ct::add(..., arg2[idx]) negates C before ReLU",
				Unpatched: float32SliceDigest(f.want),
				Patched:   float32SliceDigest(f.mutWant),
				Ran:       true,
			},
		); err != nil {
			t.Fatalf("cudaliveness.Assert failed on fixture controls: %v", err)
		}
	})
}

// TestTileIRFixtureArtifactRouting checks the fixture's actual call sites.
// This is a source guard, not a compiler acceptance or device execution test.
func TestTileIRFixtureArtifactRouting(t *testing.T) {
	source, err := parser.ParseFile(token.NewFileSet(), "codegen_native_test.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	find := func(name string) *ast.FuncDecl {
		for _, decl := range source.Decls {
			if f, ok := decl.(*ast.FuncDecl); ok && f.Name.Name == name {
				return f
			}
		}
		t.Fatalf("missing fixture function %s", name)
		return nil
	}
	selector := func(expr ast.Expr, pkg, name string) bool {
		s, ok := expr.(*ast.SelectorExpr)
		if !ok {
			return false
		}
		id, ok := s.X.(*ast.Ident)
		return ok && id.Name == pkg && s.Sel.Name == name
	}
	outputs := 0
	ast.Inspect(find("tileIRCoreCompileOptions").Body, func(node ast.Node) bool {
		call, ok := node.(*ast.CallExpr)
		if !ok {
			return true
		}
		if selector(call.Fun, "nvrtcx", "WithPhysicalArch") {
			t.Error("TileIR recipe selects a physical-only output")
		}
		if selector(call.Fun, "nvrtcx", "WithOutput") {
			outputs++
			if len(call.Args) != 1 || !selector(call.Args[0], "nvrtcx", "OutputTileIR") {
				t.Error("fixture must explicitly request OutputTileIR")
			}
		}
		return true
	})
	if outputs != 1 {
		t.Fatalf("explicit TileIR output selections = %d, want 1", outputs)
	}
	// The baseline must come from the semantic compiler's artifact and be passed
	// directly to the same loader used by the mutant, not a recompiled OutputAuto image.
	compiled, loaded := false, false
	ast.Inspect(find("runNativeTileIRNumericalParity").Body, func(node ast.Node) bool {
		assign, ok := node.(*ast.AssignStmt)
		if ok && len(assign.Lhs) == 3 && len(assign.Rhs) == 1 {
			id, isID := assign.Lhs[0].(*ast.Ident)
			call, isCall := assign.Rhs[0].(*ast.CallExpr)
			if isID && id.Name == "tileArtifact" && isCall && selector(call.Fun, "tileir", "CompileSemanticTile") {
				compiled = true
			}
		}
		call, ok := node.(*ast.CallExpr)
		if !ok {
			return true
		}
		id, ok := call.Fun.(*ast.Ident)
		if ok && id.Name == "loadArtifact" && len(call.Args) == 1 {
			arg, ok := call.Args[0].(*ast.Ident)
			if ok && arg.Name == "tileArtifact" {
				loaded = true
			}
		}
		return true
	})
	if !compiled || !loaded {
		t.Fatalf("semantic artifact routing disconnected: compiled=%v loaded=%v", compiled, loaded)
	}
}

// TestNativeTileRegionReference compiles the semantic region to a KindTileIR
// bytecode artifact via CompileSemanticTile, loads and launches that exact
// TileIR artifact on an admitted GPU under gpu-lock, poisons device output
// memory with NaN before launch, reads back the device output buffer,
// explicitly rejects NaN/Inf before tolerance comparison, compares every
// element against the independent CPU scalar reference oracle, and validates a
// cudaliveness negative control compiled and loaded as KindTileIR on device.
func TestNativeTileRegionReference(t *testing.T) {
	runNativeTileIRNumericalParity(t)
}

// TestNativeTileIRNumericalParity is the explicit numerical parity entry point
// sharing the implementation of TestNativeTileRegionReference.
func TestNativeTileIRNumericalParity(t *testing.T) {
	runNativeTileIRNumericalParity(t)
}

func runNativeTileIRNumericalParity(t *testing.T) {
	t.Helper()
	owner, ordinal, err := admitTileIRNativeGate(os.Getenv)
	if errors.Is(err, errTileIRNativeGateClosed) {
		t.Skip(err.Error())
	}
	if err != nil {
		t.Fatal(err)
	}
	if err := driver.Load(""); err != nil {
		t.Fatalf("load CUDA driver: %v", err)
	}
	if err := cuda.Init(); err != nil {
		t.Fatalf("initialize CUDA: %v", err)
	}
	rtcMajor, rtcMinor, err := nvrtc.Version()
	if err != nil {
		t.Fatalf("nvrtc.Version: %v", err)
	}
	if err := tileir.CheckTileCompiler(int(rtcMajor), int(rtcMinor)); err != nil {
		t.Fatalf("tile compiler version check: %v (%s)", err, nvrtc.Path())
	}

	dev, err := cuda.NewDevice(ordinal)
	if err != nil {
		t.Fatalf("open CUDA device %d: %v", ordinal, err)
	}
	var (
		retained tileIRNativeCleanupOwners
		stream   *cuda.Stream
		floatBuf []cudamem.Slice[float32]
	)
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		if stream != nil {
			if err := stream.Synchronize(ctx); err != nil {
				t.Errorf("retain native TileIR resources after stream drain failure: %v", err)
				return
			}
		}
		if err := retained.closeWithModules(
			func(m driver.Module) error {
				return dev.WithContext(func() error { return m.Close() })
			},
			func() error {
				for _, b := range floatBuf {
					if err := b.Free(); err != nil {
						return fmt.Errorf("free float buffer: %w", err)
					}
				}
				if stream != nil {
					if err := stream.Close(); err != nil {
						return fmt.Errorf("close stream: %w", err)
					}
					stream = nil
				}
				if err := dev.Close(); err != nil {
					return fmt.Errorf("close device: %w", err)
				}
				return nil
			},
		); err != nil {
			t.Errorf("native TileIR cleanup failed; retaining remaining resources: %v", err)
		}
	})

	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatalf("compute capability: %v", err)
	}
	f := newTileIRNumericalFixture(t)
	if err := f.verifyBindings(); err != nil {
		t.Fatalf("verifyBindings: %v", err)
	}

	// Compile semantic region to an authentic KindTileIR bytecode artifact via
	// CompileSemanticTile and load/launch that exact artifact on the device.
	compileCtx, cancelCompile := context.WithTimeout(context.Background(), time.Minute)
	tileArtifact, _, err := tileir.CompileSemanticTile(compileCtx, f.snapshot, tileir.LowerOptions{
		KernelName: f.kernelName,
		TileM:      int64(f.length),
		TileN:      1,
		TileK:      1,
	}, major, minor)
	cancelCompile()
	if err != nil {
		_ = retained.adoptError(err)
		t.Fatalf("CompileSemanticTile for sm_%d%d: %v", major, minor, err)
	}

	loadArtifact := func(art *cudakernel.Artifact) driver.Function {
		fn, err := loadTileIRArtifact(
			art,
			f.kernelName,
			f.lowering.Schema,
			&retained,
			func(a *cudakernel.Artifact) (driver.Module, error) {
				return a.Load(dev)
			},
			func(m driver.Module, name string) (fn driver.Function, err error) {
				err = dev.WithContext(func() error {
					var lookupErr error
					fn, lookupErr = m.Function(name)
					return lookupErr
				})
				return fn, err
			},
		)
		if err != nil {
			t.Fatalf("load TileIR artifact %q on device %d: %v", f.kernelName, ordinal, err)
		}
		return fn
	}

	fn := loadArtifact(tileArtifact)
	stream, err = dev.NewStream()
	if err != nil {
		t.Fatalf("create stream: %v", err)
	}

	allocFloats := func(vals []float32) cudamem.Slice[float32] {
		s, err := cudamem.Alloc[float32](dev, len(vals))
		if s.Ptr() != 0 {
			floatBuf = append(floatBuf, s)
		}
		if err != nil {
			t.Fatalf("alloc float32[%d]: %v", len(vals), err)
		}
		if err := dev.WithContext(func() error { return s.CopyFromHost(vals) }); err != nil {
			t.Fatalf("copy float32[%d] to device: %v", len(vals), err)
		}
		return s
	}

	dA := allocFloats(f.a)
	dB := allocFloats(f.b)
	dC := allocFloats(f.c)
	dD := allocFloats(f.d)

	// Allocate output with 2 trailing guard slots and poison all elements with NaN.
	poison := make([]float32, f.length+2)
	for i := range poison {
		poison[i] = float32(math.NaN())
	}
	dOutStorage := allocFloats(poison)
	dOut := dOutStorage.Slice(0, f.length)

	runKernel := func(k driver.Function) []float32 {
		if err := dev.WithContext(func() error { return dOutStorage.CopyFromHost(poison) }); err != nil {
			t.Fatalf("poison device output with NaN: %v", err)
		}
		// Launch 1 extra tile block beyond f.length to exercise the generated
		// if (idx < 16) bounds guard against writing into the trailing NaN guards.
		if err := cudalaunch.Kernel(k).
			Grid(uint32(f.length+1), 1, 1).
			Block(1, 1, 1).
			On(stream).
			Launch(dA, dB, dC, dD, dOut); err != nil {
			t.Fatalf("launch %s: %v", f.kernelName, err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		err := stream.Synchronize(ctx)
		cancel()
		if err != nil {
			t.Fatalf("synchronize %s: %v", f.kernelName, err)
		}
		readback := make([]float32, len(poison))
		if err := dev.WithContext(func() error { return dOutStorage.CopyToHost(readback) }); err != nil {
			t.Fatalf("readback device output: %v", err)
		}
		for gIdx := f.length; gIdx < len(readback); gIdx++ {
			if math.Float32bits(readback[gIdx]) != math.Float32bits(poison[gIdx]) {
				t.Fatalf("trailing guard[%d] overwritten: got %v (0x%08x), want NaN",
					gIdx, readback[gIdx], math.Float32bits(readback[gIdx]))
			}
		}
		return readback[:f.length]
	}

	got := runKernel(fn)
	if err := f.compareOutput(got, 1e-5); err != nil {
		t.Fatalf("device output parity failed: %v", err)
	}

	// Device negative control: mutate the emitted Tile C++ source by replacing
	// ct::sub with ct::add, compile explicitly to a KindTileIR bytecode artifact
	// using the same OutputTileIR option contract as CompileSemanticTile, load
	// and execute through the same Artifact.Load path, and verify via
	// cudaliveness.Assert that the probe detects the planted mutation.
	mutSource, err := f.mutatedSource()
	if err != nil {
		t.Fatalf("construct mutated Tile C++ source: %v", err)
	}
	mutCompileCtx, cancelMutCompile := context.WithTimeout(context.Background(), time.Minute)
	mutArtifact, err := compileTileIRSourceArtifact(
		mutCompileCtx,
		mutSource,
		f.kernelName,
		f.lowering.Schema,
		major,
		minor,
		rtcMajor,
		rtcMinor,
		&retained,
	)
	cancelMutCompile()
	if err != nil {
		t.Fatalf("compile mutated TileIR artifact for sm_%d%d: %v", major, minor, err)
	}
	if mutArtifact.Digest == tileArtifact.Digest {
		t.Fatalf("mutated TileIR artifact digest %s matched baseline digest", mutArtifact.Digest)
	}
	mutFn := loadArtifact(mutArtifact)
	mutGot := runKernel(mutFn)
	if err := f.compareOutput(mutGot, 1e-5); err == nil {
		t.Fatal("device mutant with ct::sub replaced by ct::add was accepted by baseline comparator")
	}
	if err := cudaliveness.Assert(
		cudaliveness.Planted{
			Site:     "device TileIR kernel ct::sub->ct::add mutation",
			Reported: f.countMismatches(mutGot, 1e-5),
			Ran:      true,
		},
		cudaliveness.Adjacent{
			Site:     "device TileIR kernel unmutated baseline",
			Reported: f.countMismatches(got, 1e-5),
			Ran:      true,
		},
		cudaliveness.Executed{
			Site:      "device TileIR kernel ct::sub->ct::add execution",
			Predicted: float32SliceDigest(f.mutWant),
			Why:       "replacing ct::sub with ct::add negates C before ReLU on device",
			Unpatched: float32SliceDigest(got),
			Patched:   float32SliceDigest(mutGot),
			Ran:       true,
		},
	); err != nil {
		t.Fatalf("device cudaliveness.Assert failed: %v", err)
	}

	drvVersion, err := driver.DriverGetVersion()
	if err != nil {
		t.Fatalf("DriverGetVersion: %v", err)
	}
	t.Logf("native TileIR numerical readback: device=%d sm_%d%d driver=%d nvrtc=%d.%d (%s) lease_pid=%d artifact_kind=%s tilebc_digest=%s tilebc_bytes=%d mutant_tilebc_digest=%s mutant_tilebc_bytes=%d elements=%d",
		ordinal, major, minor, drvVersion, rtcMajor, rtcMinor, nvrtc.Path(), owner.PID,
		tileArtifact.Kind, tileArtifact.Digest, len(tileArtifact.Raw),
		mutArtifact.Digest, len(mutArtifact.Raw), f.length)
}
