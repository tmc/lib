// SPDX-License-Identifier: Apache-2.0

package cudamtp_test

import (
	"context"
	"errors"
	"fmt"
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
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudamtp"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

var errMTPNativeGateClosed error = mtpNativeGateClosedError{}

type mtpNativeGateClosedError struct{}

func (mtpNativeGateClosedError) Error() string {
	return "native qualification pending: CUDAMTP_NATIVE=1 required"
}

func (mtpNativeGateClosedError) Unwrap() error {
	return cudaerrors.ErrHardwarePending
}

func admitMTPNativeGate(getenv func(string) string) (cudalock.Owner, int, error) {
	if getenv == nil || getenv("CUDAMTP_NATIVE") != "1" {
		return cudalock.Owner{}, 0, errMTPNativeGateClosed
	}
	owner, err := cudalock.Inherited(getenv)
	if err != nil {
		return cudalock.Owner{}, 0, fmt.Errorf("cudamtp: native gate enabled but %w: run under gpu-lock", err)
	}
	ordinal := 0
	if raw := strings.TrimSpace(getenv("CUDAMTP_DEVICE")); raw != "" {
		parsed, err := strconv.Atoi(raw)
		if err != nil || parsed < 0 {
			return cudalock.Owner{}, 0, fmt.Errorf("cudamtp: invalid CUDAMTP_DEVICE=%q", raw)
		}
		ordinal = parsed
	}
	return owner, ordinal, nil
}

func finiteWithinTol(got, want, tol float32) bool {
	if math.IsNaN(float64(got)) || math.IsInf(float64(got), 0) ||
		math.IsNaN(float64(want)) || math.IsInf(float64(want), 0) ||
		math.IsNaN(float64(tol)) || math.IsInf(float64(tol), 0) || tol < 0 {
		return false
	}
	return math.Abs(float64(got-want)) <= float64(tol)
}

// mtpNativeCleanupOwners retains io.Closer cleanup handles returned inside
// nvrtcx.Compile or CompiledModule.Load errors and closes them in LIFO order
// before dependent device resources are torn down.
type mtpNativeCleanupOwners struct {
	owners []io.Closer
}

func (r *mtpNativeCleanupOwners) adoptError(err error) error {
	var owner io.Closer
	if errors.As(err, &owner) && owner != nil {
		r.owners = append(r.owners, owner)
	}
	return err
}

func (r *mtpNativeCleanupOwners) closeBefore(dependents func() error) error {
	for len(r.owners) > 0 {
		last := len(r.owners) - 1
		if err := r.owners[last].Close(); err != nil {
			return fmt.Errorf("cudamtp: close retained compiler/module owner: %w", err)
		}
		r.owners = r.owners[:last]
	}
	if dependents == nil {
		return nil
	}
	return dependents()
}

type mtpSyntheticCleanupError struct {
	cause error
	calls int
	fail  bool
}

func (e *mtpSyntheticCleanupError) Error() string { return e.cause.Error() }
func (e *mtpSyntheticCleanupError) Unwrap() error { return e.cause }
func (e *mtpSyntheticCleanupError) Close() error {
	e.calls++
	if e.fail {
		return errors.New("synthetic owner cleanup still failing")
	}
	return nil
}

// mtpProducerStep binds an MTP draft forward invocation to an explicit step ID,
// sequence ID, branch ID, and Wait/Done quiescence callbacks.
type mtpProducerStep struct {
	StepID   int64
	SeqID    string
	BranchID int64
}

type mtpProducerOutcome struct {
	StepID         int64
	SeqID          string
	BranchID       int64
	Logits         []float32
	DraftTokens    []int32
	BufferRetained bool
}

type mtpNativeFixture struct {
	cfg        cudamtp.MTPConfig
	step       mtpProducerStep
	hTrunk     []float32
	wDraftProj []float32
	wLMHeads   []float32
	wantLogits []float32
	wantDrafts []int32
}

func newMTPNativeFixture(t *testing.T) mtpNativeFixture {
	t.Helper()
	cfg := cudamtp.MTPConfig{
		NumDraftHeads: 3,
		Dim:           8,
		VocabSize:     12,
	}
	hTrunk := make([]float32, cfg.Dim)
	for d := 0; d < cfg.Dim; d++ {
		hTrunk[d] = float32(d+1) * 0.25
	}

	// Construct distinct projection matrices for each draft head so that
	// each head routes energy to a distinct target vocabulary column.
	wDraftProj := make([]float32, cfg.NumDraftHeads*cfg.Dim*cfg.Dim)
	for h := 0; h < cfg.NumDraftHeads; h++ {
		for r := 0; r < cfg.Dim; r++ {
			for c := 0; c < cfg.Dim; c++ {
				val := float32(((h+1)*5+(r+1)*3+c)%11-5) * 0.125
				if r == (c+h)%cfg.Dim {
					val += float32(h+1) * 0.75
				}
				wDraftProj[(h*cfg.Dim+r)*cfg.Dim+c] = val
			}
		}
	}

	targetCols := []int{2, 7, 10}
	wLMHeads := make([]float32, cfg.NumDraftHeads*cfg.Dim*cfg.VocabSize)
	for h := 0; h < cfg.NumDraftHeads; h++ {
		for d := 0; d < cfg.Dim; d++ {
			for v := 0; v < cfg.VocabSize; v++ {
				val := float32(((h+1)*7+(d+1)*3+v)%9-4) * 0.0625
				if v == targetCols[h] {
					val += 2.5
				}
				wLMHeads[(h*cfg.Dim+d)*cfg.VocabSize+v] = val
			}
		}
	}

	wantLogits, err := cudamtp.MTPReference(hTrunk, wDraftProj, wLMHeads, cfg)
	if err != nil {
		t.Fatalf("MTPReference: %v", err)
	}
	wantDrafts, err := greedyDraftTokens(wantLogits, cfg)
	if err != nil {
		t.Fatalf("greedyDraftTokens: %v", err)
	}
	return mtpNativeFixture{
		cfg: cfg,
		step: mtpProducerStep{
			StepID:   1,
			SeqID:    "spec-seq-11-gen-1",
			BranchID: 1,
		},
		hTrunk:     hTrunk,
		wDraftProj: wDraftProj,
		wLMHeads:   wLMHeads,
		wantLogits: wantLogits,
		wantDrafts: wantDrafts,
	}
}

func greedyDraftTokens(logits []float32, cfg cudamtp.MTPConfig) ([]int32, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if len(logits) != cfg.NumDraftHeads*cfg.VocabSize {
		return nil, fmt.Errorf("cudamtp: logits length = %d, want %d", len(logits), cfg.NumDraftHeads*cfg.VocabSize)
	}
	drafts := make([]int32, cfg.NumDraftHeads)
	for h := 0; h < cfg.NumDraftHeads; h++ {
		row := logits[h*cfg.VocabSize : (h+1)*cfg.VocabSize]
		best := 0
		for v, val := range row {
			if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
				return nil, fmt.Errorf("cudamtp: non-finite logit at head %d vocab %d: %v", h, v, val)
			}
			if val > row[best] {
				best = v
			}
		}
		drafts[h] = int32(best)
	}
	return drafts, nil
}

// runProducerContract executes a draft producer function under the exact-ID
// and Wait/Done completion contract, retaining buffer leases whenever drain
// completion is not proven.
func (f mtpNativeFixture) runProducerContract(
	reqStep mtpProducerStep,
	produce func(mtpProducerStep) (mtpProducerStep, []float32, error),
	wait func(mtpProducerStep) error,
	done func(mtpProducerStep) bool,
) (*mtpProducerOutcome, error) {
	if err := f.cfg.Validate(); err != nil {
		return nil, err
	}
	if reqStep.StepID <= 0 || reqStep.SeqID == "" || reqStep.BranchID <= 0 {
		return nil, errors.New("cudamtp: invalid producer step identity")
	}
	if produce == nil || wait == nil {
		return nil, errors.New("cudamtp: missing producer or wait callback")
	}

	outcome := &mtpProducerOutcome{
		StepID:         reqStep.StepID,
		SeqID:          reqStep.SeqID,
		BranchID:       reqStep.BranchID,
		BufferRetained: true,
	}
	gotStep, logits, prodErr := produce(reqStep)
	if waitErr := wait(reqStep); waitErr != nil {
		if done != nil && done(reqStep) {
			outcome.BufferRetained = false
		}
		return outcome, fmt.Errorf("cudamtp: producer drain: %w", waitErr)
	}
	if done != nil && !done(reqStep) {
		return outcome, errors.New("cudamtp: producer completion not established")
	}
	outcome.BufferRetained = false
	if prodErr != nil {
		return outcome, prodErr
	}
	if gotStep != reqStep {
		return outcome, fmt.Errorf("cudamtp: foreign producer step %+v, want %+v", gotStep, reqStep)
	}
	if err := f.verifyLogits(logits, 1e-5); err != nil {
		return outcome, err
	}
	drafts, err := greedyDraftTokens(logits, f.cfg)
	if err != nil {
		return outcome, err
	}
	outcome.Logits = slices.Clone(logits)
	outcome.DraftTokens = drafts
	if !slices.Equal(outcome.DraftTokens, f.wantDrafts) {
		return outcome, fmt.Errorf("cudamtp: draft tokens = %v, want %v", outcome.DraftTokens, f.wantDrafts)
	}
	return outcome, nil
}

func (f mtpNativeFixture) verifyLogits(got []float32, tol float32) error {
	if len(got) != len(f.wantLogits) {
		return fmt.Errorf("logits length = %d, want %d", len(got), len(f.wantLogits))
	}
	for i, w := range f.wantLogits {
		if !finiteWithinTol(got[i], w, tol) {
			return fmt.Errorf("logit[%d] = %v, want %v (tol %v)", i, got[i], w, tol)
		}
	}
	gotDrafts, err := greedyDraftTokens(got, f.cfg)
	if err != nil {
		return err
	}
	if !slices.Equal(gotDrafts, f.wantDrafts) {
		return fmt.Errorf("greedy draft tokens = %v, want %v", gotDrafts, f.wantDrafts)
	}
	return nil
}

// TestMTPNativeFixtureContract exercises the MTP fixture, independent float64
// reference parity, exact-ID/Wait/Done producer lifecycle, cleanup owner
// retention, and negative controls (including NaN/Inf poison rejection) on CPU.
func TestMTPNativeFixtureContract(t *testing.T) {
	checkUnloaded := func() {
		t.Helper()
		if driver.Loaded() || nvrtc.Loaded() {
			t.Fatal("CPU fixture contract test loaded an NVIDIA library")
		}
	}
	checkUnloaded()
	defer checkUnloaded()

	f := newMTPNativeFixture(t)
	if !slices.Equal(f.wantDrafts, []int32{2, 7, 10}) {
		t.Fatalf("fixture greedy drafts = %v, want [2 7 10]", f.wantDrafts)
	}

	src := cudamtp.GenerateMTPCudaSource(f.cfg)
	for _, want := range []string{
		"MTPMultiHeadForwardCUDA",
		fmt.Sprintf("if (head_idx >= %d) return;", f.cfg.NumDraftHeads),
		"extern __shared__ float s_hidden[]",
	} {
		if !strings.Contains(src, want) {
			t.Fatalf("GenerateMTPCudaSource missing %q", want)
		}
	}

	// Positive producer contract execution.
	cpuProduce := func(s mtpProducerStep) (mtpProducerStep, []float32, error) {
		logits, err := cudamtp.MTPReference(f.hTrunk, f.wDraftProj, f.wLMHeads, f.cfg)
		return s, logits, err
	}
	outcome, err := f.runProducerContract(
		f.step,
		cpuProduce,
		func(mtpProducerStep) error { return nil },
		func(s mtpProducerStep) bool { return s == f.step },
	)
	if err != nil {
		t.Fatalf("runProducerContract failed: %v", err)
	}
	if outcome.BufferRetained || !slices.Equal(outcome.DraftTokens, f.wantDrafts) {
		t.Fatalf("unexpected outcome: %+v", outcome)
	}

	t.Run("gate_admission", func(t *testing.T) {
		if _, _, err := admitMTPNativeGate(func(string) string { return "" }); !errors.Is(err, errMTPNativeGateClosed) || !errors.Is(err, cudaerrors.ErrHardwarePending) {
			t.Fatalf("closed gate err = %v, want errMTPNativeGateClosed wrapping ErrHardwarePending", err)
		}
		if _, _, err := admitMTPNativeGate(func(k string) string {
			if k == "CUDAMTP_NATIVE" {
				return "1"
			}
			return ""
		}); !errors.Is(err, cudalock.ErrNotInherited) {
			t.Fatalf("open gate without GPU_LOCK_HELD err = %v, want cudalock.ErrNotInherited", err)
		}
		if _, _, err := admitMTPNativeGate(func(k string) string {
			switch k {
			case "CUDAMTP_NATIVE":
				return "1"
			case "GPU_LOCK_HELD":
				return "1"
			}
			return ""
		}); !errors.Is(err, cudalock.ErrNotInherited) {
			t.Fatalf("open gate with literal GPU_LOCK_HELD=1 err = %v, want cudalock.ErrNotInherited", err)
		}
	})

	t.Run("cleanup_owner_retention", func(t *testing.T) {
		for _, stage := range []string{"compile-program", "module-load"} {
			cause := errors.New(stage)
			owner := &mtpSyntheticCleanupError{cause: cause, fail: true}
			wrapped := fmt.Errorf("mtp native %s: %w", stage, owner)
			var retained mtpNativeCleanupOwners
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
	})

	t.Run("nan_and_inf_poison_rejected", func(t *testing.T) {
		allNaN := make([]float32, len(f.wantLogits))
		for i := range allNaN {
			allNaN[i] = float32(math.NaN())
		}
		if err := f.verifyLogits(allNaN, 1e-5); err == nil {
			t.Fatal("unchanged all-NaN poison logits were accepted")
		}
		if _, err := greedyDraftTokens(allNaN, f.cfg); err == nil {
			t.Fatal("greedyDraftTokens accepted all-NaN poison logits")
		}
		for _, tc := range []struct {
			name string
			idx  int
			val  float32
		}{
			{"argmax_nan", 2, float32(math.NaN())},
			{"non_argmax_nan", 0, float32(math.NaN())},
			{"tail_nan", len(f.wantLogits) - 1, float32(math.NaN())},
			{"argmax_pos_inf", 2, float32(math.Inf(1))},
			{"non_argmax_neg_inf", 1, float32(math.Inf(-1))},
		} {
			poisoned := slices.Clone(f.wantLogits)
			poisoned[tc.idx] = tc.val
			if err := f.verifyLogits(poisoned, 1e-5); err == nil {
				t.Fatalf("%s at index %d (%v) was accepted by verifyLogits", tc.name, tc.idx, tc.val)
			}
			if _, err := f.runProducerContract(
				f.step,
				func(s mtpProducerStep) (mtpProducerStep, []float32, error) {
					return s, slices.Clone(poisoned), nil
				},
				func(mtpProducerStep) error { return nil },
				func(mtpProducerStep) bool { return true },
			); err == nil {
				t.Fatalf("%s at index %d (%v) was accepted by runProducerContract", tc.name, tc.idx, tc.val)
			}
		}
		badRef := f
		badRef.wantLogits = slices.Clone(f.wantLogits)
		badRef.wantLogits[0] = float32(math.NaN())
		if err := badRef.verifyLogits(f.wantLogits, 1e-5); err == nil {
			t.Fatal("NaN reference value in verifyLogits was accepted")
		}
	})

	t.Run("foreign_step_or_branch_id_rejected", func(t *testing.T) {
		for _, mutStep := range []mtpProducerStep{
			{StepID: f.step.StepID + 1, SeqID: f.step.SeqID, BranchID: f.step.BranchID},
			{StepID: f.step.StepID, SeqID: "foreign-seq", BranchID: f.step.BranchID},
			{StepID: f.step.StepID, SeqID: f.step.SeqID, BranchID: f.step.BranchID + 1},
		} {
			_, err := f.runProducerContract(
				f.step,
				func(mtpProducerStep) (mtpProducerStep, []float32, error) {
					return mutStep, slices.Clone(f.wantLogits), nil
				},
				func(mtpProducerStep) error { return nil },
				func(mtpProducerStep) bool { return true },
			)
			if err == nil {
				t.Fatalf("foreign producer identity %+v was accepted", mutStep)
			}
		}
	})

	t.Run("wait_and_done_control_buffer_retention", func(t *testing.T) {
		// 1. Wait fails and Done is false -> buffer must stay retained.
		out, err := f.runProducerContract(
			f.step,
			cpuProduce,
			func(mtpProducerStep) error { return errors.New("stream still active") },
			func(mtpProducerStep) bool { return false },
		)
		if err == nil || out == nil || !out.BufferRetained {
			t.Fatalf("undrained wait failure: out=%+v err=%v, want BufferRetained=true", out, err)
		}

		// 2. Wait fails but Done proves exact step quiescence -> buffer may release.
		out, err = f.runProducerContract(
			f.step,
			cpuProduce,
			func(mtpProducerStep) error { return errors.New("step errored after drain") },
			func(s mtpProducerStep) bool { return s.StepID == f.step.StepID },
		)
		if err == nil || out == nil || out.BufferRetained {
			t.Fatalf("quiescent failed step: out=%+v err=%v, want BufferRetained=false", out, err)
		}

		// 3. Wait returns nil but Done is false -> completion not established, retain buffer.
		out, err = f.runProducerContract(
			f.step,
			cpuProduce,
			func(mtpProducerStep) error { return nil },
			func(mtpProducerStep) bool { return false },
		)
		if err == nil || out == nil || !out.BufferRetained {
			t.Fatalf("unconfirmed completion: out=%+v err=%v, want BufferRetained=true", out, err)
		}
	})

	t.Run("swapped_draft_head_weights_rejected", func(t *testing.T) {
		mutProj := slices.Clone(f.wDraftProj)
		headSpan := f.cfg.Dim * f.cfg.Dim
		for i := 0; i < headSpan; i++ {
			mutProj[i], mutProj[headSpan+i] = mutProj[headSpan+i], mutProj[i]
		}
		mutLogits, err := cudamtp.MTPReference(f.hTrunk, mutProj, f.wLMHeads, f.cfg)
		if err != nil {
			t.Fatalf("MTPReference: %v", err)
		}
		if err := f.verifyLogits(mutLogits, 1e-5); err == nil {
			t.Fatal("swapping head 0 and head 1 projection weights was accepted")
		}
	})

	t.Run("corrupted_tail_logit_rejected", func(t *testing.T) {
		corrupt := slices.Clone(f.wantLogits)
		corrupt[len(corrupt)-1] += 0.25
		if err := f.verifyLogits(corrupt, 1e-5); err == nil {
			t.Fatal("corrupted tail logit was accepted")
		}
	})
}

// TestNativeMTPMultiHeadForward compiles and runs MTPMultiHeadForwardCUDA on
// an admitted GPU under gpu-lock, verifying logits and greedy draft tokens
// against MTPReference as well as device mutant rejection.
func TestNativeMTPMultiHeadForward(t *testing.T) {
	owner, ordinal, err := admitMTPNativeGate(os.Getenv)
	if errors.Is(err, errMTPNativeGateClosed) {
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
	dev, err := cuda.NewDevice(ordinal)
	if err != nil {
		t.Fatalf("open CUDA device %d: %v", ordinal, err)
	}
	var (
		retained mtpNativeCleanupOwners
		stream   *cuda.Stream
		mod      driver.Module
		floatBuf []cudamem.Slice[float32]
	)
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		if stream != nil {
			if err := stream.Synchronize(ctx); err != nil {
				t.Errorf("retain native MTP resources after stream drain failure: %v", err)
				return
			}
		}
		if err := retained.closeBefore(func() error {
			if mod != 0 {
				if err := dev.WithContext(func() error { return mod.Close() }); err != nil {
					return fmt.Errorf("close module: %w", err)
				}
				mod = 0
			}
			for _, b := range floatBuf {
				if err := b.Free(); err != nil {
					return fmt.Errorf("free buffer: %w", err)
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
		}); err != nil {
			t.Errorf("native MTP cleanup failed; retaining remaining resources: %v", err)
		}
	})

	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatalf("compute capability: %v", err)
	}
	f := newMTPNativeFixture(t)
	compiled, err := nvrtcx.Compile(
		cudamtp.GenerateMTPCudaSource(f.cfg),
		nvrtcx.WithName("mtp_multi_head.cu"),
		nvrtcx.WithPhysicalArch(major, minor),
	)
	if err != nil {
		_ = retained.adoptError(err)
		t.Fatalf("compile MTPMultiHeadForwardCUDA for sm_%d%d: %v", major, minor, err)
	}
	mod, err = compiled.Load(dev)
	if err != nil {
		_ = retained.adoptError(err)
		t.Fatalf("load MTPMultiHeadForwardCUDA module: %v", err)
	}
	fn, err := mod.Function("MTPMultiHeadForwardCUDA")
	if err != nil {
		t.Fatalf("lookup MTPMultiHeadForwardCUDA: %v", err)
	}
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

	dTrunk := allocFloats(f.hTrunk)
	dProj := allocFloats(f.wDraftProj)
	dLM := allocFloats(f.wLMHeads)
	poison := make([]float32, len(f.wantLogits))
	for i := range poison {
		poison[i] = float32(math.NaN())
	}
	dOut := allocFloats(poison)

	const blockSize = 32
	gridX := uint32((f.cfg.VocabSize + blockSize - 1) / blockSize)
	gridY := uint32(f.cfg.NumDraftHeads)
	sharedBytes := uint32(f.cfg.Dim * 4)

	var stepSynced bool
	outcome, err := f.runProducerContract(
		f.step,
		func(s mtpProducerStep) (mtpProducerStep, []float32, error) {
			if err := cudalaunch.Kernel(fn).
				Grid(gridX, gridY, 1).
				Block(blockSize, 1, 1).
				SharedMem(sharedBytes).
				On(stream).
				Launch(dTrunk, dProj, dLM, dOut, int32(f.cfg.Dim), int32(f.cfg.VocabSize)); err != nil {
				return s, nil, err
			}
			ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
			defer cancel()
			if err := stream.Synchronize(ctx); err != nil {
				return s, nil, err
			}
			stepSynced = true
			got := make([]float32, len(f.wantLogits))
			if err := dev.WithContext(func() error { return dOut.CopyToHost(got) }); err != nil {
				return s, nil, err
			}
			return s, got, nil
		},
		func(mtpProducerStep) error {
			ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
			defer cancel()
			if err := stream.Synchronize(ctx); err != nil {
				return err
			}
			stepSynced = true
			return nil
		},
		func(s mtpProducerStep) bool {
			return stepSynced && s == f.step
		},
	)
	if err != nil {
		t.Fatalf("native MTP producer contract failed: %v", err)
	}

	// Device mutant: swap head 0 and head 1 projection weights on device.
	mutProj := slices.Clone(f.wDraftProj)
	headSpan := f.cfg.Dim * f.cfg.Dim
	for i := 0; i < headSpan; i++ {
		mutProj[i], mutProj[headSpan+i] = mutProj[headSpan+i], mutProj[i]
	}
	dMutProj := allocFloats(mutProj)
	if err := dev.WithContext(func() error { return dOut.CopyFromHost(poison) }); err != nil {
		t.Fatalf("poison output: %v", err)
	}
	if err := cudalaunch.Kernel(fn).
		Grid(gridX, gridY, 1).
		Block(blockSize, 1, 1).
		SharedMem(sharedBytes).
		On(stream).
		Launch(dTrunk, dMutProj, dLM, dOut, int32(f.cfg.Dim), int32(f.cfg.VocabSize)); err != nil {
		t.Fatalf("launch mutant: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	err = stream.Synchronize(ctx)
	cancel()
	if err != nil {
		t.Fatalf("synchronize mutant: %v", err)
	}
	mutGot := make([]float32, len(f.wantLogits))
	if err := dev.WithContext(func() error { return dOut.CopyToHost(mutGot) }); err != nil {
		t.Fatalf("copy mutant output: %v", err)
	}
	if err := f.verifyLogits(mutGot, 1e-5); err == nil {
		t.Fatal("device mutant with swapped draft head projection weights matched baseline reference")
	}
	t.Logf("native MTP multi-head forward qualified: device=%d sm_%d%d lease_pid=%d heads=%d dim=%d vocab=%d drafts=%v",
		ordinal, major, minor, owner.PID, f.cfg.NumDraftHeads, f.cfg.Dim, f.cfg.VocabSize, outcome.DraftTokens)
}
