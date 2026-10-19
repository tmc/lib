package cudasearch

import (
	"bytes"
	"context"
	"errors"
	"io"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaflightrecorder"
	"github.com/tmc/lib/nvidia/x/cudashape"
	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/egraph"
)

// This host executable computes arithmetic in Outputs. It does not represent
// a device measurement; the wrong candidate fails before timing is requested.
type quarantineExecutable struct {
	id                    string
	x, y                  float32
	wrong                 bool
	outputs, runs, closes int
	outputErr, closeErr   error
	outputMode            string
}

func (x *quarantineExecutable) Identity() string { return x.id }
func (x *quarantineExecutable) Outputs(context.Context) (map[string][]float32, error) {
	x.outputs++
	if x.outputErr != nil {
		return nil, x.outputErr
	}
	switch x.outputMode {
	case "missing-name":
		return map[string][]float32{"other": {5}}, nil
	case "extent":
		return map[string][]float32{"out": {5, 5}}, nil
	case "missing-all":
		return nil, nil
	}
	v := x.x + x.y
	if x.wrong {
		v = x.x * x.y
	}
	return map[string][]float32{"out": {v}}, nil
}
func (x *quarantineExecutable) Run(context.Context) (cudatrial.Receipt, error) {
	x.runs++
	return cudatrial.Receipt{}, errors.New("host fixture has no device timing")
}
func (x *quarantineExecutable) Close() error { x.closes++; return x.closeErr }

func quarantineFixture(t *testing.T) (*ExecutableEvaluator, *quarantineExecutable, *quarantineExecutable, *bytes.Buffer) {
	t.Helper()
	r, err := cudaflightrecorder.New(cudaflightrecorder.Config{Capacity: 1, MaxSnapshotBytes: 1 << 20})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { r.Close() })
	base := &quarantineExecutable{id: "baseline", x: 2, y: 3}
	candidate := &quarantineExecutable{id: "opaque executable / entry / geometry", x: 2, y: 3, wrong: true}
	buf := new(bytes.Buffer)
	e := &ExecutableEvaluator{Generation: 3, Workload: WorkloadBindings{Name: "arithmetic", InputNames: []string{"x"}, InputIdentities: map[string]string{"x": "actual-fixture-2-3"}, NumericalPolicy: NumericalPolicyFP32}, Baseline: base, References: map[string][]float32{"out": {5}}, Rounds: 2, TimingKind: cudatrial.TimingSummedKernel, HardwareLockChecker: func() bool { return true }, Compile: func(context.Context, *egraph.Expr, WorkloadBindings) (Executable, error) { return candidate, nil }, FailureEvidence: &FailureEvidence{Recorder: r, Writer: buf}}
	return e, base, candidate, buf
}
func quarantineSearch(e *ExecutableEvaluator) (*SearchResult, error) {
	return NewBeamSearch(BeamConfig{}).Search(context.Background(), executableState(), nil, e)
}

func TestQuarantinePublicSearchRoundTrip(t *testing.T) {
	e, base, candidate, buf := quarantineFixture(t)
	r, err := quarantineSearch(e)
	if r != nil {
		defer r.Close()
	}
	if !errors.Is(err, ErrVerificationFailed) || candidate.outputs != 1 || candidate.runs != 0 || candidate.closes != 1 || buf.Len() == 0 {
		t.Fatalf("err=%v candidate=%+v bytes=%d", err, candidate, buf.Len())
	}
	archive := append([]byte(nil), buf.Bytes()...)
	store, err := NewCandidateQuarantine(4, 16384)
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Import(bytes.NewReader(archive)); err != nil {
		t.Fatal(err)
	}
	e.FailureEvidence.Quarantine = store
	base.outputs = 0
	candidate.outputs = 0
	r, err = quarantineSearch(e)
	if r != nil {
		defer r.Close()
	}
	if !errors.Is(err, ErrCandidateQuarantined) || base.outputs != 0 || candidate.outputs != 0 || candidate.runs != 0 || candidate.closes != 2 {
		t.Fatalf("err=%v base=%+v candidate=%+v", err, base, candidate)
	}
	if r == nil || len(r.Rejected) != 1 || r.Rejected[0].Evidence.ExecutableID != candidate.id {
		t.Fatalf("missing concrete rejection: %+v", r)
	}
	for _, change := range []string{"executable", "input", "policy", "reference", "reference-name", "generation", "workload"} {
		t.Run(change, func(t *testing.T) {
			x, b, c, _ := quarantineFixture(t)
			x.FailureEvidence = &FailureEvidence{Quarantine: store}
			switch change {
			case "executable":
				c.id += " distinct"
			case "input":
				x.Workload.InputIdentities["x"] = "changed"
			case "policy":
				x.Policy.Absolute = 0.1
			case "reference":
				x.References["out"][0] = 7
			case "reference-name":
				x.References = map[string][]float32{"renamed": {5}}
			case "generation":
				x.Generation++
			case "workload":
				x.Workload.Name += " changed"
			}
			// Stop in baseline Outputs to observe admission without inventing timing.
			stop := errors.New("admission reached outputs")
			b.outputErr = stop
			r, err := quarantineSearch(x)
			if r != nil {
				defer r.Close()
			}
			if !errors.Is(err, stop) || b.outputs != 1 || c.closes != 1 {
				t.Fatalf("err=%v base=%+v candidate=%+v", err, b, c)
			}
		})
	}
}

func TestQuarantineOnlyCandidateComparisonExports(t *testing.T) {
	for _, stage := range []string{"baseline", "callback", "verification-callback", "compile", "cancel", "input-missing", "generation-missing"} {
		t.Run(stage, func(t *testing.T) {
			e, b, c, buf := quarantineFixture(t)
			sentinel := errors.New("callback")
			switch stage {
			case "baseline":
				b.wrong = true
			case "callback":
				c.outputErr = sentinel
			case "verification-callback":
				c.outputErr = ErrVerificationFailed
			case "compile":
				e.Compile = func(context.Context, *egraph.Expr, WorkloadBindings) (Executable, error) { return c, sentinel }
			case "cancel":
				e.Compile = func(context.Context, *egraph.Expr, WorkloadBindings) (Executable, error) { return c, context.Canceled }
			case "input-missing":
				e.Workload.InputNames = nil
				e.Workload.InputIdentities = nil
				e.Workload.Shape, _ = cudashape.New(1)
			case "generation-missing":
				e.Generation = 0
			}
			_, err := quarantineSearch(e)
			if err == nil {
				t.Fatal("expected failure")
			}
			if stage == "verification-callback" && !errors.Is(err, ErrVerificationFailed) {
				t.Fatalf("callback primary lost: %v", err)
			}
			if stage == "input-missing" || stage == "generation-missing" {
				store, _ := NewCandidateQuarantine(1, 16384)
				if !errors.Is(err, ErrVerificationFailed) || c.outputs != 1 || buf.Len() == 0 {
					t.Fatalf("missing-identity mismatch lost: %v c=%+v", err, c)
				}
				if err := store.Import(bytes.NewReader(buf.Bytes())); err == nil {
					t.Fatal("missing input association admitted")
				}
			} else if buf.Len() != 0 {
				t.Fatalf("%s exported quarantine", stage)
			}
		})
	}
}

type quarantineFailWriter struct{ err error }

func TestQuarantineIncompleteOutputsWithhold(t *testing.T) {
	for _, mode := range []string{"missing-name", "extent", "missing-all"} {
		t.Run(mode, func(t *testing.T) {
			e, _, c, archive := quarantineFixture(t)
			c.outputMode = mode
			r, err := quarantineSearch(e)
			if r != nil {
				defer r.Close()
			}
			if !errors.Is(err, ErrVerificationFailed) || c.outputs != 1 || c.closes != 1 || archive.Len() == 0 {
				t.Fatalf("primary facts lost %v candidate=%+v", err, c)
			}
			b, err := cudaflightrecorder.ReadBundle(bytes.NewReader(archive.Bytes()))
			if err != nil {
				t.Fatal(err)
			}
			f, err := cudaflightrecorder.DecodeFaultInfo(b.Files["fault.json"])
			if err != nil {
				t.Fatal(err)
			}
			if f.Candidate != nil || f.ErrorName != cudaflightrecorder.CandidateReferenceMismatch {
				t.Fatalf("incomplete output association admitted %+v", f)
			}
			store, _ := NewCandidateQuarantine(1, 16384)
			if err := store.Import(bytes.NewReader(archive.Bytes())); err == nil {
				t.Fatal("incomplete outputs established ban")
			}
			entries, _ := store.Retained()
			if entries != 0 {
				t.Fatal("store changed")
			}
		})
	}
}

func (w quarantineFailWriter) Write([]byte) (int, error) { return 0, w.err }

func TestQuarantineErrorsAndRetry(t *testing.T) {
	e, _, c, _ := quarantineFixture(t)
	writeErr := errors.New("archive write failed")
	closeErr := errors.New("close retry")
	e.FailureEvidence.Writer = quarantineFailWriter{writeErr}
	c.closeErr = closeErr
	r, err := quarantineSearch(e)
	if r != nil {
		defer r.Close()
	}
	if !errors.Is(err, ErrVerificationFailed) || !errors.Is(err, writeErr) || !errors.Is(err, closeErr) || len(e.pending) != 1 {
		t.Fatalf("error identity or owner lost: %v pending=%d", err, len(e.pending))
	}
	if err := e.Close(); !errors.Is(err, closeErr) {
		t.Fatal(err)
	}
	c.closeErr = nil
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	if len(e.pending) != 0 || c.closes != 3 {
		t.Fatalf("retry owner=%+v", c)
	}
}

func TestQuarantineImportAtomicAndLegacy(t *testing.T) {
	e, _, _, buf := quarantineFixture(t)
	quarantineSearch(e)
	b, err := cudaflightrecorder.ReadBundle(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	store, _ := NewCandidateQuarantine(1, 16384)
	if err := store.Import(bytes.NewReader(buf.Bytes())); err != nil {
		t.Fatal(err)
	}
	n, bytesBefore := store.Retained()
	if err := store.Import(bytes.NewReader(buf.Bytes())); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"different", "future", "mismatch", "module-only", "broad", "missing"} {
		t.Run(kind, func(t *testing.T) {
			fault, constraint, err := b.DecodeFaultEvidence()
			if err != nil {
				t.Fatal(err)
			}
			x := *fault.Candidate
			y := *constraint.Candidate
			fault.Candidate = &x
			constraint.Candidate = &y
			switch kind {
			case "different":
				x.ExecutableID += " changed"
				y = x
			case "future":
				x.Version = 2
				y = x
			case "mismatch":
				y.InputKey = strings.Repeat("a", 64)
			case "module-only":
				fault.Candidate = nil
				constraint.Candidate = nil
				fault.ModuleDigest = strings.Repeat("a", 64)
				constraint.CandidateDigest = fault.ModuleDigest
			case "broad":
				constraint.PruneSubtree = true
			case "missing":
				fault.Candidate = nil
			}
			var out bytes.Buffer
			if kind == "future" { // Unknown wire version cannot be written by supported encoder.
				for k, v := range b.Files {
					if k == "fault.json" || k == "negative_rules.json" {
						v = bytes.ReplaceAll(v, []byte(`"version": 1`), []byte(`"version": 2`))
						b.Files[k] = v
					}
				}
				defer func() {
					for k, v := range b.Files {
						b.Files[k] = bytes.ReplaceAll(v, []byte(`"version": 2`), []byte(`"version": 1`))
					}
				}()
				cudaflightrecorder.WriteBundle(&out, b)
			} else {
				// Inject unsupported broad policy on the wire; the supported
				// writer itself correctly refuses to produce it.
				if kind == "broad" {
					constraint.PruneSubtree = false
				}
				fb, err := fault.MarshalJSON()
				if err != nil {
					t.Fatal(err)
				}
				cb, err := constraint.MarshalJSON()
				if err != nil {
					t.Fatal(err)
				}
				if kind == "broad" {
					cb = bytes.ReplaceAll(cb, []byte(`"prune_subtree":false`), []byte(`"prune_subtree":true`))
				}
				cudaflightrecorder.WriteBundle(&out, &cudaflightrecorder.Bundle{Files: map[string][]byte{"fault.json": fb, "negative_rules.json": cb}})
			}
			if err := store.Import(&out); err == nil {
				t.Fatalf("%s admitted", kind)
			}
			now, size := store.Retained()
			if now != n || size != bytesBefore {
				t.Fatalf("store changed %d/%d", now, size)
			}
		})
	}
	if err := store.Import(io.LimitReader(bytes.NewReader(buf.Bytes()), 1)); err == nil {
		t.Fatal("truncated bundle admitted")
	}
}

func TestQuarantineMetadataBudget(t *testing.T) {
	e, _, c, _ := quarantineFixture(t)
	e.References["out"] = make([]float32, (1<<20)+1)
	_, err := quarantineSearch(e)
	if !errors.Is(err, ErrMemoryBudgetExceeded) || c.outputs != 0 || c.closes != 1 {
		t.Fatalf("err=%v c=%+v", err, c)
	}
}

func TestQuarantineRetainedByteBoundary(t *testing.T) {
	e, _, _, archive := quarantineFixture(t)
	quarantineSearch(e)
	b, err := cudaflightrecorder.ReadBundle(bytes.NewReader(archive.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	f, _, err := b.DecodeFaultEvidence()
	if err != nil {
		t.Fatal(err)
	}
	a := f.Candidate
	stringsBytes := int64(len(a.Kind) + len(a.IdentityProvenance) + len(a.ExecutableID) + len(a.WorkloadKey) + len(a.InputKey) + len(a.VerificationKey))
	probe, _ := NewCandidateQuarantine(1, 16384)
	_, base := probe.Retained()
	for _, delta := range []int64{-1, 0} {
		t.Run(map[int64]string{-1: "one-short", 0: "exact"}[delta], func(t *testing.T) {
			q, err := NewCandidateQuarantine(1, base+stringsBytes+delta)
			if err != nil {
				t.Fatal(err)
			}
			err = q.Import(bytes.NewReader(archive.Bytes()))
			n, retained := q.Retained()
			if delta < 0 {
				if !errors.Is(err, ErrMemoryBudgetExceeded) || n != 0 || retained != base {
					t.Fatalf("partial admission %v %d/%d", err, n, retained)
				}
			} else if err != nil || n != 1 || retained != base+stringsBytes {
				t.Fatalf("exact %v %d/%d", err, n, retained)
			}
		})
	}
}
