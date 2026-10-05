package cudahotswap

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/internal/nativereceipt"
	"github.com/tmc/lib/nvidia/x/cudahotswap/stage"
)

type stageNativeObservation struct {
	Index                      int
	Arm                        string
	Stage                      State
	Prepared                   bool
	ArtifactGeneration         uint64
	SourceSHA256               string
	Artifact, ArtifactSHA256   string
	Context, Stream, Function  uint64
	ExpectedStream             uint64
	StreamDomain               string
	KernelVersion              uint64
	Geometry, ExpectedGeometry [6]uint32
	Completed                  bool
}

type stageNativeOwnership struct {
	Version, KernelVersion uint64
	Function               uint64
	BeforePending          bool
	BeforeLibraryPinned    bool
	BeforeFunctionsPinned  bool
	AfterComplete          bool
	AfterLibraryReleased   bool
	AfterFunctionsReleased bool
	AfterLeaseReleased     bool
}

type stageNativeReceipt struct {
	Version      int
	Process      nativereceipt.Process
	Observations []stageNativeObservation
	Controls     []string
	Ownership    stageNativeOwnership
	Cleanup      bool
	Scope        string
}

var stageNativeControls = []string{"named_abi_refusal", "compiler_abi_refusal", "wrong_tail_refusal", "geometry_change_pass", "checker_rollback", "swap_while_pinned", "revert_output"}

func stageNativeArtifactIdentity(v *stageVersion) (stageNativeObservation, error) {
	if v == nil || v.artifact == nil {
		return stageNativeObservation{}, errors.New("stage receipt: missing actual compiled artifact")
	}
	o := stageNativeObservation{Stage: v.state, Prepared: v.state.Version == 0, ArtifactGeneration: v.artifact.Generation(), SourceSHA256: v.artifact.SourceDigest(), Artifact: v.artifact.Path()}
	if o.Prepared && o.Stage != (State{}) || !o.Prepared && o.Stage.SourceDigest != o.SourceSHA256 {
		return o, errors.New("stage receipt: publication and compiler artifact differ")
	}
	return o, nil
}

func validateStageNativeReceipt(r stageNativeReceipt) error {
	if r.Version != 1 || r.Scope != "stage-library-only" || !r.Cleanup || r.Process.PID <= 0 || r.Process.Parent <= 0 || r.Process.Group <= 0 || r.Process.Start == 0 || !filepath.IsAbs(r.Process.Executable) || !validStageDigest(r.Process.ExecutableSHA256) {
		return errors.New("stage receipt: incomplete process or cleanup")
	}
	if len(r.Controls) != len(stageNativeControls) {
		return errors.New("stage receipt: incomplete controls")
	}
	for i, name := range stageNativeControls {
		if r.Controls[i] != name {
			return errors.New("stage receipt: unexpected control order")
		}
	}
	if len(r.Observations) < 7 || len(r.Observations) > 192 {
		return errors.New("stage receipt: incomplete observations")
	}
	seen := make(map[string]bool)
	held := false
	versions := make(map[uint64]uint64)
	sources := make(map[string][6]uint32)
	artifacts := make(map[uint64]stageNativeObservation)
	streams := make(map[string]uint64)
	for i, o := range r.Observations {
		if o.Index != i || !filepath.IsAbs(o.Artifact) || o.ArtifactGeneration == 0 || !validStageDigest(o.SourceSHA256) || !validStageDigest(o.ArtifactSHA256) || o.Context == 0 || o.Stream == 0 || o.Stream != o.ExpectedStream || o.Function == 0 || o.KernelVersion == 0 || !o.Completed || o.Geometry != o.ExpectedGeometry {
			return fmt.Errorf("stage receipt: invalid observation %d", i)
		}
		domain := "replay"
		if o.Arm == "held_submission" || o.Arm == "revert_output" {
			domain = "direct"
		}
		if o.StreamDomain != domain {
			return errors.New("stage receipt: incorrect stream domain")
		}
		if old, ok := streams[domain]; ok && old != o.Stream {
			return errors.New("stage receipt: stream changed within arm domain")
		}
		streams[domain] = o.Stream
		if o.Prepared {
			if o.Stage != (State{}) || o.Arm == "original" || o.Arm == "held_submission" || o.Arm == "revert_output" {
				return errors.New("stage receipt: unpublished artifact claims selected state")
			}
		} else if o.Stage.Kind != "stage" || o.Stage.Name != "stage" || o.Stage.Version == 0 || o.Stage.Generation == 0 || o.Stage.SourceDigest != o.SourceSHA256 {
			return errors.New("stage receipt: incomplete selected artifact owner")
		}
		if old, ok := sources[o.SourceSHA256]; ok && old != o.ExpectedGeometry {
			return errors.New("stage receipt: source bound to different geometry")
		}
		sources[o.SourceSHA256] = o.ExpectedGeometry
		if old, ok := artifacts[o.ArtifactGeneration]; ok && (old.Artifact != o.Artifact || old.ArtifactSHA256 != o.ArtifactSHA256 || old.SourceSHA256 != o.SourceSHA256 || old.ExpectedGeometry != o.ExpectedGeometry) {
			return errors.New("stage receipt: artifact generation identity changed")
		}
		artifacts[o.ArtifactGeneration] = o
		if !o.Prepared {
			if old, ok := versions[o.Stage.Version]; ok && old != o.ArtifactGeneration {
				return errors.New("stage receipt: selected version aliases another artifact")
			}
			versions[o.Stage.Version] = o.ArtifactGeneration
		}
		if o.Geometry != [6]uint32{4, 1, 1, 64, 1, 1} && o.Geometry != [6]uint32{2, 1, 1, 128, 1, 1} {
			return errors.New("stage receipt: unsupported observed geometry")
		}
		if o.Function != r.Ownership.Function || o.KernelVersion != r.Ownership.KernelVersion || o.Context != r.Observations[0].Context {
			return errors.New("stage receipt: changed kernel or context owner")
		}
		switch o.Arm {
		case "original", "revert_output":
			seen[o.Arm] = seen[o.Arm] || o.Geometry[3] == 64 && !o.Prepared
		case "geometry_change_pass", "wrong_tail_refusal":
			seen[o.Arm] = seen[o.Arm] || o.Geometry[3] == 128 && o.Prepared
		case "checker_rollback", "swap_while_pinned":
			seen[o.Arm] = seen[o.Arm] || o.Geometry[3] == 64 && o.Prepared
		case "held_submission":
			held = held || !o.Prepared && o.Stage.Version == r.Ownership.Version && o.Geometry[3] == 128
		default:
			return errors.New("stage receipt: unknown execution arm")
		}
	}
	for _, arm := range []string{"original", "geometry_change_pass", "wrong_tail_refusal", "checker_rollback", "swap_while_pinned", "revert_output"} {
		if !seen[arm] {
			return fmt.Errorf("stage receipt: missing executed %s", arm)
		}
	}
	if !held {
		return errors.New("stage receipt: held ownership has no matching completed submission")
	}
	o := r.Ownership
	if o.Version == 0 || o.KernelVersion == 0 || o.Function == 0 || !o.BeforePending || !o.BeforeLibraryPinned || !o.BeforeFunctionsPinned || !o.AfterComplete || !o.AfterLibraryReleased || !o.AfterFunctionsReleased || !o.AfterLeaseReleased {
		return errors.New("stage receipt: incomplete pending ownership")
	}
	return nil
}

func validStageDigest(s string) bool {
	if len(s) != 64 {
		return false
	}
	for _, c := range s {
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f') {
			return false
		}
	}
	return true
}

func validateStageNativeMappings(r stageNativeReceipt, proof nativereceipt.Receipt) error {
	if proof.Process != r.Process || !proof.DriverObserved || !proof.NVRTCObserved {
		return errors.New("stage receipt: runtime owner or mapped driver/compiler differs")
	}
	for _, o := range r.Observations {
		file, capture := false, false
		for _, f := range proof.Files {
			file = file || f.Path == o.Artifact && f.SHA256 == o.ArtifactSHA256
		}
		for _, s := range proof.Stages {
			if s.SourceSHA256 != o.SourceSHA256 || s.ArtifactSHA256 != o.ArtifactSHA256 {
				continue
			}
			for _, m := range s.Mappings {
				capture = capture || m.Path == o.Artifact
			}
		}
		if !file || !capture {
			return errors.New("stage receipt: actual stage mapping/body not retained")
		}
	}
	return nil
}

// Check token structure first: encoding/json accepts repeated object fields.
func stageJSONValue(d *json.Decoder, depth int) error {
	if depth > 24 {
		return errors.New("stage receipt: nesting exceeds limit")
	}
	t, err := d.Token()
	if err != nil {
		return err
	}
	delim, ok := t.(json.Delim)
	if !ok {
		return nil
	}
	switch delim {
	case '{':
		seen := make(map[string]bool)
		for d.More() {
			key, err := d.Token()
			if err != nil {
				return err
			}
			name, ok := key.(string)
			if !ok || seen[name] {
				return errors.New("stage receipt: duplicate object field")
			}
			seen[name] = true
			if err := stageJSONValue(d, depth+1); err != nil {
				return err
			}
		}
	case '[':
		for d.More() {
			if err := stageJSONValue(d, depth+1); err != nil {
				return err
			}
		}
	default:
		return errors.New("stage receipt: unexpected delimiter")
	}
	_, err = d.Token()
	return err
}

func readStageNativeReceipt(data []byte) (stageNativeReceipt, error) {
	var r stageNativeReceipt
	if len(data) == 0 || len(data) > 1<<20 {
		return r, errors.New("stage receipt: size refused")
	}
	d := json.NewDecoder(bytes.NewReader(data))
	if err := stageJSONValue(d, 0); err != nil {
		return r, err
	}
	if _, err := d.Token(); err != io.EOF {
		return r, errors.New("stage receipt: trailing data")
	}
	d = json.NewDecoder(bytes.NewReader(data))
	d.DisallowUnknownFields()
	if err := d.Decode(&r); err != nil {
		return r, err
	}
	return r, validateStageNativeReceipt(r)
}

func stageReceiptCPUFixture() stageNativeReceipt {
	h := strings.Repeat("a", 64)
	r := stageNativeReceipt{Version: 1, Process: nativereceipt.Process{PID: 2, Parent: 1, Group: 2, Start: 3, Executable: "/owned/prepared.test", ExecutableSHA256: h}, Controls: append([]string(nil), stageNativeControls...), Cleanup: true, Scope: "stage-library-only"}
	for i, arm := range []string{"original", "wrong_tail_refusal", "geometry_change_pass", "checker_rollback", "swap_while_pinned", "held_submission", "revert_output"} {
		geometry := [6]uint32{4, 1, 1, 64, 1, 1}
		if arm == "wrong_tail_refusal" || arm == "geometry_change_pass" || arm == "held_submission" {
			geometry = [6]uint32{2, 1, 1, 128, 1, 1}
		}
		prepared := arm != "original" && arm != "held_submission" && arm != "revert_output"
		artifact := []uint64{1, 2, 3, 4, 5, 3, 1}[i]
		source := uint64(1)
		if arm == "wrong_tail_refusal" {
			source = 2
		}
		if arm == "geometry_change_pass" || arm == "held_submission" {
			source = 3
		}
		sha := fmt.Sprintf("%064x", source)
		state := State{}
		if !prepared {
			version := uint64(1)
			if arm == "held_submission" {
				version = 2
			}
			state = State{Name: "stage", Kind: "stage", Generation: uint64(i + 1), Version: version, SourceDigest: sha}
		}
		domain, stream := "replay", uint64(5)
		if arm == "held_submission" || arm == "revert_output" {
			domain, stream = "direct", 8
		}
		r.Observations = append(r.Observations, stageNativeObservation{Index: i, Arm: arm, Stage: state, Prepared: prepared, ArtifactGeneration: artifact, SourceSHA256: sha, Artifact: fmt.Sprintf("/owned/stage-%d.so", artifact), ArtifactSHA256: fmt.Sprintf("%064x", artifact), Context: 4, Stream: stream, ExpectedStream: stream, StreamDomain: domain, Function: 6, KernelVersion: 7, Geometry: geometry, ExpectedGeometry: geometry, Completed: true})
	}
	r.Ownership = stageNativeOwnership{Version: 2, KernelVersion: 7, Function: 6, BeforePending: true, BeforeLibraryPinned: true, BeforeFunctionsPinned: true, AfterComplete: true, AfterLibraryReleased: true, AfterFunctionsReleased: true, AfterLeaseReleased: true}
	return r
}

func TestStageNativeReceiptControls(t *testing.T) {
	for _, tt := range []struct {
		name string
		edit func(*stageNativeReceipt)
	}{
		{"positive", func(*stageNativeReceipt) {}},
		{"wrong geometry", func(r *stageNativeReceipt) { r.Observations[1].Geometry[3] = 64 }},
		{"wrong expected geometry", func(r *stageNativeReceipt) { r.Observations[1].ExpectedGeometry = [6]uint32{4, 1, 1, 64, 1, 1} }},
		{"missing context", func(r *stageNativeReceipt) { r.Observations[0].Context = 0 }},
		{"wrong kernel owner", func(r *stageNativeReceipt) { r.Observations[0].KernelVersion++ }},
		{"wrong stream owner", func(r *stageNativeReceipt) { r.Observations[0].Stream++ }},
		{"wrong expected stream", func(r *stageNativeReceipt) { r.Observations[0].ExpectedStream++ }},
		{"wrong stream domain", func(r *stageNativeReceipt) { r.Observations[5].StreamDomain = "replay" }},
		{"wrong function owner", func(r *stageNativeReceipt) { r.Observations[0].Function++ }},
		{"unrelated held version", func(r *stageNativeReceipt) { r.Ownership.Version = 99999 }},
		{"invented execution", func(r *stageNativeReceipt) {
			o := r.Observations[0]
			o.Index = len(r.Observations)
			o.Arm = "fabricated_unexecuted_arm"
			r.Observations = append(r.Observations, o)
		}},
		{"missing wrong execution", func(r *stageNativeReceipt) { r.Observations[1].Arm = "geometry_change_pass" }},
		{"missing checker execution", func(r *stageNativeReceipt) { r.Observations[3].Arm = "swap_while_pinned" }},
		{"prepared selected alias", func(r *stageNativeReceipt) { r.Observations[1].Stage = r.Observations[0].Stage }},
		{"wrong body", func(r *stageNativeReceipt) { r.Observations[0].ArtifactSHA256 = "wrong" }},
		{"missing completion", func(r *stageNativeReceipt) { r.Observations[0].Completed = false }},
		{"released early", func(r *stageNativeReceipt) { r.Ownership.BeforeLibraryPinned = false }},
		{"retained functions", func(r *stageNativeReceipt) { r.Ownership.AfterFunctionsReleased = false }},
		{"retained lease", func(r *stageNativeReceipt) { r.Ownership.AfterLeaseReleased = false }},
		{"cleanup failed", func(r *stageNativeReceipt) { r.Cleanup = false }},
		{"missing control", func(r *stageNativeReceipt) { r.Controls = r.Controls[:6] }},
		{"unexecuted", func(r *stageNativeReceipt) { r.Observations = nil }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r := stageReceiptCPUFixture()
			tt.edit(&r)
			b, err := json.Marshal(r)
			if err != nil {
				t.Fatal(err)
			}
			_, err = readStageNativeReceipt(b)
			if (err == nil) != (tt.name == "positive") {
				t.Fatalf("receipt: %v", err)
			}
		})
	}
	for _, data := range []string{`{"Version":1,"Version":1}`, `{"Unknown":1}`, `{}`, `null`, `{} {}`, strings.Repeat("x", (1<<20)+1)} {
		if _, err := readStageNativeReceipt([]byte(data)); err == nil {
			t.Fatal("accepted malformed or incomplete receipt")
		}
	}
}

func TestStageNativeMappedBodyControls(t *testing.T) {
	r := stageReceiptCPUFixture()
	proof := nativereceipt.Receipt{Process: r.Process, DriverObserved: true, NVRTCObserved: true}
	for _, o := range r.Observations {
		proof.Files = append(proof.Files, nativereceipt.File{Path: o.Artifact, SHA256: o.ArtifactSHA256})
		proof.Stages = append(proof.Stages, nativereceipt.Stage{SourceSHA256: o.SourceSHA256, ArtifactSHA256: o.ArtifactSHA256, Mappings: []nativereceipt.Mapping{{Path: o.Artifact}}})
	}
	if err := validateStageNativeMappings(r, proof); err != nil {
		t.Fatal(err)
	}
	for _, which := range []string{"process", "file", "mapping", "driver", "compiler", "source"} {
		t.Run(which, func(t *testing.T) {
			b, _ := json.Marshal(proof)
			var p nativereceipt.Receipt
			if err := json.Unmarshal(b, &p); err != nil {
				t.Fatal(err)
			}
			switch which {
			case "process":
				p.Process.PID++
			case "file":
				p.Files = nil
			case "mapping":
				for i := range p.Stages {
					p.Stages[i].Mappings = nil
				}
			case "driver":
				p.DriverObserved = false
			case "compiler":
				p.NVRTCObserved = false
			case "source":
				for i := range p.Stages {
					p.Stages[i].SourceSHA256 = strings.Repeat("b", 64)
				}
			}
			if err := validateStageNativeMappings(r, p); err == nil {
				t.Fatal("accepted counterfeit mapping/body owner")
			}
		})
	}
}

func TestStageNativeBoundaryWithoutDevice(t *testing.T) {
	s, builder, original := newHostStageService(t)
	v := s.stages.entries[original.Name].current
	prepared, err := stageNativeArtifactIdentity(&stageVersion{artifact: v.artifact})
	if err != nil || !prepared.Prepared || prepared.Stage != (State{}) || prepared.SourceSHA256 != v.artifact.SourceDigest() || !validStageDigest(prepared.SourceSHA256) || prepared.ArtifactGeneration == 0 {
		t.Fatalf("actual unpublished artifact: %+v, %v", prepared, err)
	}
	selected, err := stageNativeArtifactIdentity(v)
	if err != nil || selected.Prepared || selected.Stage != original || selected.SourceSHA256 != prepared.SourceSHA256 {
		t.Fatalf("actual selected artifact: %+v, %v", selected, err)
	}
	if _, err := s.RegisterStage(t.Context(), "native", builder, stage.Source{Code: "#error must not compile"}, StageSpec{Buffers: []StageBuffer{{Name: "output", Bytes: 16}}, Functions: []StageFunction{{Name: "entry", Kernel: "writer"}}}, "no device"); err == nil || !strings.Contains(err.Error(), "actual NVRTCLoader") {
		t.Fatalf("device boundary: %v", err)
	}
	p, err := s.RunStage(t.Context(), original.Name, StageLaunch{})
	if err != nil || p == nil || p.complete || p.call.pin == nil {
		t.Fatalf("uncompleted host owner: %v", err)
	}
	if err := p.Wait(t.Context()); err != nil || !p.complete || p.call.pin != nil || p.cleanup.lease != nil {
		t.Fatalf("completed host owner: %v", err)
	}
	if _, err := s.SetCheck(original.Name, CheckConfig{Checker: func(context.Context, State) error { return nil }}, "explicit host check"); err != nil {
		t.Fatal(err)
	}
	if _, err := s.PushStage(t.Context(), original.Name, stage.Source{Code: hostStageWrong}, StageSpec{HostOnly: true}, "wrong actual C body"); err != nil {
		t.Fatal(err)
	}
	p, err = s.RunStage(t.Context(), original.Name, StageLaunch{})
	if err == nil || p == nil || p.complete || !strings.Contains(err.Error(), "17") {
		t.Fatalf("wrong C body: pending=%v err=%v", p != nil, err)
	}
	if err := p.Wait(t.Context()); err != nil || !p.complete {
		t.Fatalf("failed submission owner completion: %v", err)
	}
}

func FuzzStageNativeReceiptBoundary(f *testing.F) {
	b, _ := json.Marshal(stageReceiptCPUFixture())
	f.Add(b)
	f.Add([]byte(`{"Version":1,"Version":2}`))
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) > 1<<20 {
			t.Skip("fuzz input exceeds receipt bound")
		}
		r, err := readStageNativeReceipt(data)
		if err != nil {
			return
		}
		if err := validateStageNativeReceipt(r); err != nil {
			t.Fatal(err)
		}
		b, err := json.Marshal(r)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := readStageNativeReceipt(b); err != nil {
			t.Fatal(err)
		}
	})
}
