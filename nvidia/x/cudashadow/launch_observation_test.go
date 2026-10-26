package cudashadow

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudabpfdev"
	"github.com/tmc/lib/nvidia/x/cudakernel"
)

// The actual object passed to BOTH read interfaces also contains the active
// backend spy. Any accidental actuator type assertion on either read path
// reaches this spy, including reads that return an error.
type launchObservationSource struct {
	*ActiveBackendSpy
	source                   *cudabpfdev.ArtifactSource
	observations             []cudabpfdev.InvocationObservation
	describeError, readError error
	descriptions, reads      int
	afterDescribe, afterRead func()
}

func (s *launchObservationSource) DescribeLaunch() (cudabpfdev.Launch, error) {
	s.descriptions++
	if s.afterDescribe != nil {
		s.afterDescribe()
	}
	if s.describeError != nil {
		return cudabpfdev.Launch{}, s.describeError
	}
	return s.source.DescribeLaunch()
}
func (s *launchObservationSource) ReadObservation() (cudabpfdev.InvocationObservation, error) {
	s.reads++
	if s.afterRead != nil {
		s.afterRead()
	}
	if s.readError != nil {
		return cudabpfdev.InvocationObservation{}, s.readError
	}
	o := s.observations[min(s.reads-1, len(s.observations)-1)]
	o.Schema.Params = slices.Clone(o.Schema.Params)
	return o, nil
}

func launchObservationFixture(t *testing.T) (*Evaluator, *launchObservationSource, LaunchObservationOptions) {
	t.Helper()
	schema, err := cudakernel.NewSchema(cudakernel.Param{Name: "x", Type: "u32"})
	if err != nil {
		t.Fatal(err)
	}
	a, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte(".version 7.0\n.target sm_80\n.address_size 64\n.visible .entry kernel() { ret; }\n"), "kernel", schema, "sm_80")
	if err != nil {
		t.Fatal(err)
	}
	g := cudabpfdev.Geometry{Grid: [3]uint32{1, 1, 1}, Block: [3]uint32{32, 1, 1}}
	source, err := cudabpfdev.NewArtifactSource(a, g)
	if err != nil {
		t.Fatal(err)
	}
	// Start with a real capability's opaque owner linkage. Modified exported
	// fields are caller assertions in this fake, not a physical execution claim.
	owner := new(cudabpfdev.LoadedInjection)
	o, err := owner.ObservationReader().ReadObservation()
	if err != nil {
		t.Fatal(err)
	}
	o.Ready = true
	o.Original = cudabpfdev.KernelIdentity{Digest: a.Digest(), EntryPoint: a.EntryPoint(), Target: a.Target()}
	o.Selection = cudabpfdev.Selection{Kernel: o.Original, Reason: "injection disabled"}
	o.Geometry = g
	o.Schema = schema
	done := o
	done.Used = true
	done.Complete = true
	done.TimestampNano = 200
	e, err := NewEvaluator(4)
	if err != nil {
		t.Fatal(err)
	}
	e.SetClock(func() int64 { return 100 })
	s := &launchObservationSource{ActiveBackendSpy: NewActiveBackendSpy(), source: source, observations: []cudabpfdev.InvocationObservation{o, done}}
	options := LaunchObservationOptions{Source: "host-adapter-epoch-1", Sequence: 1, Policy: PolicyInfo{Name: "artifact-ptx", Version: "1", Domain: DomainPTXInstrument, Seed: 23, Parameters: map[string]string{"max_entries": "16"}}}
	t.Cleanup(func() {
		if err := s.AssertZeroMutations(); err != nil {
			t.Error(err)
		}
	})
	return e, s, options
}
func beginFixture(e *Evaluator, s *launchObservationSource, o LaunchObservationOptions) (*PendingLaunchObservation, error) {
	return beginLaunchObservation(context.Background(), e, s, s, o, func() int64 { return 100 })
}

func TestLaunchObservationRoute(t *testing.T) {
	e, s, o := launchObservationFixture(t)
	p, err := beginFixture(e, s, o)
	if err != nil {
		t.Fatal(err)
	}
	if p.SnapshotID() == "" || s.descriptions != 1 || s.reads != 1 {
		t.Fatal("begin did not read actual source once")
	}
	first := e.Snapshots()[0]
	if first.Status != StatusValid || first.Policy.Seed != 23 || first.Policy.Version != "1" || first.Policy.Parameters["max_entries"] != "16" || first.Input.PTX.ArtifactDigest != s.observations[0].Original.Digest {
		t.Fatalf("lost input or policy: %+v", first)
	}
	if first.Provenance.Kind != "artifact_description" || first.Provenance.Hook != HookDescribeLaunch || first.Provenance.DeviceID != -1 || first.Provenance.DriverVersion != "" || first.Provenance.KernelModule != "" || !first.Input.PTX.FunctionOffsetUnknown {
		t.Fatal("invented native facts")
	}
	if _, ok := first.Decision.Parameters["offset"]; ok {
		t.Fatal("unknown offset interpreted as measured zero")
	}
	if err := p.Complete(context.Background()); err != nil {
		t.Fatal(err)
	}
	snapshot := e.Snapshots()[0]
	out := snapshot.ObservedOutcome
	if out == nil || !out.LatencyUnavailable || out.LatencyNano != 0 || out.TimestampNano != 200 || out.Action != "original_invocation_completion_published" || out.Details["fallback_reason"] != "injection disabled" || out.Details["source"] != "producer_reported_same_owner" {
		t.Fatalf("completion=%+v", out)
	}
	if s.descriptions != 1 || s.reads != 2 {
		t.Fatal("completion replaced source or skipped read")
	}
	if err := p.Complete(context.Background()); err == nil || s.reads != 2 {
		t.Fatal("duplicate completion read or accepted")
	}
	var b bytes.Buffer
	if err := WriteHistory(&b, e.Snapshots()); err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(b.Bytes(), []byte(`"version":2`)) {
		t.Fatal("successor history missing")
	}
	history, err := ReadHistory(&b)
	if err != nil {
		t.Fatal(err)
	}
	pol, err := NewPTXShadowPolicy(o.Policy)
	if err != nil {
		t.Fatal(err)
	}
	replay, err := e.Replay(context.Background(), pol, history)
	if err != nil || len(replay) != 1 || replay[0].ObservedOutcome == nil || !replay[0].ObservedOutcome.LatencyUnavailable {
		t.Fatalf("observation replay=%+v error=%v", replay, err)
	}
	replay[0].Input.PTX.Schema.Params[0].Name = "replay-change"
	replay[0].Input.PTX.Geometry.Grid[0] = 9
	replay[0].ObservedOutcome.Details["source"] = "replay-change"
	if history[0].Input.PTX.Schema.Params[0].Name == "replay-change" || history[0].Input.PTX.Geometry.Grid[0] == 9 || history[0].ObservedOutcome.Details["source"] == "replay-change" {
		t.Fatal("replay aliases history")
	}
	history[0].Input.PTX.Schema.Params[0].Name = "detached"
	if e.Snapshots()[0].Input.PTX.Schema.Params[0].Name == "detached" {
		t.Fatal("history aliased evaluator")
	}
}

func TestLaunchObservationBeginRefusals(t *testing.T) {
	boom := errors.New("source failure")
	rows := []struct {
		name                string
		change              func(*launchObservationSource, *LaunchObservationOptions)
		descriptions, reads int
		want                error
	}{
		{"description failure", func(s *launchObservationSource, _ *LaunchObservationOptions) { s.describeError = boom }, 1, 0, boom},
		{"reader failure", func(s *launchObservationSource, _ *LaunchObservationOptions) { s.readError = boom }, 1, 1, boom},
		{"complete", func(s *launchObservationSource, _ *LaunchObservationOptions) { s.observations[0].Complete = true }, 1, 1, nil},
		{"candidate", func(s *launchObservationSource, _ *LaunchObservationOptions) {
			s.observations[0].Selection.Candidate = true
		}, 1, 1, nil},
		{"closed", func(s *launchObservationSource, _ *LaunchObservationOptions) { s.observations[0].Closed = true }, 1, 1, nil},
		{"used failed", func(s *launchObservationSource, _ *LaunchObservationOptions) { s.observations[0].Used = true }, 1, 1, nil},
		{"pending", func(s *launchObservationSource, _ *LaunchObservationOptions) { s.observations[0].Pending = true }, 1, 1, nil},
		{"not ready", func(s *launchObservationSource, _ *LaunchObservationOptions) { s.observations[0].Ready = false }, 1, 1, nil},
		{"original identity", func(s *launchObservationSource, _ *LaunchObservationOptions) {
			s.observations[0].Original.Digest = strings.Repeat("b", 64)
		}, 1, 1, nil},
		{"selected identity", func(s *launchObservationSource, _ *LaunchObservationOptions) {
			s.observations[0].Selection.Kernel.EntryPoint = "other"
		}, 1, 1, nil},
		{"grid", func(s *launchObservationSource, _ *LaunchObservationOptions) { s.observations[0].Geometry.Grid[0]++ }, 1, 1, nil},
		{"block", func(s *launchObservationSource, _ *LaunchObservationOptions) { s.observations[0].Geometry.Block[0]++ }, 1, 1, nil},
		{"shared bytes", func(s *launchObservationSource, _ *LaunchObservationOptions) {
			s.observations[0].Geometry.SharedBytes++
		}, 1, 1, nil},
		{"schema kind", func(s *launchObservationSource, _ *LaunchObservationOptions) {
			s.observations[0].Schema.Params = slices.Clone(s.observations[0].Schema.Params)
			s.observations[0].Schema.Params[0].Type = "s32"
		}, 1, 1, nil},
		{"schema width", func(s *launchObservationSource, _ *LaunchObservationOptions) {
			s.observations[0].Schema.Params = slices.Clone(s.observations[0].Schema.Params)
			s.observations[0].Schema.Params[0].ByteWidth = 8
		}, 1, 1, nil},
		{"schema alignment", func(s *launchObservationSource, _ *LaunchObservationOptions) {
			s.observations[0].Schema.Params = slices.Clone(s.observations[0].Schema.Params)
			s.observations[0].Schema.Params[0].Alignment = 8
		}, 1, 1, nil},
		{"schema offset", func(s *launchObservationSource, _ *LaunchObservationOptions) {
			s.observations[0].Schema.Params = slices.Clone(s.observations[0].Schema.Params)
			s.observations[0].Schema.Params[0].Offset = 4
		}, 1, 1, nil},
		{"schema total", func(s *launchObservationSource, _ *LaunchObservationOptions) {
			s.observations[0].Schema.TotalParamBytes += 8
		}, 1, 1, nil},
		{"reason overflow", func(s *launchObservationSource, _ *LaunchObservationOptions) {
			s.observations[0].Selection.Reason = strings.Repeat("x", 4097)
		}, 1, 1, nil},
		{"source overflow", func(_ *launchObservationSource, o *LaunchObservationOptions) { o.Source = strings.Repeat("x", 257) }, 1, 1, nil},
		{"policy version", func(_ *launchObservationSource, o *LaunchObservationOptions) { o.Policy.Version = "" }, 1, 1, nil},
		{"negative ttl", func(_ *launchObservationSource, o *LaunchObservationOptions) { o.TTL = -1 }, 1, 1, nil},
	}
	for _, row := range rows {
		t.Run(row.name, func(t *testing.T) {
			e, s, o := launchObservationFixture(t)
			row.change(s, &o)
			p, err := beginFixture(e, s, o)
			if err == nil || p != nil || row.want != nil && !errors.Is(err, row.want) || len(e.Snapshots()) != 0 || s.descriptions != row.descriptions || s.reads != row.reads {
				t.Fatalf("refusal token=%v error=%v snapshots=%d reads=%d/%d", p, err, len(e.Snapshots()), s.descriptions, s.reads)
			}
		})
	}
}

func TestLaunchObservationCompletionRefusals(t *testing.T) {
	boom := errors.New("completion read failure")
	rows := []struct {
		name   string
		change func(*launchObservationSource)
		want   error
	}{
		{"read failure", func(s *launchObservationSource) { s.readError = boom }, boom},
		{"missing", func(s *launchObservationSource) { s.observations[1].Complete = false }, nil},
		{"failed", func(s *launchObservationSource) { s.observations[1].Complete = false; s.observations[1].Used = true }, nil},
		{"pending", func(s *launchObservationSource) { s.observations[1].Pending = true }, nil},
		{"unused", func(s *launchObservationSource) { s.observations[1].Used = false }, nil},
		{"not ready", func(s *launchObservationSource) { s.observations[1].Ready = false }, nil},
		{"candidate", func(s *launchObservationSource) { s.observations[1].Selection.Candidate = true }, nil},
		{"identity", func(s *launchObservationSource) { s.observations[1].Original.EntryPoint = "other" }, nil},
		{"geometry", func(s *launchObservationSource) { s.observations[1].Geometry.Grid[0]++ }, nil},
		{"abi", func(s *launchObservationSource) { s.observations[1].Schema.TotalParamBytes += 8 }, nil},
		{"no time", func(s *launchObservationSource) { s.observations[1].TimestampNano = 0 }, nil},
		{"preceding time", func(s *launchObservationSource) { s.observations[1].TimestampNano = 99 }, nil},
		{"same image other owner", func(s *launchObservationSource) {
			other, _ := new(cudabpfdev.LoadedInjection).ObservationReader().ReadObservation()
			previous := s.observations[1]
			other.Original = previous.Original
			other.Selection = previous.Selection
			other.Geometry = previous.Geometry
			other.Schema = previous.Schema
			other.Ready = true
			other.Used = true
			other.Complete = true
			other.TimestampNano = 200
			s.observations[1] = other
		}, nil},
	}
	for _, row := range rows {
		t.Run(row.name, func(t *testing.T) {
			e, s, o := launchObservationFixture(t)
			p, err := beginFixture(e, s, o)
			if err != nil {
				t.Fatal(err)
			}
			row.change(s)
			err = p.Complete(context.Background())
			if err == nil || row.want != nil && !errors.Is(err, row.want) || e.Snapshots()[0].ObservedOutcome != nil || s.reads != 2 || s.descriptions != 1 {
				t.Fatalf("completion refusal error=%v reads=%d", err, s.reads)
			}
		})
	}
}

func TestLaunchObservationCancellation(t *testing.T) {
	for _, stage := range []string{"before descriptor", "after descriptor", "after owner", "before completion", "after completion read"} {
		t.Run(stage, func(t *testing.T) {
			e, s, o := launchObservationFixture(t)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			wantDescriptions, wantReads := 1, 1
			switch stage {
			case "before descriptor":
				cancel()
				wantDescriptions, wantReads = 0, 0
			case "after descriptor":
				s.afterDescribe = cancel
				wantReads = 0
			case "after owner":
				s.afterRead = cancel
			}
			p, err := beginLaunchObservation(ctx, e, s, s, o, func() int64 { return 100 })
			if strings.Contains(stage, "completion") {
				if err != nil {
					t.Fatal(err)
				}
				if stage == "before completion" {
					cancel()
				} else {
					s.afterRead = cancel
					wantReads = 2
				}
				err = p.Complete(ctx)
				if e.Snapshots()[0].ObservedOutcome != nil {
					t.Fatal("cancelled completion recorded")
				}
			} else if len(e.Snapshots()) != 0 {
				t.Fatal("cancelled begin recorded")
			}
			if !errors.Is(err, context.Canceled) || s.descriptions != wantDescriptions || s.reads != wantReads {
				t.Fatalf("cancel error=%v reads=%d/%d", err, s.descriptions, s.reads)
			}
		})
	}
}

func TestLaunchObservationOrderingAndEviction(t *testing.T) {
	for _, kind := range []string{"stale", "sequence", "timestamp", "epoch", "eviction"} {
		t.Run(kind, func(t *testing.T) {
			e, s, o := launchObservationFixture(t)
			if kind == "stale" {
				e.SetClock(func() int64 { return 200 })
				o.TTL = time.Nanosecond
			}
			p, err := beginFixture(e, s, o)
			if kind == "stale" {
				if err == nil || p != nil || e.Snapshots()[0].Status != StatusStale {
					t.Fatal("stale input accepted")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			s.reads = 0
			switch kind {
			case "sequence": // Same sequence refuses a second pending token.
			case "timestamp":
				o.Sequence++
				e.SetClock(func() int64 { return 100 })
			case "epoch":
				o.Sequence++
				o.Source = "different-epoch"
			case "eviction":
				e.capacity = 1
				o.Sequence++
			}
			clock := func() int64 { return 100 }
			if kind == "timestamp" {
				clock = func() int64 { return 99 }
			}
			next, nextErr := beginLaunchObservation(context.Background(), e, s, s, o, clock)
			if kind == "eviction" {
				if nextErr != nil || next == nil {
					t.Fatal(nextErr)
				}
				if err := p.Complete(context.Background()); err == nil {
					t.Fatal("evicted token completed")
				}
			} else if nextErr == nil || next != nil {
				t.Fatal("ordering or epoch change accepted")
			}
		})
	}
}

func TestLaunchObservationSchemaAndHistory(t *testing.T) {
	e, s, o := launchObservationFixture(t)
	p, err := beginFixture(e, s, o)
	if err != nil {
		t.Fatal(err)
	}
	if err := p.Complete(context.Background()); err != nil {
		t.Fatal(err)
	}
	original := e.Snapshots()[0]
	rows := []struct {
		name   string
		change func(*Snapshot)
	}{
		{"native device", func(s *Snapshot) { s.Input.Provenance.DeviceID = 0; s.Provenance = s.Input.Provenance }},
		{"native driver", func(s *Snapshot) { s.Input.Provenance.DriverVersion = "unknown"; s.Provenance = s.Input.Provenance }},
		{"native module", func(s *Snapshot) { s.Input.Provenance.KernelModule = "stock"; s.Provenance = s.Input.Provenance }},
		{"native hook", func(s *Snapshot) { s.Input.Provenance.Hook = HookPTXInstrument; s.Provenance = s.Input.Provenance }},
		{"offset value", func(s *Snapshot) { s.Input.PTX.FunctionOffset = 1 }},
		{"measured offset zero", func(s *Snapshot) { s.Input.PTX.FunctionOffsetUnknown = false }},
		{"digest", func(s *Snapshot) { s.Input.PTX.ArtifactDigest = "" }},
		{"measured latency zero", func(s *Snapshot) { s.ObservedOutcome.LatencyUnavailable = false }},
		{"latency value", func(s *Snapshot) { s.ObservedOutcome.LatencyNano = 1 }},
		{"geometry", func(s *Snapshot) { s.Input.PTX.Geometry.Grid[0] = 0 }},
		{"schema bound", func(s *Snapshot) { s.Input.PTX.Schema.TotalParamBytes = 1 << 21 }},
	}
	for _, row := range rows {
		t.Run(row.name, func(t *testing.T) {
			bad := cloneSnapshot(original)
			row.change(&bad)
			if err := bad.Validate(); err == nil {
				t.Fatal("contradictory description accepted")
			}
		})
	}
	var b bytes.Buffer
	if err := WriteHistory(&b, []Snapshot{original}); err != nil {
		t.Fatal(err)
	}
	disguised := bytes.Replace(b.Bytes(), []byte(`"version":2`), []byte(`"version":1`), 1)
	if _, err := ReadHistory(bytes.NewReader(disguised)); err == nil {
		t.Fatal("successor disguised as legacy")
	}
	// Presence is forbidden even when its value equals the legacy zero value.
	var wire map[string]any
	if err := json.Unmarshal(b.Bytes(), &wire); err != nil {
		t.Fatal(err)
	}
	legacy := `{"version":1,"snapshots":[]}`
	if _, err := ReadHistory(strings.NewReader(legacy)); err != nil {
		t.Fatal(err)
	}
	for _, bad := range []string{strings.Replace(legacy, `"version":1`, `"version":1,"version":1`, 1), legacy + `{}`, `{"version":3,"snapshots":[]}`} {
		if _, err := ReadHistory(strings.NewReader(bad)); err == nil {
			t.Fatal("invalid history accepted")
		}
	}
	if err := WriteHistory(&bytes.Buffer{}, make([]Snapshot, maxHistorySnapshots+1)); err == nil {
		t.Fatal("history overflow accepted")
	}
}

func TestLaunchObservationReadOnlySource(t *testing.T) {
	data, err := os.ReadFile("launch_observation.go")
	if err != nil {
		t.Fatal(err)
	}
	file, err := parser.ParseFile(token.NewFileSet(), "launch_observation.go", data, 0)
	if err != nil {
		t.Fatal(err)
	}
	forbidden := map[string]bool{"Load": true, "Launch": true, "Close": true, "PrepareInjection": true, "MutatePTX": true, "EvictChunk": true, "PrefetchPages": true, "SetTimeslice": true}
	ast.Inspect(file, func(n ast.Node) bool {
		if call, ok := n.(*ast.CallExpr); ok {
			if selector, ok := call.Fun.(*ast.SelectorExpr); ok && forbidden[selector.Sel.Name] {
				t.Errorf("adapter calls active method %s", selector.Sel.Name)
			}
		}
		return true
	})
	for _, imp := range file.Imports {
		if strings.Contains(imp.Path.Value, "/driver") || strings.Contains(imp.Path.Value, "/cuda\"") {
			t.Fatal("adapter imports driver")
		}
	}
}

func ExamplePendingLaunchObservation_SnapshotID() {
	var pending *PendingLaunchObservation
	fmt.Printf("%q\n", pending.SnapshotID())
	// Output: ""
}

func TestLaunchObservationPublicRoute(t *testing.T) {
	e, s, o := launchObservationFixture(t)
	e.SetClock(func() int64 { return time.Now().UnixNano() })
	s.observations[1].TimestampNano = time.Now().Add(time.Second).UnixNano()
	var published int64
	s.afterDescribe = func() { published = time.Now().UnixNano() }
	p, err := BeginLaunchObservation(context.Background(), e, s, s, o)
	if err != nil {
		t.Fatal(err)
	}
	if err := p.Complete(context.Background()); err != nil {
		t.Fatal(err)
	}
	if e.Snapshots()[0].Provenance.TimestampNano < published {
		t.Fatal("timestamp precedes descriptor read publication")
	}
	if e.Snapshots()[0].ObservedOutcome == nil || s.descriptions != 1 || s.reads != 2 {
		t.Fatal("public route incomplete")
	}
}

func TestLaunchObservationArtifactBounds(t *testing.T) {
	for _, kind := range []string{"cubin", "image overflow", "entry overflow", "schema metadata", "schema count"} {
		t.Run(kind, func(t *testing.T) {
			e, s, o := launchObservationFixture(t)
			launch, err := s.source.DescribeLaunch()
			if err != nil {
				t.Fatal(err)
			}
			a := launch.Artifact
			raw, entry, schema, k := a.Bytes(), a.EntryPoint(), a.Schema(), a.Kind()
			switch kind {
			case "cubin":
				k = cudakernel.KindCUBIN
				raw = make([]byte, 64)
				copy(raw, "\x7fELF")
				raw[4], raw[5], raw[6] = 2, 1, 1
				binary.LittleEndian.PutUint16(raw[16:18], 2)
				binary.LittleEndian.PutUint16(raw[18:20], 190)
				binary.LittleEndian.PutUint32(raw[20:24], 1)
				binary.LittleEndian.PutUint16(raw[52:54], 64)
			case "image overflow":
				raw = bytes.Repeat([]byte{'x'}, 16<<20+1)
			case "entry overflow":
				entry = strings.Repeat("x", 4097)
			case "schema metadata":
				schema.Params[0].Name = strings.Repeat("x", 257)
			case "schema count":
				params := make([]cudakernel.Param, 129)
				for i := range params {
					params[i] = cudakernel.Param{Name: fmt.Sprintf("p%d", i), Type: "u32"}
				}
				schema, err = cudakernel.NewSchema(params...)
				if err != nil {
					t.Fatal(err)
				}
			}
			artifact, err := cudakernel.NewArtifact(k, raw, entry, schema, a.Target())
			if err != nil {
				t.Fatal(err)
			}
			identity := cudabpfdev.KernelIdentity{Digest: artifact.Digest(), EntryPoint: artifact.EntryPoint(), Target: artifact.Target()}
			for i := range s.observations {
				s.observations[i].Original = identity
				s.observations[i].Selection.Kernel = identity
				s.observations[i].Schema = artifact.Schema()
			}
			s.source, err = cudabpfdev.NewArtifactSource(artifact, launch.Geometry)
			if err != nil {
				t.Fatal(err)
			}
			p, err := beginFixture(e, s, o)
			if err == nil || p != nil || len(e.Snapshots()) != 0 {
				t.Fatal("artifact bound accepted")
			}
		})
	}
}

func TestLaunchObservationLegacyPresence(t *testing.T) {
	// A legitimate old outcome detail named source remains legal. New schema
	// field presence is rejected in v1 even when its value is false or empty.
	e, s, o := launchObservationFixture(t)
	p, err := beginFixture(e, s, o)
	if err != nil {
		t.Fatal(err)
	}
	if err := p.Complete(context.Background()); err != nil {
		t.Fatal(err)
	}
	legacy := cloneSnapshot(e.Snapshots()[0])
	legacy.Provenance = Provenance{DeviceID: 0, DriverVersion: "test", KernelModule: "test", Hook: HookPTXInstrument, Sequence: 1, TimestampNano: 100}
	legacy.Input.Provenance = legacy.Provenance
	legacy.Input.PTX.FunctionOffsetUnknown = false
	legacy.Input.PTX.ArtifactDigest = ""
	legacy.Input.PTX.Geometry = nil
	legacy.Input.PTX.Schema = nil
	legacy.ObservedOutcome.LatencyUnavailable = false
	var b bytes.Buffer
	if err := WriteHistory(&b, []Snapshot{legacy}); err != nil {
		t.Fatal(err)
	}
	data := append([]byte(nil), b.Bytes()...)
	if !bytes.Contains(data, []byte(`"version":1`)) {
		t.Fatal("legacy writer changed version")
	}
	successorLegacy := bytes.Replace(data, []byte(`"version":1`), []byte(`"version":2`), 1)
	if history, err := ReadHistory(bytes.NewReader(successorLegacy)); err != nil || len(history) != 1 || history[0].Input.Provenance.Kind != "" || history[0].ObservedOutcome.LatencyUnavailable {
		t.Fatal("version 2 rejected legacy content", err)
	}

	if _, err := ReadHistory(bytes.NewReader(data)); err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{`"function_offset_unknown":false,`, `"artifact_digest":"",`, `"geometry":null,`, `"schema":null,`} {
		bad := bytes.Replace(data, []byte(`"ptx":{`), []byte(`"ptx":{`+field), 1)
		if _, err := ReadHistory(bytes.NewReader(bad)); err == nil {
			t.Fatalf("accepted successor presence %s", field)
		}
	}
	bad := bytes.Replace(data, []byte(`"observed_outcome":{`), []byte(`"observed_outcome":{"latency_unavailable":false,`), 1)
	if _, err := ReadHistory(bytes.NewReader(bad)); err == nil {
		t.Fatal("accepted successor latency presence")
	}
}

func TestLaunchObservationLegacyCaseAliases(t *testing.T) {
	e, s, o := launchObservationFixture(t)
	p, err := beginFixture(e, s, o)
	if err != nil {
		t.Fatal(err)
	}
	if err := p.Complete(context.Background()); err != nil {
		t.Fatal(err)
	}
	legacy := cloneSnapshot(e.Snapshots()[0])
	legacy.Provenance = Provenance{DeviceID: 0, DriverVersion: "test", KernelModule: "test", Hook: HookPTXInstrument, Sequence: 1, TimestampNano: 100}
	legacy.Input.Provenance = legacy.Provenance
	legacy.Input.PTX.FunctionOffsetUnknown = false
	legacy.Input.PTX.ArtifactDigest = ""
	legacy.Input.PTX.Geometry = nil
	legacy.Input.PTX.Schema = nil
	legacy.ObservedOutcome.LatencyUnavailable = false
	var b bytes.Buffer
	if err := WriteHistory(&b, []Snapshot{legacy}); err != nil {
		t.Fatal(err)
	}
	baseline := b.String()
	if _, err := ReadHistory(strings.NewReader(baseline)); err != nil {
		t.Fatal(err)
	}
	rows := []struct {
		name, anchor, insert string
		aliases              [][2]string
	}{
		{"offset alias", `"ptx":{`, `"ptx":{"Function_Offset_Unknown":false,`, nil},
		{"schema fold", `"ptx":{`, `"ptx":{"ſchema":null,`, nil},
		{"source fold", `"provenance":{`, `"provenance":{"ſource":"",`, nil},
		{"kind alias", `"provenance":{`, `"provenance":{"Kind":"",`, nil},
		{"ancestor snapshots", `"ptx":{`, `"ptx":{"function_offset_unknown":false,`, [][2]string{{`"snapshots"`, `"Snapshots"`}}},
		{"ancestor input", `"ptx":{`, `"ptx":{"function_offset_unknown":false,`, [][2]string{{`"input"`, `"Input"`}}},
		{"ancestor ptx", `"ptx":{`, `"ptx":{"function_offset_unknown":false,`, [][2]string{{`"ptx"`, `"PTX"`}}},
		{"ancestor provenance", `"provenance":{`, `"provenance":{"kind":"",`, [][2]string{{`"provenance"`, `"Provenance"`}}},
		{"latency alias", `"observed_outcome":{`, `"observed_outcome":{"Latency_Unavailable":false,`, nil},
		{"ancestor outcome fold", `"observed_outcome":{`, `"observed_outcome":{"latency_unavailable":false,`, [][2]string{{`"observed_outcome"`, `"obſerved_outcome"`}}},
	}
	for _, row := range rows {
		t.Run(row.name, func(t *testing.T) {
			bad := strings.Replace(baseline, row.anchor, row.insert, 1)
			for _, alias := range row.aliases {
				bad = strings.ReplaceAll(bad, alias[0], alias[1])
			}
			if _, err := ReadHistory(strings.NewReader(bad)); err == nil {
				t.Fatal("accepted single successor presence through alias")
			}
		})
	}
}

func TestLaunchObservationPrecloneBounds(t *testing.T) {
	for _, kind := range []string{"source", "kind", "digest", "schema count", "schema bytes", "param name", "param type"} {
		t.Run(kind, func(t *testing.T) {
			e, s, o := launchObservationFixture(t)
			if _, err := beginFixture(e, s, o); err != nil {
				t.Fatal(err)
			}
			in := e.Snapshots()[0].Input
			in.Provenance.Sequence++
			switch kind {
			case "source":
				in.Provenance.Source = strings.Repeat("x", 257)
			case "kind":
				in.Provenance.Kind = strings.Repeat("x", 257)
			case "digest":
				in.PTX.ArtifactDigest = strings.Repeat("x", 65)
			case "schema count":
				in.PTX.Schema.Params = make([]cudakernel.Param, 129)
			case "schema bytes":
				in.PTX.Schema.TotalParamBytes = 1 << 21
			case "param name":
				in.PTX.Schema.Params[0].Name = strings.Repeat("x", 257)
			case "param type":
				in.PTX.Schema.Params[0].Type = strings.Repeat("x", 65)
			}
			pol, err := NewPTXShadowPolicy(o.Policy)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := e.Evaluate(context.Background(), pol, in); err == nil || e.TotalEvaluated() != 1 {
				t.Fatal("oversized successor stored before bounds check")
			}
		})
	}
}

func TestLaunchObservationRetryAndClosedReceipt(t *testing.T) {
	e, s, o := launchObservationFixture(t)
	p, err := beginFixture(e, s, o)
	if err != nil {
		t.Fatal(err)
	}
	sentinel := errors.New("temporary reader failure")
	s.readError = sentinel
	if err := p.Complete(context.Background()); !errors.Is(err, sentinel) {
		t.Fatal(err)
	}
	s.readError = nil
	s.observations[1].Closed = true
	if err := p.Complete(context.Background()); err != nil {
		t.Fatal(err)
	}
	if e.Snapshots()[0].ObservedOutcome.Details["closed"] != "true" {
		t.Fatal("closed receipt lost")
	}
}

func TestLaunchObservationCancelledLockWait(t *testing.T) {
	e, s, o := launchObservationFixture(t)
	p, err := beginFixture(e, s, o)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	p.mu.Lock()
	started := make(chan struct{})
	finished := make(chan error, 1)
	go func() { close(started); finished <- p.Complete(ctx) }()
	<-started
	cancel()
	p.mu.Unlock()
	if err := <-finished; !errors.Is(err, context.Canceled) || s.reads != 1 || e.Snapshots()[0].ObservedOutcome != nil {
		t.Fatal("cancelled waiter read or recorded completion", err)
	}
}

// ActiveVoid is a test-only void actuator. The semantic control inserts a call
// through the same interface object received by the production read path.
func (s *launchObservationSource) ActiveVoid() { _ = s.MutatePTX("void-actuator", nil) }
