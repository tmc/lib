package httpapi

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudahotswap"
	"github.com/tmc/lib/nvidia/x/cudahotswap/stage"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

type fakeStageService struct {
	fakeService
	source   stage.Source
	schema   cudahotswap.StageSpec
	launch   cudahotswap.StageLaunch
	config   cudahotswap.FixtureConfig
	owner    *cudahotswap.PendingStage
	stageErr error
}

func (f *fakeStageService) RegisterStage(_ context.Context, _ string, _ *stage.Builder, source stage.Source, schema cudahotswap.StageSpec, _ string) (cudahotswap.State, error) {
	f.source, f.schema = source, schema
	return f.mark("RegisterStage"), f.stageErr
}
func (f *fakeStageService) PushStage(_ context.Context, _ string, source stage.Source, schema cudahotswap.StageSpec, _ string) (*cudahotswap.PushResult, error) {
	f.source, f.schema = source, schema
	f.mark("PushStage")
	outcome := cudahotswap.OutcomeOK
	if f.stageErr != nil {
		outcome = cudahotswap.OutcomeWrongAnswer
	}
	return &cudahotswap.PushResult{Report: cudahotswap.Report{Outcome: outcome, Generation: 9, ConfigRevision: 7}}, f.stageErr
}
func (f *fakeStageService) RevertStage(context.Context, string, string) (cudahotswap.State, error) {
	return f.mark("RevertStage"), f.stageErr
}
func (f *fakeStageService) RunStage(_ context.Context, _ string, launch cudahotswap.StageLaunch) (*cudahotswap.PendingStage, error) {
	f.launch = launch
	f.mark("RunStage")
	return f.owner, f.stageErr
}
func (f *fakeStageService) RecordStageFixture(_ context.Context, _ string, config cudahotswap.FixtureConfig) (*cudahotswap.Fixture, error) {
	f.config = config
	f.mark("RecordStageFixture")
	return nil, f.stageErr
}

func stageTestHandler(t *testing.T, f *fakeStageService) *Handler {
	t.Helper()
	h, err := New(f, Resources{Stages: map[string]StageBinding{"stage": {Builder: &stage.Builder{}, Schema: cudahotswap.StageSpec{HostOnly: true}, Flags: []string{"-O1"}}}, Streams: map[string]*cuda.Stream{"main": {}}})
	if err != nil {
		t.Fatal(err)
	}
	return h
}

func stageRequest(t *testing.T, h *Handler, endpoint string, q Request) (Response, int) {
	t.Helper()
	body, err := json.Marshal(q)
	if err != nil {
		t.Fatal(err)
	}
	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("POST", "/v1/"+endpoint, strings.NewReader(string(body))))
	var response Response
	if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	return response, w.Code
}

func testStageEndpoint(t *testing.T, method, endpoint string) {
	f := new(fakeStageService)
	h := stageTestHandler(t, f)
	q := Request{Name: "stage", Label: "caller"}
	if method == "RegisterStage" || method == "PushStage" {
		q.StageCode = "int stage(struct stage_ctx *c) { return 0; }"
	}
	if method == "RecordStageFixture" {
		// Dispatcher-only fake native context. No native owner is allocated or
		// submitted. Real native schema admission is checked separately by New.
		b := h.resources.Stages["stage"]
		b.Schema.HostOnly = false
		b.Stream = "main"
		h.resources.Stages["stage"] = b
		q.Stream = "main"
	}
	response, status := stageRequest(t, h, endpoint, q)
	if status != http.StatusOK || response.Failure != nil {
		t.Fatalf("status %d: %+v", status, response)
	}
	if !reflect.DeepEqual(f.calls, []string{"Err", method}) {
		t.Fatalf("calls %v", f.calls)
	}
	if method == "RunStage" && response.State != nil {
		t.Fatal("stage launch invented a state")
	}
	if method == "RecordStageFixture" && (f.config.Launch.Stream != h.resources.Streams["main"] || f.config.Launch.Grid != (cudahotswap.LaunchGeometry{}).Grid) {
		t.Fatal("fixture lost stream or invented geometry")
	}
	if method == "RegisterStage" || method == "PushStage" {
		if f.source.Code != q.StageCode || f.source.CUDA || !reflect.DeepEqual(f.source.Flags, []string{"-O1"}) || !f.schema.HostOnly {
			t.Fatal("changed application compilation contract")
		}
	}
}

func TestStageOptionalAndMalformed(t *testing.T) {
	kernel, err := New(new(fakeService), Resources{})
	if err != nil {
		t.Fatal(err)
	}
	if _, status := stageRequest(t, kernel, "run_stage", Request{Name: "stage", Label: "caller"}); status != 400 {
		t.Fatal("kernel-only application admitted a stage")
	}
	for _, tc := range []struct {
		name string
		q    Request
	}{
		{"unknown", Request{Name: "unknown", Label: "caller"}},
		{"no-label", Request{Name: "stage"}},
		{"geometry", Request{Name: "stage", Label: "caller", Geometry: &cudahotswap.LaunchGeometry{}}},
		{"compiler-flags", Request{Name: "stage", Label: "caller", Flags: []string{"-include=/arbitrary"}}},
		{"kernel-source", Request{Name: "stage", Label: "caller", Source: cudahotswap.Source{Code: "kernel"}}},
		{"stream", Request{Name: "stage", Label: "caller", Stream: "main"}},
		{"pointer-integer", Request{Name: "stage", Label: "caller", Args: []Argument{{Type: "u64", Value: json.RawMessage("4096")}}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := new(fakeStageService)
			h := stageTestHandler(t, f)
			if _, status := stageRequest(t, h, "run_stage", tc.q); status != 400 {
				t.Fatalf("status %d", status)
			}
			if !reflect.DeepEqual(f.calls, []string{"Err"}) {
				t.Fatalf("bad request reached service %v", f.calls)
			}
		})
	}
}

func TestStageRetainsPartialOwnerWithoutState(t *testing.T) {
	f := &fakeStageService{owner: &cudahotswap.PendingStage{}, stageErr: errors.New("submitted then failed")}
	h := stageTestHandler(t, f)
	response, status := stageRequest(t, h, "run_stage", Request{Name: "stage", Label: "caller"})
	if status != 200 || response.Handle == "" || response.Failure == nil || response.State != nil {
		t.Fatal(response, status)
	}
	if !reflect.DeepEqual(f.calls, []string{"Err", "RunStage"}) {
		t.Fatal("extra status call", f.calls)
	}
	got, status := stageRequest(t, h, "wait", Request{Handle: response.Handle, Label: "retry"})
	// This explicit fake's zero owner cannot complete; the handle must survive.
	if status != 200 || got.Handle != response.Handle || got.Failure == nil {
		t.Fatal("lost incomplete owner", got, status)
	}
	if _, ok := h.handles[response.Handle]; !ok {
		t.Fatal("dropped pending stage")
	}
}

func TestStageFailedCheckClassification(t *testing.T) {
	f := &fakeStageService{stageErr: errors.New("wrong output")}
	h := stageTestHandler(t, f)
	r, status := stageRequest(t, h, "push_stage", Request{Name: "stage", Label: "caller", StageCode: "stage body"})
	if status != 200 || r.Failure == nil || r.Failure.Outcome != cudahotswap.OutcomeWrongAnswer || r.Push == nil {
		t.Fatal(r, status)
	}
}

func TestStageBindingCopiesAndRefusesHostDeviceSchema(t *testing.T) {
	flags := []string{"-O1"}
	resources := Resources{Stages: map[string]StageBinding{"stage": {Builder: &stage.Builder{}, Schema: cudahotswap.StageSpec{HostOnly: true}, Flags: flags}}}
	h, err := New(new(fakeStageService), resources)
	if err != nil {
		t.Fatal(err)
	}
	flags[0] = "-O3"
	delete(resources.Stages, "stage")
	if h.resources.Stages["stage"].Flags[0] != "-O1" {
		t.Fatal("caller mutated retained binding")
	}
	for _, b := range []StageBinding{
		{Schema: cudahotswap.StageSpec{HostOnly: true}},
		{Builder: &stage.Builder{}, Schema: cudahotswap.StageSpec{HostOnly: true}, CUDA: true},
		{Builder: &stage.Builder{}, Schema: cudahotswap.StageSpec{HostOnly: true}, Stream: "main"},
		{Builder: &stage.Builder{}, Schema: cudahotswap.StageSpec{HostOnly: true, Buffers: []cudahotswap.StageBuffer{{Name: "x", Bytes: 4}}}},
		{Builder: &stage.Builder{}, Schema: cudahotswap.StageSpec{HostOnly: true, Functions: []cudahotswap.StageFunction{{Name: "f", Kernel: "kernel"}}}},
		{Builder: &stage.Builder{}, Schema: cudahotswap.StageSpec{}},
	} {
		if _, err := New(new(fakeStageService), Resources{Stages: map[string]StageBinding{"stage": b}}); err == nil {
			t.Fatal("admitted incomplete or contradictory stage")
		}
	}
}

func TestStageFixtureDispatcherNamedOrderAndNilArgs(t *testing.T) {
	f := new(fakeStageService)
	h := stageTestHandler(t, f)
	// Synthetic native resources exercise only the thin adapter, never New's
	// native admission or a device call. Actual device qualification is pending.
	h.resources.Stages["stage"] = StageBinding{Builder: &stage.Builder{}, Schema: cudahotswap.StageSpec{Buffers: []cudahotswap.StageBuffer{{Name: "input", Bytes: 16}, {Name: "output", Bytes: 32}}, Functions: []cudahotswap.StageFunction{{Name: "writer", Kernel: "kernel"}}}, BufferNames: []string{"x", "y"}, Stream: "main"}
	h.resources.Buffers = map[string]cudamem.Slice[byte]{"x": {}, "y": {}}
	q := Request{Name: "stage", Label: "caller", Stream: "main", Args: []Argument{{Type: "buffer", Name: "x"}, {Type: "buffer", Name: "y"}}, Buffers: []Buffer{{Name: "x", Arg: 0, Bytes: 16, Stable: true}, {Name: "y", Arg: 1, Bytes: 32}}, Outputs: []cudahotswap.FixtureOutput{{Arg: 1, Elem: "f32"}}}
	r, status := stageRequest(t, h, "record_stage_fixture", q)
	if status != 200 || r.Failure != nil || len(f.config.Args) != 2 || f.config.Args[0] != nil || f.config.Args[1] != nil || len(f.config.Buffers) != 2 || f.config.Buffers[0].Arg != 0 || f.config.Buffers[1].Arg != 1 || f.config.Buffers[0].Bytes != 16 || f.config.Buffers[1].Bytes != 32 || f.config.Launch.Stream != h.resources.Streams["main"] {
		t.Fatal("changed actual fixture contract", r, status, f.config)
	}
	for _, which := range []string{"reorder", "extent", "arg-index", "pointer", "geometry", "stream"} {
		t.Run(which, func(t *testing.T) {
			bad := q
			bad.Args = append([]Argument(nil), q.Args...)
			bad.Buffers = append([]Buffer(nil), q.Buffers...)
			switch which {
			case "reorder":
				bad.Args[0], bad.Args[1] = bad.Args[1], bad.Args[0]
			case "extent":
				bad.Buffers[0].Bytes = 15
			case "arg-index":
				bad.Buffers[0].Arg = 1
			case "pointer":
				bad.Args[0].Value = json.RawMessage("1234")
			case "geometry":
				bad.Geometry = &cudahotswap.LaunchGeometry{}
			case "stream":
				bad.Stream = "other"
			}
			before := len(f.calls)
			if _, status := stageRequest(t, h, "record_stage_fixture", bad); status != 400 {
				t.Fatalf("admitted contradictory fixture %d", status)
			}
			if !reflect.DeepEqual(f.calls[before:], []string{"Err"}) {
				t.Fatal("invalid fixture reached service", f.calls[before:])
			}
		})
	}
}

func TestStageRejectsClientBuilderAndLargeSource(t *testing.T) {
	f := new(fakeStageService)
	h := stageTestHandler(t, f)
	for _, body := range []string{`{"name":"stage","label":"caller","builder":"/usr/bin/cc","stage_code":"body"}`, `{"name":"stage","label":"caller","stage_code":"body","cuda":true}`} {
		w := httptest.NewRecorder()
		h.ServeHTTP(w, httptest.NewRequest("POST", "/v1/register_stage", strings.NewReader(body)))
		if w.Code != 400 {
			t.Fatal("accepted caller build path/mode", w.Code)
		}
	}
	if _, status := stageRequest(t, h, "register_stage", Request{Name: "stage", Label: "caller", StageCode: strings.Repeat("x", (1<<20)+1)}); status != 400 {
		t.Fatal("oversized stage source admitted")
	}
	for _, call := range f.calls {
		if call != "Err" {
			t.Fatal("bad compilation reached service", f.calls)
		}
	}
}

func FuzzHostStageBinding(f *testing.F) {
	f.Add("glue", "-O1", false)
	f.Add("", "-O1", false)
	f.Add("glue", "-O1", true)
	f.Fuzz(func(t *testing.T, name, flag string, cuda bool) {
		flags := []string{flag}
		bindings := map[string]StageBinding{name: {Builder: &stage.Builder{}, Schema: cudahotswap.StageSpec{HostOnly: true}, Flags: flags, CUDA: cuda}}
		copy, err := copyStageBindings(bindings, Resources{})
		if err != nil {
			return
		}
		if cuda || name == "" || len(name) > 128 || strings.ContainsAny(name, "\x00\r\n") || flag == "" || len(flag) > 16<<10 || strings.ContainsRune(flag, 0) {
			t.Fatal("invalid host binding admitted")
		}
		flags[0] = "caller changed flag"
		delete(bindings, name)
		if copy[name].Flags[0] != flag || !copy[name].Schema.HostOnly {
			t.Fatal("caller changed frozen binding")
		}
	})
}
