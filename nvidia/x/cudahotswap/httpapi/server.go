package httpapi

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"reflect"
	"strconv"
	"strings"
	"sync"

	"github.com/tmc/lib/nvidia/x/cudahotswap"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
)

var serviceMethods = map[string]string{
	"Err": "err", "Register": "register", "Push": "push", "List": "list", "Status": "status", "Versions": "versions",
	"Select": "select", "Revert": "revert", "SetCompileFlags": "set_compile_flags", "SetGeometry": "set_geometry",
	"SetCheck": "set_check", "SetWatchPaths": "set_watch_paths", "WatchPaths": "watch_paths", "Watch": "watch",
	"Subscribe": "events", "Pin": "pin", "Launch": "launch", "RecordFixture": "record_fixture", "RegisterGraph": "register_graph",
	"LaunchGraph": "launch_graph", "Close": "close",
	"RegisterStage": "register_stage", "PushStage": "push_stage", "RevertStage": "revert_stage",
	"RunStage": "run_stage", "RecordStageFixture": "record_stage_fixture",
}

// Endpoints returns the HTTP endpoint for each exported Service method.
// Additional endpoints release_pin, wait and close_fixture release owners.
func Endpoints() map[string]string { return cloneMap(serviceMethods) }

type retained struct {
	wait    func(context.Context) error
	release func(context.Context) error
	fixture *cudahotswap.Fixture
	state   *cudahotswap.State
	states  []cudahotswap.State
}

// Handler owns transport lifetime handles, but not the application's native
// resources. Close must succeed before the application destroys those resources.
// It bounds handles to128 and streams to32. A disconnected caller does not
// release submitted work; opaque handles remain retained for retry or Close.
type Handler struct {
	service   Service
	resources Resources
	ctx       context.Context
	cancel    context.CancelFunc
	closeGate chan struct{}
	idle      chan struct{}
	active    int
	mu        sync.Mutex
	handles   map[string]retained
	next      uint64
	streams   int
	closed    bool
}

// New constructs an HTTP handler with application-owned named resources.
func New(service Service, resources Resources) (*Handler, error) {
	if service == nil || reflect.ValueOf(service).Kind() == reflect.Pointer && reflect.ValueOf(service).IsNil() {
		return nil, errors.New("httpapi: nil service")
	}
	if len(resources.Checkers)+len(resources.Buffers)+len(resources.Streams)+len(resources.Graphs)+len(resources.Arguments)+len(resources.Stages) > 4096 {
		return nil, errors.New("httpapi: resource limit exceeded")
	}
	copy := Resources{Arguments: make(map[string][]string), Checkers: make(map[string]cudahotswap.Checker), Buffers: cloneMap(resources.Buffers), Streams: cloneMap(resources.Streams), Graphs: cloneMap(resources.Graphs)}
	for name, kinds := range resources.Arguments {
		if !validName(name) || len(kinds) > 256 {
			return nil, errors.New("httpapi: invalid argument binding")
		}
		for _, kind := range kinds {
			switch kind {
			case "buffer", "u32", "i32", "u64", "i64", "f32", "f64":
			default:
				return nil, errors.New("httpapi: invalid argument kind")
			}
		}
		copy.Arguments[name] = append([]string(nil), kinds...)
	}
	for name, checker := range resources.Checkers {
		if !validName(name) || checker == nil {
			return nil, errors.New("httpapi: invalid checker")
		}
		copy.Checkers[name] = checker
	}
	for name, b := range copy.Buffers {
		if !validName(name) || b.Len() <= 0 || b.Ptr() == 0 {
			return nil, errors.New("httpapi: invalid buffer")
		}
	}
	for name, s := range copy.Streams {
		if !validName(name) || s == nil {
			return nil, errors.New("httpapi: invalid stream")
		}
	}
	for name, g := range copy.Graphs {
		if !validName(name) || g.Exec == nil || g.Node == nil || g.Stream == nil || len(g.Args) > 256 {
			return nil, errors.New("httpapi: invalid graph")
		}
		g.Args = append([]any(nil), g.Args...)
		copy.Graphs[name] = g
	}
	var err error
	copy.Stages, err = copyStageBindings(resources.Stages, copy)
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithCancel(context.Background())
	idle := make(chan struct{})
	close(idle)
	return &Handler{service: service, resources: copy, ctx: ctx, cancel: cancel, handles: make(map[string]retained), closeGate: make(chan struct{}, 1), idle: idle}, nil
}

func cloneMap[V any](m map[string]V) map[string]V {
	c := make(map[string]V, len(m))
	for k, v := range m {
		c[k] = v
	}
	return c
}
func validName(s string) bool { return s != "" && len(s) <= 128 && !strings.ContainsAny(s, "\x00\r\n") }

func failure(err error) *Failure {
	if err == nil {
		return nil
	}
	out, _ := cudahotswap.Classify(err)
	return &Failure{Message: err.Error(), Outcome: out, Poisoned: errors.Is(err, cudahotswap.ErrPoisoned)}
}

func writeResponse(w http.ResponseWriter, status int, response Response) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(response)
}

func malformed(w http.ResponseWriter, err error) {
	writeResponse(w, http.StatusBadRequest, Response{Failure: failure(err)})
}

func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	endpoint := strings.TrimPrefix(r.URL.Path, "/v1/")
	known := endpoint == "release_pin" || endpoint == "wait" || endpoint == "close_fixture"
	for _, method := range serviceMethods {
		known = known || endpoint == method
	}
	if !strings.HasPrefix(r.URL.Path, "/v1/") || !known {
		http.NotFound(w, r)
		return
	}
	if endpoint == "events" {
		if r.Method != http.MethodGet {
			malformed(w, errors.New("events requires GET"))
			return
		}
	} else if r.Method != http.MethodPost {
		malformed(w, errors.New("endpoint requires POST"))
		return
	}
	ctx, cancel := context.WithCancel(r.Context())
	stop := context.AfterFunc(h.ctx, cancel)
	defer stop()
	defer cancel()
	if endpoint == "close" {
		var request Request
		if err := decodeRequest(w, r, &request); err != nil {
			malformed(w, err)
			return
		}
		if !validName(request.Label) {
			malformed(w, errors.New("caller label required"))
			return
		}
		err := h.Close(r.Context())
		writeResponse(w, http.StatusOK, Response{RequestLabel: request.Label, Failure: failure(err)})
		return
	}
	if !h.beginRequest() {
		writeResponse(w, http.StatusOK, Response{Failure: failure(errors.New("httpapi: handler is closed"))})
		return
	}
	defer h.endRequest()
	cleanup := endpoint == "release_pin" || endpoint == "wait" || endpoint == "close_fixture"
	if err := h.service.Err(); err != nil && !cleanup {
		status := http.StatusOK
		if errors.Is(err, cudahotswap.ErrPoisoned) {
			status = http.StatusServiceUnavailable
		}
		writeResponse(w, status, Response{Failure: failure(err)})
		return
	}
	if endpoint == "events" {
		h.events(ctx, w, r)
		return
	}
	var request Request
	if err := decodeRequest(w, r, &request); err != nil {
		malformed(w, err)
		return
	}
	if endpoint == "watch" {
		h.watch(ctx, w, request)
		return
	}
	response, err := h.call(ctx, endpoint, request)
	response.RequestLabel = request.Label
	if errors.As(err, new(*requestError)) {
		malformed(w, err)
		return
	}
	response.Failure = failure(err)
	if response.Push != nil && err != nil {
		response.Failure.Outcome = response.Push.Report.Outcome
	}
	status := http.StatusOK
	if errors.Is(err, cudahotswap.ErrPoisoned) {
		status = http.StatusServiceUnavailable
	}
	writeResponse(w, status, response)
}

type requestError struct{ error }

func bad(s string) error { return &requestError{errors.New(s)} }

func decodeRequest(w http.ResponseWriter, r *http.Request, request *Request) error {
	r.Body = http.MaxBytesReader(w, r.Body, (8<<20)+(64<<10))
	body, err := io.ReadAll(r.Body)
	if err != nil {
		return fmt.Errorf("read request: %w", err)
	}
	if err := validateJSON(body); err != nil {
		return err
	}
	d := json.NewDecoder(bytes.NewReader(body))
	d.DisallowUnknownFields()
	if err := d.Decode(request); err != nil {
		return fmt.Errorf("decode request: %w", err)
	}
	if d.Decode(new(any)) != io.EOF {
		return errors.New("decode request: trailing JSON")
	}
	if len(request.Args) > 256 || len(request.Buffers) > 256 || len(request.Outputs) > 256 || len(request.Flags) > 64 || len(request.Label) > 256 || len(request.Source.Code) > 8<<20 || len(request.StageCode) > 1<<20 {
		return errors.New("decode request: field limit exceeded")
	}
	return nil
}

func (h *Handler) reserve() (string, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if len(h.handles) >= 128 || h.next == math.MaxUint64 {
		return "", bad("lifetime handle limit exceeded")
	}
	h.next++
	id := strconv.FormatUint(h.next, 10)
	h.handles[id] = retained{}
	return id, nil
}
func (h *Handler) adopt(id string, owner retained) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if owner.wait == nil && owner.release == nil && owner.fixture == nil {
		delete(h.handles, id)
	} else {
		h.handles[id] = owner
	}
}

func (h *Handler) call(ctx context.Context, method string, q Request) (response Response, err error) {
	if isStageMethod(method) {
		return h.callStage(ctx, method, q)
	}
	mutating := method == "register" || method == "push" || method == "select" || method == "revert" || method == "set_compile_flags" || method == "set_geometry" || method == "set_check" || method == "set_watch_paths" || method == "register_graph" || method == "pin" || method == "launch" || method == "record_fixture" || method == "launch_graph" || method == "wait" || method == "release_pin" || method == "close_fixture"
	if mutating && !validName(q.Label) {
		return response, bad("caller label required")
	}
	if method != "list" && method != "err" && method != "release_pin" && method != "wait" && method != "close_fixture" && method != "launch_graph" && !validName(q.Name) {
		return response, bad("kernel name required")
	}
	state := func(s cudahotswap.State, e error) (Response, error) { response.State = &s; return response, e }
	switch method {
	case "err":
		return response, h.service.Err()
	case "list":
		response.States = h.service.List()
		return response, nil
	case "status":
		return state(h.service.Status(q.Name))
	case "versions":
		response.Versions, err = h.service.Versions(q.Name)
		return response, err
	case "register":
		return state(h.service.Register(ctx, q.Name, q.Source, q.Label))
	case "push":
		response.Push, err = h.service.Push(ctx, q.Name, q.Source, q.Label)
		return response, err
	case "select":
		return state(h.service.Select(ctx, q.Name, q.Version, q.Label))
	case "revert":
		return state(h.service.Revert(ctx, q.Name, q.Label))
	case "set_compile_flags":
		return state(h.service.SetCompileFlags(q.Name, q.Flags, q.Label))
	case "set_geometry":
		return state(h.service.SetGeometry(q.Name, q.Geometry, q.Label))
	case "set_watch_paths":
		return state(h.service.SetWatchPaths(q.Name, q.Paths, q.Label))
	case "watch_paths":
		response.Paths, err = h.service.WatchPaths(q.Name)
		return response, err
	case "set_check":
		config := cudahotswap.CheckConfig{Policy: q.Policy, KeepOnFailure: q.KeepOnFailure}
		if q.Checker != "" {
			var ok bool
			config.Checker, ok = h.resources.Checkers[q.Checker]
			if !ok {
				return response, bad("unknown checker")
			}
		}
		if q.Fixture != "" {
			h.mu.Lock()
			owner, ok := h.handles[q.Fixture]
			h.mu.Unlock()
			if !ok || owner.fixture == nil {
				return response, bad("unknown fixture")
			}
			config.Fixture = owner.fixture
		}
		return state(h.service.SetCheck(q.Name, config, q.Label))
	case "register_graph", "launch_graph":
		g, ok := h.resources.Graphs[q.Graph]
		if !ok {
			return response, bad("unknown graph")
		}
		if method == "register_graph" {
			return state(h.service.RegisterGraph(ctx, q.Name, g.Exec, g.Node, g.Stream, g.Grid, g.Args...))
		}
		id, e := h.reserve()
		if e != nil {
			return response, e
		}
		p, e := h.service.LaunchGraph(ctx, g.Exec)
		owner := retained{}
		if p != nil {
			owner.wait = p.Wait
			owner.states = p.States()
			response.States = owner.states
			response.Handle = id
		}
		h.adopt(id, owner)
		return response, e
	case "pin":
		id, e := h.reserve()
		if e != nil {
			return response, e
		}
		p, e := h.service.Pin(q.Name)
		owner := retained{}
		if p != nil {
			owner.release = p.Release
			s := p.State()
			owner.state = &s
			response.State = &s
			response.Handle = id
		}
		h.adopt(id, owner)
		return response, e
	case "launch", "record_fixture":
		kinds, ok := h.resources.Arguments[q.Name]
		if !ok || len(kinds) != len(q.Args) {
			return response, bad("missing or mismatched application argument binding")
		}
		for i, kind := range kinds {
			if q.Args[i].Type != kind {
				return response, bad("argument kind differs from application binding")
			}
		}
		args, e := h.arguments(q.Args, method == "record_fixture")
		if e != nil {
			return response, e
		}
		stream, ok := h.resources.Streams[q.Stream]
		if !ok {
			return response, bad("unknown stream")
		}
		if q.Geometry == nil {
			return response, bad("launch geometry required")
		}
		cfg := cudalaunch.Config{Grid: q.Geometry.Grid, Block: q.Geometry.Block, SharedMem: q.Geometry.SharedMem, Stream: stream}
		var fc cudahotswap.FixtureConfig
		if method == "record_fixture" {
			fc = cudahotswap.FixtureConfig{Args: args, Outputs: q.Outputs, Launch: cfg, Policy: cudahotswap.DefaultReplayPolicy()}
			if q.Policy != nil {
				fc.Policy = *q.Policy
			}
			for _, b := range q.Buffers {
				buffer, ok := h.resources.Buffers[b.Name]
				if !ok || b.Bytes <= 0 || b.Bytes > buffer.Len() || b.Arg < 0 || b.Arg >= len(q.Args) || q.Args[b.Arg].Type != "buffer" || q.Args[b.Arg].Name != b.Name {
					return response, bad("invalid fixture buffer binding")
				}
				fc.Buffers = append(fc.Buffers, cudahotswap.FixtureBuffer{Arg: b.Arg, Bytes: b.Bytes, Stable: b.Stable, Buffer: buffer})
			}
		}
		id, e := h.reserve()
		if e != nil {
			return response, e
		}
		owner := retained{}
		if method == "launch" {
			p, e := h.service.Launch(ctx, q.Name, cfg, args...)
			err = e
			if p != nil {
				owner.wait = p.Wait
				s := p.State()
				owner.state = &s
				response.State = &s
				response.Handle = id
			}
		} else {
			p, e := h.service.RecordFixture(ctx, q.Name, fc)
			err = e
			if p != nil {
				owner.fixture = p
				response.Handle = id
				s := p.State()
				owner.state = &s
				response.State = &s
			}
		}
		h.adopt(id, owner)
		return response, err
	case "release_pin", "wait", "close_fixture":
		h.mu.Lock()
		owner, ok := h.handles[q.Handle]
		h.mu.Unlock()
		if !ok {
			return response, bad("unknown lifetime handle")
		}
		response.State = owner.state
		response.States = owner.states
		if method == "release_pin" && owner.release != nil {
			err = owner.release(ctx)
		} else if method == "wait" && owner.wait != nil {
			err = owner.wait(ctx)
		} else if method == "close_fixture" && owner.fixture != nil {
			err = owner.fixture.Close(ctx)
		} else {
			return response, bad("lifetime handle kind mismatch")
		}
		if err == nil {
			h.mu.Lock()
			delete(h.handles, q.Handle)
			h.mu.Unlock()
		} else {
			response.Handle = q.Handle
		}
		return response, err
	}
	return response, bad("unsupported endpoint")
}

func (h *Handler) arguments(values []Argument, fixture bool) ([]any, error) {
	args := make([]any, len(values))
	for i, v := range values {
		if v.Type == "buffer" {
			b, ok := h.resources.Buffers[v.Name]
			if !ok || len(v.Value) != 0 {
				return nil, bad("unknown buffer or pointer value")
			}
			if !fixture {
				args[i] = b
			}
			continue
		}
		if v.Name != "" || len(v.Value) == 0 {
			return nil, bad("invalid scalar argument")
		}
		var target any
		switch v.Type {
		case "u32":
			target = new(uint32)
		case "i32":
			target = new(int32)
		case "u64":
			target = new(uint64)
		case "i64":
			target = new(int64)
		case "f32":
			target = new(float32)
		case "f64":
			target = new(float64)
		default:
			return nil, bad("unsupported argument type")
		}
		if bytes.Equal(bytes.TrimSpace(v.Value), []byte("null")) || json.Unmarshal(v.Value, target) != nil {
			return nil, bad("invalid scalar value")
		}
		args[i] = reflect.ValueOf(target).Elem().Interface()
	}
	return args, nil
}

// beginRequest and endRequest retain transport ownership until the response
// actually returns, including a writer that does not respond to cancellation.
func (h *Handler) beginRequest() bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.closed {
		return false
	}
	if h.active == 0 {
		h.idle = make(chan struct{})
	}
	h.active++
	return true
}

func (h *Handler) endRequest() {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.active--
	if h.active == 0 {
		close(h.idle)
	}
}

// Close cancels transport streams, drains completion handles, releases pins,
// then closes the Service. It waits for in-flight requests to return before
// releasing owners. Cancellation leaves those requests and owners retained for
// a later Close; callers must not destroy application resources until it succeeds.
func (h *Handler) Close(ctx context.Context) error {
	if ctx == nil {
		return errors.New("httpapi: nil close context")
	}
	h.mu.Lock()
	h.closed = true
	idle := h.idle
	h.mu.Unlock()
	h.cancel()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-idle:
	}
	// Only one Close can retire owners at a time. A second Close must also
	// honor its own deadline while the first is waiting on a retained owner.
	select {
	case <-ctx.Done():
		return ctx.Err()
	case h.closeGate <- struct{}{}:
	}
	defer func() { <-h.closeGate }()
	if err := ctx.Err(); err != nil {
		return err
	}
	h.mu.Lock()
	owners := cloneMap(h.handles)
	h.mu.Unlock()
	for _, kind := range []string{"wait", "fixture", "pin"} {
		for id, owner := range owners {
			var err error
			used := false
			switch {
			case kind == "wait" && owner.wait != nil:
				used = true
				err = owner.wait(ctx)
			case kind == "fixture" && owner.fixture != nil:
				used = true
				err = owner.fixture.Close(ctx)
			case kind == "pin" && owner.release != nil:
				used = true
				err = owner.release(ctx)
			}
			if err != nil {
				return err
			}
			if used {
				h.mu.Lock()
				delete(h.handles, id)
				h.mu.Unlock()
			}
		}
	}
	return h.service.Close(ctx)
}
