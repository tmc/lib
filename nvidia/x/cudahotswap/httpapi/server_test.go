package httpapi

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudahotswap"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
)

type fakeService struct {
	Service
	mu      sync.Mutex
	calls   []string
	poison  error
	pushErr error
}

func (f *fakeService) mark(method string) cudahotswap.State {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls = append(f.calls, method)
	return cudahotswap.State{Name: "kernel", Generation: 9, ConfigRevision: 7}
}
func (f *fakeService) Err() error { f.mark("Err"); return f.poison }
func (f *fakeService) Register(context.Context, string, cudahotswap.Source, string) (cudahotswap.State, error) {
	return f.mark("Register"), nil
}
func (f *fakeService) Push(context.Context, string, cudahotswap.Source, string) (*cudahotswap.PushResult, error) {
	f.mark("Push")
	out := cudahotswap.OutcomeOK
	if f.pushErr != nil {
		out = cudahotswap.OutcomeWrongAnswer
	}
	return &cudahotswap.PushResult{Report: cudahotswap.Report{Outcome: out, Reason: "CPU reference differs", Generation: 9, ConfigRevision: 7}}, f.pushErr
}
func (f *fakeService) List() []cudahotswap.State                { return []cudahotswap.State{f.mark("List")} }
func (f *fakeService) Status(string) (cudahotswap.State, error) { return f.mark("Status"), nil }
func (f *fakeService) Versions(string) ([]cudahotswap.Version, error) {
	f.mark("Versions")
	return nil, nil
}
func (f *fakeService) Select(context.Context, string, uint64, string) (cudahotswap.State, error) {
	return f.mark("Select"), nil
}
func (f *fakeService) Revert(context.Context, string, string) (cudahotswap.State, error) {
	return f.mark("Revert"), nil
}
func (f *fakeService) SetCompileFlags(string, []string, string) (cudahotswap.State, error) {
	return f.mark("SetCompileFlags"), nil
}
func (f *fakeService) SetGeometry(string, *cudahotswap.LaunchGeometry, string) (cudahotswap.State, error) {
	return f.mark("SetGeometry"), nil
}
func (f *fakeService) SetCheck(string, cudahotswap.CheckConfig, string) (cudahotswap.State, error) {
	return f.mark("SetCheck"), nil
}
func (f *fakeService) SetWatchPaths(string, *cudahotswap.WatchPaths, string) (cudahotswap.State, error) {
	return f.mark("SetWatchPaths"), nil
}
func (f *fakeService) WatchPaths(string) (*cudahotswap.WatchPaths, error) {
	f.mark("WatchPaths")
	return nil, nil
}
func (f *fakeService) Watch(context.Context, cudahotswap.WatchConfig, func(string, *cudahotswap.PushResult, error) error) error {
	f.mark("Watch")
	return nil
}
func (f *fakeService) Subscribe(context.Context, uint64) (*cudahotswap.Subscription, error) {
	f.mark("Subscribe")
	return nil, errors.New("test subscription refused")
}
func (f *fakeService) Pin(string) (*cudahotswap.Pin, error) { f.mark("Pin"); return nil, nil }
func (f *fakeService) Launch(context.Context, string, cudalaunch.Config, ...any) (*cudahotswap.PendingLaunch, error) {
	f.mark("Launch")
	return nil, nil
}
func (f *fakeService) RecordFixture(context.Context, string, cudahotswap.FixtureConfig) (*cudahotswap.Fixture, error) {
	f.mark("RecordFixture")
	return nil, nil
}
func (f *fakeService) RegisterGraph(context.Context, string, *cudagraph.Exec, *cudagraph.Node, *cuda.Stream, driver.Dim3, ...any) (cudahotswap.State, error) {
	return f.mark("RegisterGraph"), nil
}
func (f *fakeService) LaunchGraph(context.Context, *cudagraph.Exec) (*cudahotswap.GraphLaunch, error) {
	f.mark("LaunchGraph")
	return nil, nil
}
func (f *fakeService) Close(context.Context) error { f.mark("Close"); return nil }

func TestExportedServiceEndpoints(t *testing.T) {
	typ := reflect.TypeOf((*cudahotswap.Service)(nil))
	endpoints := Endpoints()
	if len(endpoints) != typ.NumMethod() {
		t.Fatalf("endpoints %d methods %d", len(endpoints), typ.NumMethod())
	}
	for i := 0; i < typ.NumMethod(); i++ {
		if endpoints[typ.Method(i).Name] == "" {
			t.Errorf("missing endpoint for %s", typ.Method(i).Name)
		}
	}
	delete(endpoints, "Push")
	if Endpoints()["Push"] == "" {
		t.Fatal("caller mutated routing table")
	}
}

func TestEveryEndpoint(t *testing.T) {
	for method, endpoint := range Endpoints() {
		t.Run(method, func(t *testing.T) {
			f := new(fakeService)
			h, err := New(f, Resources{Arguments: map[string][]string{"kernel": {}}, Streams: map[string]*cuda.Stream{"main": {}}, Graphs: map[string]Graph{"graph": {Exec: &cudagraph.Exec{}, Node: &cudagraph.Node{}, Stream: &cuda.Stream{}}}})
			if err != nil {
				t.Fatal(err)
			}
			q := Request{Name: "kernel", Label: "caller", Stream: "main", Graph: "graph", Geometry: &cudahotswap.LaunchGeometry{}}
			b, _ := json.Marshal(q)
			r := httptest.NewRequest(http.MethodPost, "/v1/"+endpoint, bytes.NewReader(b))
			if method == "Subscribe" {
				r = httptest.NewRequest(http.MethodGet, "/v1/events?since=3", nil)
			}
			w := httptest.NewRecorder()
			h.ServeHTTP(w, r)
			if w.Code != 200 {
				t.Fatalf("status %d: %s", w.Code, w.Body.String())
			}
			found := false
			for _, call := range f.calls {
				found = found || call == method
			}
			if !found {
				t.Fatalf("%s not called: %v", method, f.calls)
			}
			if method != "Subscribe" && method != "Watch" {
				var response Response
				if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
					t.Fatal(err)
				}
				if response.State != nil && (response.State.Generation != 9 || response.State.ConfigRevision != 7) {
					t.Fatal(response)
				}
			}
		})
	}
}

func TestFailureAndPoison(t *testing.T) {
	for _, poison := range []bool{false, true} {
		t.Run(map[bool]string{false: "check", true: "poison"}[poison], func(t *testing.T) {
			f := &fakeService{pushErr: errors.New("wrong result")}
			if poison {
				f.poison = errors.Join(cudahotswap.ErrPoisoned, errors.New("device cause"))
			}
			h, _ := New(f, Resources{})
			w := httptest.NewRecorder()
			h.ServeHTTP(w, httptest.NewRequest("POST", "/v1/push", strings.NewReader(`{"name":"kernel","label":"caller"}`)))
			want := 200
			if poison {
				want = 503
			}
			if w.Code != want {
				t.Fatalf("status %d want %d", w.Code, want)
			}
			var res Response
			if err := json.Unmarshal(w.Body.Bytes(), &res); err != nil {
				t.Fatal(err)
			}
			if res.Failure == nil {
				t.Fatal("failure hidden")
			}
			if poison && !res.Failure.Poisoned {
				t.Fatal("poison identity hidden")
			}
			if !poison && (res.Push == nil || res.Failure.Outcome != cudahotswap.OutcomeWrongAnswer) {
				t.Fatal("classified check hidden")
			}
		})
	}
}

func TestRetainedCompletionAndCloseOrdering(t *testing.T) {
	f := &fakeService{poison: cudahotswap.ErrPoisoned}
	h, _ := New(f, Resources{})
	fail := true
	var order []string
	h.handles["wait"] = retained{wait: func(context.Context) error {
		order = append(order, "wait")
		if fail {
			return errors.New("not completed")
		}
		return nil
	}}
	h.handles["pin"] = retained{release: func(context.Context) error { order = append(order, "pin"); return nil }}
	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("POST", "/v1/wait", strings.NewReader(`{"handle":"wait","label":"cleanup"}`)))
	if w.Code != 200 || h.handles["wait"].wait == nil {
		t.Fatal("cleanup blocked by poison or owner lost")
	}
	if err := h.Close(context.Background()); err == nil || len(h.handles) != 2 {
		t.Fatal("failed close dropped owners")
	}
	fail = false
	order = nil
	if err := h.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(order, []string{"wait", "pin"}) || len(h.handles) != 0 {
		t.Fatalf("cleanup %v owners %v", order, h.handles)
	}
}

func TestRequestRefusals(t *testing.T) {
	for _, body := range []string{`{"name":"x","name":"y"}`, `{"unknown":1}`, `{} {}`, `{"args":[{"type":"pointer","value":42}]}`, strings.Repeat("[", 66) + "0" + strings.Repeat("]", 66)} {
		h, _ := New(new(fakeService), Resources{})
		w := httptest.NewRecorder()
		h.ServeHTTP(w, httptest.NewRequest("POST", "/v1/launch", strings.NewReader(body)))
		if w.Code != 400 {
			t.Errorf("accepted %q status %d", body, w.Code)
		}
	}
	h, _ := New(new(fakeService), Resources{})
	for _, v := range []string{"null", " null ", "4294967296", "-1", "1.5"} {
		if _, err := h.arguments([]Argument{{Type: "u32", Value: json.RawMessage(v)}}, false); err == nil {
			t.Errorf("accepted scalar %q", v)
		}
	}
}

func FuzzRequest(f *testing.F) {
	for _, s := range []string{`{}`, `{"label":"x"}`, `{"a":{"a":1}}`, `{"a":1,"a":2}`} {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, s string) {
		if len(s) > 1<<16 {
			t.Skip()
		}
		_ = validateJSON([]byte(s))
	})
}
