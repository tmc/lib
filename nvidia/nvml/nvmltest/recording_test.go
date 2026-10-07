package nvmltest

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/nvml"
)

func script() Recording {
	return Recording{Version: Version, Provenance: Provenance{Source: "synthetic host fixture"}, Calls: []Call{
		{Method: "GpmSupported", Args: json.RawMessage(`{}`), Result: json.RawMessage(`true`)},
		{Method: "ProcessesUtilization", Args: json.RawMessage(`{"LastSeen":18446744073709551615}`), Result: json.RawMessage(`[{"Pid":17,"SmUtil":23}]`)},
		{Method: "FabricInfo", Args: json.RawMessage(`{}`), Result: json.RawMessage(`{"CliqueId":29,"Status":4}`)},
		{Method: "FieldValuesFor", Args: json.RawMessage(`{"IDs":[71,83]}`), Result: json.RawMessage(`[{"FieldId":71,"NvmlReturn":4},{"FieldId":83,"NvmlReturn":0}]`)},
		{Method: "GpuInstanceList", Args: json.RawMessage(`{"ProfileID":31}`), Result: json.RawMessage(`[37,41]`)},
	}}
}

func consume(t *testing.T, q Queries) {
	t.Helper()
	if v, err := q.GpmSupported(); err != nil || !v {
		t.Fatalf("support %v, %v", v, err)
	}
	if v, err := q.ProcessesUtilization(^uint64(0)); err != nil || len(v) != 1 || v[0].Pid != 17 || v[0].SmUtil != 23 {
		t.Fatalf("processes %v, %v", v, err)
	}
	if v, err := q.FabricInfo(); err != nil || v.CliqueId != 29 || v.Status != nvml.ErrNoPermission {
		t.Fatalf("fabric %v, %v", v, err)
	}
	if v, err := q.FieldValuesFor(71, 83); err != nil || len(v) != 2 || v[0].NvmlReturn != nvml.ErrNoPermission || v[1].FieldId != 83 {
		t.Fatalf("fields %v, %v", v, err)
	}
	if v, err := q.GpuInstanceList(31); err != nil || !reflect.DeepEqual(v, []nvml.GpuInstance{37, 41}) {
		t.Fatalf("instances %v, %v", v, err)
	}
}

func TestRecorderReplayConsumerRoundTrip(t *testing.T) {
	r, err := NewReplay(script())
	if err != nil {
		t.Fatal(err)
	}
	recorder, err := NewRecorder(r, Provenance{Source: "synthetic host fixture"})
	if err != nil {
		t.Fatal(err)
	}
	consume(t, recorder)
	rec, err := recorder.Recording()
	if err != nil {
		t.Fatal(err)
	}
	var b bytes.Buffer
	if err := Write(&b, rec); err != nil {
		t.Fatal(err)
	}
	read, err := Read(&b)
	if err != nil {
		t.Fatal(err)
	}
	second, err := NewReplay(read)
	if err != nil {
		t.Fatal(err)
	}
	consume(t, second)
	if !r.Done() || !second.Done() {
		t.Fatal("unconsumed calls")
	}
	if _, err := second.GpmSupported(); err == nil {
		t.Fatal("accepted extra call")
	}
}

func TestReplayArgumentsOrderingAndCopy(t *testing.T) {
	rec := script()
	r, err := NewReplay(rec)
	if err != nil {
		t.Fatal(err)
	}
	rec.Calls[0].Result[0] = 'f'
	if _, err := r.FabricInfo(); err == nil {
		t.Fatal("accepted out-of-order call")
	}
	if v, err := r.GpmSupported(); err != nil || !v {
		t.Fatal("caller mutated replay")
	}
	if _, err := r.ProcessesUtilization(0); err == nil {
		t.Fatal("accepted wrong cursor")
	}
	processes, err := r.ProcessesUtilization(^uint64(0))
	if err != nil {
		t.Fatal(err)
	}
	processes[0].Pid = 999
	if _, err := r.FabricInfo(); err != nil {
		t.Fatal(err)
	}
	if _, err := r.FieldValuesFor(83, 71); err == nil {
		t.Fatal("accepted reversed field IDs")
	}
	if _, err := r.FieldValuesFor(71, 83); err != nil {
		t.Fatal(err)
	}
	if _, err := r.GpuInstanceList(32); err == nil {
		t.Fatal("accepted wrong profile")
	}
	if _, err := r.GpuInstanceList(31); err != nil {
		t.Fatal(err)
	}
}

func TestErrorClassificationRoundTrip(t *testing.T) {
	for _, err := range []error{nvml.ErrNotSupported, nvml.ErrNoPermission, nvml.ErrGpuIsLost, &nvml.SymbolError{Symbol: "old"}, &nvml.SymbolError{Symbol: "bad", Err: errors.New("bind failed")}, fmt.Errorf("query: %w", nvml.ErrNoPermission), errors.Join(nvml.ErrNoPermission, &nvml.SymbolError{Symbol: "old"}), &nvml.CallError{Func: "f", Result: nvml.ErrGpuIsLost}} {
		f, e := failure(err, 0)
		if e != nil {
			t.Fatal(e)
		}
		rec := Recording{Version: Version, Calls: []Call{{Method: "GpmSupported", Args: json.RawMessage(`{}`), Result: json.RawMessage(`false`), Error: f}}}
		r, e := NewReplay(rec)
		if e != nil {
			t.Fatal(e)
		}
		recorder, e := NewRecorder(r, Provenance{})
		if e != nil {
			t.Fatal(e)
		}
		_, e = recorder.GpmSupported()
		if e == nil || e.Error() != err.Error() || nvml.IsUnsupported(e) != nvml.IsUnsupported(err) {
			t.Fatalf("error %v -> %v", err, e)
		}
		for _, code := range []nvml.Result{nvml.ErrNoPermission, nvml.ErrGpuIsLost, nvml.ErrNotSupported} {
			if errors.Is(e, code) != errors.Is(err, code) {
				t.Fatalf("error %v lost code %v", err, code)
			}
		}
		out, e := recorder.Recording()
		if e != nil {
			t.Fatal(e)
		}
		if len(out.Calls) != 1 || out.Calls[0].Error == nil {
			t.Fatal("recorded error lost")
		}
	}
}

func TestReadRefusesMalformed(t *testing.T) {
	valid, _ := json.Marshal(script())
	for _, input := range []string{"", `{}`, `{"version":2}`, string(valid[:len(valid)-1]), string(valid) + `{}`, strings.Replace(string(valid), `"version":1`, `"version":1,"unknown":0`, 1), strings.Replace(string(valid), `"method":"GpmSupported"`, `"method":"unknown"`, 1), strings.Replace(string(valid), `"args":{}`, `"args":null`, 1), strings.Replace(string(valid), `"result":true`, `"result":null`, 1), strings.Replace(string(valid), `"LastSeen":18446744073709551615`, `"LastSeen":18446744073709551616`, 1)} {
		if _, err := Read(strings.NewReader(input)); err == nil {
			t.Fatalf("accepted malformed input %.100s", input)
		}
	}
	if _, err := Read(strings.NewReader(strings.Repeat(" ", maxBytes+1))); err == nil {
		t.Fatal("accepted oversized input")
	}
	bad := script()
	bad.Calls = make([]Call, maxCalls+1)
	if _, err := NewReplay(bad); err == nil {
		t.Fatal("accepted oversized script")
	}
	bad = script()
	bad.Provenance.Measured = true
	if _, err := NewReplay(bad); err == nil {
		t.Fatal("accepted measured provenance without identities")
	}
	bad = script()
	bad.Calls[0].Error = &Failure{Kind: "result", Message: "success", Code: nvml.Success}
	if _, err := NewReplay(bad); err == nil {
		t.Fatal("accepted success error")
	}
}

func TestConcurrentReplay(t *testing.T) {
	rec := Recording{Version: Version}
	for i := 0; i < 100; i++ {
		rec.Calls = append(rec.Calls, Call{Method: "GpmSupported", Args: json.RawMessage(`{}`), Result: json.RawMessage(`true`)})
	}
	r, err := NewReplay(rec)
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if v, e := r.GpmSupported(); e != nil || !v {
				t.Errorf("support=%v,%v", v, e)
			}
		}()
	}
	wg.Wait()
	if !r.Done() {
		t.Fatal("remaining calls")
	}
}

type shortWriter struct{}

func (shortWriter) Write([]byte) (int, error) { return 0, nil }
func TestWriteShort(t *testing.T) {
	if err := Write(shortWriter{}, script()); !errors.Is(err, io.ErrShortWrite) {
		t.Fatalf("error=%v", err)
	}
}

func TestRecorderBudget(t *testing.T) {
	r, err := NewReplay(script())
	if err != nil {
		t.Fatal(err)
	}
	recorder, err := NewRecorder(r, Provenance{})
	if err != nil {
		t.Fatal(err)
	}
	recorder.size = maxBytes
	if _, err := recorder.GpmSupported(); err == nil {
		t.Fatal("accepted exhausted budget")
	}
	if r.next != 0 {
		t.Fatal("executed query after exhausted budget")
	}
}
