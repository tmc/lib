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

type concurrentRecorderQueries struct {
	mu      sync.Mutex
	calls   []uint64
	block   uint64
	entered chan struct{}
	release chan struct{}
}

func (q *concurrentRecorderQueries) GpmSupported() (bool, error) { return false, nil }
func (q *concurrentRecorderQueries) FabricInfo() (nvml.GpuFabricInfoV, error) {
	return nvml.GpuFabricInfoV{}, nil
}
func (q *concurrentRecorderQueries) FieldValuesFor(...uint32) ([]nvml.FieldValue, error) {
	return nil, nil
}
func (q *concurrentRecorderQueries) GpuInstanceList(uint32) ([]nvml.GpuInstance, error) {
	return nil, nil
}
func (q *concurrentRecorderQueries) ProcessesUtilization(lastSeen uint64) ([]nvml.ProcessUtilizationInfoV1, error) {
	q.mu.Lock()
	q.calls = append(q.calls, lastSeen)
	blocked := lastSeen == q.block
	if blocked {
		close(q.entered)
	}
	q.mu.Unlock()
	if blocked {
		<-q.release
	}
	return []nvml.ProcessUtilizationInfoV1{{Pid: uint32(lastSeen)}}, nil
}

func (q *concurrentRecorderQueries) callOrder() []uint64 {
	q.mu.Lock()
	defer q.mu.Unlock()
	return append([]uint64(nil), q.calls...)
}

func TestConcurrentRecorderQueriesAndSnapshots(t *testing.T) {
	const queryCount = 32
	const snapshotCount = 4
	device := &concurrentRecorderQueries{block: 1, entered: make(chan struct{}), release: make(chan struct{})}
	recorder, err := NewRecorder(device, Provenance{Source: "synthetic concurrent host fixture"})
	if err != nil {
		t.Fatal(err)
	}
	seed, err := recorder.ProcessesUtilization(0)
	if err != nil || len(seed) != 1 || seed[0].Pid != 0 {
		t.Fatalf("seed query returned %v, %v", seed, err)
	}

	startQueries := make(chan struct{})
	queryResults := make(chan error, queryCount-1)
	var queries sync.WaitGroup
	for i := 1; i < queryCount; i++ {
		lastSeen := uint64(i)
		queries.Add(1)
		go func() {
			defer queries.Done()
			<-startQueries
			got, err := recorder.ProcessesUtilization(lastSeen)
			if err != nil {
				queryResults <- err
				return
			}
			if len(got) != 1 || got[0].Pid != uint32(lastSeen) {
				queryResults <- fmt.Errorf("query %d returned %v", lastSeen, got)
			}
		}()
	}
	close(startQueries)
	<-device.entered

	// Start each snapshot caller while query 1 is held in the fake device call.
	// The barrier overlaps the API invocations; assertions below check snapshot
	// prefix integrity and copy isolation without assuming which caller wins the
	// recorder lock after the held query returns.
	snapshotStart := make(chan struct{})
	snapshotAttempts := make(chan struct{}, snapshotCount)
	snapshots := make(chan Recording, snapshotCount)
	snapshotErrors := make(chan error, snapshotCount)
	var snapshotters sync.WaitGroup
	for i := 0; i < snapshotCount; i++ {
		snapshotters.Add(1)
		go func() {
			defer snapshotters.Done()
			<-snapshotStart
			snapshotAttempts <- struct{}{}
			snapshot, err := recorder.Recording()
			if err != nil {
				snapshotErrors <- err
				return
			}
			snapshots <- snapshot
		}()
	}
	close(snapshotStart)
	for i := 0; i < snapshotCount; i++ {
		<-snapshotAttempts
	}
	close(device.release)
	queries.Wait()
	snapshotters.Wait()
	close(queryResults)
	close(snapshotErrors)
	close(snapshots)
	for err := range queryResults {
		t.Error(err)
	}
	for err := range snapshotErrors {
		t.Error(err)
	}

	final, err := recorder.Recording()
	if err != nil {
		t.Fatal(err)
	}
	if got := len(final.Calls); got != queryCount {
		t.Fatalf("recorded %d calls, want %d", got, queryCount)
	}
	expected, err := json.Marshal(final)
	if err != nil {
		t.Fatal(err)
	}
	callOrder := device.callOrder()
	if len(callOrder) != queryCount {
		t.Fatalf("device observed %d calls, want %d", len(callOrder), queryCount)
	}
	for i, call := range final.Calls {
		if call.Method != "ProcessesUtilization" {
			t.Fatalf("call %d method = %q", i, call.Method)
		}
		var args struct{ LastSeen uint64 }
		if err := json.Unmarshal(call.Args, &args); err != nil {
			t.Fatalf("call %d arguments: %v", i, err)
		}
		var result []nvml.ProcessUtilizationInfoV1
		if err := json.Unmarshal(call.Result, &result); err != nil {
			t.Fatalf("call %d result: %v", i, err)
		}
		if args.LastSeen != callOrder[i] || len(result) != 1 || result[0].Pid != uint32(args.LastSeen) {
			t.Fatalf("call %d does not preserve invocation order and response: args=%d device=%d result=%v", i, args.LastSeen, callOrder[i], result)
		}
	}

	for snapshotIndex := 0; ; snapshotIndex++ {
		snapshot, ok := <-snapshots
		if !ok {
			break
		}
		if len(snapshot.Calls) < 1 || len(snapshot.Calls) > queryCount {
			t.Fatalf("snapshot has %d calls, want 1..%d", len(snapshot.Calls), queryCount)
		}
		for i, call := range snapshot.Calls {
			if !reflect.DeepEqual(call, final.Calls[i]) {
				t.Fatalf("snapshot row %d is not an intact prefix of the final recording", i)
			}
		}
		snapshot.Calls[0].Args[0] = byte('a' + snapshotIndex)
		snapshot.Calls[0].Result[0] = byte('A' + snapshotIndex)
		current, err := recorder.Recording()
		if err != nil {
			t.Fatal(err)
		}
		got, err := json.Marshal(current)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, expected) {
			t.Fatal("mutating a returned snapshot changed recorder state")
		}
	}
	current, err := recorder.Recording()
	if err != nil {
		t.Fatal(err)
	}
	got, err := json.Marshal(current)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, expected) {
		t.Fatal("mutating returned snapshots changed recorder state")
	}
}
