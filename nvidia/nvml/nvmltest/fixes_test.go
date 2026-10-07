package nvmltest

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/nvml"
)

func TestReadFieldIdentities(t *testing.T) {
	valid, err := json.Marshal(script())
	if err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct{ name, old, replacement string }{
		{"duplicate version", `"version":1`, `"version":999,"version":1`},
		{"aliased version", `"version":1`, `"version":999,"Version":1`},
		{"single alias", `"version":1`, `"Version":1`},
		{"provenance alias", `"source":`, `"Source":`},
		{"nested cursor duplicate", `"LastSeen":18446744073709551615`, `"LastSeen":9,"lastseen":18446744073709551615`},
		{"result duplicate", `"Pid":17`, `"Pid":13,"Pid":17`},
		{"result alias", `"Pid":17`, `"pid":17`},
	} {
		t.Run(tt.name, func(t *testing.T) {
			input := strings.Replace(string(valid), tt.old, tt.replacement, 1)
			if input == string(valid) {
				t.Fatal("control did not change input")
			}
			if _, err := Read(strings.NewReader(input)); err == nil {
				t.Fatal("ambiguous or aliased input accepted")
			}
		})
	}
}

func TestArgumentFieldsRequired(t *testing.T) {
	for _, tt := range []struct{ method, args, result string }{
		{"ProcessesUtilization", `{}`, `[]`},
		{"ProcessesUtilization", `{"lastseen":0}`, `[]`},
		{"ProcessesUtilization", `{"LastSeen":null}`, `[]`},
		{"FieldValuesFor", `{}`, `[]`},
		{"GpuInstanceList", `{}`, `[]`},
		{"GpuInstanceList", `{"ProfileID":null}`, `[]`},
	} {
		rec := Recording{Version: Version, Calls: []Call{{Method: tt.method, Args: json.RawMessage(tt.args), Result: json.RawMessage(tt.result)}}}
		if _, err := NewReplay(rec); err == nil {
			t.Errorf("accepted %s arguments %s", tt.method, tt.args)
		}
	}
	for _, tt := range []struct {
		args string
		ids  []uint32
	}{
		{`{"IDs":null}`, nil}, {`{"IDs":[]}`, []uint32{}},
	} {
		rec := Recording{Version: Version, Calls: []Call{{Method: "FieldValuesFor", Args: json.RawMessage(tt.args), Result: json.RawMessage(`[]`)}}}
		r, err := NewReplay(rec)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := r.FieldValuesFor(tt.ids...); err != nil {
			t.Fatal(err)
		}
	}
}

type fixQueries struct {
	Queries
	calls int
	err   error
}

func (q *fixQueries) GpmSupported() (bool, error) { q.calls++; return true, q.err }
func (q *fixQueries) FieldValuesFor(ids ...uint32) ([]nvml.FieldValue, error) {
	q.calls++
	ids[0] = 99
	return nil, nil
}

func TestRecorderSnapshotsInvocation(t *testing.T) {
	q := new(fixQueries)
	r, err := NewRecorder(q, Provenance{})
	if err != nil {
		t.Fatal(err)
	}
	ids := []uint32{17}
	if _, err := r.FieldValuesFor(ids...); err != nil {
		t.Fatal(err)
	}
	rec, err := r.Recording()
	if err != nil {
		t.Fatal(err)
	}
	if string(rec.Calls[0].Args) != `{"IDs":[17]}` || ids[0] != 99 {
		t.Fatalf("invocation=%s caller=%v", rec.Calls[0].Args, ids)
	}
}

func TestRecorderEnvelopeAndArgumentsBudget(t *testing.T) {
	q := new(fixQueries)
	if _, err := NewRecorder(q, Provenance{Source: strings.Repeat("a", maxBytes+1)}); err == nil {
		t.Fatal("accepted oversized envelope")
	}
	r, err := NewRecorder(q, Provenance{})
	if err != nil {
		t.Fatal(err)
	}
	initial := r.size
	r, err = NewRecorder(q, Provenance{Source: strings.Repeat("a", maxBytes-initial-16)})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := r.GpmSupported(); err == nil || q.calls != 0 {
		t.Fatalf("known exhausted request reached backend: %v calls=%d", err, q.calls)
	}
}

func TestRecorderSerializedBudget(t *testing.T) {
	r, err := NewRecorder(new(fixQueries), Provenance{Source: "source"})
	if err != nil {
		t.Fatal(err)
	}
	for range 3 {
		if _, err := r.GpmSupported(); err != nil {
			t.Fatal(err)
		}
		var b bytes.Buffer
		if err := Write(&b, r.rec); err != nil {
			t.Fatal(err)
		}
		if r.size != b.Len() {
			t.Fatalf("size=%d serialized=%d", r.size, b.Len())
		}
	}
}

func TestErrorBoundRefusesClassificationLoss(t *testing.T) {
	for _, depth := range []int{32, 40} {
		var source error = nvml.ErrNotSupported
		for range depth {
			source = fmt.Errorf("wrapped: %w", source)
		}
		r, err := NewRecorder(&fixQueries{err: source}, Provenance{})
		if err != nil {
			t.Fatal(err)
		}
		_, err = r.GpmSupported()
		if depth > 32 {
			if err == nil || nvml.IsUnsupported(err) || len(r.rec.Calls) != 0 {
				t.Fatalf("deep cause silently recorded: err=%v calls=%d", err, len(r.rec.Calls))
			}
			continue
		}
		if !nvml.IsUnsupported(err) {
			t.Fatal(err)
		}
		rec, err := r.Recording()
		if err != nil {
			t.Fatal(err)
		}
		replay, err := NewReplay(rec)
		if err != nil {
			t.Fatal(err)
		}
		_, err = replay.GpmSupported()
		if !errors.Is(err, nvml.ErrNotSupported) {
			t.Fatal(err)
		}
	}
}

func TestErrorCauseLimit(t *testing.T) {
	causes := make([]error, 4097)
	for i := range causes {
		causes[i] = nvml.ErrNoPermission
	}
	if _, err := failure(errors.Join(causes...), 0); err == nil {
		t.Fatal("accepted excessive error causes")
	}
}

func TestJSONNestingBound(t *testing.T) {
	for _, depth := range []int{96, 97} {
		input := strings.Repeat("[", depth) + "0" + strings.Repeat("]", depth)
		var value any
		err := decode([]byte(input), &value)
		if (err == nil) != (depth == 96) {
			t.Fatalf("depth=%d error=%v", depth, err)
		}
	}
}

func TestRecorderTypedNilDevice(t *testing.T) {
	var q *fixQueries
	if _, err := NewRecorder(q, Provenance{}); err == nil {
		t.Fatal("accepted typed nil device")
	}
}

func TestRecordingFormattingRoundTrip(t *testing.T) {
	rec := Recording{Version: Version, Calls: []Call{{
		Method: "ProcessesUtilization",
		Args:   json.RawMessage(`{"LastSeen": 1000000000000000000}`),
		Result: json.RawMessage(`[{"Pid": 17, "SmUtil": 19}]`),
	}}}
	var first bytes.Buffer
	if err := Write(&first, rec); err != nil {
		t.Fatal(err)
	}
	copy, err := Read(bytes.NewReader(first.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	replay, err := NewReplay(copy)
	if err != nil {
		t.Fatal(err)
	}
	result, err := replay.ProcessesUtilization(1000000000000000000)
	if err != nil || len(result) != 1 || result[0].Pid != 17 || result[0].SmUtil != 19 {
		t.Fatalf("replay result=%v error=%v", result, err)
	}
	var second bytes.Buffer
	if err := Write(&second, copy); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(first.Bytes(), second.Bytes()) {
		t.Fatal("serialized recording changed after round trip")
	}
}

func FuzzRecording(f *testing.F) {
	b, _ := json.Marshal(script())
	f.Add(b)
	f.Add([]byte(`{"version":1,"provenance":{},"calls":[]}`))
	f.Fuzz(func(t *testing.T, b []byte) {
		if len(b) > 1<<20 {
			return
		}
		rec, err := Read(bytes.NewReader(b))
		if err != nil {
			return
		}
		var out bytes.Buffer
		if err := Write(&out, rec); err != nil {
			t.Fatal(err)
		}
		copy, err := Read(bytes.NewReader(out.Bytes()))
		if err != nil {
			t.Fatal(err)
		}
		// RawMessage retains input formatting; Write compacts it. Compare
		// the serialized values, including their exact integer spellings.
		var again bytes.Buffer
		if err := Write(&again, copy); err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(again.Bytes(), out.Bytes()) {
			t.Fatal("serialized recording changed after round trip")
		}
	})
}
