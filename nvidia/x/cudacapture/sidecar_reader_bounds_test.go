package cudacapture_test

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof"
)

func stackTable(t *testing.T, process uint64, frames []string) string {
	t.Helper()
	data, err := json.Marshal(cudaprof.ExternalStackTable{Version: 1, Stacks: []cudaprof.ExternalStackRecord{{Process: process, ExternalKind: 3, ExternalID: 9, Frames: frames}}})
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

func TestTrackerProcessScopes(t *testing.T) {
	for _, tt := range []struct {
		name                  string
		stack, link, activity uint64
		wantStack, wantLink   bool
	}{
		{"known", 222, 222, 222, true, true},
		{"foreign-stack", 111, 222, 222, false, true},
		{"foreign-link", 222, 111, 222, false, false},
		{"unknown-stack", 0, 222, 222, false, true},
		{"unknown-link", 222, 0, 222, false, false},
		{"unknown-activity", 222, 222, 0, false, false},
		{"known-stack-unknown-link", 222, 0, 0, false, true},
		{"legacy", 0, 0, 0, true, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			ct := cudacapture.NewCorrelationTracker()
			if err := ct.ReadExternalStackTable(strings.NewReader(stackTable(t, tt.stack, []string{"owned"}))); err != nil {
				t.Fatal(err)
			}
			if err := ct.RecordRuntime(cudacapture.RuntimeLaunch{CorrelationID: 7}); err != nil {
				t.Fatal(err)
			}
			for _, row := range []cudacapture.ActivityRecord{
				{Process: tt.link, Correlation: 7, Subtype: "external_correlation", ExternalKind: 3, ExternalID: 9},
				{Process: tt.activity, Correlation: 7, Subtype: "kernel"},
			} {
				if err := ct.RecordActivity(row); err != nil {
					t.Fatal(err)
				}
			}
			matched, ambiguous, _, _ := ct.Join()
			if len(matched) != 1 || len(ambiguous) != 0 || matched[0].HasExternalID != tt.wantLink || (len(matched[0].Callstack) > 0) != tt.wantStack || matched[0].CallstackMissing != (tt.wantLink && !tt.wantStack) {
				t.Fatalf("matched=%+v ambiguous=%+v", matched, ambiguous)
			}
		})
	}
}

func TestTrackerReusedCUDAIDsAmbiguous(t *testing.T) {
	ct := cudacapture.NewCorrelationTracker()
	for _, pid := range []uint64{111, 222} {
		if err := ct.RecordRuntime(cudacapture.RuntimeLaunch{CorrelationID: 7}); err != nil {
			t.Fatal(err)
		}
		if err := ct.RecordActivity(cudacapture.ActivityRecord{Process: pid, Correlation: 7, Subtype: "kernel"}); err != nil {
			t.Fatal(err)
		}
	}
	matched, ambiguous, _, _ := ct.Join()
	if len(matched) != 0 || len(ambiguous) != 1 || ambiguous[0].RuntimeCount != 2 || ambiguous[0].ActivityCount != 2 {
		t.Fatalf("matched=%+v ambiguous=%+v", matched, ambiguous)
	}
}

func TestTrackerJournalCheckpoints(t *testing.T) {
	for _, count := range []string{`"00000000000000000065"`, "65"} {
		ct := cudacapture.NewCorrelationTracker()
		data := `{"version":1,"kind":"header","process":222,"epoch":"capture","losses":"00000000000000000001","losses_lower_bound":true}` + "\n" +
			`{"version":1,"kind":"stack","process":222,"epoch":"capture","external_kind":3,"external_id":9,"frames":["retained"]}` + "\n" +
			fmt.Sprintf(`{"version":1,"kind":"losses","process":222,"epoch":"capture","losses":%s}`, count) + "\n"
		if err := ct.ReadExternalStackTable(strings.NewReader(data)); err != nil {
			t.Fatal(err)
		}
		if ct.ExternalStackLosses() != 65 || !ct.ExternalStackLossesLowerBound() {
			t.Fatalf("losses=%d lower=%t", ct.ExternalStackLosses(), ct.ExternalStackLossesLowerBound())
		}
		if err := ct.RecordRuntime(cudacapture.RuntimeLaunch{CorrelationID: 7}); err != nil {
			t.Fatal(err)
		}
		for _, row := range []cudacapture.ActivityRecord{{Process: 222, Correlation: 7, Subtype: "external_correlation", ExternalKind: 3, ExternalID: 9}, {Process: 222, Correlation: 7, Subtype: "kernel"}} {
			if err := ct.RecordActivity(row); err != nil {
				t.Fatal(err)
			}
		}
		matches, _, _, _ := ct.Join()
		if len(matches) != 1 || len(matches[0].Callstack) != 1 || matches[0].Callstack[0] != "retained" {
			t.Fatalf("lost retained frames: %+v", matches)
		}
	}
}

func TestTrackerReaderRefusalsAtomic(t *testing.T) {
	header := `{"version":1,"kind":"header","losses":"5"}` + "\n"
	stack := `{"version":1,"kind":"stack","external_kind":3,"external_id":9,"frames":["injected"]}` + "\n"
	valid := stackTable(t, 0, []string{"valid"})
	for _, tt := range []struct{ name, data string }{
		{"decreasing", header + stack + `{"version":1,"kind":"losses","losses":"4"}` + "\n"},
		{"overflow-count", header + stack + `{"version":1,"kind":"losses","losses":"18446744073709551616"}` + "\n"},
		{"oversized-count", header + stack + `{"version":1,"kind":"losses","losses":"` + strings.Repeat("0", 21) + `"}` + "\n"},
		{"negative-count", header + stack + `{"version":1,"kind":"losses","losses":-1}` + "\n"},
		{"decimal-count", header + stack + `{"version":1,"kind":"losses","losses":1.0}` + "\n"},
		{"torn", header + stack + `{"version":1`},
		{"blank-row", header + stack + "\n"},
		{"unknown-key", strings.Replace(valid, `"version":1`, `"version":1,"extra":0`, 1)},
		{"duplicate-table-key", strings.Replace(valid, `"version":1`, `"version":1,"version":1`, 1)},
		{"duplicate-record-key", strings.Replace(valid, `"external_id":9`, `"external_id":9,"external_id":9`, 1)},
		{"invalid-late-row", `{"version":1,"losses":5,"stacks":[{"external_kind":3,"external_id":9,"frames":["injected"]},{"external_id":0,"frames":["bad"]}]}`},
		{"empty-frame", stackTable(t, 0, []string{""})},
		{"too-many-frames", stackTable(t, 0, strings.Split(strings.Repeat("f,", 66), ",")[:66])},
		{"large-frame", stackTable(t, 0, []string{strings.Repeat("f", 4097)})},
		{"large-total-frames", stackTable(t, 0, []string{strings.Repeat("f", 4096), strings.Repeat("f", 4096), "f"})},
		{"large-epoch", strings.Replace(header, `"losses":"5"`, `"losses":"5","epoch":"`+strings.Repeat("x", 129)+`"`, 1) + stack},
		{"mixed-process", header + strings.Replace(stack, `"kind":"stack"`, `"kind":"stack","process":222`, 1)},
		{"mixed-epoch", header + strings.Replace(stack, `"kind":"stack"`, `"kind":"stack","epoch":"other"`, 1)},
		{"large-row", header + `{"version":1,"kind":"stack","external_id":9,"frames":["` + strings.Repeat("f", 64<<10) + `"]}` + "\n"},
		{"too-many-rows", header + strings.Repeat(stack, 16385)},
		{"count-with-frames", header + stack + `{"version":1,"kind":"losses","losses":"6","frames":["wrong"]}` + "\n"},
		{"null-frame", strings.Replace(valid, `"frames":["valid"]`, `"frames":[null]`, 1)},
	} {
		t.Run(tt.name, func(t *testing.T) {
			ct := cudacapture.NewCorrelationTracker()
			if err := ct.ReadExternalStackTable(strings.NewReader(`{"version":1,"losses":2,"stacks":[]}`)); err != nil {
				t.Fatal(err)
			}
			if err := ct.ReadExternalStackTable(strings.NewReader(tt.data)); err == nil {
				t.Fatal("accepted invalid import")
			}
			if ct.ExternalStackLosses() != 2 || ct.ExternalStackLossesLowerBound() {
				t.Fatal("failed import changed losses")
			}
			if err := ct.RecordRuntime(cudacapture.RuntimeLaunch{CorrelationID: 7}); err != nil {
				t.Fatal(err)
			}
			if err := ct.RecordExternalCorrelationKind(3, 9, 7); err != nil {
				t.Fatal(err)
			}
			if err := ct.RecordActivity(cudacapture.ActivityRecord{Correlation: 7, Subtype: "kernel"}); err != nil {
				t.Fatal(err)
			}
			matches, _, _, _ := ct.Join()
			if len(matches) != 1 || !matches[0].CallstackMissing || len(matches[0].Callstack) != 0 {
				t.Fatalf("failed import injected frames: %+v", matches)
			}
		})
	}
}

type countingStackReader struct{ bytes int }

func (r *countingStackReader) Read(p []byte) (int, error) {
	for i := range p {
		p[i] = ' '
	}
	r.bytes += len(p)
	return len(p), nil
}

func TestTrackerReaderByteBound(t *testing.T) {
	r := new(countingStackReader)
	if err := cudacapture.NewCorrelationTracker().ReadExternalStackTable(r); err == nil {
		t.Fatal("accepted endless input")
	}
	if r.bytes != (16<<20)+1 {
		t.Fatalf("read %d bytes", r.bytes)
	}
}

func TestTrackerLossSaturation(t *testing.T) {
	ct := cudacapture.NewCorrelationTracker()
	for _, text := range []string{`{"version":1,"losses":18446744073709551615,"stacks":[]}`, `{"version":1,"losses":1,"stacks":[]}`} {
		if err := ct.ReadExternalStackTable(strings.NewReader(text)); err != nil {
			t.Fatal(err)
		}
	}
	if ct.ExternalStackLosses() != ^uint64(0) || !ct.ExternalStackLossesLowerBound() {
		t.Fatal("loss total wrapped or lost qualification")
	}
}

func TestTrackerReaderAdmissionBoundaries(t *testing.T) {
	for _, frames := range [][]string{{"f"}, strings.Fields(strings.Repeat("f ", 65)), {strings.Repeat("f", 4096), strings.Repeat("f", 4096)}} {
		if err := cudacapture.NewCorrelationTracker().ReadExternalStackTable(strings.NewReader(stackTable(t, 0, frames))); err != nil {
			t.Fatalf("refused valid frames: %v", err)
		}
	}
	row := `{"external_kind":3,"external_id":9,"frames":["f"]}`
	table := `{"version":1,"stacks":[` + strings.TrimSuffix(strings.Repeat(row+",", 16384), ",") + `]}`
	if err := cudacapture.NewCorrelationTracker().ReadExternalStackTable(strings.NewReader(table)); err != nil {
		t.Fatalf("refused row bound: %v", err)
	}
	data := `{"version":1,"losses":0,"stacks":[]}`
	data += strings.Repeat(" ", (16<<20)-len(data))
	if err := cudacapture.NewCorrelationTracker().ReadExternalStackTable(strings.NewReader(data)); err != nil {
		t.Fatalf("refused byte bound: %v", err)
	}
	epoch := strings.Repeat("x", 128)
	journal := fmt.Sprintf("{\"version\":1,\"kind\":\"header\",\"process\":222,\"epoch\":%q,\"losses\":\"0\"}\n{\"version\":1,\"kind\":\"stack\",\"process\":222,\"epoch\":%q,\"external_kind\":3,\"external_id\":9,\"frames\":[\"f\"]}\n", epoch, epoch)
	if err := cudacapture.NewCorrelationTracker().ReadExternalStackTable(strings.NewReader(journal)); err != nil {
		t.Fatalf("refused epoch bound: %v", err)
	}
}

func ExampleCorrelationTracker_ExternalStackLossesLowerBound() {
	tracker := cudacapture.NewCorrelationTracker()
	err := tracker.ReadExternalStackTable(bytes.NewBufferString("{\"version\":1,\"kind\":\"header\",\"losses\":\"1\",\"losses_lower_bound\":true}\n"))
	fmt.Println(err, tracker.ExternalStackLosses(), tracker.ExternalStackLossesLowerBound())
	// Output: <nil> 1 true
}

var _ io.Reader = (*countingStackReader)(nil)
