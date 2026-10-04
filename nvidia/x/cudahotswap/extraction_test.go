package cudahotswap

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
)

func TestReportFailure(t *testing.T) {
	r := &Report{Task: "retained", Unmeasured: []string{"timing"}}
	if got := r.Fail(nil); got != r || r.Outcome != "" {
		t.Fatal("nil failure changed report")
	}
	if got := r.Fail(driver.ErrIllegalAddress); got != r || r.Task != "retained" || !r.ContextPoisoned || !r.Recognized || r.Outcome != OutcomeDeviceFault || r.Reason == "" {
		t.Fatalf("failure = %+v", r)
	}
	if !reflect.DeepEqual(r.Unmeasured, []string{"timing"}) {
		t.Fatal("failure changed unmeasured evidence")
	}
}

func TestABIMismatchDistinct(t *testing.T) {
	if OutcomeABIMismatch == OutcomeLoadError || OutcomeABIMismatch == OutcomeWrongAnswer || OutcomeABIMismatch == OutcomeOK {
		t.Fatal("ABI mismatch collapsed into another outcome")
	}
	if OutcomeABIMismatch.PoisonsContext() || OutcomeABIMismatch.ExitCode() != 1 {
		t.Fatal("ABI refusal is not a non-poisoning candidate failure")
	}
}

func TestReportPreservesJSONNames(t *testing.T) {
	r := Report{Task: "task", TaskDef: "definition", Candidate: "source", Entry: "entry", Outcome: OutcomeOK, Recognized: true, Unmeasured: []string{"timing"}}
	b, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	want := `{"task":"task","task_definition":"definition","candidate":"source","entry":"entry","outcome":"ok","context_poisoned":false,"error_recognized":true,"unmeasured":["timing"]}`
	if string(b) != want {
		t.Fatalf("JSON = %s, want %s", b, want)
	}
}
