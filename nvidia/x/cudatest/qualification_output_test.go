package cudatest

import (
	"encoding/json"
	"testing"
)

func TestNativeQualification_Go127OutputFraming(t *testing.T) {
	for _, tc := range []struct {
		name, action string
		typeValue    any
		terminal     string
		admit        bool
	}{
		{"frame", "output", "frame", "pass", true},
		{"error", "output", "error", "pass", true},
		{"error-continue", "output", "error-continue", "pass", true},
		{"legacy-empty", "output", "", "pass", true},
		{"unknown", "output", "result", "pass", false},
		{"wrong-type", "output", 1, "pass", false},
		{"non-output", "pass", "frame", "pass", false},
		{"framing-does-not-hide-skip", "output", "frame", "skip", false},
		{"framing-does-not-hide-fail", "output", "frame", "fail", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := qualificationSyntheticFixture(t)
			f.changeRun(t, 0, func(r *QualificationRun) {
				log := qualificationTestLog(t, r.Package, r.Test, tc.terminal)
				annotation, err := json.Marshal(map[string]any{"Action": tc.action, "Package": r.Package, "Test": r.Test, "Output": "framed test output\n", "OutputType": tc.typeValue})
				if err != nil {
					t.Fatal(err)
				}
				r.Log = qualificationTestFile(f.files, "copy/go-test.json", append(append(annotation, '\n'), log...))
			})
			err := ValidateQualification(f.matrix, f.matrix.SourceRevision, f.files, f.files)
			if (err == nil) != tc.admit {
				t.Fatalf("admitted=%t want%t: %v", err == nil, tc.admit, err)
			}
		})
	}
}
