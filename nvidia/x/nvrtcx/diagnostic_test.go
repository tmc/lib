package nvrtcx

import (
	"errors"
	"strings"
	"testing"
)

func TestCompilerDiagnostic(t *testing.T) {
	compileErr, logErr := errors.New("compile failure"), errors.New("log failure")
	for _, tt := range []struct {
		name         string
		compile, log error
	}{
		{"success", nil, nil},
		{"compile", compileErr, nil},
		{"log", nil, logErr},
		{"both", compileErr, logErr},
	} {
		t.Run(tt.name, func(t *testing.T) {
			err := compilerDiagnostic(tt.compile, tt.log, "line 7: invalid token")
			if errors.Is(err, compileErr) != (tt.compile != nil) || errors.Is(err, logErr) != (tt.log != nil) {
				t.Fatalf("lost diagnostic cause: %v", err)
			}
			if tt.compile == nil && tt.log == nil && err != nil {
				t.Fatal(err)
			}
			if tt.compile != nil && !strings.Contains(err.Error(), "line 7: invalid token") {
				t.Fatal("lost compiler log")
			}
		})
	}
}
