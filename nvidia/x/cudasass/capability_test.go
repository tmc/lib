package cudasass

import (
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/x/internal/cudatool"
)

// recordedCaps returns what the shipped option corpus records for nvdisasm at a
// toolkit version, so the check every Disassemble makes can be exercised for
// toolkits this host does not have.
func recordedCaps(t *testing.T, v cudatool.Version) cudatool.Caps {
	t.Helper()
	opts, source, ok := cudatool.KnownOptions(ToolName, v)
	if !ok {
		t.Fatalf("no recorded option set for %s %v", ToolName, v)
	}
	return cudatool.Caps{Tool: ToolName, Version: v, Source: source, Options: opts}
}

// TestOptionsAgainstRecordedToolkits checks every command line this package can
// emit against every nvdisasm the corpus records.
//
// Mode JSON is the one option that moved: --emit-json arrived in 13.0, so a
// request for it under 12.6 must be reported as unsupported rather than handed
// to a tool that answers "Unknown option". Everything else is accepted by every
// recorded version, including the arities: --life-range-mode takes an argument
// on all of them, and a record saying otherwise fails here.
func TestOptionsAgainstRecordedToolkits(t *testing.T) {
	tests := []struct {
		name   string
		o      Options
		reject []cudatool.Version
	}{
		{name: "text", o: Options{}},
		{name: "json", o: Options{Mode: JSON}, reject: []cudatool.Version{cudatool.V(12, 6)}},
		{name: "control flow", o: Options{Mode: ControlFlow, PrintInstructionOffsetsCFG: true}},
		{name: "basic block flow", o: Options{Mode: BasicBlockFlow, PrintInstructionOffsetsCFG: true}},
		{name: "register life", o: Options{Mode: RegisterLife, LifeRangeMode: "count", PrintCode: true}},
		{name: "line info", o: Options{SourceLines: true, PTXLines: true}},
		{name: "inline line info", o: Options{InlineLines: true, PrintCode: true, PrintHex: true}},
		{name: "separate functions", o: Options{SeparateFunctions: true}},
	}
	versions := cudatool.KnownVersions(ToolName)
	if len(versions) == 0 {
		t.Fatalf("no recorded toolkit versions for %s", ToolName)
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			flags, err := tt.o.flags()
			if err != nil {
				t.Fatalf("flags: %v", err)
			}
			for _, v := range versions {
				err := recordedCaps(t, v).Check(flags...)
				reject := false
				for _, r := range tt.reject {
					reject = reject || r.Compare(v) == 0
				}
				switch {
				case reject && err == nil:
					t.Errorf("%v accepted %q, want it reported as unsupported", v, cudatool.Args(flags...))
				case reject && !errors.Is(err, cudatool.ErrUnsupportedOption):
					t.Errorf("%v: error is %v, want it to wrap ErrUnsupportedOption", v, err)
				case !reject && err != nil:
					t.Errorf("%v rejected %q: %v", v, cudatool.Args(flags...), err)
				}
			}
		})
	}
}

// TestEmitJSONArrivedIn130 states the measurement the JSON mode gate rests on,
// apart from the command lines that depend on it.
func TestEmitJSONArrivedIn130(t *testing.T) {
	tests := []struct {
		version cudatool.Version
		accepts bool
	}{
		{cudatool.V(12, 6), false},
		{cudatool.V(13, 0), true},
		{cudatool.V(13, 4), true},
	}
	for _, tt := range tests {
		t.Run(tt.version.String(), func(t *testing.T) {
			if got := recordedCaps(t, tt.version).Accepts("--emit-json"); got != tt.accepts {
				t.Errorf("nvdisasm %v accepts --emit-json = %v, want %v", tt.version, got, tt.accepts)
			}
		})
	}
}
