package main

import (
	"bytes"
	"context"
	"encoding/hex"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof"
)

var ptxLoc = regexp.MustCompile(`(?m)^(\t\.loc\t1 )(\d+)( .*)$`)

// shiftedPTX returns the fixture PTX with every source line moved by two, as a
// second compilation of a file of the same name would have. It is the same
// technique x/cudacapture's own tiering test uses: one tier is perturbed and
// the rest of the fixture is left alone, so the only thing that can fail is
// the binding between them.
//
// The perturbation is applied to the committed compiler output rather than
// written out as a second PTX. A PTX nobody compiled would differ from the
// fixture in its header, its registers and its instruction text as well as in
// the line table under test, so a loader that refused it would not have shown
// that it read the line table at all.
func shiftedPTX(t *testing.T) string {
	t.Helper()
	b, err := testFS.ReadFile("testdata/kernels.ptx")
	if err != nil {
		t.Fatalf("read kernels.ptx: %v", err)
	}
	moved := ptxLoc.ReplaceAllStringFunc(string(b), func(m string) string {
		g := ptxLoc.FindStringSubmatch(m)
		n, err := strconv.Atoi(g[2])
		if err != nil {
			t.Fatalf("unreadable .loc line %q", m)
		}
		return g[1] + strconv.Itoa(n+2) + g[3]
	})
	if moved == string(b) {
		t.Fatal("shiftedPTX changed nothing; the .loc spelling in the fixture has moved")
	}
	return moved
}

// fixtureCapture returns the committed kernels.* fixture with the PTX
// optionally replaced.
func fixtureCapture(t *testing.T, ptx string) *cudacapture.Capture {
	t.Helper()
	read := func(name string) []byte {
		b, err := testFS.ReadFile("testdata/" + name)
		if err != nil {
			t.Fatalf("read %s: %v", name, err)
		}
		return b
	}
	cubin, err := hex.DecodeString(strings.TrimSpace(string(read("kernels.cubin.hex"))))
	if err != nil {
		t.Fatalf("decode cubin hex: %v", err)
	}
	ptxBytes := read("kernels.ptx")
	if ptx != "" {
		ptxBytes = []byte(ptx)
	}
	return &cudacapture.Capture{
		Code: cudacapture.CodeFiles{Source: "kernels.cu", PTX: "kernels.ptx", CUBIN: "kernels.cubin", SASS: "kernels.sass"},
		Files: map[string][]byte{
			"kernels.cu":    read("kernels.cu"),
			"kernels.ptx":   ptxBytes,
			"kernels.cubin": cubin,
			"kernels.sass":  read("kernels.sass"),
		},
	}
}

// TestSourceFusionKeepsTheCapture is the failure half: a PTX and SASS that
// disagree about the line table must not take the capture down. Everything the
// bundle holds beyond source attribution -- the profile, the trace, the
// disassembly, the control flow graphs -- was being discarded over it.
func TestSourceFusionKeepsTheCapture(t *testing.T) {
	code, fusion, err := loadCaptureCode(fixtureCapture(t, shiftedPTX(t)))
	if err != nil {
		t.Fatalf("loadCaptureCode: %v", err)
	}
	if fusion == nil {
		t.Fatal("no fusion failure reported for a shifted PTX")
	}
	if code == nil {
		t.Fatal("loadCaptureCode withheld the kernel tier on a source-fusion failure")
	}
	k := code.Kernel("compute")
	if k == nil || len(k.Instructions) == 0 {
		t.Fatal("no kernel instructions; the capture holds nothing to write")
	}
	for _, inst := range k.Instructions {
		if inst.Text == "" {
			t.Errorf("compute+%#x: empty instruction text", inst.Offset)
		}
		if inst.SourceLine != 0 || inst.PTXLine != 0 {
			t.Errorf("compute+%#x: attribution %d/%d survived an unfused load", inst.Offset, inst.SourceLine, inst.PTXLine)
		}
	}

	// The reader of a capture sees the files, not this process's stderr, so
	// the note has to name the file the two tiers disagree about and repeat
	// the failure itself.
	note := sourceFusionNote(fusion)
	for _, want := range []string{"kernels.cu", "not from one compilation", "source attribution unavailable"} {
		if !strings.Contains(note, want) {
			t.Errorf("bundled note does not mention %q:\n%s", want, note)
		}
	}
}

// TestFusedCaptureKeepsAttribution is the near-miss half of the dissociation.
// A degrade that always fires passes the test above and loses source
// attribution on every capture, which is indistinguishable in the bundle from
// the failure it is meant to report.
func TestFusedCaptureKeepsAttribution(t *testing.T) {
	code, fusion, err := loadCaptureCode(fixtureCapture(t, ""))
	if err != nil {
		t.Fatalf("loadCaptureCode: %v", err)
	}
	if fusion != nil {
		t.Fatalf("matched fixture reported a fusion failure: %v", fusion)
	}
	attributed := 0
	for _, k := range code.Kernels {
		for _, inst := range k.Instructions {
			if inst.SourceLine > 0 && inst.PTXLine > 0 {
				attributed++
			}
		}
	}
	if attributed == 0 {
		t.Error("a matched pair produced no source attribution at all")
	}
}

// TestHotspotsWithoutSourceTier checks what the degrade puts in the profile.
// The kernel carries source and PTX lines of its own, and the instructions do
// not; falling back to the kernel's lines would put every instruction row on
// one source line, which pprof renders exactly like measured attribution.
func TestHotspotsWithoutSourceTier(t *testing.T) {
	// k is what LoadCode yields through a fusion failure: kernel-level lines
	// intact, per-instruction attribution zeroed.
	k := &cudacapture.Kernel{
		Name:       "test_kernel",
		Offset:     0xd900,
		PTXLine:    2,
		SourceLine: 40,
		Instructions: []cudacapture.Instruction{
			{Offset: 0x00, SASSLine: 10, Text: "/*0000*/ LDG.E R2, [R0];"},
			{Offset: 0x10, SASSLine: 11, Text: "/*0010*/ FFMA R3, R2, R1, R3;"},
		},
	}

	for _, tt := range []struct {
		name          string
		attribution   bool
		wantLocations int
	}{
		{name: "degraded", attribution: false, wantLocations: 1},
		{name: "attributed", attribution: true, wantLocations: 3},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			prof := cudaprof.NewSession(&buf)
			if err := prof.SetSampleTypes(sampleTypesWithHotspots(false, true)...); err != nil {
				t.Fatalf("SetSampleTypes: %v", err)
			}
			rec := newLaunchRecorder(prof, cudacapture.NewPTXAddresses("// aaaaaa\n// bbbbbb\n"))
			rec.SetSourceAttribution(tt.attribution)

			if got := recordInstructionHotspots(context.Background(), rec, k, 0, nil); got != len(k.Instructions) {
				t.Fatalf("recorded %d instructions, want %d", got, len(k.Instructions))
			}
			if err := prof.Close(); err != nil {
				t.Fatalf("Close profile: %v", err)
			}
			p, err := profile.ParseData(buf.Bytes())
			if err != nil {
				t.Fatalf("ParseData: %v", err)
			}
			if len(p.Sample) != len(k.Instructions) {
				t.Fatalf("got %d samples, want %d", len(p.Sample), len(k.Instructions))
			}
			for i, s := range p.Sample {
				if len(s.Location) != tt.wantLocations {
					t.Errorf("sample %d: %d locations, want %d", i, len(s.Location), tt.wantLocations)
				}
				for _, loc := range s.Location {
					for _, line := range loc.Line {
						name := line.Function.Filename
						if !tt.attribution && name != artifacts.SASS {
							t.Errorf("sample %d: frame in %s; only the SASS tier survives a fusion failure", i, name)
						}
						if line.Line == 0 {
							t.Errorf("sample %d: frame in %s at line 0", i, name)
						}
					}
				}
			}
		})
	}
}
