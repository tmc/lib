package main

import (
	"encoding/json"
	"maps"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof/advise"
)

// sampleRemarksYAML is the tileiras remarks fixture advise's own tests use,
// carried here so the CLI has a TMA failure to report.
const sampleRemarksYAML = `--- !Passed
Pass:            Memory
Name:            RemarkMemoryLoadInstructionSelected
DebugLoc:        { File: '/tmp/tile_matrix.cu', Line: 10, Column: 37 }
Function:        matrix
Args:
  - RemarkId:    4
  - Remark:      Load instruction selected
  - Instruction: TMA Load instruction
...
--- !Failure
Pass:            Memory
Name:            RemarkMemoryTmaLoadFailed
DebugLoc:        { File: '/tmp/tile_matrix.cu', Line: 14, Column: 5 }
Function:        matrix
Args:
  - RemarkId:    5
  - Remark:      Failed to select TMA load instruction
  - Reason:      non-contiguous stride along dim 1
  - Shape:       '[16, 32]'
...
--- !Failure
Pass:            Tensor-core
Name:            RemarkTensorCoreMMA
DebugLoc:        { File: '/tmp/tile_matrix.cu', Line: 16, Column: 9 }
Function:        matrix
Args:
  - RemarkId:    3
  - Remark:      MMA operation failed to optimize to use Tensor Cores, it is using FMA instructions instead
  - Instruction: FMA
  - Shape:       '[1, 1, 1]'
...
`

func TestAdvice_CLI(t *testing.T) {
	cap := &cudacapture.Capture{
		Profile: &profile.Profile{
			SampleType: []*profile.ValueType{{Type: "gpu_time", Unit: "nanoseconds"}},
		},
		Code: cudacapture.CodeFiles{
			Source: "kernels.cu",
			PTX:    "kernels.ptx",
			CUBIN:  "kernels.cubin",
			SASS:   "kernels.sass",
		},
		Files: map[string][]byte{
			"kernels.cu":    []byte("void matrix() {}\n"),
			"kernels.ptx":   []byte("// PTX\n"),
			"kernels.cubin": []byte("CUBIN_MOCK"),
			"kernels.sass":  []byte(".section .text.matrix\n/*0000*/ EXIT;\n"),
			"remarks.yaml":  []byte(sampleRemarksYAML),
		},
	}

	tmpDir := t.TempDir()
	capPath := filepath.Join(tmpDir, "test.pprof")
	f, err := os.Create(capPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := cudacapture.Write(f, cap); err != nil {
		t.Fatal(err)
	}
	f.Close()

	// 1. Test 'cudaprof advise capture.pprof'
	out1 := captureStdout(t, func() error { return runAdviseCommand([]string{capPath}) })
	if !strings.Contains(out1, "CUDA Kernel Advice Report") {
		t.Errorf("advise output missing report header: %s", out1)
	}
	if !strings.Contains(out1, "TMA") {
		t.Errorf("advise output missing TMA finding: %s", out1)
	}

	// 2. Test 'cudaprof advise --format=json capture.pprof'
	out2 := captureStdout(t, func() error { return runAdviseCommand([]string{"--format=json", capPath}) })
	var reportJSON advise.Report
	if err := json.NewDecoder(strings.NewReader(out2)).Decode(&reportJSON); err != nil {
		t.Fatalf("decode advise json failed: %v", err)
	}
	if len(reportJSON.Items) == 0 {
		t.Error("expected non-empty items in JSON output")
	}

	// 3. Test 'cudaprof inspect --advise capture.pprof'
	out3 := captureStdout(t, func() error { return runInspectCommand([]string{"--advise", capPath}) })
	var inspectAdv advise.Report
	if err := json.NewDecoder(strings.NewReader(out3)).Decode(&inspectAdv); err != nil {
		t.Fatalf("decode inspect --advise failed: %v", err)
	}
	if len(inspectAdv.Items) == 0 {
		t.Error("expected non-empty items in inspect --advise output")
	}
}

// TestFixtureSampleTypesAreReal is the sweep behind TestStallFixtureNamesAreReal
// in diff_test.go. That test names four types and carries the negative control;
// this one finds every sample type any fixture in this package or in
// x/cudaprof/advise declares and checks them all, so a fixture added later
// cannot invent a counter and quietly take its narrative with it. The names
// come out of the test sources rather than a list here, because a list here
// would be a third copy of the same assumption, and the oracle is
// sampleTypeIsReal so there is only one.
func TestFixtureSampleTypesAreReal(t *testing.T) {
	files, err := filepath.Glob("*_test.go")
	if err != nil {
		t.Fatal(err)
	}
	adviseFiles, err := filepath.Glob("../../cudaprof/advise/*_test.go")
	if err != nil {
		t.Fatal(err)
	}
	if len(adviseFiles) == 0 {
		t.Fatal("found no advise test files; the sweep would skip the analysis fixtures")
	}
	files = append(files, adviseFiles...)
	re := regexp.MustCompile(`\{Type:\s*"([a-z0-9_]+)"`)
	seen := make(map[string]map[string]bool)
	for _, f := range files {
		src, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		for _, m := range re.FindAllSubmatch(src, -1) {
			name := string(m[1])
			if seen[name] == nil {
				seen[name] = make(map[string]bool)
			}
			seen[name][f] = true
		}
	}
	if len(seen) == 0 {
		t.Fatal("found no sample types in the package fixtures; the pattern no longer matches anything")
	}
	for name, files := range seen {
		if sampleTypeIsReal(name) {
			continue
		}
		where := slices.Sorted(maps.Keys(files))
		t.Errorf("fixture sample type %q (%v) is emitted by no producer this command reads", name, where)
	}
}
