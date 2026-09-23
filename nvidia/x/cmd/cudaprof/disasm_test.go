package main

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
)

func TestRetainedDisasmRegionCounters(t *testing.T) {
	dir := "testdata"
	files := map[string][]byte{}
	for _, name := range []string{"kernels.cu", "kernels.ptx", "kernels.sass", "kernels.cubin.hex"} {
		data, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		if strings.HasSuffix(name, ".hex") {
			data, err = hex.DecodeString(strings.Join(strings.Fields(string(data)), ""))
			if err != nil {
				t.Fatal(err)
			}
			name = strings.TrimSuffix(name, ".hex")
		}
		files[name] = data
	}
	fn := &profile.Function{ID: 1, Name: "semantic-region"}
	loc := &profile.Location{ID: 1, Line: []profile.Line{{Function: fn}}}
	p := &profile.Profile{SampleType: []*profile.ValueType{{Type: "sm__ctas_launched.sum", Unit: "cta"}}, Function: []*profile.Function{fn}, Location: []*profile.Location{loc}, Sample: []*profile.Sample{{Location: []*profile.Location{loc}, Value: []int64{8}}}}
	before, err := json.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	cap := &cudacapture.Capture{Profile: p, Files: files, Code: cudacapture.CodeFiles{Source: "kernels.cu", PTX: "kernels.ptx", CUBIN: "kernels.cubin", SASS: "kernels.sass"}}
	code, err := cudacapture.LoadCode(cap)
	if err != nil {
		t.Fatal(err)
	}
	tool, err := newObjectTool(cap, code, t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	name := code.Kernels[0].Name
	var out bytes.Buffer
	handled, err := retainedDisasm(&out, []string{"-disasm=" + name, "capture"}, p, tool)
	if !handled || err != nil {
		t.Fatalf("handled=%v err=%v", handled, err)
	}
	for _, want := range []string{"unannotated", name + ".PTX", name + ".SASS", "No instruction costs"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("missing %q: %s", want, out.String())
		}
	}
	after, err := json.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(after, before) {
		t.Fatal("navigation changed profile counters, addresses, or frames")
	}
	out.Reset()
	if handled, err := retainedDisasm(&out, []string{"-disasm=absent_symbol", "capture"}, p, tool); !handled || err == nil || out.Len() != 0 {
		t.Fatalf("missing symbol handled=%v err=%v output=%s", handled, err, out.String())
	}
	if handled, err := retainedDisasm(&out, []string{"-disasm=" + name, "-focus=other", "capture"}, p, tool); !handled || err == nil {
		t.Fatal("silently ignored pprof filtering")
	}
	f := &fetcher{path: "capture", cap: cap}
	f.path, err = filepath.Abs(f.path)
	if err != nil {
		t.Fatal(err)
	}
	view, _, err := f.Fetch(f.path, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	comments := strings.Join(view.Comments, "\n")
	if strings.Contains(comments, "CUDA event intervals measure") || !strings.Contains(comments, "Profile values describe recorded scopes") {
		t.Fatalf("incorrect scope description: %s", comments)
	}
	capturePath := filepath.Join(t.TempDir(), "capture")
	if err := cudacapture.WriteDir(capturePath, cap); err != nil {
		t.Fatal(err)
	}
	oldArgs := os.Args
	t.Cleanup(func() { os.Args = oldArgs })
	os.Args = []string{"cudaprof", "-disasm=" + name, capturePath}
	text := captureStdout(t, run)
	if !strings.Contains(text, name+".PTX") || !strings.Contains(text, name+".SASS") || !strings.Contains(text, "unannotated") {
		t.Fatalf("CLI did not render retained code: %s", text)
	}
	loc.Address = 0x1234
	loc.Mapping = &profile.Mapping{ID: 1, Start: 0x1000, Limit: 0x2000}
	if handled, err := retainedDisasm(&out, []string{"-disasm=" + name, "capture"}, p, tool); handled || err != nil {
		t.Fatal("intercepted addressed pprof report")
	}
}
