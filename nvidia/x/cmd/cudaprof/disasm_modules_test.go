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

func TestRetainedModuleDisasm(t *testing.T) {
	raw, err := os.ReadFile("testdata/kernels.cubin.hex")
	if err != nil {
		t.Fatal(err)
	}
	cubin, err := hex.DecodeString(strings.Join(strings.Fields(string(raw)), ""))
	if err != nil {
		t.Fatal(err)
	}
	sass, err := os.ReadFile("testdata/kernels.sass")
	if err != nil {
		t.Fatal(err)
	}
	ptx, err := os.ReadFile("testdata/kernels.ptx")
	if err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		name     string
		listing  bool
		missing  bool
		primary  bool
		complete bool
	}{
		{name: "module listings", listing: true},
		{name: "module cubins"},
		{name: "missing artifact", missing: true},
		{name: "partial primary", listing: true, primary: true},
		{name: "complete primary and module", listing: true, primary: true, complete: true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			// No CUDA or GPU is needed: check the CLI's nvdisasm fallback using a
			// valid ELF fixture and a deterministic disassembler process.
			bin := filepath.Join(dir, "bin")
			if err := os.Mkdir(bin, 0700); err != nil {
				t.Fatal(err)
			}
			toolPath := filepath.Join(bin, "nvdisasm")
			script := "#!/bin/sh\ncase \"$1\" in --version|--help) exit 1;; esac\ncat <<'SASS'\n" + string(sass) + "\nSASS\n"
			if err := os.WriteFile(toolPath, []byte(script), 0700); err != nil {
				t.Fatal(err)
			}
			t.Setenv("CUDA_HOME", dir)
			fn := &profile.Function{ID: 1, Name: "compute"}
			loc := &profile.Location{ID: 1, Line: []profile.Line{{Function: fn}}}
			p := &profile.Profile{SampleType: []*profile.ValueType{{Type: "gpu_time", Unit: "nanoseconds"}}, Function: []*profile.Function{fn}, Location: []*profile.Location{loc}, Sample: []*profile.Sample{{Location: []*profile.Location{loc}, Value: []int64{73}}}}
			c := &cudacapture.Capture{Profile: p, Files: map[string][]byte{}, CodeUnavailable: map[string]string{"source": "not retained"}}
			for _, name := range []string{"first", "second"} {
				roles := cudacapture.CodeFiles{CUBIN: name + ".cubin", PTX: name + ".ptx"}
				c.Files[roles.CUBIN] = cubin
				c.Files[roles.PTX] = ptx
				if tt.listing {
					roles.SASS = name + ".sass"
					c.Files[roles.SASS] = sass
				}
				c.Modules = append(c.Modules, cudacapture.Module{ID: name, Code: roles})
			}
			if tt.primary {
				c.Code = c.Modules[0].Code
				c.Modules = c.Modules[1:]
			}
			if tt.missing {
				delete(c.Files, "second.cubin")
			}
			before, err := json.Marshal(p)
			if err != nil {
				t.Fatal(err)
			}
			code := &cudacapture.Code{}
			if tt.complete {
				source, err := os.ReadFile("testdata/kernels.cu")
				if err != nil {
					t.Fatal(err)
				}
				c.Code.Source = "kernels.cu"
				c.Files[c.Code.Source] = source
				code, err = cudacapture.LoadCode(c)
				if err != nil {
					t.Fatal(err)
				}
			}
			tool, err := newObjectTool(c, code, dir)
			if err != nil {
				t.Fatal(err)
			}
			var out bytes.Buffer
			handled, err := retainedDisasm(&out, []string{"-disasm=compute", "capture"}, p, tool)
			if !handled {
				t.Fatal("did not handle retained code")
			}
			if tt.missing {
				if err == nil || !strings.Contains(err.Error(), "second.cubin: missing artifact") {
					t.Fatalf("error=%v", err)
				}
				if out.Len() != 0 {
					t.Fatal("printed incomplete disassembly")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			for _, want := range []string{"unannotated", "No instruction costs", "first.cubin", "second.cubin", "first.ptx", "second.ptx", "0x60", "MOV R1"} {
				if !strings.Contains(out.String(), want) {
					t.Fatalf("missing %q in %s", want, &out)
				}
			}
			if strings.Count(out.String(), "compute.SASS\n") != 2 {
				t.Fatalf("lost or duplicated a module: %s", &out)
			}
			// Repeating the lookup must reuse images, not duplicate the functions.
			first := out.String()
			out.Reset()
			if _, err := retainedDisasm(&out, []string{"-disasm=compute", "capture"}, p, tool); err != nil {
				t.Fatal(err)
			}
			if out.String() != first {
				t.Fatal("repeated disassembly changed output")
			}
			after, err := json.Marshal(p)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(before, after) {
				t.Fatal("disassembly mutated profile")
			}
		})
	}
}
