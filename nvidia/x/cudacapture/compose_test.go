package cudacapture_test

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// fixtureCapture builds a one-module capture from the testdata artifacts.
func fixtureCapture(t *testing.T) (*cudacapture.Capture, string) {
	t.Helper()
	read := func(name string) []byte {
		t.Helper()
		data, err := os.ReadFile(filepath.Join("testdata", name))
		if err != nil {
			t.Fatal(err)
		}
		return data
	}
	cubin, err := hex.DecodeString(strings.TrimSpace(string(read("kernels.cubin.hex"))))
	if err != nil {
		t.Fatalf("decode cubin: %v", err)
	}
	code := cudacapture.CodeFiles{Source: "kernels.cu", PTX: "kernels.ptx", CUBIN: "kernels.cubin", SASS: "kernels.sass"}
	c := &cudacapture.Capture{
		Files: map[string][]byte{
			"kernels.cu":    read("kernels.cu"),
			"kernels.ptx":   read("kernels.ptx"),
			"kernels.sass":  read("kernels.sass"),
			"kernels.cubin": cubin,
		},
		Code:    code,
		Modules: []cudacapture.Module{{ID: "m1", Generation: 1, Code: code}},
	}
	return c, fmt.Sprintf("%x", sha256.Sum256(cubin))
}

func TestVerifyModule(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(*cudacapture.Capture, *string)
		id      string
		kernels []string
		want    string
	}{
		{name: "match", id: "m1", kernels: []string{"compute"}},
		{name: "unknown module", id: "m2", want: `no module "m2"`},
		{name: "empty module", id: "", want: "empty module id"},
		{name: "duplicate module", id: "m1", mutate: func(c *cudacapture.Capture, _ *string) {
			c.Modules = append(c.Modules, c.Modules[0])
		}, want: "duplicate module"},
		{name: "other image", id: "m1", mutate: func(c *cudacapture.Capture, _ *string) {
			c.Files["kernels.cubin"][len(c.Files["kernels.cubin"])-1] ^= 0xff
		}, want: "sha256 is"},
		{name: "no digest", id: "m1", mutate: func(_ *cudacapture.Capture, digest *string) {
			*digest = ""
		}, want: "missing sha256 digest"},
		{name: "malformed digest", id: "m1", mutate: func(_ *cudacapture.Capture, digest *string) {
			*digest = "ABC"
		}, want: "malformed sha256 digest"},
		{name: "unretained artifact", id: "m1", mutate: func(c *cudacapture.Capture, _ *string) {
			delete(c.Files, "kernels.sass")
		}, want: `sass file "kernels.sass" not retained`},
		{name: "role not recorded", id: "m1", mutate: func(c *cudacapture.Capture, _ *string) {
			c.Modules[0].Code.PTX = ""
		}, want: "has no ptx artifact"},
		{name: "artifact path", id: "m1", mutate: func(c *cudacapture.Capture, _ *string) {
			c.Modules[0].Code.Source = "src/kernels.cu"
		}, want: "not a flat relative name"},
		{name: "kernel not defined", id: "m1", kernels: []string{"project"}, want: `does not define kernel "project"`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c, digest := fixtureCapture(t)
			if tt.mutate != nil {
				tt.mutate(c, &digest)
			}
			code, err := cudacapture.VerifyModule(c, tt.id, digest, tt.kernels)
			if tt.want == "" {
				if err != nil {
					t.Fatalf("VerifyModule: %v", err)
				}
				if code.Kernel("compute") == nil {
					t.Fatal("VerifyModule returned code without the module's kernel")
				}
				return
			}
			if err == nil {
				t.Fatalf("VerifyModule succeeded, want error containing %q", tt.want)
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("VerifyModule error = %v, want it to contain %q", err, tt.want)
			}
		})
	}
}

func TestVerifyModuleNilCapture(t *testing.T) {
	if _, err := cudacapture.VerifyModule(nil, "m1", strings.Repeat("0", 64), nil); err == nil {
		t.Fatal("VerifyModule(nil) succeeded, want error")
	}
}

func ExampleVerifyModule() {
	// A reader holding only a capture file checks that its code artifacts are
	// the image the producer loaded before trusting any line attribution.
	c := &cudacapture.Capture{
		Files: map[string][]byte{"k.cu": nil, "k.ptx": nil, "k.cubin": nil, "k.sass": nil},
		Modules: []cudacapture.Module{{
			ID:   "m1",
			Code: cudacapture.CodeFiles{Source: "k.cu", PTX: "k.ptx", CUBIN: "k.cubin", SASS: "k.sass"},
		}},
	}
	_, err := cudacapture.VerifyModule(c, "m1", "not-a-digest", nil)
	fmt.Println(err)
	// Output:
	// cudacapture: module "m1" image: malformed sha256 digest "not-a-digest"
}
