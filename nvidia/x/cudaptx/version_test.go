package cudaptx_test

import (
	"errors"
	"os"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaptx"
)

// ptxModule wraps body in a module header. The body is placed inside a kernel
// so that its instructions are audited.
func ptxModule(version, target, body string) string {
	var b strings.Builder
	if version != "" {
		b.WriteString(".version " + version + "\n")
	}
	if target != "" {
		b.WriteString(".target " + target + "\n")
	}
	b.WriteString(".address_size 64\n\n")
	b.WriteString(".visible .entry k()\n{\n" + body + "\n\tret;\n}\n")
	return b.String()
}

func TestAuditVersion(t *testing.T) {
	tests := []struct {
		name    string
		version string
		target  string
		body    string
		want    string // substring of the error, or "" for success
	}{{
		// add is "Introduced in PTX ISA version 1.0" and needs nothing.
		name:    "ok",
		version: "8.8",
		target:  "sm_90",
		body:    "\tadd.s32 %r1, %r2, %r3;",
	}, {
		// A .target carrying extension targets past the architecture.
		name:    "target extensions",
		version: "8.8",
		target:  "sm_90, debug",
		body:    "\tadd.s32 %r1, %r2, %r3;",
	}, {
		name:    "target feature suffix",
		version: "8.8",
		target:  "sm_121a",
		body:    "\tadd.s32 %r1, %r2, %r3;",
	}, {
		name:    "missing version",
		version: "",
		target:  "sm_90",
		want:    "no .version directive",
	}, {
		name:    "malformed version",
		version: "8",
		target:  "sm_90",
		want:    "want major.minor",
	}, {
		name:    "non-numeric version",
		version: "eight.eight",
		target:  "sm_90",
		want:    "bad major number",
	}, {
		name:    "non-numeric minor",
		version: "8.x",
		target:  "sm_90",
		want:    "bad minor number",
	}, {
		name:    "missing target",
		version: "8.8",
		target:  "",
		want:    "no .target directive",
	}, {
		name:    "malformed target",
		version: "8.8",
		target:  "compute_90",
		want:    "malformed architecture",
	}, {
		// A feature suffix is letters only, so "a1" is not one.
		name:    "target with malformed suffix",
		version: "8.8",
		target:  "sm_90a1",
		want:    "malformed architecture",
	}, {
		name:    "target without digits",
		version: "8.8",
		target:  "sm_hopper",
		want:    "malformed architecture",
	}, {
		// tcgen05 was introduced in PTX ISA 8.6, so a module declaring
		// 8.0 cannot carry one.
		name:    "instruction too new",
		version: "8.0",
		target:  "sm_100a",
		body:    "\ttcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%r1], 32;",
		want:    "requires PTX ISA 8.6, module declares 8.0",
	}, {
		// The same instruction under the .version that introduced it.
		name:    "instruction at introduction",
		version: "8.6",
		target:  "sm_100a",
		body:    "\ttcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%r1], 32;",
	}, {
		name:    "version past tables",
		version: "99.0",
		target:  "sm_90",
		want:    "tables describe PTX ISA",
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m, err := cudaptx.ParseString(ptxModule(tt.version, tt.target, tt.body))
			if err != nil {
				t.Fatalf("ParseString: %v", err)
			}
			err = m.AuditVersion("")
			if tt.want == "" {
				if err != nil {
					t.Fatalf("AuditVersion() = %v, want nil", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("AuditVersion() = nil, want error containing %q", tt.want)
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("AuditVersion() = %v, want error containing %q", err, tt.want)
			}
		})
	}
}

// TestAuditVersionUnaudited checks that a .version past the generated tables is
// an explicit unknown rather than a pass.
func TestAuditVersionUnaudited(t *testing.T) {
	m, err := cudaptx.ParseString(ptxModule("99.0", "sm_90", "\tadd.s32 %r1, %r2, %r3;"))
	if err != nil {
		t.Fatalf("ParseString: %v", err)
	}
	err = m.AuditVersion("")
	if !errors.Is(err, cudaptx.ErrVersionUnaudited) {
		t.Fatalf("AuditVersion() = %v, want ErrVersionUnaudited", err)
	}
	if !strings.Contains(err.Error(), cudaptx.PTXISAVersion) {
		t.Errorf("AuditVersion() = %v, want the error to name PTX ISA %s", err, cudaptx.PTXISAVersion)
	}

	// The tables' own version is audited, not deferred.
	m, err = cudaptx.ParseString(ptxModule(cudaptx.PTXISAVersion, "sm_90", "\tadd.s32 %r1, %r2, %r3;"))
	if err != nil {
		t.Fatalf("ParseString: %v", err)
	}
	if err := m.AuditVersion(""); err != nil {
		t.Errorf("AuditVersion() at PTXISAVersion = %v, want nil", err)
	}
}

// TestAuditVersionKernel checks kernel selection: naming the clean kernel must
// not report the defect in its neighbour, and an unknown name is an error.
func TestAuditVersionKernel(t *testing.T) {
	src := ".version 8.0\n.target sm_100a\n.address_size 64\n\n" +
		".visible .entry clean()\n{\n\tadd.s32 %r1, %r2, %r3;\n\tret;\n}\n" +
		".visible .entry dirty()\n{\n\ttcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%r1], 32;\n\tret;\n}\n"
	m, err := cudaptx.ParseString(src)
	if err != nil {
		t.Fatalf("ParseString: %v", err)
	}
	if err := m.AuditVersion("clean"); err != nil {
		t.Errorf("AuditVersion(clean) = %v, want nil", err)
	}
	err = m.AuditVersion("dirty")
	if err == nil || !strings.Contains(err.Error(), "requires PTX ISA 8.6") {
		t.Errorf("AuditVersion(dirty) = %v, want the 8.6 requirement", err)
	}
	if err := m.AuditVersion("missing"); err == nil || !strings.Contains(err.Error(), "not found") {
		t.Errorf("AuditVersion(missing) = %v, want a not-found error", err)
	}
	// The whole-module form reaches the defect in dirty.
	if err := m.AuditVersion(""); err == nil {
		t.Error("AuditVersion() = nil, want the defect in dirty")
	}
}

// TestAuditVersionFixtures audits the nvcc fixtures, which declare .version 9.0
// against tables generated from a later ISA.
func TestAuditVersionFixtures(t *testing.T) {
	files, err := os.ReadDir("testdata")
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range files {
		if !strings.HasSuffix(f.Name(), ".ptx") {
			continue
		}
		t.Run(f.Name(), func(t *testing.T) {
			b, err := os.ReadFile("testdata/" + f.Name())
			if err != nil {
				t.Fatal(err)
			}
			m, err := cudaptx.ParseBytes(b)
			if err != nil {
				t.Fatalf("ParseBytes: %v", err)
			}
			if err := m.AuditVersion(""); err != nil {
				t.Errorf("AuditVersion() = %v, want nil", err)
			}
		})
	}
}

func TestExplainLoadError(t *testing.T) {
	m, err := cudaptx.ParseString(ptxModule("8.8", "sm_121a", "\tadd.s32 %r1, %r2, %r3;"))
	if err != nil {
		t.Fatalf("ParseString: %v", err)
	}
	load := errors.New("CUDA_ERROR_UNSUPPORTED_PTX_VERSION")
	got := m.ExplainLoadError(load)
	if !errors.Is(got, load) {
		t.Errorf("ExplainLoadError() = %v, want it to wrap the driver error", got)
	}
	for _, want := range []string{"8.8", "sm_121a", "LD_LIBRARY_PATH"} {
		if !strings.Contains(got.Error(), want) {
			t.Errorf("ExplainLoadError() = %v, want it to mention %q", got, want)
		}
	}
	if got := m.ExplainLoadError(nil); got != nil {
		t.Errorf("ExplainLoadError(nil) = %v, want nil", got)
	}

	// A module whose header never parsed still names what it has.
	empty := &cudaptx.Module{}
	if got := empty.ExplainLoadError(load).Error(); !strings.Contains(got, "<none>") {
		t.Errorf("ExplainLoadError() on an empty module = %v, want <none> placeholders", got)
	}
}

func TestAuditVersionNilModule(t *testing.T) {
	var m *cudaptx.Module
	if err := m.AuditVersion(""); err == nil {
		t.Error("AuditVersion() on a nil module = nil, want an error")
	}
}
