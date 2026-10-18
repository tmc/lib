//go:build darwin || linux

package hosttest

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// shimSource returns ../interpose.c with each edit applied. Every edit's old
// text must occur exactly once, so a mutant cannot silently apply to nothing.
func shimSource(t *testing.T, edits ...[2]string) string {
	t.Helper()
	b, err := os.ReadFile("../interpose.c")
	if err != nil {
		t.Fatal(err)
	}
	source := string(b)
	for _, e := range edits {
		if n := strings.Count(source, e[0]); n != 1 {
			t.Fatalf("mutant precondition: %q occurs %d times, want 1", e[0], n)
		}
		source = strings.Replace(source, e[0], e[1], 1)
	}
	return source
}

// buildFixture compiles testdata/<fixture>.c against shim with warnings as
// errors. The shim is written as interpose-source.c next to copies of the
// package headers, matching how the fixtures include it.
func buildFixture(t *testing.T, fixture, shim string) string {
	t.Helper()
	cc, err := exec.LookPath("cc")
	if err != nil {
		t.Fatal("host fixture requires a C compiler:", err)
	}
	version, err := exec.Command(cc, "--version").CombinedOutput()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("compiler=%s version=%s", cc, version)
	dir := t.TempDir()
	headers, err := filepath.Glob("../*.h")
	if err != nil {
		t.Fatal(err)
	}
	for _, h := range headers {
		b, err := os.ReadFile(h)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, filepath.Base(h)), b, 0600); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(dir, "interpose-source.c"), []byte(shim), 0600); err != nil {
		t.Fatal(err)
	}
	src, err := os.ReadFile(filepath.Join("testdata", fixture+".c"))
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, fixture+".c")
	if err := os.WriteFile(path, src, 0600); err != nil {
		t.Fatal(err)
	}
	binary := filepath.Join(dir, fixture)
	args := []string{"-std=c11", "-pthread", "-Wall", "-Wextra", "-Werror", path, "-o", binary}
	t.Logf("compile: %s %v", cc, args)
	if out, err := exec.Command(cc, args...).CombinedOutput(); err != nil {
		t.Fatalf("compile: %v\n%s", err, out)
	}
	return binary
}

// runIdentity runs the identity fixture with CUDAINTERCEPT_OBSERVE set to
// observe. The fixture launches once with eleven pairwise-distinct arguments,
// has the real cuLaunchKernel return result, and drains one record.
func runIdentity(t *testing.T, binary, observe, result string) string {
	t.Helper()
	t.Setenv("CUDAINTERCEPT_OBSERVE", observe)
	timedOut, err, out, elapsed := runFixture(t, binary, result)
	t.Logf("elapsed=%s timeout=%v result=%v output=%s", elapsed, timedOut, err, out)
	if timedOut || err != nil {
		t.Fatal("identity fixture did not exit cleanly")
	}
	return strings.TrimSpace(out)
}

func TestIdentityForwarding(t *testing.T) {
	binary := buildFixture(t, "identity", shimSource(t))
	for _, observe := range []struct{ value, drain string }{{"", "-1"}, {"1", "1"}} {
		// -1 and 0x7fffffff are outside the CUresult enum but must still
		// pass through, and be recorded, as a full int.
		for _, result := range []string{"0", "1", "700", "-1", "2147483647"} {
			t.Run("observe_"+observe.value+"/result_"+result, func(t *testing.T) {
				want := "result=" + result + " calls=1 drain=" + observe.drain + " mismatch=none"
				if got := runIdentity(t, binary, observe.value, result); got != want {
					t.Fatalf("got %q, want %q", got, want)
				}
			})
		}
	}
}

func TestIdentityMutants(t *testing.T) {
	for _, tt := range []struct {
		name     string
		old, new string
		observe  string
		result   string
		want     string
	}{
		{
			name:   "swap_block_y_z",
			old:    "blockDimY, blockDimZ, sharedMemBytes, hStream",
			new:    "blockDimZ, blockDimY, sharedMemBytes, hStream",
			result: "700",
			want:   "result=700 calls=1 drain=-1 mismatch=blockY blockZ",
		},
		{
			name:   "drop_extra",
			old:    "kernelParams, extra);",
			new:    "kernelParams, ((void)extra, NULL));",
			result: "700",
			want:   "result=700 calls=1 drain=-1 mismatch=extra",
		},
		{
			name:    "record_result_int16",
			old:     "r->result = result;",
			new:     "r->result = (int16_t)result;",
			observe: "1",
			result:  "2147483647",
			want:    "result=2147483647 calls=1 drain=1 mismatch=record result",
		},
		{
			name:    "record_swap_block_y_z",
			old:     "r->block[1] = blockDimY;\n    r->block[2] = blockDimZ;",
			new:     "r->block[1] = blockDimZ;\n    r->block[2] = blockDimY;",
			observe: "1",
			result:  "0",
			want:    "result=0 calls=1 drain=1 mismatch=record block",
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			binary := buildFixture(t, "identity", shimSource(t, [2]string{tt.old, tt.new}))
			if got := runIdentity(t, binary, tt.observe, tt.result); got != tt.want {
				t.Fatalf("got %q, want %q", got, tt.want)
			}
		})
	}
}
