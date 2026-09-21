//go:build darwin || linux

package nvtx

import (
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/ebitengine/purego"
)

func TestOpenDynamicLibrary(t *testing.T) {
	cc, err := exec.LookPath("cc")
	if err != nil {
		t.Skip("native loader test requires cc")
	}
	for _, test := range []struct {
		name       string
		unresolved bool
	}{
		{"valid", false},
		{"unresolved", true},
	} {
		t.Run(test.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "provider.so")
			args := []string{"-shared", "-fPIC"}
			if runtime.GOOS == "darwin" {
				args = []string{"-dynamiclib", "-Wl,-undefined,dynamic_lookup"}
			}
			if test.unresolved {
				args = append(args, "-DUNRESOLVED")
			}
			args = append(args, "-x", "c", "-", "-o", path)
			cmd := exec.Command(cc, args...)
			cmd.Stdin = strings.NewReader(`
int nvtx_loader_test_value(void) { return 42; }
#ifdef UNRESOLVED
extern int nvtx_loader_test_missing_dependency(void);
int nvtx_loader_test_unused(void) { return nvtx_loader_test_missing_dependency(); }
#endif
`)
			if out, err := cmd.CombinedOutput(); err != nil {
				t.Fatalf("compile provider: %v\n%s", err, out)
			}
			h, err := openDynamicLibrary(path)
			if h != 0 {
				t.Cleanup(func() {
					if err := purego.Dlclose(h); err != nil {
						t.Errorf("close provider: %v", err)
					}
				})
			}
			if test.unresolved {
				if err == nil || h != 0 {
					t.Fatalf("open unresolved provider = %#x, %v, want zero handle and error", h, err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			sym, err := lookupLibrarySymbol(h, "nvtx_loader_test_value")
			if err != nil {
				t.Fatal(err)
			}
			if got, _, _ := purego.SyscallN(sym); got != 42 {
				t.Fatalf("provider value = %d, want 42", got)
			}
			global, err := purego.Dlsym(purego.RTLD_DEFAULT, "nvtx_loader_test_value")
			if err != nil || global != sym {
				t.Fatalf("global symbol = %#x, %v, want %#x", global, err, sym)
			}
		})
	}
}
