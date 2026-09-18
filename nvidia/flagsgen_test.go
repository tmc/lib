package nvidia_test

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// genoptions is the command that writes the flags.gen.go files. This test names
// it by import path, the same way the go:generate directives do, so it keeps
// working when x becomes its own module and the path resolves through go.work.
const genoptions = "github.com/tmc/lib/nvidia/x/internal/cudatool/cmd/genoptions"

// TestFlagsGenUpToDate regenerates each flags.gen.go and compares it to the
// committed file.
//
// nvgen does not write these files and nvgen -diff does not read them, so a
// clean -diff reports the package unchanged however far flags.gen.go has
// drifted. Corrupting nvrtc/flags.gen.go leaves -diff reporting the same four
// differing files as an untouched tree, while corrupting a file nvgen does own
// raises the count and names it. This test is the only check that covers them.
//
// Today genoptions is in this module and always resolves. When x becomes its
// own module the command moves out of it, and this module must not require x,
// so go.work becomes the only thing that can reach it. The skip keeps the test
// honest then: outside the workspace it reports that it could not run, rather
// than failing for a reason that has nothing to do with the tree.
func TestFlagsGenUpToDate(t *testing.T) {
	if testing.Short() {
		t.Skip("runs the generator")
	}
	if out, err := exec.Command("go", "list", genoptions).CombinedOutput(); err != nil {
		t.Skipf("genoptions unreachable, needs the go.work workspace: %s", bytes.TrimSpace(out))
	}

	for _, target := range []string{"nvjitlink", "nvrtc", "nvvm"} {
		t.Run(target, func(t *testing.T) {
			committed := filepath.Join(target, "flags.gen.go")
			want, err := os.ReadFile(committed)
			if err != nil {
				t.Fatal(err)
			}
			fresh := filepath.Join(t.TempDir(), "flags.gen.go")
			cmd := exec.Command("go", "run", genoptions, "-target="+target, "-out="+fresh)
			if out, err := cmd.CombinedOutput(); err != nil {
				t.Fatalf("genoptions -target=%s: %v\n%s", target, err, out)
			}
			got, err := os.ReadFile(fresh)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(got, want) {
				t.Errorf("%s is stale; regenerate with go generate ./%s", committed, target)
			}
		})
	}
}
