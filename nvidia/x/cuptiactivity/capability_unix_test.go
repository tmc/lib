//go:build (linux || darwin) && cgo

package cuptiactivity

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/cupti"
)

// The stub only reports a version. Registering or enabling collection aborts
// the subprocess: a missing layout API must be refused before those calls.
const missingLayoutAPIStub = `
#include <stdint.h>
#include <stdlib.h>
int cuptiGetVersion(uint32_t *version) { *version = 130001; return 0; }
int cuptiActivityRegisterCallbacks(void *request, void *complete) { abort(); }
int cuptiActivityEnable(uint32_t kind) { abort(); }
`

func TestStartMissingLayoutAPI(t *testing.T) {
	if path := os.Getenv("CUPTIACTIVITY_LAYOUT_STUB"); path != "" {
		if err := cupti.Load(path); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := cupti.Close(); err != nil {
				t.Error(err)
			}
		})
		session, err := Start(Config{Kinds: []cupti.ActivityKind{cupti.ActivityKindConcurrentKernel}, BufferBytes: 64, MaxBuffers: 1, MaxRecords: 1})
		var symbol *cupti.SymbolError
		if session != nil || !errors.As(err, &symbol) || symbol.Symbol != "cuptiActivityGetStructSize" || symbol.Err != nil {
			t.Fatalf("session=%v error=%v; want missing-layout-API refusal", session, err)
		}
		if !strings.Contains(err.Error(), "CUPTI version 130001 cannot verify activity layouts") {
			t.Fatal(err)
		}
		nativeSessions.Lock()
		defer nativeSessions.Unlock()
		if nativeSessions.active != nil || nativeSessions.registered || nativeSessions.request != 0 || nativeSessions.complete != 0 || nativeSessions.callbacks != 0 {
			t.Fatal("refused layout acquired collection ownership or callbacks")
		}
		return
	}
	cc, err := exec.LookPath("cc")
	if err != nil {
		t.Skip("CPU layout capability control requires a C compiler")
	}
	dir := t.TempDir()
	source, library := filepath.Join(dir, "stub.c"), filepath.Join(dir, "stub.so")
	if err := os.WriteFile(source, []byte(missingLayoutAPIStub), 0600); err != nil {
		t.Fatal(err)
	}
	flag := "-shared"
	if runtime.GOOS == "darwin" {
		flag = "-dynamiclib"
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if output, err := exec.CommandContext(ctx, cc, flag, "-fPIC", source, "-o", library).CombinedOutput(); err != nil {
		t.Fatalf("build CPU CUPTI stub: %v\n%s", err, output)
	}
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	child := exec.CommandContext(ctx, executable, "-test.run=^TestStartMissingLayoutAPI$", "-test.v")
	child.Env = append(os.Environ(), "CUPTIACTIVITY_LAYOUT_STUB="+library, "CUPTI_LIB_PATH="+library, "CODEGEN_SKIP_BINDINGS=")
	if output, err := child.CombinedOutput(); err != nil {
		t.Fatalf("CPU missing-layout-API control: %v\n%s", err, output)
	} else {
		t.Log(string(output))
	}
}
