package openxla

import (
	"runtime"
	"strings"
	"testing"

	"github.com/ebitengine/purego"
)

func TestOpenPluginMissingSymbol(t *testing.T) {
	var path string
	switch runtime.GOOS {
	case "darwin":
		path = "/usr/lib/libSystem.B.dylib"
	case "linux":
		path = "libc.so.6"
	default:
		t.Skip("no system library fixture for this platform")
	}
	// Confirm this is a loadable library, so the test reaches symbol lookup.
	handle, err := purego.Dlopen(path, purego.RTLD_NOW)
	if err != nil {
		t.Fatal(err)
	}
	if err := purego.Dlclose(handle); err != nil {
		t.Fatal(err)
	}
	plugin, err := OpenPlugin(path)
	if plugin != nil || err == nil || !strings.Contains(err.Error(), "GetPjrtApi") {
		t.Fatalf("plugin=%v err=%v", plugin, err)
	}
}
