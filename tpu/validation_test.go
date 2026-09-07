package tpu

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"unsafe"

	"github.com/tmc/lib/openxla"
)

// These assignments require type identity, including nested PJRT declarations.
var (
	_ *openxla.Api                 = (*Api)(nil)
	_ *openxla.ClientWrapper       = (*ClientWrapper)(nil)
	_ *openxla.Buffer_MemoryLayout = (*Buffer_MemoryLayout)(nil)
)

func testPlugin(t *testing.T, source string) string {
	t.Helper()
	if runtime.GOOS != "darwin" && runtime.GOOS != "linux" {
		t.Skip("native fixture requires Unix shared libraries")
	}
	cc, err := exec.LookPath("cc")
	if err != nil {
		t.Skip("native fixture requires a C compiler")
	}
	dir := t.TempDir()
	src, lib := filepath.Join(dir, "plugin.c"), filepath.Join(dir, "plugin.so")
	if err := os.WriteFile(src, []byte(source), 0o600); err != nil {
		t.Fatal(err)
	}
	args := []string{"-shared", "-fPIC", "-o", lib, src}
	if runtime.GOOS == "darwin" {
		args[0] = "-dynamiclib"
	}
	if out, err := exec.Command(cc, args...).CombinedOutput(); err != nil {
		t.Fatalf("compile fixture: %v\n%s", err, out)
	}
	return lib
}

func resetLoader(t *testing.T) {
	t.Helper()
	Close()
	t.Cleanup(func() { Close() })
	t.Setenv("CODEGEN_SKIP_BINDINGS", "")
}

func TestMalformedAPI(t *testing.T) {
	resetLoader(t)
	path := testPlugin(t, "#include <stdint.h>\nvoid *GetPjrtApi(void) { static uintptr_t api[4096]; return api; }\n")
	if err := Load(path); err != nil {
		t.Fatal(err)
	}
	api, err := GetApi()
	if err != nil {
		t.Fatal(err)
	}
	valid := Api{StructSize: unsafe.Sizeof(Api{}), Error_Message: 1, Error_Destroy: 1, Client_Create: 1, Client_Destroy: 1}
	tests := []struct {
		name   string
		change func(*Api)
		want   string
	}{
		{"version outside table", func(a *Api) { a.StructSize = unsafe.Sizeof(uintptr(0)) }, "too small"},
		{"unsupported version", func(a *Api) { a.ApiVersion.Major_version = 99 }, "major version"},
		{"destroy outside table", func(a *Api) { a.StructSize = unsafe.Offsetof(a.Client_Destroy) }, "too small"},
		{"missing error destroy", func(a *Api) { a.Error_Destroy = 0 }, "error reporting"},
		{"missing error message", func(a *Api) { a.Error_Message = 0 }, "error reporting"},
		{"missing create", func(a *Api) { a.Client_Create = 0 }, "Client_Create"},
		{"missing destroy", func(a *Api) { a.Client_Destroy = 0 }, "Client_Destroy"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			*api = valid
			tt.change(api)
			client, err := NewClient()
			if client != nil || err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("NewClient() = %v, %v; want %q", client, err, tt.want)
			}
		})
	}
	// Nonzero sentinel pointers must not be inspected beyond StructSize.
	*api = valid
	api.Plugin_Initialize, api.Client_Compile = 1, 1
	for _, symbol := range []string{"Client_Create", "Client_Destroy", "Client_Compile", "Plugin_Initialize"} {
		api.StructSize = unsafe.Sizeof(uintptr(0))
		if Has(symbol) {
			t.Errorf("Has(%q) read outside table", symbol)
		}
		api.StructSize = unsafe.Sizeof(Api{})
		if !Has(symbol) {
			t.Errorf("Has(%q) ignored a present slot", symbol)
		}
	}
}

func TestLoadInvalidPlugin(t *testing.T) {
	resetLoader(t)
	for _, tt := range []struct{ name, source, want string }{
		{"missing symbol", "int unrelated(void) { return 0; }", "GetPjrtApi"},
		{"nil table", "void *GetPjrtApi(void) { return 0; }", "nil"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			err := Load(testPlugin(t, tt.source))
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("Load: %v; want %q", err, tt.want)
			}
			if Loaded() || Path() != "" {
				t.Fatal("failed load retained selected plugin")
			}
		})
	}
}

func TestLoaderResetConcurrent(t *testing.T) {
	resetLoader(t)
	path := testPlugin(t, "#include <stdint.h>\nvoid *GetPjrtApi(void) { static uintptr_t api[4096]; return api; }\n")
	t.Setenv("TPU_LIB_PATH", path)
	if err := Load(path); err != nil {
		t.Fatal(err)
	}
	var conflict *LibraryConflictError
	if err := Load("another-plugin"); !errors.As(err, &conflict) {
		t.Fatalf("conflicting path: %v", err)
	}
	if conflict.Loaded != path || conflict.Requested != "another-plugin" {
		t.Fatalf("conflict: %+v", conflict)
	}
	original, err := GetApi()
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for range 4 {
		wg.Go(func() {
			for range 8 {
				Close()
				api, err := GetApi()
				if err != nil || api == nil || api != original {
					t.Errorf("GetApi after reset: %p, %v; want %p", api, err, original)
				}
				Has("GetPjrtApi")
				Loaded()
				Path()
			}
		})
	}
	wg.Wait()
}

func TestNativeClient(t *testing.T) {
	path := os.Getenv("XLA_PJRT_CPU_PLUGIN")
	if path == "" {
		t.Skip("set XLA_PJRT_CPU_PLUGIN to exercise a native PJRT client")
	}
	resetLoader(t)
	if err := Load(path); err != nil {
		t.Fatal(err)
	}
	client, err := NewClient()
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := client.Destroy(); err != nil {
			t.Error(err)
		}
	}()
	Close()
	name, err := client.PlatformName()
	if err != nil || name == "" {
		t.Fatalf("PlatformName after loader reset = %q, %v", name, err)
	}
	if version, err := client.PlatformVersion(); err != nil || version == "" {
		t.Fatalf("PlatformVersion = %q, %v", version, err)
	}
}
