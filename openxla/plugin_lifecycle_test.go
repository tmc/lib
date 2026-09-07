package openxla

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
	"unsafe"

	"github.com/ebitengine/purego"
)

// strictPlugin models an initializer that rejects a second successful setup.
// Offsets come from Go solely to test lifecycle behavior, not C ABI conformance.
func strictPlugin(t *testing.T, initFailures, createFailures int, copyTable bool) (string, func(int32) int32) {
	t.Helper()
	if runtime.GOOS != "linux" && runtime.GOOS != "darwin" {
		t.Skip("native shared-library fixture requires Linux or Darwin")
	}
	cc, err := exec.LookPath("cc")
	if err != nil {
		t.Skip("native fixture requires a C compiler")
	}
	source := `
#include <stdint.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <string.h>
static uintptr_t api[{{APIWORDS}}];
static atomic_int initialized, inits, creates, destroys, errors;
static int init_failures = {{INITFAIL}}, create_failures = {{CREATEFAIL}};
static void *failure(void) { return malloc(1); }
static void *initialize(void *args) {
 atomic_fetch_add(&inits, 1);
 if (*(uintptr_t *)args != {{INITSIZE}}) return failure();
 if (init_failures > 0) { init_failures--; return failure(); }
 if (atomic_exchange(&initialized, 1)) return failure();
 return NULL;
}
static void *create(void *args) {
 atomic_fetch_add(&creates, 1);
 if (!atomic_load(&initialized) || *(uintptr_t *)args != {{CREATESIZE}}) return failure();
 if (create_failures > 0) { create_failures--; return failure(); }
 *(void **)((char *)args + {{CREATECLIENT}}) = malloc(1);
 return NULL;
}
static void *destroy(void *args) {
 atomic_fetch_add(&destroys, 1);
 free(*(void **)((char *)args + {{DESTROYCLIENT}}));
 return NULL;
}
static void error_destroy(void *args) {
 atomic_fetch_add(&errors, 1);
 free(*(void **)((char *)args + {{ERROR}}));
}
static void error_message(void *args) {
 *(const char **)((char *)args + {{MESSAGE}}) = "strict plugin error";
 *(uintptr_t *)((char *)args + {{MESSAGESIZE}}) = 19;
}
__attribute__((constructor)) static void setup(void) {
 api[0] = sizeof(api);
 *(uintptr_t *)((char *)api + {{INITIALIZE}}) = (uintptr_t)initialize;
 *(uintptr_t *)((char *)api + {{CREATE}}) = (uintptr_t)create;
 *(uintptr_t *)((char *)api + {{DESTROY}}) = (uintptr_t)destroy;
 *(uintptr_t *)((char *)api + {{ERRORDESTROY}}) = (uintptr_t)error_destroy;
 *(uintptr_t *)((char *)api + {{ERRORMESSAGE}}) = (uintptr_t)error_message;
}
void *GetPjrtApi(void) { return api; }
int Counter(int which) {
 switch (which) {
 case 0: return atomic_load(&inits);
 case 1: return atomic_load(&creates);
 case 2: return atomic_load(&destroys);
 case 3: return atomic_load(&errors);
 default: return -1;
 }
}
`
	if copyTable {
		source = strings.Replace(source, "void *GetPjrtApi(void) { return api; }", "void *GetPjrtApi(void) { void *copy = malloc(sizeof(api)); memcpy(copy, api, sizeof(api)); return copy; }", 1)
	}
	values := map[string]uintptr{
		"APIWORDS": unsafe.Sizeof(Api{}) / unsafe.Sizeof(uintptr(0)),
		"INITFAIL": uintptr(initFailures), "CREATEFAIL": uintptr(createFailures),
		"INITSIZE": PJRT_Plugin_Initialize_Args_STRUCT_SIZE, "CREATESIZE": PJRT_Client_Create_Args_STRUCT_SIZE,
		"CREATECLIENT": unsafe.Offsetof(Client_Create_Args{}.Client), "DESTROYCLIENT": unsafe.Offsetof(Client_Destroy_Args{}.Client),
		"ERROR": unsafe.Offsetof(Error_Destroy_Args{}.Error), "MESSAGE": unsafe.Offsetof(Error_Message_Args{}.Message), "MESSAGESIZE": unsafe.Offsetof(Error_Message_Args{}.Message_size),
		"INITIALIZE": unsafe.Offsetof(Api{}.Plugin_Initialize), "CREATE": unsafe.Offsetof(Api{}.Client_Create), "DESTROY": unsafe.Offsetof(Api{}.Client_Destroy),
		"ERRORDESTROY": unsafe.Offsetof(Api{}.Error_Destroy), "ERRORMESSAGE": unsafe.Offsetof(Api{}.Error_Message),
	}
	for name, value := range values {
		source = strings.ReplaceAll(source, "{{"+name+"}}", strconv.FormatUint(uint64(value), 10))
	}
	dir := t.TempDir()
	src, path := filepath.Join(dir, "plugin.c"), filepath.Join(dir, "plugin.so")
	if err := os.WriteFile(src, []byte(source), 0o600); err != nil {
		t.Fatal(err)
	}
	args := []string{"-shared", "-fPIC", "-std=c11", "-o", path, src}
	if runtime.GOOS == "darwin" {
		args[0] = "-dynamiclib"
	}
	if out, err := exec.Command(cc, args...).CombinedOutput(); err != nil {
		t.Fatalf("compile strict plugin: %v\n%s", err, out)
	}
	handle, err := purego.Dlopen(path, purego.RTLD_NOW|purego.RTLD_LOCAL)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := purego.Dlclose(handle); err != nil {
			t.Error(err)
		}
	})
	var count func(int32) int32
	purego.RegisterLibFunc(&count, handle, "Counter")
	return path, count
}

func TestPluginInitializationShared(t *testing.T) {
	for _, copyTable := range []bool{false, true} {
		t.Run(strconv.FormatBool(copyTable), func(t *testing.T) {
			path, count := strictPlugin(t, 0, 0, copyTable)
			a, err := OpenPlugin(path)
			if err != nil {
				t.Fatal(err)
			}
			alias := filepath.Join(t.TempDir(), "alias.so")
			if err := os.Symlink(path, alias); err != nil {
				t.Fatal(err)
			}
			b, err := OpenPlugin(alias)
			if err != nil {
				t.Fatal(err)
			}
			if (a.Api() != b.Api()) != copyTable {
				t.Fatal("fixture did not reach requested API identity")
			}
			var wg sync.WaitGroup
			for i := 0; i < 8; i++ {
				wg.Add(1)
				go func(i int) {
					defer wg.Done()
					var client *ClientWrapper
					var err error
					switch i % 3 {
					case 0:
						client, err = a.NewClient()
					case 1:
						client, err = b.NewClient()
					case 2:
						client, err = NewClientFromApi(a.Api())
					}
					if err != nil {
						t.Error(err)
						return
					}
					if err := client.Destroy(); err != nil {
						t.Error(err)
					}
				}(i)
			}
			wg.Wait()
			if got := count(0); got != 1 {
				t.Errorf("initializations = %d, want 1", got)
			}
			if got := count(1); got != 8 {
				t.Errorf("creates = %d, want 8", got)
			}
			if got := count(2); got != 8 {
				t.Errorf("destroys = %d, want 8", got)
			}
			if got := count(3); got != 0 {
				t.Errorf("errors = %d, want 0", got)
			}
		})
	}
}

func TestPluginInitializationRetry(t *testing.T) {
	path, count := strictPlugin(t, 1, 1, false)
	p, err := OpenPlugin(path)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 2; i++ {
		client, err := p.NewClient()
		if client != nil || err == nil || !strings.Contains(err.Error(), "strict plugin error") {
			t.Fatalf("attempt %d: client=%v err=%v", i, client, err)
		}
	}
	for i := 0; i < 2; i++ {
		client, err := p.NewClient()
		if err != nil {
			t.Fatal(err)
		}
		if err := client.Destroy(); err != nil {
			t.Fatal(err)
		}
	}
	for which, want := range []int32{2, 3, 2, 2} {
		if got := count(int32(which)); got != want {
			t.Errorf("counter %d = %d, want %d", which, got, want)
		}
	}
}

func TestPluginValidationBeforeInitialization(t *testing.T) {
	path, count := strictPlugin(t, 0, 0, false)
	p, err := OpenPlugin(path)
	if err != nil {
		t.Fatal(err)
	}
	api := p.Api()
	destroy := api.Client_Destroy
	api.Client_Destroy = 0
	_, err = p.NewClient()
	api.Client_Destroy = destroy
	if err == nil || !strings.Contains(err.Error(), "Client_Destroy") {
		t.Fatalf("validation: %v", err)
	}
	if count(0) != 0 {
		t.Fatal("initialized before validation")
	}
	client, err := p.NewClient()
	if err != nil {
		t.Fatal(err)
	}
	if err := client.Destroy(); err != nil {
		t.Fatal(err)
	}
	if got := count(0); got != 1 {
		t.Fatalf("initializations = %d, want 1", got)
	}
}

func TestExternalAPIInitializationIsNotRetained(t *testing.T) {
	inits := 0
	api := &Api{
		StructSize:        unsafe.Sizeof(Api{}),
		Error_Destroy:     purego.NewCallback(func(*Error_Destroy_Args) {}),
		Error_Message:     purego.NewCallback(func(*Error_Message_Args) {}),
		Plugin_Initialize: purego.NewCallback(func(*Plugin_Initialize_Args) *Error { inits++; return nil }),
		Client_Create:     purego.NewCallback(func(args *Client_Create_Args) *Error { args.Client = &Client{}; return nil }),
		Client_Destroy:    purego.NewCallback(func(*Client_Destroy_Args) *Error { return nil }),
	}
	for i := 0; i < 2; i++ {
		client, err := NewClientFromApi(api)
		if err != nil {
			t.Fatal(err)
		}
		if err := client.Destroy(); err != nil {
			t.Fatal(err)
		}
	}
	if inits != 2 {
		t.Fatalf("external initializations = %d, want 2", inits)
	}
	if managedInitialization(api) != nil {
		t.Fatal("retained externally owned API table")
	}
}
