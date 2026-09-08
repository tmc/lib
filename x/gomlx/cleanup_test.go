package gomlx

import (
	"strings"
	"testing"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/gomlx/compute"
	"github.com/tmc/lib/openxla"
	"github.com/tmc/lib/x/xla"
)

func TestExecutableCleanupErrorRetained(t *testing.T) {
	api := &openxla.Api{StructSize: unsafe.Sizeof(openxla.Api{})}
	message := []byte("injected executable cleanup failure")
	executableCloses, clientCloses, errorCloses := 0, 0, 0
	api.Error_Message = purego.NewCallback(func(a *openxla.Error_Message_Args) { a.Message = &message[0]; a.Message_size = uintptr(len(message)) })
	api.Error_Destroy = purego.NewCallback(func(*openxla.Error_Destroy_Args) { errorCloses++ })
	api.Client_Create = purego.NewCallback(func(a *openxla.Client_Create_Args) *openxla.Error { a.Client = &openxla.Client{}; return nil })
	api.Client_Destroy = purego.NewCallback(func(*openxla.Client_Destroy_Args) *openxla.Error { clientCloses++; return nil })
	api.Client_Compile = purego.NewCallback(func(a *openxla.Client_Compile_Args) *openxla.Error {
		a.Executable = &openxla.LoadedExecutable{}
		return nil
	})
	api.LoadedExecutable_Destroy = purego.NewCallback(func(*openxla.LoadedExecutable_Destroy_Args) *openxla.Error {
		executableCloses++
		return &openxla.Error{}
	})
	engine, err := xla.NewEngineFromAPI(api)
	if err != nil {
		t.Fatal(err)
	}
	b := &Backend{engine: engine, buffers: make(map[*buffer]struct{}), executables: make(map[*executable]struct{})}
	defer b.Close()
	builder := b.Builder("cleanup")
	f := builder.Main()
	v, err := f.Constant([]float32{1})
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Return([]compute.Value{v}, nil); err != nil {
		t.Fatal(err)
	}
	e, err := builder.Compile()
	if err != nil {
		t.Fatal(err)
	}
	e.Finalize()
	for range 2 {
		err := b.Close()
		if err == nil || !strings.Contains(err.Error(), string(message)) {
			t.Errorf("Close error = %v, want injected failure", err)
		}
	}
	e.Finalize()
	if executableCloses != 1 || clientCloses != 1 || errorCloses != 1 {
		t.Fatalf("cleanup counts: executable=%d client=%d error=%d", executableCloses, clientCloses, errorCloses)
	}
}
