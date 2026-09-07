package xla

import (
	"bytes"
	"context"
	"errors"
	"io"
	"testing"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/tmc/lib/openxla"
)

func testExecutable(t *testing.T, onAwait func(), badShape, failDestroy bool) (*Executable, *int, *int) {
	t.Helper()
	api := &openxla.Api{StructSize: unsafe.Sizeof(openxla.Api{})}
	destroyed, downloads := new(int), new(int)
	api.Error_Destroy = purego.NewCallback(func(*openxla.Error_Destroy_Args) {})
	api.Error_Message = purego.NewCallback(func(*openxla.Error_Message_Args) {})
	api.Client_Create = purego.NewCallback(func(a *openxla.Client_Create_Args) *openxla.Error { a.Client = &openxla.Client{}; return nil })
	api.Client_Destroy = purego.NewCallback(func(*openxla.Client_Destroy_Args) *openxla.Error { return nil })
	api.Client_BufferFromHostBuffer = purego.NewCallback(func(a *openxla.Client_BufferFromHostBuffer_Args) *openxla.Error {
		a.Buffer = &openxla.Buffer{}
		return nil
	})
	api.Client_Compile = purego.NewCallback(func(a *openxla.Client_Compile_Args) *openxla.Error {
		a.Executable = &openxla.LoadedExecutable{}
		return nil
	})
	api.LoadedExecutable_GetExecutable = purego.NewCallback(func(a *openxla.LoadedExecutable_GetExecutable_Args) *openxla.Error {
		a.Executable = &openxla.Executable{}
		return nil
	})
	api.Executable_Destroy = purego.NewCallback(func(*openxla.Executable_Destroy_Args) *openxla.Error { return nil })
	api.LoadedExecutable_Destroy = purego.NewCallback(func(*openxla.LoadedExecutable_Destroy_Args) *openxla.Error { return nil })
	api.Executable_NumOutputs = purego.NewCallback(func(a *openxla.Executable_NumOutputs_Args) *openxla.Error { a.Num_outputs = 2; return nil })
	api.LoadedExecutable_Execute = purego.NewCallback(func(a *openxla.LoadedExecutable_Execute_Args) *openxla.Error {
		ptrs := unsafe.Slice((***openxla.Buffer)(a.Output_lists), 1)
		out := unsafe.Slice(ptrs[0], 2)
		out[0], out[1] = &openxla.Buffer{}, &openxla.Buffer{}
		unsafe.Slice((**openxla.Event)(a.Device_complete_events), 1)[0] = &openxla.Event{}
		return nil
	})
	api.Event_Await = purego.NewCallback(func(*openxla.Event_Await_Args) *openxla.Error {
		if onAwait != nil {
			onAwait()
		}
		return nil
	})
	api.Event_Destroy = purego.NewCallback(func(*openxla.Event_Destroy_Args) *openxla.Error { return nil })
	api.Buffer_Destroy = purego.NewCallback(func(*openxla.Buffer_Destroy_Args) *openxla.Error {
		*destroyed++
		if failDestroy {
			return &openxla.Error{}
		}
		return nil
	})
	api.Buffer_ElementType = purego.NewCallback(func(a *openxla.Buffer_ElementType_Args) *openxla.Error { a.Type = openxla.Buffer_Type_U8; return nil })
	dims := []int64{3}
	api.Buffer_Dimensions = purego.NewCallback(func(a *openxla.Buffer_Dimensions_Args) *openxla.Error {
		a.Num_dims = 1
		if !badShape {
			a.Dims = &dims[0]
		}
		return nil
	})
	api.Buffer_ToHostBuffer = purego.NewCallback(func(a *openxla.Buffer_ToHostBuffer_Args) *openxla.Error {
		*downloads++
		copy(unsafe.Slice((*byte)(a.Dst), a.Dst_size), []byte{1, 2, 3})
		return nil
	})
	client, err := openxla.NewClientFromApi(api)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := client.Destroy(); err != nil {
			t.Error(err)
		}
	})
	wrapper, err := client.Compile([]byte("unused fake program"), "mlir", nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := wrapper.Destroy(); err != nil {
			t.Error(err)
		}
	})
	return &Executable{wrapper: wrapper, engine: &Engine{client: client}}, destroyed, downloads
}

func TestExecuteCancelledCleansOutputs(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	e, destroyed, _ := testExecutable(t, cancel, false, false)
	out, err := e.Execute(ctx, nil)
	if !errors.Is(err, context.Canceled) || out != nil || *destroyed != 2 {
		t.Fatalf("outputs=%v err=%v destroyed=%d", out, err, *destroyed)
	}
}

func TestExecuteBadShapeCleansAllOutputs(t *testing.T) {
	e, destroyed, _ := testExecutable(t, nil, true, false)
	out, err := e.Execute(context.Background(), nil)
	if err == nil || out != nil || *destroyed != 2 {
		t.Fatalf("outputs=%v err=%v destroyed=%d", out, err, *destroyed)
	}
}

type shortWriter struct{}

func (shortWriter) Write(p []byte) (int, error) { return 1, nil }

func TestExecuteOutputMetadata(t *testing.T) {
	e, destroyed, downloads := testExecutable(t, nil, false, false)
	out, err := e.Execute(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		for _, b := range out {
			if err := b.Close(); err != nil {
				t.Error(err)
			}
		}
	}()
	if err := out[0].ToHost(make([]byte, 2)); err == nil || *downloads != 0 {
		t.Fatalf("short read err=%v dispatches=%d", err, *downloads)
	}
	var buf bytes.Buffer
	n, err := out[0].WriteTo(&buf)
	if err != nil || n != 3 || !bytes.Equal(buf.Bytes(), []byte{1, 2, 3}) {
		t.Fatalf("n=%d bytes=%v err=%v", n, buf.Bytes(), err)
	}
	n, err = out[0].WriteTo(shortWriter{})
	if n != 1 || !errors.Is(err, io.ErrShortWrite) {
		t.Fatalf("n=%d err=%v", n, err)
	}
	if *destroyed != 0 {
		t.Fatalf("outputs destroyed before Close: %d", *destroyed)
	}
}

func TestFailedCloseInvalidatesBuffer(t *testing.T) {
	e, destroyed, downloads := testExecutable(t, nil, false, true)
	out, err := e.Execute(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer out[1].Close()
	if err := out[0].Close(); err == nil {
		t.Fatal("missing destruction error")
	}
	if err := out[0].ToHost(make([]byte, 3)); err == nil || *downloads != 0 {
		t.Fatalf("closed buffer err=%v downloads=%d", err, *downloads)
	}
	if err := out[0].Close(); err != nil || *destroyed != 1 {
		t.Fatalf("second close err=%v destroys=%d", err, *destroyed)
	}
}
