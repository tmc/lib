package openxla

import (
	"strings"
	"testing"
	"unsafe"
	"github.com/ebitengine/purego"
)

func TestNewClientFromApi_Truncated(t *testing.T) {
	// 1. Api table too small for version
	api1 := &Api{
		StructSize: 8,
	}
	_, err := NewClientFromApi(api1)
	if err == nil || err.Error() != "openxla: API table too small to contain API version" {
		t.Errorf("expected version truncation error, got %v", err)
	}

	// 2. Unsupported major version
	api2 := &Api{
		StructSize: unsafe.Offsetof(Api{}.ApiVersion) + unsafe.Sizeof(Api{}.ApiVersion),
		ApiVersion: Api_Version{Major_version: 1, Minor_version: 0},
	}
	_, err = NewClientFromApi(api2)
	if err == nil || err.Error() != "openxla: unsupported PJRT API major version: 1 (expected 0)" {
		t.Errorf("expected version error, got %v", err)
	}

	// 3. API table too small for Client_Create
	api3 := &Api{
		StructSize: unsafe.Offsetof(Api{}.Plugin_Initialize) + 8,
		ApiVersion: Api_Version{Major_version: 0, Minor_version: 30},
	}
	_, err = NewClientFromApi(api3)
	if err == nil || err.Error() != "openxla: API table too small to contain Client_Create" {
		t.Errorf("expected Client_Create truncation error, got %v", err)
	}
}

func TestNewClientFromApi_Success(t *testing.T) {
	api := &Api{
		StructSize: unsafe.Offsetof(Api{}.Client_Create) + 8,
		ApiVersion: Api_Version{Major_version: 0, Minor_version: 30},
	}
	
	clientCreated := false
	cbCreate := func(args *Client_Create_Args) *Error {
		clientCreated = true
		args.Client = &Client{}
		return nil
	}
	api.Client_Create = purego.NewCallback(cbCreate)

	client, err := NewClientFromApi(api)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if client == nil {
		t.Fatalf("client is nil")
	}
	if !clientCreated {
		t.Errorf("Client_Create was not called")
	}
}

func TestExecutableWrapper_NumOutputs_Zero(t *testing.T) {
	api := &Api{
		StructSize: 1000,
	}
	// Missing LoadedExecutable_GetExecutable -> should return error
	exec := &ExecutableWrapper{
		api: api,
		executable: &LoadedExecutable{},
	}
	num, err := exec.NumOutputs()
	if err == nil || num != 0 {
		t.Errorf("expected error for missing query functions, got %d, %v", num, err)
	}
	
	api.LoadedExecutable_GetExecutable = purego.NewCallback(func(args *LoadedExecutable_GetExecutable_Args) *Error {
		args.Executable = &Executable{}
		return nil
	})
	api.Executable_NumOutputs = purego.NewCallback(func(args *Executable_NumOutputs_Args) *Error {
		args.Num_outputs = 0
		return nil
	})
	api.Executable_Destroy = purego.NewCallback(func(args *Executable_Destroy_Args) *Error { return nil })
	
	num, err = exec.NumOutputs()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if num != 0 {
		t.Errorf("expected 0 outputs, got %d", num)
	}
}

func TestBufferFromHost_EventCleanup(t *testing.T) {
	api := &Api{
		StructSize: 1000,
	}
	client := &ClientWrapper{
		api: api,
		client: &Client{},
	}
	
	destroyed := false
	awaited := false
	
	api.Client_BufferFromHostBuffer = purego.NewCallback(func(args *Client_BufferFromHostBuffer_Args) *Error {
		args.Buffer = &Buffer{}
		args.Done_with_host_buffer = &Event{}
		return nil
	})
	api.Event_Await = purego.NewCallback(func(args *Event_Await_Args) *Error {
		awaited = true
		return nil
	})
	api.Event_Destroy = purego.NewCallback(func(args *Event_Destroy_Args) *Error {
		destroyed = true
		return nil
	})
	
	var dummyData int32 = 42
	buf, err := client.BufferFromHost(unsafe.Pointer(&dummyData), BufferTypeS32, []int64{1}, &Device{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if buf == nil {
		t.Fatalf("expected buffer, got nil")
	}
	if !awaited {
		t.Errorf("Event_Await not called")
	}
	if !destroyed {
		t.Errorf("Event_Destroy not called")
	}
}


func TestNewClientFromApi_ShortNativeAllocation(t *testing.T) {
	// Allocate a raw byte buffer sized strictly up to Client_Create + 8.
	// This simulates a truncated native C library function table.
	neededSize := unsafe.Offsetof(Api{}.Client_Create) + 8
	rawBuf := make([]byte, neededSize)

	// Cast the buffer header to *Api
	shortApi := (*Api)(unsafe.Pointer(&rawBuf[0]))
	shortApi.StructSize = neededSize
	shortApi.ApiVersion = Api_Version{Major_version: 0, Minor_version: 30}

	clientCreated := false
	cbCreate := purego.NewCallback(func(args *Client_Create_Args) *Error {
		clientCreated = true
		args.Client = &Client{}
		return nil
	})
	shortApi.Client_Create = cbCreate

	client, err := NewClientFromApi(shortApi)
	if err != nil {
		t.Fatalf("unexpected error with short native allocation: %v", err)
	}
	if !clientCreated {
		t.Fatalf("Client_Create was not called")
	}

	// Any access to functions beyond the short allocation should return "not available",
	// and MUST NOT segfault or read garbage pointers.
	if _, err := client.PlatformName(); err == nil || !strings.Contains(err.Error(), "not available") {
		t.Errorf("expected not available for PlatformName on truncated table, got %v", err)
	}
	if _, err := client.BufferFromHost(nil, BufferTypeF32, nil, nil); err == nil || !strings.Contains(err.Error(), "not available") {
		t.Errorf("expected not available for BufferFromHost on truncated table, got %v", err)
	}
}

func TestToHost_MissingAwaitRejection(t *testing.T) {
	api := &Api{
		StructSize: 1000,
	}
	// Missing Event_Await and Event_Destroy
	api.Buffer_ToHostBuffer = purego.NewCallback(func(args *Buffer_ToHostBuffer_Args) *Error {
		t.Fatal("Buffer_ToHostBuffer should not be called when Await/Destroy are missing")
		return nil
	})

	buf := &BufferWrapper{
		api: api,
		buffer: &Buffer{},
	}
	var dst int32
	err := buf.ToHost(unsafe.Pointer(&dst), 4)
	if err == nil {
		t.Fatal("expected error for missing Event_Await/Event_Destroy in ToHost, got nil")
	}
	if !strings.Contains(err.Error(), "required Event_Await or Event_Destroy not available") {
		t.Errorf("unexpected error message: %v", err)
	}
}

func TestExecute_MissingAwaitRejection(t *testing.T) {
	api := &Api{
		StructSize: 1000,
	}
	// Missing Event_Await and Event_Destroy
	api.LoadedExecutable_Execute = purego.NewCallback(func(args *LoadedExecutable_Execute_Args) *Error {
		t.Fatal("LoadedExecutable_Execute should not be called when Await/Destroy are missing")
		return nil
	})
	api.LoadedExecutable_GetExecutable = purego.NewCallback(func(args *LoadedExecutable_GetExecutable_Args) *Error {
		args.Executable = &Executable{}
		return nil
	})
	api.Executable_NumOutputs = purego.NewCallback(func(args *Executable_NumOutputs_Args) *Error {
		args.Num_outputs = 1
		return nil
	})
	api.Executable_Destroy = purego.NewCallback(func(args *Executable_Destroy_Args) *Error { return nil })

	exec := &ExecutableWrapper{
		api: api,
		executable: &LoadedExecutable{},
	}
	b := &BufferWrapper{api: api, buffer: &Buffer{}}
	_, err := exec.Execute([][]*BufferWrapper{{b}})
	if err == nil {
		t.Fatal("expected error for missing Event_Await/Event_Destroy in Execute, got nil")
	}
	if !strings.Contains(err.Error(), "required Event_Await or Event_Destroy not available") {
		t.Errorf("unexpected error message: %v", err)
	}
}

func TestBufferFromHost_AwaitFailureCleansBuffer(t *testing.T) {
	api := &Api{
		StructSize: 1000,
	}
	client := &ClientWrapper{
		api: api,
		client: &Client{},
	}

	bufferDestroyed := false
	eventDestroyed := false
	errorDestroyed := false

	mockErr := &Error{}

	api.Client_BufferFromHostBuffer = purego.NewCallback(func(args *Client_BufferFromHostBuffer_Args) *Error {
		args.Buffer = &Buffer{}
		args.Done_with_host_buffer = &Event{}
		return nil
	})
	api.Event_Await = purego.NewCallback(func(args *Event_Await_Args) *Error {
		return mockErr
	})
	api.Event_Destroy = purego.NewCallback(func(args *Event_Destroy_Args) *Error {
		eventDestroyed = true
		return nil
	})
	api.Buffer_Destroy = purego.NewCallback(func(args *Buffer_Destroy_Args) *Error {
		bufferDestroyed = true
		return nil
	})
	api.Error_Destroy = purego.NewCallback(func(args *Error_Destroy_Args) {
		errorDestroyed = true
	})
	api.Error_Message = purego.NewCallback(func(args *Error_Message_Args) {
		msg := "await failed"
		args.Message = &([]byte(msg))[0]
		args.Message_size = uintptr(len(msg))
	})

	var dummyData int32 = 42
	buf, err := client.BufferFromHost(unsafe.Pointer(&dummyData), BufferTypeS32, []int64{1}, &Device{})
	if err == nil {
		t.Fatal("expected error from failed Event_Await, got nil")
	}
	if buf != nil {
		t.Errorf("expected nil buffer on error, got %v", buf)
	}
	if !eventDestroyed {
		t.Errorf("Event_Destroy was not called on await failure")
	}
	if !bufferDestroyed {
		t.Errorf("Buffer_Destroy was not called on await failure (buffer leaked!)")
	}
	if !errorDestroyed {
		t.Errorf("Error_Destroy was not called for returned error")
	}
}
