package openxla

import (
	"bytes"
	"fmt"
	"os"
	"slices"
	"testing"
	"unsafe"

	"github.com/ebitengine/purego"
)

func TestDefaultDeviceAssignment(t *testing.T) {
	for _, tt := range []struct {
		name                 string
		replicas, partitions int
		fail                 bool
	}{
		{"valid", 2, 3, false}, {"error", 2, 3, true},
		{"zero", 0, 1, false}, {"negative", 1, -1, false},
		{"overflow", int(^uint(0) >> 1), 2, false},
		{"product overflow", 1<<31 - 1, 1<<31 - 1, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			calls, destroys := 0, 0
			api := &Api{}
			api.Error_Destroy = purego.NewCallback(func(*Error_Destroy_Args) { destroys++ })
			api.Client_DefaultDeviceAssignment = purego.NewCallback(func(a *Client_DefaultDeviceAssignment_Args) *ErrorHandle {
				calls++
				if a.Struct_size != PJRT_Client_DefaultDeviceAssignment_Args_STRUCT_SIZE || a.Num_replicas != 2 || a.Num_partitions != 3 || a.Default_assignment_size != 6 {
					t.Errorf("invalid args: %+v", a)
				}
				if tt.fail {
					return &ErrorHandle{}
				}
				copy(unsafe.Slice(a.Default_assignment, a.Default_assignment_size), []int32{9, 4, 8, 2, 7, 1})
				return nil
			})
			c := &ClientWrapper{api: api, client: &Client{}}
			got, err := c.DefaultDeviceAssignment(tt.replicas, tt.partitions)
			if tt.name != "valid" {
				if err == nil || got != nil {
					t.Fatalf("got %v, %v", got, err)
				}
				if tt.fail {
					if calls != 1 || destroys != 1 {
						t.Fatalf("calls=%d destroys=%d", calls, destroys)
					}
				} else if calls != 0 {
					t.Fatal("invalid input dispatched")
				}
				return
			}
			if err != nil || !slices.Equal(got, []int32{9, 4, 8, 2, 7, 1}) || calls != 1 {
				t.Fatalf("got %v, %v; calls=%d", got, err, calls)
			}
		})
	}
}

func TestSerializedDeviceAssignment(t *testing.T) {
	for _, name := range []string{"valid", "empty", "nil bytes", "overflow", "nil owner", "nil deleter", "error"} {
		t.Run(name, func(t *testing.T) {
			data := []byte{8, 2, 16, 3}
			owner := &DeviceAssignmentSerialized{}
			calls, frees, destroys := 0, 0, 0
			api := &Api{}
			api.Error_Destroy = purego.NewCallback(func(*Error_Destroy_Args) { destroys++ })
			free := purego.NewCallback(func(p *DeviceAssignmentSerialized) {
				frees++
				if p != owner {
					t.Error("wrong backing allocation")
				}
				clear(data)
			})
			api.LoadedExecutable_GetDeviceAssignment = purego.NewCallback(func(a *LoadedExecutable_GetDeviceAssignment_Args) *ErrorHandle {
				calls++
				if a.Struct_size != PJRT_LoadedExecutable_GetDeviceAssignment_Args_STRUCT_SIZE {
					t.Error("wrong struct size")
				}
				if name == "error" {
					return &ErrorHandle{}
				}
				a.Serialized_bytes, a.Serialized_bytes_size = &data[0], uintptr(len(data))
				a.Serialized_device_assignment, a.Serialized_device_assignment_deleter = owner, free
				switch name {
				case "empty":
					a.Serialized_bytes = nil
					a.Serialized_bytes_size = 0
				case "nil bytes":
					a.Serialized_bytes = nil
				case "overflow":
					a.Serialized_bytes_size = ^uintptr(0)
				case "nil owner":
					a.Serialized_device_assignment = nil
				case "nil deleter":
					a.Serialized_device_assignment_deleter = 0
				}
				return nil
			})
			e := &ExecutableWrapper{api: api, executable: &LoadedExecutable{}}
			got, err := e.SerializedDeviceAssignment()
			if calls != 1 {
				t.Fatal("did not dispatch")
			}
			switch name {
			case "valid":
				if err != nil || !bytes.Equal(got, []byte{8, 2, 16, 3}) {
					t.Fatalf("got %v, %v", got, err)
				}
			case "empty":
				if err != nil || len(got) != 0 {
					t.Fatalf("got %v, %v", got, err)
				}
			default:
				if err == nil || got != nil {
					t.Fatalf("got %v, %v", got, err)
				}
			}
			wantFrees := 1
			if name == "nil owner" || name == "nil deleter" || name == "error" {
				wantFrees = 0
			}
			if frees != wantFrees {
				t.Fatalf("frees=%d, want %d", frees, wantFrees)
			}
			if name == "error" && destroys != 1 {
				t.Fatal("native error was not destroyed")
			}
		})
	}
}

func TestAssignmentUnavailable(t *testing.T) {
	for _, c := range []*ClientWrapper{nil, {}, {api: &Api{}, client: &Client{}}} {
		if _, err := c.DefaultDeviceAssignment(1, 1); err == nil {
			t.Fatal("missing client/function accepted")
		}
	}
	for _, e := range []*ExecutableWrapper{nil, {}, {api: &Api{}, executable: &LoadedExecutable{}}} {
		if _, err := e.SerializedDeviceAssignment(); err == nil {
			t.Fatal("missing executable/function accepted")
		}
	}
}

func ExampleClientWrapper_DefaultDeviceAssignment() {
	var client ClientWrapper
	_, err := client.DefaultDeviceAssignment(0, 1)
	fmt.Println(err)
	// Output: openxla: replica and partition counts must be positive C ints
}

func ExampleExecutableWrapper_SerializedDeviceAssignment() {
	var executable ExecutableWrapper
	_, err := executable.SerializedDeviceAssignment()
	fmt.Println(err)
	// Output: openxla: executable is closed
}

func TestNativeDeviceAssignment(t *testing.T) {
	path := os.Getenv("XLA_PJRT_CPU_PLUGIN")
	if path == "" {
		t.Skip("set XLA_PJRT_CPU_PLUGIN to a CPU PJRT library")
	}
	plugin, err := OpenPlugin(path)
	if err != nil {
		t.Fatal(err)
	}
	client, err := plugin.NewClient()
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := client.Destroy(); err != nil {
			t.Error(err)
		}
	}()
	ids, err := client.DefaultDeviceAssignment(1, 1)
	if err != nil || len(ids) != 1 || ids[0] < 0 {
		t.Fatalf("default assignment %v, %v", ids, err)
	}
	t.Logf("default assignment: %v", ids)
	executable, err := client.Compile([]byte(`module {
 func.func @main(%arg0: tensor<1xf32>) -> tensor<1xf32> { return %arg0 : tensor<1xf32> }
}`), "mlir", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := executable.Destroy(); err != nil {
			t.Error(err)
		}
	}()
	t.Run("serialized", func(t *testing.T) {
		if executable.api.LoadedExecutable_GetDeviceAssignment == 0 {
			t.Skip("CPU plugin does not expose serialized device assignments")
		}
		first, err := executable.SerializedDeviceAssignment()
		if err != nil {
			t.Fatal(err)
		}
		if len(first) == 0 {
			t.Fatal("empty device assignment proto")
		}
		want := bytes.Clone(first)
		clear(first)
		second, err := executable.SerializedDeviceAssignment()
		if err != nil || !bytes.Equal(second, want) {
			t.Fatalf("assignment copies differ: %x, want %x; %v", second, want, err)
		}
		t.Logf("serialized assignment: %x", second)
	})
}
