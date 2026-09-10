package openxla

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/ebitengine/purego"
)

func TestCompiledMemoryStats(t *testing.T) {
	for _, mode := range []string{"values", "acquire error", "nil handle", "query error", "destroy error", "both errors"} {
		t.Run(mode, func(t *testing.T) {
			raw, loaded := &Executable{}, &LoadedExecutable{}
			queries, destroys, errorsDestroyed := 0, 0, 0
			api := &Api{}
			api.Error_Destroy = purego.NewCallback(func(*Error_Destroy_Args) { errorsDestroyed++ })
			api.LoadedExecutable_GetExecutable = purego.NewCallback(func(a *LoadedExecutable_GetExecutable_Args) *ErrorHandle {
				if a.Struct_size != PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE || a.Loaded_executable != loaded {
					t.Error("invalid acquire arguments")
				}
				if mode == "acquire error" {
					return &ErrorHandle{}
				}
				if mode != "nil handle" {
					a.Executable = raw
				}
				return nil
			})
			api.Executable_GetCompiledMemoryStats = purego.NewCallback(func(a *Executable_GetCompiledMemoryStats_Args) *ErrorHandle {
				queries++
				if a.Struct_size != PJRT_Executable_GetCompiledMemoryStats_Args_STRUCT_SIZE || a.Executable != raw || a.Extension_start != nil {
					t.Error("invalid query arguments")
				}
				v := reflect.ValueOf(a).Elem()
				for i := 3; i < v.NumField(); i++ {
					v.Field(i).SetInt(int64(100 + i))
				}
				if mode == "query error" || mode == "both errors" {
					return &ErrorHandle{}
				}
				return nil
			})
			api.Executable_Destroy = purego.NewCallback(func(a *Executable_Destroy_Args) *ErrorHandle {
				destroys++
				if a.Struct_size != PJRT_Executable_Destroy_Args_STRUCT_SIZE || a.Executable != raw {
					t.Error("invalid destroy arguments")
				}
				if mode == "destroy error" || mode == "both errors" {
					return &ErrorHandle{}
				}
				return nil
			})
			e := &ExecutableWrapper{api: api, executable: loaded}
			got, err := e.CompiledMemoryStats()
			if mode == "values" {
				if err != nil {
					t.Fatal(err)
				}
				v := reflect.ValueOf(got)
				for i := 0; i < v.NumField(); i++ {
					if v.Field(i).Int() != int64(103+i) {
						t.Errorf("%s: got %d, want %d", v.Type().Field(i).Name, v.Field(i).Int(), 103+i)
					}
				}
			} else if err == nil || got != (CompiledMemoryStats{}) {
				t.Fatalf("got %+v, %v", got, err)
			}
			wantCalls, wantErrors := 1, 0
			switch mode {
			case "nil handle":
				wantCalls = 0
			case "acquire error":
				wantCalls, wantErrors = 0, 1
			case "query error", "destroy error":
				wantErrors = 1
			case "both errors":
				wantErrors = 2
			}
			if queries != wantCalls || destroys != wantCalls || errorsDestroyed != wantErrors {
				t.Fatalf("queries=%d destroys=%d errors=%d, want %d/%d/%d", queries, destroys, errorsDestroyed, wantCalls, wantCalls, wantErrors)
			}
			if e.executable != loaded {
				t.Fatal("query changed loaded executable")
			}
		})
	}
}

func TestCompiledMemoryStatsUnavailable(t *testing.T) {
	for _, e := range []*ExecutableWrapper{nil, {}, {api: &Api{}, executable: &LoadedExecutable{}}} {
		if _, err := e.CompiledMemoryStats(); err == nil {
			t.Fatal("accepted unavailable statistics")
		}
	}
	for _, field := range []string{"LoadedExecutable_GetExecutable", "Executable_GetCompiledMemoryStats", "Executable_Destroy"} {
		api := &Api{LoadedExecutable_GetExecutable: 1, Executable_GetCompiledMemoryStats: 1, Executable_Destroy: 1}
		reflect.ValueOf(api).Elem().FieldByName(field).SetUint(0)
		e := &ExecutableWrapper{api: api, executable: &LoadedExecutable{}}
		if _, err := e.CompiledMemoryStats(); err == nil {
			t.Fatalf("accepted missing %s", field)
		}
	}
}

func ExampleCompiledMemoryStats() {
	stats := CompiledMemoryStats{ArgumentBytes: 64, OutputBytes: 32}
	fmt.Println(stats.ArgumentBytes, stats.OutputBytes)
	// Output: 64 32
}

func ExampleExecutableWrapper_CompiledMemoryStats() {
	var executable *ExecutableWrapper
	_, err := executable.CompiledMemoryStats()
	fmt.Println(err)
	// Output: openxla: executable is closed
}
