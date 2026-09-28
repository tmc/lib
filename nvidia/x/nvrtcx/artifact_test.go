package nvrtcx

import (
	"bytes"
	"fmt"
	"reflect"
	"strings"
	"testing"
)

func TestArtifactCache(t *testing.T) {
	for _, cm := range []*CompiledModule{
		{output: OutputAuto, ptx: "ptx", cubin: []byte{1}},
		{output: OutputPTX, ptx: "ptx"},
		{output: OutputCUBIN, cubin: []byte{1}},
		{output: OutputCUBIN, ptx: "ptx", cubin: []byte{1}},
		{output: OutputLTOIR, ltoir: []byte{2}},
		{output: OutputTileIR, tileir: []byte{3}},
		{output: OutputPTXTileIR, ptx: "ptx", tileir: []byte{3}},
		{output: OutputOptiXIR, optixir: []byte{4}},
	} {
		t.Run(fmt.Sprint(cm.output), func(t *testing.T) {
			cm.key, cm.log = "key", "compiler warning"
			encoded, err := encodeCacheEntry(cm)
			if err != nil {
				t.Fatal(err)
			}
			got, err := decodeCacheEntry(cm.key, encoded)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, cm) {
				t.Fatalf("got %#v, want %#v", got, cm)
			}
			if len(cm.ltoir) > 0 {
				copy := got.LTOIR()
				copy[0]++
				if !bytes.Equal(got.LTOIR(), cm.ltoir) {
					t.Fatal("LTOIR aliases cached data")
				}
				if _, err := got.Load(nil); err == nil {
					t.Fatal("unlinked LTOIR loaded")
				}
			}
			if len(cm.tileir) > 0 {
				copy := got.TileIR()
				copy[0]++
				if !bytes.Equal(got.TileIR(), cm.tileir) {
					t.Fatal("TileIR aliases cached data")
				}
			}
			if len(cm.optixir) > 0 {
				copy := got.OptiXIR()
				copy[0]++
				if !bytes.Equal(got.OptiXIR(), cm.optixir) {
					t.Fatal("OptiXIR aliases cached data")
				}
				if _, err := got.Load(nil); err == nil || err.Error() != "nvrtcx: OptiX IR must be passed to optixModuleCreate" {
					t.Fatalf("got err %v, want OptiX IR load rejection", err)
				}
				if _, err := got.LoadOutput(nil, OutputOptiXIR); err == nil || err.Error() != "nvrtcx: OptiX IR must be passed to optixModuleCreate" {
					t.Fatalf("got err %v, want OptiX IR load output rejection", err)
				}
			}
		})
	}
	for _, cm := range []*CompiledModule{
		{},
		{output: OutputLTOIR, ptx: "wrong"},
		{output: OutputPTX, tileir: []byte{1}},
		{output: OutputPTXTileIR, ptx: "ptx"},
		{output: OutputPTXTileIR, tileir: []byte{1}},
		{output: OutputOptiXIR, ptx: "wrong"},
		{output: OutputOptiXIR, cubin: []byte{1}},
		{output: 255, cubin: []byte{1}},
	} {
		if _, err := encodeCacheEntry(cm); err == nil {
			t.Fatalf("accepted invalid artifact %#v", cm)
		}
	}
}

func TestOutputFlags(t *testing.T) {
	for _, kind := range []OutputKind{OutputAuto, OutputPTX, OutputCUBIN, OutputLTOIR, OutputTileIR, OutputOptiXIR} {
		var input []string
		if kind == OutputCUBIN {
			input = []string{"--gpu-architecture=sm_80"}
		}
		flags, err := outputFlags(kind, input)
		if err != nil {
			t.Fatal(err)
		}
		if kind == OutputLTOIR && !reflect.DeepEqual(flags, []string{"-dlto"}) {
			t.Fatal(flags)
		}
		if kind == OutputTileIR && !reflect.DeepEqual(flags, []string{"--tile-only"}) {
			t.Fatal(flags)
		}
		if kind == OutputOptiXIR && !reflect.DeepEqual(flags, []string{"--optix-ir"}) {
			t.Fatal(flags)
		}
	}
	for _, flag := range []string{"-dlto", "--dlink-time-opt", "--tile-only", "--optix-ir"} {
		if _, err := outputFlags(OutputPTX, []string{flag}); err == nil {
			t.Fatalf("accepted conflicting flag %s", flag)
		}
	}
	if _, err := Compile("source", WithOutput(255)); err == nil {
		t.Fatal("accepted unknown output")
	}
}

func TestCUBINRequiresPhysicalTarget(t *testing.T) {
	for _, tt := range []struct {
		name  string
		flags []string
		valid bool
	}{
		{"default", nil, false},
		{"virtual", []string{"--gpu-architecture=compute_80"}, false},
		{"physical", []string{"--gpu-architecture=sm_80"}, true},
		{"specific", []string{"--gpu-architecture=sm_121a"}, true},
		{"short", []string{"-arch=sm_80"}, true},
		{"separate", []string{"--gpu-architecture", "sm_80"}, true},
		{"short separate", []string{"-arch", "sm_80"}, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			_, err := outputFlags(OutputCUBIN, tt.flags)
			if tt.valid {
				if err != nil {
					t.Fatal(err)
				}
			} else if err == nil || !strings.Contains(err.Error(), "requires a physical architecture") {
				t.Fatalf("error = %v, want physical architecture diagnostic", err)
			}
		})
	}
	for _, opts := range [][]Option{
		{WithArch(12, 1), WithOutput(OutputCUBIN)},
		{WithOutput(OutputCUBIN), WithArch(12, 1)},
	} {
		// No compiler is needed to diagnose an incompatible output request.
		_, err := Compile("extern \"C\" __global__ void empty() {}", opts...)
		if err == nil || !strings.Contains(err.Error(), "requires a physical architecture") {
			t.Fatalf("Compile error = %v, want physical architecture diagnostic", err)
		}
	}
}

func ExampleWithOutput() {
	// Pass this option to Compile with source and an appropriate target.
	option := WithOutput(OutputLTOIR)
	fmt.Println(option != nil)
	// Output: true
}

func TestMixedOutputSelection(t *testing.T) {
	flags, err := outputFlags(OutputPTXTileIR, []string{"--std=c++20"})
	if err != nil || !reflect.DeepEqual(flags, []string{"--std=c++20", "--enable-tile"}) {
		t.Fatalf("mixed flags = %v, %v", flags, err)
	}
	for _, flag := range []string{"--tile-only", "-dlto", "--simt-only"} {
		if _, err := outputFlags(OutputPTXTileIR, []string{flag}); err == nil {
			t.Fatalf("accepted %s", flag)
		}
	}
	cm := &CompiledModule{output: OutputPTXTileIR, ptx: "ptx", tileir: []byte{1}}
	if _, err := cm.Load(nil); err == nil || err.Error() != "nvrtcx: mixed output requires LoadOutput" {
		t.Fatalf("Load = %v", err)
	}
	for _, kind := range []OutputKind{OutputAuto, OutputCUBIN, OutputLTOIR, OutputPTXTileIR, 255} {
		if _, err := cm.LoadOutput(nil, kind); err == nil {
			t.Fatalf("accepted unavailable or non-loadable output %d", kind)
		}
	}
}

func ExampleCompiledModule_LoadOutput() {
	// A compiled result must contain the selected artifact.
	cm := new(CompiledModule)
	_, err := cm.LoadOutput(nil, OutputPTX)
	fmt.Println(err)
	// Output: nvrtcx: output 1 is unavailable
}
