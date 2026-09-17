package nvjitlinkx

import (
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudakernel"
)

func TestOptions(t *testing.T) {
	var o linkOptions
	for _, opt := range []Option{WithArch("sm_90"), WithLTO(), WithPTX(), WithMaxRegisters(64), WithVerbose()} {
		if err := opt(&o); err != nil {
			t.Fatal(err)
		}
	}
	want := []string{"-arch=sm_90", "-lto", "-ptx", "-maxrregcount=64", "-verbose"}
	if !reflect.DeepEqual(o.flags, want) {
		t.Fatalf("flags = %q, want %q", o.flags, want)
	}
	for _, opt := range []Option{WithArch("bad"), WithMaxRegisters(-1), WithFlag("-Werror")} {
		if err := opt(&o); err == nil {
			t.Fatal("invalid option accepted")
		}
	}
}

func TestInputFromArtifactNil(t *testing.T) {
	if _, err := InputFromArtifact("unit", nil); err == nil {
		t.Fatal("nil artifact accepted")
	}
}

func TestInputFromKernelArtifact(t *testing.T) {
	if _, err := InputFromKernelArtifact("unit", nil); err == nil {
		t.Fatal("nil artifact accepted")
	}
	if _, err := InputFromKernelArtifact("", &cudakernel.Artifact{}); err == nil {
		t.Fatal("empty name accepted")
	}

	schema, _ := cudakernel.NewSchema(cudakernel.Param{Name: "p", Type: "u32"})
	ltoArt, err := cudakernel.NewArtifact(cudakernel.KindLTOIR, []byte("lto-data"), "entry", schema, "sm_90")
	if err != nil {
		t.Fatalf("NewArtifact LTOIR: %v", err)
	}

	in, err := InputFromKernelArtifact("module.lto", ltoArt)
	if err != nil {
		t.Fatalf("InputFromKernelArtifact: %v", err)
	}
	if in.Name != "module.lto" {
		t.Errorf("in.Name = %q, want %q", in.Name, "module.lto")
	}
	if string(in.Data) != "lto-data" {
		t.Errorf("in.Data = %q, want %q", string(in.Data), "lto-data")
	}

	// TileIR cannot be linked via nvjitlink
	tileArt, err := cudakernel.NewArtifact(cudakernel.KindTileIR, []byte("tile-data"), "entry", schema, "sm_121a")
	if err != nil {
		t.Fatalf("NewArtifact TileIR: %v", err)
	}
	if _, err := InputFromKernelArtifact("module.tile", tileArt); err == nil {
		t.Fatal("accepted TileIR for linking, expected error")
	}
}
