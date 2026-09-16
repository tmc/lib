package nvjitlinkx

import (
	"reflect"
	"testing"
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
