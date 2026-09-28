package cudalaunch

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
)

func TestSignatureFromPTX(t *testing.T) {
	const ptx = `.version 8.0
.target sm_90
.address_size 64
.visible .entry add(.param .u64 add_param_0, .param .u32 add_param_1) {
	ret;
}`
	sig, err := SignatureFromPTX(ptx, "add")
	if err != nil {
		t.Fatal(err)
	}
	if len(sig.Params) != 2 || sig.Params[0].Size != 8 || sig.Params[1].Size != 4 {
		t.Fatalf("signature = %+v", sig)
	}
	for _, tc := range []struct {
		name string
		args []any
		want string
	}{
		{"valid", []any{driver.DevicePtr(0x1000), uint32(3)}, ""},
		{"count", []any{driver.DevicePtr(0x1000)}, "count mismatch"},
		{"width", []any{driver.DevicePtr(0x1000), uint64(3)}, "size mismatch"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := Validate(sig, tc.args...)
			if tc.want == "" && err != nil {
				t.Fatal(err)
			}
			if tc.want != "" && (err == nil || !strings.Contains(err.Error(), tc.want)) {
				t.Fatalf("error = %v, want %q", err, tc.want)
			}
		})
	}
}

func TestLaunchSignatureRejectsBeforeDriver(t *testing.T) {
	sig := Signature{KernelName: "add", Params: []Param{{Name: "n", Type: "u32", Size: 4}}}
	l := Kernel(driver.Function(1)).WithSignature(sig)
	sig.Params[0].Size = 8
	if err := l.Launch(uint64(3)); err == nil || !strings.Contains(err.Error(), "size mismatch") {
		t.Fatalf("Launch error = %v, want size mismatch", err)
	}
}

// Equal-width arguments must still be checked before the driver sees them.
func TestLaunchSignatureRejectsFloatForInteger(t *testing.T) {
	for _, typ := range []string{"u32", "s32"} {
		t.Run(typ, func(t *testing.T) {
			sig := Signature{KernelName: "vecadd", Params: []Param{{Name: "n", Type: typ, Size: 4}}}
			if err := Validate(sig, int32(1024)); err != nil {
				t.Fatal(err)
			}
			l := Kernel(driver.Function(1)).WithSignature(sig)
			if err := l.Launch(float32(1024)); err == nil || !strings.Contains(err.Error(), "type mismatch") {
				t.Fatalf("Launch error = %v, want type mismatch before driver call", err)
			}
		})
	}
}

func TestSignatureBitStorage(t *testing.T) {
	sig := Signature{Params: []Param{{Type: "b32", Size: 4}}}
	for _, arg := range []any{float32(1), int32(1)} {
		if err := Validate(sig, arg); err != nil {
			t.Fatalf("untyped b32 storage rejected %T: %v", arg, err)
		}
	}
	if err := Validate(sig, float64(1)); err == nil {
		t.Fatal("accepted wrong storage width")
	}
}
