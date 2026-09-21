package cudatma

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
)

func TestElementTypeDriverType(t *testing.T) {
	for _, tt := range []struct {
		name         string
		elementType  ElementType
		elementBytes uint32
		want         driver.TensorMapDataType
		wantErr      string
	}{
		{"default_2b_is_float16", ElementTypeDefault, 2, driver.TensorMapDataTypeFloat16, ""},
		{"default_4b_is_float32", ElementTypeDefault, 4, driver.TensorMapDataTypeFloat32, ""},
		{"default_8b_rejected", ElementTypeDefault, 8, 0, "no admitted driver format"},
		{"explicit_float16", ElementTypeFloat16, 2, driver.TensorMapDataTypeFloat16, ""},
		{"explicit_bfloat16", ElementTypeBfloat16, 2, driver.TensorMapDataTypeBfloat16, ""},
		{"explicit_float32", ElementTypeFloat32, 4, driver.TensorMapDataTypeFloat32, ""},
		{"unknown_rejected", ElementType(9), 2, 0, "unsupported element type"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.elementType.driverType(tt.elementBytes)
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("driverType = %v, %v, want an error containing %q", got, err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("driverType: %v", err)
			}
			if got != tt.want {
				t.Errorf("driverType = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestElementTypeBfloat16DiffersFromFloat16 is the point of the whole field:
// the two share a width, so only an explicit type separates them.
func TestElementTypeBfloat16DiffersFromFloat16(t *testing.T) {
	bf, err := ElementTypeBfloat16.driverType(2)
	if err != nil {
		t.Fatalf("driverType: %v", err)
	}
	f16, err := ElementTypeFloat16.driverType(2)
	if err != nil {
		t.Fatalf("driverType: %v", err)
	}
	if bf == f16 {
		t.Fatalf("bfloat16 and float16 both encode as %v", bf)
	}
}

func TestDescriptorValidateElementType(t *testing.T) {
	base := func(et ElementType, bytes uint32) Descriptor {
		d := descriptor()
		d.ElementType = et
		d.ElementBytes = bytes
		return d
	}
	for _, tt := range []struct {
		name    string
		desc    Descriptor
		wantErr string
	}{
		{"default_2b", base(ElementTypeDefault, 2), ""},
		{"bfloat16_2b", base(ElementTypeBfloat16, 2), ""},
		{"float16_2b", base(ElementTypeFloat16, 2), ""},
		{"float32_declared_2b", base(ElementTypeFloat32, 2), "is 4 bytes, not the 2 declared"},
		{"unknown_type", base(ElementType(9), 2), "unsupported element type"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.desc.Validate()
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("Validate: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("Validate = %v, want an error containing %q", err, tt.wantErr)
			}
		})
	}
}

func TestElementTypeString(t *testing.T) {
	for _, tt := range []struct {
		elementType ElementType
		want        string
	}{
		{ElementTypeDefault, "default"},
		{ElementTypeFloat16, "float16"},
		{ElementTypeBfloat16, "bfloat16"},
		{ElementTypeFloat32, "float32"},
		{ElementType(9), "ElementType(9)"},
	} {
		if got := tt.elementType.String(); got != tt.want {
			t.Errorf("ElementType(%d).String() = %q, want %q", uint8(tt.elementType), got, tt.want)
		}
	}
}
