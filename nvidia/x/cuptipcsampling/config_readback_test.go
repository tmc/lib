package cuptipcsampling

import (
	"errors"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

func testAppliedSamplingConfig(t *testing.T) *samplingConfig {
	t.Helper()
	o, inventory := testSamplingOptions()
	c, err := newSamplingConfig(o, inventory, testSamplingBuffer(t))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(c.release)
	if err := c.apply(5, func(*cupti.PCSamplingConfigurationInfoParams) error {
		for i := range c.attrs {
			c.attrs[i].AttributeStatus = 0
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	return c
}

func TestSamplingConfigReadback(t *testing.T) {
	failure := errors.New("native readback failure")
	for _, name := range []string{"same", "different value", "unwritten value", "unwritten status", "native", "pointer", "type"} {
		t.Run(name, func(t *testing.T) {
			c := testAppliedSamplingConfig(t)
			err := c.verify(5, func(p *cupti.PCSamplingConfigurationInfoParams) error {
				if name == "native" {
					return failure
				}
				attrs := unsafe.Slice((*cupti.PCSamplingConfigurationInfo)(nativePointer(p.PPCSamplingConfigurationInfo)), int(p.NumAttributes))
				for i := range attrs {
					if name != "unwritten status" {
						attrs[i].AttributeStatus = 0
					}
					for _, a := range c.attrs {
						if a.AttributeType == attrs[i].AttributeType && name != "unwritten value" {
							// CUPTI writes only four bytes for uint32/enum fields.
							if a.AttributeType == cupti.PcSamplingConfigurationAttrTypeScratchBufferSize || a.AttributeType == cupti.PcSamplingConfigurationAttrTypeHardwareBufferSize {
								attrs[i].AttributeData[0] = a.AttributeData[0]
							} else {
								attrs[i].AttributeData[0] = 0xffffffff00000000 | a.AttributeData[0]
							}
						}
					}
				}
				switch name {
				case "different value":
					attrs[0].AttributeData[0]++
				case "pointer":
					p.PPCSamplingConfigurationInfo++
				case "type":
					attrs[0].AttributeType = 0
				}
				return nil
			})
			if (err == nil) != (name == "same") || c.verified != (name == "same") || len(c.effective) != 6 {
				t.Fatalf("verified=%v err=%v effective=%v", c.verified, err, c.effective)
			}
			if name == "native" && !errors.Is(err, failure) {
				t.Fatal("lost native error")
			}
		})
	}
}

func TestSamplingConfigReadbackContext(t *testing.T) {
	c := testAppliedSamplingConfig(t)
	if err := c.verify(6, func(*cupti.PCSamplingConfigurationInfoParams) error { t.Fatal("read different context"); return nil }); err == nil {
		t.Fatal("accepted another context")
	}
}
