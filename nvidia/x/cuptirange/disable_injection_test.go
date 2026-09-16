package cuptirange

import (
	"errors"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

type disableAttempt struct {
	Size           uint
	Object         uintptr
	ReturnedSize   uint
	ReturnedObject uintptr
	Err            error
}
type disableInjection struct {
	call     func(*cupti.RangeProfilerDisableParams) error
	attempts []disableAttempt
}

func (d *disableInjection) disable(p *cupti.RangeProfilerDisableParams) error {
	copy := *p
	if len(d.attempts) == 0 {
		copy.StructSize = 0
	}
	attempt := disableAttempt{Size: copy.StructSize, Object: copy.PRangeProfilerObject}
	err := d.call(&copy)
	attempt.ReturnedSize = copy.StructSize
	attempt.ReturnedObject = copy.PRangeProfilerObject
	attempt.Err = err
	d.attempts = append(d.attempts, attempt)
	return err // In particular, do not disguise unexpected native success.
}
func TestDisableInjection(t *testing.T) {
	for _, first := range []error{cupti.ErrInvalidParameter, nil} {
		d := disableInjection{call: func(p *cupti.RangeProfilerDisableParams) error {
			if p.StructSize == 0 {
				return first
			}
			p.PRangeProfilerObject = 0 // Native callees may modify even input-marked fields.
			return nil
		}}
		p := cupti.RangeProfilerDisableParams{PRangeProfilerObject: 123}
		p.StructSize = uint(unsafe.Sizeof(p))
		if err := d.disable(&p); !errors.Is(err, first) {
			t.Fatalf("first=%v want=%v", err, first)
		}
		if p.StructSize != 24 || p.PRangeProfilerObject != 123 {
			t.Fatalf("modified original: %+v", p)
		}
		if first != nil {
			if err := d.disable(&p); err != nil {
				t.Fatal(err)
			}
			if d.attempts[1].Size != 24 || d.attempts[1].Object != 123 || d.attempts[1].ReturnedObject != 0 {
				t.Fatal(d.attempts)
			}
		}
	}
}
