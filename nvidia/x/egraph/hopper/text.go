package hopper

import "fmt"

// The text forms below let kernel specs round-trip through JSON
// (cmd/egvariants reads them).

// The zero DType (no operand) is the empty string.
func (d DType) MarshalText() ([]byte, error) {
	if d == 0 {
		return nil, nil
	}
	return []byte(d.String()), nil
}

func (d *DType) UnmarshalText(b []byte) error {
	if len(b) == 0 {
		*d = 0
		return nil
	}
	v, err := ParseDType(string(b))
	if err != nil {
		return err
	}
	*d = v
	return nil
}

func (f WeightFormat) MarshalText() ([]byte, error) { return []byte(f.String()), nil }

func (f *WeightFormat) UnmarshalText(b []byte) error {
	for _, v := range []WeightFormat{WeightBF16, WeightBF16HiLo, WeightNVFP4, WeightE4M3} {
		if v.String() == string(b) {
			*f = v
			return nil
		}
	}
	return fmt.Errorf("unknown weight format %q", b)
}

func (b Broadcast) String() string {
	switch b {
	case PerCol:
		return "col"
	case PerRow:
		return "row"
	case Full:
		return "full"
	}
	return fmt.Sprintf("Broadcast(%d)", uint8(b))
}

func (b Broadcast) MarshalText() ([]byte, error) { return []byte(b.String()), nil }

func (b *Broadcast) UnmarshalText(t []byte) error {
	for _, v := range []Broadcast{PerCol, PerRow, Full} {
		if v.String() == string(t) {
			*b = v
			return nil
		}
	}
	return fmt.Errorf("unknown broadcast %q", t)
}

func (l Lambda) MarshalText() ([]byte, error) { return []byte(l.String()), nil }

// The zero Lambda (the default) is the empty string.
func (l *Lambda) UnmarshalText(b []byte) error {
	if len(b) == 0 {
		*l = Lambda{}
		return nil
	}
	v, err := ParseLambda(string(b))
	if err != nil {
		return err
	}
	*l = v
	return nil
}
