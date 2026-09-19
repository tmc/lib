package cudalayout

import (
	"errors"
	"testing"
)

// TestErrorRefinementsWrap checks the relation the two refinements promise: a
// caller testing for ErrUnsupported keeps matching, and a caller testing for
// one refinement does not match the other.
func TestErrorRefinementsWrap(t *testing.T) {
	for _, err := range []error{ErrUnmodeled, ErrUninitialized} {
		if !errors.Is(err, ErrUnsupported) {
			t.Errorf("%v does not wrap ErrUnsupported", err)
		}
	}
	if errors.Is(ErrUnmodeled, ErrUninitialized) || errors.Is(ErrUninitialized, ErrUnmodeled) {
		t.Error("the refinements match one another")
	}
	if errors.Is(ErrUnsupported, ErrUnmodeled) {
		t.Error("plain ErrUnsupported matches ErrUnmodeled")
	}
}

// TestUnmodeledIsNotEveryRejection is the gate on ErrUnmodeled's meaning. It
// says the hardware has a layout this package does not describe, so a merely
// invalid description must not carry it. If every rejection were tagged
// ErrUnmodeled the sentinel would say nothing.
func TestUnmodeledIsNotEveryRejection(t *testing.T) {
	unmodeled := []struct {
		name string
		call func() error
	}{
		// fp8 base tiles exist; this package has no verified mapping.
		// The one-byte row layout is measured; the column layout is not,
		// and a 16x32 base tile is not a square tile's transpose.
		{"one-byte column fragment", func() error {
			_, err := NewFragment(1, FragmentCol)
			return err
		}},
		// sm_90a is a real target with no fixture written for it.
		{"unadmitted fixture target", func() error {
			tile, err := NewTile(16, 16, 2)
			if err != nil {
				t.Fatal(err)
			}
			_, err = FixtureSourceForTarget(tile, "sm_90a")
			return err
		}},
	}
	for _, tt := range unmodeled {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.call()
			if !errors.Is(err, ErrUnmodeled) {
				t.Errorf("error = %v, want ErrUnmodeled", err)
			}
		})
	}

	// These are not unmodeled: no hardware has them, and the caller's
	// description is wrong.
	invalid := []struct {
		name string
		call func() error
	}{
		{"eight-byte fragment", func() error {
			_, err := NewFragment(8, FragmentRow)
			return err
		}},
		{"unknown fragment layout", func() error {
			_, err := NewFragment(4, FragmentLayout(7))
			return err
		}},
		{"swizzle span 16", func() error {
			_, err := TMASwizzleFromBytes(16)
			return err
		}},
		{"row straddles the span", func() error {
			_, err := NewSwizzledTile(16, 48, 2, TMASwizzle64B)
			return err
		}},
	}
	for _, tt := range invalid {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.call()
			if !errors.Is(err, ErrUnsupported) {
				t.Fatalf("error = %v, want ErrUnsupported", err)
			}
			if errors.Is(err, ErrUnmodeled) {
				t.Errorf("error = %v, tagged ErrUnmodeled; an invalid description is not unmodeled", err)
			}
		})
	}
}

// TestUninitializedOnZeroValues checks that a zero value reports never having
// been constructed rather than describing an unsupported layout.
func TestUninitializedOnZeroValues(t *testing.T) {
	var (
		tile     Tile
		fragment Fragment
		swizzled SwizzledTile
	)
	calls := []struct {
		name string
		call func() error
	}{
		{"Tile.FootprintBytes", func() error { _, err := tile.FootprintBytes(); return err }},
		{"Tile.OwnerFor", func() error { _, _, err := tile.OwnerFor(0, 0); return err }},
		{"Fragment.Owner", func() error { _, err := fragment.Owner(0, 0); return err }},
		{"Fragment.Element", func() error { _, _, err := fragment.Element(FragmentOwner{}); return err }},
		{"SwizzledTile.ByteOffset", func() error { _, err := swizzled.ByteOffset(0, 0); return err }},
	}
	for _, tt := range calls {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.call()
			if !errors.Is(err, ErrUninitialized) {
				t.Errorf("error = %v, want ErrUninitialized", err)
			}
			if errors.Is(err, ErrUnmodeled) {
				t.Errorf("error = %v, tagged ErrUnmodeled", err)
			}
		})
	}
}
