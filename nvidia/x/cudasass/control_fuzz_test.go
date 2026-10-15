package cudasass

import (
	"errors"
	"testing"
)

// FuzzSASSDecodeControlIsTotalOverEncodingDomain checks DecodeControl and
// EncodeControl on every encoding and architecture spelling: a decode is
// reported exactly for a captured sm_121a encoding, a refused decode returns a
// typed error and zero ControlCode, a reported one is Valid and re-encodes to
// the encoding it came from, and an encode touches only the control field.
func FuzzSASSDecodeControlIsTotalOverEncodingDomain(f *testing.F) {
	f.Add(uint64(0x0000df00ff017b82), uint64(0x000fe20000000800), string(SM121a), uint8(0), false, uint8(0), uint8(0), uint8(0), uint8(0))
	f.Add(uint64(0x000000000000094d), uint64(0x000fea0003800000), string(SM121a), uint8(15), true, uint8(7), uint8(0), uint8(0x3f), uint8(0xf))
	f.Add(uint64(0), uint64(0), string(SM121a), uint8(1), true, uint8(2), uint8(3), uint8(4), uint8(5))
	f.Add(uint64(1), uint64(^uint64(0)), "sm_90", uint8(16), false, uint8(8), uint8(8), uint8(0x40), uint8(0x10))
	f.Fuzz(func(t *testing.T, lo, hi uint64, arch string, stall uint8, yield bool, wr, rd, wait, reuse uint8) {
		e := Encoding{Lo: lo, Hi: hi}
		a := Arch(arch)
		c, decodeErr := DecodeControl(e, a)
		if want := a == SM121a && e.Present(); (decodeErr == nil) != want {
			t.Fatalf("DecodeControl(%v, %q) error = %v, want success %v", e, a, decodeErr, want)
		}
		if decodeErr != nil {
			var refusal *ControlDecodeError
			if !errors.As(decodeErr, &refusal) || refusal.Arch != a || refusal.MissingEncoding != (a == SM121a) {
				t.Fatalf("DecodeControl(%v, %q) error = %v, want typed refusal", e, a, decodeErr)
			}
			if c != (ControlCode{}) {
				t.Fatalf("DecodeControl(%v, %q) refused but returned %v", e, a, c)
			}
		} else {
			if err := c.Valid(); err != nil {
				t.Fatalf("DecodeControl(%v) = %v: %v", e, c, err)
			}
			if got, err := EncodeControl(e, c, a); err != nil || got != e {
				t.Fatalf("EncodeControl(%v, DecodeControl) = %v, %v; want the encoding back", e, got, err)
			}
		}

		want := ControlCode{Stall: stall, Yield: yield, WriteBarrier: wr, ReadBarrier: rd, WaitMask: wait, Reuse: reuse}
		got, err := EncodeControl(e, want, a)
		switch {
		case a != SM121a || want.Valid() != nil:
			if err == nil || got != (Encoding{}) {
				t.Fatalf("EncodeControl(%v, %v, %q) = %v, %v; want absence and an error", e, want, a, got, err)
			}
		case err != nil:
			t.Fatalf("EncodeControl(%v, %v) = %v", e, want, err)
		default:
			// The observed control field runs from the stall bits through
			// the reuse bits; the bits above it are preserved.
			const observed = (1<<(reuseShift+reuseWidth) - 1) &^ (1<<ctrlShift - 1)
			if got.Lo != e.Lo || got.Hi&^observed != e.Hi&^observed {
				t.Fatalf("EncodeControl(%v, %v) = %v: changed bits outside the observed control field", e, want, got)
			}
			back, err := DecodeControl(got, a)
			if !got.Present() {
				if err == nil {
					t.Fatalf("DecodeControl of the zero encoding reported %v", back)
				}
				return
			}
			if err != nil || back != want {
				t.Fatalf("DecodeControl(EncodeControl(%v, %v)) = %v, %v", e, want, back, err)
			}
		}
	})
}

// TestDecodeControlExhaustiveControlField decodes every value of the observed
// control bits and checks that decode is a bijection from them onto the field
// tuples, and that each value re-encodes to itself. The unobserved bits above
// the field take every value along the way, so a decoder that reads them, or
// an encoder that clears them, fails too.
func TestDecodeControlExhaustiveControlField(t *testing.T) {
	const (
		observedBits = reuseShift + reuseWidth - ctrlShift
		tupleBits    = stallWidth + 1 + 2*barWidth + waitWidth + reuseWidth
		otherHi      = uint64(0x123456789ab) & (1<<ctrlShift - 1)
		failureLimit = 10
	)
	if observedBits != tupleBits {
		t.Fatalf("%d observed control bits hold %d bits of fields", observedBits, tupleBits)
	}
	counts := make([]uint8, 1<<tupleBits)
	failures := 0
	for v := uint64(0); v < 1<<observedBits; v++ {
		above := v % (1 << (64 - ctrlShift - observedBits))
		e := Encoding{Lo: 1, Hi: above<<(ctrlShift+observedBits) | v<<ctrlShift | otherHi}
		c, err := DecodeControl(e, SM121a)
		if err != nil {
			t.Fatalf("DecodeControl(%v) refused a present encoding: %v", e, err)
		}
		if err := c.Valid(); err != nil {
			t.Fatalf("DecodeControl(%v) = %v: %v", e, c, err)
		}
		if got, err := EncodeControl(e, c, SM121a); (err != nil || got != e) && failures < failureLimit {
			failures++
			t.Errorf("EncodeControl(%v, %v) = %v, %v; want the encoding back", e, c, got, err)
		}
		// The key packs the decoded fields, each within its Valid width, and
		// does not depend on where decode found them.
		key := uint64(c.Stall)
		key = key<<1 | b2u(c.Yield)
		key = key<<barWidth | uint64(c.WriteBarrier)
		key = key<<barWidth | uint64(c.ReadBarrier)
		key = key<<waitWidth | uint64(c.WaitMask)
		key = key<<reuseWidth | uint64(c.Reuse)
		counts[key]++
	}
	for key, n := range counts {
		if n != 1 && failures < failureLimit {
			failures++
			t.Errorf("field tuple %#x decoded from %d values, want 1", key, n)
		}
	}
}

func b2u(b bool) uint64 {
	if b {
		return 1
	}
	return 0
}

// TestDecodeControlRefusesUnsupportedArch checks that an architecture other
// than sm_121a gets absence from both directions: no control code and no
// encoding, rather than a field read or written at sm_121a offsets.
func TestDecodeControlRefusesUnsupportedArch(t *testing.T) {
	e := Encoding{Lo: 0x0000df00ff017b82, Hi: 0x000fe20000000800}
	for _, a := range []Arch{"", "sm_90", "sm_121", "sm_120a", "SM_121A", "sm_121a ", "sm_121a\x00"} {
		if c, err := DecodeControl(e, a); err == nil || c != (ControlCode{}) {
			t.Errorf("DecodeControl(%q) = %v, %v; want absence", a, c, err)
		}
		if got, err := EncodeControl(e, ControlCode{}, a); err == nil || got != (Encoding{}) {
			t.Errorf("EncodeControl(%q) = %v, %v; want absence and an error", a, got, err)
		}
	}
	if c, err := DecodeControl(Encoding{}, SM121a); err == nil || c != (ControlCode{}) {
		t.Errorf("DecodeControl(no encoding) = %v, %v; want absence", c, err)
	}
}
