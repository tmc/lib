package xidcatalog

import (
	"fmt"
	"strconv"
	"strings"
	"testing"
)

func TestValidationAllSourceMaskBits(t *testing.T) {
	for _, raw := range literalDecodeRows(t) {
		id, _ := strconv.ParseUint(raw.Cells[fmt.Sprint("A", raw.Row)], 10, 64)
		text := raw.Cells[fmt.Sprint("E", raw.Row)]
		status := uint32(0)
		if text != "" {
			n, _ := strconv.ParseUint(strings.Split(text, "/")[0][2:], 16, 32)
			status = uint32(n)
		}
		for _, v := range []struct{ driver, column string }{{"574.99.99", "C"}, {"575", "D"}} {
			for _, variant := range []struct {
				offset byte
				kind   DecodeVariant
			}{{0, PrimaryDecode}, {4, SecondaryDecode}} {
				mask := raw.Cells[string(v.column[0]+variant.offset)+strconv.Itoa(raw.Row)]
				if mask == "" {
					continue
				}
				base, _ := strconv.ParseUint(strings.ReplaceAll(mask, "-", "0"), 2, 32)
				for bit, c := range mask {
					info := uint32(base) ^ (uint32(1) << uint(31-bit))
					r := Decode(id, B100, v.driver, info, status)
					found := false
					for _, m := range r.Matches {
						if m.Rule.Row == raw.Row && m.Variant == variant.kind {
							found = true
						}
					}
					if found != (c == '-') {
						t.Fatalf("row%d driver%s variant%d bit%d literal%s unexpectedly matches=%v", raw.Row, v.driver, variant.kind, bit, mask, found)
					}
				}
			}
		}
	}
}

func TestValidationAllSourceStatusAlternatives(t *testing.T) {
	for _, raw := range literalDecodeRows(t) {
		text := raw.Cells[fmt.Sprint("E", raw.Row)]
		if text == "" {
			continue
		}
		for _, part := range strings.Split(text, "/") {
			n, err := strconv.ParseUint(part[2:], 16, 32)
			if err != nil {
				t.Fatal(err)
			}
			if !matchesStatus(text, uint32(n)) {
				t.Fatalf("row%d exact alternative%s rejected", raw.Row, part)
			}
		}
		for _, value := range []uint32{0, 1, 0x80000000, 0xffffffff} {
			want := false
			for _, part := range strings.Split(text, "/") {
				n, _ := strconv.ParseUint(part[2:], 16, 32)
				want = want || value == uint32(n)
			}
			if matchesStatus(text, value) != want {
				t.Fatalf("row%d status%x differs from literal%s", raw.Row, value, text)
			}
		}
	}
}

func TestValidationRecoveryLastToken(t *testing.T) {
	s := "Xid 154 GPU recovery action changed from 0xffffffff (Node Reboot Required) to 0x0 (None)"
	got, err := ParseRecoveryTransition(s)
	if err != nil || got.From.Value != 0xffffffff || got.To.Value != 0 || got.From.State != RecoveryNodeReboot || got.To.State != RecoveryNone || got.Message != s {
		t.Fatalf("last transition lost %+v %v", got, err)
	}
	for _, text := range []string{"Future", "None-like", "GPU Reset Required (future)"} {
		s := "Xid 154 GPU recovery action changed from 0x0 (None) to 0xff (" + text + ")"
		got, err := ParseRecoveryTransition(s)
		if strings.Contains(text, "(") {
			if err == nil {
				t.Fatal("nested ambiguous token accepted")
			}
		} else if err != nil || got.To.Recognized || got.To.Value != 255 || got.To.Text != text {
			t.Fatalf("future token lost %+v %v", got, err)
		}
	}
}

func FuzzValidationDecode(f *testing.F) {
	f.Add(uint64(144), "575.0", uint32(1), uint32(1))
	f.Fuzz(func(t *testing.T, id uint64, driver string, info, status uint32) {
		if len(driver) > 256 {
			return
		}
		r := Decode(id, B100, driver, info, status)
		if r.Driver != driver {
			t.Fatal("raw driver lost")
		}
		if r.Outcome == UnknownDriver && len(r.Matches) != 0 {
			t.Fatal("unknown driver selected masks")
		}
		for _, m := range r.Matches {
			if m.Rule.Code != id || m.Rule.Row < 2 || m.Rule.Row > 96 {
				t.Fatal("foreign source row")
			}
		}
	})
}
