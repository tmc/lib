package cudalineinfo

import (
	"encoding/hex"
	"os"
	"strings"
	"testing"
)

// FuzzNewResolverNeverPanics exercises the ELF bytes NewResolver accepts from
// callers. It starts with a real CUBIN built from the checked-in round-trip
// fixture, plus malformed and truncated forms of that same image.
func FuzzNewResolverNeverPanics(f *testing.F) {
	const cubinHex = "../cudasass/testdata/roundtrip/sm121a-roundtrip.cubin.hex"
	encoded, err := os.ReadFile(cubinHex)
	if err != nil {
		f.Fatal(err)
	}
	cubin, err := hex.DecodeString(strings.Join(strings.Fields(string(encoded)), ""))
	if err != nil {
		f.Fatal(err)
	}
	if len(cubin) == 0 {
		f.Fatal("round-trip fixture has an empty CUBIN")
	}

	a := &Attribution{}
	f.Add([]byte(nil))
	f.Add([]byte("not an ELF image"))
	f.Add(cubin[:min(len(cubin), 64)])
	f.Add(cubin[:len(cubin)-1])
	f.Add(cubin)

	f.Fuzz(func(t *testing.T, data []byte) {
		// Limit supplied input bytes; this does not bound declared ELF sizes or
		// parser work.
		if len(data) > 1<<20 {
			t.Skip()
		}
		resolver, err := NewResolver(a, data)
		if err != nil {
			return
		}
		if resolver == nil {
			t.Fatal("NewResolver returned a nil resolver without an error")
		}
	})
}

func TestNewResolverAcceptsCheckedInCUBIN(t *testing.T) {
	data, err := os.ReadFile("../cudasass/testdata/roundtrip/sm121a-roundtrip.cubin.hex")
	if err != nil {
		t.Fatal(err)
	}
	cubin, err := hex.DecodeString(strings.Join(strings.Fields(string(data)), ""))
	if err != nil {
		t.Fatal(err)
	}
	r, err := NewResolver(loadFixture(t, "fused_gate", ""), cubin)
	if err != nil {
		t.Fatal(err)
	}
	if r == nil {
		t.Fatal("NewResolver returned nil without an error")
	}
	if got := r.text["intmix"]; got == 0 {
		t.Fatalf("checked-in CUBIN has no .text.intmix offset: %v", r.text)
	}
}

func TestNewResolverRejectsMalformedCUBIN(t *testing.T) {
	data, err := os.ReadFile("../cudasass/testdata/roundtrip/sm121a-roundtrip.cubin.hex")
	if err != nil {
		t.Fatal(err)
	}
	cubin, err := hex.DecodeString(strings.Join(strings.Fields(string(data)), ""))
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name string
		data []byte
	}{
		{name: "non-ELF", data: []byte("not an ELF image")},
		{name: "short header", data: cubin[:min(len(cubin), 64)]},
		{name: "truncated image", data: cubin[:len(cubin)-1]},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if r, err := NewResolver(&Attribution{}, tc.data); err == nil || r != nil {
				t.Fatalf("NewResolver() = (%v, %v), want an error and no resolver", r, err)
			}
		})
	}
}
