package nvfatbinx

import "testing"

func TestBuilderInputValidation(t *testing.T) {
	for _, tc := range []struct {
		name string
		in   Input
		ok   bool
	}{
		{"ptx", Input{Kind: PTX, Name: "k", Arch: "90", Data: []byte("ptx")}, true},
		{"blackwell", Input{Kind: CUBIN, Name: "k", Arch: "100a", Data: []byte("cubin")}, true},
		{"unknown arch", Input{Kind: PTX, Name: "k", Arch: "xyz", Data: []byte("ptx")}, false},
		{"bad kind", Input{Kind: 99, Name: "k", Arch: "90", Data: []byte("ptx")}, false},
		{"cubin options", Input{Kind: CUBIN, Name: "k", Arch: "90", Data: []byte("cubin"), JITOptions: "-O3"}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var b Builder
			err := b.AddInput(tc.in)
			if (err == nil) != tc.ok {
				t.Fatalf("AddInput error = %v, want ok=%t", err, tc.ok)
			}
		})
	}
}

func TestBuilderCopiesInput(t *testing.T) {
	data := []byte("ptx")
	var b Builder
	if err := b.AddInput(Input{Kind: PTX, Name: "k", Arch: "90", Data: data}); err != nil {
		t.Fatal(err)
	}
	data[0] = 'X'
	if string(b.inputs[0].Data) != "ptx" {
		t.Fatalf("builder data = %q", b.inputs[0].Data)
	}
}
