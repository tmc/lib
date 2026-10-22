package dsv4vision

import "testing"

func TestSpanOwnership(t *testing.T) {
	r := newSidecarFixture().reader()
	sidecar, err := ReadSidecar(r, r.size)
	if err != nil {
		t.Fatal(err)
	}
	values := make([]float32, 4096)
	values[0] = 7
	span, err := NewSpan(sidecar, 3, 3, 0, 129280, values, [32]byte{9})
	if err != nil {
		t.Fatal(err)
	}
	tokens, rows := span.Tokens(), span.Embeddings()
	if len(tokens) != 9 || len(rows) != 9*4096 || rows[4*4096] != 7 {
		t.Fatal("wrong layout or encoder payload")
	}
	values[0] = 99
	tokens[0] = -1
	rows[4*4096] = 99
	if span.Tokens()[0] < 0 || span.Embeddings()[4*4096] != 7 || span.Fingerprint() != [32]byte{9} {
		t.Fatal("span aliases mutable input or getter storage")
	}
}

func TestSpanAdmission(t *testing.T) {
	for _, test := range []struct {
		name                        string
		height, width, start, vocab int
		values                      []float32
	}{
		{"dimensions", 0, 1, 0, 129280, nil},
		{"position", 1, 1, -1, 129280, make([]float32, 4096)},
		{"vocabulary", 1, 1, 0, 0, make([]float32, 4096)},
		{"extent", 1, 1, 0, 129280, nil},
		{"sidecar", 1, 1, 0, 129280, make([]float32, 4096)},
	} {
		t.Run(test.name, func(t *testing.T) {
			if _, err := NewSpan(nil, test.height, test.width, test.start, test.vocab, test.values, [32]byte{}); err == nil {
				t.Fatal("invalid span admitted")
			}
		})
	}
}
