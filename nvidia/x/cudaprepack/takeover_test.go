package cudaprepack

import "testing"

func TestTakeoverRejectsUnknownType(t *testing.T) {
	if _, err := NewLayoutTransformer(ArchSM121a, DataType("future")); err == nil {
		t.Fatal("accepted unknown data type")
	}
}

func TestTakeoverRejectsExtentOverflow(t *testing.T) {
	lt, err := NewLayoutTransformer(ArchSM121a, TypeFloat32)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := lt.Transform2D(nil, int(^uint(0)>>1), int(^uint(0)>>1)); err == nil {
		t.Fatal("accepted overflowing dimensions")
	}
	if _, err := lt.Transform3D(nil, int(^uint(0)>>1), 2, 2); err == nil {
		t.Fatal("accepted overflowing 3D dimensions")
	}
}

func TestTakeoverLayoutTailPreserved(t *testing.T) {
	lt, err := NewLayoutTransformer(ArchSM121a, TypeFP8_E4M3)
	if err != nil {
		t.Fatal(err)
	}
	src := []byte{1, 2, 3, 4, 5}
	p, err := lt.Transform2D(src, 1, len(src))
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range src {
		if p.Payload[i] != want {
			t.Fatalf("tail byte %d = %d, want %d", i, p.Payload[i], want)
		}
	}
}
