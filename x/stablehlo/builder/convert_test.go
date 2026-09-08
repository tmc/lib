package builder

import "testing"

func TestConvert(t *testing.T) {
	for _, element := range []string{"f16", "bf16", "f32", "f64", "i1", "i32", "i64", "ui32", "ui64"} {
		t.Run(element, func(t *testing.T) {
			b := NewFunc("main", Type{Shape: []int64{-1, 4}, ElementType: "f32"})
			out, err := b.Convert(b.Arg(0), element)
			if err != nil {
				t.Fatal(err)
			}
			if out.Type.ElementType != element || out.Type.Shape[0] != -1 || out.Type.Shape[1] != 4 {
				t.Fatalf("type = %s", out.Type)
			}
			if element != "f32" {
				out.Type.Shape[1] = 9
				if b.Args[0].Type.Shape[1] != 4 {
					t.Fatal("result aliases input shape")
				}
			}
		})
	}
}

func TestConvertRejectsInvalid(t *testing.T) {
	for _, tt := range []struct {
		src, dst string
		shape    []int64
	}{
		{"invalid", "f32", nil}, {"f32", "invalid", nil}, {"f32", "f16", []int64{-2}},
	} {
		b := NewFunc("main", Type{Shape: tt.shape, ElementType: tt.src})
		if _, err := b.Convert(b.Arg(0), tt.dst); err == nil {
			t.Fatalf("accepted %v", tt)
		}
		if len(b.Ops) != 0 || b.nextID != 0 {
			t.Fatal("invalid conversion changed builder")
		}
	}
}
