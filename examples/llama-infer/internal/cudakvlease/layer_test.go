package cudakvlease

import "testing"

func TestProduct(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	tests := []struct {
		name   string
		a, b   int
		want   int
		wantOK bool
	}{
		{name: "valid", a: 3, b: 7, want: 21, wantOK: true},
		{name: "zero", a: 0, b: 7},
		{name: "negative", a: -1, b: 7},
		{name: "overflow", a: maxInt, b: 2},
		{name: "boundary", a: maxInt, b: 1, want: maxInt, wantOK: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, ok := product(test.a, test.b)
			if got != test.want || ok != test.wantOK {
				t.Fatalf("product(%d, %d) = %d, %v; want %d, %v", test.a, test.b, got, ok, test.want, test.wantOK)
			}
		})
	}
}
