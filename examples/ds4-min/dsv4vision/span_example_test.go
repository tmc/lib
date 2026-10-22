package dsv4vision

import "fmt"

func ExampleSpan_zero() {
	var span Span
	start, end := span.Bounds()
	fmt.Println(start, end, len(span.Tokens()), len(span.Embeddings()))
	// Output: 0 0 0 0
}

func ExampleNewSpan_nilSidecar() {
	_, err := NewSpan(nil, 1, 1, 0, 129280, make([]float32, 4096), [32]byte{})
	fmt.Println(err != nil)
	// Output: true
}

func ExampleSpan_Bounds_zero() {
	start, end := (Span{}).Bounds()
	fmt.Println(start, end)
	// Output: 0 0
}

func ExampleSpan_Tokens_zero() {
	fmt.Println(len((Span{}).Tokens()))
	// Output: 0
}

func ExampleSpan_Embeddings_zero() {
	fmt.Println(len((Span{}).Embeddings()))
	// Output: 0
}

func ExampleSpan_Fingerprint_zero() {
	fmt.Println((Span{}).Fingerprint() == [32]byte{})
	// Output: true
}
