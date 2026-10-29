package cudadecomp_test

import (
	"context"
	"fmt"
	"slices"

	"github.com/tmc/lib/nvidia/x/cudadecomp"
)

func ExampleCompressBF16() {
	// 1.0, -0.5 and 0.25 in BF16, repeated.
	w := make([]uint16, 3*40000)
	for i := range w {
		w[i] = []uint16{0x3f80, 0xbf00, 0x3e80}[i%3]
	}
	ct, err := cudadecomp.CompressBF16("w", w, cudadecomp.CodecLZ4)
	if err != nil {
		panic(err)
	}
	got, err := cudadecomp.DecompressBF16(context.Background(), ct)
	if err != nil {
		panic(err)
	}
	fmt.Println(len(ct.Chunks), "chunks, equal:", slices.Equal(got, w))
	// Output: 4 chunks, equal: true
}
