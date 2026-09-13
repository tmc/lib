package cudalowering_test

import (
	"fmt"
	"log"
	"os"

	"github.com/tmc/lib/nvidia/x/cudalowering"
	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

func parse(ptxFile, sassFile string) (*cudaptx.Module, *cudasass.Disassembly) {
	pf, err := os.Open(ptxFile)
	if err != nil {
		log.Fatal(err)
	}
	defer pf.Close()
	m, err := cudaptx.Parse(pf)
	if err != nil {
		log.Fatal(err)
	}
	sf, err := os.Open(sassFile)
	if err != nil {
		log.Fatal(err)
	}
	defer sf.Close()
	d, err := cudasass.Parse(sf)
	if err != nil {
		log.Fatal(err)
	}
	return m, d
}

// Two kernels call atomicAdd on the same address with the same arguments, and
// compile to the same atom.global.add.f32 in PTX. One of them reads the value
// the atomic returns; the other throws it away, and ptxas substitutes a
// reduction. Nothing in the PTX says which kernel got which.
func Example() {
	m, d := parse("testdata/lowering.sm90.ptx", "testdata/lowering.sm90.sass")

	r, err := cudalowering.Compare(m, d, nil)
	if err != nil {
		log.Fatal(err)
	}
	for _, div := range r.Divergences {
		fmt.Println(div)
	}
	// Output:
	// _Z10atomicDropPfPKfi: erased: 1 atom.global, no ATOMG
}

// A report turns into one error, for use as a gate.
func ExampleReport_Err() {
	m, d := parse("testdata/lowering.sm90.ptx", "testdata/lowering.sm90.rr24.sass")

	r, err := cudalowering.Compare(m, d, &cudalowering.Options{
		Lowerings: []cudalowering.Lowering{
			{PTX: "st.local", SASS: []string{"STL"}, Exclusive: true, Spill: true},
		},
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(r.Err())
	// Output:
	// cudalowering: _Z7spillerPfPKfi: spill: 5 STL, no st.local
}
