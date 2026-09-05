//go:build linux

package nvshmem_test

import (
	"fmt"
	"log"

	"github.com/tmc/lib/nvidia/nvshmem"
)

// Every PE in a job calls the same sequence: agree on a unique ID, initialize
// with it, allocate symmetrically, and finalize. Distributing the ID from PE 0
// to the others is the caller's job; NVSHMEM does not do it. Here it stands in
// as the distribute function.
//
// This example does not run: it needs a multi-PE job with the library present.
func Example_uniqueIDBootstrap() {
	var distribute func(nvshmem.UniqueID) nvshmem.UniqueID

	// Every PE learns its rank and the job size from the launcher.
	pe, npes := 0, 2

	var id nvshmem.UniqueID
	if pe == 0 {
		var err error
		if id, err = nvshmem.NewUniqueID(); err != nil {
			log.Fatal(err)
		}
	}
	id = distribute(id)

	if err := nvshmem.InitWithUniqueID(pe, npes, &id); err != nil {
		log.Fatal(err)
	}
	defer nvshmem.Finalize()

	// The allocation is collective: every PE makes it, at the same size, in
	// the same order, and the address is valid on all of them.
	buf := nvshmem.Malloc(4096)
	defer nvshmem.Free(buf)

	nvshmem.BarrierAll()
	if nvshmem.MyPE() == 0 {
		nvshmem.PutMem(buf, buf, 4096, 1)
		nvshmem.Quiet()
	}
	nvshmem.BarrierAll()
}

// Symbols lists the C entry points the package binds. It is the package's own
// account of its surface and needs neither the library nor a GPU.
func ExampleSymbols() {
	syms := nvshmem.Symbols()
	fmt.Println(len(syms) > 0, syms[0])
	// Output: true nvshmemx_hostlib_init_attr
}
