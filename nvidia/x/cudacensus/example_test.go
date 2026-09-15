package cudacensus_test

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudacensus"
	"github.com/tmc/lib/nvidia/x/cudaroofline"
)

// The common case: census one token's launches and report what the cost model
// accounts for before reading anything off it.
func Example() {
	c, err := cudacensus.ReadDir("testdata/capture")
	if err != nil {
		panic(err)
	}
	tok, _ := c.Span("token")
	// A nil Coster is the honest first look: the capture alone knows what ran
	// and for how long, and nothing about bytes.
	entries := cudacensus.Census(c.Window(tok.StartNs, tok.EndNs), nil)

	modeled, total := cudacensus.Coverage(entries)
	fmt.Printf("%d kernels, %d launches, %.3f ms device time\n",
		len(entries), countLaunches(entries), float64(total)/1e6)
	fmt.Printf("bytes known for %.0f%% of that time\n", 100*float64(modeled)/float64(total))
	// Output:
	// 10 kernels, 132 launches, 2.284 ms device time
	// bytes known for 0% of that time
}

func countLaunches(entries []cudacensus.Entry) int {
	n := 0
	for _, e := range entries {
		n += e.Launches
	}
	return n
}

// A Table costs a kernel by symbol. Use it when a kernel reads the same
// operands on every launch.
func ExampleParseTable() {
	table, err := cudacensus.ParseTable(strings.NewReader(`
# bytes derived from attn_q_b [1024,32768] Q8_0: (1024/32)*34*32768
# residency: the GGUF is mmap'd, so the GPU reads these pages through host
# page tables and the 162.4 GB/s ceiling applies, not the 239.6 GB/s one.
attn_q_b_matvec host_page_table 35651584
`))
	if err != nil {
		panic(err)
	}
	cost, ok := table.Cost(cudacensus.Launch{Name: "attn_q_b_matvec"})
	fmt.Println(cost.Bytes, cost.Residency, ok)
	_, ok = table.Cost(cudacensus.Launch{Name: "something_else"})
	fmt.Println(ok)

	// Residency is a required column, so a table written before it existed
	// fails loudly at the residency rather than being read as device-resident.
	_, err = cudacensus.ParseTable(strings.NewReader("attn_q_b_matvec 35651584 2048\n"))
	fmt.Println(err)
	// Output:
	// 35651584 host_page_table true
	// false
	// cudacensus: line 1: residency "35651584" is not device, host_page_table or unknown
}

// When a kernel's bytes depend on its launch geometry, a Table cannot describe
// it and a CosterFunc reads the geometry the capture does record.
func ExampleCosterFunc() {
	rowsPerBlock := 32
	coster := cudacensus.CosterFunc(func(l cudacensus.Launch) (cudacensus.Cost, bool) {
		var gx, gy, gz int
		if _, err := fmt.Sscanf(l.Grid, "%dx%dx%d", &gx, &gy, &gz); err != nil {
			return cudacensus.Cost{}, false
		}
		// Q8_0: 34 bytes per 32-element block, one block row per thread block.
		// These weights are mmap'd from the model file, so they are read
		// through host page tables.
		return cudacensus.Cost{
			Bytes:     uint64(gx * gy * gz * rowsPerBlock * 34),
			Residency: cudaroofline.HostPageTableResident,
		}, true
	})
	cost, ok := coster.Cost(cudacensus.Launch{Grid: "1x128x1"})
	fmt.Println(cost.Bytes, cost.Residency, ok)
	// Output:
	// 139264 host_page_table true
}
