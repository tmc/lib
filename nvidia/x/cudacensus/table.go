package cudacensus

import (
	"bufio"
	"fmt"
	"io"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaroofline"
)

// Table is a Coster keyed by exact kernel symbol. It is the simplest useful
// cost model: one cost per launch of a named kernel, which holds when a kernel
// reads the same operands every time it runs. A kernel whose bytes depend on
// its grid needs a CosterFunc instead.
type Table map[string]Cost

// Cost looks up l by name.
func (t Table) Cost(l Launch) (Cost, bool) {
	c, ok := t[l.Name]
	return c, ok
}

// residencyNames are the spellings ParseTable accepts. "unknown" is spelled
// out so that declaring ignorance is a deliberate act a reader can see, rather
// than an omission that reads like device-resident.
var residencyNames = map[string]cudaroofline.Residency{
	string(cudaroofline.DeviceResident):        cudaroofline.DeviceResident,
	string(cudaroofline.HostPageTableResident): cudaroofline.HostPageTableResident,
	"unknown": UnknownResidency,
}

// ParseTable reads a cost table. Each line is
//
//	symbol residency bytes [flops]
//
// where symbol is the kernel symbol exactly as the capture records it,
// residency is "device", "host_page_table" or "unknown", bytes is the bytes
// one launch moves, and flops is optional. Blank lines and lines beginning
// with # are ignored.
//
// Residency is required rather than optional, and it sits before the number it
// qualifies, because a byte count is only meaningful against the ceiling for
// the path those bytes travel; the two paths differ by 1.43x on a measured
// GB10. A table that omits it does not parse.
//
// Writing such a table is the work a capture cannot do for you: the numbers
// come from the shapes and element formats of the tensors the kernel reads and
// from how those tensors were allocated, not from anything the profiler saw.
// Say in a comment where each number came from, so a later reader can check it
// against the model rather than trust it.
func ParseTable(r io.Reader) (Table, error) {
	t := make(Table)
	sc := bufio.NewScanner(r)
	for n := 1; sc.Scan(); n++ {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		f := strings.Fields(line)
		if len(f) < 3 || len(f) > 4 {
			return nil, fmt.Errorf("cudacensus: line %d: want \"symbol residency bytes [flops]\", got %d fields", n, len(f))
		}
		if _, dup := t[f[0]]; dup {
			return nil, fmt.Errorf("cudacensus: line %d: duplicate symbol %q", n, f[0])
		}
		residency, ok := residencyNames[f[1]]
		if !ok {
			return nil, fmt.Errorf("cudacensus: line %d: residency %q is not device, host_page_table or unknown", n, f[1])
		}
		bytes, err := strconv.ParseUint(f[2], 10, 64)
		if err != nil {
			return nil, fmt.Errorf("cudacensus: line %d: bytes: %w", n, err)
		}
		var flops uint64
		if len(f) == 4 {
			if flops, err = strconv.ParseUint(f[3], 10, 64); err != nil {
				return nil, fmt.Errorf("cudacensus: line %d: flops: %w", n, err)
			}
		}
		t[f[0]] = Cost{Bytes: bytes, FLOPs: flops, Residency: residency}
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return t, nil
}
