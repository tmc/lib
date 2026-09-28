package cudasanitizer

import (
	"regexp"
	"strconv"
	"strings"
)

// Access describes an invalid global-memory access. Address is the accessed
// address; Allocation and AllocationBytes, when present, are the tool's nearest
// allocation, which is evidence rather than proof of ownership. Unknown report
// formats remain available in Finding's original text.
type Access struct {
	Address         uint64
	Bytes           uint64
	Write           bool
	Allocation      uint64
	AllocationBytes uint64
}

var accessHeadline = regexp.MustCompile(`^Invalid __global__ (read|write) of size ([0-9]+) bytes?`)
var accessAddress = regexp.MustCompile(`(?m)^\s*(?:Address|Access to) (0x[0-9a-fA-F]+)\b`)
var nearestAllocation = regexp.MustCompile(`nearest allocation at (0x[0-9a-fA-F]+) of size ([0-9,]+) bytes`)

// MemoryAccess decodes supported memcheck global-access reports. It returns
// false if the address or access size cannot be established without guessing.
func (f Finding) MemoryAccess() (Access, bool) {
	h := accessHeadline.FindStringSubmatch(f.Headline)
	if h == nil {
		return Access{}, false
	}
	text := strings.Join(f.Detail, "\n")
	a := accessAddress.FindStringSubmatch(text)
	if a == nil {
		return Access{}, false
	}
	address, err := strconv.ParseUint(a[1][2:], 16, 64)
	if err != nil {
		return Access{}, false
	}
	size, err := strconv.ParseUint(h[2], 10, 64)
	if err != nil || size == 0 {
		return Access{}, false
	}
	out := Access{Address: address, Bytes: size, Write: h[1] == "write"}
	if n := nearestAllocation.FindStringSubmatch(text); n != nil {
		base, e1 := strconv.ParseUint(n[1][2:], 16, 64)
		count, e2 := strconv.ParseUint(strings.ReplaceAll(n[2], ",", ""), 10, 64)
		if e1 == nil && e2 == nil {
			out.Allocation = base
			out.AllocationBytes = count
		}
	}
	return out, true
}
