package cudasanitizer

import (
	"fmt"
	"testing"
)

func TestMemoryAccess(t *testing.T) {
	for _, tt := range []struct {
		name, headline, detail string
		ok                     bool
	}{
		{"access", "Invalid __global__ write of size 4 bytes", "Access to 0x1004 is out of bounds", true},
		{"write", "Invalid __global__ write of size 4 bytes", "Address 0x1004 is out of bounds", true},
		{"unknown", "Invalid read", "Address 0x1004 is out of bounds", false},
		{"noaddress", "Invalid __global__ read of size 4 bytes", "at kernel+0x1004", false},
		{"overflow", "Invalid __global__ read of size 4 bytes", "Address 0x10000000000000000 is out of bounds", false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			_, ok := (Finding{Headline: tt.headline, Detail: []string{tt.detail}}).MemoryAccess()
			if ok != tt.ok {
				t.Fatalf("ok=%v", ok)
			}
		})
	}
}
func ExampleFinding_MemoryAccess() {
	finding := Finding{Headline: "Invalid __global__ write of size 4 bytes", Detail: []string{"Address 0x1010 is out of bounds"}}
	access, ok := finding.MemoryAccess()
	fmt.Printf("parsed=%t address=%#x bytes=%d write=%t\n", ok, access.Address, access.Bytes, access.Write)
	// Output: parsed=true address=0x1010 bytes=4 write=true
}
