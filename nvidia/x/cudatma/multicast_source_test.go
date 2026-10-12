package cudatma

import (
	"fmt"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
)

func multicastConfig(rank int, elem uint32, swizzle uint32) MulticastConfig {
	d := Descriptor{Target: TargetSM100a, Rank: rank, Base: 0x1000, ElementBytes: elem, Swizzle: swizzle}
	width := uint32(128)
	if swizzle != 0 {
		width = swizzle
	}
	d.Box[0] = width / elem
	d.Dims[0] = uint64(d.Box[0]) * 2
	stride := d.Dims[0] * uint64(elem)
	for i := 1; i < rank; i++ {
		d.Dims[i] = 4
		d.Box[i] = 2
		d.Strides[i-1] = stride
		stride *= 4
	}
	bytes := uint64(width)
	for i := 1; i < rank; i++ {
		bytes *= 2
	}
	c := MulticastConfig{Descriptor: d, Cluster: [3]uint32{2, 1, 1}, Destinations: []uint32{0, 1}, AvailableSharedBytes: 65536, SharedLimitProvenance: "host test selected bound"}
	for i := uint32(0); i < 2; i++ {
		c.Windows = append(c.Windows, SharedWindow{Rank: i, Descriptor: d, BarrierOffset: bytes, ReservedBytes: bytes + 8})
	}
	return c
}

func TestMulticastPlanMatrix(t *testing.T) {
	for rank := 1; rank <= 5; rank++ {
		for _, typ := range []ElementType{ElementTypeDefault, ElementTypeFloat16, ElementTypeBfloat16, ElementTypeFloat32} {
			for _, sw := range []uint32{0, 32, 64, 128} {
				elem := uint32(2)
				if typ == ElementTypeFloat32 {
					elem = 4
				}
				c := multicastConfig(rank, elem, sw)
				c.Descriptor.ElementType = typ
				for i := range c.Windows {
					c.Windows[i].Descriptor = c.Descriptor
				}
				c.Coordinates[0] = -1
				p, err := NewMulticastPlan(c)
				if err != nil {
					t.Fatalf("rank %d type %v swizzle %d: %v", rank, typ, sw, err)
				}
				r, _ := p.Requirements()
				if r.Mask != 3 || r.CTAs != 2 || r.OutputBytes != uint64(r.TileBytes)*2 {
					t.Fatalf("incorrect requirements: %+v", r)
				}
				s, err := p.Source()
				if err != nil || protocolError(s, p) != "" {
					t.Fatalf("rank %d type %v swizzle %d source: %v", rank, typ, sw, err)
				}
			}
		}
	}
}

func TestMulticastPlanRejects(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*MulticastConfig)
		want   string
	}{
		{"unsupported target", func(c *MulticastConfig) { c.Descriptor.Target = TargetSM121a }, "target"},
		{"zero dimension", func(c *MulticastConfig) { c.Descriptor.Dims[0] = 0 }, "dimension"},
		{"zero base", func(c *MulticastConfig) { c.Descriptor.Base = 0 }, "base"},
		{"global address overflow", func(c *MulticastConfig) { c.Descriptor.Base = ^uint64(0) - 127 }, "address extent"},
		{"dimension overflow", func(c *MulticastConfig) { c.Descriptor.Dims[0] = 1<<32 + 1 }, "limits"},
		{"box overflow", func(c *MulticastConfig) { c.Descriptor.Box[1] = 257; c.Descriptor.Dims[1] = 512 }, "limits"},
		{"stride overflow", func(c *MulticastConfig) { c.Descriptor.Strides[0] = 1 << 40 }, "stride"},
		{"overlapping global axes", func(c *MulticastConfig) { c.Descriptor.Strides[0] = 16 }, "overlaps"},
		{"inactive coordinate", func(c *MulticastConfig) { c.Coordinates[2] = 1 }, "inactive"},
		{"inactive extent", func(c *MulticastConfig) { c.Descriptor.Box[2] = 1 }, "inactive"},
		{"zero cluster", func(c *MulticastConfig) { c.Cluster[0] = 0 }, "cluster"},
		{"mask width", func(c *MulticastConfig) { c.Cluster = [3]uint32{4, 4, 2} }, "cluster"},
		{"source outside cluster", func(c *MulticastConfig) { c.SourceRank = 2 }, "source"},
		{"empty destinations", func(c *MulticastConfig) { c.Destinations = nil }, "destination"},
		{"destination outside cluster", func(c *MulticastConfig) { c.Destinations = []uint32{2} }, "destination"},
		{"duplicate destinations", func(c *MulticastConfig) { c.Destinations = []uint32{1, 1} }, "duplicate"},
		{"missing window", func(c *MulticastConfig) { c.Windows = c.Windows[:1] }, "every"},
		{"duplicate window", func(c *MulticastConfig) { c.Windows[1].Rank = 0 }, "duplicate"},
		{"different descriptor", func(c *MulticastConfig) { c.Windows[1].Descriptor.Base = 0x2000 }, "descriptor"},
		{"misaligned tile", func(c *MulticastConfig) { c.Windows[1].TileOffset = 16 }, "misaligned"},
		{"misaligned barrier", func(c *MulticastConfig) { c.Windows[1].BarrierOffset++ }, "misaligned"},
		{"overlap", func(c *MulticastConfig) { c.Windows[1].BarrierOffset = 8 }, "overlaps"},
		{"different offset", func(c *MulticastConfig) {
			c.Windows[1].TileOffset = 1024
			c.Windows[1].BarrierOffset = 2048
			c.Windows[1].ReservedBytes = 4096
		}, "different"},
		{"truncated shared address", func(c *MulticastConfig) { c.Windows[1].TileOffset = 1 << 32 }, "address"},
		{"small reservation", func(c *MulticastConfig) { c.Windows[1].ReservedBytes = 8 }, "reservation"},
		{"small selected limit", func(c *MulticastConfig) { c.AvailableSharedBytes = 8 }, "reservation"},
		{"missing limit provenance", func(c *MulticastConfig) { c.SharedLimitProvenance = "" }, "provenance"},
		{"nan fill", func(c *MulticastConfig) { c.Descriptor.OOBFill = driver.TensorMapFloatOobFillNanRequestZeroFma }, "zero-fill"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := multicastConfig(2, 4, 128)
			tt.mutate(&c)
			_, err := NewMulticastPlan(c)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error %v, want %q", err, tt.want)
			}
		})
	}
}

func TestMulticastPhysicalAgreementPending(t *testing.T) {
	for rank := 3; rank <= 5; rank++ {
		for _, iv := range []driver.TensorMapInterleave{driver.TensorMapInterleave16B, driver.TensorMapInterleave32B} {
			c := multicastConfig(rank, 2, 32)
			c.Descriptor.Interleave = iv
			if _, err := NewMulticastPlan(c); err == nil || !strings.Contains(err.Error(), "physical storage") {
				t.Fatalf("rank %d interleave %d: %v", rank, iv, err)
			}
		}
	}
	c := multicastConfig(2, 2, 128)
	c.Descriptor.Box[0] = 32
	if _, err := NewMulticastPlan(c); err == nil || !strings.Contains(err.Error(), "narrow") {
		t.Fatalf("narrow swizzle: %v", err)
	}
}

func TestMulticastPlanOwnsConfiguration(t *testing.T) {
	c := multicastConfig(1, 4, 0)
	p, err := NewMulticastPlan(c)
	if err != nil {
		t.Fatal(err)
	}
	c.Destinations[0] = 15
	c.Windows[0].TileOffset = 16
	c.Descriptor.Base = 0
	r, _ := p.Requirements()
	if r.Mask != 3 || p.config.Windows[0].TileOffset != 0 || p.config.Descriptor.Base == 0 {
		t.Fatal("caller mutation changed plan")
	}
	r.Mask = 0
	again, _ := p.Requirements()
	if again.Mask != 3 {
		t.Fatal("requirements mutation changed plan")
	}
	if _, err := (*MulticastPlan)(nil).Requirements(); err == nil {
		t.Fatal("nil plan admitted")
	}
	if _, err := new(MulticastPlan).Requirements(); err == nil {
		t.Fatal("zero plan admitted")
	}
}

func TestMulticastMaskRank15(t *testing.T) {
	c := multicastConfig(1, 4, 0)
	c.Cluster = [3]uint32{4, 2, 2}
	c.SourceRank = 15
	c.Destinations = []uint32{0, 15}
	w := c.Windows[0]
	c.Windows = nil
	for i := uint32(0); i < 16; i++ {
		w.Rank = i
		c.Windows = append(c.Windows, w)
	}
	p, err := NewMulticastPlan(c)
	if err != nil {
		t.Fatal(err)
	}
	r, _ := p.Requirements()
	if r.Mask != 0x8001 || r.CTAs != 16 {
		t.Fatalf("requirements: %+v", r)
	}
}

// protocolError checks the emitted one-shot sequence, including every
// destination's expectation. These controls observe source, not assembled PTX
// or hardware completion.
func protocolError(s string, p *MulticastPlan) string {
	r := p.requirements
	wants := []string{"__grid_constant__ const CUtensorMap map", "fence.mbarrier_init.release.cluster;", ".tile.mbarrier::complete_tx::bytes.multicast::cluster", `"h"(mask)`, ".reg .pred p;", "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64", "cluster.sync(); // retirement after all selected waits and reads"}
	for _, w := range wants {
		if !strings.Contains(s, w) {
			return w
		}
	}
	if !strings.Contains(s, fmt.Sprintf("const uint16_t mask = %d;", r.Mask)) {
		return "mask"
	}
	for rank := uint32(0); rank < r.CTAs; rank++ {
		if r.Mask&(1<<rank) != 0 && !strings.Contains(s, fmt.Sprintf("if (rank == %d && threadIdx.x == 0) {\n    asm volatile(\"mbarrier.arrive.expect_tx", rank)) {
			return "destination expectation"
		}
	}
	if strings.Index(s, "mbarrier.init") > strings.Index(s, "fence.mbarrier_init") || strings.Index(s, "initialized expectations before issue") > strings.Index(s, "cp.async.bulk.tensor") || strings.Index(s, "mbarrier.try_wait") > strings.Index(s, "output[uint64_t") || strings.Index(s, "output[uint64_t") > strings.Index(s, "retirement after") {
		return "ordering"
	}
	return ""
}

func TestMulticastSourceRankABI(t *testing.T) {
	for rank := 1; rank <= 5; rank++ {
		p, err := NewMulticastPlan(multicastConfig(rank, 4, 128))
		if err != nil {
			t.Fatal(err)
		}
		s, err := p.Source()
		if err != nil {
			t.Fatal(err)
		}
		if got := protocolError(s, p); got != "" {
			t.Fatalf("rank%d: %s", rank, got)
		}
		opcode := fmt.Sprintf("cp.async.bulk.tensor.%dd.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster", rank)
		if strings.Count(s, opcode) != 1 || strings.Count(s, "int32_t c") != rank {
			t.Fatalf("rank%d ABI missing", rank)
		}
		for i := 0; i < rank; i++ {
			if !strings.Contains(s, fmt.Sprintf(`"r"(c%d)`, i)) {
				t.Fatalf("rank%d coordinate%d missing", rank, i)
			}
		}
		if !strings.Contains(s, fmt.Sprintf("}], [%%%d], %%%d;", rank+2, rank+3)) {
			t.Fatalf("rank%d operand indexes", rank)
		}
	}
}

func TestMulticastSourceCausalOmissions(t *testing.T) {
	p, _ := NewMulticastPlan(multicastConfig(2, 4, 128))
	s, _ := p.Source()
	tests := []struct{ name, old, replacement string }{
		{"qualifier", ".multicast::cluster", ""},
		{"destination bit", "const uint16_t mask = 3;", "const uint16_t mask = 1;"},
		{"destination expectation", "if (rank == 1 && threadIdx.x == 0) {\n    asm volatile(\"mbarrier.arrive.expect_tx", "if (false) {\n    asm volatile(\"mbarrier.arrive.expect_tx"},
		{"init fence", "fence.mbarrier_init.release.cluster;", ""},
		{"retirement", "cluster.sync(); // retirement after all selected waits and reads", ""},
		{"descriptor parameter", "__grid_constant__ const CUtensorMap map", "const CUtensorMap map"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mutant := strings.Replace(s, tt.old, tt.replacement, 1)
			if mutant == s {
				t.Fatal("mutation did not apply")
			}
			if protocolError(mutant, p) == "" {
				t.Fatal("mutation escaped host protocol control")
			}
		})
	}
}

func TestMulticastPhysicalFullWidthOracle(t *testing.T) {
	// PTX Table24's explicit 16-byte-cell permutations at a zero base phase.
	tables := map[uint32][][]int{
		32:  {{0, 1, 2, 3, 4, 5, 6, 7}, {1, 0, 3, 2, 5, 4, 7, 6}},
		64:  {{0, 1, 2, 3, 4, 5, 6, 7}, {1, 0, 3, 2, 5, 4, 7, 6}, {2, 3, 0, 1, 6, 7, 4, 5}, {3, 2, 1, 0, 7, 6, 5, 4}},
		128: {{0, 1, 2, 3, 4, 5, 6, 7}, {1, 0, 3, 2, 5, 4, 7, 6}, {2, 3, 0, 1, 6, 7, 4, 5}, {3, 2, 1, 0, 7, 6, 5, 4}, {4, 5, 6, 7, 0, 1, 2, 3}, {5, 4, 7, 6, 1, 0, 3, 2}, {6, 7, 4, 5, 2, 3, 0, 1}, {7, 6, 5, 4, 3, 2, 1, 0}},
	}
	for sw, table := range tables {
		for row := 0; row < 24; row++ {
			for cell := 0; cell < 8; cell++ {
				for byteInCell := 0; byteInCell < 16; byteInCell++ {
					logical := row*128 + cell*16 + byteInCell
					want := row*128 + table[row%len(table)][cell]*16 + byteInCell
					got := logical ^ (((logical >> 7) & int(sw/16-1)) << 4)
					if got != want {
						t.Fatalf("swizzle%d logical%d: %d want%d", sw, logical, got, want)
					}
				}
			}
		}
		for rank := 1; rank <= 5; rank++ {
			c := multicastConfig(rank, 2, sw)
			p, err := NewMulticastPlan(c)
			if err != nil {
				t.Fatal(err)
			}
			bytes := int(p.requirements.TileBytes)
			seen := make([]bool, bytes)
			for i := 0; i < bytes; i++ {
				j := i ^ (((i >> 7) & int(sw/16-1)) << 4)
				if j >= bytes || seen[j] {
					t.Fatalf("rank%d swizzle%d unsafe physical span", rank, sw)
				}
				seen[j] = true
			}
		}
	}
}

func TestMulticastUnswizzledWidths(t *testing.T) {
	for rank := 1; rank <= 5; rank++ {
		for _, elem := range []uint32{2, 4} {
			for width := uint32(16); width <= 128; width += 16 {
				c := multicastConfig(rank, elem, 0)
				c.Descriptor.Box[0] = width / elem
				bytes := uint64(width)
				for i := 1; i < rank; i++ {
					bytes *= 2
				}
				for i := range c.Windows {
					c.Windows[i].Descriptor = c.Descriptor
					c.Windows[i].BarrierOffset = bytes
					c.Windows[i].ReservedBytes = bytes + 8
				}
				p, err := NewMulticastPlan(c)
				if err != nil {
					t.Fatalf("rank%d elem%d width%d: %v", rank, elem, width, err)
				}
				s, err := p.Source()
				if err != nil || !strings.Contains(s, "((i >> 7) & 0u)") {
					t.Fatalf("unswizzled source: %v", err)
				}
			}
		}
	}
}

func TestMulticastTransactionOverflow(t *testing.T) {
	c := multicastConfig(5, 2, 128)
	stride := uint64(128)
	c.Descriptor.Dims[0] = 64
	for i := 1; i < 5; i++ {
		c.Descriptor.Dims[i] = 256
		c.Descriptor.Box[i] = 256
		c.Descriptor.Strides[i-1] = stride
		stride *= 256
	}
	if _, err := NewMulticastPlan(c); err == nil || !strings.Contains(err.Error(), "transaction bytes") {
		t.Fatalf("overflow: %v", err)
	}
}

func TestMulticastSparseSourceParticipation(t *testing.T) {
	c := multicastConfig(2, 4, 128)
	c.Cluster = [3]uint32{4, 1, 1}
	c.SourceRank = 2
	c.Destinations = []uint32{1, 3}
	w := c.Windows[0]
	c.Windows = nil
	for i := uint32(0); i < 4; i++ {
		w.Rank = i
		c.Windows = append(c.Windows, w)
	}
	p, err := NewMulticastPlan(c)
	if err != nil {
		t.Fatal(err)
	}
	s, _ := p.Source()
	if protocolError(s, p) != "" || !strings.Contains(s, "const uint16_t mask = 10;") || strings.Count(s, "mbarrier.arrive.expect_tx") != 2 {
		t.Fatal("incorrect sparse participation")
	}
	if strings.Contains(s, "if (rank == 2 && threadIdx.x == 0) {\n    asm volatile(\"mbarrier.arrive.expect_tx") {
		t.Fatal("nonselected source acquired expectation")
	}
}

func transactionBoundaryConfig(width uint32, axes [4]uint32) MulticastConfig {
	c := multicastConfig(5, 2, 0)
	d := c.Descriptor
	d.Box[0] = width / 2
	d.Dims[0] = uint64(d.Box[0])
	stride, bytes := uint64(width), uint64(width)
	for i, extent := range axes {
		d.Box[i+1] = extent
		d.Dims[i+1] = uint64(extent)
		d.Strides[i] = stride
		stride *= uint64(extent)
		bytes *= uint64(extent)
	}
	c.Descriptor = d
	// Hypothetical host admission metadata, not an actual device capability.
	c.AvailableSharedBytes = bytes + 8
	c.SharedLimitProvenance = "hypothetical host counter boundary; no device resource admission"
	for i := range c.Windows {
		c.Windows[i].Descriptor = d
		c.Windows[i].BarrierOffset = bytes
		c.Windows[i].ReservedBytes = bytes + 8
	}
	return c
}

func TestMulticastTransactionCounterBoundary(t *testing.T) {
	// Every admitted tile size is a multiple of 16, so the exact architectural
	// maximum (2^20-1) is not representable. 2^20-16 has factor 257 in its
	// element-cell product, outside the <=256 axis bound. 2^20-32 is admitted
	// at rank five as 16*2*7*31*151, without claiming actual shared capacity.
	const near = 1<<20 - 32
	c := transactionBoundaryConfig(16, [4]uint32{2, 7, 31, 151})
	p, err := NewMulticastPlan(c)
	if err != nil {
		t.Fatal(err)
	}
	r, _ := p.Requirements()
	if r.TileBytes != near {
		t.Fatalf("bytes%d want%d", r.TileBytes, near)
	}
	s, err := p.Source()
	if err != nil || !strings.Contains(s, fmt.Sprintf(`"r"(%du)`, near)) {
		t.Fatalf("near-boundary source: %v", err)
	}
	tests := []struct {
		name  string
		width uint32
		axes  [4]uint32
	}{
		{"exact two to twenty", 16, [4]uint32{256, 256, 1, 1}},
		{"above two to twenty", 32, [4]uint32{256, 256, 1, 1}},
		{"two to thirty one", 128, [4]uint32{256, 256, 256, 1}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := transactionBoundaryConfig(tt.width, tt.axes)
			_, err := NewMulticastPlan(c)
			if err == nil || !strings.Contains(err.Error(), "mbarrier tx-count limit") {
				t.Fatalf("counter refusal: %v", err)
			}
		})
	}
	// The same counter-admissible tile still needs independent resource
	// admission: a smaller selected shared bound must refuse it.
	c.AvailableSharedBytes = 65536
	if _, err := NewMulticastPlan(c); err == nil || !strings.Contains(err.Error(), "shared limit") {
		t.Fatalf("resource refusal: %v", err)
	}
}

func TestMulticastSourceCounterAdmission(t *testing.T) {
	p, err := NewMulticastPlan(multicastConfig(1, 4, 0))
	if err != nil {
		t.Fatal(err)
	}
	// Defensive source admission uses the counter limit as well as constructor
	// admission. Deliberately corrupt private test metadata, not public config.
	p.requirements.TileBytes = 1 << 20
	s, err := p.Source()
	if s != "" || err == nil || !strings.Contains(err.Error(), "mbarrier tx-count limit") {
		t.Fatalf("source counter refusal: %v", err)
	}
}
