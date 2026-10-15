package cudashared

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudalayout"
	"strings"
	"testing"
)

func testLimits() Limits {
	return Limits{StaticMax: 128, DynamicMax: 256, TotalMax: 512, Source: "test-target"}
}

func TestSharedAlignment(t *testing.T) {
	p, _ := NewPlan(testLimits())
	r, err := p.Add(Request{Name: "a", Size: 17, Alignment: 16, FirstStage: 0, LastStage: 0})
	if err != nil || r.Offset != 0 {
		t.Fatalf("allocation = %#v, %v", r, err)
	}
	r, err = p.Add(Request{Name: "b", Size: 16, Alignment: 32, FirstStage: 0, LastStage: 0})
	if err != nil || r.Offset%32 != 0 {
		t.Fatalf("alignment = %#v, %v", r, err)
	}
	if err := p.ValidateBase(8); err == nil {
		t.Fatal("accepted unaligned base")
	}
}

func TestSharedLifetimeOverlap(t *testing.T) {
	p, _ := NewPlan(testLimits())
	a, err := p.Add(Request{Name: "a", Size: 64, Alignment: 16, FirstStage: 1, LastStage: 3})
	if err != nil {
		t.Fatal(err)
	}
	b, err := p.Add(Request{Name: "live", Size: 64, Alignment: 16, FirstStage: 3, LastStage: 4})
	if err != nil {
		t.Fatal(err)
	}
	if a.Offset == b.Offset {
		t.Fatal("overlapping live regions reused storage")
	}
}

func TestStageReuse(t *testing.T) {
	p, _ := NewPlan(testLimits())
	a, _ := p.Add(Request{Name: "a", Size: 200, Alignment: 16, FirstStage: 0, LastStage: 1, Dynamic: true})
	b, err := p.Add(Request{Name: "b", Size: 200, Alignment: 16, FirstStage: 2, LastStage: 3, Dynamic: true})
	if err != nil || a.Offset != b.Offset {
		t.Fatalf("stage reuse = %#v %#v, %v", a, b, err)
	}
	if p.DynamicBytes() != 200 || p.FootprintBytes() != 200 {
		t.Fatalf("peak accounting = %d/%d", p.DynamicBytes(), p.FootprintBytes())
	}
}

func TestSharedResourceLimit(t *testing.T) {
	p, _ := NewPlan(testLimits())
	if _, err := p.Add(Request{Name: "too-large", Size: 129, Alignment: 1, FirstStage: 0, LastStage: 0}); err == nil {
		t.Fatal("accepted static limit overflow")
	}
	if _, err := p.Add(Request{Name: "bad-align", Size: 1, Alignment: 3, FirstStage: 0, LastStage: 0}); err == nil {
		t.Fatal("accepted non-power-of-two alignment")
	}
}

func TestTailLayout(t *testing.T) {
	p, _ := NewPlan(testLimits())
	if _, err := p.Add(Request{Name: "tail", Size: 31, Alignment: 16, FirstStage: 0, LastStage: 0}); err != nil {
		t.Fatal(err)
	}
	if p.FootprintBytes() != 31 {
		t.Fatalf("footprint = %d", p.FootprintBytes())
	}
}

func TestCUDALayoutConsumer(t *testing.T) {
	tile, err := cudalayout.NewTile(3, 5, 4, cudalayout.WithAlignment(16))
	if err != nil {
		t.Fatal(err)
	}
	r, err := RequestForTile(tile, "tile", 0, 1, true)
	if err != nil || r.Size != 3*5*4 || r.Alignment != 16 {
		t.Fatalf("tile request = %#v, %v", r, err)
	}
}

func TestHugeStageAndPhysicalPools(t *testing.T) {
	p, _ := NewPlan(testLimits())
	a, err := p.Add(Request{Name: "static", Size: 32, Alignment: 16, FirstStage: 0, LastStage: 0})
	if err != nil {
		t.Fatal(err)
	}
	b, err := p.Add(Request{Name: "dynamic", Size: 32, Alignment: 16, FirstStage: 0, LastStage: int(^uint(0) >> 1), Dynamic: true})
	if err != nil {
		t.Fatal(err)
	}
	if a.Offset != 0 || b.Offset != 0 || p.FootprintBytes() != 64 {
		t.Fatal("separate pools must start at their own bases and both count toward total")
	}
}

func TestAlignedFootprintLimit(t *testing.T) {
	p, _ := NewPlan(Limits{StaticMax: 512, DynamicMax: 64, TotalMax: 512, Source: "test"})
	if _, err := p.Add(Request{Name: "first", Size: 1, Alignment: 128, FirstStage: 0, LastStage: 0, Dynamic: true}); err != nil {
		t.Fatal(err)
	}
	if _, err := p.Add(Request{Name: "dynamic", Size: 64, Alignment: 16, FirstStage: 0, LastStage: 0, Dynamic: true}); err == nil {
		t.Fatal("accepted dynamic aligned footprint beyond dynamic pool")
	}
}

func TestStageFixtureUsesPlanOffsetAndBytes(t *testing.T) {
	limits, err := LimitsFromDevice("sm_121a")
	if err != nil {
		t.Fatal(err)
	}
	p, err := NewPlan(limits)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := p.Add(Request{Name: "prefix", Size: 17, Alignment: 16, FirstStage: 1, LastStage: 1, Dynamic: true}); err != nil {
		t.Fatal(err)
	}
	region, err := p.Add(Request{Name: "stage0", Size: 64, Alignment: 16, FirstStage: 1, LastStage: 1, Dynamic: true})
	if err != nil {
		t.Fatal(err)
	}
	src, bytes, err := StagedCopyFixture(p, "sm_121a", 8)
	if err != nil {
		t.Fatal(err)
	}
	if region.Offset == 0 || bytes != p.DynamicBytes() || !strings.Contains(src, fmt.Sprintf("shared_offset=%d", region.Offset)) || !strings.Contains(src, fmt.Sprintf("shared_bytes=%d", p.DynamicBytes())) {
		t.Fatalf("fixture does not carry plan allocation: bytes=%d src=%q", bytes, src)
	}
}
