package nvtx_test

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/nvtx"
)

func TestNoOpPath(t *testing.T) {
	// With NVTX_INJECTION64_PATH unset, all calls must be safe no-ops
	nvtx.Mark("TestMarker")
	depth := nvtx.RangePush("TestRange")
	if depth != 0 {
		t.Errorf("expected 0 depth in no-op mode, got %d", depth)
	}

	remaining := nvtx.RangePop()
	if remaining != 0 {
		t.Errorf("expected 0 remaining depth in no-op mode, got %d", remaining)
	}

	rID := nvtx.RangeStart("AsyncRange")
	nvtx.RangeEnd(rID)

	done := nvtx.ScopedRange("Scoped")
	done()

	nvtx.NameOSThread(1234, "WorkerGoroutine")
}

func TestDomainNoOp(t *testing.T) {
	dom := nvtx.CreateDomain("InferenceDomain")
	defer dom.Destroy()

	dom.Mark("Layer0Start")
	dom.RangePush("AttentionBlock")
	dom.RangePop()

	rID := dom.RangeStart("AsyncKernel")
	dom.RangeEnd(rID)

	handle := dom.RegisterString("TokenGeneration")
	if handle != 0 {
		t.Errorf("expected 0 handle in no-op mode, got %v", handle)
	}

	nvtx.NameCategory(dom, 1, "Compute")
}
