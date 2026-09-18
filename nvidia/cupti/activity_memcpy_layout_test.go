package cupti

import (
	"testing"
	"unsafe"
)

// The sizes and the two new offsets are measured, not derived: compiled on an
// sbsa-linux host against each toolkit's own header.
//
//	Memcpy6 96  Memcpy7 112  sourceGraphId 96  sourceGraphNodeId 104
//	Memset4 88  Memset5 104  sourceGraphId 88  sourceGraphNodeId 96
func TestActivityMemcpy7Layout(t *testing.T) {
	if got := unsafe.Sizeof(ActivityMemcpy7{}); got != ActivityMemcpy7Size {
		t.Errorf("sizeof(ActivityMemcpy7) = %d, want %d", got, ActivityMemcpy7Size)
	}
	if got := unsafe.Sizeof(ActivityMemcpy6{}); got != ActivityMemcpy6Size {
		t.Errorf("sizeof(ActivityMemcpy6) = %d, want %d", got, ActivityMemcpy6Size)
	}
	if got := unsafe.Alignof(ActivityMemcpy7{}); got != 8 {
		t.Errorf("alignof(ActivityMemcpy7) = %d, want 8", got)
	}
	if got := unsafe.Offsetof(ActivityMemcpy7{}.SourceGraphId); got != 96 {
		t.Errorf("offsetof(ActivityMemcpy7.SourceGraphId) = %d, want 96", got)
	}
	if got := unsafe.Offsetof(ActivityMemcpy7{}.SourceGraphNodeId); got != 104 {
		t.Errorf("offsetof(ActivityMemcpy7.SourceGraphNodeId) = %d, want 104", got)
	}
}

func TestActivityMemset5Layout(t *testing.T) {
	if got := unsafe.Sizeof(ActivityMemset5{}); got != ActivityMemset5Size {
		t.Errorf("sizeof(ActivityMemset5) = %d, want %d", got, ActivityMemset5Size)
	}
	if got := unsafe.Sizeof(ActivityMemset4{}); got != ActivityMemset4Size {
		t.Errorf("sizeof(ActivityMemset4) = %d, want %d", got, ActivityMemset4Size)
	}
	if got := unsafe.Alignof(ActivityMemset5{}); got != 8 {
		t.Errorf("alignof(ActivityMemset5) = %d, want 8", got)
	}
	if got := unsafe.Offsetof(ActivityMemset5{}.SourceGraphId); got != 88 {
		t.Errorf("offsetof(ActivityMemset5.SourceGraphId) = %d, want 88", got)
	}
	if got := unsafe.Offsetof(ActivityMemset5{}.SourceGraphNodeId); got != 96 {
		t.Errorf("offsetof(ActivityMemset5.SourceGraphNodeId) = %d, want 96", got)
	}
}

// TestActivityMemcpyPrefixIdentity is the assertion the decoder actually leans
// on. x/cuptiactivity reads an ActivityMemcpy6 out of whatever the loaded CUPTI
// emitted, including a 112-byte ActivityMemcpy7, which is only sound while
// every shared field sits at the same offset in both. A future version that
// inserts a field mid-record rather than appending one would keep the size
// check happy -- the size would still differ and still be admitted by its own
// constant -- while silently shifting every field after the insertion.
func TestActivityMemcpyPrefixIdentity(t *testing.T) {
	var v6 ActivityMemcpy6
	var v7 ActivityMemcpy7
	for _, f := range []struct {
		name     string
		old, new uintptr
	}{
		{"Kind", unsafe.Offsetof(v6.Kind), unsafe.Offsetof(v7.Kind)},
		{"CopyKind", unsafe.Offsetof(v6.CopyKind), unsafe.Offsetof(v7.CopyKind)},
		{"SrcKind", unsafe.Offsetof(v6.SrcKind), unsafe.Offsetof(v7.SrcKind)},
		{"DstKind", unsafe.Offsetof(v6.DstKind), unsafe.Offsetof(v7.DstKind)},
		{"Flags", unsafe.Offsetof(v6.Flags), unsafe.Offsetof(v7.Flags)},
		{"Bytes", unsafe.Offsetof(v6.Bytes), unsafe.Offsetof(v7.Bytes)},
		{"Start", unsafe.Offsetof(v6.Start), unsafe.Offsetof(v7.Start)},
		{"End", unsafe.Offsetof(v6.End), unsafe.Offsetof(v7.End)},
		{"DeviceId", unsafe.Offsetof(v6.DeviceId), unsafe.Offsetof(v7.DeviceId)},
		{"ContextId", unsafe.Offsetof(v6.ContextId), unsafe.Offsetof(v7.ContextId)},
		{"StreamId", unsafe.Offsetof(v6.StreamId), unsafe.Offsetof(v7.StreamId)},
		{"CorrelationId", unsafe.Offsetof(v6.CorrelationId), unsafe.Offsetof(v7.CorrelationId)},
		{"RuntimeCorrelationId", unsafe.Offsetof(v6.RuntimeCorrelationId), unsafe.Offsetof(v7.RuntimeCorrelationId)},
		{"Reserved0", unsafe.Offsetof(v6.Reserved0), unsafe.Offsetof(v7.Reserved0)},
		{"GraphNodeId", unsafe.Offsetof(v6.GraphNodeId), unsafe.Offsetof(v7.GraphNodeId)},
		{"GraphId", unsafe.Offsetof(v6.GraphId), unsafe.Offsetof(v7.GraphId)},
		{"ChannelID", unsafe.Offsetof(v6.ChannelID), unsafe.Offsetof(v7.ChannelID)},
		{"ChannelType", unsafe.Offsetof(v6.ChannelType), unsafe.Offsetof(v7.ChannelType)},
		{"IsDeviceLaunched", unsafe.Offsetof(v6.IsDeviceLaunched), unsafe.Offsetof(v7.IsDeviceLaunched)},
		{"CopyCount", unsafe.Offsetof(v6.CopyCount), unsafe.Offsetof(v7.CopyCount)},
	} {
		if f.old != f.new {
			t.Errorf("%s moved: ActivityMemcpy6 offset %d, ActivityMemcpy7 offset %d", f.name, f.old, f.new)
		}
	}
	if unsafe.Sizeof(v7) <= unsafe.Sizeof(v6) {
		t.Errorf("ActivityMemcpy7 (%d) must be larger than ActivityMemcpy6 (%d)", unsafe.Sizeof(v7), unsafe.Sizeof(v6))
	}
}

func TestActivityMemsetPrefixIdentity(t *testing.T) {
	var v4 ActivityMemset4
	var v5 ActivityMemset5
	for _, f := range []struct {
		name     string
		old, new uintptr
	}{
		{"Kind", unsafe.Offsetof(v4.Kind), unsafe.Offsetof(v5.Kind)},
		{"Value", unsafe.Offsetof(v4.Value), unsafe.Offsetof(v5.Value)},
		{"Bytes", unsafe.Offsetof(v4.Bytes), unsafe.Offsetof(v5.Bytes)},
		{"Start", unsafe.Offsetof(v4.Start), unsafe.Offsetof(v5.Start)},
		{"End", unsafe.Offsetof(v4.End), unsafe.Offsetof(v5.End)},
		{"DeviceId", unsafe.Offsetof(v4.DeviceId), unsafe.Offsetof(v5.DeviceId)},
		{"ContextId", unsafe.Offsetof(v4.ContextId), unsafe.Offsetof(v5.ContextId)},
		{"StreamId", unsafe.Offsetof(v4.StreamId), unsafe.Offsetof(v5.StreamId)},
		{"CorrelationId", unsafe.Offsetof(v4.CorrelationId), unsafe.Offsetof(v5.CorrelationId)},
		{"Flags", unsafe.Offsetof(v4.Flags), unsafe.Offsetof(v5.Flags)},
		{"MemoryKind", unsafe.Offsetof(v4.MemoryKind), unsafe.Offsetof(v5.MemoryKind)},
		{"Reserved0", unsafe.Offsetof(v4.Reserved0), unsafe.Offsetof(v5.Reserved0)},
		{"GraphNodeId", unsafe.Offsetof(v4.GraphNodeId), unsafe.Offsetof(v5.GraphNodeId)},
		{"GraphId", unsafe.Offsetof(v4.GraphId), unsafe.Offsetof(v5.GraphId)},
		{"ChannelID", unsafe.Offsetof(v4.ChannelID), unsafe.Offsetof(v5.ChannelID)},
		{"ChannelType", unsafe.Offsetof(v4.ChannelType), unsafe.Offsetof(v5.ChannelType)},
		{"IsDeviceLaunched", unsafe.Offsetof(v4.IsDeviceLaunched), unsafe.Offsetof(v5.IsDeviceLaunched)},
	} {
		if f.old != f.new {
			t.Errorf("%s moved: ActivityMemset4 offset %d, ActivityMemset5 offset %d", f.name, f.old, f.new)
		}
	}
}
