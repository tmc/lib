package custatevec

import (
	"testing"
	"unsafe"
)

// These constants and field positions are independently read from the pinned
// custatevec.h and CUDA vector_types.h, not derived from generated source.
func TestDistributedABI(t *testing.T) {
	for _, tt := range []struct {
		name      string
		got, want uintptr
	}{
		{"Int2 size", unsafe.Sizeof(Int2{}), 8},
		{"Int2 alignment", unsafe.Alignof(Int2{}), 8},
		{"Int2 Y", unsafe.Offsetof(Int2{}.Y), 4},
		{"parameters size", unsafe.Sizeof(SVSwapParameters{}), 608},
		{"transfer size offset", unsafe.Offsetof(SVSwapParameters{}.TransferSize), 600},
		{"index size", unsafe.Sizeof(Index(0)), 8},
		{"communicator enum size", unsafe.Sizeof(CommunicatorType(0)), 4},
		{"transfer enum size", unsafe.Sizeof(DataTransferType(0)), 4},
		{"external", uintptr(CommunicatorTypeExternal), 0},
		{"OpenMPI", uintptr(CommunicatorTypeOpenmpi), 1},
		{"MPICH", uintptr(CommunicatorTypeMpich), 2},
		{"send and receive", uintptr(DataTransferTypeSendRecv), 3},
	} {
		if tt.got != tt.want {
			t.Errorf("%s = %d, want %d", tt.name, tt.got, tt.want)
		}
	}
}
