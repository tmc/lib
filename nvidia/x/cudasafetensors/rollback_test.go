package cudasafetensors

import (
	"errors"
	"strings"
	"testing"
)

type failingAllocation struct {
	freed   *bool
	freeErr error
}

type fixedTensor struct {
	data []byte
	info TensorInfo
}

func (p fixedTensor) TensorBytes(string) ([]byte, TensorInfo, error) {
	return p.data, p.info, nil
}

func (a failingAllocation) CopyFromHost([]float32) error {
	return errors.New("injected copy failure")
}

func (a failingAllocation) Free() error {
	*a.freed = true
	return a.freeErr
}

func TestLoadSliceReportsRollbackFailure(t *testing.T) {
	provider := fixedTensor{
		data: make([]byte, 4),
		info: TensorInfo{Name: "weight", DType: "F32", Shape: []int{1}, OffsetEnd: 4},
	}
	freed := false
	_, err := loadSlice[float32](provider, "weight", func(int) (failingAllocation, error) {
		return failingAllocation{freed: &freed, freeErr: errors.New("injected free failure")}, nil
	})
	if err == nil || !strings.Contains(err.Error(), "injected copy failure") || !strings.Contains(err.Error(), "injected free failure") {
		t.Fatalf("loadSlice error = %v, want copy and free failures", err)
	}
	if !freed {
		t.Fatal("loadSlice did not attempt to free allocation")
	}
}

func TestLoadSliceCopyFailureFreesAllocation(t *testing.T) {
	provider := fixedTensor{
		data: make([]byte, 4),
		info: TensorInfo{Name: "weight", DType: "F32", Shape: []int{1}, OffsetEnd: 4},
	}
	freed := false
	_, err := loadSlice[float32](provider, "weight", func(length int) (failingAllocation, error) {
		if length != 1 {
			t.Fatalf("allocation length = %d, want 1", length)
		}
		return failingAllocation{freed: &freed}, nil
	})
	if err == nil || !strings.Contains(err.Error(), "injected copy failure") {
		t.Fatalf("loadSlice error = %v, want injected copy failure", err)
	}
	if !freed {
		t.Fatal("loadSlice did not free allocation after copy failure")
	}
}
