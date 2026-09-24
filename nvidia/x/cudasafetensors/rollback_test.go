package cudasafetensors

import (
	"errors"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/safetensors"
)

type failingAllocation struct {
	copied  *bool
	freed   *bool
	freeErr error
}

func (a failingAllocation) CopyFromHost([]float32) error {
	if a.copied != nil {
		*a.copied = true
	}
	return errors.New("injected copy failure")
}

func (a failingAllocation) Free() error {
	*a.freed = true
	return a.freeErr
}

type fixedTensor safetensors.Tensor

func (t fixedTensor) Tensor(string) (safetensors.Tensor, error) {
	return safetensors.Tensor(t), nil
}

var weight = fixedTensor{Name: "weight", Dtype: safetensors.F32, Shape: []int{1}, Data: make([]byte, 4)}

func TestLoadSliceReportsRollbackFailure(t *testing.T) {
	freed := false
	owner, err := loadSlice[float32](weight, "weight", func(int) (failingAllocation, error) {
		return failingAllocation{freed: &freed, freeErr: errors.New("injected free failure")}, nil
	})
	if err == nil || !strings.Contains(err.Error(), "injected copy failure") || !strings.Contains(err.Error(), "injected free failure") {
		t.Fatalf("loadSlice error = %v, want copy and free failures", err)
	}
	if !freed {
		t.Fatal("loadSlice did not attempt to free allocation")
	}
	if owner.freed != &freed {
		t.Fatal("failed rollback lost buffer owner")
	}
	freed = false
	owner.freeErr = nil
	if err := owner.Free(); err != nil || !freed {
		t.Fatalf("retry cleanup: %v", err)
	}
}

func TestLoadSliceCopyFailureFreesAllocation(t *testing.T) {
	freed := false
	owner, err := loadSlice[float32](weight, "weight", func(length int) (failingAllocation, error) {
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
	if owner.freed != nil {
		t.Fatal("returned already freed buffer")
	}
}

func TestLoadSlicePreservesAllocationErrorOwner(t *testing.T) {
	failure := errors.New("restore context")
	freed, copied := false, false
	owner, err := loadSlice[float32](weight, "weight", func(int) (failingAllocation, error) {
		return failingAllocation{freed: &freed, copied: &copied}, failure
	})
	if !errors.Is(err, failure) || owner.freed != &freed {
		t.Fatalf("allocation owner lost: %v", err)
	}
	if copied || freed {
		t.Fatal("used or discarded partial allocation")
	}
	if err := owner.Free(); err != nil || !freed {
		t.Fatalf("caller cleanup: %v", err)
	}
}
