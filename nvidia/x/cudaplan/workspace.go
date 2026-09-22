package cudaplan

import (
	"errors"
	"fmt"
	"math"
	"sync"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// WorkspaceManager manages temporary scratchpad memory across execution steps and branches.
// It enforces 256-byte alignment (standard for Tensor Cores and cuBLASLt) and verifies
// that memory usage stays within the configured peak budget.
type WorkspaceManager struct {
	mu          sync.Mutex
	dev         *cuda.Device
	buffer      cudamem.Slice[byte]
	totalBytes  int
	peakBudget  int
	ownedBuffer bool
	closed      bool
}

// NewWorkspaceManager creates a WorkspaceManager with a configured peak budget.
// A zero peakBudget selects a 1 GiB budget. A negative budget is rejected.
func NewWorkspaceManager(dev *cuda.Device, peakBudget int) (*WorkspaceManager, error) {
	if dev == nil {
		return nil, errors.New("cudaplan: nil device passed to NewWorkspaceManager")
	}
	if peakBudget < 0 {
		return nil, fmt.Errorf("cudaplan: negative peak workspace budget %d", peakBudget)
	}
	if peakBudget == 0 {
		peakBudget = 1024 * 1024 * 1024 // 1 GiB default budget
	}
	return &WorkspaceManager{
		dev:        dev,
		peakBudget: peakBudget,
	}, nil
}

// PeakBudget returns the configured maximum workspace memory budget.
func (wm *WorkspaceManager) PeakBudget() int {
	if wm == nil {
		return 0
	}
	wm.mu.Lock()
	defer wm.mu.Unlock()
	return wm.peakBudget
}

// EnsureSize ensures that an internal workspace buffer of at least requiredBytes is allocated.
// If requiredBytes > peakBudget, ErrWorkspaceBudgetExceeded is returned.
func (wm *WorkspaceManager) EnsureSize(requiredBytes int) error {
	if wm == nil {
		return errors.New("cudaplan: nil workspace manager")
	}
	wm.mu.Lock()
	defer wm.mu.Unlock()

	if wm.closed {
		return ErrClosed
	}
	if requiredBytes < 0 || requiredBytes > math.MaxInt-255 {
		return errors.New("cudaplan: invalid workspace size")
	}
	if rem := requiredBytes % 256; rem != 0 {
		requiredBytes += 256 - rem
	}

	if requiredBytes > wm.peakBudget {
		return fmt.Errorf("%w: required %d bytes exceeds peak budget %d bytes",
			ErrWorkspaceBudgetExceeded, requiredBytes, wm.peakBudget)
	}

	if wm.buffer.Len() >= requiredBytes {
		return nil // Already large enough
	}

	// Free existing buffer if owned
	if wm.ownedBuffer && wm.buffer.Len() > 0 {
		if err := wm.buffer.Free(); err != nil {
			return fmt.Errorf("cudaplan: release previous workspace: %w", err)
		}
		wm.buffer = cudamem.Slice[byte]{}
		wm.ownedBuffer = false
	}

	buf, err := cudamem.Alloc[byte](wm.dev, requiredBytes)
	// Adopt before inspecting the error: context restoration can fail after
	// allocation succeeds. Close must retain the opportunity to free it.
	wm.buffer = buf
	wm.totalBytes = requiredBytes
	wm.ownedBuffer = buf.Ptr() != 0
	if err != nil {
		wm.closed = true
		return fmt.Errorf("cudaplan: alloc workspace buffer (%d bytes): %w", requiredBytes, err)
	}

	// Verify 256-byte alignment
	if uintptr(buf.Ptr())%256 != 0 {
		wm.closed = true
		return fmt.Errorf("cudaplan: allocated workspace pointer 0x%x not 256-byte aligned", buf.Ptr())
	}

	return nil
}

// SliceAt returns a checked byte range within the workspace. The base is
// aligned to 256 bytes; the caller must choose an appropriately aligned offset.
func (wm *WorkspaceManager) SliceAt(offset, size int) (driver.DevicePtr, error) {
	if wm == nil {
		return 0, errors.New("cudaplan: nil workspace manager")
	}
	wm.mu.Lock()
	defer wm.mu.Unlock()

	if wm.closed {
		return 0, ErrClosed
	}
	if offset < 0 || size < 0 || offset > wm.totalBytes || size > wm.totalBytes-offset {
		return 0, fmt.Errorf("cudaplan: workspace offset [%d, %d) out of bounds (%d total)",
			offset, offset+size, wm.totalBytes)
	}

	basePtr := wm.buffer.Ptr()
	if basePtr == 0 {
		return 0, ErrNilBuffer
	}

	subPtr := basePtr + driver.DevicePtr(offset)
	return subPtr, nil
}

// Buffer returns the underlying allocated workspace buffer.
func (wm *WorkspaceManager) Buffer() cudamem.Slice[byte] {
	if wm == nil {
		return cudamem.Slice[byte]{}
	}
	wm.mu.Lock()
	defer wm.mu.Unlock()
	return wm.buffer
}

// Close frees the owned workspace memory. Failure retains ownership for retry
// while preventing further allocations and slices from this manager.
func (wm *WorkspaceManager) Close() error {
	return wm.closeWith(func() error { return wm.buffer.Free() })
}

func (wm *WorkspaceManager) closeWith(free func() error) error {
	if wm == nil {
		return nil
	}
	wm.mu.Lock()
	defer wm.mu.Unlock()

	wm.closed = true

	if wm.ownedBuffer {
		if err := free(); err != nil {
			return fmt.Errorf("cudaplan: free workspace: %w", err)
		}
		wm.buffer = cudamem.Slice[byte]{}
		wm.ownedBuffer = false
		wm.totalBytes = 0
	}
	return nil
}
