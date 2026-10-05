package cudaschedule

import (
	"fmt"
	"sync"
)

// recordedOutput retains at most limit bytes. Returning len(p) after overflow
// allows the pipe to keep draining; the owner must stop the attempt when
// overflow is observed and retain that refusal in its execution record.
type recordedOutput struct {
	mu       sync.Mutex
	limit    int
	data     []byte
	overflow bool
}

func newRecordedOutput(limit int) (*recordedOutput, error) {
	if limit <= 0 || limit > 64<<20 {
		return nil, fmt.Errorf("recorded run: output limit out of range")
	}
	return &recordedOutput{limit: limit}, nil
}

func (b *recordedOutput) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	keep := len(p)
	if keep > b.limit-len(b.data) {
		keep = b.limit - len(b.data)
		b.overflow = true
	}
	b.data = append(b.data, p[:keep]...)
	return len(p), nil
}

func (b *recordedOutput) snapshot() (string, bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return string(b.data), b.overflow
}
