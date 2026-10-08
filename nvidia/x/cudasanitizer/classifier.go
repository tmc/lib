package cudasanitizer

import (
	"fmt"
	"sync"
)

// SanitizerClassifier classifies sanitizer defect findings and maps them to unique
// crash signatures, deduplicating recurring defects across trial runs.
type SanitizerClassifier struct {
	mu   sync.Mutex
	seen map[[32]byte]CrashObservation
}

// NewSanitizerClassifier allocates an empty classifier.
func NewSanitizerClassifier() *SanitizerClassifier {
	return &SanitizerClassifier{
		seen: make(map[[32]byte]CrashObservation),
	}
}

// Classify calculates the CrashObservation for result using preceding trace and failure identity.
// It reports whether the resulting defect signature is novel (seen for the first time).
// If the signature is novel, it is recorded. The resulting CrashObservation is attached
// to result.CrashObservation.
func (c *SanitizerClassifier) Classify(result *Result, preceding CompletedCrashTrace, failure CrashIdentity) (CrashObservation, bool, error) {
	if c == nil {
		return CrashObservation{}, false, fmt.Errorf("sanitizer classifier: uninitialized")
	}
	obs, err := CrashSignature(result, preceding, failure)
	if err != nil {
		return CrashObservation{}, false, err
	}
	if result != nil {
		result.CrashObservation = &obs
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if _, ok := c.seen[obs.Signature]; ok {
		return obs, false, nil
	}
	c.seen[obs.Signature] = obs
	return obs, true, nil
}

// UniqueCount returns the number of distinct crash signatures observed.
func (c *SanitizerClassifier) UniqueCount() int {
	if c == nil {
		return 0
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.seen)
}

// Observations returns all distinct crash observations recorded so far.
func (c *SanitizerClassifier) Observations() []CrashObservation {
	if c == nil {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]CrashObservation, 0, len(c.seen))
	for _, obs := range c.seen {
		out = append(out, obs)
	}
	return out
}
