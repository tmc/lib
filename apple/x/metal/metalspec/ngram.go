// SPDX-License-Identifier: Apache-2.0

package metalspec

import (
	"fmt"
	"sync"
)

// NgramDrafter drafts candidate tokens by finding recurrent n-gram matches in the sequence history.
type NgramDrafter struct {
	n  int
	mu sync.RWMutex
}

// NewNgramDrafter creates an N-gram drafter looking for matches of order n.
func NewNgramDrafter(n int) *NgramDrafter {
	if n <= 0 {
		n = 3
	}
	return &NgramDrafter{n: n}
}

// Draft examines history and proposes up to maxDraft candidate continuation tokens.
func (d *NgramDrafter) Draft(history []int32, maxDraft int) ([]int32, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()

	if len(history) < d.n || maxDraft <= 0 {
		return nil, nil
	}

	matchLen := d.n
	suffix := history[len(history)-matchLen:]

	// Scan backwards for the most recent match
	for i := len(history) - matchLen - 1; i >= 0; i-- {
		match := true
		for j := 0; j < matchLen; j++ {
			if history[i+j] != suffix[j] {
				match = false
				break
			}
		}
		if match {
			startPos := i + matchLen
			var candidates []int32
			for k := 0; k < maxDraft && (startPos+k) < len(history); k++ {
				candidates = append(candidates, history[startPos+k])
			}
			if len(candidates) > 0 {
				return candidates, nil
			}
		}
	}

	return nil, nil
}

// String returns human-readable drafter name.
func (d *NgramDrafter) String() string {
	return fmt.Sprintf("NgramDrafter(n=%d)", d.n)
}
