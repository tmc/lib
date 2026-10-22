package main

import (
	"fmt"
	"math"

	"github.com/tmc/lib/examples/ds4-min/dsv4expertprofile"
	"github.com/tmc/lib/examples/ds4-min/dsv4imatrix"
)

// profileCollectors belong to one request. They are never inherited by a new
// independent State, and their input buffers are not retained.
type profileCollectors struct {
	imatrix     *dsv4imatrix.Collector
	experts     *dsv4expertprofile.Collector
	expertStart int
	imatrixRows [43]uint64
	expertRows  [43]uint64
}

func completeProfileRows(rows [43]uint64, want uint64) error {
	if want == 0 {
		return fmt.Errorf("profile collectors: no rows produced")
	}
	for layer, got := range rows {
		if got != want {
			return fmt.Errorf("profile collectors: layer %d produced %d rows, want %d", layer, got, want)
		}
	}
	return nil
}

func newProfileCollectors(imatrix, experts bool, maxBytes uint64) (*profileCollectors, error) {
	if !imatrix && !experts {
		return nil, nil
	}
	if maxBytes == 0 || maxBytes > 1<<30 {
		return nil, fmt.Errorf("profile collectors: invalid byte limit")
	}
	p := &profileCollectors{}
	var err error
	if imatrix {
		p.imatrix, err = dsv4imatrix.New(43, 256, 4096, 2048, maxBytes/2)
		if err != nil {
			return nil, err
		}
	}
	if experts {
		// Histograms, previous selection and nine bounded unique-ID caches
		// use under 1 MiB at the pinned Flash shape.
		if maxBytes < 1<<20 {
			return nil, fmt.Errorf("expert profile exceeds collector byte limit")
		}
		p.experts, err = dsv4expertprofile.New(43, 256, []int{1, 2, 4, 8, 16, 32, 64, 128, 256})
		if err != nil {
			return nil, err
		}
	}
	return p, nil
}

// The method is promoted by batchStateSessionBackend. Each request receives
// its own collector; the model owner and sibling States remain untouched.
func (s *stateSessionBackend) setProfileCollectors(p *profileCollectors) error {
	if s == nil || s.state == nil {
		return fmt.Errorf("profile collectors: missing State")
	}
	owner, ok := s.state.(interface {
		setProfileCollectors(*profileCollectors) error
	})
	if !ok {
		return fmt.Errorf("profile collectors: backend has no producer")
	}
	return owner.setProfileCollectors(p)
}

func (p *profileCollectors) observe(layer, position, rows int, ids32 []int32, weights, input, mid []float32) error {
	if p == nil {
		return nil
	}
	if rows <= 0 || len(ids32) != rows*6 || len(weights) != len(ids32) {
		return fmt.Errorf("profile collectors: invalid route shape")
	}
	if layer < 0 || layer >= 43 || position < 0 {
		return fmt.Errorf("profile collectors: invalid layer or position")
	}
	for row := 0; row < rows; row++ {
		var seen [256]bool
		for slot := 0; slot < 6; slot++ {
			i := row*6 + slot
			id := ids32[i]
			w := weights[i]
			if id < 0 || id >= 256 || seen[id] || w < 0 || math.IsNaN(float64(w)) || math.IsInf(float64(w), 0) {
				return fmt.Errorf("profile collectors: invalid route")
			}
			seen[id] = true
		}
	}
	ids := make([]int, len(ids32))
	for i, id := range ids32 {
		ids[i] = int(id)
	}
	if p.imatrix != nil {
		if uint64(rows) > math.MaxUint64-p.imatrixRows[layer] {
			return fmt.Errorf("profile collectors: row count overflow")
		}
		if err := p.imatrix.ObserveBatch(layer, 6, ids, input, mid); err != nil {
			return err
		}
		p.imatrixRows[layer] += uint64(rows)
	}
	if p.experts != nil {
		for row := 0; row < rows; row++ {
			if position+row < p.expertStart {
				continue
			}
			if p.expertRows[layer] == math.MaxUint64 {
				return fmt.Errorf("profile collectors: row count overflow")
			}
			if err := p.experts.Observe(layer, position+row, ids[row*6:(row+1)*6], weights[row*6:(row+1)*6]); err != nil {
				return err
			}
			p.expertRows[layer]++
		}
	}
	return nil
}
