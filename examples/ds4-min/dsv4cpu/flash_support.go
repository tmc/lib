package dsv4cpu

import (
	"bytes"
	"errors"
	"io"
	"math"
	"slices"

	"github.com/tmc/lib/examples/ds4-min/internal/dsparkstate"
)

// flashSupport owns target-derived rows, not draft KV or shared model weights.
// It is optional and must be attached before the first target evaluation.
type flashSupport struct {
	state    dsparkstate.Envelope
	staged   []float32
	captured []bool
	epoch    uint64
}

// combineLayer is the final target FFN HC operation, followed by observation.
// It is shared with actual-body controls; no second target evaluation occurs.
func (s *flashState) combineLayer(dst, block, after, post, comb []float32, il int) error {
	if err := Combine(dst, block, after, post, comb); err != nil {
		return err
	}
	if s.support != nil {
		return s.support.capture(il, dst)
	}
	return nil
}

func (s *flashState) attachSupport(targets []int, block int) error {
	if s == nil || s.closed || s.failed || s.position != 0 || s.support != nil || s.model == nil || s.model.digest == ([32]byte{}) || len(targets) == 0 || len(targets) > 8 || block < 1 || block > 16 {
		return errors.New("CPU support: invalid attachment")
	}
	for i, layer := range targets {
		if layer < 0 || layer >= 43 || (i > 0 && targets[i-1] >= layer) {
			return errors.New("CPU support: invalid selected layer order")
		}
	}
	rows, values, err := dsparkstate.Shape(s.context, block, len(targets), 4096)
	if err != nil {
		return err
	}
	j, err := dsparkstate.NewJournal(rows, 1)
	if err != nil {
		return err
	}
	s.support = &flashSupport{
		state:  dsparkstate.Envelope{Platform: 3, Model: s.model.digest, Context: s.context, Block: block, Targets: slices.Clone(targets), Journal: j, History: make([]float32, values)},
		staged: make([]float32, len(targets)*4096), captured: make([]bool, len(targets)), epoch: 1,
	}
	return nil
}

func (v *flashSupport) capture(layer int, hc []float32) error {
	i, selected := slices.BinarySearch(v.state.Targets, layer)
	if !selected {
		return nil
	}
	if len(hc) != 4*4096 || v.captured[i] {
		return errors.New("CPU support: invalid or repeated capture")
	}
	row := v.staged[i*4096 : (i+1)*4096]
	for d := range row {
		var sum float32
		for lane := 0; lane < 4; lane++ {
			sum += hc[lane*4096+d] * 0.25
		}
		if math.IsNaN(float64(sum)) || math.IsInf(float64(sum), 0) {
			return errors.New("CPU support: nonfinite capture")
		}
		row[d] = sum
	}
	v.captured[i] = true
	return nil
}

func (v *flashSupport) publish(position, token int) error {
	for _, captured := range v.captured {
		if !captured {
			return errors.New("CPU support: incomplete capture")
		}
	}
	// Journal append is the last fallible operation; no partial trusted row is
	// published when its contiguous target frontier check fails.
	if err := v.state.Journal.Append(position, token, v.epoch); err != nil {
		return err
	}
	n := len(v.staged)
	slot := position % len(v.state.Journal.Rows)
	copy(v.state.History[slot*n:(slot+1)*n], v.staged)
	clear(v.captured)
	clear(v.staged)
	return nil
}

func (v *flashSupport) clone(frontier int) (*flashSupport, error) {
	if err := v.state.Validate(frontier, 0); err != nil {
		return nil, err
	}
	for _, captured := range v.captured {
		if captured {
			return nil, errors.New("CPU support: pending capture")
		}
	}
	out := *v
	out.state.Targets = slices.Clone(v.state.Targets)
	out.state.Journal = v.state.Journal.Clone()
	out.state.History = slices.Clone(v.state.History)
	out.staged = slices.Clone(v.staged)
	out.captured = slices.Clone(v.captured)
	return &out, nil
}

func (v *flashSupport) sameIdentity(other *flashSupport) bool {
	return other != nil && v.state.Platform == other.state.Platform && v.state.Model == other.state.Model && v.state.Context == other.state.Context && v.state.Block == other.state.Block && slices.Equal(v.state.Targets, other.state.Targets)
}

func (v *flashSupport) reset() error {
	if v.state.Journal == nil {
		return errors.New("CPU support: missing journal")
	}
	j, err := dsparkstate.NewJournal(len(v.state.Journal.Rows), 1)
	if err != nil {
		return err
	}
	clear(v.state.History)
	clear(v.staged)
	clear(v.captured)
	// Epoch is local ownership metadata; reset discards the complete frontier.
	v.state.Journal = j
	v.epoch = 1
	return nil
}

func (v *flashSupport) close() {
	clear(v.state.History)
	clear(v.staged)
	clear(v.captured)
	clear(v.state.Journal.Rows)
	*v = flashSupport{}
}

func (s *flashState) saveSupport(dst io.Writer) error {
	if err := s.support.state.Validate(s.position, 0); err != nil {
		return err
	}
	for _, captured := range s.support.captured {
		if captured {
			return errors.New("CPU support: pending capture")
		}
	}
	var core bytes.Buffer
	if err := s.saveCore(&core); err != nil {
		return err
	}
	return dsparkstate.WriteContainer(dst, core.Bytes(), &s.support.state, s.position, 0, 1<<30)
}

func (s *flashState) loadSupport(src io.Reader) error {
	core, payload, _, err := dsparkstate.ReadContainer(src, 3, 1<<30)
	if err != nil {
		return err
	}
	next, err := s.model.newState(s.context)
	if err != nil {
		return err
	}
	defer next.close()
	r := bytes.NewReader(core)
	if err := next.loadCore(r); err != nil {
		return err
	}
	if r.Len() != 0 {
		return errors.New("CPU support: trailing core bytes")
	}
	envelope, err := dsparkstate.Read(payload, &s.support.state, next.position, 0)
	if err != nil {
		return err
	}
	next.support = &flashSupport{state: *envelope, staged: make([]float32, len(envelope.Targets)*4096), captured: make([]bool, len(envelope.Targets)), epoch: envelope.Journal.Epoch}
	// All core and support identity, journal, count and checksum checks have
	// completed. Only Go-owned buffers are replaced, so cleanup cannot fail.
	old := *s
	*s = *next
	*next = old
	return nil
}
