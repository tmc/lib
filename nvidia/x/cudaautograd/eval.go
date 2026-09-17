package cudaautograd

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudaref"
)

// ActivationStore manages retained forward activations and checks for modifications or overwrites.
type ActivationStore struct {
	snapshots map[string][]float32
	checksums map[string][32]byte
}

// NewActivationStore creates an empty activation store.
func NewActivationStore() *ActivationStore {
	return &ActivationStore{
		snapshots: make(map[string][]float32),
		checksums: make(map[string][32]byte),
	}
}

// Save records an activation tensor and records its checksum to guard against silent in-place overwrite.
func (s *ActivationStore) Save(name string, data []float32) {
	cp := make([]float32, len(data))
	copy(cp, data)
	s.snapshots[name] = cp
	s.checksums[name] = computeChecksum(cp)
}

// Get retrieves a saved activation while verifying that the stored buffer was not corrupted.
func (s *ActivationStore) Get(name string) ([]float32, error) {
	buf, ok := s.snapshots[name]
	if !ok {
		return nil, fmt.Errorf("%w: missing saved activation %q", ErrSavedActivationStale, name)
	}
	// Verify checksum integrity
	expected := s.checksums[name]
	current := computeChecksum(buf)
	if current != expected {
		return nil, fmt.Errorf("%w: activation %q checksum mismatch (overwritten)", ErrSavedActivationStale, name)
	}
	cp := make([]float32, len(buf))
	copy(cp, buf)
	return cp, nil
}

// Invalidate corrupts or removes a saved activation to test negative lifetime controls.
func (s *ActivationStore) Invalidate(name string) {
	if buf, ok := s.snapshots[name]; ok && len(buf) > 0 {
		buf[0] += 999.0 // Deliberate corruption to trigger checksum mismatch
	} else {
		delete(s.snapshots, name)
	}
}

func computeChecksum(data []float32) [32]byte {
	h := sha256.New()
	var buf [4]byte
	for _, v := range data {
		bits := math.Float32bits(v)
		buf[0] = byte(bits)
		buf[1] = byte(bits >> 8)
		buf[2] = byte(bits >> 16)
		buf[3] = byte(bits >> 24)
		h.Write(buf[:])
	}
	var res [32]byte
	copy(res[:], h.Sum(nil))
	return res
}

// EvaluateForward executes the forward graph on CPU and returns all node values (both outputs and intermediate nodes).
func EvaluateForward(g *cudair.Graph, inputs map[string][]float32) (map[string][]float32, error) {
	// Temporarily mark all non-input nodes as outputs to capture intermediate activations
	origOutputs := g.Outputs()
	allNodes := g.Nodes()
	var newOutputs []*cudair.Node
	for _, n := range allNodes {
		if n.Kind != cudair.KindInput {
			newOutputs = append(newOutputs, n)
		}
	}

	if err := g.SetOutputs(newOutputs...); err != nil {
		return nil, fmt.Errorf("cudaautograd: failed setting intermediate outputs: %w", err)
	}
	defer func() {
		_ = g.SetOutputs(origOutputs...)
	}()

	return cudaref.Evaluate(g, inputs)
}

// EvaluateBackward executes the backward graph using the forward inputs and intermediate activations.
// Returns map from forward target node ID to computed gradient slice.
func EvaluateBackward(
	bwdRes *BackwardResult,
	fwdInputs map[string][]float32,
	fwdOutputs map[string][]float32,
) (map[int][]float32, error) {
	if bwdRes == nil {
		return nil, errors.New("cudaautograd: nil backward result")
	}

	bwdInputs := make(map[string][]float32, len(bwdRes.SavedActivations)+1)
	// Seed gradient dLoss/dLoss = 1.0
	bwdInputs[bwdRes.LossGradInput.Name] = []float32{1.0}

	// Populate saved activations
	for id, savedInputNode := range bwdRes.SavedActivations {
		fwdNode := bwdRes.ForwardNodes[id]
		if fwdNode == nil {
			return nil, fmt.Errorf("cudaautograd: missing forward node for ID %d", id)
		}

		var data []float32
		if fwdNode.Kind == cudair.KindInput {
			data = fwdInputs[fwdNode.Name]
		} else {
			data = fwdOutputs[fwdNode.Name]
		}

		if data == nil {
			return nil, fmt.Errorf("cudaautograd: missing activation buffer for %q", fwdNode.Name)
		}
		bwdInputs[savedInputNode.Name] = data
	}

	// Evaluate backward graph
	bwdOutputs, err := cudaref.Evaluate(bwdRes.BackwardGraph, bwdInputs)
	if err != nil {
		return nil, fmt.Errorf("cudaautograd: backward graph evaluation failed: %w", err)
	}

	// Collect gradients for targets
	grads := make(map[int][]float32, len(bwdRes.Gradients))
	for targetID, gradNode := range bwdRes.Gradients {
		val, ok := bwdOutputs[gradNode.Name]
		if !ok {
			return nil, fmt.Errorf("cudaautograd: missing output gradient for %q", gradNode.Name)
		}
		grads[targetID] = val
	}

	return grads, nil
}
