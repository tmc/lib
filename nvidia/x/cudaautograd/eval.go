package cudaautograd

import (
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"slices"
	"sync/atomic"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudaref"
)

// Activations records the forward values that one backward pass of a
// [BackwardResult] reads. [Forward] creates it and [Backward] consumes it.
//
// The saved buffers alias the values returned by [Forward] and the caller's
// input slices; they are not copied. Activations binds the result it was
// recorded for, the length of each buffer, and a SHA-256 digest of their
// contents, so [Backward] refuses buffers written after [Forward] instead of
// silently differentiating at the wrong point.
//
// Activations covers host buffers only. For native execution the saved
// device buffers must stay live and unwritten until the backward
// invocation completes, which this type does not observe.
type Activations struct {
	result   *BackwardResult
	saved    []savedActivation // ordered by forward node ID
	digest   [32]byte
	released atomic.Bool
}

type savedActivation struct {
	id    int    // forward node ID
	input string // backward graph input name
	data  []float32
}

func digestActivations(saved []savedActivation) [32]byte {
	h := sha256.New()
	var buf [8]byte
	for _, a := range saved {
		binary.LittleEndian.PutUint64(buf[:], uint64(a.id))
		h.Write(buf[:])
		binary.LittleEndian.PutUint64(buf[:], uint64(len(a.data)))
		h.Write(buf[:])
		for _, v := range a.data {
			binary.LittleEndian.PutUint32(buf[:4], math.Float32bits(v))
			h.Write(buf[:4])
		}
	}
	var sum [32]byte
	h.Sum(sum[:0])
	return sum
}

// digestSnapshot hashes the executable graph contract in stable node-ID order.
// Source locations are attribution only and do not change computation.
func digestSnapshot(snap *cudair.Snapshot) [32]byte {
	h := sha256.New()
	var buf [8]byte
	writeInt := func(v int) {
		binary.LittleEndian.PutUint64(buf[:], uint64(int64(v)))
		h.Write(buf[:])
	}
	writeCount := func(n int) { writeInt(n) }
	writeString := func(s string) {
		writeCount(len(s))
		h.Write([]byte(s))
	}
	writeInts := func(values []int) {
		writeCount(len(values))
		for _, v := range values {
			writeInt(v)
		}
	}

	h.Write([]byte("cudaautograd/cudair-snapshot/v1"))
	nodes := snap.Nodes()
	writeCount(len(nodes))
	for _, n := range nodes {
		writeInt(n.ID)
		writeString(n.Name)
		writeInt(int(n.Kind))
		writeString(n.Op)
		writeCount(len(n.Inputs))
		for _, in := range n.Inputs {
			writeInt(in.ID)
		}
		writeInts(n.Shape)
		writeInt(int(n.Type))
		writeInt(int(n.Layout))
		writeInts(n.Strides)
		writeInt(int(n.Effect))
		if n.StorageAlias == nil {
			writeInt(0)
		} else {
			writeInt(1)
			writeInt(n.StorageAlias.ID)
		}
		writeCount(len(n.ConstValues))
		for _, v := range n.ConstValues {
			binary.LittleEndian.PutUint32(buf[:4], math.Float32bits(v))
			h.Write(buf[:4])
		}
		writeInts(n.Axes)
		writeInts(n.Perm)
		writeInts(n.SliceStarts)
		writeInts(n.SliceEnds)
		writeInts(n.SliceSteps)
		writeInt(n.Axis)
	}
	outputs := snap.Outputs()
	writeCount(len(outputs))
	for _, out := range outputs {
		writeInt(out.ID)
	}

	var digest [32]byte
	copy(digest[:], h.Sum(nil))
	return digest
}

// Forward evaluates g on the CPU and records the activations that the
// backward pass of res reads. It returns the value of every non-input node,
// keyed by node name, and the activations to pass to [Backward].
//
// The graph g must retain the executable semantics res was derived from;
// Forward reports [ErrActivationsMismatch] if any node or output differs.
// Evaluation uses the retained immutable snapshot. A saved buffer whose
// length differs from its forward node at differentiation time is reported
// as [ErrSavedActivationStale].
func Forward(res *BackwardResult, g *cudair.Graph, inputs map[string][]float32) (map[string][]float32, *Activations, error) {
	if res == nil || res.forward == nil {
		return nil, nil, fmt.Errorf("%w: backward result not created by Differentiate", ErrActivationsMismatch)
	}
	if g == nil {
		return nil, nil, errors.New("cudaautograd: nil forward graph")
	}
	if len(g.Nodes()) != len(res.forward.Nodes()) {
		return nil, nil, fmt.Errorf("%w: forward graph node set changed", ErrActivationsMismatch)
	}
	for _, n := range g.Nodes() {
		if n == nil {
			return nil, nil, fmt.Errorf("%w: forward graph contains a nil node", ErrActivationsMismatch)
		}
		if !res.forward.Contains(n) {
			return nil, nil, fmt.Errorf("%w: forward node %s (id %d) is not from the differentiated graph", ErrActivationsMismatch, n.Name, n.ID)
		}
	}
	current, err := g.Snapshot()
	if err != nil {
		return nil, nil, fmt.Errorf("%w: current forward graph is invalid: %w", ErrActivationsMismatch, err)
	}
	if digestSnapshot(current) != res.forwardDigest {
		for id := range res.SavedActivations {
			want := res.forward.NodeByID(id)
			got := current.NodeByID(id)
			if want != nil && got != nil && want.NumElements() != got.NumElements() {
				return nil, nil, fmt.Errorf("%w: %w: saved node %s changed length from %d to %d", ErrActivationsMismatch, ErrSavedActivationStale, want.Name, want.NumElements(), got.NumElements())
			}
		}
		return nil, nil, fmt.Errorf("%w: forward graph semantics changed after differentiation", ErrActivationsMismatch)
	}

	valuesByID, err := cudaref.EvaluateSnapshotAll(res.forward, inputs)
	if err != nil {
		return nil, nil, err
	}
	values := make(map[string][]float32, len(res.forward.Nodes()))
	for _, n := range res.forward.Nodes() {
		if n.Kind != cudair.KindInput {
			values[n.Name] = valuesByID[n.ID]
		}
	}

	saved := make([]savedActivation, 0, len(res.SavedActivations))
	for id, in := range res.SavedActivations {
		fwd := res.forward.NodeByID(id)
		if fwd == nil {
			return nil, nil, fmt.Errorf("%w: saved node id %d not in forward snapshot", ErrActivationsMismatch, id)
		}
		var data []float32
		if fwd.Kind == cudair.KindInput {
			data = inputs[fwd.Name]
		} else {
			data = valuesByID[id]
		}
		if len(data) != fwd.NumElements() {
			return nil, nil, fmt.Errorf("%w: %s has %d elements, want %d", ErrSavedActivationStale, fwd.Name, len(data), fwd.NumElements())
		}
		saved = append(saved, savedActivation{id: id, input: in.Name, data: data})
	}
	slices.SortFunc(saved, func(a, b savedActivation) int { return a.id - b.id })
	acts := &Activations{result: res, saved: saved, digest: digestActivations(saved)}
	return values, acts, nil
}

// EvaluateForward executes the forward graph on CPU and returns all node values (both outputs and intermediate nodes).
func EvaluateForward(g *cudair.Graph, inputs map[string][]float32) (map[string][]float32, error) {
	if g == nil {
		return nil, errors.New("cudaautograd: nil forward graph")
	}
	snap, err := g.Snapshot()
	if err != nil {
		return nil, fmt.Errorf("cudaautograd: forward snapshot failed: %w", err)
	}
	valuesByID, err := cudaref.EvaluateSnapshotAll(snap, inputs)
	if err != nil {
		return nil, err
	}
	values := make(map[string][]float32, len(snap.Nodes()))
	for _, n := range snap.Nodes() {
		if n.Kind != cudair.KindInput {
			values[n.Name] = valuesByID[n.ID]
		}
	}
	return values, nil
}

// Backward evaluates the backward graph of res on the CPU using activations
// recorded by [Forward] for the same res, and returns the gradient of each
// target keyed by forward node ID.
//
// Backward consumes acts once it has matched them to res, including when it
// then refuses them: a second call with the same activations reports
// [ErrSavedActivationStale] ("already consumed"), as does any saved buffer
// written since [Forward] ("changed after forward"). Nil arguments and
// activations recorded for a different result report
// [ErrActivationsMismatch] and do not consume acts.
func Backward(res *BackwardResult, acts *Activations) (map[int][]float32, error) {
	switch {
	case res == nil:
		return nil, fmt.Errorf("%w: nil backward result", ErrActivationsMismatch)
	case acts == nil:
		return nil, fmt.Errorf("%w: nil activations", ErrActivationsMismatch)
	case acts.result != res:
		return nil, fmt.Errorf("%w: activations recorded for a different backward result", ErrActivationsMismatch)
	}
	if acts.released.Swap(true) {
		return nil, fmt.Errorf("%w: activations already consumed", ErrSavedActivationStale)
	}
	// Digest and evaluate the same private copies, so a buffer cannot
	// change between the check and its use. A caller writing a saved
	// buffer concurrently with Backward is a data race regardless; this
	// copy is defensive and no test exercises it.
	saved := make([]savedActivation, len(acts.saved))
	for i, a := range acts.saved {
		saved[i] = savedActivation{id: a.id, input: a.input, data: slices.Clone(a.data)}
	}
	if digestActivations(saved) != acts.digest {
		return nil, fmt.Errorf("%w: saved buffers changed after forward evaluation", ErrSavedActivationStale)
	}

	bwdInputs := make(map[string][]float32, len(saved)+1)
	bwdInputs[res.LossGradInput.Name] = []float32{1.0}
	for _, a := range saved {
		bwdInputs[a.input] = a.data
	}
	bwdOutputs, err := cudaref.Evaluate(res.BackwardGraph, bwdInputs)
	if err != nil {
		return nil, fmt.Errorf("cudaautograd: backward graph evaluation failed: %w", err)
	}

	grads := make(map[int][]float32, len(res.Gradients))
	for targetID, gradNode := range res.Gradients {
		val, ok := bwdOutputs[gradNode.Name]
		if !ok {
			return nil, fmt.Errorf("cudaautograd: missing output gradient for %q", gradNode.Name)
		}
		grads[targetID] = val
	}
	return grads, nil
}
