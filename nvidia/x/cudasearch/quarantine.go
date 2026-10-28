package cudasearch

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"math"
	"reflect"
	"sort"
	"strings"
	"sync"

	"github.com/tmc/lib/nvidia/x/cudaflightrecorder"
)

// ErrCandidateQuarantined reports an exact offline comparison-failure match.
// Compilation may already have loaded or uploaded resources. The evaluator
// closes its compiled owner, but never calls either arm's Outputs or Run.
var ErrCandidateQuarantined = errors.New("cudasearch: candidate quarantined by offline comparison evidence")

// FailureEvidence opts an executable evaluator into comparison-failure export
// and offline quarantine. Recorder and Writer must be both set or both nil.
// All three resources belong to the caller; the evaluator never closes them.
// The caller also budgets Writer's retention across snapshots; each individual
// archive is bounded by Recorder's snapshot and TAR limits.
// Only an actual candidate-arm reference mismatch is exported. Missing input
// identity or zero execution generation withholds the association; it never
// becomes a wildcard. Set a positive execution-owner Generation to associate.
// An exact ban also requires complete same-name, same-extent output vectors.
// Incomplete outputs remain failed evaluations and readable diagnostic facts,
// but do not establish a comparable numeric mismatch.
type FailureEvidence struct {
	Recorder   *cudaflightrecorder.Recorder
	Writer     io.Writer
	Quarantine *CandidateQuarantine
}

// CandidateQuarantine retains exact associations imported from diagnostic TAR
// bundles. Use NewCandidateQuarantine; the zero value is unusable. Imported
// identities are locally supplied policy, not authenticated native evidence.
// It is safe for concurrent import and evaluation. Caller lifetime is separate
// from any SearchResult; its storage is not charged to MaxMemoryBytes.
type CandidateQuarantine struct {
	mu              sync.Mutex
	entries         []cudaflightrecorder.CandidateAssociation
	maxBytes, bytes int64
}

// NewCandidateQuarantine creates a store with finite count and byte limits.
// Bytes count the store value, reserved entry array and copied string bytes, excluding
// allocator rounding, mutex implementation and transient ReadBundle storage
// (bounded independently to 256 MiB by the recorder reader).
func NewCandidateQuarantine(maxEntries int, maxBytes int64) (*CandidateQuarantine, error) {
	if maxEntries <= 0 || maxEntries > 65536 || maxBytes <= 0 || maxBytes > 64<<20 {
		return nil, fmt.Errorf("cudasearch: invalid quarantine limits")
	}
	base := int64(reflect.TypeFor[CandidateQuarantine]().Size()) + int64(maxEntries)*int64(reflect.TypeFor[cudaflightrecorder.CandidateAssociation]().Size())
	if base > maxBytes {
		return nil, ErrMemoryBudgetExceeded
	}
	return &CandidateQuarantine{entries: make([]cudaflightrecorder.CandidateAssociation, 0, maxEntries), maxBytes: maxBytes, bytes: base}, nil
}

// Import reads one bounded diagnostic TAR through the recorder's strict reader
// and schema decoder. Module-only records cannot enter this store. Refusal
// leaves the store unchanged; duplicate exact associations consume no storage.
func (q *CandidateQuarantine) Import(r io.Reader) error {
	if q == nil || q.maxBytes == 0 {
		return fmt.Errorf("cudasearch: uninitialized quarantine")
	}
	b, err := cudaflightrecorder.ReadBundle(r)
	if err != nil {
		return err
	}
	fault, _, err := b.DecodeFaultEvidence()
	if err != nil {
		return err
	}
	if fault.Candidate == nil {
		return fmt.Errorf("cudasearch: module facts lack exact search association")
	}
	a := *fault.Candidate
	q.mu.Lock()
	defer q.mu.Unlock()
	for _, old := range q.entries {
		if old == a {
			return nil
		}
	}
	n := int64(len(a.Kind) + len(a.IdentityProvenance) + len(a.ExecutableID) + len(a.WorkloadKey) + len(a.InputKey) + len(a.VerificationKey))
	if len(q.entries) == cap(q.entries) || n > q.maxBytes-q.bytes {
		return ErrMemoryBudgetExceeded
	}
	// Admit the entire retained copy before allocating any string.
	a.Kind = strings.Clone(a.Kind)
	a.IdentityProvenance = strings.Clone(a.IdentityProvenance)
	a.ExecutableID = strings.Clone(a.ExecutableID)
	a.WorkloadKey = strings.Clone(a.WorkloadKey)
	a.InputKey = strings.Clone(a.InputKey)
	a.VerificationKey = strings.Clone(a.VerificationKey)
	q.entries = append(q.entries, a)
	q.bytes += n
	return nil
}

// Retained returns the admitted entry count and declared retained bytes.
func (q *CandidateQuarantine) Retained() (int, int64) {
	if q == nil {
		return 0, 0
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	return len(q.entries), q.bytes
}

func (q *CandidateQuarantine) check(a *cudaflightrecorder.CandidateAssociation) error {
	if q == nil {
		return nil
	}
	if q.maxBytes == 0 {
		return fmt.Errorf("cudasearch: uninitialized quarantine")
	}
	if a == nil {
		return nil
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	for _, old := range q.entries {
		if old == *a {
			return ErrCandidateQuarantined
		}
	}
	return nil
}

// candidateAssociation admits temporary identity construction before sorting or
// copying. Fixed caps bound work even outside a memory-limited Search. Reference
// values are hashed in place; no map, slice or executable is retained.
func (e *ExecutableEvaluator) candidateAssociation(ctx context.Context, id string) (*cudaflightrecorder.CandidateAssociation, *memoryReservation, error) {
	if e.FailureEvidence == nil || (e.FailureEvidence.Recorder == nil && e.FailureEvidence.Quarantine == nil) {
		return nil, nil, nil
	}
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	var bytes int64 = 4096
	add := func(n int) bool {
		if n < 0 || int64(n) > (8<<20)-bytes {
			return false
		}
		bytes += int64(n)
		return true
	}
	if len(id) == 0 || len(id) > 4096 {
		return nil, nil, fmt.Errorf("cudasearch: unavailable or oversized candidate identity")
	}
	b := e.Workload
	if len(b.InputNames) > 16384 || len(b.InputIdentities) > 16384 || len(b.WeightIdentities) > 16384 || len(e.References) > 16384 {
		return nil, nil, ErrMemoryBudgetExceeded
	}
	for _, s := range []string{id, b.Name, b.Layout, string(b.NumericalPolicy), string(b.Objective)} {
		if !add(len(s)) {
			return nil, nil, ErrMemoryBudgetExceeded
		}
	}
	for _, s := range b.InputNames {
		if !add(len(s) + 32) {
			return nil, nil, ErrMemoryBudgetExceeded
		}
	}
	for _, m := range []map[string]string{b.InputIdentities, b.WeightIdentities} {
		for k, v := range m {
			if !add(len(k)) || !add(len(v)) || !add(64) {
				return nil, nil, ErrMemoryBudgetExceeded
			}
		}
	}
	if b.Shape.Rank() > 65536 || !add(b.Shape.Rank()*32) {
		return nil, nil, ErrMemoryBudgetExceeded
	}
	for k, v := range e.References {
		if len(v) > 1<<20 || !add(len(k)+32) || !add(len(v)*4) {
			return nil, nil, ErrMemoryBudgetExceeded
		}
	}
	// Key construction, sorted-name populations and both JSON association
	// copies coexist. The factor also covers JSON's six-byte character escapes.
	memory, err := reserveSearchMemory(ctx, bytes*16)
	if err != nil {
		return nil, nil, err
	}
	if e.Generation == 0 || inputKey(b) == "" {
		return nil, memory, nil
	}
	h := sha256.New()
	var word [8]byte
	writeInt := func(n uint64) { binary.LittleEndian.PutUint64(word[:], n); h.Write(word[:]) }
	writeInt(math.Float64bits(e.Policy.Absolute))
	writeInt(math.Float64bits(e.Policy.Relative))
	var flags uint64
	if e.Policy.NaNsEqual {
		flags |= 1
	}
	if e.Policy.IgnoreZeroSign {
		flags |= 2
	}
	writeInt(flags)
	names := make([]string, 0, len(e.References))
	for name := range e.References {
		names = append(names, name)
	}
	sort.Strings(names)
	writeInt(uint64(len(names)))
	for _, name := range names {
		if err := ctx.Err(); err != nil {
			memory.release()
			return nil, nil, err
		}
		writeInt(uint64(len(name)))
		io.WriteString(h, name)
		writeInt(uint64(len(e.References[name])))
		for i, f := range e.References[name] {
			if i%4096 == 0 {
				if err := ctx.Err(); err != nil {
					memory.release()
					return nil, nil, err
				}
			}
			writeInt(uint64(math.Float32bits(f)))
		}
	}
	a := &cudaflightrecorder.CandidateAssociation{Version: 1, Kind: cudaflightrecorder.CandidateReferenceMismatch, IdentityProvenance: cudaflightrecorder.CallerAssertedCandidateIdentity, ExecutableID: id, WorkloadKey: b.Key(), InputKey: inputKey(b), Generation: e.Generation, VerificationKey: hex.EncodeToString(h.Sum(nil))}
	if err := a.Validate(); err != nil {
		memory.release()
		return nil, nil, err
	}
	return a, memory, nil
}

func (e *ExecutableEvaluator) recordComparisonFailure(ctx context.Context, a *cudaflightrecorder.CandidateAssociation, outputs map[string][]float32, failure error) error {
	c := e.FailureEvidence
	if c == nil || c.Recorder == nil {
		return failure
	}
	// Missing names or incompatible extents cannot establish the supported
	// numeric-comparison association. The original verification error survives.
	if len(outputs) != len(e.References) {
		a = nil
	}
	for name, reference := range e.References {
		output, ok := outputs[name]
		if !ok || len(output) != len(reference) {
			a = nil
			break
		}
	}
	_, err := c.Recorder.FaultSnapshot(ctx, c.Writer, cudaflightrecorder.FaultInfo{ErrorName: cudaflightrecorder.CandidateReferenceMismatch, ErrorMessage: "candidate outputs incomplete or different from declared reference; compiler/input/reference identities asserted", Candidate: a})
	return errors.Join(failure, err)
}
