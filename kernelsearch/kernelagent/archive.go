package kernelagent

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sync"

	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudaroofline"
	"github.com/tmc/lib/nvidia/x/cudasearch"
	"github.com/tmc/lib/nvidia/x/egraph"
)

// archiveVersion is the only Record version OpenArchive resumes.
const archiveVersion = 2

// Record is the archive's versioned envelope. Attempt is an archive sequence
// reference, not a candidate identity. CandidateID is copied unchanged from
// the executable/search owner; it is empty if compilation produced no identity.
// Inputs and references retain FP32 bits, including signed zero and NaN payloads.
//
// Attempt and outcome records are written only by the owner evaluator's
// observer. Evaluation and OwnerAttempt copy the owner's
// [cudasearch.EvaluationKind] and per-evaluator attempt sequence; BaselineID,
// ConfirmationOf and MeasurementReused copy the owner's start and finish
// records. None of them is an identity scheme of this package.
type Record struct {
	Version, Generation                           int
	Sequence, Attempt, OwnerAttempt               uint64
	Epoch, Kind, Phase, CandidateID, StructuralID string
	Evaluation, BaselineID, ConfirmationOf        string                  `json:",omitempty"`
	MeasurementReused                             bool                    `json:",omitempty"`
	Identity                                      *Identity               `json:",omitempty"`
	Proposal                                      *ProposalEvidence       `json:",omitempty"`
	Diagnostics                                   *Summary                `json:",omitempty"`
	Inputs, References                            map[string][]uint32     `json:",omitempty"`
	WorkloadKey                                   string                  `json:",omitempty"`
	Workload                                      *WorkloadEvidence       `json:",omitempty"`
	Roofline                                      *cudaroofline.Ceiling   `json:",omitempty"`
	Operation                                     *cudaroofline.Operation `json:",omitempty"`
	Numerical                                     *NumericalDeclaration   `json:",omitempty"`
	Derivation                                    []egraph.Action         `json:",omitempty"`
	Compile, Verification, Outcome, Reason        string
	VerificationArm, VerificationIdentity         string
	Unmeasured                                    bool
	CostNs                                        *float64 `json:",omitempty"`
	FaultCauseUnknown                             bool
	Trials                                        []Trial                       `json:",omitempty"`
	Samples                                       []Sample                      `json:",omitempty"`
	Comparison                                    map[string]Comparison         `json:",omitempty"`
	Policy                                        *cudacache.SearchPolicyID     `json:",omitempty"`
	Promotion                                     *cudasearch.PromotionProposal `json:",omitempty"`
}

// Trial transports owner receipts without converting absent timing to zero.
// Timings are separate, non-additive channels, in nanoseconds.
type Trial struct {
	Name, Identity, Kind, Error string
	Round                       int
	Complete                    bool
	Timings                     map[string]int64
}

// Sample records the two condition observations surrounding an owner Run call.
// A rejected sample is diagnostic evidence and cannot supply the search cost.
type Sample struct {
	Identity, Status, Error string
	Before, After           DeviceState
	Floor                   cudaroofline.Judgment
}

// Comparison preserves nonfinite rejected deviations as text, not invalid JSON.
type Comparison struct {
	Elements                 int
	MaxAbsolute, MaxRelative string
}

// WorkloadEvidence transports the existing owner's binding facts. WorkloadKey
// remains WorkloadBindings.Key; this type defines no identity scheme. ResetState
// is a trusted callback identified by the configured evaluator policy, not JSON.
type WorkloadEvidence struct {
	Name, DType, Layout, NumericalPolicy, Objective string
	Shape                                           []int
	InputNames                                      []string
	InputIdentities, WeightIdentities               map[string]string
}

// Archive owns one bounded JSONL append path. Use one process writer at a time.
// It serializes goroutines, syncs every complete record, and refuses malformed
// or torn tails; it never truncates or silently repairs existing evidence.
type Archive struct {
	mu                    sync.Mutex
	root                  *os.Root
	file                  *os.File
	records               []Record
	maxRecord, maxRecords int
	failed                error
}

// OpenArchive opens archive.jsonl inside an existing directory. Bounds are
// positive. Incomplete attempts from earlier epochs are quarantined as unknown
// interruptions through the same append path; no proposal is blamed for a fault.
func OpenArchive(dir string, maxRecordBytes, maxRecords int) (_ *Archive, err error) {
	if maxRecordBytes <= 0 || maxRecordBytes > 16<<20 || maxRecords <= 0 {
		return nil, fmt.Errorf("kernelagent: invalid archive bounds")
	}
	root, err := os.OpenRoot(dir)
	if err != nil {
		return nil, err
	}
	f, err := root.OpenFile("archive.jsonl", os.O_CREATE|os.O_APPEND|os.O_RDWR, 0600)
	if err != nil {
		root.Close()
		return nil, err
	}
	directory, err := root.Open(".")
	if err == nil {
		err = directory.Sync()
		directory.Close()
	}
	if err != nil {
		f.Close()
		root.Close()
		return nil, fmt.Errorf("kernelagent: sync archive directory: %w", err)
	}
	a := &Archive{root: root, file: f, maxRecord: maxRecordBytes, maxRecords: maxRecords}
	defer func() {
		if err != nil {
			a.Close()
		}
	}()
	reader := bufio.NewReaderSize(f, 4096)
	pending := make(map[uint64]Record)
	for {
		var line []byte
		for {
			part, readErr := reader.ReadSlice('\n')
			if len(part) > maxRecordBytes-len(line) {
				return nil, fmt.Errorf("kernelagent: archive record too large")
			}
			line = append(line, part...)
			if readErr == bufio.ErrBufferFull {
				continue
			}
			if readErr == io.EOF && len(line) == 0 {
				break
			}
			if readErr != nil {
				return nil, fmt.Errorf("kernelagent: torn or unreadable archive tail: %w", readErr)
			}
			break
		}
		if len(line) == 0 {
			break
		}
		if len(a.records) >= maxRecords {
			return nil, fmt.Errorf("kernelagent: archive record limit reached")
		}
		var r Record
		d := json.NewDecoder(bytes.NewReader(line))
		d.DisallowUnknownFields()
		if err := d.Decode(&r); err != nil {
			return nil, fmt.Errorf("kernelagent: invalid archive: %w", err)
		}
		if err := d.Decode(new(any)); err != io.EOF {
			return nil, fmt.Errorf("kernelagent: trailing data in archive record")
		}
		if r.Version != archiveVersion || r.Sequence != uint64(len(a.records)+1) || r.Epoch == "" {
			return nil, fmt.Errorf("kernelagent: invalid archive sequence or version")
		}
		if r.Kind == "attempt" {
			pending[r.Sequence] = r
		}
		if r.Kind == "outcome" {
			begin, ok := pending[r.Attempt]
			if !ok || begin.Epoch != r.Epoch {
				return nil, fmt.Errorf("kernelagent: outcome without matching attempt")
			}
			delete(pending, r.Attempt)
		}
		a.records = append(a.records, r)
	}
	// Preserve original sequence order even though the lookup is a map.
	for _, begin := range a.records {
		if _, ok := pending[begin.Sequence]; !ok {
			continue
		}
		r := Record{Epoch: begin.Epoch, Kind: "outcome", Phase: begin.Phase, Evaluation: begin.Evaluation, OwnerAttempt: begin.OwnerAttempt, Attempt: begin.Sequence,
			Outcome: "quarantined", Verification: "unverified", Unmeasured: true, FaultCauseUnknown: true,
			Reason: "interrupted attempt; execution and fault cause unknown; original inputs retained by attempt reference"}
		if _, err := a.append(r); err != nil {
			return nil, err
		}
	}
	return a, nil
}

func (a *Archive) append(r Record) (uint64, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.file == nil {
		return 0, fmt.Errorf("kernelagent: archive closed")
	}
	if a.failed != nil {
		return 0, a.failed
	}
	if len(a.records) >= a.maxRecords {
		return 0, fmt.Errorf("kernelagent: archive record limit reached")
	}
	r.Version, r.Sequence = archiveVersion, uint64(len(a.records)+1)
	data, err := json.Marshal(r)
	if err != nil {
		return 0, err
	}
	data = append(data, '\n')
	if len(data) > a.maxRecord {
		return 0, fmt.Errorf("kernelagent: archive record too large")
	}
	n, err := a.file.Write(data)
	if err == nil && n != len(data) {
		err = io.ErrShortWrite
	}
	if err == nil {
		err = a.file.Sync()
	}
	if err != nil {
		a.failed = err
		return 0, err
	}
	// Retain a detached value with exactly the representation written to disk.
	var saved Record
	if err := json.Unmarshal(data, &saved); err != nil {
		a.failed = err
		return 0, err
	}
	a.records = append(a.records, saved)
	return r.Sequence, nil
}

// Records returns detached envelopes, in append order.
func (a *Archive) Records() []Record {
	a.mu.Lock()
	defer a.mu.Unlock()
	data, _ := json.Marshal(a.records)
	var out []Record
	_ = json.Unmarshal(data, &out)
	return out
}

// CandidateRecords returns detached records indexed by the executable owner's
// exact identity string. It does not hash or reinterpret candidate identity.
func (a *Archive) CandidateRecords(id string) []Record {
	var out []Record
	if id == "" {
		return out
	}
	for _, r := range a.Records() {
		if r.CandidateID == id {
			out = append(out, r)
		}
	}
	return out
}

// Close closes the file and directory handle. It does not remove evidence.
func (a *Archive) Close() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.file == nil {
		return nil
	}
	err := a.file.Close()
	a.file = nil
	if e := a.root.Close(); err == nil {
		err = e
	}
	return err
}
