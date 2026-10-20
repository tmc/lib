package ncclinspector

import (
	"bytes"
	"crypto/sha256"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"strings"
)

// ProgressBinding is an explicit caller-owned provenance association. It names
// an actual Inspector identity and monitor registration, not an operation or
// collective sequence derived from counter slots. Evidence names the caller's
// independent association artifact; this package does not authenticate it.
type ProgressBinding struct {
	Capture      ProgressCapture
	Communicator string // 0x plus16 complete lowercase hex digits
	Rank         int
	Registration uint64
	Evidence     string
}

// ProgressObservation retains a selected reported row and copy admission.
// Fresh means the reader verified that row's successful, nonstale copy chain;
// it does not calibrate GPU timestamps or establish atomic GPU progress.
type ProgressObservation struct {
	Record ProgressRecord
	Fresh  bool
}

// ProgressSlotChange compares raw diagnostic counts for one source-defined
// slot. Delta is observed arithmetic, not bytes, operations or completed
// collectives. A decrease withholds the entire pair; reset/wrap is unknown.
type ProgressSlotChange struct {
	Slot          int
	Before, After uint64
	Status        string // increased or unchanged
	Delta         uint64
}

// ProgressComparison contains adjacent successful copy generations under the
// selected registration and a complete, omission-free exporter artifact.
// Even unchanged counts never establish a stall, failure or complete GPU work.
type ProgressComparison struct {
	FromGeneration, ToGeneration uint64
	Slots                        []ProgressSlotChange
}

// ProgressSummary separates reported source facts from caller assertions.
// ExporterComplete concerns this bounded file only, never upstream population.
// Missing terminal, output loss, failed/stale copies and decreases withhold
// dependent comparisons. Inspector operation/drop facts are retained separately
// and never fill mirror, phase or proxy loss. No diagnostic verdict is issued.
type ProgressSummary struct {
	Asserted         ProgressBinding
	Inspector        Record
	InspectorSHA256  string
	Observations     []ProgressObservation
	Comparisons      []ProgressComparison
	ExporterComplete bool
	ExporterDropped  uint64
	UpstreamLoss     string
	ResetCoverage    string
	Withheld         []string
}

// SummarizeProgress binds a privately checked custom sidecar to an actual
// existing Inspector reader record. It re-reads Raw and rejects edited decoded
// fields. Full communicator/rank/host/PID and supplied boot/process/capture/
// build/worker/registration identities must agree. Inspector clocks or drops
// do not calibrate the monitor or qualify its loss. Source or identity failure
// returns an error, never fabricated healthy or zero values.
func SummarizeProgress(t *ProgressTrace, b ProgressBinding, record Record) (ProgressSummary, error) {
	var out ProgressSummary
	if t == nil || !t.read || len(t.rows) == 0 {
		return out, errors.New("progress: trace was not read")
	}
	if b.Capture != t.capture || b.Registration == 0 || b.Rank < 0 || b.Evidence == "" || len(b.Evidence) > 4096 {
		return out, errors.New("progress: invalid asserted binding")
	}
	if err := ValidateCommunicatorHash(b.Communicator); err != nil {
		return out, err
	}
	if b.Communicator != strings.ToLower(b.Communicator) {
		return out, errors.New("progress: communicator must be canonical")
	}
	actual, err := Read(bytes.NewReader(record.Raw), Limits{Bytes: 1 << 20, Line: 1 << 20, Records: 1})
	if err != nil || len(actual) != 1 || !reflect.DeepEqual(actual[0], record) {
		return out, errors.New("progress: Inspector record is absent or edited")
	}
	if record.Header.ID != b.Communicator || record.Header.Rank != b.Rank || record.Metadata.Hostname != b.Capture.Host || uint64(record.Metadata.PID) != b.Capture.PID {
		return out, errors.New("progress: Inspector identity differs")
	}
	out.Asserted = b
	out.Inspector = actual[0]
	out.InspectorSHA256 = fmt.Sprintf("%x", sha256.Sum256(record.Raw))
	out.ExporterComplete = t.complete
	out.ExporterDropped = t.rows[len(t.rows)-1].DroppedBefore
	out.UpstreamLoss, out.ResetCoverage = "unknown", "unknown"
	out.Withheld = []string{"GPU completion", "stall", "GPU failure", "population zero loss", "counter reset or wrap", "GPU progress age", "cross-clock ordering"}
	var previous *ProgressRecord
	for _, row := range t.Records() {
		if row.Terminal {
			continue
		}
		if row.Registration != b.Registration {
			continue
		}
		if "0x"+row.Communicator != b.Communicator || row.Rank != b.Rank {
			return ProgressSummary{}, errors.New("progress: selected registration identity differs")
		}
		fresh := row.MirrorPresent && row.copyAdmitted
		out.Observations = append(out.Observations, ProgressObservation{row, fresh})
		if !fresh {
			switch row.Event {
			case "copy_launched", "copy_pending": // not a sample; wait for this attempt
			default:
				previous = nil
			}
			continue
		}
		if previous != nil && t.complete && previous.Generation != ^uint64(0) && row.Generation == previous.Generation+1 {
			pair := ProgressComparison{FromGeneration: previous.Generation, ToGeneration: row.Generation}
			decreased := false
			for slot, before := range previous.Mirror.Counts {
				after := row.Mirror.Counts[slot]
				if after < before {
					decreased = true
					break
				}
				status := "unchanged"
				if after > before {
					status = "increased"
				}
				pair.Slots = append(pair.Slots, ProgressSlotChange{slot, before, after, status, after - before})
			}
			if !decreased {
				out.Comparisons = append(out.Comparisons, pair)
			} else {
				out.Withheld = append(out.Withheld, "decreasing diagnostic counter pair")
			}
		}
		copy := row
		previous = &copy
	}
	if len(out.Observations) == 0 {
		return ProgressSummary{}, errors.New("progress: selected registration absent")
	}
	if !t.complete {
		out.Withheld = append(out.Withheld, "incomplete or lossy exporter artifact")
	}
	// Detach optional Inspector bodies from caller-owned pointers as well as Raw.
	out.Inspector.Raw = slices.Clone(out.Inspector.Raw)
	return out, nil
}
