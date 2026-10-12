package cuptipcsampling

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"math/bits"
	"strings"
	"time"

	"github.com/tmc/lib/nvidia/cupti"
)

// PhaseRequest records what the caller asked of a phase capture, so a
// manifest can show it beside what the session actually did.
type PhaseRequest struct {
	Period     uint32        // requested Config.Period
	MaxBatches uint          // requested PhaseLimits.MaxBatches
	Window     time.Duration // requested capture duration; zero if none was requested
}

// PhaseObservation records what the caller observed around a phase capture
// that [PhaseReport] does not carry.
type PhaseObservation struct {
	// Start and End bound the capture. Zero times mean not observed.
	Start, End time.Time

	// Raw is the retained output: the concatenation of Batch.AppendBinary
	// over the report's batches, in order.
	Raw []byte
}

// Coverage classifies how much of a phase's sample population the stall
// breakdown represents.
type Coverage string

const (
	// CoverageMeasured means every eligible sample is in a named PC's
	// selected reasons, collection completed at a verified period, and
	// nothing was lost.
	CoverageMeasured Coverage = "measured"
	// CoverageReduced means samples exist but some were lost, unattributed,
	// unreported or left in the native buffer, or the phase or its period
	// readback was not clean.
	CoverageReduced Coverage = "reduced"
	// CoverageUnavailable means no reason samples were collected. It is not
	// a measurement of zero stalls.
	CoverageUnavailable Coverage = "unavailable"
)

// CleanupState classifies native resource ownership after a phase.
type CleanupState string

const (
	CleanupReleased CleanupState = "released" // session storage released
	CleanupPending  CleanupState = "pending"  // session not closed; caller still owns cleanup
	CleanupFailed   CleanupState = "failed"   // session not closed after a failure; retry Close
)

// PhaseManifest seals the accounting of one phase capture. Requested values
// are recorded beside actual ones rather than substituted for them.
//
// Sample counts are exact native integers. TotalSamples includes non-user
// and dropped samples; EligibleSamples excludes them. Named and Unattributed
// sum non-companion reason samples on PCs with and without a kernel name;
// Companion sums _not_issued reasons, which are subsets of their base reasons
// and are never added to Named or Unattributed. Unreported is the eligible
// population no selected reason accounts for.
type PhaseManifest struct {
	RequestedPeriod uint32
	ActualPeriod    uint32 // native readback; zero if the session reported none
	PeriodVerified  bool   // readback matched the configuration

	RequestedWindow time.Duration
	Start, End      time.Time

	RequestedBatches uint
	Batches          int // batches retained in the report
	InvalidBatches   int // retained batches that failed decoding; not counted below

	FinalDrain   bool // a read after Stop found no remaining PC records
	DrainLimited bool // MaxBatches stopped the final drain; records may remain
	Complete     bool // final drain reached and no PC records remain

	TotalSamples    uint64
	NonUserSamples  uint64
	LostSamples     uint64 // dropped by hardware, buffers or failed reads
	EligibleSamples uint64
	Named           uint64
	Unattributed    uint64
	Companion       uint64
	Unreported      uint64

	// RemainingPCs is the latest PC record count reported remaining by a
	// read after Stop plus the final batch's count, copied from the report.
	// The counts may describe the same records, so this conservative record
	// count is not a distinct-record total or a sample-loss count. Zero does
	// not mean nothing remained; DrainLimited reports records of unknown count.
	RemainingPCs uint64
	HardwareFull bool

	RawSHA256 [sha256.Size]byte

	Cleanup       CleanupState
	SessionFailed bool
	Coverage      Coverage
}

// SealPhase builds a manifest from a phase request, its report and the
// caller's observation. r.Batches must hold every batch the phase drained,
// including those delivered to a consumer, and obs.Raw must be their encoding.
//
// SealPhase refuses inconsistent accounting: a requested period other than
// the session's, a batch count that disagrees with the session's reads, a
// sample total or dropped count that disagrees with the batches, raw output
// that does not encode them, a reason absent from the session inventory,
// reason samples exceeding a batch's eligible population, or a companion
// reason exceeding its selected base reason on the same PC.
func SealPhase(req PhaseRequest, r PhaseReport, obs PhaseObservation) (PhaseManifest, error) {
	m := PhaseManifest{
		RequestedPeriod:  req.Period,
		PeriodVerified:   r.Info.Verified,
		RequestedWindow:  req.Window,
		Start:            obs.Start,
		End:              obs.End,
		RequestedBatches: req.MaxBatches,
		Batches:          len(r.Batches),
		FinalDrain:       r.FinalDrained,
		DrainLimited:     r.DrainLimited,
		LostSamples:      r.DroppedSamples,
		RemainingPCs:     r.RemainingPCs,
		HardwareFull:     r.HardwareFull,
		RawSHA256:        sha256.Sum256(obs.Raw),
		SessionFailed:    r.Info.Failed,
	}
	if !obs.Start.IsZero() && !obs.End.IsZero() && obs.End.Before(obs.Start) {
		return PhaseManifest{}, errors.New("cuptipcsampling: phase window ends before it starts")
	}
	for _, a := range r.Info.Effective {
		if a.AttributeType == cupti.PcSamplingConfigurationAttrTypeSamplingPeriod && a.AttributeStatus == 0 {
			m.ActualPeriod = uint32(a.AttributeData[0])
		}
	}
	if req.Period != r.Info.Config.Period {
		return PhaseManifest{}, fmt.Errorf("cuptipcsampling: requested period %d differs from session period %d", req.Period, r.Info.Config.Period)
	}
	if uint64(r.BatchesDrained) != uint64(len(r.Batches)) {
		return PhaseManifest{}, fmt.Errorf("cuptipcsampling: batch count %d differs from %d drained", len(r.Batches), r.BatchesDrained)
	}
	// Every successful session read yields one batch, and the final batch is
	// not a read. A failed read counts without yielding a batch, and fails
	// the session.
	reads := uint64(r.Info.Reads)
	if r.FinalBatch {
		reads++
	}
	if n := uint64(len(r.Batches)); n != reads && (!r.Info.Failed || n > reads) {
		return PhaseManifest{}, fmt.Errorf("cuptipcsampling: %d batches from %d session reads (final batch %v)", n, r.Info.Reads, r.FinalBatch)
	}

	var raw []byte
	var dropped uint64
	var ok bool
	for i, b := range r.Batches {
		var err error
		if raw, err = b.AppendBinary(raw); err != nil {
			return PhaseManifest{}, fmt.Errorf("cuptipcsampling: batch %d: %w", i, err)
		}
		if dropped, ok = add(dropped, b.DroppedSamples); !ok {
			return PhaseManifest{}, errors.New("cuptipcsampling: dropped sample count overflow")
		}
		if !b.Valid {
			m.InvalidBatches++
			continue
		}
		if err := m.addBatch(r.Info.Inventory, r.Info.Config.ReasonIDs, b); err != nil {
			return PhaseManifest{}, fmt.Errorf("cuptipcsampling: batch %d: %w", i, err)
		}
	}
	if !bytes.Equal(raw, obs.Raw) {
		return PhaseManifest{}, errors.New("cuptipcsampling: raw output does not encode report batches")
	}
	if m.TotalSamples != r.TotalSamples {
		return PhaseManifest{}, fmt.Errorf("cuptipcsampling: report total %d differs from batch total %d", r.TotalSamples, m.TotalSamples)
	}
	if r.DroppedSamples < dropped {
		return PhaseManifest{}, fmt.Errorf("cuptipcsampling: report dropped %d below batch dropped %d", r.DroppedSamples, dropped)
	}

	last := uint64(0)
	if n := len(r.Batches); n > 0 {
		last = r.Batches[n-1].RemainingPCs
	}
	m.Complete = r.FinalDrained && !r.DrainLimited && r.RemainingPCs == 0 && last == 0

	switch {
	case r.Info.Closed:
		m.Cleanup = CleanupReleased
	case r.Info.Failed:
		m.Cleanup = CleanupFailed
	default:
		m.Cleanup = CleanupPending
	}

	switch {
	case m.Named == 0 && m.Unattributed == 0:
		m.Coverage = CoverageUnavailable
	case !m.Complete || m.LostSamples != 0 || m.HardwareFull ||
		!m.PeriodVerified || m.ActualPeriod != m.RequestedPeriod ||
		m.Unattributed != 0 || m.Unreported != 0 || m.InvalidBatches != 0 ||
		m.SessionFailed || r.Canceled || r.ConsumerErr != nil:
		m.Coverage = CoverageReduced
	default:
		m.Coverage = CoverageMeasured
	}
	return m, nil
}

// addBatch adds one valid batch's populations to m. selected lists the
// session's reason indexes.
func (m *PhaseManifest) addBatch(inv ReasonInventory, selected []uint32, b Batch) error {
	if b.CUPTIVersion != inv.CUPTIVersion {
		return fmt.Errorf("batch version %d differs from inventory version %d", b.CUPTIVersion, inv.CUPTIVersion)
	}
	if b.NonUserSamples > b.TotalSamples || b.DroppedSamples > b.TotalSamples-b.NonUserSamples {
		return errors.New("non-user and dropped samples exceed batch total")
	}
	eligible := b.TotalSamples - b.NonUserSamples - b.DroppedSamples
	var named, unattributed, companion uint64
	for _, pc := range b.PCs {
		if err := checkCompanions(inv, selected, pc); err != nil {
			return err
		}
		for _, r := range pc.Reasons {
			name := inv.Lookup(r.Index)
			if name == "" {
				return fmt.Errorf("reason index %d not in inventory", r.Index)
			}
			n := uint64(r.Samples) // uint32 counts: these sums cannot overflow before the check below
			switch {
			case isCompanion(name):
				companion += n
			case pc.Name == "":
				unattributed += n
			default:
				named += n
			}
		}
	}
	if named+unattributed > eligible {
		return fmt.Errorf("reason samples %d exceed eligible %d", named+unattributed, eligible)
	}
	var ok [7]bool
	m.TotalSamples, ok[0] = add(m.TotalSamples, b.TotalSamples)
	m.NonUserSamples, ok[1] = add(m.NonUserSamples, b.NonUserSamples)
	m.EligibleSamples, ok[2] = add(m.EligibleSamples, eligible)
	m.Named, ok[3] = add(m.Named, named)
	m.Unattributed, ok[4] = add(m.Unattributed, unattributed)
	m.Companion, ok[5] = add(m.Companion, companion)
	m.Unreported, ok[6] = add(m.Unreported, eligible-named-unattributed)
	for _, o := range ok {
		if !o {
			return errors.New("sample count overflow")
		}
	}
	return nil
}

// checkCompanions refuses a _not_issued companion count above its base
// reason's count on the same PC. A companion counts a subset of its base
// reason's samples. The check applies only when the base reason was selected;
// otherwise its count was never collected.
func checkCompanions(inv ReasonInventory, selected []uint32, pc PC) error {
	counts := make(map[string]uint64, len(pc.Reasons))
	for _, r := range pc.Reasons {
		counts[strings.ToLower(inv.Lookup(r.Index))] += uint64(r.Samples)
	}
	for name, n := range counts {
		if !isCompanion(name) {
			continue
		}
		base := strings.TrimSuffix(name, "_not_issued")
		for _, id := range selected {
			if strings.ToLower(inv.Lookup(id)) == base && n > counts[base] {
				return fmt.Errorf("companion %s count %d exceeds base count %d", name, n, counts[base])
			}
		}
	}
	return nil
}

func add(a, b uint64) (uint64, bool) {
	s, carry := bits.Add64(a, b, 0)
	return s, carry == 0
}

// AppendBinary appends a deterministic little-endian encoding of b to dst.
// Every field is encoded, so two batches encode equally only if they are equal.
// It implements [encoding.BinaryAppender].
func (b Batch) AppendBinary(dst []byte) ([]byte, error) {
	le := binary.LittleEndian
	dst = le.AppendUint32(dst, b.CUPTIVersion)
	dst = le.AppendUint32(dst, uint32(b.CollectionMode))
	dst = appendBool(dst, b.Valid)
	for _, v := range []uint64{b.TotalSamples, b.DroppedSamples, b.NonUserSamples, b.RangeID, b.RemainingPCs} {
		dst = le.AppendUint64(dst, v)
	}
	dst = append(dst, b.HardwareBufferFull)
	if uint64(len(b.PCs)) > math.MaxUint32 {
		return nil, errors.New("cuptipcsampling: too many PCs to encode")
	}
	dst = le.AppendUint32(dst, uint32(len(b.PCs)))
	for _, pc := range b.PCs {
		if uint64(len(pc.Name)) > math.MaxUint32 || uint64(len(pc.Reasons)) > math.MaxUint32 {
			return nil, errors.New("cuptipcsampling: PC record too large to encode")
		}
		dst = le.AppendUint64(dst, pc.CRC)
		dst = le.AppendUint64(dst, pc.Offset)
		dst = le.AppendUint32(dst, pc.FunctionIndex)
		dst = le.AppendUint32(dst, pc.CorrelationID)
		dst = le.AppendUint32(dst, uint32(len(pc.Name)))
		dst = append(dst, pc.Name...)
		dst = le.AppendUint32(dst, uint32(len(pc.Reasons)))
		for _, r := range pc.Reasons {
			dst = le.AppendUint32(dst, r.Index)
			dst = le.AppendUint32(dst, r.Samples)
		}
	}
	return dst, nil
}

func appendBool(dst []byte, v bool) []byte {
	if v {
		return append(dst, 1)
	}
	return append(dst, 0)
}
