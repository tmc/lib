package cuptipcsampling

import (
	"errors"
	"fmt"
	"runtime"
	"slices"

	"github.com/tmc/lib/nvidia/cupti"
)

// Config specifies PC sampling policy. There are no implicit memory limits:
// every limit and byte size must be positive and is enforced before any native
// call. The zero Config is not usable.
type Config struct {
	// Period selects the native sampling period, which is 2^Period GPU cycles.
	// Valid values are 5 through 31, that is 32 through 2^31 cycles. Smaller
	// periods sample more often and make hardware buffer overflow more likely.
	Period uint32

	// WorkerSleepMS is the sleep span, in milliseconds, between wakeups of the
	// CUPTI worker threads that decode hardware data and correlate PCs to SASS.
	// It must be nonzero; the native default is 100.
	WorkerSleepMS uint32

	// CollectionMode selects continuous or serialized collection. Zero selects
	// continuous. Serialized collection also requires AllowSerialization.
	CollectionMode cupti.PCSamplingCollectionMode

	// AllowSerialization authorizes kernel-serialized collection. Serializing
	// kernels changes concurrency, so it changes the workload being measured.
	AllowSerialization bool

	// ReasonIDs selects native reason indexes, as returned by StallReasons for
	// the same context and version. Duplicates are rejected, every index must
	// appear in the inventory, and the count must not exceed MaxReasons.
	// Start clones the slice; later caller edits do not affect the session.
	ReasonIDs []uint32

	// MaxReasons bounds the reason inventory query, from 1 through 65536.
	MaxReasons int

	// ScratchBytes is the software buffer holding raw counter data downloaded
	// from the hardware buffer. HardwareBytes is the hardware buffer itself; a
	// small buffer with a short Period drops samples. MaxNativeBytes bounds the
	// two together: ScratchBytes+HardwareBytes must not exceed it. These are
	// native allocations made by CUPTI, separate from the output buffers below.
	ScratchBytes, HardwareBytes, MaxNativeBytes uint64

	// PCsPerBuffer is the PC record capacity of one output buffer. Start owns
	// two such buffers, the configured output and a separate read drain, so a
	// single Read never returns more than PCsPerBuffer records.
	PCsPerBuffer int

	// MaxOutputBytes bounds both output buffers together, each receiving
	// MaxOutputBytes/2; an odd value loses the remaining byte to truncation.
	// The budget covers the output header, PC records and reason arrays, and
	// excludes Go bookkeeping and the native buffers sized above.
	MaxOutputBytes uint64

	// MaxReads bounds successful Read calls for the session's whole lifetime.
	// Exactly MaxReads reads are admitted; the next one fails the session
	// rather than silently dropping records.
	MaxReads uint

	// MaxNames bounds distinct native name pointers retained for the session,
	// MaxNameBytes bounds one name including its terminating NUL, and
	// MaxTotalNameBytes bounds their sum.
	//
	// Names accumulate for the whole session: every distinct pointer stays
	// registered until Close, because native names may be shared with Activity
	// records and are freed only once. Repeated names cost nothing, so size
	// MaxNames for the distinct functions the workload is expected to sample,
	// not for one read. Start enforces only the per-drain worst case, that is
	// MaxNames >= 2*PCsPerBuffer, since each drain reserves both buffers. A
	// session that then exhausts MaxNames fails its next Read visibly; it
	// never drops a name silently.
	MaxNames, MaxNameBytes, MaxTotalNameBytes int
}

// StallSamples is a native count for one queried reason, not a legacy stall enum.
type StallSamples struct{ Index, Samples uint32 }

// PC retains one sampled code coordinate. CRC alone does not identify a loaded
// module generation. CorrelationID is kernel-specific only in serialized mode.
type PC struct {
	CRC, Offset                  uint64
	FunctionIndex, CorrelationID uint32
	Name                         string
	Reasons                      []StallSamples
}

// Batch owns one native output snapshot. Valid means the output passed decoding,
// not that collection was complete or loss-free. Totals include dropped,
// non-user and unselected-reason samples; they need not equal the reason sums.
// Empty data does not establish workload completion. Callers must retain run,
// process, device and module identities separately when transporting samples.
type Batch struct {
	CUPTIVersion                                          uint32
	CollectionMode                                        cupti.PCSamplingCollectionMode
	Valid                                                 bool
	TotalSamples, DroppedSamples, NonUserSamples, RangeID uint64
	RemainingPCs                                          uint64
	HardwareBufferFull                                    uint8
	PCs                                                   []PC
}

// Info owns session configuration and quality metadata. Effective attributes
// remain diagnostic unless Verified is true. Closed indicates storage release,
// not collection success. Failed is sticky across successful cleanup retries.
type Info struct {
	CUPTIVersion             uint32
	Context                  cupti.Context
	Config                   Config
	Inventory                ReasonInventory
	Effective                []cupti.PCSamplingConfigurationInfo
	Verified, Failed, Closed bool
	Reads                    uint
	Starts, Stops            uint64
}

// Session owns an exclusively coordinated PC collector. Its zero value is not
// usable. Serialize method calls and keep the CUDA context alive through Close.
type Session struct {
	core      *samplingSession
	config    Config
	inventory ReasonInventory
}

// Start configures, verifies, enables and starts an explicit sampling range.
// It launches no workload and does not synchronize or replay CUDA work. Only
// Linux amd64/arm64 with CUPTI 130001 is admitted; this is a header-layout gate,
// not a claim of native hardware qualification. Period must be 5 through 31.
// A zero CollectionMode selects continuous collection. Serialized collection
// requires AllowSerialization because it changes kernel concurrency.
//
// The caller must own exclusive PC sampling and coordinate other profilers and
// shared native names. The package rejects its own concurrent sessions but
// cannot detect every external registrant. A nonnil Session returned with an
// error retains cleanup ownership: call Close, retrying explicitly if native
// teardown fails. Do not destroy the context or discard that Session first.
func Start(ctx cupti.Context, config Config) (*Session, error) {
	if err := validateConfig(ctx, config); err != nil {
		return nil, err
	}
	if runtime.GOOS != "linux" || (runtime.GOARCH != "amd64" && runtime.GOARCH != "arm64") {
		return nil, errors.New("cuptipcsampling: native sampling requires Linux amd64 or arm64")
	}
	return startSession(ctx, config, nativeSamplingControlAPI(), StallReasons, newNativeSamplingNames, cupti.PCSamplingGetData)
}

func validateConfig(ctx cupti.Context, c Config) error {
	if ctx == 0 || c.Period < 5 || c.Period > 31 || c.WorkerSleepMS == 0 || len(c.ReasonIDs) == 0 || c.MaxReasons < 1 || c.MaxReasons > 65536 || len(c.ReasonIDs) > c.MaxReasons || c.PCsPerBuffer < 1 || c.MaxReads == 0 || c.MaxNames < 1 || c.MaxNameBytes < 1 || c.MaxTotalNameBytes < 1 {
		return errors.New("cuptipcsampling: invalid sampling configuration or limits")
	}
	if c.CollectionMode != 0 && c.CollectionMode != cupti.PcSamplingCollectionModeContinuous && (c.CollectionMode != cupti.PcSamplingCollectionModeKernelSerialized || !c.AllowSerialization) {
		return errors.New("cuptipcsampling: invalid or unauthorized sampling mode")
	}
	if c.ScratchBytes == 0 || c.HardwareBytes == 0 || c.ScratchBytes > c.MaxNativeBytes || c.HardwareBytes > c.MaxNativeBytes-c.ScratchBytes {
		return errors.New("cuptipcsampling: native sampling memory exceeds budget")
	}
	stride := uint64(64) + uint64(len(c.ReasonIDs))*8
	half := c.MaxOutputBytes / 2
	if half < 80 || uint64(c.PCsPerBuffer) > (half-80)/stride || c.PCsPerBuffer > c.MaxNames/2 {
		return errors.New("cuptipcsampling: output or name capacity budget too small")
	}
	seen := make(map[uint32]bool, len(c.ReasonIDs))
	for _, id := range c.ReasonIDs {
		if seen[id] {
			return errors.New("cuptipcsampling: duplicate selected reason")
		}
		seen[id] = true
	}
	return nil
}

func startSession(ctx cupti.Context, c Config, api samplingControlAPI, inventoryQuery func(cupti.Context, int) (ReasonInventory, error), nameOwner func(int, int, int) (*samplingNames, error), query func(*cupti.PCSamplingGetDataParams) error) (*Session, error) {
	if err := validateConfig(ctx, c); err != nil {
		return nil, err
	}
	version, err := api.version()
	if err != nil {
		return nil, err
	}
	if version != 130001 {
		return nil, fmt.Errorf("cuptipcsampling: unqualified sampling version %d", version)
	}
	inventory, err := inventoryQuery(ctx, c.MaxReasons)
	if err != nil {
		return nil, err
	}
	if inventory.CUPTIVersion != version {
		return nil, errors.New("cuptipcsampling: reason inventory version changed")
	}
	if c.CollectionMode == 0 {
		c.CollectionMode = cupti.PcSamplingCollectionModeContinuous
	}
	c.ReasonIDs = slices.Clone(c.ReasonIDs)
	out, err := newSamplingBuffer(c.PCsPerBuffer, len(c.ReasonIDs), c.MaxOutputBytes/2)
	if err != nil {
		return nil, err
	}
	drain, err := newSamplingBuffer(c.PCsPerBuffer, len(c.ReasonIDs), c.MaxOutputBytes/2)
	if err != nil {
		out.release()
		return nil, err
	}
	names, err := nameOwner(c.MaxNames, c.MaxNameBytes, c.MaxTotalNameBytes)
	if err != nil {
		out.release()
		drain.release()
		return nil, err
	}
	o := samplingOptions{period: c.Period, workerSleepMS: c.WorkerSleepMS, mode: c.CollectionMode, allowSerialization: c.AllowSerialization, scratchBytes: c.ScratchBytes, hardwareBytes: c.HardwareBytes, maxNativeBytes: c.MaxNativeBytes, reasons: c.ReasonIDs}
	cfg, err := newSamplingConfig(o, inventory, out)
	if err != nil {
		names.release()
		out.release()
		drain.release()
		return nil, err
	}
	core := &samplingSession{control: &samplingControl{ctx: ctx, config: cfg, api: api}, drain: drain, names: names, query: query, maxReads: c.MaxReads}
	s := &Session{core: core, config: c, inventory: inventory}
	if err := core.begin(); err != nil {
		if !core.control.touched {
			_, _ = core.finish(true)
			return nil, err
		}
		return s, err
	}
	return s, nil
}

// Read drains one bounded batch. It never synchronizes the device. An error
// may accompany diagnostic counters, with Valid false. Keep all earlier
// batches; Final is a separate native buffer, not a replacement for them.
func (s *Session) Read() (Batch, error) {
	if s == nil || s.core == nil {
		return Batch{}, errors.New("cuptipcsampling: invalid session")
	}
	b, err := s.core.read()
	return s.batch(b), err
}

// Stop ends the sampling range. It does not wait for workload completion.
// Drain remaining records with Read before Close at the coordinated boundary.
func (s *Session) Stop() error {
	if s == nil || s.core == nil {
		return errors.New("cuptipcsampling: invalid session")
	}
	return s.core.control.stop()
}

// Close disables sampling and copies final output before freeing owned storage.
// The caller must first reach its workload boundary and finish all consumers
// sharing native names. A failed disable retains ownership for explicit retry.
// Prior failures/loss remain errors even after successful cleanup. Close does
// not destroy the CUDA context. Final remains available after storage release.
func (s *Session) Close() error {
	if s == nil || s.core == nil {
		return errors.New("cuptipcsampling: invalid session")
	}
	_, err := s.core.finish(true)
	return err
}

// Final returns an owned final-buffer snapshot after successful native disable
// and local finalization. The boolean means availability, not capture success;
// inspect Valid, loss fields, Info and the Close error separately.
func (s *Session) Final() (Batch, bool) {
	if s == nil || s.core == nil || !s.core.closed || !s.core.control.disabled {
		return Batch{}, false
	}
	return s.batch(s.core.final), true
}

// Info returns owned configuration and diagnostic metadata.
func (s *Session) Info() Info {
	if s == nil || s.core == nil {
		return Info{}
	}
	c := s.core.control
	config := s.config
	config.ReasonIDs = slices.Clone(config.ReasonIDs)
	inventory := s.inventory
	inventory.Reasons = slices.Clone(inventory.Reasons)
	return Info{CUPTIVersion: inventory.CUPTIVersion, Context: c.ctx, Config: config, Inventory: inventory, Effective: slices.Clone(c.config.effective), Verified: c.config.verified, Failed: c.failed, Closed: s.core.closed, Reads: s.core.reads, Starts: c.starts, Stops: c.stops}
}

func (s *Session) batch(b samplingBatch) Batch {
	out := Batch{CUPTIVersion: s.inventory.CUPTIVersion, CollectionMode: s.config.CollectionMode, Valid: b.valid, TotalSamples: b.total, DroppedSamples: b.dropped, NonUserSamples: b.nonUser, RangeID: b.rangeID, RemainingPCs: uint64(b.remaining), HardwareBufferFull: b.hardwareFull}
	for _, p := range b.pcs {
		pc := PC{CRC: p.crc, Offset: p.offset, FunctionIndex: p.functionIndex, CorrelationID: p.correlation, Name: p.name}
		for _, r := range p.reasons {
			pc.Reasons = append(pc.Reasons, StallSamples{Index: r.PcSamplingStallReasonIndex, Samples: r.Samples})
		}
		out.PCs = append(out.PCs, pc)
	}
	return out
}
