package nccltest

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/ncclinspector"
	"github.com/tmc/lib/nvidia/x/nccljoin"
	"github.com/tmc/lib/nvidia/x/ncclras"
)

type collectorArmMode string

const (
	collectorArmOff collectorArmMode = "off"
	collectorArmOn  collectorArmMode = "on"

	collectorNativeWorkloadBytes      = 65536
	collectorLiveCaptureMaxBytes      = 4 << 20
	collectorLiveCaptureMaxTotalBytes = 32 << 20
)

var errPendingCollectorCapture = errors.New("collector cost: capture incomplete")

func isPendingCollectorCaptureError(err error) bool {
	return errors.Is(err, errPendingCollectorCapture)
}

type collectorPluginPin struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
}

type collectorExpectedCapture struct {
	Scope               nccljoin.Scope       `json:"scope"`
	CommunicatorHash    string               `json:"communicator_hash"`
	Operation           string               `json:"operation"`
	Bytes               uint64               `json:"bytes"`
	Ranks               int                  `json:"ranks"`
	Nodes               uint64               `json:"nodes"`
	RankPIDs            [2]int               `json:"rank_pids"`
	RankHosts           [2]string            `json:"rank_hosts"`
	Sequences           []uint64             `json:"sequences"`
	RequireLossComplete bool                 `json:"require_loss_complete"`
	Limits              ncclinspector.Limits `json:"limits"`
}

type collectorArmConfig struct {
	Mode      collectorArmMode   `json:"mode"`
	Case      Case               `json:"case"`
	Options   Options            `json:"options"`
	Cycles    int                `json:"cycles"`
	OutputDir string             `json:"output_dir"`
	DumpDir   string             `json:"dump_dir"`
	RASAddr   string             `json:"ras_addr"`
	Plugin    collectorPluginPin `json:"plugin"`
	Env       map[string]string  `json:"env"`
}

type collectorArmTiming struct {
	SetupDuration time.Duration   `json:"setup_duration"`
	WallDuration  time.Duration   `json:"wall_duration"`
	FlushDuration time.Duration   `json:"flush_duration"`
	GPUTime       time.Duration   `json:"gpu_time"`
	RankTimes     []time.Duration `json:"rank_times,omitempty"`
}

type collectorArmExecution struct {
	Mode              collectorArmMode                     `json:"mode"`
	PID               int                                  `json:"pid"`
	Hostname          string                               `json:"hostname"`
	BootID            string                               `json:"boot_id"`
	ProcessStartTicks uint64                               `json:"process_start_ticks"`
	CollectorEnabled  bool                                 `json:"collector_enabled"`
	LiveCommunicator  bool                                 `json:"live_communicator"`
	CollectorCalls    int                                  `json:"collector_calls"`
	ObservedFiles     int                                  `json:"observed_files"`
	Plugin            collectorPluginPin                   `json:"plugin"`
	OutputDir         string                               `json:"output_dir"`
	DumpDir           string                               `json:"dump_dir"`
	Env               map[string]string                    `json:"env"`
	Result            Result                               `json:"result"`
	CyclesCompleted   int                                  `json:"cycles_completed"`
	Timing            collectorArmTiming                   `json:"timing"`
	CaptureBytes      []byte                               `json:"capture_bytes,omitempty"`
	ReadCapture       func() ([]byte, error)               `json:"-"`
	Stop              func() error                         `json:"-"`
	Sink              func(name string, data []byte) error `json:"-"`
	Owner             io.Closer                            `json:"-"`
}

type collectorCaptureReceipt struct {
	Records          []ncclinspector.Record
	Joined           *nccljoin.Result
	CommunicatorHash string
	Operation        string
	Bytes            uint64
	RankCounts       [2]uint64
	RankAnnounced    [2]uint64
	RankStatsPresent [2]bool
	LossKnown        bool
	LossStatus       string
	Versions         []string
	Mechanisms       []string
	Plugin           collectorPluginPin
	RawBytes         []byte
}

type collectorCostComparisonSpec struct {
	ParentPID         int
	Case              Case
	Options           Options
	Cycles            int
	OffOutputDir      string
	OnOutputDir       string
	RASAddr           string
	Plugin            collectorPluginPin
	Expected          collectorExpectedCapture
	ResolveOnExpected func(onExec collectorArmExecution) (collectorExpectedCapture, error)
}

type collectorCostComparisonReceipt struct {
	Off           collectorArmExecution
	On            collectorArmExecution
	Capture       *collectorCaptureReceipt
	RetainedOwner io.Closer
}

type collectorPipelineHooks struct {
	skipReaderAdmission     bool
	skipPositiveRecordCheck bool
}

// nativeCollectorWorkloadPlan describes the fixed two-GPU AllReduce Float32
// ordinary-launch RunCycles workload used by the native collector-off/on
// comparison fixture. In run.go:417-506, each RunCycles cycle executes:
//   - o.Checks correctness submit(s) per rank (run.go:417-433),
//   - o.Warmup warmup submit(s) per rank (run.go:466-470), and
//   - o.Iterations*aggregation timed submit(s) per rank (run.go:474-480).
//
// Per-rank collective counts are kept distinct from two-rank aggregate counts.
type nativeCollectorWorkloadPlan struct {
	Case                         Case
	Options                      Options
	Cycles                       int
	PerRankCheckSubmitsPerCycle  uint64
	PerRankWarmupSubmitsPerCycle uint64
	PerRankTimedSubmitsPerCycle  uint64
	PerRankCollectivesPerCycle   uint64
	PerRankCollectivesTotal      uint64
	RankTotalCollectives         uint64
	Sequences                    []uint64
}

func defaultNativeCollectorCostWorkload(devices []int32) (nativeCollectorWorkloadPlan, error) {
	return planNativeCollectorCostWorkload(
		Case{Operation: AllReduce, DataType: Float32, Bytes: collectorNativeWorkloadBytes},
		Options{Devices: append([]int32(nil), devices...), Warmup: 1, Iterations: 1, Checks: 1},
		2,
	)
}

func planNativeCollectorCostWorkload(c Case, o Options, cycles int) (nativeCollectorWorkloadPlan, error) {
	if c.DataType == "" {
		c.DataType = Float32
	}
	if c.Operation != AllReduce || c.DataType != Float32 || c.Bytes != collectorNativeWorkloadBytes || c.InPlace || c.Spread != nil || (c.Reduction != "" && c.Reduction != Sum) || c.Root != 0 {
		return nativeCollectorWorkloadPlan{}, errors.New("collector cost: native workload requires out-of-place 64 KiB Float32 AllReduce Sum")
	}
	if cycles != 2 || o.Checks != 1 || o.Warmup != 1 || o.Iterations != 1 {
		return nativeCollectorWorkloadPlan{}, errors.New("collector cost: native workload requires two cycles with one check, warmup, and timed iteration each")
	}
	if len(o.Devices) != 2 || o.Devices[0] < 0 || o.Devices[1] < 0 || o.Devices[0] == o.Devices[1] {
		return nativeCollectorWorkloadPlan{}, errors.New("collector cost: two distinct non-negative device ordinals required")
	}
	if o.GraphLaunches != 0 || o.Aggregation < 0 || o.Aggregation > 1 || o.SyncIterations || o.Register || o.DefaultStream || o.Offset != 0 || o.PeerDelay != 0 || o.DelayRank != 0 {
		return nativeCollectorWorkloadPlan{}, errors.New("collector cost: native workload refuses graph, aggregated, sync-iteration, registered, default-stream, offset, or delayed modes")
	}
	if _, err := c.Layout(2); err != nil {
		return nativeCollectorWorkloadPlan{}, err
	}
	checksPerCycle := uint64(o.Checks)
	warmupPerCycle := uint64(o.Warmup)
	timedPerCycle := uint64(o.Iterations)
	perRankPerCycle := checksPerCycle + warmupPerCycle + timedPerCycle
	perRankTotal := uint64(cycles) * perRankPerCycle
	rankTotal := uint64(len(o.Devices)) * perRankTotal
	sequences := make([]uint64, perRankTotal)
	for i := range sequences {
		sequences[i] = uint64(i)
	}
	optsCopy := o
	optsCopy.Devices = append([]int32(nil), o.Devices...)
	return nativeCollectorWorkloadPlan{
		Case:                         c.copy(),
		Options:                      optsCopy,
		Cycles:                       cycles,
		PerRankCheckSubmitsPerCycle:  checksPerCycle,
		PerRankWarmupSubmitsPerCycle: warmupPerCycle,
		PerRankTimedSubmitsPerCycle:  timedPerCycle,
		PerRankCollectivesPerCycle:   perRankPerCycle,
		PerRankCollectivesTotal:      perRankTotal,
		RankTotalCollectives:         rankTotal,
		Sequences:                    sequences,
	}, nil
}

func (p nativeCollectorWorkloadPlan) cumulativePerRankCollectivesAfterCycle(cycleIndex int) (uint64, error) {
	if cycleIndex < 0 || cycleIndex >= p.Cycles || p.PerRankCollectivesPerCycle == 0 {
		return 0, fmt.Errorf("collector cost: invalid cycle index %d for %d cycles", cycleIndex, p.Cycles)
	}
	return uint64(cycleIndex+1) * p.PerRankCollectivesPerCycle, nil
}

func (p nativeCollectorWorkloadPlan) expectedCapture(scope nccljoin.Scope, commHash string, rankPIDs [2]int, rankHosts [2]string) collectorExpectedCapture {
	return collectorExpectedCapture{
		Scope:               scope,
		CommunicatorHash:    commHash,
		Operation:           "AllReduce",
		Bytes:               uint64(p.Case.Bytes),
		Ranks:               2,
		Nodes:               1,
		RankPIDs:            rankPIDs,
		RankHosts:           rankHosts,
		Sequences:           append([]uint64(nil), p.Sequences...),
		RequireLossComplete: true,
	}
}

func (p nativeCollectorWorkloadPlan) validateRASCycleStatus(
	status *ncclras.Status,
	cycleIndex int,
	wantNCCLVersion int32,
	wantPID int,
	prevCommHash string,
) (string, error) {
	wantOps, err := p.cumulativePerRankCollectivesAfterCycle(cycleIndex)
	if err != nil {
		return "", err
	}
	if status == nil || status.Collection.Timeouts != 0 || len(status.Communicators) != 1 {
		return "", errors.New("collector cost native: incomplete live RAS collection")
	}
	comm := status.Communicators[0]
	if err := ncclinspector.ValidateCommunicatorHash(comm.Hash); err != nil {
		return "", err
	}
	version := fmt.Sprintf("%d.%d.%d", wantNCCLVersion/10000, (wantNCCLVersion%10000)/100, wantNCCLVersion%100)
	if status.Version != version || (prevCommHash != "" && comm.Hash != prevCommHash) {
		return "", errors.New("collector cost native: RAS version or communicator hash changed")
	}
	if comm.Size != 2 || comm.RanksCount != 2 || comm.MissingRanksCount != 0 || len(comm.Ranks) != 2 || len(comm.MissingRanks) != 0 || len(p.Options.Devices) != 2 {
		return "", errors.New("collector cost native: RAS membership differs from two-rank job")
	}
	var seenRank [2]bool
	for _, r := range comm.Ranks {
		if r.Rank < 0 || r.Rank >= len(p.Options.Devices) || seenRank[r.Rank] {
			return "", fmt.Errorf("%w: invalid or duplicate RAS rank %d", ncclinspector.ErrRankMismatch, r.Rank)
		}
		seenRank[r.Rank] = true
		observed, ok := r.Counts["AllReduce"]
		if r.PID != wantPID || r.CUDADevice != int(p.Options.Devices[r.Rank]) || !ok || observed != wantOps {
			return "", fmt.Errorf("collector cost native: RAS rank %d PID/device/AllReduce count mismatch (got pid=%d dev=%d count=%d, want pid=%d dev=%d count=%d)", r.Rank, r.PID, r.CUDADevice, observed, wantPID, p.Options.Devices[r.Rank], wantOps)
		}
	}
	if !seenRank[0] || !seenRank[1] {
		return "", fmt.Errorf("%w: missing expected RAS rank in communicator", ncclinspector.ErrMissingRanks)
	}
	return comm.Hash, nil
}

type collectorOnceCloser struct {
	mu     sync.Mutex
	closer io.Closer
	closed bool
	err    error
}

func (c *collectorOnceCloser) Close() error {
	if c == nil || c.closer == nil {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil
	}
	err := c.closer.Close()
	if err == nil {
		c.closed = true
	}
	c.err = err
	return err
}

func validateCollectorPluginPin(pin collectorPluginPin) error {
	if !filepath.IsAbs(pin.Path) || filepath.Clean(pin.Path) != pin.Path {
		return errors.New("collector cost: absolute clean Inspector plugin path required")
	}
	if len(pin.SHA256) != sha256.Size*2 || strings.ToLower(pin.SHA256) != pin.SHA256 {
		return errors.New("collector cost: canonical 64-hex Inspector plugin SHA-256 required")
	}
	decoded, err := hex.DecodeString(pin.SHA256)
	if err != nil || len(decoded) != sha256.Size {
		return errors.New("collector cost: invalid Inspector plugin SHA-256")
	}
	return nil
}

func validateCollectorExpectedCapture(expected collectorExpectedCapture, requireLiveIdentity bool) error {
	if expected.Scope.Capture == "" || expected.Scope.Job == "" || expected.Scope.Epoch == "" {
		return errors.New("collector cost: explicit capture scope required")
	}
	if expected.Ranks != 2 {
		return errors.New("collector cost: expected rank count must be 2")
	}
	if expected.Nodes < 1 || expected.Nodes > uint64(expected.Ranks) {
		return errors.New("collector cost: invalid expected node count")
	}
	if expected.Operation == "" || expected.Bytes == 0 {
		return errors.New("collector cost: expected collective operation and positive byte size required")
	}
	if len(expected.Sequences) == 0 {
		return errors.New("collector cost: expected collective sequence list required")
	}
	for i, seq := range expected.Sequences {
		if i > 0 && seq <= expected.Sequences[i-1] {
			return errors.New("collector cost: expected collective sequences must be strictly increasing")
		}
	}
	if requireLiveIdentity {
		if err := ncclinspector.ValidateCommunicatorHash(expected.CommunicatorHash); err != nil {
			return err
		}
		for rank := 0; rank < 2; rank++ {
			if expected.RankPIDs[rank] <= 0 || strings.TrimSpace(expected.RankHosts[rank]) == "" {
				return fmt.Errorf("collector cost: rank %d expected process/host identity missing", rank)
			}
		}
	}
	return nil
}

func buildCollectorArmEnv(mode collectorArmMode, rasAddr, dumpDir string, plugin collectorPluginPin) (map[string]string, error) {
	env := make(map[string]string, 12)
	if rasAddr != "" {
		env["NCCL_RAS_ENABLE"] = "1"
		env["NCCL_RAS_ADDR"] = rasAddr
	}
	switch mode {
	case collectorArmOff:
		if dumpDir != "" || plugin != (collectorPluginPin{}) {
			return nil, errors.New("collector cost: off arm must not configure Inspector plugin or dump directory")
		}
		env["NCCL_INSPECTOR_ENABLE"] = "0"
		env["NCCL_PROFILER_PLUGIN"] = ""
		env["NCCL_INSPECTOR_DUMP_DIR"] = ""
		env["NCCL_INSPECTOR_DUMP_THREAD_ENABLE"] = ""
		env["NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS"] = ""
		env["NCCL_INSPECTOR_DUMP_MIN_SIZE_BYTES"] = ""
		env["NCCL_INSPECTOR_PROM_DUMP"] = ""
		env["NCCL_INSPECTOR_OTEL_EXPORT"] = ""
		env["NCCL_INSPECTOR_DUMP_VERBOSE"] = ""
		return env, nil
	case collectorArmOn:
		if err := validateCollectorPluginPin(plugin); err != nil {
			return nil, err
		}
		if !filepath.IsAbs(dumpDir) || filepath.Clean(dumpDir) != dumpDir {
			return nil, errors.New("collector cost: on arm requires clean absolute dump directory")
		}
		env["NCCL_PROFILER_PLUGIN"] = plugin.Path
		env["NCCL_INSPECTOR_ENABLE"] = "1"
		env["NCCL_INSPECTOR_DUMP_DIR"] = dumpDir
		env["NCCL_INSPECTOR_PROM_DUMP"] = "0"
		env["NCCL_INSPECTOR_OTEL_EXPORT"] = "0"
		env["NCCL_INSPECTOR_DUMP_VERBOSE"] = "0"
		env["NCCL_INSPECTOR_DUMP_THREAD_ENABLE"] = "1"
		env["NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS"] = "100000"
		env["NCCL_INSPECTOR_DUMP_MIN_SIZE_BYTES"] = "0"
		return env, nil
	default:
		return nil, fmt.Errorf("collector cost: unknown arm mode %q", mode)
	}
}

func buildCollectorArmConfigs(spec collectorCostComparisonSpec) (collectorArmConfig, collectorArmConfig, error) {
	if spec.ParentPID <= 0 {
		return collectorArmConfig{}, collectorArmConfig{}, errors.New("collector cost: positive parent PID required")
	}
	if spec.Cycles <= 0 {
		return collectorArmConfig{}, collectorArmConfig{}, errors.New("collector cost: positive cycle count required")
	}
	if len(spec.Options.Devices) != 2 || spec.Options.Devices[0] < 0 || spec.Options.Devices[1] < 0 || spec.Options.Devices[0] == spec.Options.Devices[1] {
		return collectorArmConfig{}, collectorArmConfig{}, errors.New("collector cost: two distinct non-negative device ordinals required")
	}
	if spec.Options.Iterations <= 0 || spec.Options.Checks <= 0 || spec.Options.Warmup < 0 {
		return collectorArmConfig{}, collectorArmConfig{}, errors.New("collector cost: invalid workload iteration/check counts")
	}
	if _, err := spec.Case.Layout(2); err != nil {
		return collectorArmConfig{}, collectorArmConfig{}, err
	}
	if err := validateCollectorPluginPin(spec.Plugin); err != nil {
		return collectorArmConfig{}, collectorArmConfig{}, err
	}
	if err := validateCollectorExpectedCapture(spec.Expected, spec.ResolveOnExpected == nil); err != nil {
		return collectorArmConfig{}, collectorArmConfig{}, err
	}
	if !filepath.IsAbs(spec.OffOutputDir) || filepath.Clean(spec.OffOutputDir) != spec.OffOutputDir ||
		!filepath.IsAbs(spec.OnOutputDir) || filepath.Clean(spec.OnOutputDir) != spec.OnOutputDir {
		return collectorArmConfig{}, collectorArmConfig{}, errors.New("collector cost: clean absolute per-arm output directories required")
	}
	if spec.OffOutputDir == spec.OnOutputDir ||
		strings.HasPrefix(spec.OffOutputDir, spec.OnOutputDir+string(filepath.Separator)) ||
		strings.HasPrefix(spec.OnOutputDir, spec.OffOutputDir+string(filepath.Separator)) {
		return collectorArmConfig{}, collectorArmConfig{}, errors.New("collector cost: off and on output directories must be disjoint")
	}
	offEnv, err := buildCollectorArmEnv(collectorArmOff, spec.RASAddr, "", collectorPluginPin{})
	if err != nil {
		return collectorArmConfig{}, collectorArmConfig{}, err
	}
	onDumpDir := filepath.Join(spec.OnOutputDir, "inspector")
	onEnv, err := buildCollectorArmEnv(collectorArmOn, spec.RASAddr, onDumpDir, spec.Plugin)
	if err != nil {
		return collectorArmConfig{}, collectorArmConfig{}, err
	}
	copyOpts := func(o Options) Options {
		o.Devices = append([]int32(nil), o.Devices...)
		return o
	}
	offCfg := collectorArmConfig{
		Mode:      collectorArmOff,
		Case:      spec.Case.copy(),
		Options:   copyOpts(spec.Options),
		Cycles:    spec.Cycles,
		OutputDir: spec.OffOutputDir,
		DumpDir:   "",
		RASAddr:   spec.RASAddr,
		Plugin:    collectorPluginPin{},
		Env:       offEnv,
	}
	onCfg := collectorArmConfig{
		Mode:      collectorArmOn,
		Case:      spec.Case.copy(),
		Options:   copyOpts(spec.Options),
		Cycles:    spec.Cycles,
		OutputDir: spec.OnOutputDir,
		DumpDir:   onDumpDir,
		RASAddr:   spec.RASAddr,
		Plugin:    spec.Plugin,
		Env:       onEnv,
	}
	return offCfg, onCfg, nil
}

func admitCollectorOnCapture(raw []byte, expected collectorExpectedCapture, plugin collectorPluginPin) (*collectorCaptureReceipt, error) {
	return admitCollectorOnCaptureWithHooks(raw, expected, plugin, collectorPipelineHooks{})
}

func admitCollectorOnCaptureWithHooks(raw []byte, expected collectorExpectedCapture, plugin collectorPluginPin, hooks collectorPipelineHooks) (*collectorCaptureReceipt, error) {
	if err := validateCollectorPluginPin(plugin); err != nil {
		return nil, err
	}
	if err := validateCollectorExpectedCapture(expected, true); err != nil {
		return nil, err
	}
	if hooks.skipReaderAdmission {
		if len(raw) == 0 {
			return nil, errors.New("collector cost: empty on-arm inspector capture")
		}
		return &collectorCaptureReceipt{
			CommunicatorHash: expected.CommunicatorHash,
			Operation:        expected.Operation,
			Bytes:            expected.Bytes,
			RankCounts:       [2]uint64{uint64(len(expected.Sequences)), uint64(len(expected.Sequences))},
			RankAnnounced:    [2]uint64{uint64(len(expected.Sequences)), uint64(len(expected.Sequences))},
			RankStatsPresent: [2]bool{true, true},
			LossKnown:        true,
			LossStatus:       "complete_zero_drops",
			Plugin:           plugin,
			RawBytes:         append([]byte(nil), raw...),
		}, nil
	}
	if len(raw) == 0 {
		if hooks.skipPositiveRecordCheck {
			return &collectorCaptureReceipt{
				CommunicatorHash: expected.CommunicatorHash,
				Operation:        expected.Operation,
				Bytes:            expected.Bytes,
				LossKnown:        false,
				LossStatus:       "loss_unknown",
				Plugin:           plugin,
			}, nil
		}
		return nil, errors.New("collector cost: empty on-arm inspector capture")
	}
	records, err := ncclinspector.Read(bytes.NewReader(raw), expected.Limits)
	if err != nil {
		return nil, fmt.Errorf("collector cost inspector read: %w", err)
	}
	expectedSeqSet := make(map[uint64]bool, len(expected.Sequences))
	for _, seq := range expected.Sequences {
		expectedSeqSet[seq] = true
	}
	var counts, announced [2]uint64
	var statsPresent [2]bool
	seenSeq := [2]map[uint64]bool{{}, {}}
	var versions, mechanisms []string
	for _, r := range records {
		if r.PointToPoint != nil {
			return nil, errors.New("collector cost: unexpected point-to-point observation in collective capture")
		}
		if err := ncclinspector.ValidateCommunicatorHash(r.Header.ID); err != nil {
			return nil, err
		}
		if r.Header.ID != expected.CommunicatorHash {
			return nil, fmt.Errorf("collector cost: communicator hash %q != expected %q", r.Header.ID, expected.CommunicatorHash)
		}
		if r.Header.Ranks != expected.Ranks || r.Header.Rank < 0 || r.Header.Rank >= expected.Ranks || r.Header.Nodes != expected.Nodes {
			return nil, fmt.Errorf("%w: rank=%d n_ranks=%d nnodes=%d", ncclinspector.ErrRankMismatch, r.Header.Rank, r.Header.Ranks, r.Header.Nodes)
		}
		rank := r.Header.Rank
		if r.Metadata.PID != expected.RankPIDs[rank] {
			return nil, fmt.Errorf("collector cost: rank %d PID %d != expected %d", rank, r.Metadata.PID, expected.RankPIDs[rank])
		}
		if r.Metadata.Hostname != expected.RankHosts[rank] {
			return nil, fmt.Errorf("collector cost: rank %d hostname %q != expected %q", rank, r.Metadata.Hostname, expected.RankHosts[rank])
		}
		if r.Metadata.Version != "v4.1" || r.Metadata.Mechanism == "" || r.Metadata.TimestampMicros == 0 {
			return nil, errors.New("collector cost: invalid record metadata")
		}
		if !slices.Contains(versions, r.Metadata.Version) {
			versions = append(versions, r.Metadata.Version)
		}
		if !slices.Contains(mechanisms, r.Metadata.Mechanism) {
			mechanisms = append(mechanisms, r.Metadata.Mechanism)
		}
		if r.Stats != nil {
			if r.Stats.PointToPoints != 0 || r.Stats.PointToPointDroppedTotal != 0 || r.Stats.PointToPointDropped != 0 {
				return nil, errors.New("collector cost: unexpected point-to-point dump statistics")
			}
			if r.Stats.CollectiveDroppedTotal != 0 || r.Stats.CollectiveDropped != 0 {
				return nil, fmt.Errorf("collector cost: rank %d dropped collectives (total=%d since_last=%d)", rank, r.Stats.CollectiveDroppedTotal, r.Stats.CollectiveDropped)
			}
			statsPresent[rank] = true
			announced[rank] += r.Stats.Collectives
			if announced[rank] > uint64(len(expected.Sequences)) {
				return nil, fmt.Errorf("collector cost: rank %d dump_stats announced %d collectives, exceeds expected %d", rank, announced[rank], len(expected.Sequences))
			}
		}
		if r.Collective != nil {
			if r.Collective.Name != expected.Operation || r.Collective.Bytes != expected.Bytes {
				return nil, fmt.Errorf("collector cost: rank %d collective %s/%d != expected %s/%d", rank, r.Collective.Name, r.Collective.Bytes, expected.Operation, expected.Bytes)
			}
			if seenSeq[rank][r.Collective.Sequence] {
				return nil, fmt.Errorf("%w: rank %d sequence %d", ncclinspector.ErrDuplicateSequence, rank, r.Collective.Sequence)
			}
			if !expectedSeqSet[r.Collective.Sequence] {
				return nil, fmt.Errorf("collector cost: rank %d unexpected collective sequence %d", rank, r.Collective.Sequence)
			}
			seenSeq[rank][r.Collective.Sequence] = true
			counts[rank]++
			if counts[rank] > uint64(len(expected.Sequences)) {
				return nil, fmt.Errorf("collector cost: rank %d observed %d collectives, exceeds expected %d", rank, counts[rank], len(expected.Sequences))
			}
		}
	}
	if !hooks.skipPositiveRecordCheck {
		for rank := 0; rank < 2; rank++ {
			if counts[rank] < uint64(len(expected.Sequences)) {
				return nil, fmt.Errorf("%w (%w): rank %d observed %d collectives, want %d", ncclinspector.ErrMissingRanks, errPendingCollectorCapture, rank, counts[rank], len(expected.Sequences))
			}
			for _, seq := range expected.Sequences {
				if !seenSeq[rank][seq] {
					return nil, fmt.Errorf("%w (%w): rank %d missing collective sequence %d", ncclinspector.ErrMissingRanks, errPendingCollectorCapture, rank, seq)
				}
			}
		}
	}
	joined, err := nccljoin.Join([]nccljoin.Capture{{
		ID:      "on-arm",
		Records: records,
	}}, nccljoin.Options{
		Scope:      expected.Scope,
		Expected:   map[string][]int{expected.CommunicatorHash: {0, 1}},
		MaxRecords: expected.Limits.Records,
		MaxBytes:   expected.Limits.Bytes,
	})
	if err != nil {
		return nil, fmt.Errorf("collector cost join: %w", err)
	}
	if !hooks.skipPositiveRecordCheck {
		if len(joined.Collectives) != len(expected.Sequences) {
			return nil, fmt.Errorf("%w (%w): joined %d collectives, want %d", ncclinspector.ErrMissingRanks, errPendingCollectorCapture, len(joined.Collectives), len(expected.Sequences))
		}
		for i, g := range joined.Collectives {
			if g.Communicator != expected.CommunicatorHash || g.Sequence != expected.Sequences[i] || g.Name != expected.Operation || g.Bytes != expected.Bytes || len(g.Unobserved) != 0 || !slices.Equal(g.Present, []int{0, 1}) {
				return nil, fmt.Errorf("%w (%w): joined collective %d incomplete or mismatched", ncclinspector.ErrMissingRanks, errPendingCollectorCapture, i)
			}
		}
	}
	for rank := 0; rank < 2; rank++ {
		if statsPresent[rank] && announced[rank] != counts[rank] {
			return nil, fmt.Errorf("collector cost: rank %d dump_stats announced %d collectives, observed %d", rank, announced[rank], counts[rank])
		}
	}
	receipt := &collectorCaptureReceipt{
		Records:          records,
		Joined:           joined,
		CommunicatorHash: expected.CommunicatorHash,
		Operation:        expected.Operation,
		Bytes:            expected.Bytes,
		RankCounts:       counts,
		RankAnnounced:    announced,
		RankStatsPresent: statsPresent,
		Versions:         versions,
		Mechanisms:       mechanisms,
		Plugin:           plugin,
		RawBytes:         append([]byte(nil), raw...),
	}
	if statsPresent[0] && statsPresent[1] {
		receipt.LossKnown = true
		receipt.LossStatus = "complete_zero_drops"
		return receipt, nil
	}
	receipt.LossKnown = false
	receipt.LossStatus = "loss_unknown"
	if statsPresent[0] != statsPresent[1] {
		return receipt, fmt.Errorf("collector cost: asymmetric rank dump_stats; cannot establish complete capture (%w)", errPendingCollectorCapture)
	}
	if expected.RequireLossComplete {
		return receipt, fmt.Errorf("collector cost: collector capture loss statistics unavailable; cannot admit loss-unknown output as complete (%w)", errPendingCollectorCapture)
	}
	return receipt, nil
}

func pollLiveCollectorCaptureReader(
	ctx context.Context,
	read func() ([]byte, error),
	expected collectorExpectedCapture,
	plugin collectorPluginPin,
	interval time.Duration,
) ([]byte, error) {
	if read == nil {
		return nil, errors.New("collector cost: nil live capture reader")
	}
	totalPolledBytes := 0
	for attempt := 0; attempt < 80; attempt++ {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("collector cost live capture incomplete: %w", err)
		}
		b, err := read()
		if err != nil {
			return nil, fmt.Errorf("collector cost live capture read: %w", err)
		}
		totalPolledBytes += len(b)
		if len(b) > collectorLiveCaptureMaxBytes || totalPolledBytes > collectorLiveCaptureMaxTotalBytes {
			return nil, errors.New("collector cost: live capture poll byte budget exceeded")
		}
		end := bytes.LastIndexByte(b, '\n') + 1
		if end > 0 {
			_, admitErr := admitCollectorOnCapture(b[:end], expected, plugin)
			if admitErr == nil && end == len(b) {
				return append([]byte(nil), b...), nil
			}
			if admitErr != nil && !isPendingCollectorCaptureError(admitErr) {
				return nil, admitErr
			}
		}
		timer := time.NewTimer(interval)
		select {
		case <-ctx.Done():
			timer.Stop()
			return nil, fmt.Errorf("collector cost live capture incomplete: %w", ctx.Err())
		case <-timer.C:
		}
	}
	return nil, errors.New("collector cost: live capture observation budget exhausted")
}

func runCollectorCostComparison(
	ctx context.Context,
	spec collectorCostComparisonSpec,
	runArm func(context.Context, collectorArmConfig) (collectorArmExecution, error),
) (*collectorCostComparisonReceipt, error) {
	return runCollectorCostComparisonWithHooks(ctx, spec, runArm, collectorPipelineHooks{})
}

func runCollectorCostComparisonWithHooks(
	ctx context.Context,
	spec collectorCostComparisonSpec,
	runArm func(context.Context, collectorArmConfig) (collectorArmExecution, error),
	hooks collectorPipelineHooks,
) (*collectorCostComparisonReceipt, error) {
	if ctx == nil {
		return nil, errors.New("collector cost: nil context")
	}
	if runArm == nil {
		return nil, errors.New("collector cost: nil arm runner")
	}
	offCfg, onCfg, err := buildCollectorArmConfigs(spec)
	if err != nil {
		return nil, err
	}
	receipt := &collectorCostComparisonReceipt{}

	offExec, offRetained, offErr := executeCollectorArm(ctx, offCfg, runArm)
	receipt.Off = offExec
	if offRetained != nil {
		receipt.RetainedOwner = offRetained
	}
	if offErr != nil {
		return receipt, fmt.Errorf("collector cost off arm: %w", offErr)
	}
	if err := validateOffArmExecution(spec, offCfg, offExec); err != nil {
		return receipt, err
	}

	onExec, onRetained, onErr := executeCollectorArm(ctx, onCfg, runArm)
	receipt.On = onExec
	if onRetained != nil {
		receipt.RetainedOwner = onRetained
	}
	if onErr != nil {
		return receipt, fmt.Errorf("collector cost on arm: %w", onErr)
	}
	if err := validateOnArmExecution(spec, offCfg, onCfg, offExec, onExec, hooks); err != nil {
		return receipt, err
	}

	expected := spec.Expected
	if spec.ResolveOnExpected != nil {
		resolved, err := spec.ResolveOnExpected(onExec)
		if err != nil {
			return receipt, fmt.Errorf("collector cost resolve on expected identity: %w", err)
		}
		if resolved.Scope != spec.Expected.Scope ||
			resolved.Operation != spec.Expected.Operation ||
			resolved.Bytes != spec.Expected.Bytes ||
			resolved.Ranks != spec.Expected.Ranks ||
			resolved.Nodes != spec.Expected.Nodes ||
			resolved.RequireLossComplete != spec.Expected.RequireLossComplete ||
			!slices.Equal(resolved.Sequences, spec.Expected.Sequences) {
			return receipt, errors.New("collector cost: resolved expected identity altered workload or completeness contract")
		}
		if resolved.RankPIDs != [2]int{onExec.PID, onExec.PID} || resolved.RankHosts != [2]string{onExec.Hostname, onExec.Hostname} {
			return receipt, errors.New("collector cost: resolved expected identity must bind to live on-arm child PID and hostname")
		}
		expected = resolved
	}
	captureReceipt, admitErr := admitCollectorOnCaptureWithHooks(onExec.CaptureBytes, expected, spec.Plugin, hooks)
	receipt.Capture = captureReceipt
	if admitErr != nil {
		return receipt, admitErr
	}
	return receipt, nil
}

func executeCollectorArm(
	ctx context.Context,
	cfg collectorArmConfig,
	runArm func(context.Context, collectorArmConfig) (collectorArmExecution, error),
) (collectorArmExecution, io.Closer, error) {
	if err := ctx.Err(); err != nil {
		return collectorArmExecution{}, nil, err
	}
	execState, startErr := runArm(ctx, cfg)
	var onceOwner *collectorOnceCloser
	if execState.Owner != nil {
		onceOwner = &collectorOnceCloser{closer: execState.Owner}
		execState.Owner = onceOwner
	}
	var readErr, stopErr, sinkErr, closeErr error
	if startErr == nil && cfg.Mode == collectorArmOn && execState.ReadCapture != nil {
		if !execState.LiveCommunicator {
			readErr = errors.New("collector cost: on arm attempted capture read without live communicator")
		} else {
			var data []byte
			data, readErr = execState.ReadCapture()
			execState.CollectorCalls++
			if readErr == nil {
				execState.CaptureBytes = append([]byte(nil), data...)
			}
		}
	}
	if execState.Stop != nil {
		stopErr = execState.Stop()
	}
	if startErr == nil && readErr == nil && stopErr == nil && cfg.Mode == collectorArmOn && execState.Sink != nil && len(execState.CaptureBytes) > 0 {
		sinkErr = execState.Sink("inspector-capture.jsonl", execState.CaptureBytes)
	}
	if onceOwner != nil {
		closeErr = onceOwner.Close()
	}
	var retained io.Closer
	if onceOwner != nil && (stopErr != nil || closeErr != nil) {
		retained = onceOwner
	}
	return execState, retained, errors.Join(startErr, readErr, stopErr, sinkErr, closeErr)
}

func validateArmNumericalAndTiming(spec collectorCostComparisonSpec, cfg collectorArmConfig, execState collectorArmExecution) error {
	if execState.Mode != cfg.Mode {
		return fmt.Errorf("collector cost: arm mode %q != %q", execState.Mode, cfg.Mode)
	}
	if execState.PID <= 0 || execState.PID == spec.ParentPID {
		return fmt.Errorf("collector cost: %s arm must run in a fresh child process (pid=%d parent=%d)", cfg.Mode, execState.PID, spec.ParentPID)
	}
	if strings.TrimSpace(execState.Hostname) == "" {
		return fmt.Errorf("collector cost: %s arm hostname missing", cfg.Mode)
	}
	if execState.OutputDir != cfg.OutputDir {
		return fmt.Errorf("collector cost: %s arm output directory %q != %q", cfg.Mode, execState.OutputDir, cfg.OutputDir)
	}
	if execState.CyclesCompleted != spec.Cycles {
		return fmt.Errorf("collector cost: %s arm completed %d cycles, want %d", cfg.Mode, execState.CyclesCompleted, spec.Cycles)
	}
	wantLayout, err := spec.Case.Layout(2)
	if err != nil {
		return err
	}
	if execState.Result.Ranks != 2 || execState.Result.Comparisons <= 0 || execState.Result.Wrong != 0 {
		return fmt.Errorf("collector cost: %s arm numerical verification failed: %+v", cfg.Mode, execState.Result)
	}
	if execState.Result.Case.Operation != spec.Case.Operation || execState.Result.Case.Bytes != spec.Case.Bytes || execState.Result.Layout != wantLayout {
		return fmt.Errorf("collector cost: %s arm workload case/layout mismatch: %+v", cfg.Mode, execState.Result)
	}
	if execState.Timing.SetupDuration < 0 || execState.Timing.WallDuration <= 0 || execState.Timing.FlushDuration < 0 || execState.Timing.GPUTime <= 0 || execState.Result.GPUTime <= 0 || execState.Timing.GPUTime != execState.Result.GPUTime {
		return fmt.Errorf("collector cost: %s arm invalid timing channels: %+v (result GPUTime=%s)", cfg.Mode, execState.Timing, execState.Result.GPUTime)
	}
	return nil
}

func validateOffArmExecution(spec collectorCostComparisonSpec, offCfg collectorArmConfig, offExec collectorArmExecution) error {
	if err := validateArmNumericalAndTiming(spec, offCfg, offExec); err != nil {
		return err
	}
	if offExec.CollectorEnabled || offExec.CollectorCalls != 0 || offExec.ObservedFiles != 0 || len(offExec.CaptureBytes) != 0 || offExec.ReadCapture != nil {
		return errors.New("collector cost: off arm violated collector exclusion (recorded or invoked collector)")
	}
	if offExec.Plugin != (collectorPluginPin{}) || offExec.DumpDir != "" {
		return errors.New("collector cost: off arm retained non-empty plugin pin or dump directory")
	}
	if offExec.Env["NCCL_INSPECTOR_ENABLE"] != "0" || offExec.Env["NCCL_PROFILER_PLUGIN"] != "" || offExec.Env["NCCL_INSPECTOR_DUMP_DIR"] != "" {
		return errors.New("collector cost: off arm environment failed to disable Inspector and clear profiler plugin")
	}
	if spec.RASAddr != "" && (offExec.Env["NCCL_RAS_ENABLE"] != "1" || offExec.Env["NCCL_RAS_ADDR"] != spec.RASAddr) {
		return errors.New("collector cost: off arm RAS configuration mismatch")
	}
	return nil
}

func validateOnArmExecution(
	spec collectorCostComparisonSpec,
	offCfg, onCfg collectorArmConfig,
	offExec, onExec collectorArmExecution,
	hooks collectorPipelineHooks,
) error {
	if err := validateArmNumericalAndTiming(spec, onCfg, onExec); err != nil {
		return err
	}
	if onExec.PID == offExec.PID {
		return fmt.Errorf("collector cost: off and on arms reused child PID %d", onExec.PID)
	}
	if onExec.Hostname != offExec.Hostname {
		return fmt.Errorf("collector cost: arm hostnames differ (%q vs %q)", offExec.Hostname, onExec.Hostname)
	}
	if offExec.BootID != "" && onExec.BootID != "" && offExec.BootID != onExec.BootID {
		return fmt.Errorf("collector cost: arm boot IDs differ (%q vs %q)", offExec.BootID, onExec.BootID)
	}
	if onExec.Result.Comparisons != offExec.Result.Comparisons ||
		onExec.Result.TimedOperations != offExec.Result.TimedOperations ||
		onExec.Result.Iterations != offExec.Result.Iterations ||
		onExec.Result.Warmup != offExec.Result.Warmup ||
		onExec.Result.Checks != offExec.Result.Checks ||
		onExec.Result.Aggregation != offExec.Result.Aggregation {
		return errors.New("collector cost: off and on arms executed mismatched numerical workloads")
	}
	if !onExec.CollectorEnabled || !onExec.LiveCommunicator || onExec.CollectorCalls != 1 {
		return fmt.Errorf("collector cost: on arm did not execute exactly one live collector capture (enabled=%v live=%v calls=%d)", onExec.CollectorEnabled, onExec.LiveCommunicator, onExec.CollectorCalls)
	}
	if !hooks.skipPositiveRecordCheck && onExec.ObservedFiles <= 0 {
		return errors.New("collector cost: on arm observed zero collector dump files")
	}
	if onExec.Plugin != spec.Plugin || onExec.DumpDir != onCfg.DumpDir {
		return errors.New("collector cost: on arm plugin pin or dump directory differs from specification")
	}
	if onExec.Env["NCCL_INSPECTOR_ENABLE"] != "1" ||
		onExec.Env["NCCL_PROFILER_PLUGIN"] != spec.Plugin.Path ||
		onExec.Env["NCCL_INSPECTOR_DUMP_DIR"] != onCfg.DumpDir ||
		onExec.Env["NCCL_INSPECTOR_DUMP_THREAD_ENABLE"] != "1" {
		return errors.New("collector cost: on arm environment did not pin Inspector plugin and dump directory")
	}
	if offCfg.RASAddr != onCfg.RASAddr || onExec.Env["NCCL_RAS_ADDR"] != offExec.Env["NCCL_RAS_ADDR"] || onExec.Env["NCCL_RAS_ENABLE"] != offExec.Env["NCCL_RAS_ENABLE"] {
		return errors.New("collector cost: RAS observer configuration changed between off and on arms")
	}
	return nil
}

type constructedTwoRankManifest struct {
	SchemaVersion    int            `json:"schema_version"`
	Provenance       string         `json:"provenance"`
	AuthenticCapture bool           `json:"authentic_capture"`
	CommunicatorHash string         `json:"communicator_hash"`
	Operation        string         `json:"operation"`
	Sequence         uint64         `json:"sequence"`
	Bytes            uint64         `json:"bytes"`
	Scope            nccljoin.Scope `json:"scope"`
	ExpectedRanks    []int          `json:"expected_ranks"`
	RankFiles        struct {
		Normal  []string `json:"normal"`
		Delayed []string `json:"delayed"`
	} `json:"rank_files"`
	SHA256 map[string]string `json:"sha256"`
}

type constructedTwoRankFixture struct {
	manifest       constructedTwoRankManifest
	rank0Normal    []byte
	rank1Normal    []byte
	lossUnknownRaw []byte
	completeRaw    []byte
	spec           collectorCostComparisonSpec
}

func loadConstructedTwoRankFixture(tb testing.TB) constructedTwoRankFixture {
	tb.Helper()
	baseDir := filepath.Join("..", "ncclinspector", "testdata", "two-rank-stall")
	manifestBytes, err := os.ReadFile(filepath.Join(baseDir, "manifest.json"))
	if err != nil {
		tb.Fatalf("read two-rank-stall manifest: %v", err)
	}
	var manifest constructedTwoRankManifest
	if err := json.Unmarshal(manifestBytes, &manifest); err != nil {
		tb.Fatalf("decode two-rank-stall manifest: %v", err)
	}
	if manifest.AuthenticCapture || manifest.Provenance != "constructed_host_control" || len(manifest.RankFiles.Normal) != 2 {
		tb.Fatalf("unexpected two-rank-stall manifest provenance: %+v", manifest)
	}
	readVerified := func(name string) []byte {
		raw, err := os.ReadFile(filepath.Join(baseDir, name))
		if err != nil {
			tb.Fatalf("read %s: %v", name, err)
		}
		sum := sha256.Sum256(raw)
		if got, want := hex.EncodeToString(sum[:]), manifest.SHA256[name]; got != want {
			tb.Fatalf("%s sha256 = %s, want %s", name, got, want)
		}
		return raw
	}
	r0 := readVerified(manifest.RankFiles.Normal[0])
	r1 := readVerified(manifest.RankFiles.Normal[1])
	var lossUnknown bytes.Buffer
	lossUnknown.Write(r0)
	lossUnknown.Write(r1)

	// Append explicitly constructed zero-drop dump_stats lines for ranks 0 and 1
	// matching the manifest's communicator, PIDs (2100, 2101), and hostnames.
	stats0 := fmt.Sprintf(`{"header":{"id":%q,"name":"constructed-comm","rank":0,"n_ranks":2,"nnodes":2},"metadata":{"inspector_output_format_version":"v4.1","git_rev":"constructed-producer-r1","rec_mechanism":"event_trace","dump_timestamp_us":9100,"hostname":"constructed-node-r0","pid":2100},"dump_stats":{"coll_records":1,"coll_dropped_total":0,"coll_dropped_since_last_dump":0,"p2p_records":0,"p2p_dropped_total":0,"p2p_dropped_since_last_dump":0}}`+"\n", manifest.CommunicatorHash)
	stats1 := fmt.Sprintf(`{"header":{"id":%q,"name":"constructed-comm","rank":1,"n_ranks":2,"nnodes":2},"metadata":{"inspector_output_format_version":"v4.1","git_rev":"constructed-producer-r1","rec_mechanism":"event_trace","dump_timestamp_us":9100,"hostname":"constructed-node-r1","pid":2101},"dump_stats":{"coll_records":1,"coll_dropped_total":0,"coll_dropped_since_last_dump":0,"p2p_records":0,"p2p_dropped_total":0,"p2p_dropped_since_last_dump":0}}`+"\n", manifest.CommunicatorHash)
	var complete bytes.Buffer
	complete.Write(lossUnknown.Bytes())
	complete.WriteString(stats0)
	complete.WriteString(stats1)

	rootDir := tb.TempDir()
	c := Case{Operation: AllReduce, DataType: Float32, Bytes: int(manifest.Bytes)}
	spec := collectorCostComparisonSpec{
		ParentPID:    1000,
		Case:         c,
		Options:      Options{Devices: []int32{0, 1}, Warmup: 1, Iterations: 1, Checks: 1},
		Cycles:       1,
		OffOutputDir: filepath.Join(rootDir, "off"),
		OnOutputDir:  filepath.Join(rootDir, "on"),
		RASAddr:      "127.0.0.1:29500",
		Plugin: collectorPluginPin{
			Path:   "/opt/nccl-inspector/libnccl-profiler-inspector.so",
			SHA256: strings.Repeat("ab", sha256.Size),
		},
		Expected: collectorExpectedCapture{
			Scope:               manifest.Scope,
			CommunicatorHash:    manifest.CommunicatorHash,
			Operation:           manifest.Operation,
			Bytes:               manifest.Bytes,
			Ranks:               2,
			Nodes:               2,
			RankPIDs:            [2]int{2100, 2101},
			RankHosts:           [2]string{"constructed-node-r0", "constructed-node-r1"},
			Sequences:           []uint64{manifest.Sequence},
			RequireLossComplete: true,
		},
	}
	return constructedTwoRankFixture{
		manifest:       manifest,
		rank0Normal:    r0,
		rank1Normal:    r1,
		lossUnknownRaw: lossUnknown.Bytes(),
		completeRaw:    complete.Bytes(),
		spec:           spec,
	}
}

func constructedArmExecution(cfg collectorArmConfig, pid int, onCapture []byte) collectorArmExecution {
	layout, _ := cfg.Case.Layout(2)
	res := Result{
		Case:            cfg.Case.copy(),
		Layout:          layout,
		Ranks:           2,
		NCCLVersion:     22800,
		Iterations:      cfg.Options.Iterations,
		Warmup:          cfg.Options.Warmup,
		Checks:          cfg.Options.Checks,
		Aggregation:     1,
		TimedOperations: cfg.Options.Iterations,
		Comparisons:     layout.RecvCount * 2 * cfg.Options.Checks,
		Wrong:           0,
		GPUTime:         350 * time.Microsecond,
		RankTimes:       []time.Duration{350 * time.Microsecond, 350 * time.Microsecond},
	}
	envCopy := make(map[string]string, len(cfg.Env))
	for k, v := range cfg.Env {
		envCopy[k] = v
	}
	execState := collectorArmExecution{
		Mode:              cfg.Mode,
		PID:               pid,
		Hostname:          "constructed-node-r0",
		BootID:            "constructed-boot-r0",
		ProcessStartTicks: uint64(pid * 10),
		OutputDir:         cfg.OutputDir,
		DumpDir:           cfg.DumpDir,
		Plugin:            cfg.Plugin,
		Env:               envCopy,
		Result:            res,
		CyclesCompleted:   cfg.Cycles,
		Timing: collectorArmTiming{
			SetupDuration: 1200 * time.Microsecond,
			WallDuration:  2500 * time.Microsecond,
			FlushDuration: 400 * time.Microsecond,
			GPUTime:       350 * time.Microsecond,
			RankTimes:     []time.Duration{350 * time.Microsecond, 350 * time.Microsecond},
		},
	}
	if cfg.Mode == collectorArmOn {
		execState.CollectorEnabled = true
		execState.LiveCommunicator = true
		execState.ObservedFiles = 2
		captureCopy := append([]byte(nil), onCapture...)
		execState.ReadCapture = func() ([]byte, error) {
			return captureCopy, nil
		}
	}
	return execState
}

func testCollectorCostComparisonContract(t *testing.T) {
	t.Helper()
	fx := loadConstructedTwoRankFixture(t)

	// 1. Prove the old defective pattern (same-process env toggle with positive
	// GPUTime and zero collected records) is refused.
	_, err := runCollectorCostComparison(t.Context(), fx.spec, func(_ context.Context, cfg collectorArmConfig) (collectorArmExecution, error) {
		execState := constructedArmExecution(cfg, fx.spec.ParentPID, nil)
		execState.ReadCapture = nil
		execState.ObservedFiles = 0
		return execState, nil
	})
	if err == nil {
		t.Fatal("same-process uncollected env-toggle run was unexpectedly admitted")
	}

	// 2. Prove the connected two-rank constructed fixture is admitted and retains
	// separate timing channels and decoded Inspector records without claiming a
	// physical GPU overhead measurement.
	var sunkName string
	var sunkBytes []byte
	receipt, err := runCollectorCostComparison(t.Context(), fx.spec, func(_ context.Context, cfg collectorArmConfig) (collectorArmExecution, error) {
		if cfg.Mode == collectorArmOff {
			return constructedArmExecution(cfg, 2099, nil), nil
		}
		execState := constructedArmExecution(cfg, 2100, fx.completeRaw)
		execState.Sink = func(name string, data []byte) error {
			sunkName = name
			sunkBytes = append([]byte(nil), data...)
			return nil
		}
		return execState, nil
	})
	if err != nil {
		t.Fatalf("connected constructed two-rank collector cost comparison failed: %v", err)
	}
	if receipt.Off.CollectorCalls != 0 || receipt.Off.CollectorEnabled || receipt.On.CollectorCalls != 1 || !receipt.On.CollectorEnabled {
		t.Fatalf("unexpected arm collector call counts: off=%+v on=%+v", receipt.Off, receipt.On)
	}
	if receipt.Capture == nil || !receipt.Capture.LossKnown || receipt.Capture.LossStatus != "complete_zero_drops" {
		t.Fatalf("unexpected capture loss receipt: %+v", receipt.Capture)
	}
	if receipt.Capture.RankCounts != [2]uint64{1, 1} || receipt.Capture.RankAnnounced != [2]uint64{1, 1} || len(receipt.Capture.Records) != 4 {
		t.Fatalf("unexpected capture record counts: %+v", receipt.Capture)
	}
	if sunkName != "inspector-capture.jsonl" || !bytes.Equal(sunkBytes, fx.completeRaw) {
		t.Fatalf("unexpected sunk capture artifact %q (%d bytes)", sunkName, len(sunkBytes))
	}
	verifyNativeWorkloadSixOpAndRASControls(t)
}

func buildConstructedNativeWorkloadCapture(commHash, hostname string, pid, nodes, opsPerRank int, bytesPerOp uint64) []byte {
	var buf bytes.Buffer
	for rank := 0; rank < 2; rank++ {
		for seq := 0; seq < opsPerRank; seq++ {
			fmt.Fprintf(
				&buf,
				`{"header":{"id":%q,"name":"constructed-native-comm","rank":%d,"n_ranks":2,"nnodes":%d},"metadata":{"inspector_output_format_version":"v4.1","git_rev":"constructed-producer-r1","rec_mechanism":"event_trace","dump_timestamp_us":%d,"hostname":%q,"pid":%d},"coll_perf":{"coll":"AllReduce","coll_sn":%d,"coll_msg_size_bytes":%d,"coll_exec_time_us":350,"coll_timing_source":"event_trace","coll_algobw_gbs":0,"coll_busbw_gbs":0}}`+"\n",
				commHash,
				rank,
				nodes,
				9000+seq*10,
				hostname,
				pid,
				seq,
				bytesPerOp,
			)
		}
	}
	for rank := 0; rank < 2; rank++ {
		fmt.Fprintf(
			&buf,
			`{"header":{"id":%q,"name":"constructed-native-comm","rank":%d,"n_ranks":2,"nnodes":%d},"metadata":{"inspector_output_format_version":"v4.1","git_rev":"constructed-producer-r1","rec_mechanism":"event_trace","dump_timestamp_us":9500,"hostname":%q,"pid":%d},"dump_stats":{"coll_records":%d,"coll_dropped_total":0,"coll_dropped_since_last_dump":0,"p2p_records":0,"p2p_dropped_total":0,"p2p_dropped_since_last_dump":0}}`+"\n",
			commHash,
			rank,
			nodes,
			hostname,
			pid,
			opsPerRank,
		)
	}
	return buf.Bytes()
}

func verifyNativeWorkloadSixOpAndRASControls(t *testing.T) {
	t.Helper()
	plan, err := defaultNativeCollectorCostWorkload([]int32{0, 1})
	if err != nil {
		t.Fatalf("defaultNativeCollectorCostWorkload failed: %v", err)
	}
	if plan.PerRankCheckSubmitsPerCycle != 1 ||
		plan.PerRankWarmupSubmitsPerCycle != 1 ||
		plan.PerRankTimedSubmitsPerCycle != 1 ||
		plan.PerRankCollectivesPerCycle != 3 ||
		plan.PerRankCollectivesTotal != 6 ||
		plan.RankTotalCollectives != 12 {
		t.Fatalf("unexpected native workload collective accounting: %+v", plan)
	}
	if !slices.Equal(plan.Sequences, []uint64{0, 1, 2, 3, 4, 5}) {
		t.Fatalf("plan.Sequences = %v, want [0 1 2 3 4 5]", plan.Sequences)
	}
	if got0, err := plan.cumulativePerRankCollectivesAfterCycle(0); err != nil || got0 != 3 {
		t.Fatalf("cycle 0 cumulative collectives = %d, %v; want 3", got0, err)
	}
	if got1, err := plan.cumulativePerRankCollectivesAfterCycle(1); err != nil || got1 != 6 {
		t.Fatalf("cycle 1 cumulative collectives = %d, %v; want 6", got1, err)
	}
	if _, err := plan.cumulativePerRankCollectivesAfterCycle(2); err == nil {
		t.Fatal("out-of-range cycle index accepted")
	}

	// Verify unsupported RunCycles modes are refused explicitly by the native workload planner.
	for _, bad := range []struct {
		name   string
		c      Case
		o      Options
		cycles int
	}{
		{"graphs", plan.Case, Options{Devices: []int32{0, 1}, Warmup: 1, Iterations: 1, Checks: 1, GraphLaunches: 1}, 2},
		{"aggregation", plan.Case, Options{Devices: []int32{0, 1}, Warmup: 1, Iterations: 1, Checks: 1, Aggregation: 2}, 2},
		{"sync_iterations", plan.Case, Options{Devices: []int32{0, 1}, Warmup: 1, Iterations: 1, Checks: 1, SyncIterations: true}, 2},
		{"register", plan.Case, Options{Devices: []int32{0, 1}, Warmup: 1, Iterations: 1, Checks: 1, Register: true}, 2},
		{"default_stream", plan.Case, Options{Devices: []int32{0, 1}, Warmup: 1, Iterations: 1, Checks: 1, DefaultStream: true}, 2},
		{"offset", plan.Case, Options{Devices: []int32{0, 1}, Warmup: 1, Iterations: 1, Checks: 1, Offset: 4}, 2},
		{"peer_delay", plan.Case, Options{Devices: []int32{0, 1}, Warmup: 1, Iterations: 1, Checks: 1, PeerDelay: time.Millisecond}, 2},
		{"in_place", Case{Operation: AllReduce, DataType: Float32, Bytes: collectorNativeWorkloadBytes, InPlace: true}, plan.Options, 2},
		{"wrong_op", Case{Operation: Broadcast, DataType: Float32, Bytes: collectorNativeWorkloadBytes}, plan.Options, 2},
		{"wrong_bytes", Case{Operation: AllReduce, DataType: Float32, Bytes: 4096}, plan.Options, 2},
		{"single_device", plan.Case, Options{Devices: []int32{0}, Warmup: 1, Iterations: 1, Checks: 1}, 2},
	} {
		if _, err := planNativeCollectorCostWorkload(bad.c, bad.o, bad.cycles); err == nil {
			t.Fatalf("planNativeCollectorCostWorkload(%s) unexpectedly succeeded", bad.name)
		}
	}

	// Counts belong to this fixed six-operation fixture. Refuse every other
	// count before the planner performs arithmetic or allocates sequences.
	maxInt := int(^uint(0) >> 1)
	minInt := -maxInt - 1
	for _, bad := range []struct {
		name                          string
		cycles, checks, warmup, timed int
	}{
		{"negative_cycles", -1, 1, 1, 1},
		{"zero_cycles", 0, 1, 1, 1},
		{"one_cycle", 1, 1, 1, 1},
		{"three_cycles", 3, 1, 1, 1},
		{"max_cycles", maxInt, 1, 1, 1},
		{"min_cycles", minInt, 1, 1, 1},
		{"negative_checks", 2, -1, 1, 1},
		{"zero_checks", 2, 0, 1, 1},
		{"two_checks", 2, 2, 1, 1},
		{"max_checks", 2, maxInt, 1, 1},
		{"min_checks", 2, minInt, 1, 1},
		{"negative_warmup", 2, 1, -1, 1},
		{"zero_warmup", 2, 1, 0, 1},
		{"two_warmups", 2, 1, 2, 1},
		{"max_warmup", 2, 1, maxInt, 1},
		{"min_warmup", 2, 1, minInt, 1},
		{"negative_timed", 2, 1, 1, -1},
		{"zero_timed", 2, 1, 1, 0},
		{"two_timed", 2, 1, 1, 2},
		{"max_timed", 2, 1, 1, maxInt},
		{"min_timed", 2, 1, 1, minInt},
		{"all_max", maxInt, maxInt, maxInt, maxInt},
	} {
		opts := plan.Options
		opts.Checks, opts.Warmup, opts.Iterations = bad.checks, bad.warmup, bad.timed
		if _, err := planNativeCollectorCostWorkload(plan.Case, opts, bad.cycles); err == nil {
			t.Fatalf("planNativeCollectorCostWorkload(%s) accepted out-of-contract counts", bad.name)
		}
	}

	// Verify RAS cycle validation requires 3 collectives/rank after cycle 0 and
	// 6 collectives/rank after cycle 1, refuses the old buggy 2/4 counts, and
	// refuses out-of-range or duplicate rank ordinals before indexing Devices[r.Rank].
	const commHash = "0x1234567890abcdef"
	makeRASStatus := func(r0, d0 int, c0 uint64, r1, d1 int, c1 uint64) *ncclras.Status {
		return &ncclras.Status{
			Version: "2.28.0",
			Communicators: []ncclras.Communicator{{
				Hash:       commHash,
				Size:       2,
				RanksCount: 2,
				Ranks: []ncclras.Rank{
					{Rank: r0, PID: 2100, CUDADevice: d0, Counts: map[string]uint64{"AllReduce": c0}},
					{Rank: r1, PID: 2100, CUDADevice: d1, Counts: map[string]uint64{"AllReduce": c1}},
				},
			}},
		}
	}
	if gotHash, err := plan.validateRASCycleStatus(makeRASStatus(0, 0, 3, 1, 1, 3), 0, 22800, 2100, ""); err != nil || gotHash != commHash {
		t.Fatalf("cycle 0 RAS validation (count=3) failed: hash=%q err=%v", gotHash, err)
	}
	if gotHash, err := plan.validateRASCycleStatus(makeRASStatus(0, 0, 6, 1, 1, 6), 1, 22800, 2100, commHash); err != nil || gotHash != commHash {
		t.Fatalf("cycle 1 RAS validation (count=6) failed: hash=%q err=%v", gotHash, err)
	}
	if _, err := plan.validateRASCycleStatus(makeRASStatus(0, 0, 2, 1, 1, 2), 0, 22800, 2100, ""); err == nil {
		t.Fatal("cycle 0 RAS validation accepted stale count=2 instead of 3")
	}
	if _, err := plan.validateRASCycleStatus(makeRASStatus(0, 0, 4, 1, 1, 4), 1, 22800, 2100, commHash); err == nil {
		t.Fatal("cycle 1 RAS validation accepted stale count=4 instead of 6")
	}
	for _, badRank := range []struct {
		name   string
		r0, r1 int
	}{
		{"negative_rank", -1, 1},
		{"rank_ge_devices", 0, 2},
		{"duplicate_rank_zero", 0, 0},
	} {
		_, err := plan.validateRASCycleStatus(makeRASStatus(badRank.r0, 0, 3, badRank.r1, 1, 3), 0, 22800, 2100, "")
		if !errors.Is(err, ncclinspector.ErrRankMismatch) {
			t.Fatalf("validateRASCycleStatus(%s) err = %v; want ErrRankMismatch", badRank.name, err)
		}
	}

	// Also verify ncclras.Read rejects out-of-range and duplicate ranks at the JSON parser boundary.
	for _, rawCase := range []struct {
		name  string
		rank1 int
	}{
		{"out_of_range_rank_2", 2},
		{"duplicate_rank_0", 0},
	} {
		rawJSON := fmt.Sprintf(`{"nccl_version":"2.32.3","cuda_runtime_version":12080,"cuda_driver_version":12080,"timestamp":"2026-10-25T00:00:00Z","communicators_count":1,"communicators":[{"hash":%q,"secondary_hash":"0xfedcba0987654321","size":2,"ranks_count":2,"missing_ranks_count":0,"ranks":[{"rank":0,"host":"constructed-node-r0","pid":2100,"cuda_dev":0,"nvml_dev":0,"status":{"init_state":4,"async_error":0,"finalize_called":false,"destroy_flag":false,"abort_flag":false},"collective_counts":{"Broadcast":0,"Reduce":0,"AllGather":0,"ReduceScatter":0,"AllReduce":3}},{"rank":%d,"host":"constructed-node-r0","pid":2100,"cuda_dev":1,"nvml_dev":1,"status":{"init_state":4,"async_error":0,"finalize_called":false,"destroy_flag":false,"abort_flag":false},"collective_counts":{"Broadcast":0,"Reduce":0,"AllGather":0,"ReduceScatter":0,"AllReduce":3}}],"missing_ranks":[]}],"ras":{"collection_time_sec":0.001,"timeouts_count":0}}`, commHash, rawCase.rank1)
		if _, err := ncclras.Read(strings.NewReader(rawJSON), ncclras.Limits{}); err == nil {
			t.Fatalf("ncclras.Read(%s) unexpectedly accepted invalid rank membership", rawCase.name)
		}
	}

	// Instantiate the actual native workload spec and verify that a constructed
	// 6-collective-per-rank capture passes public ncclinspector.Read + nccljoin.Join
	// while a 4-collective-per-rank capture fails with ErrMissingRanks.
	rootDir := t.TempDir()
	scope := nccljoin.Scope{
		Capture: "constructed-native-workload-6op",
		Job:     "constructed-native-job",
		Epoch:   "on-arm",
	}
	nativeSpec := collectorCostComparisonSpec{
		ParentPID:    1000,
		Case:         plan.Case,
		Options:      plan.Options,
		Cycles:       plan.Cycles,
		OffOutputDir: filepath.Join(rootDir, "off"),
		OnOutputDir:  filepath.Join(rootDir, "on"),
		RASAddr:      "127.0.0.1:29500",
		Plugin: collectorPluginPin{
			Path:   "/opt/nccl-inspector/libnccl-profiler-inspector.so",
			SHA256: strings.Repeat("ab", sha256.Size),
		},
		Expected: plan.expectedCapture(scope, commHash, [2]int{2100, 2100}, [2]string{"constructed-node-r0", "constructed-node-r0"}),
	}
	sixOpRaw := buildConstructedNativeWorkloadCapture(commHash, "constructed-node-r0", 2100, 1, 6, collectorNativeWorkloadBytes)
	sixReceipt, err := runCollectorCostComparison(t.Context(), nativeSpec, func(_ context.Context, cfg collectorArmConfig) (collectorArmExecution, error) {
		if cfg.Mode == collectorArmOff {
			return constructedArmExecution(cfg, 2099, nil), nil
		}
		return constructedArmExecution(cfg, 2100, sixOpRaw), nil
	})
	if err != nil {
		t.Fatalf("native 6-op workload contract failed on matching 6-op capture: %v", err)
	}
	if sixReceipt.Capture.RankCounts != [2]uint64{6, 6} || sixReceipt.Capture.RankAnnounced != [2]uint64{6, 6} || len(sixReceipt.Capture.Joined.Collectives) != 6 || len(sixReceipt.Capture.Records) != 14 {
		t.Fatalf("unexpected 6-op capture receipt: %+v", sixReceipt.Capture)
	}

	fourOpRaw := buildConstructedNativeWorkloadCapture(commHash, "constructed-node-r0", 2100, 1, 4, collectorNativeWorkloadBytes)
	_, err = runCollectorCostComparison(t.Context(), nativeSpec, func(_ context.Context, cfg collectorArmConfig) (collectorArmExecution, error) {
		if cfg.Mode == collectorArmOff {
			return constructedArmExecution(cfg, 2099, nil), nil
		}
		return constructedArmExecution(cfg, 2100, fourOpRaw), nil
	})
	if !errors.Is(err, ncclinspector.ErrMissingRanks) {
		t.Fatalf("native 6-op workload contract on 4-op capture returned %v; want ErrMissingRanks", err)
	}
}

func TestCollectorCostNativeWorkloadSixOpAndRASControls(t *testing.T) {
	verifyNativeWorkloadSixOpAndRASControls(t)
}

func TestCollectorCostConstructedTwoRankPositiveAndLossUnknown(t *testing.T) {
	fx := loadConstructedTwoRankFixture(t)

	// Without dump_stats, the raw two-rank-stall JSONL must remain explicitly
	// loss-unknown and be refused when complete capture is required.
	lossReceipt, err := runCollectorCostComparison(t.Context(), fx.spec, func(_ context.Context, cfg collectorArmConfig) (collectorArmExecution, error) {
		if cfg.Mode == collectorArmOff {
			return constructedArmExecution(cfg, 2099, nil), nil
		}
		return constructedArmExecution(cfg, 2100, fx.lossUnknownRaw), nil
	})
	if err == nil || !strings.Contains(err.Error(), "loss-unknown") {
		t.Fatalf("loss-unknown capture admitted as complete: receipt=%+v err=%v", lossReceipt, err)
	}
	if lossReceipt == nil || lossReceipt.Capture == nil || lossReceipt.Capture.LossKnown || lossReceipt.Capture.LossStatus != "loss_unknown" {
		t.Fatalf("loss-unknown capture did not preserve unavailable loss status: %+v", lossReceipt)
	}
	if lossReceipt.Capture.RankStatsPresent != [2]bool{false, false} || lossReceipt.Capture.RankCounts != [2]uint64{1, 1} {
		t.Fatalf("unexpected loss-unknown rank counts/stats: %+v", lossReceipt.Capture)
	}

	// When RequireLossComplete is false, the partial/loss-unknown disposition is
	// preserved explicitly without inventing zero drops.
	allowLossUnknownSpec := fx.spec
	allowLossUnknownSpec.Expected.RequireLossComplete = false
	partialReceipt, err := runCollectorCostComparison(t.Context(), allowLossUnknownSpec, func(_ context.Context, cfg collectorArmConfig) (collectorArmExecution, error) {
		if cfg.Mode == collectorArmOff {
			return constructedArmExecution(cfg, 2099, nil), nil
		}
		return constructedArmExecution(cfg, 2100, fx.lossUnknownRaw), nil
	})
	if err != nil {
		t.Fatalf("explicitly allowed loss-unknown capture failed: %v", err)
	}
	if partialReceipt.Capture.LossKnown || partialReceipt.Capture.LossStatus != "loss_unknown" {
		t.Fatalf("invented loss completeness on stats-absent capture: %+v", partialReceipt.Capture)
	}

	// With constructed zero-drop dump_stats for both ranks, the comparison is admitted.
	completeReceipt, err := runCollectorCostComparison(t.Context(), fx.spec, func(_ context.Context, cfg collectorArmConfig) (collectorArmExecution, error) {
		if cfg.Mode == collectorArmOff {
			return constructedArmExecution(cfg, 2099, nil), nil
		}
		return constructedArmExecution(cfg, 2100, fx.completeRaw), nil
	})
	if err != nil {
		t.Fatalf("complete constructed two-rank comparison failed: %v", err)
	}
	if !completeReceipt.Capture.LossKnown || completeReceipt.Capture.LossStatus != "complete_zero_drops" {
		t.Fatalf("expected complete_zero_drops status, got %+v", completeReceipt.Capture)
	}
	if completeReceipt.Capture.Joined == nil || len(completeReceipt.Capture.Joined.Collectives) != 1 || !slices.Equal(completeReceipt.Capture.Joined.Collectives[0].Present, []int{0, 1}) {
		t.Fatalf("unexpected joined collectives: %+v", completeReceipt.Capture.Joined)
	}

	// Verify live capture polling defers an unfinished trailing line until the
	// snapshot is complete, and fails immediately on the first poll when a
	// complete line carries a permanent identity defect (e.g. 17-byte truncated hash).
	polls := 0
	polled, err := pollLiveCollectorCaptureReader(t.Context(), func() ([]byte, error) {
		polls++
		if polls == 1 {
			return fx.completeRaw[:len(fx.completeRaw)-12], nil
		}
		return fx.completeRaw, nil
	}, fx.spec.Expected, fx.spec.Plugin, time.Millisecond)
	if err != nil || polls != 2 || !bytes.Equal(polled, fx.completeRaw) {
		t.Fatalf("pollLiveCollectorCaptureReader: polls=%d err=%v", polls, err)
	}

	badPolls := 0
	truncatedHashRaw := bytes.ReplaceAll(fx.completeRaw, []byte(fx.manifest.CommunicatorHash), []byte("0x1234567890abcd"))
	_, err = pollLiveCollectorCaptureReader(t.Context(), func() ([]byte, error) {
		badPolls++
		return truncatedHashRaw, nil
	}, fx.spec.Expected, fx.spec.Plugin, time.Millisecond)
	if !errors.Is(err, ncclinspector.ErrTruncatedHash) || badPolls != 1 {
		t.Fatalf("pollLiveCollectorCaptureReader on truncated hash: badPolls=%d err=%v; want 1 poll and ErrTruncatedHash", badPolls, err)
	}
}

type collectorNegativeControlCase struct {
	name    string
	mutate  func(base collectorCostComparisonSpec, fx constructedTwoRankFixture) (collectorCostComparisonSpec, func(context.Context, collectorArmConfig) (collectorArmExecution, error))
	wantErr error
}

func collectorNegativeControlCases(fx constructedTwoRankFixture) []collectorNegativeControlCase {
	defaultRunner := func(onCapture []byte) func(context.Context, collectorArmConfig) (collectorArmExecution, error) {
		return func(_ context.Context, cfg collectorArmConfig) (collectorArmExecution, error) {
			if cfg.Mode == collectorArmOff {
				return constructedArmExecution(cfg, 2099, nil), nil
			}
			return constructedArmExecution(cfg, 2100, onCapture), nil
		}
	}
	return []collectorNegativeControlCase{
		{
			name: "both_arms_off",
			mutate: func(spec collectorCostComparisonSpec, _ constructedTwoRankFixture) (collectorCostComparisonSpec, func(context.Context, collectorArmConfig) (collectorArmExecution, error)) {
				return spec, func(_ context.Context, cfg collectorArmConfig) (collectorArmExecution, error) {
					execState := constructedArmExecution(cfg, 2099, nil)
					if cfg.Mode == collectorArmOn {
						execState.PID = 2100
						execState.CollectorEnabled = false
						execState.ReadCapture = nil
						execState.ObservedFiles = 0
					}
					return execState, nil
				}
			},
		},
		{
			name: "both_arms_recording",
			mutate: func(spec collectorCostComparisonSpec, fx constructedTwoRankFixture) (collectorCostComparisonSpec, func(context.Context, collectorArmConfig) (collectorArmExecution, error)) {
				return spec, func(_ context.Context, cfg collectorArmConfig) (collectorArmExecution, error) {
					if cfg.Mode == collectorArmOff {
						execState := constructedArmExecution(cfg, 2099, nil)
						execState.CollectorEnabled = true
						execState.CollectorCalls = 1
						execState.ObservedFiles = 1
						execState.CaptureBytes = fx.completeRaw
						return execState, nil
					}
					return constructedArmExecution(cfg, 2100, fx.completeRaw), nil
				}
			},
		},
		{
			name: "empty_on_capture",
			mutate: func(spec collectorCostComparisonSpec, _ constructedTwoRankFixture) (collectorCostComparisonSpec, func(context.Context, collectorArmConfig) (collectorArmExecution, error)) {
				return spec, defaultRunner(nil)
			},
		},
		{
			name: "same_process_pid",
			mutate: func(spec collectorCostComparisonSpec, fx constructedTwoRankFixture) (collectorCostComparisonSpec, func(context.Context, collectorArmConfig) (collectorArmExecution, error)) {
				return spec, func(_ context.Context, cfg collectorArmConfig) (collectorArmExecution, error) {
					if cfg.Mode == collectorArmOff {
						return constructedArmExecution(cfg, 2100, nil), nil
					}
					return constructedArmExecution(cfg, 2100, fx.completeRaw), nil
				}
			},
		},
		{
			name: "shared_or_overlapping_output_dir",
			mutate: func(spec collectorCostComparisonSpec, fx constructedTwoRankFixture) (collectorCostComparisonSpec, func(context.Context, collectorArmConfig) (collectorArmExecution, error)) {
				spec.OnOutputDir = spec.OffOutputDir
				return spec, defaultRunner(fx.completeRaw)
			},
		},
		{
			name: "mismatched_workload_or_ras",
			mutate: func(spec collectorCostComparisonSpec, fx constructedTwoRankFixture) (collectorCostComparisonSpec, func(context.Context, collectorArmConfig) (collectorArmExecution, error)) {
				return spec, func(_ context.Context, cfg collectorArmConfig) (collectorArmExecution, error) {
					if cfg.Mode == collectorArmOff {
						return constructedArmExecution(cfg, 2099, nil), nil
					}
					execState := constructedArmExecution(cfg, 2100, fx.completeRaw)
					execState.Result.Comparisons += 16
					return execState, nil
				}
			},
		},
		{
			name: "stale_or_swapped_arm_capture",
			mutate: func(spec collectorCostComparisonSpec, fx constructedTwoRankFixture) (collectorCostComparisonSpec, func(context.Context, collectorArmConfig) (collectorArmExecution, error)) {
				swapped := bytes.ReplaceAll(fx.completeRaw, []byte(`"pid":2100`), []byte(`"pid":2099`))
				return spec, defaultRunner(swapped)
			},
		},
		{
			name: "wrong_pid",
			mutate: func(spec collectorCostComparisonSpec, fx constructedTwoRankFixture) (collectorCostComparisonSpec, func(context.Context, collectorArmConfig) (collectorArmExecution, error)) {
				bad := bytes.ReplaceAll(fx.completeRaw, []byte(`"pid":2101`), []byte(`"pid":9999`))
				return spec, defaultRunner(bad)
			},
		},
		{
			name: "wrong_hostname",
			mutate: func(spec collectorCostComparisonSpec, fx constructedTwoRankFixture) (collectorCostComparisonSpec, func(context.Context, collectorArmConfig) (collectorArmExecution, error)) {
				bad := bytes.ReplaceAll(fx.completeRaw, []byte(`"hostname":"constructed-node-r1"`), []byte(`"hostname":"foreign-node"`))
				return spec, defaultRunner(bad)
			},
		},
		{
			name: "wrong_communicator_hash",
			mutate: func(spec collectorCostComparisonSpec, fx constructedTwoRankFixture) (collectorCostComparisonSpec, func(context.Context, collectorArmConfig) (collectorArmExecution, error)) {
				bad := bytes.ReplaceAll(fx.completeRaw, []byte(fx.manifest.CommunicatorHash), []byte("0xdeadbeefcafebabe"))
				return spec, defaultRunner(bad)
			},
		},
		{
			name: "truncated_communicator_hash",
			mutate: func(spec collectorCostComparisonSpec, fx constructedTwoRankFixture) (collectorCostComparisonSpec, func(context.Context, collectorArmConfig) (collectorArmExecution, error)) {
				bad := bytes.ReplaceAll(fx.completeRaw, []byte(fx.manifest.CommunicatorHash), []byte("0x1234567890abcd"))
				return spec, defaultRunner(bad)
			},
			wantErr: ncclinspector.ErrTruncatedHash,
		},
		{
			name: "wrong_rank_membership",
			mutate: func(spec collectorCostComparisonSpec, fx constructedTwoRankFixture) (collectorCostComparisonSpec, func(context.Context, collectorArmConfig) (collectorArmExecution, error)) {
				bad := bytes.ReplaceAll(fx.completeRaw, []byte(`"n_ranks":2`), []byte(`"n_ranks":4`))
				return spec, defaultRunner(bad)
			},
			wantErr: ncclinspector.ErrRankMismatch,
		},
		{
			name: "wrong_operation_or_bytes",
			mutate: func(spec collectorCostComparisonSpec, fx constructedTwoRankFixture) (collectorCostComparisonSpec, func(context.Context, collectorArmConfig) (collectorArmExecution, error)) {
				bad := bytes.ReplaceAll(fx.completeRaw, []byte(`"coll":"AllReduce"`), []byte(`"coll":"Broadcast"`))
				return spec, defaultRunner(bad)
			},
		},
		{
			name: "wrong_sequence",
			mutate: func(spec collectorCostComparisonSpec, fx constructedTwoRankFixture) (collectorCostComparisonSpec, func(context.Context, collectorArmConfig) (collectorArmExecution, error)) {
				bad := bytes.ReplaceAll(fx.completeRaw, []byte(`"coll_sn":7`), []byte(`"coll_sn":99`))
				return spec, defaultRunner(bad)
			},
		},
		{
			name: "duplicate_sequence",
			mutate: func(spec collectorCostComparisonSpec, fx constructedTwoRankFixture) (collectorCostComparisonSpec, func(context.Context, collectorArmConfig) (collectorArmExecution, error)) {
				var dup bytes.Buffer
				dup.Write(fx.rank0Normal)
				dup.Write(fx.rank0Normal)
				dup.Write(fx.rank1Normal)
				return spec, defaultRunner(dup.Bytes())
			},
			wantErr: ncclinspector.ErrDuplicateSequence,
		},
		{
			name: "missing_rank",
			mutate: func(spec collectorCostComparisonSpec, fx constructedTwoRankFixture) (collectorCostComparisonSpec, func(context.Context, collectorArmConfig) (collectorArmExecution, error)) {
				return spec, defaultRunner(fx.rank0Normal)
			},
			wantErr: ncclinspector.ErrMissingRanks,
		},
		{
			name: "truncated_jsonl",
			mutate: func(spec collectorCostComparisonSpec, fx constructedTwoRankFixture) (collectorCostComparisonSpec, func(context.Context, collectorArmConfig) (collectorArmExecution, error)) {
				truncated := fx.completeRaw[:len(fx.completeRaw)-10]
				return spec, defaultRunner(truncated)
			},
		},
		{
			name: "dropped_collectives",
			mutate: func(spec collectorCostComparisonSpec, fx constructedTwoRankFixture) (collectorCostComparisonSpec, func(context.Context, collectorArmConfig) (collectorArmExecution, error)) {
				bad := bytes.ReplaceAll(fx.completeRaw, []byte(`"coll_dropped_total":0,"coll_dropped_since_last_dump":0`), []byte(`"coll_dropped_total":1,"coll_dropped_since_last_dump":1`))
				return spec, defaultRunner(bad)
			},
		},
		{
			name: "underannounced_or_overannounced_stats",
			mutate: func(spec collectorCostComparisonSpec, fx constructedTwoRankFixture) (collectorCostComparisonSpec, func(context.Context, collectorArmConfig) (collectorArmExecution, error)) {
				bad := bytes.ReplaceAll(fx.completeRaw, []byte(`"coll_records":1`), []byte(`"coll_records":2`))
				return spec, defaultRunner(bad)
			},
		},
		{
			name: "unexpected_p2p_record_or_stats",
			mutate: func(spec collectorCostComparisonSpec, fx constructedTwoRankFixture) (collectorCostComparisonSpec, func(context.Context, collectorArmConfig) (collectorArmExecution, error)) {
				bad := bytes.ReplaceAll(fx.completeRaw, []byte(`"p2p_records":0`), []byte(`"p2p_records":1`))
				return spec, defaultRunner(bad)
			},
		},
		{
			name: "byte_or_record_budget_exhausted",
			mutate: func(spec collectorCostComparisonSpec, fx constructedTwoRankFixture) (collectorCostComparisonSpec, func(context.Context, collectorArmConfig) (collectorArmExecution, error)) {
				spec.Expected.Limits = ncclinspector.Limits{Bytes: 64, Line: 64, Records: 1}
				return spec, defaultRunner(fx.completeRaw)
			},
		},
		{
			name: "native_six_op_contract_rejects_four_op_capture",
			mutate: func(spec collectorCostComparisonSpec, fx constructedTwoRankFixture) (collectorCostComparisonSpec, func(context.Context, collectorArmConfig) (collectorArmExecution, error)) {
				plan, err := defaultNativeCollectorCostWorkload([]int32{0, 1})
				if err != nil {
					panic(err)
				}
				spec.Case = plan.Case
				spec.Options = plan.Options
				spec.Cycles = plan.Cycles
				spec.Expected = plan.expectedCapture(spec.Expected.Scope, fx.manifest.CommunicatorHash, [2]int{2100, 2100}, [2]string{"constructed-node-r0", "constructed-node-r0"})
				fourOpRaw := buildConstructedNativeWorkloadCapture(fx.manifest.CommunicatorHash, "constructed-node-r0", 2100, 1, 4, collectorNativeWorkloadBytes)
				return spec, defaultRunner(fourOpRaw)
			},
			wantErr: ncclinspector.ErrMissingRanks,
		},
	}
}

func verifyNegativeControlsRefused(ctx context.Context, fx constructedTwoRankFixture, hooks collectorPipelineHooks) error {
	for _, tc := range collectorNegativeControlCases(fx) {
		spec, runner := tc.mutate(fx.spec, fx)
		if _, err := runCollectorCostComparisonWithHooks(ctx, spec, runner, hooks); err == nil {
			return fmt.Errorf("negative control %q unexpectedly admitted", tc.name)
		}
	}
	return nil
}

func TestCollectorCostBehavioralNegatives(t *testing.T) {
	fx := loadConstructedTwoRankFixture(t)
	for _, tc := range collectorNegativeControlCases(fx) {
		t.Run(tc.name, func(t *testing.T) {
			spec, runner := tc.mutate(fx.spec, fx)
			_, err := runCollectorCostComparison(t.Context(), spec, runner)
			if err == nil {
				t.Fatalf("negative control %s unexpectedly succeeded", tc.name)
			}
			if tc.wantErr != nil && !errors.Is(err, tc.wantErr) {
				t.Fatalf("negative control %s error = %v, want errors.Is(_, %v)", tc.name, err, tc.wantErr)
			}
		})
	}
}

type trackingArmOwner struct {
	closes    int
	failFirst error
}

func (o *trackingArmOwner) Close() error {
	o.closes++
	if o.closes == 1 && o.failFirst != nil {
		return o.failFirst
	}
	return nil
}

func TestCollectorCostLifecycleErrorsAndOnceCleanup(t *testing.T) {
	fx := loadConstructedTwoRankFixture(t)
	startErr := errors.New("synthetic arm start failure")
	readErr := errors.New("synthetic capture read failure")
	stopErr := errors.New("synthetic communicator stop failure")
	sinkErr := errors.New("synthetic capture sink failure")
	closeErr := errors.New("synthetic owner close failure")

	t.Run("start_error", func(t *testing.T) {
		owner := &trackingArmOwner{}
		_, err := runCollectorCostComparison(t.Context(), fx.spec, func(_ context.Context, cfg collectorArmConfig) (collectorArmExecution, error) {
			execState := constructedArmExecution(cfg, 2099, nil)
			execState.Owner = owner
			return execState, startErr
		})
		if !errors.Is(err, startErr) || owner.closes != 1 {
			t.Fatalf("err=%v closes=%d; want startErr and exactly 1 close", err, owner.closes)
		}
	})

	t.Run("read_error", func(t *testing.T) {
		offOwner := &trackingArmOwner{}
		onOwner := &trackingArmOwner{}
		_, err := runCollectorCostComparison(t.Context(), fx.spec, func(_ context.Context, cfg collectorArmConfig) (collectorArmExecution, error) {
			if cfg.Mode == collectorArmOff {
				execState := constructedArmExecution(cfg, 2099, nil)
				execState.Owner = offOwner
				return execState, nil
			}
			execState := constructedArmExecution(cfg, 2100, fx.completeRaw)
			execState.Owner = onOwner
			execState.ReadCapture = func() ([]byte, error) { return nil, readErr }
			return execState, nil
		})
		if !errors.Is(err, readErr) || offOwner.closes != 1 || onOwner.closes != 1 {
			t.Fatalf("err=%v offCloses=%d onCloses=%d", err, offOwner.closes, onOwner.closes)
		}
	})

	t.Run("stop_error", func(t *testing.T) {
		onOwner := &trackingArmOwner{}
		receipt, err := runCollectorCostComparison(t.Context(), fx.spec, func(_ context.Context, cfg collectorArmConfig) (collectorArmExecution, error) {
			if cfg.Mode == collectorArmOff {
				return constructedArmExecution(cfg, 2099, nil), nil
			}
			execState := constructedArmExecution(cfg, 2100, fx.completeRaw)
			execState.Owner = onOwner
			execState.Stop = func() error { return stopErr }
			return execState, nil
		})
		if !errors.Is(err, stopErr) || onOwner.closes != 1 || receipt == nil || receipt.RetainedOwner == nil {
			t.Fatalf("err=%v closes=%d retained=%v", err, onOwner.closes, receipt)
		}
		// Retrying RetainedOwner.Close after a successful initial close must be a no-op.
		if retryErr := receipt.RetainedOwner.Close(); retryErr != nil || onOwner.closes != 1 {
			t.Fatalf("retryErr=%v closes=%d; want idempotent once-close", retryErr, onOwner.closes)
		}
	})

	t.Run("sink_error", func(t *testing.T) {
		onOwner := &trackingArmOwner{}
		_, err := runCollectorCostComparison(t.Context(), fx.spec, func(_ context.Context, cfg collectorArmConfig) (collectorArmExecution, error) {
			if cfg.Mode == collectorArmOff {
				return constructedArmExecution(cfg, 2099, nil), nil
			}
			execState := constructedArmExecution(cfg, 2100, fx.completeRaw)
			execState.Owner = onOwner
			execState.Sink = func(string, []byte) error { return sinkErr }
			return execState, nil
		})
		if !errors.Is(err, sinkErr) || onOwner.closes != 1 {
			t.Fatalf("err=%v closes=%d", err, onOwner.closes)
		}
	})

	t.Run("close_error_and_retained_retry", func(t *testing.T) {
		onOwner := &trackingArmOwner{failFirst: closeErr}
		receipt, err := runCollectorCostComparison(t.Context(), fx.spec, func(_ context.Context, cfg collectorArmConfig) (collectorArmExecution, error) {
			if cfg.Mode == collectorArmOff {
				return constructedArmExecution(cfg, 2099, nil), nil
			}
			execState := constructedArmExecution(cfg, 2100, fx.completeRaw)
			execState.Owner = onOwner
			return execState, nil
		})
		if !errors.Is(err, closeErr) || onOwner.closes != 1 || receipt == nil || receipt.RetainedOwner == nil {
			t.Fatalf("err=%v closes=%d retained=%v", err, onOwner.closes, receipt)
		}
		if retryErr := receipt.RetainedOwner.Close(); retryErr != nil || onOwner.closes != 2 {
			t.Fatalf("retryErr=%v closes=%d; want successful retry on 2nd call", retryErr, onOwner.closes)
		}
		if secondRetryErr := receipt.RetainedOwner.Close(); secondRetryErr != nil || onOwner.closes != 2 {
			t.Fatalf("secondRetryErr=%v closes=%d; want no-op after successful retry", secondRetryErr, onOwner.closes)
		}
	})

	t.Run("stop_and_close_joined_errors", func(t *testing.T) {
		onOwner := &trackingArmOwner{failFirst: closeErr}
		receipt, err := runCollectorCostComparison(t.Context(), fx.spec, func(_ context.Context, cfg collectorArmConfig) (collectorArmExecution, error) {
			if cfg.Mode == collectorArmOff {
				return constructedArmExecution(cfg, 2099, nil), nil
			}
			execState := constructedArmExecution(cfg, 2100, fx.completeRaw)
			execState.Owner = onOwner
			execState.Stop = func() error { return stopErr }
			return execState, nil
		})
		if !errors.Is(err, stopErr) || !errors.Is(err, closeErr) || receipt == nil || receipt.RetainedOwner == nil {
			t.Fatalf("joined error did not preserve both stopErr and closeErr: %v", err)
		}
	})
}

func TestCollectorCostCompilingOmissionControls(t *testing.T) {
	fx := loadConstructedTwoRankFixture(t)
	if err := verifyNegativeControlsRefused(t.Context(), fx, collectorPipelineHooks{}); err != nil {
		t.Fatalf("baseline negative-control suite failed: %v", err)
	}
	if err := verifyNegativeControlsRefused(t.Context(), fx, collectorPipelineHooks{skipReaderAdmission: true}); err == nil {
		t.Fatal("omitting ncclinspector.Read admission did not cause negative-control suite to fail")
	}
	if err := verifyNegativeControlsRefused(t.Context(), fx, collectorPipelineHooks{skipPositiveRecordCheck: true}); err == nil {
		t.Fatal("omitting positive collective record check did not cause negative-control suite to fail")
	}
}
