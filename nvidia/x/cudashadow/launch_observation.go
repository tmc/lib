package cudashadow

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/tmc/lib/nvidia/x/cudabpfdev"
	"github.com/tmc/lib/nvidia/x/cudakernel"
)

// InvocationObservationReader supplies host receipts. Implementations other than
// cudabpfdev.InvocationReader supply caller assertions, not independent attestation.
type InvocationObservationReader interface {
	ReadObservation() (cudabpfdev.InvocationObservation, error)
}

// LaunchObservationOptions identifies the adapter epoch, receipt order and fixed
// PTX policy configuration. Source must change when the adapter epoch changes.
// Sequence is host receipt order; TTL uses host Unix nanoseconds.
type LaunchObservationOptions struct {
	Source   string
	Sequence uint64
	TTL      time.Duration
	Policy   PolicyInfo
}

// PendingLaunchObservation retains the same read-only owner view used at begin.
// Its zero value is unavailable. It does not execute or request baseline work.
// Complete may be retried after a read failure, but records at most one outcome.
type PendingLaunchObservation struct {
	mu         sync.Mutex
	evaluator  *Evaluator
	reader     InvocationObservationReader
	original   cudabpfdev.InvocationObservation
	snapshotID string
	complete   bool
}

// BeginLaunchObservation reads an existing PTX descriptor and fresh original
// fallback owner, then evaluates the built-in PTX shadow policy. It never loads,
// launches or closes an owner. Baseline execution remains with the caller.
// The returned snapshot ID identifies observation replay, not kernel replay.
func BeginLaunchObservation(ctx context.Context, evaluator *Evaluator, source cudabpfdev.LaunchSource, reader InvocationObservationReader, options LaunchObservationOptions) (*PendingLaunchObservation, error) {
	return beginLaunchObservation(ctx, evaluator, source, reader, options, func() int64 { return time.Now().UnixNano() })
}

func beginLaunchObservation(ctx context.Context, evaluator *Evaluator, source cudabpfdev.LaunchSource, reader InvocationObservationReader, options LaunchObservationOptions, clock func() int64) (*PendingLaunchObservation, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if evaluator == nil || source == nil || reader == nil {
		return nil, errors.New("cudashadow: unavailable observation source or evaluator")
	}
	launch, err := source.DescribeLaunch()
	if err != nil {
		return nil, fmt.Errorf("cudashadow: describe launch: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	a := launch.Artifact
	if a == nil {
		return nil, errors.New("cudashadow: missing launch artifact")
	}
	if err := a.Validate(); err != nil {
		return nil, fmt.Errorf("cudashadow: launch artifact: %w", err)
	}
	if a.Kind() != cudakernel.KindPTX {
		return nil, errors.New("cudashadow: observation requires PTX")
	}
	raw := a.Bytes()
	if len(raw) == 0 || len(raw) > 16<<20 || strings.ContainsRune(a.EntryPoint(), 0) {
		return nil, errors.New("cudashadow: invalid PTX description bounds")
	}
	schema := a.Schema()
	if err := validateLaunchDescriptor(launch.Geometry, schema); err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	observed, err := reader.ReadObservation()
	if err != nil {
		return nil, fmt.Errorf("cudashadow: read invocation: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if !observed.Ready || observed.Used || observed.Pending || observed.Closed || observed.Complete || observed.TimestampNano != 0 {
		return nil, errors.New("cudashadow: observation requires fresh original owner")
	}
	identity := cudabpfdev.KernelIdentity{Digest: a.Digest(), EntryPoint: a.EntryPoint(), Target: a.Target()}
	if observed.Selection.Candidate || observed.Original != identity || observed.Selection.Kernel != identity || observed.Geometry != launch.Geometry || !sameSchema(observed.Schema, schema) || len(observed.Selection.Reason) > 4096 {
		return nil, errors.New("cudashadow: launch descriptor does not match original owner")
	}
	if !observed.SameInvocation(observed) {
		return nil, errors.New("cudashadow: unavailable owner linkage")
	}
	in := Input{Provenance: Provenance{Kind: "artifact_description", Source: options.Source, DeviceID: -1, Hook: HookDescribeLaunch, Sequence: options.Sequence, TimestampNano: clock(), TTL: options.TTL}, Domain: DomainPTXInstrument,
		PTX: &PTXData{KernelName: a.EntryPoint(), TargetArch: a.Target(), FunctionOffsetUnknown: true, CodeSize: uint64(len(raw)), ArtifactDigest: a.Digest(), Geometry: &launch.Geometry, Schema: &schema}}
	if err := in.Validate(); err != nil {
		return nil, err
	}
	policy, err := NewPTXShadowPolicy(options.Policy)
	if err != nil {
		return nil, err
	}
	snapshot, err := evaluator.Evaluate(ctx, policy, in)
	if err != nil {
		return nil, err
	}
	if snapshot.Status != StatusValid {
		return nil, fmt.Errorf("cudashadow: observation input %s", snapshot.Status)
	}
	observed.Schema.Params = slices.Clone(observed.Schema.Params)
	return &PendingLaunchObservation{evaluator: evaluator, reader: reader, original: observed, snapshotID: snapshot.ID}, nil
}

// SnapshotID returns the evaluated input's ID, or empty for an unavailable token.
func (p *PendingLaunchObservation) SnapshotID() string {
	if p == nil {
		return ""
	}
	return p.snapshotID
}

// Complete records producer-reported completion from the retained owner view.
// Its timestamp is host completion publication, and latency is unavailable.
// Missing, failed, pending or changed-owner receipts leave the outcome absent.
func (p *PendingLaunchObservation) Complete(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if p == nil {
		return errors.New("cudashadow: unavailable pending observation")
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.evaluator == nil || p.reader == nil {
		return errors.New("cudashadow: unavailable pending observation")
	}
	if p.complete {
		return errors.New("cudashadow: observation already completed")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	observed, err := p.reader.ReadObservation()
	if err != nil {
		return fmt.Errorf("cudashadow: read completion: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	before := p.original
	if !before.SameInvocation(observed) || observed.Original != before.Original || observed.Selection != before.Selection || observed.Selection.Candidate || observed.Geometry != before.Geometry || !sameSchema(observed.Schema, before.Schema) {
		return errors.New("cudashadow: completion owner or descriptor changed")
	}
	if !observed.Ready || !observed.Used || observed.Pending || !observed.Complete || observed.TimestampNano <= 0 {
		return errors.New("cudashadow: original completion unavailable")
	}
	outcome := ObservedOutcome{TimestampNano: observed.TimestampNano, Action: "original_invocation_completion_published", LatencyUnavailable: true,
		Details: map[string]string{"source": "producer_reported_same_owner", "timestamp_source": "host_completion_publication", "candidate": strconv.FormatBool(observed.Selection.Candidate), "fallback_reason": observed.Selection.Reason, "image_digest": observed.Original.Digest, "entry_point": observed.Original.EntryPoint, "target": observed.Original.Target, "closed": strconv.FormatBool(observed.Closed)}}
	if err := p.evaluator.RecordOutcome(p.snapshotID, outcome); err != nil {
		return err
	}
	p.complete = true
	return nil
}

func sameSchema(a, b cudakernel.Schema) bool {
	return a.TotalParamBytes == b.TotalParamBytes && slices.Equal(a.Params, b.Params)
}

func validateLaunchDescriptor(g cudabpfdev.Geometry, s cudakernel.Schema) error {
	for i := range g.Grid {
		if g.Grid[i] == 0 || g.Block[i] == 0 {
			return errors.New("cudashadow: zero launch dimension")
		}
	}
	if len(s.Params) > 128 || s.TotalParamBytes > 1<<20 {
		return errors.New("cudashadow: parameter schema exceeds bound")
	}
	for _, p := range s.Params {
		if len(p.Name) > 256 || len(p.Type) > 64 {
			return errors.New("cudashadow: parameter metadata exceeds bound")
		}
	}
	if err := s.Validate(); err != nil {
		return fmt.Errorf("cudashadow: parameter schema: %w", err)
	}
	return nil
}
