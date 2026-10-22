//go:build linux

package main

import (
	"bytes"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"io"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

func cudaSupportShape(f *cudaDSparkFeatures) (*dsparkSupportEnvelope, error) {
	return cudaSupportIdentity(f.modelSHA, f.r.ctx, f.block, f.targets)
}

func cudaSupportIdentity(modelSHA string, context, block int, targets []int) (*dsparkSupportEnvelope, error) {
	identity, err := hex.DecodeString(modelSHA)
	if err != nil || len(identity) != 32 {
		return nil, errors.New("cuda support codec: invalid model identity")
	}
	e := &dsparkSupportEnvelope{context: context, block: block, targets: targets, capture: &dsparkSupportCapture{}}
	copy(e.model[:], identity)
	return e, nil
}

func (s *cudaRequestState) saveSupport(w io.Writer) (err error) {
	return saveCUDAState(s, w)
}

func writeCUDASupportSnapshot(w io.Writer, v *cudaStateSnapshot, model, quant [32]byte) error {
	f := v.dspark
	e, err := cudaSupportIdentity(f.modelSHA, f.context, f.block, f.targets)
	if err != nil {
		return err
	}
	// The snapshot owns these allocations independently of the destination.
	e.context, e.block, e.targets, e.journal = f.context, f.block, f.targets, f.journal
	e.history = make([]float32, f.history.Len())
	e.capture = &dsparkSupportCapture{generation: f.generation, position: f.position, token: f.token, published: f.published, captured: f.captured, features: make([]float32, f.features.Len())}
	if err := f.history.CopyToHost(e.history); err != nil {
		return err
	}
	if err := f.features.CopyToHost(e.capture.features); err != nil {
		return err
	}
	var status [1]uint32
	if err := f.status.CopyToHost(status[:]); err != nil {
		return err
	}
	e.capture.status = status[0]
	var core bytes.Buffer
	if err := writeCUDAStateCoreSnapshot(&core, v, model, quant); err != nil {
		return err
	}
	return writeDSparkSupportContainer(w, core.Bytes(), e, v.pos, v.generation, uint64(cudaStateMaxBytes))
}

type cudaSupportLoadHooks struct {
	snapshot           func() (dsv4.Snapshot, error)
	afterStageTransfer func() error
	afterCore          func() error
	beforeRollback     func() error
}

func (s *cudaRequestState) loadSupport(r io.Reader) error {
	return s.loadSupportWithHooks(r, nil)
}

// Hooks follow the existing core Load transaction fault-control convention.
// Public Load always supplies nil and uses the actual snapshot/transfer paths.
func (s *cudaRequestState) loadSupportWithHooks(r io.Reader, hooks *cudaSupportLoadHooks) (err error) {
	if s.lifecycleEpoch == ^uint64(0) {
		return errors.New("cuda support load: speculative epoch overflow")
	}
	core, support, generation, err := readDSparkSupportContainer(r, true, uint64(cudaStateMaxBytes))
	if err != nil {
		return err
	}
	if len(core) < cudaStateHeaderBytes || !bytes.Equal(core[:8], cudaStateMagic[:]) {
		return errors.New("cuda support load: invalid core header")
	}
	position := binary.LittleEndian.Uint64(core[12:20])
	if position > uint64(s.runtime.ctx) {
		return errors.New("cuda support load: invalid frontier")
	}
	expected, err := cudaSupportShape(s.dspark)
	if err != nil {
		return err
	}
	e, err := readDSparkSupport(support, expected, int(position), generation)
	if err != nil {
		return err
	}
	// Staging is a separately owned native snapshot. Failed transfers must
	// retain its cleanup owner; no destination allocation has changed yet.
	snapshot := s.Snapshot
	if hooks != nil && hooks.snapshot != nil {
		snapshot = hooks.snapshot
	}
	stage, err := snapshot()
	if err != nil {
		return err
	}
	if stage == nil {
		return errors.New("cuda support load: missing staging owner")
	}
	defer func() { err = finishBackendCleanup(err, stage) }()
	nativeStage, ok := stage.(*cudaStateSnapshot)
	if !ok || nativeStage == nil || nativeStage.closed || nativeStage.closing || nativeStage.owner != s.runtime || nativeStage.dspark == nil || nativeStage.dspark.owner != s.dspark {
		return errors.New("cuda support load: foreign staging owner")
	}
	v := nativeStage.dspark
	if err := v.history.CopyFromHost(e.history); err != nil {
		return err
	}
	if err := v.features.CopyFromHost(e.capture.features); err != nil {
		return err
	}
	if err := v.status.CopyFromHost([]uint32{e.capture.status}); err != nil {
		return err
	}
	if hooks != nil && hooks.afterStageTransfer != nil {
		if err := hooks.afterStageTransfer(); err != nil {
			return err
		}
	}
	v.journal, v.captured = e.journal, e.capture.captured
	v.position, v.token, v.generation, v.published = e.capture.position, e.capture.token, e.capture.generation, e.capture.published
	if err := s.dspark.validateSnapshot(v, int(position), generation); err != nil {
		return err
	}
	backup, err := snapshot()
	if err != nil {
		return err
	}
	if backup == nil {
		return errors.New("cuda support load: missing backup owner")
	}
	defer func() { err = finishBackendCleanup(err, backup) }()
	nativeBackup, ok := backup.(*cudaStateSnapshot)
	if !ok || nativeBackup == nil || nativeBackup.owner != s.runtime {
		return errors.New("cuda support load: foreign backup owner")
	}
	err = cudaStateTransaction(func() error {
		if err := loadCUDAState(s, bytes.NewReader(core)); err != nil {
			return err
		}
		if hooks != nil && hooks.afterCore != nil {
			if err := hooks.afterCore(); err != nil {
				return err
			}
		}
		if err := syncStream(s.runtime.stream()); err != nil {
			return err
		}
		if err := s.dspark.restore(v, int(position), generation); err != nil {
			return err
		}
		s.generation = generation
		return nil
	}, func() error {
		if hooks != nil && hooks.beforeRollback != nil {
			if err := hooks.beforeRollback(); err != nil {
				return err
			}
		}
		return s.restoreSnapshot(nativeBackup)
	}, func(err error) {
		s.runtime.failed = err
		s.runtime.headObserver.invalidate()
	})
	return s.completeObservedCacheReplacement(err)
}
