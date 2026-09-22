package cudacensus

import (
	"fmt"
	"strings"
	"time"

	"github.com/tmc/lib/nvidia/dcgm"
)

// WatchDCGM creates a GPU group, field group and watch in session, then returns
// a reader for that watch. name must be unique on the host engine. updateFreq
// is the requested sampling cadence, not a measured integration interval;
// maxAge bounds retained sample age and the reader's freshness check. The watch
// retains at most two samples per field.
//
// The caller must close session even when this function fails: the session owns
// every group and watch created before a failure. This function never closes a
// caller's session. Prefer a dedicated session when independent cleanup is
// required. Do not read after closing it. Session.Close errors must be reported;
// the current binding does not retry failed teardown operations.
//
// Watching does not guarantee data or metric support. An auto-mode engine
// samples asynchronously; a manual-mode engine requires UpdateAllFields.
func WatchDCGM(session *dcgm.Session, name string, gpuID int32, want []Counter, updateFreq, maxAge time.Duration) (*DCGMReader, error) {
	if session == nil {
		return nil, fmt.Errorf("cudacensus: nil DCGM session")
	}
	return watchDCGM(session, session.Handle, name, gpuID, want, updateFreq, maxAge)
}

type dcgmWatchSession interface {
	NewGroup(string, dcgm.GroupType) (dcgm.GpuGrp, error)
	AddEntity(dcgm.GpuGrp, dcgm.FieldEntityGroup, dcgm.FieldEid) error
	NewFieldGroup(string, []dcgm.FieldID) (dcgm.FieldGrp, error)
	Watch(dcgm.GpuGrp, dcgm.FieldGrp, time.Duration, time.Duration, int32) error
}

func watchDCGM(session dcgmWatchSession, handle dcgm.Handle, name string, gpuID int32, want []Counter, updateFreq, maxAge time.Duration) (*DCGMReader, error) {
	if name == "" || len(name) >= dcgm.MaxStrLength || strings.ContainsRune(name, 0) {
		return nil, fmt.Errorf("cudacensus: invalid DCGM watch name")
	}
	if updateFreq < time.Microsecond || updateFreq%time.Microsecond != 0 || maxAge < updateFreq {
		return nil, fmt.Errorf("cudacensus: DCGM cadence must be whole positive microseconds and no greater than maximum age")
	}
	r, err := NewDCGMReader(handle, gpuID, want, maxAge)
	if err != nil {
		return nil, err
	}
	fields := r.Fields()
	if len(fields) == 0 {
		return nil, fmt.Errorf("cudacensus: no requested counters map to DCGM fields")
	}
	group, err := session.NewGroup(name, dcgm.GroupEmpty)
	if err != nil {
		return nil, fmt.Errorf("cudacensus: create DCGM GPU group: %w", err)
	}
	if err := session.AddEntity(group, dcgm.FeGpu, dcgm.FieldEid(gpuID)); err != nil {
		return nil, fmt.Errorf("cudacensus: add DCGM GPU: %w", err)
	}
	fieldGroup, err := session.NewFieldGroup(name, fields)
	if err != nil {
		return nil, fmt.Errorf("cudacensus: create DCGM field group: %w", err)
	}
	if err := session.Watch(group, fieldGroup, updateFreq, maxAge, 2); err != nil {
		return nil, fmt.Errorf("cudacensus: watch DCGM fields: %w", err)
	}
	return r, nil
}
