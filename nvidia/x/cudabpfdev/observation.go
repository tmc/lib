package cudabpfdev

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"slices"
)

// InvocationObservation is a detached host receipt from one loaded owner.
// TimestampNano is completion publication time, available only when Complete.
// It contains no device identity, native function offset or device duration.
type InvocationObservation struct {
	Original                               KernelIdentity
	Selection                              Selection
	Geometry                               Geometry
	Schema                                 cudakernel.Schema
	Ready, Used, Pending, Closed, Complete bool
	TimestampNano                          int64
	owner                                  *LoadedInjection
}

// SameInvocation reports whether two receipts came from the same in-process
// owner. It is not an authenticated claim about serialized or caller data.
func (o InvocationObservation) SameInvocation(other InvocationObservation) bool {
	return o.owner != nil && o.owner == other.owner
}

// InvocationReader is a read-only capability for one LoadedInjection.
// Its zero value is unavailable. It exposes no driver handles or actuators.
type InvocationReader struct{ owner *LoadedInjection }

// ObservationReader returns a read-only view that remains readable after Close.
func (l *LoadedInjection) ObservationReader() InvocationReader { return InvocationReader{l} }

// ReadObservation copies retained host state under the owner's mutex without
// calling native operations. Selection alone does not imply execution.
func (r InvocationReader) ReadObservation() (InvocationObservation, error) {
	l := r.owner
	if l == nil {
		return InvocationObservation{}, fmt.Errorf("cudabpfdev: unavailable invocation reader")
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	schema := l.schema
	schema.Params = slices.Clone(schema.Params)
	o := InvocationObservation{Original: l.originalKernel, Selection: l.selection,
		Geometry: l.geometry, Schema: schema, Ready: l.ready, Used: l.used, Pending: l.pending,
		Closed: l.closed, Complete: l.complete, owner: l}
	if l.complete {
		o.TimestampNano = l.timestamp
	}
	return o, nil
}
