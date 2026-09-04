package main

import (
	"errors"

	"github.com/tmc/lib/examples/internal/continuousbatch"
)

var errServingBackendUnavailable = errors.New("serving backend unavailable")

// servingBackend owns request state and device resources used by the serving
// scheduler. PrefillBatch and DecodeBatch return one result per request, in
// order. They observe cancellation before submission; after submission they
// wait for device completion before publishing results or releasing state.
// Close is called only after the scheduler has drained all requests.
type servingBackend interface {
	continuousbatch.Backend
	continuousbatch.BatchPrefiller
	Close()
}

type servingPrefixCache interface {
	prefixCacheMetrics() (PrefixCacheMetrics, bool)
}

func newServingScheduler(backend servingBackend, cfg continuousbatch.Config) (*continuousbatch.Scheduler, error) {
	if backend == nil {
		return nil, errServingBackendUnavailable
	}
	return continuousbatch.New(backend, cfg)
}
