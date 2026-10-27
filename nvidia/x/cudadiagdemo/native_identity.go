package cudadiagdemo

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

// nativeIdentityQuery maps owned CUDA handles to CUPTI's numeric activity IDs.
// Handle values are never cast to activity identities. The function fields
// allow CPU controls to execute this same query path without loading a driver.
type nativeIdentityQuery struct {
	current   func() (driver.Context, error)
	contextID func(driver.Context) (uint32, error)
	streamID  func(driver.Context, driver.Stream, uint8) (uint32, error)
}

func queryNativeIdentity(withContext func(func() error) error, withHandle func(func(driver.Stream) error) error, q nativeIdentityQuery) (cudaworkload.ContextStreamID, error) {
	var identity cudaworkload.ContextStreamID
	if withContext == nil || withHandle == nil || q.current == nil || q.contextID == nil || q.streamID == nil {
		return identity, errors.New("cudadiagdemo: incomplete context identity query")
	}
	err := withHandle(func(stream driver.Stream) error {
		return withContext(func() error {
			context, err := q.current()
			if err != nil {
				return fmt.Errorf("current context: %w", err)
			}
			contextID, err := q.contextID(context)
			if err != nil {
				return fmt.Errorf("CUPTI context ID: %w", err)
			}
			// This fixture owns an explicit non-default stream, not the per-thread
			// default stream. Passing its actual context requests membership checking.
			streamID, err := q.streamID(context, stream, 0)
			if err != nil {
				return fmt.Errorf("CUPTI stream ID: %w", err)
			}
			identity = cudaworkload.ContextStreamID{Context: uint64(contextID), Stream: uint64(streamID)}
			return nil
		})
	})
	if err != nil {
		return cudaworkload.ContextStreamID{}, err
	}
	return identity, nil
}

func (s *cudaNativeStream) attributionIdentity() (cudaworkload.ContextStreamID, error) {
	if s == nil || s.stream == nil {
		return cudaworkload.ContextStreamID{}, errors.New("cudadiagdemo: missing owned stream")
	}
	// Obtain the immutable owner before WithHandle: its callback may not call
	// stream methods. The lifetime lock excludes stream destruction throughout
	// the context-scoped queries; device context restoration is separately owned.
	dev := s.stream.Device()
	if dev == nil {
		return cudaworkload.ContextStreamID{}, errors.New("cudadiagdemo: missing stream device")
	}
	return queryNativeIdentity(dev.WithContext, s.stream.WithHandle, nativeIdentityQuery{driver.CtxGetCurrent, cupti.GetContextId, cupti.GetStreamIdEx})
}
