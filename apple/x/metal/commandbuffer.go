//go:build darwin

package metal

import (
	"context"
	"errors"
	"fmt"

	apple "github.com/tmc/apple/metal"
)

// CommandBufferFailure classifies an unsuccessful command buffer wait.
type CommandBufferFailure uint8

const (
	CommandBufferFailed CommandBufferFailure = iota + 1
	CommandBufferCanceled
	CommandBufferDeadlineExceeded
	CommandBufferDeviceTimeout
	CommandBufferUnexpectedStatus
)

// CommandBufferError reports why a committed command buffer did not produce a
// usable result. DeviceTerminal is true once Metal has reached a terminal
// status, even when the caller's context was canceled while the GPU was still
// running.
type CommandBufferError struct {
	Failure           CommandBufferFailure
	Status            apple.MTLCommandBufferStatus
	DeviceTerminal    bool
	DetailAvailable   bool
	DeviceErrorCode   int
	DeviceErrorDomain string
	DeviceErrorText   string
	cause             error
}

func (e *CommandBufferError) Error() string {
	switch e.Failure {
	case CommandBufferCanceled:
		return "metal: command buffer completed after context cancellation"
	case CommandBufferDeadlineExceeded:
		return "metal: command buffer completed after context deadline"
	case CommandBufferDeviceTimeout:
		return e.deviceError("metal: command buffer timed out")
	case CommandBufferUnexpectedStatus:
		return fmt.Sprintf("metal: command buffer wait ended with status %s", e.Status)
	default:
		return e.deviceError("metal: command buffer execution failed")
	}
}

func (e *CommandBufferError) deviceError(prefix string) string {
	if !e.DetailAvailable {
		return prefix + ": error detail unavailable"
	}
	return fmt.Sprintf("%s: domain=%s code=%d: %s", prefix, e.DeviceErrorDomain, e.DeviceErrorCode, e.DeviceErrorText)
}

// Unwrap exposes context cancellation and deadline errors to errors.Is.
func (e *CommandBufferError) Unwrap() error { return e.cause }

type commandBufferDetail struct {
	available bool
	code      int
	domain    string
	text      string
}

// WaitCommandBuffer waits until Metal reaches a terminal status and checks it.
// Context cancellation does not return early: Metal provides no safe way to
// cancel arbitrary committed work, and callers may release referenced buffers
// immediately after this function returns.
func WaitCommandBuffer(ctx context.Context, cb apple.MTLCommandBuffer) error {
	if cb == nil || cb.GetID() == 0 {
		return errors.New("metal: nil command buffer")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	cb.WaitUntilCompleted()
	detail := commandBufferDetail{}
	if nsErr := cb.Error(); nsErr.GetID() != 0 {
		detail = commandBufferDetail{
			available: true,
			code:      nsErr.Code(),
			domain:    string(nsErr.Domain()),
			text:      nsErr.LocalizedDescription(),
		}
	}
	return commandBufferCompletion(cb.Status(), detail, ctx.Err())
}

func commandBufferCompletion(status apple.MTLCommandBufferStatus, detail commandBufferDetail, contextErr error) error {
	deviceTerminal := status == apple.MTLCommandBufferStatusCompleted || status == apple.MTLCommandBufferStatusError
	if status == apple.MTLCommandBufferStatusError {
		failure := CommandBufferFailed
		if detail.available && detail.domain == string(apple.MTLCommandBufferErrorDomain) && apple.MTLCommandBufferError(detail.code) == apple.MTLCommandBufferErrorTimeout {
			failure = CommandBufferDeviceTimeout
		}
		return &CommandBufferError{Failure: failure, Status: status, DeviceTerminal: deviceTerminal, DetailAvailable: detail.available, DeviceErrorCode: detail.code, DeviceErrorDomain: detail.domain, DeviceErrorText: detail.text}
	}
	if status != apple.MTLCommandBufferStatusCompleted {
		return &CommandBufferError{Failure: CommandBufferUnexpectedStatus, Status: status, DeviceTerminal: deviceTerminal}
	}
	if errors.Is(contextErr, context.DeadlineExceeded) {
		return &CommandBufferError{Failure: CommandBufferDeadlineExceeded, Status: status, DeviceTerminal: true, cause: context.DeadlineExceeded}
	}
	if errors.Is(contextErr, context.Canceled) {
		return &CommandBufferError{Failure: CommandBufferCanceled, Status: status, DeviceTerminal: true, cause: context.Canceled}
	}
	return nil
}
