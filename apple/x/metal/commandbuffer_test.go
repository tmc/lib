//go:build darwin

package metal

import (
	"context"
	"errors"
	"strings"
	"testing"

	apple "github.com/tmc/apple/metal"
)

func TestCommandBufferCompletion(t *testing.T) {
	tests := []struct {
		name    string
		status  apple.MTLCommandBufferStatus
		detail  commandBufferDetail
		ctxErr  error
		failure CommandBufferFailure
		is      error
		text    string
	}{
		{name: "completed", status: apple.MTLCommandBufferStatusCompleted},
		{name: "canceled", status: apple.MTLCommandBufferStatusCompleted, ctxErr: context.Canceled, failure: CommandBufferCanceled, is: context.Canceled},
		{name: "deadline", status: apple.MTLCommandBufferStatusCompleted, ctxErr: context.DeadlineExceeded, failure: CommandBufferDeadlineExceeded, is: context.DeadlineExceeded},
		{name: "error without detail", status: apple.MTLCommandBufferStatusError, failure: CommandBufferFailed, text: "detail unavailable"},
		{name: "error with detail", status: apple.MTLCommandBufferStatusError, detail: commandBufferDetail{available: true, code: 3, domain: "MTLCommandBufferErrorDomain", text: "page fault"}, failure: CommandBufferFailed, text: "page fault"},
		{name: "device timeout", status: apple.MTLCommandBufferStatusError, detail: commandBufferDetail{available: true, code: int(apple.MTLCommandBufferErrorTimeout), domain: "MTLCommandBufferErrorDomain", text: "timeout"}, failure: CommandBufferDeviceTimeout, text: "timed out"},
		{name: "nonterminal", status: apple.MTLCommandBufferStatusScheduled, failure: CommandBufferUnexpectedStatus, text: "Scheduled"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := commandBufferCompletion(test.status, test.detail, test.ctxErr)
			if test.failure == 0 {
				if err != nil {
					t.Fatalf("commandBufferCompletion: %v", err)
				}
				return
			}
			var got *CommandBufferError
			if !errors.As(err, &got) || got.Failure != test.failure {
				t.Fatalf("error = %#v, want failure %d", err, test.failure)
			}
			if test.is != nil && !errors.Is(err, test.is) {
				t.Fatalf("errors.Is(%v) = false", test.is)
			}
			if test.text != "" && !strings.Contains(err.Error(), test.text) {
				t.Fatalf("error %q does not contain %q", err, test.text)
			}
		})
	}
}

func TestDeviceFailureWinsOverContextCancellation(t *testing.T) {
	err := commandBufferCompletion(apple.MTLCommandBufferStatusError, commandBufferDetail{}, context.Canceled)
	if errors.Is(err, context.Canceled) {
		t.Fatal("device failure was hidden by context cancellation")
	}
	var got *CommandBufferError
	if !errors.As(err, &got) || got.Failure != CommandBufferFailed || !got.DeviceTerminal {
		t.Fatalf("error = %#v", err)
	}
}

func TestWaitCommandBufferHardware(t *testing.T) {
	dev, err := NewDevice()
	if err != nil {
		t.Skip(err)
	}
	t.Run("completed", func(t *testing.T) {
		cb := dev.Queue().CommandBuffer()
		cb.Commit()
		if err := WaitCommandBuffer(t.Context(), cb); err != nil {
			t.Fatalf("WaitCommandBuffer: %v", err)
		}
	})
	t.Run("canceled context still waits", func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		cancel()
		cb := dev.Queue().CommandBuffer()
		cb.Commit()
		err := WaitCommandBuffer(ctx, cb)
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("WaitCommandBuffer error = %v, want context cancellation", err)
		}
		if cb.Status() != apple.MTLCommandBufferStatusCompleted {
			t.Fatalf("status = %s, want completed", cb.Status())
		}
		var completion *CommandBufferError
		if !errors.As(err, &completion) || !completion.DeviceTerminal {
			t.Fatalf("error = %#v, want completed device state", err)
		}
	})
}
