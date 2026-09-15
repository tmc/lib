// Copyright 2026 The Go Authors. All rights reserved.

package metalprof

import "time"

// CommandBufferTiming is the GPU execution window of one completed command
// buffer, as reported by Metal itself.
type CommandBufferTiming struct {
	Start time.Duration // GPUStartTime, relative to an arbitrary origin
	End   time.Duration // GPUEndTime, on the same origin as Start
}

// GPU reports how long the GPU spent executing the command buffer.
func (t CommandBufferTiming) GPU() time.Duration { return t.End - t.Start }

// Valid reports whether the timing describes a real execution window. Metal
// leaves both timestamps at zero for a command buffer that has not completed,
// and a scheduling error can leave End before Start.
func (t CommandBufferTiming) Valid() bool { return t.Start > 0 && t.End > t.Start }

func seconds(v float64) time.Duration {
	return time.Duration(v * float64(time.Second))
}
