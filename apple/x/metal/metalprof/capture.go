//go:build darwin

package metalprof

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/tmc/apple/foundation"
	apple "github.com/tmc/apple/metal"
	"github.com/tmc/lib/apple/x/metal"
)

// ErrReplayUnresolvedOwnership is returned when replay cancellation occurs but the underlying
// process cannot be proven terminated via direct process handle ownership.
var ErrReplayUnresolvedOwnership = errors.New("metalprof: replay cancelled with unresolved process ownership")

// CaptureSession tracks an active programmatic Metal GPU trace capture.
type CaptureSession struct {
	mgr       apple.MTLCaptureManager
	outputURL string
}

// StartCapture initiates programmatic capture of Metal GPU workloads to a
// .gputrace directory. Metal only inserts the capture layer when the process
// starts with MTL_CAPTURE_ENABLED=1 in its environment; without it StartCapture
// reports that the capture layer is not inserted.
func StartCapture(dev *metal.Device, outputDirectory string) (*CaptureSession, error) {
	if dev == nil || dev.Raw().GetID() == 0 {
		return nil, fmt.Errorf("metalprof: nil or uninitialized device")
	}
	if outputDirectory == "" {
		outputDirectory = "capture.gputrace"
	}

	mgr := apple.GetMTLCaptureManagerClass().SharedCaptureManager()
	if mgr.GetID() == 0 {
		return nil, fmt.Errorf("metalprof: MTLCaptureManager unavailable")
	}

	if mgr.IsCapturing() {
		return nil, fmt.Errorf("metalprof: capture already in progress")
	}

	desc := apple.GetMTLCaptureDescriptorClass().Alloc().Init()
	desc.SetCaptureObject(dev.Raw())
	desc.SetDestination(apple.MTLCaptureDestinationGPUTraceDocument)

	nsURL := foundation.GetNSURLClass().FileURLWithPath(outputDirectory)
	desc.SetOutputURL(nsURL)

	ok, err := mgr.StartCaptureWithDescriptorError(desc)
	if err != nil || !ok {
		return nil, fmt.Errorf("metalprof: start capture failed: %w", err)
	}

	return &CaptureSession{
		mgr:       mgr,
		outputURL: outputDirectory,
	}, nil
}

// Stop terminates the active GPU capture session and flushes the .gputrace archive to disk.
func (cs *CaptureSession) Stop() error {
	if cs == nil || cs.mgr.GetID() == 0 {
		return nil
	}
	if !cs.mgr.IsCapturing() {
		return nil
	}
	cs.mgr.StopCapture()
	return nil
}

// ReplayTrace invokes the macOS system MTLReplayer headlessly to profile a
// captured trace. It writes <outTracePath>/<trace base name>.gpuprofiler_raw,
// holding the replayer's per-frame counter, timeline, and shader profiler
// payloads.
func ReplayTrace(inTracePath, outTracePath string) error {
	return ReplayTraceContext(context.Background(), inTracePath, outTracePath)
}

// ReplayTraceContext invokes MTLReplayer with context cancellation.
// Because MTLReplayer on macOS is launched via LaunchServices (`open -W -n -a ...`),
// child process ownership is not directly held by the caller. If the context expires or is
// cancelled, ReplayTraceContext leaves the scratch directory intact and returns a typed
// ErrReplayUnresolvedOwnership rather than attempting broad, unsafe process kills by name/path.
func ReplayTraceContext(ctx context.Context, inTracePath, outTracePath string) error {
	if _, err := os.Stat(inTracePath); err != nil {
		return fmt.Errorf("metalprof: input trace not found: %w", err)
	}
	if outTracePath == "" {
		outTracePath = inTracePath + ".profiled"
	}

	replayerApp := "/System/Library/CoreServices/MTLReplayer.app"
	if _, err := os.Stat(replayerApp); err != nil {
		return fmt.Errorf("metalprof: MTLReplayer not available at %s", replayerApp)
	}

	absIn, err := filepath.Abs(inTracePath)
	if err != nil {
		return err
	}
	absOut, err := filepath.Abs(outTracePath)
	if err != nil {
		return err
	}

	cmd := exec.CommandContext(ctx, "open", "-W", "-n", "-a", replayerApp, "--args",
		"-CLI", absIn,
		"-profileTrace",
		"-collectProfilerData",
		"-outputPath", absOut,
	)

	out, err := cmd.CombinedOutput()
	if err != nil {
		if ctx.Err() != nil {
			return fmt.Errorf("%w: %w (output: %s)", ErrReplayUnresolvedOwnership, ctx.Err(), string(out))
		}
		return fmt.Errorf("metalprof: headless replay failed: %w (output: %s)", err, string(out))
	}
	return nil
}
