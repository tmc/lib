//go:build darwin

package metalprof

import (
	"fmt"
	"os"
	"os/exec"

	"github.com/tmc/apple/foundation"
	apple "github.com/tmc/apple/metal"
	"github.com/tmc/lib/apple/x/metal"
)

// CaptureSession tracks an active programmatic Metal GPU trace capture.
type CaptureSession struct {
	mgr       apple.MTLCaptureManager
	outputURL string
}

// StartCapture initiates programmatic capture of Metal GPU workloads to a .gputrace directory.
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

// ReplayTrace invokes the macOS system MTLReplayer headlessly to profile and extract hardware counters.
// Uses the recipe discovered in gputrace: open -W -n -a MTLReplayer.app --args -CLI in.gputrace ...
func ReplayTrace(inTracePath, outTracePath string) error {
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

	cmd := exec.Command("open", "-W", "-n", "-a", replayerApp, "--args",
		"-CLI", inTracePath,
		"-profileTrace",
		"-collectProfilerData",
		"-outputPath", outTracePath,
	)

	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("metalprof: headless replay failed: %w (output: %s)", err, string(out))
	}
	return nil
}
