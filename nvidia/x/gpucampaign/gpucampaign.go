package gpucampaign

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

// Status is a phase's outcome.
type Status string

const (
	Pass   Status = "PASS"    // exited 0 within the deadline with no terminal CUDA error
	Hold   Status = "HOLD"    // ran and failed, timed out, logged a terminal CUDA error, or could not start
	NotRun Status = "NOT_RUN" // skipped; Reason says why
)

// A Phase is one subprocess of a campaign.
type Phase struct {
	Name string
	Argv []string
	Cap  time.Duration // longest the phase may run
	GPU  bool          // uses the GPU, so a lost GPU skips it
	Dir  string        // working directory; "" for the current one
	Env  []string      // added to the runner's environment
	Log  string        // file for combined stdout and stderr; "" discards after terminal-error scan
}

// A Result is a phase's receipt.
type Result struct {
	Name          string    `json:"name"`
	Status        Status    `json:"status"`
	Reason        string    `json:"reason,omitempty"`
	TerminalError string    `json:"terminal_error,omitempty"`
	ExitCode      int       `json:"exit_code"` // -1 if it did not exit normally
	TimedOut      bool      `json:"timed_out,omitempty"`
	Start         time.Time `json:"start,omitzero"`
	End           time.Time `json:"end,omitzero"`
	Log           string    `json:"log,omitempty"`
}

// A Runner runs phases against one deadline.
type Runner struct {
	// Health, if set, is called after a GPU phase fails. A non-nil error
	// marks the GPU lost for the rest of the campaign.
	Health func(context.Context) error
	// Grace is how long a killed process group is given to exit before the
	// runner stops waiting for its output; default one second.
	Grace time.Duration

	deadline time.Time
	lost     string
	now      func() time.Time
}

// NewRunner returns a runner whose phases must all finish within budget.
func NewRunner(budget time.Duration) *Runner {
	return &Runner{deadline: time.Now().Add(budget), now: time.Now}
}

// Remaining returns the time left before the deadline.
func (r *Runner) Remaining() time.Duration { return r.deadline.Sub(r.now()) }

// Lost returns the health or terminal CUDA failure that stopped GPU phases, or "".
func (r *Runner) Lost() string { return r.lost }

// Run runs p and returns its receipt. It never returns an error: every
// failure is a receipt.
func (r *Runner) Run(ctx context.Context, p Phase) Result {
	res := Result{Name: p.Name, Status: NotRun, ExitCode: -1, Log: p.Log}
	switch {
	case p.GPU && r.lost != "":
		res.Reason = "GPU lost earlier: " + r.lost
		return res
	case len(p.Argv) == 0:
		res.Status, res.Reason = Hold, "empty command"
		return res
	case p.Cap <= 0:
		res.Status, res.Reason = Hold, "phase cap must be positive"
		return res
	case p.Cap > r.Remaining():
		res.Reason = fmt.Sprintf("cap %v does not fit in the %v left", p.Cap, r.Remaining().Round(time.Millisecond))
		return res
	}
	res.Start = r.now()
	var output []byte
	res.ExitCode, res.TimedOut, res.Reason, output = r.exec(ctx, p)
	res.End = r.now()

	if term := DetectTerminalCUDAError(output); term != "" {
		res.TerminalError = term
		msg := "terminal CUDA error: " + term
		if res.Reason == "" {
			res.Reason = msg
		} else {
			res.Reason += "; " + msg
		}
		if p.GPU && r.lost == "" {
			r.lost = msg
		}
	}
	if !res.TimedOut && res.End.After(r.deadline) {
		res.TimedOut = true
		msg := "campaign deadline expired before phase completed"
		if res.Reason == "" {
			res.Reason = msg
		} else {
			res.Reason += "; " + msg
		}
	}
	if res.ExitCode == 0 && !res.TimedOut && res.Reason == "" {
		res.Status = Pass
		return res
	}
	res.Status = Hold
	if p.GPU && r.Health != nil {
		if err := r.Health(ctx); err != nil {
			if r.lost == "" {
				r.lost = err.Error()
			}
			res.Reason += "; GPU health failed: " + err.Error()
		}
	}
	return res
}

const maxCapturedOutput = 1 << 20

type boundedBuffer struct {
	buf bytes.Buffer
	max int
}

func (b *boundedBuffer) Write(p []byte) (int, error) {
	orig := len(p)
	if rem := b.max - b.buf.Len(); rem > 0 {
		if len(p) > rem {
			p = p[:rem]
		}
		b.buf.Write(p)
	}
	return orig, nil
}

func (r *Runner) exec(ctx context.Context, p Phase) (code int, timedOut bool, reason string, output []byte) {
	ctx, cancel := context.WithTimeout(ctx, p.Cap)
	defer cancel()
	cmd := exec.CommandContext(ctx, p.Argv[0], p.Argv[1:]...)
	cmd.Dir = p.Dir
	if len(p.Env) > 0 {
		cmd.Env = append(os.Environ(), p.Env...)
	}
	capBuf := &boundedBuffer{max: maxCapturedOutput}
	var outWriter io.Writer = capBuf
	if p.Log != "" {
		f, err := os.Create(p.Log)
		if err != nil {
			return -1, false, err.Error(), nil
		}
		defer f.Close()
		outWriter = io.MultiWriter(f, capBuf)
	}
	cmd.Stdout, cmd.Stderr = outWriter, outWriter
	setProcessGroup(cmd)
	cmd.Cancel = func() error { return killProcessGroup(cmd.Process) }
	cmd.WaitDelay = r.Grace
	if cmd.WaitDelay <= 0 {
		cmd.WaitDelay = time.Second
	}
	err := cmd.Run()
	output = capBuf.buf.Bytes()
	if ctx.Err() == context.DeadlineExceeded {
		return -1, true, fmt.Sprintf("cap %v expired; process group killed", p.Cap), output
	}
	if err == nil {
		return 0, false, "", output
	}
	var ee *exec.ExitError
	if errors.As(err, &ee) {
		if ee.Exited() {
			return ee.ExitCode(), false, fmt.Sprintf("exit status %d", ee.ExitCode()), output
		}
		return -1, false, ee.String(), output
	}
	return -1, false, err.Error(), output
}

var terminalCUDAPattern = regexp.MustCompile(`(?i)` +
	`\bCUDA\s+(?:is\s+)?unavailable\b` +
	`|\bno CUDA GPUs are available\b` +
	`|\bone [A-Za-z0-9_-]+ CUDA device required\b` +
	`|\bone visible GPU required\b` +
	`|\bCUDA\s+(?:driver\s+)?initialization (?:failed|error)\b` +
	`|\bCUDA (?:error|warning):\s*(?:an illegal memory access|an illegal instruction|misaligned address|device-side assert|unspecified launch failure|the launch timed out and was terminated|uncorrectable ECC error|unknown error)\b` +
	`|\ban illegal memory access was encountered\b` +
	`|\bdevice-side assert triggered\b` +
	`|\bcudaError(?:NoDevice|InsufficientDriver|InitializationError|IllegalAddress|IllegalInstruction|MisalignedAddress|Assert|LaunchFailure|LaunchTimeout|HardwareStackError|ECCUncorrectable|DevicesUnavailable|Unknown)\b` +
	`|\bCUDA_ERROR_(?:NO_DEVICE|SYSTEM_DRIVER_MISMATCH|NOT_INITIALIZED|ILLEGAL_ADDRESS|ILLEGAL_INSTRUCTION|MISALIGNED_ADDRESS|ASSERT|LAUNCH_FAILED|LAUNCH_TIMEOUT|HARDWARE_STACK_ERROR|ECC_UNCORRECTABLE|DEVICES_UNAVAILABLE)\b` +
	`|\bReturning \d+ \(CUDA_ERROR_[A-Z_]+\) from [A-Za-z_]\w*\b` +
	`|\bNVRM:\s*Xid\b` +
	`|\bXid\s*\([^)]*\):\s*\d+` +
	`|\bDriver could not be loaded\b` +
	`|\bCUPTI_ERROR_UNKNOWN\b`,
)

var ignoredMetadataKeys = map[string]bool{
	"source":          true,
	"sources":         true,
	"source_code":     true,
	"config":          true,
	"configuration":   true,
	"metadata":        true,
	"expected_output": true,
	"expected_errors": true,
}

var diagnosticStringKeys = map[string]bool{
	"error":                true,
	"initialization_error": true,
	"exception":            true,
	"stdout":               true,
	"stderr":               true,
	"capture_error":        true,
	"failure":              true,
	"reason":               true,
	"diagnostic":           true,
	"message":              true,
}

// DetectTerminalCUDAError inspects phase output (raw stdout/stderr, JSON lines,
// or structured JSON receipts) and returns the first terminal CUDA or device
// diagnostic found, or "" if none is present. Metadata fields such as
// source_code or expected_errors in JSON objects are ignored.
func DetectTerminalCUDAError(output []byte) string {
	trimmed := bytes.TrimSpace(output)
	if len(trimmed) == 0 {
		return ""
	}
	if trimmed[0] == '{' {
		var obj map[string]any
		if err := json.Unmarshal(trimmed, &obj); err == nil {
			return detectInJSONValue(obj)
		}
	}
	for _, rawLine := range bytes.Split(output, []byte("\n")) {
		line := strings.TrimSpace(string(rawLine))
		if line == "" {
			continue
		}
		if strings.HasPrefix(line, "{") {
			var obj map[string]any
			if err := json.Unmarshal([]byte(line), &obj); err == nil {
				if m := detectInJSONValue(obj); m != "" {
					return m
				}
				continue
			}
		}
		if m := terminalCUDAPattern.FindString(line); m != "" {
			return line
		}
	}
	return ""
}

func detectInJSONValue(v any) string {
	switch x := v.(type) {
	case map[string]any:
		for k, val := range x {
			if ignoredMetadataKeys[k] {
				continue
			}
			if str, ok := val.(string); ok && diagnosticStringKeys[k] {
				if m := DetectTerminalCUDAError([]byte(str)); m != "" {
					return m
				}
			}
		}
		for k, val := range x {
			if ignoredMetadataKeys[k] {
				continue
			}
			switch val.(type) {
			case map[string]any, []any:
				if m := detectInJSONValue(val); m != "" {
					return m
				}
			}
		}
	case []any:
		for _, item := range x {
			if m := detectInJSONValue(item); m != "" {
				return m
			}
		}
	}
	return ""
}

// WriteCheckpoint writes v as indented JSON to path if and only if the
// runner's monotonic deadline has not expired before or during the write and
// sync. If the deadline expires before commit, any prior file at path is
// preserved (rolled back) and an error wrapping [context.DeadlineExceeded] is
// returned.
func (r *Runner) WriteCheckpoint(path string, v any) error {
	if r.Remaining() <= 0 {
		return fmt.Errorf("write checkpoint: %w", context.DeadlineExceeded)
	}
	prev, prevErr := os.ReadFile(path)
	hadPrev := prevErr == nil
	if prevErr != nil && !errors.Is(prevErr, os.ErrNotExist) {
		return fmt.Errorf("write checkpoint: %w", prevErr)
	}
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return fmt.Errorf("write checkpoint: %w", err)
	}
	f, err := os.CreateTemp(filepath.Dir(path), "."+filepath.Base(path)+".pending-*")
	if err != nil {
		return fmt.Errorf("write checkpoint: %w", err)
	}
	tmp := f.Name()
	_, err = f.Write(append(data, '\n'))
	if err == nil {
		err = f.Sync()
	}
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		os.Remove(tmp)
		return fmt.Errorf("write checkpoint: %w", err)
	}
	if r.Remaining() <= 0 {
		os.Remove(tmp)
		return fmt.Errorf("write checkpoint: %w", context.DeadlineExceeded)
	}
	if err := os.Rename(tmp, path); err != nil {
		os.Remove(tmp)
		return fmt.Errorf("write checkpoint: %w", err)
	}
	if r.Remaining() <= 0 {
		if hadPrev {
			_ = WriteJSON(path, json.RawMessage(prev))
		} else {
			_ = os.Remove(path)
		}
		return fmt.Errorf("write checkpoint: %w", context.DeadlineExceeded)
	}
	return nil
}

// WriteJSON writes v as indented JSON to path by writing a temporary file
// in the same directory and renaming it, so readers see either the old
// receipt or the new one.
func WriteJSON(path string, v any) error {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return fmt.Errorf("write receipt: %w", err)
	}
	f, err := os.CreateTemp(filepath.Dir(path), "."+filepath.Base(path)+".pending-*")
	if err != nil {
		return fmt.Errorf("write receipt: %w", err)
	}
	tmp := f.Name()
	_, err = f.Write(append(data, '\n'))
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err == nil {
		err = os.Rename(tmp, path)
	}
	if err != nil {
		os.Remove(tmp)
		return fmt.Errorf("write receipt: %w", err)
	}
	return nil
}
