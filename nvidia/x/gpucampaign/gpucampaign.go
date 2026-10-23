package gpucampaign

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"time"
)

// Status is a phase's outcome.
type Status string

const (
	Pass   Status = "PASS"    // exited 0
	Hold   Status = "HOLD"    // ran and failed, timed out, or could not start
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
	Log  string        // file for combined stdout and stderr; "" discards
}

// A Result is a phase's receipt.
type Result struct {
	Name     string    `json:"name"`
	Status   Status    `json:"status"`
	Reason   string    `json:"reason,omitempty"`
	ExitCode int       `json:"exit_code"` // -1 if it did not exit normally
	TimedOut bool      `json:"timed_out,omitempty"`
	Start    time.Time `json:"start,omitzero"`
	End      time.Time `json:"end,omitzero"`
	Log      string    `json:"log,omitempty"`
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

// Lost returns the health failure that stopped GPU phases, or "".
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
	res.ExitCode, res.TimedOut, res.Reason = r.exec(ctx, p)
	res.End = r.now()
	if res.ExitCode == 0 && !res.TimedOut && res.Reason == "" {
		res.Status = Pass
		return res
	}
	res.Status = Hold
	if p.GPU && r.Health != nil {
		if err := r.Health(ctx); err != nil {
			r.lost = err.Error()
			res.Reason += "; GPU health failed: " + r.lost
		}
	}
	return res
}

func (r *Runner) exec(ctx context.Context, p Phase) (code int, timedOut bool, reason string) {
	ctx, cancel := context.WithTimeout(ctx, p.Cap)
	defer cancel()
	cmd := exec.CommandContext(ctx, p.Argv[0], p.Argv[1:]...)
	cmd.Dir = p.Dir
	if len(p.Env) > 0 {
		cmd.Env = append(os.Environ(), p.Env...)
	}
	if p.Log != "" {
		f, err := os.Create(p.Log)
		if err != nil {
			return -1, false, err.Error()
		}
		defer f.Close()
		cmd.Stdout, cmd.Stderr = f, f
	}
	setProcessGroup(cmd)
	cmd.Cancel = func() error { return killProcessGroup(cmd.Process) }
	cmd.WaitDelay = r.Grace
	if cmd.WaitDelay <= 0 {
		cmd.WaitDelay = time.Second
	}
	err := cmd.Run()
	if ctx.Err() == context.DeadlineExceeded {
		return -1, true, fmt.Sprintf("cap %v expired; process group killed", p.Cap)
	}
	if err == nil {
		return 0, false, ""
	}
	var ee *exec.ExitError
	if errors.As(err, &ee) {
		if ee.Exited() {
			return ee.ExitCode(), false, fmt.Sprintf("exit status %d", ee.ExitCode())
		}
		return -1, false, ee.String()
	}
	return -1, false, err.Error()
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
