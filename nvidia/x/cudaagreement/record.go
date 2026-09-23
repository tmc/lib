package cudaagreement

import (
	"errors"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudaplan"
)

// Buffer identifies one device buffer by address and extent. Two arms that
// name the same address wrote the same memory, which is why the record carries
// it: outputs read from different buffers are not evidence of agreement.
type Buffer struct {
	Ptr   driver.DevicePtr
	Bytes int
}

func (b Buffer) String() string { return fmt.Sprintf("0x%x+%d", uint64(b.Ptr), b.Bytes) }

// Step is one kernel dispatch. The artifact's parameter schema takes the input
// buffers in order followed by the output buffer, and nothing else.
type Step struct {
	Name     string
	Artifact *cudakernel.Artifact
	Inputs   []string
	Output   string
	Grid     uint32
	Block    uint32
}

// Plan is a chain of steps over named buffers. Step i is ordered after step
// i-1; the chain, not a dependency analysis, is what both executors preserve.
type Plan struct {
	Name  string
	Steps []Step
}

// Validate reports whether the plan can be submitted at all: every step has a
// name, an artifact whose schema matches its buffer count, and a nonzero
// launch geometry.
func (p *Plan) Validate() error {
	if p == nil || len(p.Steps) == 0 {
		return errors.New("cudaagreement: plan has no steps")
	}
	for i, s := range p.Steps {
		switch {
		case s.Name == "":
			return fmt.Errorf("cudaagreement: step %d has no name", i)
		case s.Artifact == nil:
			return fmt.Errorf("cudaagreement: step %s has no artifact", s.Name)
		case len(s.Inputs) == 0:
			return fmt.Errorf("cudaagreement: step %s has no inputs", s.Name)
		case s.Output == "":
			return fmt.Errorf("cudaagreement: step %s has no output", s.Name)
		case s.Grid == 0 || s.Block == 0:
			return fmt.Errorf("cudaagreement: step %s geometry is %dx%d", s.Name, s.Grid, s.Block)
		}
		if want := len(s.Inputs) + 1; len(s.Artifact.Schema.Params) != want {
			return fmt.Errorf("cudaagreement: step %s artifact takes %d parameters, want %d", s.Name, len(s.Artifact.Schema.Params), want)
		}
	}
	return nil
}

// StepRecord is what an executor reports about one step after submitting it.
// Executed is the backend that ran the step, which differs from Requested
// whenever the executor declined the step and took another path; Declined then
// says why.
type StepRecord struct {
	Step       string
	Requested  cudaplan.TargetBackend
	Executed   cudaplan.TargetBackend
	Declined   string
	Digest     string
	EntryPoint string
	Grid       uint32
	Block      uint32
	Output     Buffer
}

func (s StepRecord) String() string {
	line := fmt.Sprintf("step=%s requested=%s executed=%s grid=%d block=%d digest=%s entry=%s out=%s",
		s.Step, s.Requested, s.Executed, s.Grid, s.Block, s.Digest, s.EntryPoint, s.Output)
	if s.Declined != "" {
		line += " declined=" + s.Declined
	}
	return line
}

// Record is one arm of a run: the backend the arm asked for and, per step, the
// backend that ran it.
type Record struct {
	Arm       string
	Requested cudaplan.TargetBackend
	Steps     []StepRecord
}

func (r *Record) String() string {
	if r == nil {
		return "<nil record>"
	}
	lines := make([]string, 0, len(r.Steps)+1)
	lines = append(lines, fmt.Sprintf("arm=%s requested=%s steps=%d", r.Arm, r.Requested, len(r.Steps)))
	for _, s := range r.Steps {
		lines = append(lines, "  "+s.String())
	}
	return strings.Join(lines, "\n")
}

// Declined returns the steps whose executed backend is not the requested one.
func (r *Record) Declined() []StepRecord {
	if r == nil {
		return nil
	}
	var out []StepRecord
	for _, s := range r.Steps {
		if s.Executed != s.Requested {
			out = append(out, s)
		}
	}
	return out
}

// ExecutedOn reports an error naming every step that did not record b. It reads
// the executed backend, never the requested one.
func (r *Record) ExecutedOn(b cudaplan.TargetBackend) error {
	if r == nil || len(r.Steps) == 0 {
		return errors.New("cudaagreement: record has no steps")
	}
	var bad []string
	for _, s := range r.Steps {
		if s.Executed != b {
			bad = append(bad, fmt.Sprintf("%s ran on %s (%s)", s.Step, s.Executed, s.Declined))
		}
	}
	if len(bad) > 0 {
		return fmt.Errorf("cudaagreement: arm %s did not execute on %s: %s", r.Arm, b, strings.Join(bad, "; "))
	}
	return nil
}
