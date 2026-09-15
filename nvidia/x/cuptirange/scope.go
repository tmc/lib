package cuptirange

import (
	"errors"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudagraph"
)

// Mode selects native range attribution.
type Mode int

const (
	Whole   Mode = iota // One inclusive operation range.
	Regions             // The operation and nested caller regions.
	Kernels             // Native automatic kernel ranges.
)

// ReplayMode selects who repeats the GPU workload to collect counters.
type ReplayMode int

const (
	UserReplay   ReplayMode = iota // The caller restores and repeats each profiling pass.
	KernelReplay                   // CUPTI repeats kernels internally; the host sequence runs once.
)

// String returns the replay mode used in capture metadata.
func (m ReplayMode) String() string {
	switch m {
	case UserReplay:
		return "user"
	case KernelReplay:
		return "kernel"
	default:
		return "unknown"
	}
}

// CaptureOptions bounds nesting, including the operation root. Zero MaxDepth
// selects one for Whole/Kernels and sixteen for Regions. KernelReplay requires
// Kernels at depth one and a qualified CUPTI 13.2 runtime.
type CaptureOptions struct {
	CachePolicy CachePolicy
	ReplayMode  ReplayMode
	// Graphs retains native graph identity witnesses for automatic kernel ranges.
	Graphs   []*cudagraph.Exec
	Mode     Mode
	MaxDepth uint16
}

// Replay describes one repeatable profiling sequence. Prepare runs before Start.
// Synchronize must join all participating work; it runs before and after Prepare,
// and after Run before the operation range closes. All callbacks run on the
// calling goroutine with the retained CUDA context current. KernelReplay calls
// Prepare and Run once, with pass zero; internal kernel replays are not callbacks.
type Replay struct {
	Prepare     func(uint) error
	Run         func(uint, *Scope) error
	Synchronize func() error
}

// RangeInfo maps a native evaluation index to its retained identity. User range
// IDs contain occurrence numbers. Kernel names come from observed launches or
// retained native graph proofs.
type RangeInfo struct {
	Index                    uint
	ID, ParentID, Name, Kind string
	NativeName               string // Native AutoRange launch ordinal; empty for user ranges.
	Inclusive                bool
}

// Scope records nested regions during a replay callback. It is valid only on
// that callback's goroutine. Its zero value is not usable.
type Scope struct {
	active  bool
	mode    Mode
	depth   uint16
	limit   uint
	stack   []string
	ranges  []RangeInfo
	counts  map[string]int
	push    func(string) error
	pop     func() error
	failure error
}

// Region executes run immediately. Whole and Kernels ignore annotations. Region
// balances successful pushes during errors and panic unwinding. Callers must
// propagate its error; ignored instrumentation errors still invalidate capture.
func (s *Scope) Region(name string, run func() error) (err error) {
	if s == nil || !s.active {
		return errors.New("cuptirange: inactive scope")
	}
	defer func() { s.failure = errors.Join(s.failure, err) }()
	if name == "" || strings.ContainsAny(name, "/\x00") || run == nil {
		return errors.New("cuptirange: invalid region")
	}
	if s.mode != Regions {
		return run()
	}
	return s.region(name, "region", run)
}

func (s *Scope) region(name, kind string, run func() error) (err error) {
	if len(s.stack) >= int(s.depth) || uint(len(s.ranges)) >= s.limit {
		return errors.New("cuptirange: region bounds exceeded")
	}
	parent := ""
	if len(s.stack) > 0 {
		parent = s.stack[len(s.stack)-1]
	}
	key := parent + "/" + name
	s.counts[key]++
	// Escape literal percent and hash so caller names cannot alias occurrence IDs.
	component := strings.NewReplacer("%", "%25", "#", "%23").Replace(name) + fmt.Sprintf("#%d", s.counts[key])
	id := component
	if parent != "" {
		id = parent + "/" + component
	}
	if err := s.push(component); err != nil {
		return err
	}
	s.ranges = append(s.ranges, RangeInfo{ID: id, ParentID: parent, Name: name, Kind: kind, Inclusive: true})
	s.stack = append(s.stack, id)
	defer func() { s.stack = s.stack[:len(s.stack)-1]; err = errors.Join(err, s.pop()) }()
	return run()
}
