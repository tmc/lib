package cuptibench

import (
	"errors"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cuptirange"
	"strings"
)

// Workload describes one replayable sequence. Streams must list every stream
// participating in Run. Prepare restores the sequence's starting state once per
// pass; Preparation documents fixture, state and cache policy even when Prepare
// is nil. Run and Synchronize are required. Callbacks must not change contexts.
type Workload struct {
	// Graphs supplies retained native kernel-order proofs for graph attribution.
	// Executables must remain live and unmodified throughout collection.
	Graphs []*cudagraph.Exec
	// View is "full", "isolated", or empty for an unspecified measurement view.
	View        string
	Streams     []*cuda.Stream
	Prepare     func() error
	Run         func(*Scope) error
	Synchronize func() error
	Preparation string
}

// Scope provides immediate named regions during Run. It is valid only on the
// calling goroutine until Run returns. Its zero value is not usable.
type Scope struct {
	region  func(string, func() error) error
	active  bool
	failure error
}

// Region executes run immediately. Regions mode records a balanced nested range;
// other modes and timing simply execute run. Names are nonempty path components.
// Panic unwinding closes native regions and preserves the panic.
func (s *Scope) Region(name string, run func() error) (err error) {
	if s == nil || !s.active {
		return errors.New("cuptibench: scope is not active")
	}
	defer func() { s.failure = errors.Join(s.failure, err) }()
	if name == "" || strings.ContainsAny(name, "/\x00") || run == nil {
		return errors.New("cuptibench: invalid region name or callback")
	}
	if s.region != nil {
		return s.region(name, run)
	}
	return run()
}

func runWorkload(w Workload, region func(string, func() error) error) error {
	s := &Scope{region: region, active: true}
	defer func() { s.active = false }()
	return errors.Join(w.Run(s), s.failure)
}
func (p *Plan) validateWorkload(w Workload) error {
	if p.config.Mode == Kernels && len(w.Graphs) > 0 {
		if p.config.ReplayMode != cuptirange.KernelReplay {
			return errors.New("cuptibench: graph kernel attribution requires kernel replay")
		}
		seenGraphs := make(map[*cudagraph.Exec]bool)
		for _, graph := range w.Graphs {
			if graph == nil || seenGraphs[graph] {
				return errors.New("cuptibench: nil or duplicate graph executable")
			}
			seenGraphs[graph] = true
			if _, _, err := graph.KernelOrder(); err != nil {
				return err
			}
		}
	}
	if w.View != "" && w.View != "full" && w.View != "isolated" {
		return errors.New("cuptibench: invalid workload view")
	}
	if w.Run == nil || w.Synchronize == nil || strings.TrimSpace(w.Preparation) == "" || len(w.Streams) == 0 {
		return errors.New("cuptibench: workload needs streams, run, synchronization, and preparation policy")
	}
	seen := make(map[*cuda.Stream]bool)
	for _, s := range w.Streams {
		if s == nil || seen[s] || s.Device() != p.dev {
			return errors.New("cuptibench: nil, duplicate, or incompatible stream")
		}
		seen[s] = true
	}
	return nil
}
