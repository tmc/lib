package nccltest

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/driver"
)

type rankGraph struct {
	capturing  bool
	graph      driver.Graph
	executable driver.GraphExec
}

// captureGraphs captures submissions on every rank's stream. The caller owns
// returned resources even on error and must close them before freeing buffers.
func captureGraphs(states []rankState, submit func() error) ([]rankGraph, error) {
	if len(states) == 0 || submit == nil {
		return nil, fmt.Errorf("nccltest: empty graph capture")
	}
	for i := range states {
		if states[i].stream == 0 {
			return nil, fmt.Errorf("nccltest: graph capture requires a dedicated stream for rank %d", i)
		}
	}
	graphs := make([]rankGraph, len(states))
	for i := range states {
		s := &states[i]
		if err := s.context.SetCurrent(); err != nil {
			return graphs, err
		}
		if err := s.stream.BeginCapture(driver.StreamCaptureModeThreadLocal); err != nil {
			return graphs, err
		}
		graphs[i].capturing = true
	}
	if err := submit(); err != nil {
		return graphs, err
	}
	for i := range states {
		if err := states[i].context.SetCurrent(); err != nil {
			return graphs, err
		}
		graph, err := states[i].stream.EndCapture()
		graphs[i].capturing = false // EndCapture terminates even an invalidated capture.
		graphs[i].graph = graph
		if err != nil {
			return graphs, err
		}
	}
	for i := range states {
		if err := states[i].context.SetCurrent(); err != nil {
			return graphs, err
		}
		executable, err := graphs[i].graph.Instantiate(0)
		graphs[i].executable = executable
		if err != nil {
			return graphs, err
		}
	}
	return graphs, nil
}

func launchGraphs(states []rankState, graphs []rankGraph) error {
	if len(states) == 0 || len(states) != len(graphs) {
		return fmt.Errorf("nccltest: graph rank count mismatch")
	}
	for i := range states {
		if graphs[i].executable == 0 {
			return fmt.Errorf("nccltest: missing executable graph for rank %d", i)
		}
		if err := states[i].context.SetCurrent(); err != nil {
			return err
		}
		if err := graphs[i].executable.GraphLaunch(states[i].stream); err != nil {
			return err
		}
	}
	return nil
}

// closeGraphs ends partial captures and drains launches before destruction.
// It leaves context-bound resources alone when selecting their context fails.
func closeGraphs(states []rankState, graphs []rankGraph) error {
	defer func() {
		for _, g := range graphs {
			if g.capturing || g.graph != 0 || g.executable != 0 {
				// A captured collective can reference peer ranks. Retain the
				// entire run, not just the rank whose graph failed to close.
				for i := range states {
					states[i].retainResources = true
				}
				return
			}
		}
	}()
	if len(states) != len(graphs) {
		return fmt.Errorf("nccltest: graph rank count mismatch")
	}
	var result error
	for i := len(graphs) - 1; i >= 0; i-- {
		g := &graphs[i]
		if !g.capturing && g.graph == 0 && g.executable == 0 {
			continue
		}
		if err := states[i].context.SetCurrent(); err != nil {
			result = errors.Join(result, fmt.Errorf("nccltest: graph cleanup rank %d (resources retained): %w", i, err))
			continue
		}
		if g.capturing {
			graph, err := states[i].stream.EndCapture()
			g.capturing = false
			g.graph = graph
			result = errors.Join(result, err)
		}
		if g.executable != 0 {
			if err := states[i].stream.Synchronize(); err != nil {
				result = errors.Join(result, err)
				continue
			}
			if err := g.executable.Destroy(); err != nil {
				result = errors.Join(result, err)
				continue
			}
			g.executable = 0
		}
		if g.graph != 0 {
			if err := g.graph.Destroy(); err != nil {
				result = errors.Join(result, err)
				continue
			}
			g.graph = 0
		}
	}
	return result
}
