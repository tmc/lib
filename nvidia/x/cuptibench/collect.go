package cuptibench

import (
	"crypto/rand"
	"errors"
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cuptirange"
	"math"
	"strings"
)

// Collect runs bounded replay and returns counters without benchmark timing.
// It does not consult process flags. A failed capture returns diagnostic quality
// where available, never admissible counter values.
func (p *Plan) Collect(w Workload) (*Result, error) {
	if err := p.begin(); err != nil {
		return nil, err
	}
	defer p.end()
	return p.collect(w)
}

func (p *Plan) collect(w Workload) (*Result, error) {
	return p.collectMode(w, p.config.Mode)
}

func (p *Plan) collectMode(w Workload, mode Mode) (result *Result, err error) {
	defer func() {
		if value := recover(); value != nil {
			p.closed = true
			panic(value)
		}
	}()
	if err = p.validateWorkload(w); err != nil {
		return nil, err
	}
	w.Streams = append(w.Streams[:0:0], w.Streams...)
	w.Graphs = append([]*cudagraph.Exec(nil), w.Graphs...)
	replayMode := p.config.ReplayMode
	if mode != Kernels {
		replayMode = cuptirange.UserReplay
	}
	id := make([]byte, 16)
	if _, err = rand.Read(id); err != nil {
		return nil, fmt.Errorf("cuptibench: capture identity: %w", err)
	}
	result = &Result{Operations: p.config.Operations, Metadata: make(map[string]string)}
	for k, v := range p.metadata {
		result.Metadata[k] = v
	}
	result.Metadata["capture_id"] = fmt.Sprintf("%x", id)
	result.Metadata["mode"] = []string{"whole", "regions", "kernels"}[mode]
	result.Metadata["replay_mode"] = replayMode.String()
	depth := captureDepth(mode, p.config.MaxDepth)
	if depth == 0 {
		depth = 16
	}
	result.Metadata["configured_nesting_levels"] = fmt.Sprint(depth)
	result.Metadata["replay_pass_accounting"] = "host start/stop calls"
	if replayMode == cuptirange.KernelReplay {
		result.Metadata["replay_pass_accounting"] = "host start/stop calls; native internal kernel passes not observed"
	}
	result.Metadata["metrics"] = strings.Join(p.config.Metrics, ",")
	result.Metadata["preparation"] = w.Preparation
	result.Metadata["view"] = w.View
	if w.View == "" {
		result.Metadata["view"] = "unspecified"
	}
	err = p.context.Run(func() error {
		capture, e := p.backend.CaptureReplay(p.dev, p.config.Limits, cuptirange.CaptureOptions{ReplayMode: replayMode, Graphs: w.Graphs, CachePolicy: p.config.CachePolicy, Mode: cuptirange.Mode(mode), MaxDepth: captureDepth(mode, p.config.MaxDepth)}, cuptirange.Replay{
			Prepare: func(uint) error {
				if w.Prepare != nil {
					return p.call(w.Prepare)
				}
				return nil
			},
			Run: func(_ uint, s *cuptirange.Scope) error {
				for i := 0; i < p.config.Operations; i++ {
					if e := p.call(func() error {
						return runWorkload(w, func(name string, run func() error) error { return s.Region(name, func() error { return p.call(run) }) })
					}); e != nil {
						return e
					}
				}
				return nil
			},
			Synchronize: func() error { return p.call(w.Synchronize) },
		})
		if capture != nil {
			result.Quality = capture.Quality
			if cleanup := capture.Close(); cleanup != nil {
				p.pending = capture
				e = errors.Join(e, cleanup)
			}
		}
		if e != nil {
			return e
		}
		if capture == nil {
			return errors.New("cuptibench: collector returned no capture")
		}
		q := capture.Quality
		if !q.AllPassesSubmitted || !q.Decoded || q.DroppedRanges != 0 || q.PassesStarted == 0 || q.PassesStarted != q.PassesStopped || q.PassesStarted > p.config.Limits.MaxPasses {
			return errors.New("cuptibench: incomplete or lossy capture")
		}
		if len(capture.RangeInfo) != int(capture.Ranges) || capture.Ranges == 0 || capture.Ranges > p.config.Limits.MaxRanges {
			return errors.New("cuptibench: missing range identities")
		}
		seen := make(map[string]bool)
		indexes := make(map[uint]bool)
		for _, r := range capture.RangeInfo {
			if r.ID == "" || seen[r.ID] || indexes[r.Index] || r.Index >= capture.Ranges {
				return errors.New("cuptibench: invalid range identity")
			}
			seen[r.ID] = true
			indexes[r.Index] = true
			values, e := p.backend.Evaluate(capture.Image, r.Index)
			if e != nil {
				return e
			}
			if len(values) != len(p.config.Metrics) {
				return errors.New("cuptibench: metric count mismatch")
			}
			out := Range{ID: r.ID, ParentID: r.ParentID, Name: r.Name, NativeName: r.NativeName, Kind: r.Kind, Inclusive: r.Inclusive}
			for i, v := range values {
				if math.IsNaN(v) || math.IsInf(v, 0) {
					return errors.New("cuptibench: nonfinite metric")
				}
				unit := "unknown"
				if i < len(p.units) && p.units[i] != "" {
					unit = p.units[i]
				}
				additive := i < len(p.additive) && p.additive[i]
				if additive && v < 0 {
					return errors.New("cuptibench: negative event counter")
				}
				out.Metrics = append(out.Metrics, Metric{Name: p.config.Metrics[i], Unit: unit, Value: v, Additive: additive})
			}
			result.Ranges = append(result.Ranges, out)
		}
		return nil
	})
	if err != nil {
		result.Ranges = nil
	}
	return result, err
}

func captureDepth(mode Mode, depth uint16) uint16 {
	if mode != Regions {
		return 1
	}
	return depth
}
