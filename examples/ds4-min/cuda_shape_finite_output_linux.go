//go:build linux

package main

import "errors"

func readShapeFiniteOutput(state *cudaShapeForward, name string) error {
	if state == nil || state.closed {
		return errors.New("cuda shape forward: closed state")
	}
	if err := state.r.checkOpen(); err != nil {
		return err
	}
	v := state.scratch[name]
	if v.Len() != state.plan.config.Embed || v.Ptr() == 0 {
		return errors.New("cuda shape forward: invalid output extent")
	}
	if err := state.r.withContext(func() error { return syncStream(state.r.stream()) }); err != nil {
		return err
	}
	values := make([]float32, v.Len())
	if err := v.CopyToHost(values); err != nil {
		return err
	}
	return checkShapeFiniteOutput(values, state.plan.config.Embed)
}
