//go:build darwin && arm64

package main

import "errors"

func (r *metalRuntime) nativeStateIdentity() ([32]byte, [32]byte, error) {
	if r == nil || r.stateIdentity == nil || r.file == nil || r.model == nil {
		return [32]byte{}, [32]byte{}, errors.New("metal state: model identity unavailable")
	}
	info, err := r.file.Stat()
	if err != nil {
		return [32]byte{}, [32]byte{}, err
	}
	return r.stateIdentity.get(r.file, info.Size(), r.model)
}
