package cudaprofagent

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// WriteUnit publishes one generated translation unit to the capture this
// process is running under, and reports whether there was a capture to
// publish to.
//
// The agent collects what the driver and the hardware saw: modules, kernels,
// and program counters. None of it says what the program asked for. A unit is
// that missing half -- the generated CUDA the program's own code was lowered
// to, and the map from each of its lines back to the line and call stack it
// came from -- and only the process that generated it has it, which is why
// this is published from the target rather than collected by the agent.
//
// unitJSON is the JSON form of a cudalineinfo.Unit. It is taken as bytes so
// that finalizing a capture stays free of the compiler stack: a target that
// generates device code already has the unit encoded, and a target that does
// not has no reason to link the packages that define it.
//
// Publishing the same name twice replaces it. Names are taken from the unit,
// so two processes of one capture that generate the same unit publish the
// same bytes; two that generate different code under one name publish a
// capture that can only describe one of them, and the last one wins.
//
// Without a capture it writes nothing and returns false, so the call is safe
// to leave in a program that is usually run without one.
func WriteUnit(name string, unitJSON []byte) (bool, error) {
	dir := os.Getenv("CUDAPROF_TRANSPORT")
	if dir == "" {
		return false, nil
	}
	if err := validUnitName(name); err != nil {
		return false, err
	}
	dir = filepath.Join(dir, "units")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return false, fmt.Errorf("publish unit %s: %w", name, err)
	}
	// Written through a temporary file: the capture harvests this directory
	// when the target exits, and a half-written unit is worse than a missing
	// one because it reads as a unit that lost its line map.
	tmp, err := os.CreateTemp(dir, ".unit-")
	if err != nil {
		return false, fmt.Errorf("publish unit %s: %w", name, err)
	}
	defer os.Remove(tmp.Name())
	if _, err := tmp.Write(unitJSON); err != nil {
		tmp.Close()
		return false, fmt.Errorf("publish unit %s: %w", name, err)
	}
	if err := tmp.Close(); err != nil {
		return false, fmt.Errorf("publish unit %s: %w", name, err)
	}
	if err := os.Rename(tmp.Name(), filepath.Join(dir, name+".json")); err != nil {
		return false, fmt.Errorf("publish unit %s: %w", name, err)
	}
	return true, nil
}

// validUnitName rejects a name that would name a file outside the capture's
// own directory. A unit name reaches this from a code generator, which is one
// step from the program's own identifiers.
func validUnitName(name string) error {
	if name == "" {
		return errors.New("unit has no name")
	}
	if strings.ContainsAny(name, `/\`) || strings.Contains(name, "..") {
		return fmt.Errorf("invalid unit name %q", name)
	}
	return nil
}
