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
	switch err := publish("units", name, unitJSON); {
	case errors.Is(err, errNoCapture):
		return false, nil
	case err != nil:
		return false, fmt.Errorf("publish unit %s: %w", name, err)
	}
	return true, nil
}

// publish writes data as <transport>/<kind>/<name>.json, and reports
// errNoCapture when this process is running under no capture.
//
// Written through a temporary file: the capture harvests these directories
// when the target exits, and a half-written file is worse than a missing one,
// because it reads as one that lost part of what it was published to carry.
func publish(kind, name string, data []byte) error {
	transport := os.Getenv("CUDAPROF_TRANSPORT")
	if transport == "" {
		return errNoCapture
	}
	if err := validName(name); err != nil {
		return err
	}
	dir := filepath.Join(transport, kind)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, "."+kind+"-")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), filepath.Join(dir, name+".json"))
}

// errNoCapture reports that this process runs under no capture. It never
// reaches a caller: the exported functions turn it into a false return, so
// that a call left in a program usually run without a capture is not an
// error path.
var errNoCapture = errors.New("no capture")

// validName rejects a name that would name a file outside the capture's own
// directory. A name reaches this from a code generator, which is one step
// from the program's own identifiers.
func validName(name string) error {
	if name == "" {
		return errors.New("no name")
	}
	if strings.ContainsAny(name, `/\`) || strings.Contains(name, "..") {
		return fmt.Errorf("invalid name %q", name)
	}
	return nil
}
