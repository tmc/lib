package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/tmc/lib/nvidia/x/cudalineinfo"
	"github.com/tmc/lib/nvidia/x/cudaprofagent"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

// publish hands a unit to the cudaprof capture this process is running
// under, if it is running under one, so a per-line profile of it can name
// the program's own lines. Without a capture it does nothing.
func publish(u cudalineinfo.Unit) error {
	data, err := json.Marshal(u)
	if err != nil {
		return fmt.Errorf("encode %s: %w", u.Name, err)
	}
	if _, err := cudaprofagent.WriteUnit(u.Name, data); err != nil {
		return err
	}
	return nil
}

// export writes a kernel's unit and build artifacts into dir, named after
// the kernel, in the forms the cudalineinfo command reads: the unit as JSON
// with its functions and callers, the annotated source, the PTX, the CUBIN,
// and the disassembly.
func export(ctx context.Context, dir string, a *cudalineinfo.Attribution, nvdisasm string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	unit, err := json.MarshalIndent(a.Unit, "", "\t")
	if err != nil {
		return fmt.Errorf("encode %s: %w", a.Unit.Name, err)
	}
	sass, err := cudasass.Disassembler{Path: nvdisasm}.Disassemble(ctx, a.Module.CUBIN(), cudasass.Options{PrintCode: true, SourceLines: true})
	if err != nil {
		return fmt.Errorf("disassemble %s: %w", a.Unit.Name, err)
	}
	for ext, data := range map[string][]byte{
		".json":  unit,
		".cu":    []byte(a.Unit.SourceWithLineDirectives()),
		".ptx":   []byte(a.PTX),
		".cubin": a.Module.CUBIN(),
		".sass":  sass.Stdout,
	} {
		if err := os.WriteFile(filepath.Join(dir, a.Unit.Name+ext), data, 0o644); err != nil {
			return err
		}
	}
	return nil
}
