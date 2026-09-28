package main

import (
	"bytes"
	"crypto/sha256"
	"debug/elf"
	"fmt"
	"path/filepath"
	"strings"

	"github.com/google/pprof/driver"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

// loadRetainedImages indexes code kept without the complete source/PTX/SASS
// bundle LoadCode requires. In particular, activity captures keep individual
// module CUBINs, not a whole-capture CUBIN. Disassemble them only when the user
// requests retained code; ordinary profile reports do not need nvdisasm.
func (t *objectTool) loadRetainedImages() error {
	if t.capture == nil {
		return nil
	}
	roles := []cudacapture.CodeFiles{t.capture.Code}
	for _, m := range t.capture.Modules {
		roles = append(roles, m.Code)
	}
	for _, role := range roles {
		for _, item := range []struct {
			name  string
			cubin bool
		}{{role.CUBIN, true}, {role.PTX, false}} {
			if item.name == "" {
				continue
			}
			path := filepath.Join(t.dir, item.name)
			if old := t.images[path]; old != nil && len(old.functions) > 0 {
				for i := range old.functions {
					old.functions[i].artifact = item.name
				}
				continue
			}
			data := t.capture.Files[item.name]
			if len(data) == 0 {
				return fmt.Errorf("retained code %s: missing artifact", item.name)
			}
			var image *objectImage
			var err error
			if item.cubin {
				// A retained listing also works on hosts without the CUDA tools.
				var dis *cudasass.Disassembly
				if listing := t.capture.Files[role.SASS]; len(listing) > 0 {
					dis, err = cudasass.ParseString(string(listing))
				} else {
					dis, err = cudasass.DisassembleCUBIN(data)
				}
				if err == nil {
					image, err = retainedCUBINImage(path, data, dis)
				}
			} else {
				image, err = ptxImage(path, fmt.Sprintf("%x", sha256.Sum256(data)), string(data))
			}
			if err != nil {
				return fmt.Errorf("retained code %s: %w", item.name, err)
			}
			for i := range image.functions {
				image.functions[i].artifact = item.name
			}
			t.images[path] = image
		}
	}
	return nil
}

// retainedCUBINImage uses ELF file offsets, not device PCs. No profile samples,
// instruction costs, or source associations are manufactured here.
func retainedCUBINImage(path string, data []byte, dis *cudasass.Disassembly) (*objectImage, error) {
	e, err := elf.NewFile(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("read cubin: %w", err)
	}
	defer e.Close()
	if e.Machine != elf.EM_CUDA {
		return nil, fmt.Errorf("cubin architecture is %s, want CUDA", e.Machine)
	}
	image := &objectImage{file: path, buildID: fmt.Sprintf("%x", sha256.Sum256(data)), size: uint64(len(data))}
	for _, f := range dis.Functions {
		sec := e.Section(".text." + f.Name)
		if sec == nil {
			return nil, fmt.Errorf("missing cubin section for %s", f.Name)
		}
		fn := objectFunction{name: f.Name + ".SASS", start: sec.Offset, end: sec.Offset + sec.Size}
		for _, in := range f.Instructions {
			if in.Address >= sec.Size {
				return nil, fmt.Errorf("instruction offset %#x outside %s", in.Address, sec.Name)
			}
			fn.instructions = append(fn.instructions, driver.Inst{Addr: sec.Offset + in.Address, Text: strings.TrimSpace(in.Raw), Function: fn.name})
		}
		image.functions = append(image.functions, fn)
	}
	return image, nil
}
