package main

import (
	"encoding/json"
	"sort"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// attachUnits names, for each module, the generated translation unit its
// kernels were lowered from, so a reader can report the module's instructions
// under the program's own lines rather than under generated text that was
// never written to disk.
//
// A unit is matched to a module through the kernels that ran: a unit is named
// after the kernel it generates, and a kernel activity names both the kernel
// and the module it came from. Matching on the image would be tighter, but
// the image says only what the compiler emitted, and the compiler does not
// record which generator produced its input. Two builds of one unit therefore
// both match it, which is right: they are two builds of one unit.
func attachUnits(c *cudacapture.Capture, activities []captureActivity, attachments map[string][]byte) {
	units := map[string]string{} // unit name -> attachment name
	for name, data := range attachments {
		if !strings.HasPrefix(name, "unit-") || !strings.HasSuffix(name, ".json") {
			continue
		}
		var u struct {
			Name string `json:"Name"`
		}
		if err := json.Unmarshal(data, &u); err != nil || u.Name == "" {
			continue
		}
		units[u.Name] = name
	}
	if len(units) == 0 {
		return
	}
	// A capture of one unit needs no match: there is one program, and every
	// module in the capture is a build of it.
	if len(units) == 1 && len(c.Modules) <= 1 {
		for _, file := range units {
			c.Code.Unit = file
			for i := range c.Modules {
				c.Modules[i].Code.Unit = file
			}
		}
		return
	}
	// Longest name first, so a unit named for a kernel another unit's name is
	// a prefix of is matched to its own kernel.
	names := make([]string, 0, len(units))
	for name := range units {
		names = append(names, name)
	}
	sort.Slice(names, func(i, j int) bool { return len(names[i]) > len(names[j]) })
	for i := range c.Modules {
		id := c.Modules[i].DriverID
		if id == 0 {
			continue
		}
		for _, a := range activities {
			if a.Subtype != "kernel" || a.ModuleID != id {
				continue
			}
			for _, name := range names {
				if strings.Contains(a.Name, name) {
					c.Modules[i].Code.Unit = units[name]
					break
				}
			}
			if c.Modules[i].Code.Unit != "" {
				break
			}
		}
	}
}
