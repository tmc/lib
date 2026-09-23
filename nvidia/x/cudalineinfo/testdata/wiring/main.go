// Command wiring exports the units the package's fixtures are compiled
// from. Its kernels.go is kept here, beside the fixtures, so that the lines
// the fixtures name, and the callers recorded above them, change only when
// this directory does.
//
// Usage:
//
//	go run ./testdata/wiring dir
//
// testdata/gen.go runs it; see there.
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/tmc/lib/nvidia/x/cudafuse"
	"github.com/tmc/lib/nvidia/x/cudair"
)

func main() {
	log.SetFlags(0)
	log.SetPrefix("wiring: ")
	if len(os.Args) != 2 {
		log.Fatal("usage: wiring dir")
	}
	if err := run(os.Args[1]); err != nil {
		log.Fatal(err)
	}
}

func run(dir string) error {
	for _, k := range kernels {
		if err := attribute(dir, k); err != nil {
			return err
		}
	}
	return nil
}

// attribute builds a kernel's graph, lowers it, and writes its unit to dir.
func attribute(dir string, k kernel) error {
	g := cudair.NewGraph()
	k.build(g)
	fk, err := cudafuse.FuseWithSourceMap(g, k.name)
	if err != nil {
		return fmt.Errorf("%s: fuse: %w", k.name, err)
	}
	data, err := json.MarshalIndent(fk.Unit(), "", "\t")
	if err != nil {
		return fmt.Errorf("encode %s: %w", k.name, err)
	}
	return os.WriteFile(filepath.Join(dir, k.name+".json"), data, 0o644)
}
