// Command ds4-stats summarizes a bounded little-endian float32 vector.
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"

	"github.com/tmc/lib/examples/ds4-min/dsv4tools"
)

const maxValues = 1 << 20

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

func run(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("ds4-stats", flag.ContinueOnError)
	fs.SetOutput(stderr)
	name := fs.String("name", "", "name printed in the summary")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 1 {
		fmt.Fprintln(stderr, "ds4-stats: exactly one vector file is required")
		return 2
	}
	path := fs.Arg(0)
	values, err := readVector(path)
	if err != nil {
		fmt.Fprintf(stderr, "ds4-stats: %v\n", err)
		return 1
	}
	if *name == "" {
		*name = filepath.Base(path)
	}
	if err := dsv4tools.WriteVectorStats(stdout, *name, values); err != nil {
		fmt.Fprintf(stderr, "ds4-stats: %v\n", err)
		return 1
	}
	return 0
}

func readVector(path string) ([]float32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open vector: %w", err)
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat vector: %w", err)
	}
	if info.Size() == 0 {
		return nil, fmt.Errorf("vector is empty")
	}
	maxBytes := int64(maxValues * 4)
	if info.Size() > maxBytes {
		return nil, fmt.Errorf("vector has %d bytes, exceeds %d-byte limit", info.Size(), maxBytes)
	}
	if info.Size()%4 != 0 {
		return nil, fmt.Errorf("vector size %d is not a multiple of 4", info.Size())
	}
	data, err := io.ReadAll(io.LimitReader(f, maxBytes+1))
	if err != nil {
		return nil, fmt.Errorf("read vector: %w", err)
	}
	if int64(len(data)) != info.Size() {
		return nil, fmt.Errorf("vector changed while reading")
	}
	values := make([]float32, len(data)/4)
	for i := range values {
		bits := binary.LittleEndian.Uint32(data[i*4:])
		values[i] = math.Float32frombits(bits)
	}
	if _, err := dsv4tools.SummarizeVector(values); err != nil {
		return nil, err
	}
	return values, nil
}
