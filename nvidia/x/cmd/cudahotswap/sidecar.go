package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"syscall"

	"github.com/tmc/lib/nvidia/x/cudaref"
	"github.com/tmc/lib/nvidia/x/egraph"
)

func sidecarReadFile(path string) ([]byte, error) {
	f, err := os.OpenFile(path, os.O_RDONLY|syscall.O_NONBLOCK, 0)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil || !info.Mode().IsRegular() || info.Size() > 8<<20 {
		return nil, errors.New("sidecar file must be a bounded regular file")
	}
	b, err := io.ReadAll(io.LimitReader(f, (8<<20)+1))
	if err != nil {
		return nil, err
	}
	if len(b) > 8<<20 {
		return nil, errors.New("sidecar file exceeds size limit")
	}
	return b, nil
}

func loadSidecarReference(sourcePath, metadataPath, inputsPath string) (*sidecarReference, error) {
	code, err := sidecarReadFile(sourcePath)
	if err != nil {
		return nil, fmt.Errorf("sidecar source: %w", err)
	}
	meta, err := sidecarReadFile(metadataPath)
	if err != nil {
		return nil, fmt.Errorf("sidecar metadata: %w", err)
	}
	data, err := sidecarReadFile(inputsPath)
	if err != nil {
		return nil, fmt.Errorf("sidecar inputs: %w", err)
	}
	inputs, err := decodeSidecarInputs(data)
	if err != nil {
		return nil, fmt.Errorf("sidecar inputs: %w", err)
	}
	return readSidecarReference(bytes.NewReader(meta), string(code), inputs)
}

// sidecarReference holds the expression oracle and the emitted pointer order.
// CUDA edits invalidate attribution, but never replace this reference.
type sidecarReference struct {
	source egraph.Source
	inputs [][]float32
	want   []float32
}

func readSidecarReference(r io.Reader, code string, inputs map[string][]float32) (*sidecarReference, error) {
	source, err := egraph.ReadMetadata(r, code)
	if err != nil {
		return nil, fmt.Errorf("sidecar metadata: %w", err)
	}
	if len(source.Parameters) > 64 {
		return nil, errors.New("sidecar inputs: more than 64 parameters")
	}
	for _, p := range source.Parameters {
		size := 1
		for _, n := range p.Shape {
			size *= n
		}
		if len(inputs[p.Name]) != size {
			return nil, fmt.Errorf("sidecar inputs: shape extent differs for %q", p.Name)
		}
	}
	graph, err := source.ReferenceGraph()
	if err != nil {
		return nil, fmt.Errorf("sidecar reference: %w", err)
	}
	remaining := 1 << 20
	for _, node := range graph.Nodes() {
		size := 1
		for _, dimension := range node.Shape {
			if dimension <= 0 || dimension > remaining/size {
				return nil, errors.New("sidecar reference: intermediate element bound exceeded")
			}
			size *= dimension
		}
		if size > remaining {
			return nil, errors.New("sidecar reference: intermediate element bound exceeded")
		}
		remaining -= size
	}
	if len(inputs) != len(source.Parameters) {
		return nil, fmt.Errorf("sidecar inputs: parameter count mismatch")
	}
	total := 0
	owned := make(map[string][]float32, len(inputs))
	ref := &sidecarReference{source: source}
	for _, parameter := range source.Parameters {
		values, ok := inputs[parameter.Name]
		if !ok {
			return nil, fmt.Errorf("sidecar inputs: missing %q", parameter.Name)
		}
		if len(values) == 0 || len(values) > 65536-total {
			return nil, fmt.Errorf("sidecar inputs: bounded nonempty extent required for %q", parameter.Name)
		}
		total += len(values)
		for _, value := range values {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return nil, fmt.Errorf("sidecar inputs: nonfinite %q", parameter.Name)
			}
		}
		copy := append([]float32(nil), values...)
		owned[parameter.Name] = copy
		ref.inputs = append(ref.inputs, copy)
	}
	outputs, err := cudaref.Evaluate(graph, owned)
	if err != nil {
		return nil, fmt.Errorf("sidecar evaluate: %w", err)
	}
	want, ok := outputs["result"]
	if !ok || len(outputs) != 1 || len(want) == 0 || len(want) > 65536 {
		return nil, fmt.Errorf("sidecar reference: missing single result")
	}
	for _, value := range want {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, fmt.Errorf("sidecar reference: nonfinite result")
		}
	}
	ref.want = append([]float32(nil), want...)
	return ref, nil
}

// Decode tokens before allocating each array so duplicate names and declared
// resource bounds are enforced rather than overwritten by encoding/json.
func decodeSidecarInputs(data []byte) (map[string][]float32, error) {
	if len(data) == 0 || len(data) > 8<<20 {
		return nil, errors.New("input JSON size limit")
	}
	d := json.NewDecoder(bytes.NewReader(data))
	d.UseNumber()
	token, err := d.Token()
	if err != nil || token != json.Delim('{') {
		return nil, errors.New("expected named input object")
	}
	inputs := make(map[string][]float32)
	total := 0
	for d.More() {
		token, err := d.Token()
		if err != nil {
			return nil, err
		}
		name, ok := token.(string)
		if !ok || name == "" || len(name) > 256 || len(inputs) >= 64 {
			return nil, errors.New("input name or parameter bound exceeded")
		}
		if _, ok := inputs[name]; ok {
			return nil, fmt.Errorf("duplicate input %q", name)
		}
		token, err = d.Token()
		if err != nil || token != json.Delim('[') {
			return nil, fmt.Errorf("input %q must be an array", name)
		}
		var values []float32
		for d.More() {
			if total >= 65536 {
				return nil, errors.New("input element bound exceeded")
			}
			token, err := d.Token()
			if err != nil {
				return nil, err
			}
			number, ok := token.(json.Number)
			if !ok {
				return nil, fmt.Errorf("input %q requires finite FP32 numbers", name)
			}
			value, err := strconv.ParseFloat(string(number), 32)
			if err != nil || math.IsNaN(value) || math.IsInf(value, 0) {
				return nil, fmt.Errorf("input %q requires finite FP32 numbers", name)
			}
			values = append(values, float32(value))
			total++
		}
		if _, err := d.Token(); err != nil {
			return nil, err
		}
		if len(values) == 0 {
			return nil, fmt.Errorf("input %q is empty", name)
		}
		inputs[name] = values
	}
	if _, err := d.Token(); err != nil {
		return nil, err
	}
	if _, err := d.Token(); err != io.EOF {
		return nil, errors.New("trailing input JSON")
	}
	return inputs, nil
}
