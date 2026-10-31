package main

import (
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/safetensors"
)

// openAfmoeCheckpoint reads the original shared HF shards. It never writes
// a combined checkpoint or expands the model into host float32 weights.
func openAfmoeCheckpoint(dir string) (checkpoint, *afmoeConfig, error) {
	b, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, nil, fmt.Errorf("afmoe config: %w", err)
	}
	cfg, err := parseAfmoeConfig(b)
	if err != nil {
		return nil, nil, err
	}
	ck, err := safetensors.OpenSharded(dir)
	if err != nil {
		return nil, nil, fmt.Errorf("open afmoe shards: %w", err)
	}
	expected := cfg.tensors()
	if len(ck.Shard) != len(expected) {
		err = fmt.Errorf("afmoe checkpoint: %d tensors, want %d", len(ck.Shard), len(expected))
	} else {
		for _, spec := range expected {
			if err = checkAfmoeTensor(ck, spec); err != nil {
				break
			}
		}
	}
	if err != nil {
		if cerr := ck.Close(); cerr != nil {
			return ck, nil, errors.Join(err, cerr)
		}
		return nil, nil, err
	}
	return ck, cfg, nil
}

func checkAfmoeTensor(ck checkpoint, spec afmoeTensor) error {
	info, ok := ck.Info(spec.Name)
	if !ok {
		return fmt.Errorf("afmoe checkpoint: missing tensor %s", spec.Name)
	}
	if !slices.Equal(info.Shape, spec.Shape) || info.Dtype != safetensors.BF16 {
		return fmt.Errorf("afmoe tensor %s: shape %v dtype %s, want %v BF16", spec.Name, info.Shape, info.Dtype, spec.Shape)
	}
	return nil
}

// afmoeWeights owns each uploaded tensor once. Matrices remain BF16;
// one-dimensional learned norms and expert biases widen to float32 for
// kernels, preserving the original BF16 values exactly.
type afmoeWeights struct {
	Tensors map[string]deviceWeight
}

func (w *afmoeWeights) Free() error {
	if w == nil {
		return nil
	}
	var names []string
	for name := range w.Tensors {
		names = append(names, name)
	}
	slices.Sort(names)
	var err error
	for _, name := range names {
		tensor := w.Tensors[name]
		if e := tensor.release(); e != nil {
			err = errors.Join(err, fmt.Errorf("free afmoe tensor %s: %w", name, e))
			continue
		}
		delete(w.Tensors, name)
	}
	return err
}

func streamAfmoeWeights(dev *cuda.Device, ck checkpoint, cfg *afmoeConfig) (*afmoeWeights, error) {
	return streamAfmoeWeightsWith(ck, cfg.tensors(), make([]byte, streamChunk),
		func(n int, dtype weightDType) (deviceWeight, error) { return allocWeight(dev, n, dtype) }, writeDevice)
}

func streamAfmoeWeightsWith(ck checkpoint, specs []afmoeTensor, buf []byte,
	alloc func(int, weightDType) (deviceWeight, error),
	write func(deviceBuffer, int, []byte) error) (res *afmoeWeights, err error) {
	if ck == nil || len(specs) == 0 || len(buf) < 4 || alloc == nil || write == nil {
		return nil, fmt.Errorf("afmoe stream: missing buffer or device operation")
	}
	// Inspect every header before allocating any device weight.
	seen := make(map[string]bool)
	for _, spec := range specs {
		if spec.Name == "" || seen[spec.Name] || len(spec.Shape) == 0 {
			return nil, fmt.Errorf("afmoe stream: empty or duplicate tensor specification %q", spec.Name)
		}
		seen[spec.Name] = true
		if err := checkAfmoeTensor(ck, spec); err != nil {
			return nil, err
		}
	}
	w := &afmoeWeights{Tensors: make(map[string]deviceWeight)}
	defer func() {
		if err != nil {
			if cerr := w.Free(); cerr != nil {
				res = w
				err = errors.Join(err, cerr)
			}
		}
	}()
	for _, spec := range specs {
		n := 1
		for _, d := range spec.Shape {
			if d <= 0 || n > math.MaxInt/d {
				return nil, fmt.Errorf("afmoe tensor %s: element count overflows", spec.Name)
			}
			n *= d
		}
		if n > math.MaxInt/4 {
			return nil, fmt.Errorf("afmoe tensor %s: byte count overflows", spec.Name)
		}
		to := weightBF16
		if len(spec.Shape) == 1 {
			to = weightF32
		}
		tensor, aerr := alloc(n, to)
		w.Tensors[spec.Name] = tensor // Retain ownership even on partial allocation failure.
		if aerr != nil {
			return nil, fmt.Errorf("allocate afmoe tensor %s: %w", spec.Name, aerr)
		}
		if tensor.Ptr() == 0 || tensor.dtype != to {
			return nil, fmt.Errorf("allocate afmoe tensor %s: invalid owner or dtype", spec.Name)
		}
		r, rerr := ck.TensorReader(spec.Name)
		if rerr != nil {
			return nil, fmt.Errorf("read afmoe tensor %s: %w", spec.Name, rerr)
		}
		if r.Size() != int64(n)*2 {
			return nil, fmt.Errorf("afmoe tensor %s: source length %d, want %d", spec.Name, r.Size(), int64(n)*2)
		}
		source := &afmoeReadCount{Reader: r}
		expected := n * 2
		if to == weightF32 {
			expected = n * 4
		}
		written := 0
		if err := streamTensor(source, safetensors.BF16, to, buf, func(off int, b []byte) error {
			if off != written || len(b) > expected-written {
				return fmt.Errorf("noncontiguous or excess output at %d, length %d", off, len(b))
			}
			if err := write(tensor.buf, off, b); err != nil {
				return err
			}
			written += len(b)
			return nil
		}); err != nil {
			return nil, fmt.Errorf("upload afmoe tensor %s: %w", spec.Name, err)
		}
		if source.n != int64(n)*2 || written != expected {
			return nil, fmt.Errorf("afmoe tensor %s: short upload (read %d/%d, wrote %d/%d bytes)", spec.Name, source.n, int64(n)*2, written, expected)
		}
	}
	return w, nil
}

// afmoeReadCount checks a live read as well as the section's declared size.
// A checkpoint can be truncated after its header was inspected.
type afmoeReadCount struct {
	io.Reader
	n int64
}

func (r *afmoeReadCount) Read(b []byte) (int, error) {
	n, err := r.Reader.Read(b)
	r.n += int64(n)
	return n, err
}
