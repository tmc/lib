package cudadiagdemo

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"

	"github.com/tmc/lib/nvidia/x/cudaregress"
)

// Retained numerical vector encoding and bounds.
const (
	nativeOperation = "output[i] = 2*input[i] + 1 in IEEE-754 binary32, round to nearest even, no fused multiply-add"
	nativeEncoding  = "little-endian IEEE-754 binary32, Elements values per file"
	// nativeMaxNumericBytes bounds the input, reference and output bytes
	// that every planned attempt may retain together.
	nativeMaxNumericBytes = 64 << 20
	nativeRetentionNone   = "none: OutputDir empty; no vector files retained"
)

// errNativeNumericBound reports a configuration whose planned retained
// vectors exceed nativeMaxNumericBytes.
var errNativeNumericBound = errors.New("cudadiagdemo: retained numerical vectors exceed 64 MiB")

// NativeNumerics describes the numerical vectors RunNative retains.
//
// With an empty OutputDir, Retention says that no vector files are retained;
// the trial digests are then the only numerical record. Otherwise
// PlannedBytes is the checked worst-case total of input, reference and
// output bytes for every planned attempt, at most 64 MiB, and Vectors lists,
// in plan order, each dispatched attempt that constructed its input.
//
// Retention is "retained" when every constructed vector was written to a new
// file and closed, and "incomplete" otherwise, with the failures in Error.
// It describes the files, not numerical correctness. Files are never removed
// or rewritten, including partially written ones.
type NativeNumerics struct {
	Retention    string          `json:"retention"`
	Operation    string          `json:"operation"`
	Encoding     string          `json:"encoding"`
	PlannedBytes int64           `json:"planned_bytes,omitempty"`
	Vectors      []NativeVectors `json:"vectors,omitempty"`
	Error        string          `json:"error,omitempty"`
}

// NativeVectors references the retained vectors of one request attempt.
// Plan is the attempt's plan index, which names its files. Reference is
// computed on the CPU from the input by NativeNumerics.Operation, never from
// the device output. Output is present only after a complete download,
// including an incorrect output; Error is the attempt's failure. A nil
// reference was not constructed or not completely written.
type NativeVectors struct {
	Plan      int                      `json:"plan"`
	RequestID string                   `json:"request_id"`
	Attempt   string                   `json:"attempt"`
	Input     *cudaregress.ArtifactRef `json:"input,omitempty"`
	Reference *cudaregress.ArtifactRef `json:"reference,omitempty"`
	Output    *cudaregress.ArtifactRef `json:"output,omitempty"`
	Error     string                   `json:"error,omitempty"`
}

// nativeVectors holds one attempt's encoded vectors until export. Output is
// nil unless the download completed.
type nativeVectors struct {
	index                    int
	requestID, attempt, err  string
	input, reference, output []byte
}

// nativeNumericsFor returns the retention record for cfg, whose other
// bounds must already be valid. It refuses an OutputDir configuration whose
// planned vectors exceed the bound, before any lease or device work.
func nativeNumericsFor(cfg NativeConfig) (NativeNumerics, error) {
	n := NativeNumerics{Operation: nativeOperation, Encoding: nativeEncoding}
	if cfg.OutputDir == "" {
		n.Retention = nativeRetentionNone
		return n, nil
	}
	total, err := nativeNumericBytes(cfg.Elements, cfg.Rounds, nativeMaxNumericBytes)
	if err != nil {
		return n, err
	}
	n.Retention = "pending"
	n.PlannedBytes = total
	return n, nil
}

// nativeNumericBytes returns the worst-case retained bytes for rounds of
// planned attempts of elements float32 values: an input, reference and
// output vector for each of 3*rounds attempts. It refuses a total above
// limit without overflowing.
func nativeNumericBytes(elements, rounds int, limit int64) (int64, error) {
	if elements < 0 || rounds < 0 || limit < 0 {
		return 0, errNativeNumericBound
	}
	total := int64(4)
	for _, n := range []int64{int64(elements), 3, int64(len(nativeArms)), int64(rounds)} {
		if n != 0 && total > limit/n {
			return 0, errNativeNumericBound
		}
		total *= n
	}
	return total, nil
}

// nativeReference computes the transform kernel's output for input on the
// CPU. Doubling is exact, so the single rounding is that of the addition.
func nativeReference(input []float32) []float32 {
	ref := make([]float32, len(input))
	for i, v := range input {
		ref[i] = float32(2*v) + 1
	}
	return ref
}

// encodeFloats returns values as little-endian binary32, preserving bits.
func encodeFloats(values []float32) []byte {
	b := make([]byte, 4*len(values))
	for i, v := range values {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(v))
	}
	return b
}

// recordAttempts appends the dispatched attempts, in plan order, to r.
func (r *NativeResult) recordAttempts(attempts []nativeAttempt) {
	for _, a := range attempts {
		r.Requests = append(r.Requests, a.log)
		r.Trials = append(r.Trials, a.trial)
		if a.annotation != nil {
			r.Annotations = append(r.Annotations, *a.annotation)
		}
		if a.incorrect {
			r.NumericalGate = "failed"
		}
		if a.vectors != nil {
			v := *a.vectors
			v.requestID, v.attempt, v.err = a.trial.RequestID, a.trial.Attempt, a.trial.Error
			r.vectors = append(r.vectors, v)
		}
	}
}

// createNativeFile creates path for writing. It refuses an existing file or
// symbolic link rather than overwriting or following it.
func createNativeFile(path string) (io.WriteCloser, error) {
	return os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0644)
}

// writeNativeFile writes data to a file from create. A failed or short
// write or a failed close is returned; the file is left in place.
func writeNativeFile(create func(path string) (io.WriteCloser, error), path string, data []byte) error {
	f, err := create(path)
	if err != nil {
		return err
	}
	n, writeErr := f.Write(data)
	if writeErr == nil && n != len(data) {
		writeErr = io.ErrShortWrite
	}
	return errors.Join(writeErr, f.Close())
}

// retainNativeVectors writes each vector that r's attempts constructed to its
// own file in dir, named by plan index and role, and records the references
// in r.Numerics. It attempts every file once and joins the failures, which
// leave the vector unreferenced and the retention incomplete.
func retainNativeVectors(dir string, r *NativeResult, create func(path string) (io.WriteCloser, error)) error {
	n := &r.Numerics
	n.Vectors = nil
	if len(r.vectors) == 0 {
		n.Retention = "none: no request attempt constructed vectors"
		return nil
	}
	var errs []error
	for _, v := range r.vectors {
		out := NativeVectors{Plan: v.index, RequestID: v.requestID, Attempt: v.attempt, Error: v.err}
		for _, f := range []struct {
			role string
			data []byte
			ref  **cudaregress.ArtifactRef
		}{{"input", v.input, &out.Input}, {"reference", v.reference, &out.Reference}, {"output", v.output, &out.Output}} {
			if f.data == nil {
				continue
			}
			name := fmt.Sprintf("numeric-%03d-%s.f32", v.index, f.role)
			path := filepath.Join(dir, name)
			if err := writeNativeFile(create, path, f.data); err != nil {
				errs = append(errs, fmt.Errorf("retain %s: %w", name, err))
				continue
			}
			*f.ref = &cudaregress.ArtifactRef{Name: name, Path: path, Digest: digestBytes(f.data), SizeBytes: int64(len(f.data)), Role: f.role}
		}
		n.Vectors = append(n.Vectors, out)
	}
	if err := errors.Join(errs...); err != nil {
		n.Retention = "incomplete"
		n.Error = err.Error()
		addNativeGate(r, "complete numerical vector retention")
		return err
	}
	n.Retention = "retained"
	return nil
}
