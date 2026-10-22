//go:build linux

package main

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"

	"github.com/tmc/lib/nvidia/x/cudamem"
)

// encodeObservedOutputHead preserves the ordinary head path when disabled.
// The diagnostic path copies live operands before scratch can be reused. It
// intentionally synchronizes and is not suitable for performance captures.
func (r *cudaRuntime) encodeObservedOutputHead(mixed, out cudamem.Slice[float32], position int) error {
	t := r.model.output.weight
	if mixed.Len() != 8192 {
		return fmt.Errorf("head observer: invalid mixed input extent")
	}
	in := mixed.Slice(4096, 8192)
	prefix := os.Getenv("DS4_HEAD_OBSERVER_PREFIX")
	if prefix == "" {
		return r.encodeQ80Matvec(t, r.denseOutput, in, out)
	}
	target, err := strconv.Atoi(os.Getenv("DS4_HEAD_OBSERVER_POSITION"))
	if err != nil || target < 0 {
		return fmt.Errorf("head observer: require nonnegative DS4_HEAD_OBSERVER_POSITION")
	}
	if position != target {
		return r.encodeQ80Matvec(t, r.denseOutput, in, out)
	}
	history, err := r.headObserver.hash(position)
	if err != nil {
		return err
	}
	model := os.Getenv("DS4_HEAD_OBSERVER_MODEL_SHA256")
	if len(model) != 64 {
		return fmt.Errorf("head observer: require externally verified model checksum assertion")
	}
	source, err := os.ReadFile(os.Getenv("DS4_HEAD_OBSERVER_SOURCE_MANIFEST"))
	if err != nil {
		return fmt.Errorf("head observer: read source manifest: %w", err)
	}
	executable, err := os.Executable()
	if err != nil {
		return err
	}
	binary, err := os.ReadFile(executable)
	if err != nil {
		return fmt.Errorf("head observer: hash executable: %w", err)
	}
	if mixed.Len() != 8192 || out.Len() != 129280 || t.typeID != 8 ||
		len(t.dimensions) != 2 || t.dimensions[0] != 4096 || t.dimensions[1] != 129280 || t.bytes < 2<<20 {
		return fmt.Errorf("head observer: unsupported head shape")
	}
	q := r.state.q80Input.Slice(0, 128*36)
	if err := r.encodeQuantQ81(in, q, 4096); err != nil {
		return err
	}
	if err := syncStream(r.stream()); err != nil {
		return err
	}
	values := make([]float32, 8192)
	if err := mixed.CopyToHost(values); err != nil {
		return fmt.Errorf("head observer: copy mixed input: %w", err)
	}
	codes := make([]byte, q.Len())
	if err := q.CopyToHost(codes); err != nil {
		return fmt.Errorf("head observer: copy live q8_1: %w", err)
	}
	for _, stage := range []struct {
		name string
		data []byte
	}{
		{".hc.f32", float32Bytes(values[:4096])},
		{".norm.f32", float32Bytes(values[4096:])},
		{".q81.bin", codes},
	} {
		if err := writeHeadSnapshot(prefix+stage.name, stage.data); err != nil {
			return err
		}
	}
	route := "raw-q8_0-q8_1"
	if r.denseOutput.Len() > 0 && r.denseArtifactMatvecEnabled() {
		route = "aligned-q8_0-q8_1"
		if err := r.encodeQ80MatvecQ81Aligned(r.denseOutput, t.dimensions[0], t.dimensions[1], q, out); err != nil {
			return err
		}
	} else {
		if err := r.encodeQ80MatvecQ81(t, q, out); err != nil {
			return err
		}
	}
	if err := syncStream(r.stream()); err != nil {
		return err
	}
	logits := make([]float32, out.Len())
	if err := out.CopyToHost(logits); err != nil {
		return fmt.Errorf("head observer: copy logits: %w", err)
	}
	data := float32Bytes(logits)
	if err := writeHeadSnapshot(prefix+".logits.f32", data); err != nil {
		return err
	}
	type stage struct {
		File string `json:"file"`
		SHA  string `json:"sha256"`
	}
	stages := make(map[string]stage)
	for _, s := range []struct {
		name, suffix string
		data         []byte
	}{
		{"hc", ".hc.f32", float32Bytes(values[:4096])},
		{"norm", ".norm.f32", float32Bytes(values[4096:])},
		{"q81", ".q81.bin", codes},
		{"logits", ".logits.f32", data},
	} {
		stages[s.name] = stage{filepath.Base(prefix + s.suffix), fmt.Sprintf("%x", sha256.Sum256(s.data))}
	}
	packet := struct {
		Version     int              `json:"version"`
		Model       string           `json:"model_sha256"`
		History     string           `json:"history_sha256"`
		Source      string           `json:"source_sha256"`
		Binary      string           `json:"binary_sha256"`
		Position    int              `json:"position"`
		Context     int              `json:"context"`
		Route       string           `json:"route"`
		ModelBasis  string           `json:"model_sha256_basis"`
		SourceBasis string           `json:"source_sha256_basis"`
		BinaryBasis string           `json:"binary_sha256_basis"`
		Stages      map[string]stage `json:"stages"`
	}{1, model, history, fmt.Sprintf("%x", sha256.Sum256(source)), fmt.Sprintf("%x", sha256.Sum256(binary)), position, r.ctx, route, "external-receipt-assertion", "actual-source-manifest-file", "actual-executable-file", stages}
	metadata, err := json.Marshal(packet)
	if err != nil {
		return err
	}
	return writeHeadSnapshot(prefix+".json", append(metadata, '\n'))
}
