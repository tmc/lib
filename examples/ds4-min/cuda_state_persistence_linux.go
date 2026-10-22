//go:build linux

package main

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

const (
	cudaStateFormatVersion = 3
	cudaStateHeaderBytes   = 104
	cudaStateMaxBytes      = 1 << 30
	cudaStateFieldFloat    = 1
	cudaStateFieldByte     = 2
	cudaStateFieldUint32   = 3
	cudaStateFieldInt32    = 4
)

var cudaStateMagic = [8]byte{'D', 'S', '4', 'C', 'S', 'T', 'A', 'T'}

type cudaStateField struct {
	name       string
	kind       byte
	f          cudamem.Slice[float32]
	b          cudamem.Slice[byte]
	u          cudamem.Slice[uint32]
	i          cudamem.Slice[int32]
	hostFloats []float32
}

func cudaStateFields(s *cudaSessionState, hostLogits []float32) []cudaStateField {
	if s == nil {
		return nil
	}
	var fields []cudaStateField
	addF := func(name string, value cudamem.Slice[float32]) {
		fields = append(fields, cudaStateField{name: name, kind: cudaStateFieldFloat, f: value})
	}
	addB := func(name string, value cudamem.Slice[byte]) {
		fields = append(fields, cudaStateField{name: name, kind: cudaStateFieldByte, b: value})
	}
	addU := func(name string, value cudamem.Slice[uint32]) {
		fields = append(fields, cudaStateField{name: name, kind: cudaStateFieldUint32, u: value})
	}
	addI := func(name string, value cudamem.Slice[int32]) {
		fields = append(fields, cudaStateField{name: name, kind: cudaStateFieldInt32, i: value})
	}
	for i := range s.hc {
		addF(fmt.Sprintf("hc-%d", i), s.hc[i])
	}
	for _, item := range []struct {
		name  string
		value cudamem.Slice[float32]
	}{
		{"attention-pre", s.attnPre}, {"query-lora", s.qLora}, {"query-heads", s.qHeads},
		{"kv-raw", s.kvRaw}, {"kv-current", s.kvCur}, {"compressor-kv", s.compressorKV},
		{"compressor-gate", s.compressorGate}, {"indexer-kv", s.indexerKV}, {"indexer-gate", s.indexerGate},
		{"indexer-query", s.indexerQ}, {"indexer-weights", s.indexerWeights}, {"indexer-query-scratch", s.indexerScratch},
		{"indexer-scores", s.indexerScores}, {"raw-scores", s.rawScores}, {"raw-output", s.rawOutput},
		{"mixed-scores", s.mixedScores}, {"mixed-denominators", s.mixedDenominators}, {"mixed-output", s.mixedOutput},
		{"attention-low", s.attnLow}, {"attention-output", s.attnOut}, {"ffn-pre", s.ffnPre},
		{"router", s.router}, {"router-probabilities", s.probs}, {"expert-weights", s.expertW},
		{"shared-gate", s.sharedGate}, {"shared-up", s.sharedUp}, {"shared-mid", s.sharedMid},
		{"expert-gate", s.expertGate}, {"expert-up", s.expertUp}, {"expert-mid", s.expertMid},
		{"expert-down", s.expertDown}, {"ffn-blocks", s.ffnBlocks}, {"ffn-output", s.ffnOut},
		{"output-mixed", s.outputMixed}, {"logits", s.logits},
	} {
		addF(item.name, item.value)
	}
	addB("q80-input", s.q80Input)
	addB("q81-input", s.q81Input)
	addB("q81-experts", s.q81Experts)
	if s.q8KInput.Len() != 0 {
		addB("q8k-input", s.q8KInput)
		addB("q8k-experts", s.q8KExperts)
	}
	addU("allowed-rows", s.allowed)
	addI("expert-ids", s.expertIDs)
	for i := range s.layers {
		addF(fmt.Sprintf("layer-%d-raw", i), s.layers[i].rawKV)
		addF(fmt.Sprintf("layer-%d-attention-cache", i), s.layers[i].attn.cache)
		addF(fmt.Sprintf("layer-%d-attention-kv", i), s.layers[i].attn.stateKV)
		addF(fmt.Sprintf("layer-%d-attention-scores", i), s.layers[i].attn.stateScore)
		addF(fmt.Sprintf("layer-%d-attention-pooled", i), s.layers[i].attn.pooled)
		addF(fmt.Sprintf("layer-%d-indexer-cache", i), s.layers[i].indexer.cache)
		addF(fmt.Sprintf("layer-%d-indexer-kv", i), s.layers[i].indexer.stateKV)
		addF(fmt.Sprintf("layer-%d-indexer-scores", i), s.layers[i].indexer.stateScore)
		addF(fmt.Sprintf("layer-%d-indexer-pooled", i), s.layers[i].indexer.pooled)
	}
	fields = append(fields, cudaStateField{name: "host-logits", kind: cudaStateFieldFloat, hostFloats: append([]float32(nil), hostLogits...)})
	return fields
}

func cudaStateFieldLength(f cudaStateField) int {
	if f.hostFloats != nil {
		return len(f.hostFloats)
	}
	switch f.kind {
	case cudaStateFieldFloat:
		return f.f.Len()
	case cudaStateFieldByte:
		return f.b.Len()
	case cudaStateFieldUint32:
		return f.u.Len()
	case cudaStateFieldInt32:
		return f.i.Len()
	}
	return -1
}

func cudaStateFieldBytes(f cudaStateField) ([]byte, error) {
	n := cudaStateFieldLength(f)
	if n < 0 {
		return nil, errors.New("cuda state: invalid field kind")
	}
	switch f.kind {
	case cudaStateFieldFloat:
		out := make([]byte, n*4)
		values := f.hostFloats
		if values == nil {
			values = make([]float32, n)
			if err := f.f.CopyToHost(values); err != nil {
				return nil, err
			}
		}
		for i, value := range values {
			binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(value))
		}
		return out, nil
	case cudaStateFieldByte:
		out := make([]byte, n)
		if err := f.b.CopyToHost(out); err != nil {
			return nil, err
		}
		return out, nil
	case cudaStateFieldUint32:
		values := make([]uint32, n)
		if err := f.u.CopyToHost(values); err != nil {
			return nil, err
		}
		out := make([]byte, n*4)
		for i, value := range values {
			binary.LittleEndian.PutUint32(out[i*4:], value)
		}
		return out, nil
	case cudaStateFieldInt32:
		values := make([]int32, n)
		if err := f.i.CopyToHost(values); err != nil {
			return nil, err
		}
		out := make([]byte, n*4)
		for i, value := range values {
			binary.LittleEndian.PutUint32(out[i*4:], uint32(value))
		}
		return out, nil
	}
	return nil, errors.New("cuda state: invalid field kind")
}

func writeCUDAStateField(w *bytes.Buffer, f cudaStateField, data []byte) error {
	if len(f.name) > math.MaxUint16 {
		return errors.New("cuda state: field name too long")
	}
	if len(data) > cudaStateMaxBytes {
		return errors.New("cuda state: field is too large")
	}
	var head [20]byte
	binary.LittleEndian.PutUint16(head[0:2], uint16(len(f.name)))
	head[2] = f.kind
	binary.LittleEndian.PutUint64(head[4:12], uint64(cudaStateFieldLength(f)))
	binary.LittleEndian.PutUint64(head[12:20], uint64(len(data)))
	if _, err := w.Write(head[:]); err != nil {
		return err
	}
	if _, err := w.WriteString(f.name); err != nil {
		return err
	}
	_, err := w.Write(data)
	return err
}

func saveCUDAState(s *cudaRequestState, w io.Writer) (err error) {
	if s == nil || s.closed || s.runtime == nil {
		return dsv4.ErrClosed
	}
	if w == nil {
		return errors.New("cuda state save: nil writer")
	}
	if s.runtime.failed != nil {
		return s.runtime.failed
	}
	model, quant, err := s.runtime.cudaFoundation.nativeStateIdentity()
	if err != nil {
		return fmt.Errorf("cuda state save identity: %w", err)
	}
	snapshot, err := s.Snapshot()
	if err != nil {
		return fmt.Errorf("cuda state save snapshot: %w", err)
	}
	defer func() { err = finishBackendCleanup(err, snapshot) }()
	v := snapshot.(*cudaStateSnapshot)
	return writeCUDAStateSnapshot(w, v, model, quant)
}

func writeCUDAStateSnapshot(w io.Writer, v *cudaStateSnapshot, model, quant [32]byte) error {
	if v == nil || v.closed || v.closing {
		return dsv4.ErrClosed
	}
	if v.dspark != nil {
		if v.supportModel == ([32]byte{}) || v.supportQuant == ([32]byte{}) {
			return errors.New("cuda support snapshot: missing frozen core identity")
		}
		return writeCUDASupportSnapshot(w, v, v.supportModel, v.supportQuant)
	}
	return writeCUDAStateCoreSnapshot(w, v, model, quant)
}

// The core payload stays byte-for-byte compatible with the legacy serializer.
func writeCUDAStateCoreSnapshot(w io.Writer, v *cudaStateSnapshot, model, quant [32]byte) error {
	fields := cudaStateFields(&v.state, v.logits)
	var body bytes.Buffer
	for _, field := range fields {
		data, err := cudaStateFieldBytes(field)
		if err != nil {
			return fmt.Errorf("cuda state save %s: %w", field.name, err)
		}
		if err := writeCUDAStateField(&body, field, data); err != nil {
			return err
		}
		if body.Len() > cudaStateMaxBytes {
			return errors.New("cuda state save: state exceeds size limit")
		}
	}
	var header [cudaStateHeaderBytes]byte
	copy(header[:8], cudaStateMagic[:])
	binary.LittleEndian.PutUint32(header[8:12], cudaStateFormatVersion)
	binary.LittleEndian.PutUint64(header[12:20], uint64(v.pos))
	binary.LittleEndian.PutUint32(header[20:24], uint32(len(fields)))
	binary.LittleEndian.PutUint64(header[24:32], uint64(body.Len()))
	binary.LittleEndian.PutUint32(header[32:36], uint32(len(v.state.layers)))
	copy(header[40:72], model[:])
	copy(header[72:104], quant[:])
	var meta bytes.Buffer
	meta.Write(header[:])
	steering, err := v.steering.encode()
	if err != nil {
		return fmt.Errorf("cuda state save steering: %w", err)
	}
	meta.Write(steering[:])
	for _, layer := range v.state.layers {
		var n [8]byte
		binary.LittleEndian.PutUint32(n[0:4], uint32(layer.attn.n))
		binary.LittleEndian.PutUint32(n[4:8], uint32(layer.indexer.n))
		meta.Write(n[:])
	}
	meta.Write(body.Bytes())
	if meta.Len() > cudaStateMaxBytes {
		return errors.New("cuda state save: state exceeds size limit")
	}
	if err := writeSessionFull(w, meta.Bytes()); err != nil {
		return fmt.Errorf("cuda state save: %w", err)
	}
	return nil
}

type cudaDecodedField struct {
	name  string
	kind  byte
	elems uint64
	data  []byte
}

func validateCUDAStateLayerCounts(s cudaSessionState, attnN, indexerN []int) error {
	if len(attnN) != len(s.layers) || len(indexerN) != len(s.layers) {
		return errors.New("cuda state load: layer count length mismatch")
	}
	for i, layer := range s.layers {
		if attnN[i] < 0 || attnN[i] > layer.attn.cap {
			return fmt.Errorf("cuda state load: layer %d attention count %d exceeds capacity %d", i, attnN[i], layer.attn.cap)
		}
		if indexerN[i] < 0 || indexerN[i] > layer.indexer.cap {
			return fmt.Errorf("cuda state load: layer %d indexer count %d exceeds capacity %d", i, indexerN[i], layer.indexer.cap)
		}
	}
	return nil
}

func readCUDAStateFields(data []byte, count int) ([]cudaDecodedField, int, error) {
	fields := make([]cudaDecodedField, 0, count)
	off := 0
	for i := 0; i < count; i++ {
		if len(data)-off < 20 {
			return nil, 0, errors.New("cuda state load: truncated field header")
		}
		nameLen := int(binary.LittleEndian.Uint16(data[off:]))
		kind := data[off+2]
		elems := binary.LittleEndian.Uint64(data[off+4:])
		bytesLen := binary.LittleEndian.Uint64(data[off+12:])
		off += 20
		if nameLen > 255 || bytesLen > uint64(cudaStateMaxBytes) || uint64(nameLen)+bytesLen > uint64(len(data)-off) {
			return nil, 0, errors.New("cuda state load: invalid field size")
		}
		name := string(data[off : off+nameLen])
		off += nameLen
		payload := append([]byte(nil), data[off:off+int(bytesLen)]...)
		off += int(bytesLen)
		fields = append(fields, cudaDecodedField{name: name, kind: kind, elems: elems, data: payload})
	}
	return fields, off, nil
}

func applyCUDAField(f cudaStateField, data []byte) error {
	want := cudaStateFieldLength(f)
	if want < 0 {
		return errors.New("cuda state load: invalid field kind")
	}
	if f.kind == cudaStateFieldByte {
		if len(data) != want {
			return errors.New("cuda state load: field byte length mismatch")
		}
	} else if len(data) != want*4 {
		return errors.New("cuda state load: field byte length mismatch")
	}
	switch f.kind {
	case cudaStateFieldFloat:
		values := make([]float32, len(data)/4)
		for i := range values {
			values[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
		}
		if f.hostFloats != nil {
			copy(f.hostFloats, values)
			return nil
		}
		return f.f.CopyFromHost(values)
	case cudaStateFieldByte:
		return f.b.CopyFromHost(data)
	case cudaStateFieldUint32:
		values := make([]uint32, len(data)/4)
		for i := range values {
			values[i] = binary.LittleEndian.Uint32(data[i*4:])
		}
		return f.u.CopyFromHost(values)
	case cudaStateFieldInt32:
		values := make([]int32, len(data)/4)
		for i := range values {
			values[i] = int32(binary.LittleEndian.Uint32(data[i*4:]))
		}
		return f.i.CopyFromHost(values)
	}
	return errors.New("cuda state load: invalid field kind")
}

func loadCUDAState(s *cudaRequestState, r io.Reader) error {
	return loadCUDAStateWithHooks(s, r, nil)
}

// cudaStateLoadHooks is a persistence-only test seam. The callback runs after
// the field's actual device upload; production Load always passes nil.
type cudaStateLoadHooks struct {
	afterUpload    func(int) error
	beforeRollback func() error
}

func loadCUDAStateWithHooks(s *cudaRequestState, r io.Reader, hooks *cudaStateLoadHooks) error {
	if s == nil || s.closed || s.runtime == nil {
		return dsv4.ErrClosed
	}
	if r == nil {
		return errors.New("cuda state load: nil reader")
	}
	data, err := io.ReadAll(io.LimitReader(r, cudaStateMaxBytes+1))
	if err != nil {
		return fmt.Errorf("cuda state load: read: %w", err)
	}
	if len(data) > cudaStateMaxBytes || len(data) < cudaStateHeaderBytes {
		return errors.New("cuda state load: invalid size")
	}
	if !bytes.Equal(data[:8], cudaStateMagic[:]) {
		return errors.New("cuda state load: invalid format")
	}
	headerBytes, err := cudaStateSteeringHeader(data, s.runtime.steeringStateIdentity())
	if err != nil {
		return err
	}
	if s.runtime.failed != nil {
		return s.runtime.failed
	}
	model, quant, err := s.runtime.cudaFoundation.nativeStateIdentity()
	if err != nil {
		return fmt.Errorf("cuda state load identity: %w", err)
	}
	if err := validateCUDAStateIdentity(data, model, quant); err != nil {
		return err
	}
	pos := binary.LittleEndian.Uint64(data[12:20])
	count := int(binary.LittleEndian.Uint32(data[20:24]))
	bodyBytes := binary.LittleEndian.Uint64(data[24:32])
	layers := int(binary.LittleEndian.Uint32(data[32:36]))
	if layers != len(s.runtime.state.layers) || layers > 512 || pos > uint64(s.runtime.ctx) || count < 1 || count > 512 {
		return errors.New("cuda state load: invalid header")
	}
	metaBytes := headerBytes + 8*layers
	if metaBytes > len(data) || bodyBytes > uint64(len(data)-metaBytes) || uint64(metaBytes)+bodyBytes != uint64(len(data)) {
		return errors.New("cuda state load: invalid header")
	}
	off := headerBytes
	attnN := make([]int, layers)
	indexerN := make([]int, layers)
	for i := range attnN {
		if len(data)-off < 8 {
			return errors.New("cuda state load: truncated layer metadata")
		}
		attnN[i] = int(binary.LittleEndian.Uint32(data[off:]))
		indexerN[i] = int(binary.LittleEndian.Uint32(data[off+4:]))
		off += 8
	}
	if err := validateCUDAStateLayerCounts(s.runtime.state, attnN, indexerN); err != nil {
		return err
	}
	fields, used, err := readCUDAStateFields(data[off:], count)
	if err != nil || uint64(used) != bodyBytes {
		if err == nil {
			err = errors.New("field bytes do not match header")
		}
		return err
	}
	expected := cudaStateFields(&s.runtime.state, s.runtime.logits)
	if len(fields) != len(expected) {
		return errors.New("cuda state load: field count mismatch")
	}
	for i := range expected {
		want := cudaStateFieldLength(expected[i])
		wantBytes := want * 4
		if expected[i].kind == cudaStateFieldByte {
			wantBytes = want
		}
		if fields[i].name != expected[i].name || fields[i].kind != expected[i].kind || fields[i].elems != uint64(want) || len(fields[i].data) != wantBytes {
			return fmt.Errorf("cuda state load: field %q shape mismatch", fields[i].name)
		}
	}
	var hostLogits []float32
	for _, field := range expected {
		if field.name == "host-logits" {
			hostLogits = field.hostFloats
			break
		}
	}
	if len(hostLogits) != len(s.runtime.logits) {
		return errors.New("cuda state load: host logits missing or wrong length")
	}
	backup, err := s.Snapshot()
	if err != nil {
		return fmt.Errorf("cuda state load backup: %w", err)
	}
	defer backup.Close()
	err = cudaStateTransaction(func() error {
		for i := range expected {
			if err := applyCUDAField(expected[i], fields[i].data); err != nil {
				return fmt.Errorf("cuda state load %s: %w", expected[i].name, err)
			}
			if hooks != nil && hooks.afterUpload != nil {
				if err := hooks.afterUpload(i + 1); err != nil {
					return err
				}
			}
		}
		return nil
	}, func() error {
		if hooks != nil && hooks.beforeRollback != nil {
			if err := hooks.beforeRollback(); err != nil {
				return err
			}
		}
		return s.restoreSnapshot(backup.(*cudaStateSnapshot))
	}, func(err error) {
		s.runtime.failed = err
		s.runtime.headObserver.invalidate()
	})
	if err != nil {
		return err
	}
	for i := range s.runtime.state.layers {
		s.runtime.state.layers[i].attn.n = attnN[i]
		s.runtime.state.layers[i].indexer.n = indexerN[i]
	}
	// cudaStateFields owns its host-logit slice while parsing and uploading.
	// Publish it only after every device field was installed successfully.
	copy(s.runtime.logits, hostLogits)
	s.runtime.pos = int(pos)
	s.runtime.failed = nil
	return nil
}
