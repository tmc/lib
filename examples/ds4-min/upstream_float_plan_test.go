package main

import (
	"encoding/binary"
	"errors"
	"math"
	"os"
	"reflect"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
	"github.com/tmc/lib/examples/ds4-min/dsv4kvstore"
)

// TestUpstreamFloatPlanRecord checks a retained upstream payload against the
// Metal float-buffer shapes without opening a device or claiming continuation.
func TestUpstreamFloatPlanRecord(t *testing.T) {
	path := os.Getenv("DS4_UPSTREAM_FLOAT_PLAN_RECORD")
	if path == "" {
		t.Skip("retained upstream record required")
	}
	record, err := dsv4kvstore.ReadUpstreamPath(path, dsv4kvstore.UpstreamInspectOptions{MaxFileBytes: 2 << 30, MaxPayloadBytes: 2 << 30})
	if err != nil {
		t.Fatal(err)
	}
	ratios := make([]uint32, Flash.Layers)
	for i := 2; i < len(ratios); i++ {
		ratios[i] = 4
		if i%2 != 0 {
			ratios[i] = 128
		}
	}
	layout, err := dsv4kvstore.ValidateUpstreamRecordLayout(record, ratios)
	if err != nil {
		t.Fatal(err)
	}
	m := layout.Metadata
	layers := make([]upstreamFloatLayer, len(ratios))
	for i, ratio := range ratios {
		l := &layers[i]
		l.raw = make([]float32, 128*int(m.HeadDim))
		l.ratio = int(ratio)
		if ratio == 0 {
			continue
		}
		l.capacity = int(m.Context)/int(ratio) + 2
		l.cache = make([]float32, l.capacity*int(m.HeadDim))
		l.pooled = make([]float32, m.HeadDim)
		rows, width := int(ratio), int(m.HeadDim)
		if ratio == 4 {
			rows *= 2
			width *= 2
		}
		l.kv, l.score = make([]float32, rows*width), make([]float32, rows*width)
		if ratio == 4 {
			l.indexRatio, l.indexCapacity = 4, l.capacity
			l.indexCache = make([]float32, l.capacity*int(m.IndexerHeadDim))
			l.indexPooled = make([]float32, m.IndexerHeadDim)
			l.indexKV, l.indexScore = make([]float32, 8*2*int(m.IndexerHeadDim)), make([]float32, 8*2*int(m.IndexerHeadDim))
		}
	}
	writes, ns, indexNs, err := planUpstreamFloatRestore(record.Payload, layout, layers, make([]float32, m.Vocab))
	if err != nil {
		t.Fatal(err)
	}
	if err := commitUpstreamFloatRestore(writes, func(dst, v []float32) error { copy(dst, v); return nil }); err != nil {
		t.Fatal(err)
	}
	t.Logf("payload_sha256=%x tokens=%d layers=%d writes=%d counts=%d/%d host_only", record.PayloadSHA256, m.Tokens, m.Layers, len(writes), len(ns), len(indexNs))
}

func upstreamFloatPlanFixture() ([]byte, dsv4kvstore.UpstreamPayloadLayout, []upstreamFloatLayer, []float32) {
	m := dsv4kvstore.UpstreamPayloadMetadata{Context: 16, Tokens: 5, Layers: 1, HeadDim: 1, IndexerHeadDim: 1, Vocab: 2, RawLive: 2, RawCapacity: 2, RawWindow: 2}
	payload := make([]byte, 52+5*4+2*4+8)
	for i := 0; i < 5; i++ {
		binary.LittleEndian.PutUint32(payload[52+i*4:], uint32(i%2))
	}
	for i, v := range []float32{7, 8} {
		binary.LittleEndian.PutUint32(payload[72+i*4:], math.Float32bits(v))
	}
	binary.LittleEndian.PutUint32(payload[80:], 1)
	binary.LittleEndian.PutUint32(payload[84:], 1)
	layout := dsv4kvstore.UpstreamPayloadLayout{Metadata: m}
	for _, section := range []struct {
		name   string
		values []float32
	}{
		{"raw", []float32{11, 12}}, {"compressed", []float32{13}}, {"attention_state_kv", []float32{14, 15}}, {"attention_state_score", []float32{float32(math.Inf(-1)), 16}},
		{"indexer", []float32{17}}, {"index_state_kv", []float32{18, 19}}, {"index_state_score", []float32{20, 21}},
	} {
		start := len(payload)
		for _, v := range section.values {
			payload = binary.LittleEndian.AppendUint32(payload, math.Float32bits(v))
		}
		layout.Sections = append(layout.Sections, dsv4kvstore.UpstreamPayloadSection{Layer: 0, Name: section.name, Offset: uint64(start), Bytes: uint64(len(payload) - start)})
	}
	layers := []upstreamFloatLayer{{raw: []float32{99, 99}, cache: []float32{99, 99}, kv: []float32{99, 99}, score: []float32{99, 99}, pooled: []float32{99}, indexCache: []float32{99, 99}, indexKV: []float32{99, 99}, indexScore: []float32{99, 99}, indexPooled: []float32{99}, ratio: 4, indexRatio: 4, capacity: 2, indexCapacity: 2}}
	return payload, layout, layers, []float32{99, 99}
}

func TestUpstreamFloatPlanStagesRingAndScratch(t *testing.T) {
	payload, layout, layers, logits := upstreamFloatPlanFixture()
	writes, ns, indexNs, err := planUpstreamFloatRestore(payload, layout, layers, logits)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(layers[0].raw, []float32{99, 99}) || logits[0] != 99 {
		t.Fatal("staging changed live buffers")
	}
	if err := commitUpstreamFloatRestore(writes, func(dst, v []float32) error { copy(dst, v); return nil }); err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct{ got, want []float32 }{{layers[0].raw, []float32{12, 11}}, {layers[0].cache, []float32{13, 0}}, {layers[0].pooled, []float32{0}}, {layers[0].indexPooled, []float32{0}}, {logits, []float32{7, 8}}} {
		if !reflect.DeepEqual(test.got, test.want) {
			t.Fatalf("got %v, want %v", test.got, test.want)
		}
	}
	if ns[0] != 1 || indexNs[0] != 1 {
		t.Fatal("counts not restored")
	}
}

func TestUpstreamFloatPlanRejectsMutationsBeforeWrite(t *testing.T) {
	for _, name := range []string{"nan-logit", "infinite-kv", "count", "missing", "duplicate", "offset", "token", "shape"} {
		t.Run(name, func(t *testing.T) {
			payload, layout, layers, logits := upstreamFloatPlanFixture()
			switch name {
			case "nan-logit":
				binary.LittleEndian.PutUint32(payload[72:], math.Float32bits(float32(math.NaN())))
			case "infinite-kv":
				binary.LittleEndian.PutUint32(payload[layout.Sections[2].Offset:], math.Float32bits(float32(math.Inf(1))))
			case "count":
				binary.LittleEndian.PutUint32(payload[80:], 2)
			case "missing":
				layout.Sections = layout.Sections[1:]
			case "duplicate":
				layout.Sections = append(layout.Sections, layout.Sections[0])
			case "offset":
				layout.Sections[0].Offset = uint64(len(payload) + 1)
			case "token":
				binary.LittleEndian.PutUint32(payload[52:], 2)
			case "shape":
				layers[0].kv = layers[0].kv[:1]
			}
			if _, _, _, err := planUpstreamFloatRestore(payload, layout, layers, logits); err == nil {
				t.Fatal("accepted mutation")
			}
			if logits[0] != 99 || layers[0].raw[0] != 99 {
				t.Fatal("rejected input changed live buffers")
			}
		})
	}
}

func TestUpstreamFloatCommitRollback(t *testing.T) {
	for _, rollbackFails := range []bool{false, true} {
		dst := []float32{1, 2}
		writes := []upstreamFloatWrite{{dst: dst[:1], values: []float32{3}}, {dst: dst[1:], values: []float32{4}}}
		calls := 0
		err := commitUpstreamFloatRestore(writes, func(d, v []float32) error {
			calls++
			copy(d, v)
			if calls == 2 || rollbackFails && calls == 3 {
				return errors.New("injected write failure")
			}
			return nil
		})
		if err == nil || errors.Is(err, dsv4.ErrDesynchronized) != rollbackFails {
			t.Fatalf("rollback failure=%v: %v", rollbackFails, err)
		}
		if !reflect.DeepEqual(dst, []float32{1, 2}) {
			t.Fatalf("rollback data=%v", dst)
		}
	}
}
