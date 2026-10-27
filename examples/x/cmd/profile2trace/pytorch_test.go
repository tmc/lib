package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/big"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
)

func pytorchFixture(t *testing.T, name string) []byte {
	t.Helper()
	b, err := os.ReadFile("../../../../nvidia/x/nvtelemetry/testdata/" + name)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func TestPyTorchAuthentic(t *testing.T) {
	tests := []struct {
		name, digest string
		count        int
	}{
		{"pytorch-2.10.0-cpu-eager.chrome.json", "da43e72ecac5387566568d1f139af9b4aaefa3cb5861ce11ff5028e6e96cad47", 2},
		{"pytorch-2.10.0-cpu-inductor.chrome.json", "39a23a35df382fc29e27e3781849810c74d66d294ed3eac4a9bed401486f01ad", 4},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			raw := pytorchFixture(t, tt.name)
			if fmt.Sprintf("%x", sha256.Sum256(raw)) != tt.digest {
				t.Fatal("authentic fixture changed")
			}
			var trace struct {
				Events []struct {
					Name, Cat, Ph  string
					TS             json.Number `json:"ts"`
					Dur            json.Number `json:"dur"`
					PID, TID, Args json.RawMessage
				} `json:"traceEvents"`
			}
			if err := json.Unmarshal(raw, &trace); err != nil {
				t.Fatal(err)
			}
			rows, err := readPyTorch(context.Background(), bytes.NewReader(raw), "2.10.0", tt.digest, 10)
			if err != nil {
				t.Fatal(err)
			}
			if len(rows) != tt.count {
				t.Fatalf("annotations %d, want %d", len(rows), tt.count)
			}
			ns := func(n json.Number) int64 {
				r, ok := new(big.Rat).SetString(string(n))
				if !ok {
					t.Fatal(n)
				}
				r.Mul(r, big.NewRat(1000, 1))
				if !r.IsInt() || !r.Num().IsInt64() {
					t.Fatal(r)
				}
				return r.Num().Int64()
			}
			want := make(map[string][]uint64)
			i := 0
			for _, e := range trace.Events {
				if e.Cat != "user_annotation" {
					continue
				}
				row := rows[i]
				if e.Ph != "X" || row.name != e.Name || row.start != ns(e.TS) || row.duration != ns(e.Dur) {
					t.Fatalf("event %d: %+v", i, row)
				}
				for k, v := range map[string]any{
					"source_report_sha256": tt.digest, "source_annotation_ordinal": i,
					"source_pid_json": string(e.PID), "source_tid_json": string(e.TID), "source_args_json": string(e.Args),
					"source_start_raw": string(e.TS), "source_duration_raw": string(e.Dur), "source_time_unit": "microseconds",
					"native_membership": "unavailable", "source_clock": "unestablished", "capture_completeness": "unestablished",
				} {
					if row.args[k] != v {
						t.Fatalf("event %d %s = %v, want %v", i, k, row.args[k], v)
					}
				}
				if row.process != "pytorch:"+tt.digest+"/pid:"+string(e.PID) || row.thread != "tid:"+string(e.TID) {
					t.Fatal("trace-local identity lost")
				}
				want[e.Name] = append(want[e.Name], uint64(ns(e.Dur)))
				i++
			}
			var out, log bytes.Buffer
			file := filepath.Join(t.TempDir(), "capture.json")
			if err := os.WriteFile(file, raw, 0600); err != nil {
				t.Fatal(err)
			}
			if err := runContext(context.Background(), []string{"-pytorch", file, "-pytorch-version", "2.10.0", "-pytorch-sha256", tt.digest}, &out, &log, sqliteCommand); err != nil {
				t.Fatal(err)
			}
			got := decodePyTorchSlices(t, out.Bytes())
			for _, values := range got {
				sort.Slice(values, func(i, j int) bool { return values[i] < values[j] })
			}
			for _, values := range want {
				sort.Slice(values, func(i, j int) bool { return values[i] < values[j] })
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("protobuf slices %v, want %v", got, want)
			}
			if !bytes.Contains(out.Bytes(), []byte(tt.digest)) || !bytes.Contains(out.Bytes(), []byte("source_annotation_ordinal")) || !strings.Contains(log.String(), "other source events not transformed") {
				t.Fatal("transform provenance missing")
			}
		})
	}
}

func TestPyTorchRefusals(t *testing.T) {
	raw := []byte(`{"schemaVersion":1,"deviceProperties":[],"traceEvents":[{"name":"x","cat":"user_annotation","ph":"X","ts":1,"dur":2,"pid":1,"tid":2}]}`)
	for _, tt := range []struct {
		name, version, old, new string
		max                     int
	}{
		{"version", "2.7.0", "", "", 2},
		{"schema", "2.10.0", `"schemaVersion":1`, `"schemaVersion":2`, 2},
		{"device", "2.10.0", `"deviceProperties":[]`, `"deviceProperties":[{}]`, 2},
		{"native category", "2.10.0", `"user_annotation"`, `"kernel"`, 2},
		{"overflow", "2.10.0", `"ts":1`, `"ts":9223372036854775807`, 2},
		{"fractional ns", "2.10.0", `"ts":1`, `"ts":0.0001`, 2},
		{"slice limit", "2.10.0", "", "", 0},
	} {
		t.Run(tt.name, func(t *testing.T) {
			b := raw
			if tt.old != "" {
				b = bytes.ReplaceAll(b, []byte(tt.old), []byte(tt.new))
			}
			digest := fmt.Sprintf("%x", sha256.Sum256(b))
			if _, err := readPyTorch(context.Background(), bytes.NewReader(b), tt.version, digest, tt.max); err == nil {
				t.Fatal("accepted invalid profile")
			}
		})
	}
	t.Run("digest", func(t *testing.T) {
		if _, err := readPyTorch(context.Background(), bytes.NewReader(raw), "2.10.0", strings.Repeat("0", 64), 2); err == nil {
			t.Fatal("accepted wrong digest")
		}
	})
	t.Run("bytes", func(t *testing.T) {
		b := bytes.Repeat([]byte(" "), 4<<20+1)
		if _, err := readPyTorch(context.Background(), bytes.NewReader(b), "2.10.0", fmt.Sprintf("%x", sha256.Sum256(b)), 2); err == nil {
			t.Fatal("accepted oversized input")
		}
	})
	t.Run("mixed clocks", func(t *testing.T) {
		var out bytes.Buffer
		err := runContext(context.Background(), []string{"-pytorch", "unused", "-spans", "unused"}, &out, io.Discard, sqliteCommand)
		if err == nil || out.Len() != 0 {
			t.Fatal("mixed input accepted")
		}
	})
}

func TestPyTorchCancellationAndWriter(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := readPyTorch(ctx, strings.NewReader(""), "2.10.0", strings.Repeat("0", 64), 1); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	raw := pytorchFixture(t, "pytorch-2.10.0-cpu-eager.chrome.json")
	rows, err := readPyTorch(context.Background(), bytes.NewReader(raw), "2.10.0", fmt.Sprintf("%x", sha256.Sum256(raw)), 10)
	if err != nil {
		t.Fatal(err)
	}
	if err := writeTrace(errorWriter{}, rows); !errors.Is(err, io.ErrClosedPipe) {
		t.Fatal(err)
	}
}

func decodePyTorchSlices(t *testing.T, data []byte) map[string][]uint64 {
	t.Helper()
	type field struct {
		n int
		v uint64
		b []byte
	}
	fields := func(data []byte) []field {
		var out []field
		for len(data) > 0 {
			tag, n := binary.Uvarint(data)
			if n <= 0 {
				t.Fatal("bad tag")
			}
			data = data[n:]
			f := field{n: int(tag >> 3)}
			v, n := binary.Uvarint(data)
			if n <= 0 {
				t.Fatal("bad value")
			}
			data = data[n:]
			switch tag & 7 {
			case 0:
				f.v = v
			case 2:
				if v > uint64(len(data)) {
					t.Fatal("short field")
				}
				f.b = data[:v]
				data = data[v:]
			default:
				t.Fatalf("unexpected wire type %d", tag&7)
			}
			out = append(out, f)
		}
		return out
	}
	type begin struct {
		name string
		ts   uint64
	}
	active := make(map[uint64]begin)
	result := make(map[string][]uint64)
	for _, packet := range fields(data) {
		if packet.n != 1 {
			continue
		}
		var ts uint64
		var ev []byte
		for _, f := range fields(packet.b) {
			if f.n == 8 {
				ts = f.v
			}
			if f.n == 11 {
				ev = f.b
			}
		}
		if ev == nil {
			continue
		}
		var kind, track uint64
		var name string
		for _, f := range fields(ev) {
			switch f.n {
			case 9:
				kind = f.v
			case 11:
				track = f.v
			case 23:
				name = string(f.b)
			}
		}
		switch kind {
		case 1:
			if _, ok := active[track]; ok {
				t.Fatal("overlapping slices on one track")
			}
			active[track] = begin{name, ts}
		case 2:
			b, ok := active[track]
			if !ok || ts < b.ts {
				t.Fatal("unmatched end")
			}
			result[b.name] = append(result[b.name], ts-b.ts)
			delete(active, track)
		default:
			t.Fatalf("unexpected event %d", kind)
		}
	}
	if len(active) != 0 {
		t.Fatal("unclosed slices")
	}
	return result
}
