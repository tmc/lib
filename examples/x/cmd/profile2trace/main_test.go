package main

import (
	"bytes"
	"encoding/binary"
	"io"
	"math"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

const fixtureSQL = `
CREATE TABLE StringIds(id INTEGER, value TEXT);
INSERT INTO StringIds VALUES(1,'gemm'),(2,'decode');
CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,deviceId INTEGER,contextId INTEGER,streamId INTEGER,globalPid INTEGER,correlationId INTEGER,demangledName INTEGER);
INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(1000000,2000000,0,1,7,123,1,1),(3000000,4000000,0,2,7,123,2,999);
CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,globalTid INTEGER,endGlobalTid INTEGER,text TEXT,textId INTEGER);
INSERT INTO NVTX_EVENTS VALUES(900000,4100000,1234,NULL,'prefill',NULL),(3000000,4000000,1234,1234,NULL,2),(4200000,NULL,1234,NULL,'mark',NULL);
`

func database(t *testing.T, sql string) string {
	t.Helper()
	if _, err := exec.LookPath("sqlite3"); err != nil {
		t.Fatal("sqlite3 is required for reader tests")
	}
	path := filepath.Join(t.TempDir(), "capture.sqlite")
	if out, err := exec.Command("sqlite3", path, sql).CombinedOutput(); err != nil {
		t.Fatalf("sqlite fixture: %s: %v", out, err)
	}
	return path
}

func TestNSYS(t *testing.T) {
	path := database(t, fixtureSQL)
	rows, err := readNSYS(path, wholeCapture(), 500000, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 4 || rows[0].name != "gemm" || rows[1].name != "kernel name unavailable" || rows[2].name != "prefill" || rows[3].name != "decode" {
		t.Fatalf("rows: %+v", rows)
	}
	if rows[0].thread == rows[1].thread {
		t.Fatal("distinct contexts collapsed")
	}
	var out bytes.Buffer
	if err := run([]string{"-nsys", path}, &out, io.Discard); err != nil {
		t.Fatal(err)
	}
	slices := decodeSlices(t, out.Bytes())
	if len(slices) != 4 || slices["gemm"] != 1000000 || slices["prefill"] != 3200000 {
		t.Fatalf("decoded durations: %v", slices)
	}
	for _, mutation := range []string{
		"DROP TABLE CUPTI_ACTIVITY_KIND_KERNEL;", "DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL;",
		"UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET end=start-1;", "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET start=NULL;",
		"UPDATE NVTX_EVENTS SET end=start-1 WHERE end IS NOT NULL;",
	} {
		t.Run(mutation, func(t *testing.T) {
			if _, err := readNSYS(database(t, fixtureSQL+mutation), wholeCapture(), 500000, io.Discard); err == nil {
				t.Fatal("accepted invalid capture")
			}
		})
	}
}

func TestNSYSDiagnostics(t *testing.T) {
	const diagnostics = `
CREATE TABLE DIAGNOSTIC_EVENT(timestamp INTEGER,severity INTEGER,text TEXT);
INSERT INTO DIAGNOSTIC_EVENT VALUES(0,1,'capture started'),(9000000,2,'CUDA events may be missing'),(9000001,3,'capture failed'),(9000002,99,'unknown diagnostic');
`
	const levels = `
CREATE TABLE ENUM_DIAGNOSTIC_SEVERITY_LEVEL(id INTEGER,name TEXT,label TEXT);
INSERT INTO ENUM_DIAGNOSTIC_SEVERITY_LEVEL VALUES(1,'Info','Info'),(2,'Warning','Warning'),(3,'Error','Error');
`
	for _, tt := range []struct {
		name, schema, warning string
		wantInfo              bool
	}{
		{"named levels", levels, "Nsight diagnostic (Warning): CUDA events may be missing", false},
		{"missing levels", "", "Nsight diagnostic (2): CUDA events may be missing", true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var want, got, log bytes.Buffer
			// The warning is outside this window, but concerns the whole capture.
			args := []string{"-nsys", database(t, fixtureSQL), "-nsys-window", "0:2500000"}
			if err := run(args, &want, io.Discard); err != nil {
				t.Fatal(err)
			}
			args[1] = database(t, fixtureSQL+diagnostics+tt.schema)
			if err := run(args, &got, &log); err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(got.Bytes(), want.Bytes()) {
				t.Fatal("diagnostics changed the trace")
			}
			for _, text := range []string{tt.warning, "capture failed", "unknown diagnostic"} {
				if !strings.Contains(log.String(), text) {
					t.Errorf("missing %q in log: %s", text, &log)
				}
			}
			if strings.Contains(log.String(), "capture started") != tt.wantInfo {
				t.Errorf("unexpected info diagnostic: %s", &log)
			}
		})
	}
}

func TestReaders(t *testing.T) {
	host := `{"schema":"gputrace.host-event/v1","kind":"interval","run_id":"test","id":"0","name":"prefill","clock_domain":"cpu_uptime_ns","timestamp_ns":1000,"duration_ns":250}`
	rows, err := readHost(strings.NewReader(host))
	if err != nil || len(rows) != 1 || rows[0].duration != 250 {
		t.Fatalf("host: %+v %v", rows, err)
	}
	for _, bad := range []string{"", strings.Replace(host, "cpu_uptime_ns", "unix", 1), strings.Replace(host, "250", "-1", 1), strings.Replace(host, "1000", "null", 1), host + strings.Replace(host, `"test"`, `"other"`, 1)} {
		if _, err := readHost(strings.NewReader(bad)); err == nil {
			t.Fatalf("accepted host %s", bad)
		}
	}
	metal := `{"regions":[{"region":"ffn","layer":1,"position":2048,"start_s":1,"end_s":1.002}]}`
	rows, err = readMetal(strings.NewReader(metal))
	if err != nil || len(rows) != 1 || rows[0].duration != 2000000 {
		t.Fatalf("metal: %+v %v", rows, err)
	}
	for _, bad := range []string{`{}`, strings.Replace(metal, "1.002", "0.9", 1), strings.Replace(metal, `"start_s":1`, `"start_s":null`, 1), metal + `{}`} {
		if _, err := readMetal(strings.NewReader(bad)); err == nil {
			t.Fatalf("accepted Metal %s", bad)
		}
	}
}

func TestTraceOverlap(t *testing.T) {
	rows := []slice{
		{name: "outer", process: "host", thread: "unknown", start: 100, duration: 100},
		{name: "cross", process: "host", thread: "unknown", start: 150, duration: 100},
		{name: "zero", process: "host", thread: "unknown", start: 250},
		{name: "next", process: "host", thread: "unknown", start: 250, duration: 10},
	}
	var out bytes.Buffer
	if err := writeTrace(&out, rows); err != nil {
		t.Fatal(err)
	}
	got := decodeSlices(t, out.Bytes())
	for _, row := range rows {
		if d, ok := got[row.name]; !ok || d != uint64(row.duration) {
			t.Fatalf("%s: %d, %v", row.name, d, ok)
		}
	}
	if err := writeTrace(errorWriter{}, rows); err == nil {
		t.Fatal("lost write error")
	}
	if err := run([]string{"-nsys", "missing", "-spans", "missing"}, io.Discard, io.Discard); err == nil || !strings.Contains(err.Error(), "offset") {
		t.Fatalf("clock check: %v", err)
	}
}

type errorWriter struct{}

func (errorWriter) Write([]byte) (int, error) { return 0, io.ErrClosedPipe }

// Decode protobuf independently and pair events per track to catch unit errors,
// overlapping intervals, and unmatched begin/end events.
func decodeSlices(t *testing.T, data []byte) map[string]uint64 {
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
	result := make(map[string]uint64)
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
			result[b.name] = ts - b.ts
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

func TestClockOffset(t *testing.T) {
	rows := []slice{{name: "host", start: 1000, duration: 10}, {name: "device", start: 1020, duration: 5}}
	if err := offset(rows, -2000); err != nil {
		t.Fatal(err)
	}
	if rows[0].start != -1000 || rows[0].args["source_start_ns"] != int64(1000) || rows[0].args["clock_offset_ns"] != int64(-2000) {
		t.Fatalf("offset: %+v", rows[0])
	}
	var out bytes.Buffer
	if err := writeTrace(&out, rows); err != nil {
		t.Fatal(err)
	}
	if got := decodeSlices(t, out.Bytes()); len(got) != 2 || got["host"] != 10 || got["device"] != 5 {
		t.Fatalf("offset changed intervals: %v", got)
	}
	for _, tt := range []struct{ start, delta int64 }{{1<<63 - 1, 1}, {-1 << 63, -1}} {
		if err := offset([]slice{{start: tt.start}}, tt.delta); err == nil {
			t.Fatal("accepted overflow")
		}
	}
	if err := writeTrace(io.Discard, []slice{{start: 1<<63 - 1, duration: 1}}); err == nil {
		t.Fatal("accepted interval overflow")
	}
}

func TestNSYSWindow(t *testing.T) {
	path := database(t, fixtureSQL)
	var w window
	if err := w.Set("2000000:"); err != nil {
		t.Fatal(err)
	}
	rows, err := readNSYS(path, w, 500000, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	// The kernel at 1000000 and the NVTX range opening at 900000 start before
	// the window; an interval is selected by its start, never by its overlap.
	if len(rows) != 2 || rows[0].category != "cuda_kernel" || rows[0].start != 3000000 || rows[1].name != "decode" {
		t.Fatalf("rows: %+v", rows)
	}
	if _, err := readNSYS(path, wholeCapture(), 1, io.Discard); err == nil {
		t.Fatal("accepted a window over the slice limit")
	}
	var empty window
	if err := empty.Set("5000000:6000000"); err != nil {
		t.Fatal(err)
	}
	if _, err := readNSYS(path, empty, 500000, io.Discard); err == nil {
		t.Fatal("accepted a window selecting no kernels")
	}
}

func TestWindowFlag(t *testing.T) {
	for _, bad := range []string{"", "1000", "abc:1", "1:xyz", "5:5", "9:1"} {
		var w window
		if err := w.Set(bad); err == nil {
			t.Errorf("Set(%q) = nil, want error", bad)
		}
	}
	var w window
	if err := w.Set(":42"); err != nil {
		t.Fatal(err)
	}
	if w.lo != math.MinInt64 || w.hi != 42 || w.String() != ":42" {
		t.Fatalf("window = %+v, String = %q", w, w.String())
	}
	if err := w.Set("7:"); err != nil {
		t.Fatal(err)
	}
	if w.lo != 7 || w.hi != math.MaxInt64 || w.String() != "7:" {
		t.Fatalf("window = %+v, String = %q", w, w.String())
	}
	if !wholeCapture().unbounded() || wholeCapture().String() != "whole capture" {
		t.Fatal("wholeCapture is not unbounded")
	}
}
