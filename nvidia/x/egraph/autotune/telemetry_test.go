package autotune

import (
	"bytes"
	"encoding/json"
	"testing"
	"time"
)

func TestTelemetryChromeTraceJSONExport(t *testing.T) {
	rec := NewFlightTelemetryRecorder()
	t0 := time.Date(2026, 10, 31, 9, 0, 0, 0, time.UTC)

	rec.RecordStageSpan("l0_post_gate_up", 1, t0, 42*time.Microsecond, 0, 16)
	rec.RecordPCStallAttribution("l0_post_gate_up", t0.Add(20*time.Microsecond), 180, 45, 30, 300)
	rec.RecordPTXASIteration("cand_rpb16_u2", 1, t0.Add(50*time.Microsecond), 120*time.Microsecond, 96, 0)
	rec.RecordHoldoutAndHotSwap("cand_rpb16_u2", 2, t0.Add(200*time.Microsecond), 310*time.Microsecond, 1.084, 0.015625, true)
	rec.RecordBucketTransition(t0.Add(550*time.Microsecond), 1, 4, 0.78)
	rec.RecordBucketTransition(t0.Add(620*time.Microsecond), 4, 8, 0.89)

	var buf bytes.Buffer
	if err := rec.ExportChromeTraceJSON(&buf); err != nil {
		t.Fatalf("ExportChromeTraceJSON: %v", err)
	}

	var doc ChromeTraceDocument
	if err := json.Unmarshal(buf.Bytes(), &doc); err != nil {
		t.Fatalf("json.Unmarshal ChromeTraceDocument: %v", err)
	}
	if len(doc.TraceEvents) != 7 {
		t.Fatalf("expected 7 trace events, got %d", len(doc.TraceEvents))
	}

	var (
		completeCount int
		instantCount  int
		sawStallArgs  bool
		sawPTXASArgs  bool
		sawBucketM1M4 bool
		sawBucketM4M8 bool
	)
	for _, ev := range doc.TraceEvents {
		switch ev.Ph {
		case "X":
			completeCount++
			if ev.Dur <= 0 {
				t.Fatalf("complete event %q has non-positive dur=%v", ev.Name, ev.Dur)
			}
		case "i":
			instantCount++
		default:
			t.Fatalf("unexpected event phase %q in %+v", ev.Ph, ev)
		}

		if ev.Cat == "cupti.pcsampling" {
			if _, ok := ev.Args["stall_long_score"]; ok {
				if _, ok2 := ev.Args["stall_barrier"]; ok2 {
					if _, ok3 := ev.Args["stall_mio"]; ok3 {
						sawStallArgs = true
					}
				}
			}
		}
		if ev.Cat == "ptxas.compile_loop" {
			if _, ok := ev.Args["regs"]; ok {
				if _, ok2 := ev.Args["spill_bytes"]; ok2 {
					sawPTXASArgs = true
				}
			}
		}
		if ev.Name == "M=1 -> M=4" {
			sawBucketM1M4 = true
		}
		if ev.Name == "M=4 -> M=8" {
			sawBucketM4M8 = true
		}
	}

	if completeCount != 3 || instantCount != 4 {
		t.Fatalf("got completeCount=%d instantCount=%d, want 3 and 4", completeCount, instantCount)
	}
	if !sawStallArgs {
		t.Fatalf("missing stall_long_score / stall_barrier / stall_mio in cupti.pcsampling event")
	}
	if !sawPTXASArgs {
		t.Fatalf("missing regs / spill_bytes in ptxas.compile_loop event")
	}
	if !sawBucketM1M4 || !sawBucketM4M8 {
		t.Fatalf("missing bucket transition markers M=1 -> M=4 (%v) or M=4 -> M=8 (%v)", sawBucketM1M4, sawBucketM4M8)
	}
}
