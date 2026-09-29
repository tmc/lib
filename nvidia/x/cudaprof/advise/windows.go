package advise

import (
	"encoding/json"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

// A captureWindow is one measured capture range: the interval between a
// qualified start and stop edge, in device timestamps.
type captureWindow struct{ start, end uint64 }

// captureWindows returns the measured capture ranges recorded in cap's phase
// receipt. A target that opened the range several times has one window per
// pair; one that opened it once has one. It returns nil when the receipt is
// absent or unread.
func captureWindows(cap *cudacapture.Capture) []captureWindow {
	if cap == nil {
		return nil
	}
	type receipt struct {
		Status string `json:"status"`
		Start  uint64 `json:"start_timestamp"`
		Stop   uint64 `json:"stop_timestamp"`
	}
	var status struct {
		PhaseReceipt *struct {
			receipt
			Phases []receipt `json:"phases"`
		} `json:"phase_receipt"`
	}
	data := cap.Files["capture-status.json"]
	var wrapped struct {
		Status *json.RawMessage `json:"status"`
	}
	if json.Unmarshal(data, &wrapped) == nil && wrapped.Status != nil {
		data = *wrapped.Status
	}
	if json.Unmarshal(data, &status) != nil || status.PhaseReceipt == nil {
		return nil
	}
	phases := status.PhaseReceipt.Phases
	if len(phases) == 0 {
		phases = []receipt{status.PhaseReceipt.receipt}
	}
	var windows []captureWindow
	for _, p := range phases {
		if p.Status == "MEASURED" && p.Stop > p.Start {
			windows = append(windows, captureWindow{p.Start, p.Stop})
		}
	}
	return windows
}

// windowedDutyCycle is profileDutyCycle over the captured windows only.
//
// Between two windows the device was not observed, so that time is neither
// busy nor idle. Measuring from the first activity of the first window to the
// last activity of the last counts it as idle, and a trainer sampled in five
// short windows read as a device 0.76% busy. The span here is the sum of each
// window's own first-to-last activity span, and busy time is the union of the
// activities clipped to the windows.
//
// It also returns the clipped streams, so a caller measuring the same span
// another way reads the same records.
func windowedDutyCycle(streams map[uint32][]timeline.Event, windows []captureWindow) (pct float64, spanNs, busyNs uint64, clipped map[uint32][]timeline.Event, ok bool) {
	clipped = make(map[uint32][]timeline.Event)
	for _, w := range windows {
		in := make(map[uint32][]timeline.Event)
		for stream, evs := range streams {
			for _, ev := range evs {
				lo, hi := max(ev.Start, w.start), min(ev.End, w.end)
				if ev.End == 0 || lo >= hi {
					continue
				}
				ev.Start, ev.End = lo, hi
				in[stream] = append(in[stream], ev)
				clipped[stream] = append(clipped[stream], ev)
			}
		}
		// A window holding a single activity has no idle to measure, and
		// profileDutyCycle declines it rather than call it 100% busy.
		if _, span, busy, wok := profileDutyCycle(in); wok {
			spanNs += span
			busyNs += busy
		}
	}
	if spanNs == 0 {
		return 0, 0, 0, nil, false
	}
	return float64(busyNs) * 100.0 / float64(spanNs), spanNs, busyNs, clipped, true
}
