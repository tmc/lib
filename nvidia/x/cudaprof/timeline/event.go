package timeline

import (
	"encoding/json"
	"strconv"
	"strings"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// JournalEvents reconstructs the device timeline from the per-record
// journal, for consumers that otherwise read the pprof projection.
//
// The projection is an aggregate: it carries no activity timestamps, so a
// reader built on it reports that the capture does not have them. On a capture
// whose journal is attached that statement is false about the file, and the
// recommendation it produced -- re-capture with timestamps -- sends someone
// back under GPU lock for work already on disk. Everything needed is in the
// member gaps reads by default.
//
// The second return says the replay envelopes are the only record of their
// graphs (see [Records].GraphNodes), so a caller can carry that caveat forward instead
// of silently reporting envelope time as kernel time.
func JournalEvents(cap *cudacapture.Capture) (events []Event, graphsOpaque bool, err error) {
	journals, err := Journals(cap)
	if err != nil {
		return nil, false, err
	}
	records, err := Decode(journals, MaxRecords, cudacapture.MaxCaptureInputBytes)
	if err != nil {
		return nil, false, err
	}
	acts := records.Activities
	if len(records.Graphs) != 0 && records.GraphNodes == 0 {
		graphsOpaque = true
		acts = append(acts, records.Graphs...)
	}
	for _, a := range acts {
		if a.End <= a.Start {
			continue
		}
		subtype := a.Subtype
		if subtype == "" {
			subtype = "kernel"
		}
		events = append(events, Event{
			Subtype:     subtype,
			Name:        a.Name,
			Stream:      a.Stream,
			DurationNs:  uint64(a.End - a.Start),
			Start:       uint64(a.Start),
			End:         uint64(a.End),
			Correlation: a.Correlation,
			GraphID:     a.GraphID,
			Grid:        a.Grid,
			Block:       a.Block,
		})
	}
	return events, graphsOpaque, nil
}

// journalAPIRecords reads the host launch API records from the per-record
// journal, indexed by correlation id.
//
// Captures written through compactCaptureStatus omit the per-record arrays, so
// a reader that looks only in capture-status.json finds none and reports that
// the capture has no host launch timestamps. The journal that same capture
// carries held 124834 of them on the run that found this, all correlated.
func journalAPIRecords(cap *cudacapture.Capture) (map[uint32]APICall, error) {
	journals, err := Journals(cap)
	if err != nil {
		return nil, err
	}
	records, err := Decode(journals, MaxRecords, cudacapture.MaxCaptureInputBytes)
	if err != nil {
		return nil, err
	}
	res := make(map[uint32]APICall, len(records.APIs))
	for corr, api := range records.APIs {
		if corr == 0 || api.Start <= 0 || api.End < api.Start {
			continue
		}
		res[corr] = APICall{Start: uint64(api.Start), End: uint64(api.End)}
	}
	return res, nil
}

// parseDim3 parses a grid or block label such as "128x1x1".
func parseDim3(s string) [3]int {
	s = strings.Trim(s, "()[]{} ")
	parts := strings.FieldsFunc(s, func(r rune) bool {
		return r == ',' || r == 'x' || r == 'X' || r == ' ' || r == '	'
	})
	var res [3]int
	res[0], res[1], res[2] = 1, 1, 1
	for i, p := range parts {
		if i >= 3 {
			break
		}
		if n, err := strconv.Atoi(strings.TrimSpace(p)); err == nil && n > 0 {
			res[i] = n
		}
	}
	return res
}

// parseDim3FromSlice parses a grid or block given as a JSON array.
func parseDim3FromSlice(items []any) [3]int {
	var res [3]int
	res[0], res[1], res[2] = 1, 1, 1
	for i, it := range items {
		if i >= 3 {
			break
		}
		if f, ok := it.(float64); ok && f > 0 {
			res[i] = int(f)
		}
	}
	return res
}

// An Event is one device activity on a stream.
type Event struct {
	Subtype    string
	Name       string
	Stream     uint32
	DurationNs uint64
	// Start and End are device timestamps. They are zero when the capture does
	// not carry per-activity timestamps.
	Start       uint64
	End         uint64
	Correlation uint32
	// GraphID names the CUDA graph this ran as a node of, or the graph a
	// replay envelope replayed. Zero means the record named none.
	GraphID uint32
	Shape   string
	Buffer  string
	Grid    [3]int
	Block   [3]int
}

// An APICall is a host driver or runtime API call. Its timestamps describe the
// host call, not the GPU work it launched.
type APICall struct {
	Start uint64
	End   uint64
}

// APIRecords indexes host launch API records by correlation ID. They come
// from the api_records array of a status file when the capture retains one,
// and from the journals otherwise: captures written through
// compactCaptureStatus omit the per-record array.
func APIRecords(cap *cudacapture.Capture) map[uint32]APICall {
	res := make(map[uint32]APICall)
	if cap == nil {
		return res
	}
	for _, name := range []string{"capture-status.json", "status.json", "activities.json", "trace.json"} {
		data, ok := cap.Files[name]
		if !ok || len(data) == 0 {
			continue
		}
		var raw map[string]any
		if json.Unmarshal(data, &raw) != nil {
			continue
		}
		items, _ := raw["api_records"].([]any)
		if items == nil {
			if nested, ok := raw["status"].(map[string]any); ok {
				items, _ = nested["api_records"].([]any)
			}
		}
		for _, item := range items {
			m, ok := item.(map[string]any)
			if !ok {
				continue
			}
			corr, ok := m["correlation"].(float64)
			if !ok || corr <= 0 {
				continue
			}
			start, _ := m["start"].(float64)
			if start <= 0 {
				continue
			}
			end, _ := m["end"].(float64)
			res[uint32(corr)] = APICall{Start: uint64(start), End: uint64(end)}
		}
		if len(res) > 0 {
			return res
		}
	}
	// Same reason the timeline falls back to the journal: a status file that
	// omits the per-record array says nothing about the capture, and the
	// finding built on its silence claims the launches were never recorded.
	if fromJournal, err := journalAPIRecords(cap); err == nil && len(fromJournal) > 0 {
		return fromJournal
	}
	return res
}

// SampleCorrelation returns the CUPTI correlation ID recorded on a sample, or
// zero when the capture carries none.
func SampleCorrelation(s *profile.Sample) uint32 {
	if s == nil {
		return 0
	}
	if len(s.NumLabel["correlation"]) > 0 && s.NumLabel["correlation"][0] > 0 {
		return uint32(s.NumLabel["correlation"][0])
	}
	if len(s.Label["correlation"]) > 0 {
		if n, err := strconv.ParseUint(s.Label["correlation"][0], 10, 32); err == nil {
			return uint32(n)
		}
	}
	return 0
}

// Events returns the device activities of a capture by stream. They come from
// the pprof samples; from the journals when no sample carries a device
// interval; and from a JSON activity file when neither yields an event.
//
// The second result reports that the events came from a journal whose graph
// replays are opaque: the envelopes are on the timeline because no
// device record named a graph, so their interiors are counted as busy. A
// caller stating a duty cycle has to carry that caveat, and this is the only
// place that knows it -- re-deriving it would mean decoding the journal twice.
func Events(cap *cudacapture.Capture) (map[uint32][]Event, bool) {
	res := make(map[uint32][]Event)
	if cap == nil {
		return res, false
	}

	if cap.Profile != nil && len(cap.Profile.Sample) > 0 {
		kernelTimeIdx := -1
		gpuTimeIdx := -1
		copyTimeIdx := -1
		for i, st := range cap.Profile.SampleType {
			switch st.Type {
			case "kernel_time":
				kernelTimeIdx = i
			case "gpu_time":
				gpuTimeIdx = i
			case "copy_time":
				copyTimeIdx = i
			}
		}

		for _, s := range cap.Profile.Sample {
			var streamID uint32
			if len(s.NumLabel["stream"]) > 0 {
				streamID = uint32(s.NumLabel["stream"][0])
			} else if len(s.NumLabel["stream_id"]) > 0 {
				streamID = uint32(s.NumLabel["stream_id"][0])
			} else if len(s.Label["stream"]) > 0 {
				if n, err := strconv.ParseUint(s.Label["stream"][0], 10, 32); err == nil {
					streamID = uint32(n)
				}
			}

			name := ""
			if len(s.Label["name"]) > 0 {
				name = s.Label["name"][0]
			} else if len(s.Label["kernel"]) > 0 {
				name = s.Label["kernel"][0]
			} else if len(s.Location) > 0 && len(s.Location[0].Line) > 0 && s.Location[0].Line[0].Function != nil {
				name = s.Location[0].Line[0].Function.Name
			}

			subtype := ""
			if len(s.Label["subtype"]) > 0 {
				subtype = s.Label["subtype"][0]
			} else {
				nLower := strings.ToLower(name)
				if copyTimeIdx >= 0 && s.Value[copyTimeIdx] > 0 {
					subtype = "memcpy"
				} else if strings.Contains(nLower, "memcpy") {
					subtype = "memcpy"
				} else if strings.Contains(nLower, "synchroniz") || strings.Contains(nLower, "sync") {
					subtype = "synchronization"
				} else if name != "" {
					subtype = "kernel"
				}
			}

			var durationNs uint64
			if kernelTimeIdx >= 0 && s.Value[kernelTimeIdx] > 0 {
				durationNs = uint64(s.Value[kernelTimeIdx])
			} else if gpuTimeIdx >= 0 && s.Value[gpuTimeIdx] > 0 {
				durationNs = uint64(s.Value[gpuTimeIdx])
			} else if copyTimeIdx >= 0 && s.Value[copyTimeIdx] > 0 {
				durationNs = uint64(s.Value[copyTimeIdx])
			} else if len(s.NumLabel["duration_ns"]) > 0 {
				durationNs = uint64(s.NumLabel["duration_ns"][0])
			} else if len(s.NumLabel["end"]) > 0 && len(s.NumLabel["start"]) > 0 && s.NumLabel["end"][0] >= s.NumLabel["start"][0] {
				durationNs = uint64(s.NumLabel["end"][0] - s.NumLabel["start"][0])
			} else if len(s.Label["duration_ns"]) > 0 {
				durationNs, _ = strconv.ParseUint(s.Label["duration_ns"][0], 10, 64)
			}

			var startNs, endNs uint64
			if len(s.NumLabel["start"]) > 0 && s.NumLabel["start"][0] > 0 {
				startNs = uint64(s.NumLabel["start"][0])
			} else if len(s.Label["start"]) > 0 {
				startNs, _ = strconv.ParseUint(s.Label["start"][0], 10, 64)
			}
			if len(s.NumLabel["end"]) > 0 && s.NumLabel["end"][0] > 0 {
				endNs = uint64(s.NumLabel["end"][0])
			} else if len(s.Label["end"]) > 0 {
				endNs, _ = strconv.ParseUint(s.Label["end"][0], 10, 64)
			}
			if endNs == 0 && startNs > 0 && durationNs > 0 {
				endNs = startNs + durationNs
			}

			hasStream := len(s.NumLabel["stream"]) > 0 || len(s.NumLabel["stream_id"]) > 0 || len(s.Label["stream"]) > 0
			hasTiming := durationNs > 0 || startNs > 0 || endNs > 0
			hasActivity := len(s.Label["subtype"]) > 0 || len(s.NumLabel["correlation"]) > 0 || len(s.Label["correlation_id"]) > 0 || len(s.Label["grid"]) > 0
			if !hasStream && !hasTiming && !hasActivity {
				continue
			}

			correlation := SampleCorrelation(s)
			var graphID uint32
			if len(s.NumLabel["graph_id"]) > 0 {
				graphID = uint32(s.NumLabel["graph_id"][0])
			}

			shape := ""
			if len(s.Label["shape"]) > 0 {
				shape = s.Label["shape"][0]
			} else if len(s.Label["tensor"]) > 0 {
				shape = s.Label["tensor"][0]
			}

			buffer := ""
			if len(s.Label["buffer"]) > 0 {
				buffer = s.Label["buffer"][0]
			}

			var grid, block [3]int
			if len(s.Label["grid"]) > 0 {
				grid = parseDim3(s.Label["grid"][0])
			}
			if len(s.Label["block"]) > 0 {
				block = parseDim3(s.Label["block"][0])
			}

			if name != "" || subtype != "" {
				res[streamID] = append(res[streamID], Event{
					Subtype:     subtype,
					Name:        name,
					Stream:      streamID,
					DurationNs:  durationNs,
					Start:       startNs,
					End:         endNs,
					Correlation: correlation,
					GraphID:     graphID,
					Shape:       shape,
					Buffer:      buffer,
					Grid:        grid,
					Block:       block,
				})
			}
		}
	}

	// The journal comes first among the fallbacks because it is the only
	// source that carries per-activity timestamps. The pprof samples above
	// are an aggregate, and a reader that stops there reports the capture as
	// having no timestamps when the file it was handed has them.
	//
	// The test is whether a timestamp was actually produced, not whether the
	// projection produced anything. A projection yielding hundreds of samples
	// that all carry Start == End == 0 is the case this exists for: it fills
	// res, so a len(res) == 0 guard would never fire, and the reader would go
	// on to announce that the capture has no timestamps while holding them.
	if !anyStreamTimed(res) {
		if events, opaque, err := JournalEvents(cap); err == nil && anyEventTimed(events) {
			// Replaces what the projection produced rather than joining it.
			// The two sources describe the same device work at different
			// fidelities, and a union of them would double every activity the
			// projection also carries.
			timed := make(map[uint32][]Event)
			for _, ev := range events {
				timed[ev.Stream] = append(timed[ev.Stream], ev)
			}
			return timed, opaque
		}
	}
	if len(res) == 0 && len(cap.Files) > 0 {
		for _, fname := range []string{"activities.json", "capture-status.json", "status.json", "trace.json"} {
			if data, ok := cap.Files[fname]; ok && len(data) > 0 {
				events := parseStreamEventsFromJSON(data)
				for _, ev := range events {
					res[ev.Stream] = append(res[ev.Stream], ev)
				}
				if len(res) > 0 {
					break
				}
			}
		}
	}

	return res, false
}

// parseStreamEventsFromJSON reads the activities array of a JSON file.
func parseStreamEventsFromJSON(data []byte) []Event {
	var events []Event
	var raw any
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil
	}

	var items []any
	switch v := raw.(type) {
	case []any:
		items = v
	case map[string]any:
		if acts, ok := v["activities"].([]any); ok {
			items = acts
		} else if status, ok := v["status"].(map[string]any); ok {
			if acts, ok := status["activities"].([]any); ok {
				items = acts
			}
		}
	}

	for _, item := range items {
		m, ok := item.(map[string]any)
		if !ok {
			continue
		}
		var ev Event
		if s, ok := m["subtype"].(string); ok {
			ev.Subtype = s
		} else if s, ok := m["kind"].(string); ok {
			ev.Subtype = s
		}
		if s, ok := m["name"].(string); ok {
			ev.Name = s
		}
		if s, ok := m["stream"].(float64); ok {
			ev.Stream = uint32(s)
		} else if s, ok := m["stream_id"].(float64); ok {
			ev.Stream = uint32(s)
		}

		var start, end uint64
		if s, ok := m["start"].(float64); ok {
			start = uint64(s)
		}
		if e, ok := m["end"].(float64); ok {
			end = uint64(e)
		}
		if end >= start && start > 0 {
			ev.DurationNs = end - start
			ev.Start, ev.End = start, end
		}
		if c, ok := m["correlation"].(float64); ok && c > 0 {
			ev.Correlation = uint32(c)
		}
		if g, ok := m["graph_id"].(float64); ok && g > 0 {
			ev.GraphID = uint32(g)
		}
		if d, ok := m["duration"].(float64); ok && d > 0 {
			ev.DurationNs = uint64(d)
		}
		if d, ok := m["duration_ns"].(float64); ok && d > 0 {
			ev.DurationNs = uint64(d)
		}

		if s, ok := m["shape"].(string); ok {
			ev.Shape = s
		}
		if b, ok := m["buffer"].(string); ok {
			ev.Buffer = b
		}

		if blk, ok := m["block"].([]any); ok && len(blk) > 0 {
			ev.Block = parseDim3FromSlice(blk)
		}
		if grd, ok := m["grid"].([]any); ok && len(grd) > 0 {
			ev.Grid = parseDim3FromSlice(grd)
		}

		if ev.Name != "" || ev.Subtype != "" {
			events = append(events, ev)
		}
	}
	return events
}

// anyEventTimed reports whether any event carries a usable device interval.
// End is the field tested, not Start: a timeline whose first activity begins
// at zero is a legitimate one.
func anyEventTimed(events []Event) bool {
	for _, ev := range events {
		if ev.End != 0 && ev.End > ev.Start {
			return true
		}
	}
	return false
}

// anyStreamTimed reports whether any stream holds a timed event.
func anyStreamTimed(streams map[uint32][]Event) bool {
	for _, evs := range streams {
		if anyEventTimed(evs) {
			return true
		}
	}
	return false
}
