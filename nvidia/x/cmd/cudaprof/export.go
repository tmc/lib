package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
	"github.com/tmc/lib/nvidia/x/perfetto"
)

// cudaprof export writes a capture's device timeline as a Perfetto trace.
//
// The per-record journal is the only tier that carries timestamps: pprof
// samples are aggregates, so a trace cannot be built from the projection. This
// command therefore refuses exactly the captures gaps refuses, and for the same
// reason -- a timeline assembled from a journal that dropped records has holes
// no viewer displays, and a hole is indistinguishable from device idle.
//
// The trace is standalone. Timestamps are CUPTI's own domain rebased on the
// earliest record, and no clock-synchronization packet is emitted, so it must
// not be merged with a host trace taken from another clock.

type exportOptions struct {
	capturePath  string
	outPath      string
	maxRecords   int64
	journalBytes int64
	limits       cudacapture.Limits
}

var errExportHelp = errors.New("export: help requested")

func runExportCommand(args []string) error {
	if len(args) == 0 {
		printExportHelp()
		return nil
	}
	opts, err := parseExportArgs(args)
	if err != nil {
		if errors.Is(err, errExportHelp) {
			return nil
		}
		return err
	}
	path, err := filepath.Abs(opts.capturePath)
	if err != nil {
		return err
	}
	cap, err := readCapturePathLimits(path, opts.limits)
	if err != nil {
		return fmt.Errorf("read capture %s: %w", opts.capturePath, err)
	}
	journals, _, err := requireJournals(cap)
	if err != nil {
		return err
	}
	records, err := decodeTraceJournals(journals, opts.maxRecords, opts.journalBytes)
	if err != nil {
		return err
	}
	f, err := os.Create(opts.outPath)
	if err != nil {
		return fmt.Errorf("create trace %s: %w", opts.outPath, err)
	}
	n, err := writeTrace(f, records)
	if err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("close trace %s: %w", opts.outPath, err)
	}
	fmt.Printf("wrote %s: %d record(s) on %d track(s)\n", opts.outPath, len(records), n)
	return nil
}

func parseExportArgs(rawArgs []string) (exportOptions, error) {
	args, limits, err := parseReadLimits(rawArgs)
	if err != nil {
		return exportOptions{}, err
	}
	opts := exportOptions{maxRecords: controllerMaxJournalRecords, journalBytes: cudacapture.MaxCaptureInputBytes, limits: limits}
	for i := 0; i < len(args); i++ {
		arg := args[i]
		switch {
		case arg == "-h" || arg == "--help" || arg == "-help":
			printExportHelp()
			return exportOptions{}, errExportHelp
		case arg == "-o" && i+1 < len(args):
			i++
			opts.outPath = args[i]
		case strings.HasPrefix(arg, "-o="), strings.HasPrefix(arg, "--out="):
			opts.outPath = arg[strings.Index(arg, "=")+1:]
		case strings.HasPrefix(arg, "--max-records="), strings.HasPrefix(arg, "-max-records="):
			v, err := strconv.ParseInt(arg[strings.Index(arg, "=")+1:], 10, 64)
			if err != nil || v <= 0 {
				return exportOptions{}, fmt.Errorf("invalid max records: %s", arg)
			}
			opts.maxRecords = v
		case strings.HasPrefix(arg, "--journal-bytes="), strings.HasPrefix(arg, "-journal-bytes="):
			v, err := strconv.ParseInt(arg[strings.Index(arg, "=")+1:], 10, 64)
			if err != nil || v <= 0 {
				return exportOptions{}, fmt.Errorf("invalid journal bytes: %s", arg)
			}
			opts.journalBytes = v
		case strings.HasPrefix(arg, "-"):
			return exportOptions{}, fmt.Errorf("unknown flag: %s", arg)
		case opts.capturePath == "":
			opts.capturePath = arg
		default:
			return exportOptions{}, fmt.Errorf("unexpected argument: %s", arg)
		}
	}
	if opts.capturePath == "" {
		return exportOptions{}, errors.New("usage: cudaprof export [-o out.pftrace] [--max-records=N] [--journal-bytes=N] capture.pprof")
	}
	if opts.outPath == "" {
		opts.outPath = strings.TrimSuffix(opts.capturePath, ".pprof") + ".pftrace"
	}
	return opts, nil
}

func printExportHelp() {
	fmt.Fprint(os.Stdout, `Usage: cudaprof export [options] capture.pprof

Writes the capture's device timeline as a Perfetto trace (.pftrace), openable
at https://ui.perfetto.dev.

Kernels and memcpys go on one track per device and stream; driver and runtime
API calls go on one track per thread. Activities that overlap on the same
stream are split across numbered lanes rather than nested, so no slice is
drawn inside another it did not contain.

The timeline comes from the per-record journal, not from the pprof samples,
which are aggregates and carry no timestamps. A capture whose journal dropped
records is refused: a hole in a timeline is indistinguishable from device idle
once drawn.

Timestamps are CUPTI's domain rebased on the earliest record. No clock
synchronization packet is emitted, so this trace must not be merged with a
host trace taken from another clock.

Options:
  -o <path>          output file (default: the capture path with .pftrace)
  --max-records=N    maximum journal records to decode (default 1048576)
  --journal-bytes=N  maximum decompressed bytes per journal
`)
}

// tracePayload is the typed journal payload, in full. gapPayload reads the
// four fields a duty cycle needs; a timeline draws the rest.
type tracePayload struct {
	Kind        string `json:"kind"`
	Subtype     string `json:"subtype"`
	Domain      string `json:"domain"`
	Name        string `json:"name"`
	Valid       bool   `json:"valid"`
	Start       uint64 `json:"start"`
	End         uint64 `json:"end"`
	Bytes       uint64 `json:"bytes"`
	Queued      uint64 `json:"queued"`
	Submitted   uint64 `json:"submitted"`
	CopyKind    uint32 `json:"copy_kind"`
	RuntimeCorr uint32 `json:"runtime_correlation"`
	Device      uint32 `json:"device"`
	Context     uint32 `json:"context"`
	Stream      uint32 `json:"stream"`
	Correlation uint32 `json:"correlation"`
	GridX       uint32 `json:"grid_x"`
	GridY       uint32 `json:"grid_y"`
	GridZ       uint32 `json:"grid_z"`
	BlockX      uint32 `json:"block_x"`
	BlockY      uint32 `json:"block_y"`
	BlockZ      uint32 `json:"block_z"`
	Registers   uint32 `json:"registers_per_thread"`
	DynShared   uint32 `json:"dynamic_shared_memory"`
	GraphID     uint32 `json:"graph_id"`
	GraphNodeID uint64 `json:"graph_node_id"`
	ProcessID   uint32 `json:"process_id"`
	ThreadID    uint32 `json:"thread_id"`
	CallbackID  uint32 `json:"callback_id"`
	ReturnValue uint32 `json:"return_value"`
	Dropped     uint64 `json:"dropped"`
}

// traceRecord is one activity with the track it belongs on. The track key is
// separate from the record because a device stream and a host thread are
// different namespaces that would otherwise collide on small integers.
type traceRecord struct {
	rec   cuptiactivity.Record
	host  bool // true for an API record, which lives on a thread track
	group [2]uint32
}

// decodeTraceJournals decodes every journal member into activities and API
// calls. The refusals mirror decodeJournals: a journal that never finalized,
// lost records, or decoded short leaves a timeline whose missing spans read as
// idle device, which is a wrong answer rather than a partial one.
func decodeTraceJournals(journals map[string][]byte, maxRecords, maxBytes int64) ([]traceRecord, error) {
	names := make([]string, 0, len(journals))
	for name := range journals {
		names = append(names, name)
	}
	sort.Strings(names)
	var out []traceRecord
	for _, name := range names {
		reader, _, err := decompressJournal(journals[name], maxBytes)
		if err != nil {
			return nil, fmt.Errorf("read journal %s: %w", name, err)
		}
		err = decodeTraceJournal(name, reader, maxRecords, &out)
		if cerr := reader.Close(); err == nil && cerr != nil {
			err = fmt.Errorf("close journal %s: %w", name, cerr)
		}
		if err != nil {
			return nil, err
		}
	}
	if len(out) == 0 {
		return nil, errors.New("journals hold no complete activity records, so there is no timeline to draw")
	}
	return out, nil
}

func decodeTraceJournal(name string, reader io.Reader, maxRecords int64, out *[]traceRecord) error {
	limit := uint32(math.MaxUint32)
	if maxRecords > 0 && maxRecords < math.MaxUint32 {
		limit = uint32(maxRecords)
	}
	d := wire.NewDecoder(reader, wire.Limits{MaxRecords: limit})
	first, finalized, decoded := true, false, 0
	for {
		r, err := d.Next()
		if err != nil {
			switch {
			case errors.Is(err, io.EOF):
				if decoded == 0 {
					return fmt.Errorf("journal %s is empty, so no timeline was recorded", name)
				}
				if !finalized {
					return fmt.Errorf("journal %s decoded %d record(s) but never finalized, so the timeline has no known end", name, decoded)
				}
				if stats := d.Stats(); stats.Gaps > 0 || stats.Dropped > 0 {
					return fmt.Errorf("journal %s lost records (%d sequence gap(s), %d dropped), so the timeline is missing spans that would draw as idle", name, stats.Gaps, stats.Dropped)
				}
				return nil
			case errors.Is(err, io.ErrUnexpectedEOF):
				return fmt.Errorf("journal %s is truncated after %d record(s): the timeline ends at an unknown point", name, decoded)
			default:
				return fmt.Errorf("decode journal %s after %d record(s): %w", name, decoded, err)
			}
		}
		if first {
			d.ExpectSession(r.Session)
			first = false
		}
		decoded++
		var p tracePayload
		if len(r.Payload) > 0 && json.Unmarshal(r.Payload, &p) != nil {
			return fmt.Errorf("journal %s record %d is not a typed capture record", name, r.Sequence)
		}
		switch r.Kind {
		case recordActivity:
			if p.Dropped > 0 {
				return fmt.Errorf("journal %s reports %d dropped record(s), so the timeline is missing activities", name, p.Dropped)
			}
			if !p.Valid || p.Start == 0 || p.End < p.Start {
				continue
			}
			*out = append(*out, traceRecord{rec: activityRecord(p), group: [2]uint32{p.Device, p.Stream}})
		case recordAPI:
			if p.Start == 0 || p.End < p.Start {
				continue
			}
			*out = append(*out, traceRecord{rec: apiRecord(p), host: true, group: [2]uint32{p.ProcessID, p.ThreadID}})
		case recordFinalized:
			finalized = true
			if p.Dropped > 0 {
				return fmt.Errorf("journal %s finalized with %d dropped record(s), so the timeline is missing activities", name, p.Dropped)
			}
		}
	}
}

func activityRecord(p tracePayload) cuptiactivity.Record {
	r := cuptiactivity.Record{
		Name:          p.Name,
		Start:         p.Start,
		End:           p.End,
		Complete:      true,
		CorrelationID: p.Correlation,
		DeviceID:      p.Device,
		ContextID:     p.Context,
		StreamID:      p.Stream,
		Bytes:         p.Bytes,
		GraphID:       p.GraphID,
		GraphNodeID:   p.GraphNodeID,
	}
	switch p.Subtype {
	case "memcpy":
		r.Kind = cupti.ActivityKindMemcpy
		r.CopyKind = cupti.ActivityMemcpyKind(p.CopyKind)
		r.RuntimeCorrelationID = p.RuntimeCorr
	default:
		r.Kind = cupti.ActivityKindConcurrentKernel
		r.Queued, r.Submitted = p.Queued, p.Submitted
		r.Grid = [3]int32{int32(p.GridX), int32(p.GridY), int32(p.GridZ)}
		r.Block = [3]int32{int32(p.BlockX), int32(p.BlockY), int32(p.BlockZ)}
		r.RegistersPerThread = uint16(p.Registers)
		r.DynamicSharedMemory = int32(p.DynShared)
	}
	return r
}

func apiRecord(p tracePayload) cuptiactivity.Record {
	r := cuptiactivity.Record{
		Kind:          cupti.ActivityKindRuntime,
		Name:          p.Domain,
		Start:         p.Start,
		End:           p.End,
		Complete:      true,
		CorrelationID: p.Correlation,
		ProcessID:     p.ProcessID,
		ThreadID:      p.ThreadID,
		CallbackID:    p.CallbackID,
		ReturnValue:   p.ReturnValue,
	}
	if p.Domain == "driver" {
		r.Kind = cupti.ActivityKindDriver
	}
	if r.Name == "" {
		r.Name = r.Kind.String()
	}
	return r
}

// writeTrace emits descriptors and slices, returning the track count.
//
// Lanes exist because a Perfetto track is a stack: two slices that overlap on
// one track are drawn as one containing the other. CUPTI reports concurrent
// kernels, and a memcpy can overlap a kernel on the same stream id, so an
// overlapping activity goes on a numbered lane of the same group instead. That
// is a drawing decision, not a reinterpretation: no record moves in time.
func writeTrace(w io.Writer, records []traceRecord) (int, error) {
	origin := uint64(math.MaxUint64)
	for _, t := range records {
		if t.rec.Start < origin {
			origin = t.rec.Start
		}
	}
	uuids, names := assignTracks(records)
	pw := perfetto.NewWriter(w)
	for i, name := range names {
		if err := pw.WriteTrackDescriptor(uint64(i+1), name, 0); err != nil {
			return 0, fmt.Errorf("write track descriptor: %w", err)
		}
	}
	for i, t := range records {
		if err := cuptiactivity.WriteRecord(pw, uuids[i], origin, t.rec); err != nil {
			return len(names), fmt.Errorf("write record %d: %w", i, err)
		}
	}
	return len(names), nil
}

// assignTracks sorts records into drawing order and gives each one a track,
// returning the per-record track uuid and the track names in uuid order
// (uuid i+1 is names[i]).
//
// Lanes exist because a Perfetto track is a stack: two slices that overlap on
// one track are drawn as one CONTAINING the other, which asserts a nesting the
// hardware never reported. CUPTI reports concurrent kernels, and a memcpy can
// overlap a kernel carrying the same stream id, so an overlapping activity
// goes on a numbered lane of the same group instead. That is a drawing
// decision and not a reinterpretation: no record moves in time, and a record
// stays in its own device/stream or pid/thread group.
func assignTracks(records []traceRecord) ([]uint64, []string) {
	sort.SliceStable(records, func(i, j int) bool {
		a, b := records[i], records[j]
		if a.host != b.host {
			return !a.host
		}
		if a.group != b.group {
			if a.group[0] != b.group[0] {
				return a.group[0] < b.group[0]
			}
			return a.group[1] < b.group[1]
		}
		return a.rec.Start < b.rec.Start
	})
	type lane struct {
		uuid uint64
		end  uint64
	}
	var (
		lanes = map[[3]uint32][]lane{}
		uuids = make([]uint64, len(records))
		names []string
	)
	for i, t := range records {
		key := [3]uint32{boolUint(t.host), t.group[0], t.group[1]}
		group := lanes[key]
		slot := -1
		for j := range group {
			if group[j].end <= t.rec.Start {
				slot = j
				break
			}
		}
		if slot < 0 {
			names = append(names, trackName(t.host, t.group, len(group)))
			group = append(group, lane{uuid: uint64(len(names))})
			slot = len(group) - 1
		}
		group[slot].end = t.rec.End
		lanes[key] = group
		uuids[i] = group[slot].uuid
	}
	return uuids, names
}

func trackName(host bool, group [2]uint32, lane int) string {
	var name string
	if host {
		name = fmt.Sprintf("pid %d thread %d", group[0], group[1])
	} else {
		name = fmt.Sprintf("device %d stream %d", group[0], group[1])
	}
	if lane > 0 {
		name += fmt.Sprintf(" (overlap %d)", lane)
	}
	return name
}

func boolUint(b bool) uint32 {
	if b {
		return 1
	}
	return 0
}
