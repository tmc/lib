// profile2trace converts Nsight SQLite, ds4-min host spans, and Metal regions
// to native Perfetto protobuf. See README.md for clock alignment.
package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"sort"

	"github.com/tmc/lib/nvidia/x/perfetto"
)

type slice struct {
	name, category, process, thread string
	start, duration                 int64
	args                            map[string]any
}

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, "profile2trace:", err)
		os.Exit(1)
	}
}

func run(args []string, out, log io.Writer) error {
	fs := flag.NewFlagSet("profile2trace", flag.ContinueOnError)
	fs.SetOutput(log)
	nsys := fs.String("nsys", "", "Nsight Systems SQLite export (requires sqlite3)")
	spans := fs.String("spans", "", "gputrace.host-event/v1 JSONL")
	metal := fs.String("metal", "", "Metal region receipt JSON")
	hostOffset := fs.Int64("spans-offset-ns", 0, "add to host timestamps to align with the reference clock")
	metalOffset := fs.Int64("metal-offset-ns", 0, "add to Metal timestamps to align with the reference clock")
	nsysWindow := wholeCapture()
	fs.Var(&nsysWindow, "nsys-window", "keep only Nsight intervals starting in START:END, in source-clock nanoseconds")
	maxSlices := fs.Int("max-slices", 500000, "refuse an Nsight window selecting more kernel rows than this")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}
	if fs.NArg() != 0 {
		return errors.New("unexpected arguments")
	}
	count := 0
	for _, path := range []string{*nsys, *spans, *metal} {
		if path != "" {
			count++
		}
	}
	if count == 0 {
		return errors.New("specify -nsys, -spans, or -metal")
	}
	if *maxSlices < 1 {
		return errors.New("-max-slices must be positive")
	}
	seen := make(map[string]bool)
	fs.Visit(func(f *flag.Flag) { seen[f.Name] = true })
	if count > 1 {
		if *nsys != "" && *spans != "" && !seen["spans-offset-ns"] {
			return errors.New("merging Nsight and host spans requires explicit -spans-offset-ns")
		}
		if *metal != "" && !seen["metal-offset-ns"] {
			return errors.New("merging Metal regions requires explicit -metal-offset-ns")
		}
	}
	var all []slice
	if *nsys != "" {
		rows, err := readNSYS(*nsys, nsysWindow, *maxSlices, log)
		if err != nil {
			return err
		}
		all = append(all, rows...)
	}
	if *spans != "" {
		rows, err := readFile(*spans, readHost)
		if err != nil {
			return err
		}
		if err := offset(rows, *hostOffset); err != nil {
			return err
		}
		fmt.Fprintf(log, "host slices: %d\n", len(rows))
		all = append(all, rows...)
	}
	if *metal != "" {
		rows, err := readFile(*metal, readMetal)
		if err != nil {
			return err
		}
		if err := offset(rows, *metalOffset); err != nil {
			return err
		}
		fmt.Fprintf(log, "Metal device slices: %d\n", len(rows))
		all = append(all, rows...)
	}
	return writeTrace(out, all)
}

func readFile(path string, read func(io.Reader) ([]slice, error)) ([]slice, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	rows, err := read(f)
	return rows, errors.Join(err, f.Close())
}

func offset(rows []slice, delta int64) error {
	for i := range rows {
		n := rows[i].start
		if delta > 0 && n > math.MaxInt64-delta || delta < 0 && n < math.MinInt64-delta {
			return errors.New("clock offset overflows timestamp")
		}
		if rows[i].args == nil {
			rows[i].args = make(map[string]any)
		}
		rows[i].args["source_start_ns"] = n
		rows[i].args["clock_offset_ns"] = delta
		rows[i].start += delta
	}
	return nil
}

// writeTrace assigns overlapping intervals to separate tracks because the host
// receipt does not record thread IDs, and arbitrary ranges need not be nested.
func writeTrace(out io.Writer, rows []slice) error {
	if len(rows) == 0 {
		return errors.New("input contains no intervals")
	}
	sort.SliceStable(rows, func(i, j int) bool { return rows[i].start < rows[j].start })
	origin := rows[0].start
	for _, s := range rows {
		if s.duration < 0 || origin < 0 && s.start > math.MaxInt64+origin || s.start-origin > math.MaxInt64-s.duration || s.start > math.MaxInt64-s.duration {
			return errors.New("invalid interval or timestamp range")
		}
	}
	pw := perfetto.NewWriter(out)
	processes := make(map[string]uint64)
	type lane struct {
		id  uint64
		end int64
	}
	tracks := make(map[string][]lane)
	type edge struct {
		time  uint64
		track uint64
		row   int
		begin bool
	}
	var edges []edge
	var next uint64
	for i, s := range rows {
		parent := processes[s.process]
		if parent == 0 {
			next++
			parent = next
			processes[s.process] = parent
			if err := pw.WriteTrackDescriptor(parent, s.process, 0); err != nil {
				return err
			}
		}
		key := s.process + "/" + s.thread
		lanes := tracks[key]
		j := 0
		for j < len(lanes) && lanes[j].end > s.start {
			j++
		}
		if j == len(lanes) {
			next++
			lanes = append(lanes, lane{id: next})
			name := s.thread
			if j > 0 {
				name = fmt.Sprintf("%s (overlap %d)", name, j)
			}
			if err := pw.WriteTrackDescriptor(next, name, parent); err != nil {
				return err
			}
		}
		start := uint64(s.start - origin)
		edges = append(edges, edge{start, lanes[j].id, i, true}, edge{start + uint64(s.duration), lanes[j].id, i, false})
		lanes[j].end = s.start + s.duration
		tracks[key] = lanes
	}
	sort.SliceStable(edges, func(i, j int) bool { return edges[i].time < edges[j].time })
	for _, e := range edges {
		if !e.begin {
			if err := pw.WriteSliceEnd(e.track, e.time); err != nil {
				return err
			}
			continue
		}
		s := rows[e.row]
		annotations := []perfetto.DebugAnnotation{perfetto.NewIntAnnotation("trace_origin_ns", origin), perfetto.NewIntAnnotation("aligned_start_ns", s.start)}
		keys := make([]string, 0, len(s.args))
		for key := range s.args {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for _, key := range keys {
			annotations = append(annotations, perfetto.NewStringAnnotation(key, fmt.Sprint(s.args[key])))
		}
		if err := pw.WriteSliceBeginWithAnnotations(e.track, e.time, s.name, s.category, annotations, nil); err != nil {
			return err
		}
	}
	return nil
}
