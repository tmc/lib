package nccltest

import (
	"bufio"
	"fmt"
	"io"
	"math"
	"regexp"
	"strconv"
	"strings"
)

// Measurement is one nccl-tests output column group. Checked distinguishes a
// reported zero mismatch count from N/A (such as unsupported in-place alltoall).
type Measurement struct {
	Microseconds float64
	AlgorithmGBs float64
	BusGBs       float64
	Checked      bool
	Wrong        uint64
}

// ReferenceConfig records the controls printed by the upstream test harness.
// GPUs is per thread; the comparison command currently uses one process/thread.
type ReferenceConfig struct {
	Threads       int
	GPUs          int
	Warmup        int
	Iterations    int
	Aggregation   int
	Checks        int
	GraphLaunches int
	Offset        int
}

var referenceConfigPattern = regexp.MustCompile(`^# nThread ([0-9]+) nGpus ([0-9]+).* warmup iters: ([0-9]+) iters: ([0-9]+) agg iters: ([0-9]+) validation: ([0-9]+) graph: ([0-9]+) unalign: ([0-9]+)(?: |$)`)

// ReferenceRow is one size row from the standard nccl-tests text format.
type ReferenceRow struct {
	Config      ReferenceConfig
	NCCLVersion int32
	Bytes       int
	Count       int
	DataType    string
	Reduction   string
	Root        int
	OutOfPlace  Measurement
	InPlace     Measurement
}

// ParseReference reads the ordinary nccl-tests text table. It requires the
// final correctness summary and completion marker; truncated or empty output
// is an error. The process's exit status must also be checked by the caller.
func ParseReference(r io.Reader) ([]ReferenceRow, error) {
	scanner := bufio.NewScanner(r)
	var rows []ReferenceRow
	summary, complete := false, false
	configSeen := false
	var libraryVersion int64
	var config ReferenceConfig
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if strings.HasPrefix(line, "# nThread ") {
			if configSeen {
				return nil, fmt.Errorf("nccltest: duplicate reference controls")
			}
			configSeen = true
			m := referenceConfigPattern.FindStringSubmatch(line)
			if m == nil {
				return nil, fmt.Errorf("nccltest: malformed reference controls")
			}
			values := []*int{&config.Threads, &config.GPUs, &config.Warmup, &config.Iterations, &config.Aggregation, &config.Checks, &config.GraphLaunches, &config.Offset}
			for i, p := range values {
				n, err := strconv.Atoi(m[i+1])
				if err != nil || n < 0 || (i != 2 && i != 6 && i != 7 && n == 0) {
					return nil, fmt.Errorf("nccltest: invalid reference control %q", m[i+1])
				}
				*p = n
			}
		}
		if strings.HasPrefix(line, "# nccl-tests version ") {
			if libraryVersion != 0 || len(rows) != 0 {
				return nil, fmt.Errorf("nccltest: duplicate or late version banner")
			}
			for _, field := range strings.Fields(line) {
				if value, ok := strings.CutPrefix(field, "nccl-library="); ok {
					var err error
					libraryVersion, err = strconv.ParseInt(value, 10, 32)
					if err != nil || libraryVersion <= 0 {
						return nil, fmt.Errorf("nccltest: invalid NCCL library version")
					}
				}
			}
		}
		if strings.HasPrefix(line, "# Out of bounds values") {
			if summary || complete || len(rows) == 0 {
				return nil, fmt.Errorf("nccltest: misplaced correctness summary")
			}
			f := strings.Fields(line)
			if len(f) != 8 || f[6] != "0" || f[7] != "OK" {
				return nil, fmt.Errorf("nccltest: unsuccessful correctness summary %q", line)
			}
			summary = true
		}
		if strings.HasPrefix(line, "# Collective test concluded:") {
			if !summary || complete {
				return nil, fmt.Errorf("nccltest: misplaced completion marker")
			}
			complete = true
		}
		if line == "" || line[0] == '#' {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) == 0 || fields[0][0] < '0' || fields[0][0] > '9' {
			continue
		}
		if !configSeen || libraryVersion == 0 || summary || complete {
			return nil, fmt.Errorf("nccltest: data row outside reference result section")
		}
		// Hypercube prints an empty reduction column instead of "none".
		if len(fields) == 12 {
			fields = append(fields[:3], append([]string{""}, fields[3:]...)...)
		}
		if len(fields) != 13 {
			return nil, fmt.Errorf("nccltest: expected 13 data columns, got %d", len(fields))
		}
		row := ReferenceRow{Config: config, NCCLVersion: int32(libraryVersion), DataType: fields[2], Reduction: fields[3]}
		var err error
		row.Bytes, err = strconv.Atoi(fields[0])
		if err != nil || row.Bytes <= 0 {
			return nil, fmt.Errorf("nccltest: invalid reference size %q", fields[0])
		}
		row.Count, err = strconv.Atoi(fields[1])
		if err != nil || row.Count <= 0 {
			return nil, fmt.Errorf("nccltest: invalid reference count %q", fields[1])
		}
		row.Root, err = strconv.Atoi(fields[4])
		if err != nil {
			return nil, fmt.Errorf("nccltest: invalid reference root %q", fields[4])
		}
		row.OutOfPlace, err = parseMeasurement(fields[5:9])
		if err != nil {
			return nil, err
		}
		row.InPlace, err = parseMeasurement(fields[9:13])
		if err != nil {
			return nil, err
		}
		rows = append(rows, row)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if len(rows) == 0 || !summary || !complete || libraryVersion == 0 || config.Aggregation == 0 {
		return nil, fmt.Errorf("nccltest: incomplete reference output: rows=%d summary=%v complete=%v", len(rows), summary, complete)
	}
	return rows, nil
}
func parseMeasurement(f []string) (m Measurement, err error) {
	values := []*float64{&m.Microseconds, &m.AlgorithmGBs, &m.BusGBs}
	for i, p := range values {
		*p, err = strconv.ParseFloat(f[i], 64)
		if err != nil || math.IsNaN(*p) || math.IsInf(*p, 0) || *p < 0 {
			return m, fmt.Errorf("nccltest: invalid timing/bandwidth %q", f[i])
		}
	}
	if f[3] == "N/A" {
		return m, nil
	}
	m.Wrong, err = strconv.ParseUint(f[3], 10, 64)
	if err != nil {
		return m, fmt.Errorf("nccltest: invalid correctness count %q", f[3])
	}
	m.Checked = true
	return m, nil
}
