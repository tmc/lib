package nsightcompute

import (
	"bytes"
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"math"
	"regexp"
	"strconv"
	"strings"
)

// A RawTable is one ncu --csv --page raw export: one row per profiled
// launch, one column per metric, and a units row.
//
// Unlike [Read] and [Import], ReadRaw accepts any Nsight Compute version and
// any metric set. It retains what the CSV says and joins nothing: a kernel
// name in a row is a display string, not an identity.
type RawTable struct {
	Preamble []string    // lines before the header, such as ==PROF== notices
	Metrics  []string    // metric column names in CSV order
	Units    []string    // unit of each metric column; "" when the export gives none
	Launches []RawLaunch // in CSV order
}

// A RawLaunch is one row of a raw-page export.
type RawLaunch struct {
	ID      string // the export's launch ID, as written
	Process string // "Process ID"
	Kernel  string // "Kernel Name", as exported (possibly demangled or truncated)
	Context string
	Stream  string
	Block   string // "Block Size", such as "(128, 1, 1)"
	Grid    string // "Grid Size"
	Device  string
	CC      string

	// Raw holds every metric value as written, indexed like RawTable.Metrics.
	Raw []string
}

// rawMeta maps the identifying columns of a raw-page export to RawLaunch
// fields. Every other column is a metric.
var rawMeta = map[string]func(*RawLaunch) *string{
	"ID":           func(l *RawLaunch) *string { return &l.ID },
	"Process ID":   func(l *RawLaunch) *string { return &l.Process },
	"Process Name": nil,
	"Host Name":    nil,
	"Kernel Name":  func(l *RawLaunch) *string { return &l.Kernel },
	"Context":      func(l *RawLaunch) *string { return &l.Context },
	"Stream":       func(l *RawLaunch) *string { return &l.Stream },
	"Block Size":   func(l *RawLaunch) *string { return &l.Block },
	"Grid Size":    func(l *RawLaunch) *string { return &l.Grid },
	"Device":       func(l *RawLaunch) *string { return &l.Device },
	"CC":           func(l *RawLaunch) *string { return &l.CC },
}

// ReadRaw reads a raw-page CSV export of at most maxBytes bytes, as written
// by ncu --csv --page raw --log-file, including the ==PROF== lines ncu puts
// before the header. It refuses an export containing an ==ERROR== line (such
// as ERR_NVGPUCTRPERM, profiling not permitted), a header without ID and
// Kernel Name, a duplicate column, rows of the wrong length, and duplicate
// launch IDs, so a failed profile is never read as an empty one.
func ReadRaw(r io.Reader, maxBytes int) (*RawTable, error) {
	if maxBytes <= 0 || maxBytes > MaxBytes {
		return nil, fmt.Errorf("nsightcompute: raw byte budget %d outside 1..%d", maxBytes, MaxBytes)
	}
	data, err := io.ReadAll(io.LimitReader(r, int64(maxBytes)+1))
	if err != nil {
		return nil, fmt.Errorf("nsightcompute: read raw csv: %w", err)
	}
	if len(data) > maxBytes {
		return nil, fmt.Errorf("nsightcompute: raw csv exceeds %d bytes", maxBytes)
	}
	t := &RawTable{}
	off := 0
	for off < len(data) {
		end := bytes.IndexByte(data[off:], '\n')
		if end < 0 {
			end = len(data) - off
		}
		line := strings.TrimSuffix(string(data[off:off+end]), "\r")
		if strings.HasPrefix(line, "==ERROR==") {
			return nil, fmt.Errorf("nsightcompute: ncu reported an error: %s", line)
		}
		if strings.HasPrefix(line, `"ID",`) {
			break
		}
		t.Preamble = append(t.Preamble, line)
		off += end + 1
	}
	if off >= len(data) {
		return nil, errors.New("nsightcompute: raw csv has no header row")
	}
	if i := bytes.Index(data[off:], []byte("\n==ERROR==")); i >= 0 {
		end := bytes.IndexByte(data[off+i+1:], '\n')
		if end < 0 {
			end = len(data) - (off + i + 1)
		}
		return nil, fmt.Errorf("nsightcompute: ncu reported an error: %s", data[off+i+1:off+i+1+end])
	}
	cr := csv.NewReader(bytes.NewReader(data[off:]))
	cr.FieldsPerRecord = 0 // every row must match the header
	records, err := cr.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("nsightcompute: parse raw csv: %w", err)
	}
	if len(records) < 2 {
		return nil, errors.New("nsightcompute: raw csv has no units row")
	}
	if len(records)-2 > MaxRows {
		return nil, fmt.Errorf("nsightcompute: raw csv has more than %d launches", MaxRows)
	}
	header, units := records[0], records[1]
	seen := map[string]bool{}
	var metricCols []int
	for i, name := range header {
		if seen[name] {
			return nil, fmt.Errorf("nsightcompute: duplicate raw column %q", name)
		}
		seen[name] = true
		if _, ok := rawMeta[name]; ok {
			if units[i] != "" {
				return nil, fmt.Errorf("nsightcompute: launch column %q has unit %q", name, units[i])
			}
			continue
		}
		metricCols = append(metricCols, i)
		t.Metrics = append(t.Metrics, name)
		t.Units = append(t.Units, units[i])
	}
	if !seen["ID"] || !seen["Kernel Name"] {
		return nil, errors.New("nsightcompute: raw header lacks ID or Kernel Name")
	}
	ids := map[string]bool{}
	for n, rec := range records[2:] {
		var l RawLaunch
		for i, name := range header {
			if f := rawMeta[name]; f != nil {
				*f(&l) = rec[i]
			}
		}
		if l.ID == "" || ids[l.ID] {
			return nil, fmt.Errorf("nsightcompute: raw row %d: missing or duplicate launch ID %q", n+1, l.ID)
		}
		ids[l.ID] = true
		if l.Kernel == "" {
			return nil, fmt.Errorf("nsightcompute: raw row %d: empty kernel name", n+1)
		}
		l.Raw = make([]string, len(metricCols))
		for j, i := range metricCols {
			l.Raw[j] = rec[i]
		}
		t.Launches = append(t.Launches, l)
	}
	return t, nil
}

// Column returns the index of the named metric in Metrics, or -1.
func (t *RawTable) Column(metric string) int {
	for i, m := range t.Metrics {
		if m == metric {
			return i
		}
	}
	return -1
}

var groupedNumber = regexp.MustCompile(`^-?(\d{1,3}(,\d{3})+|\d+)(\.\d+)?([eE][-+]?\d+)?$`)

// Value returns launch l's value of the named metric and its unit. Thousands
// separators, which ncu writes in raw-page values ("1,980,000"), are
// removed. It returns an error if the metric is absent or its value is not a
// finite number, such as "n/a".
func (t *RawTable) Value(l *RawLaunch, metric string) (float64, string, error) {
	i := t.Column(metric)
	if i < 0 {
		return 0, "", fmt.Errorf("nsightcompute: no raw metric %q", metric)
	}
	s := l.Raw[i]
	if !groupedNumber.MatchString(s) {
		return 0, t.Units[i], fmt.Errorf("nsightcompute: launch %s %s = %q, not a number", l.ID, metric, s)
	}
	v, err := strconv.ParseFloat(strings.ReplaceAll(s, ",", ""), 64)
	if err != nil || math.IsInf(v, 0) {
		return 0, t.Units[i], fmt.Errorf("nsightcompute: launch %s %s = %q, not a finite number", l.ID, metric, s)
	}
	return v, t.Units[i], nil
}
