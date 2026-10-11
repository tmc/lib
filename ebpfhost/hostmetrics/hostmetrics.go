package hostmetrics

import (
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

// Context identifies the source and the caller's query interval. Query times
// are not sensor timestamps. Version names the source kernel or exporter.
type Context struct {
	NodeID, SourceID, Version, ClockDomain, ClockEpoch string
	QueryStartNanos, QueryEndNanos                     int64
}

// Metric preserves integer counters exactly. Resource is the explicitly
// selected interface, RDMA port, block device, or pressure resource.
type Metric struct {
	Name, Unit, Resource string
	Value                float64
	Integer              bool
	UintValue            uint64
}

// Snapshot is version 1 node evidence. A failed read returns no snapshot;
// missing fields are absent, never synthesized as zero.
type Snapshot struct {
	Version int
	Context Context
	Metrics []Metric
}

func identity(s string) bool {
	if s == "" || len(s) > 256 || !utf8.ValidString(s) {
		return false
	}
	for _, r := range s {
		if unicode.IsControl(r) || unicode.IsSpace(r) {
			return false
		}
	}
	return true
}
func (c Context) validate() error {
	for _, s := range []string{c.NodeID, c.SourceID, c.Version, c.ClockDomain, c.ClockEpoch} {
		if !identity(s) {
			return fmt.Errorf("invalid node source identity")
		}
	}
	if c.QueryStartNanos < 0 || c.QueryEndNanos < c.QueryStartNanos {
		return fmt.Errorf("invalid node query interval")
	}
	return nil
}
func read(r io.Reader, c Context, maxBytes int64) (string, error) {
	if err := c.validate(); err != nil {
		return "", err
	}
	if r == nil || maxBytes <= 0 || maxBytes > 1<<20 {
		return "", fmt.Errorf("invalid node reader or byte limit")
	}
	b, err := io.ReadAll(io.LimitReader(r, maxBytes+1))
	if err != nil {
		return "", fmt.Errorf("read node statistics: %w", err)
	}
	if int64(len(b)) > maxBytes {
		return "", fmt.Errorf("node statistics exceed byte limit")
	}
	if len(b) == 0 {
		return "", fmt.Errorf("empty node statistics")
	}
	return string(b), nil
}
func integer(s string) (uint64, error) {
	if s == "" {
		return 0, fmt.Errorf("missing integer")
	}
	for _, r := range s {
		if r < '0' || r > '9' {
			return 0, fmt.Errorf("invalid unsigned integer")
		}
	}
	return strconv.ParseUint(s, 10, 64)
}
func result(c Context, m []Metric) (Snapshot, error) { s := Snapshot{1, c, m}; return s, s.Validate() }

// Validate checks identities, supported units, finite values and duplicates.
func (s Snapshot) Validate() error {
	if s.Version != 1 {
		return fmt.Errorf("unsupported node snapshot version")
	}
	if err := s.Context.validate(); err != nil {
		return err
	}
	if len(s.Metrics) == 0 || len(s.Metrics) > 64 {
		return fmt.Errorf("invalid node metric count")
	}
	seen := map[string]bool{}
	for _, m := range s.Metrics {
		unit, ok := Unit(m.Name)
		if !ok || m.Unit != unit || !identity(m.Resource) {
			return fmt.Errorf("invalid node metric identity or unit")
		}
		if strings.HasPrefix(m.Name, "node.psi.") && strings.Split(m.Name, ".")[2] != m.Resource {
			return fmt.Errorf("pressure resource mismatch")
		}
		if strings.HasPrefix(m.Name, "node.memory.") && m.Resource != "memory" {
			return fmt.Errorf("memory resource mismatch")
		}
		k := m.Name + "\x00" + m.Resource
		if seen[k] {
			return fmt.Errorf("duplicate node metric")
		}
		seen[k] = true
		if math.IsNaN(m.Value) || math.IsInf(m.Value, 0) || (m.Value < 0 && unit != "Cel") || (!m.Integer && m.UintValue != 0) || (m.Integer && m.Value != float64(m.UintValue)) {
			return fmt.Errorf("invalid node metric value")
		}
		if unit == "percent" && (m.Integer || m.Value > 100) {
			return fmt.Errorf("invalid pressure percentage")
		}
		if unit != "percent" && unit != "W" && unit != "Cel" && !m.Integer {
			return fmt.Errorf("node counter requires exact integer")
		}
	}
	return nil
}

// Unit returns the canonical unit of a supported node field.
func Unit(name string) (string, bool) {
	p := strings.Split(name, ".")
	if len(p) == 5 && p[0] == "node" && p[1] == "psi" && p[2] == "cpu" && p[3] == "full" {
		return "", false
	}
	if len(p) == 5 && p[0] == "node" && p[1] == "psi" && (p[2] == "cpu" || p[2] == "memory" || p[2] == "io") && (p[3] == "some" || p[3] == "full") {
		switch p[4] {
		case "avg10", "avg60", "avg300":
			return "percent", true
		case "total":
			return "us", true
		}
	}
	switch name {
	case "node.bmc.power":
		return "W", true
	case "node.bmc.temperature":
		return "Cel", true
	case "node.memory.MemTotal", "node.memory.MemAvailable", "node.memory.MemFree", "node.memory.SwapTotal", "node.memory.SwapFree", "node.net.rx_bytes", "node.net.tx_bytes":
		return "bytes", true
	case "node.net.rx_packets", "node.net.tx_packets", "node.rdma.port_rcv_packets", "node.rdma.port_xmit_packets":
		return "packets", true
	case "node.net.rx_errors", "node.net.tx_errors", "node.block.reads_completed", "node.block.writes_completed":
		return "count", true
	case "node.rdma.port_rcv_data", "node.rdma.port_xmit_data":
		return "4-byte_words", true
	case "node.block.sectors_read", "node.block.sectors_written":
		return "512-byte_sectors", true
	}
	return "", false
}

// ReadPSI reads /proc/pressure/{cpu,memory,io}. The system CPU full line is
// undefined by Linux and is omitted even when the kernel emits compatibility
// zeros. This reader does not accept cgroup PSI semantics.
func ReadPSI(r io.Reader, c Context, resource string, maxBytes int64) (Snapshot, error) {
	if resource != "cpu" && resource != "memory" && resource != "io" {
		return Snapshot{}, fmt.Errorf("invalid pressure resource")
	}
	b, err := read(r, c, maxBytes)
	if err != nil {
		return Snapshot{}, err
	}
	seen := map[string]bool{}
	var out []Metric
	for _, line := range strings.Split(strings.TrimSpace(b), "\n") {
		f := strings.Fields(line)
		if len(f) != 5 || (f[0] != "some" && f[0] != "full") || seen[f[0]] {
			return Snapshot{}, fmt.Errorf("invalid pressure row")
		}
		seen[f[0]] = true
		for i, key := range []string{"avg10", "avg60", "avg300", "total"} {
			kv := strings.Split(f[i+1], "=")
			if len(kv) != 2 || kv[0] != key {
				return Snapshot{}, fmt.Errorf("invalid pressure field")
			}
			m := Metric{Name: "node.psi." + resource + "." + f[0] + "." + key, Resource: resource}
			m.Unit, _ = Unit(m.Name)
			if key == "total" {
				m.UintValue, err = integer(kv[1])
				m.Integer = true
				m.Value = float64(m.UintValue)
			} else {
				m.Value, err = strconv.ParseFloat(kv[1], 64)
			}
			if err != nil || math.IsNaN(m.Value) || math.IsInf(m.Value, 0) || m.Value < 0 || (key != "total" && m.Value > 100) {
				return Snapshot{}, fmt.Errorf("invalid pressure value")
			}
			if resource != "cpu" || f[0] != "full" {
				out = append(out, m)
			}
		}
	}
	if !seen["some"] || (resource != "cpu" && !seen["full"]) {
		return Snapshot{}, fmt.Errorf("missing pressure row")
	}
	return result(c, out)
}

// ReadMeminfo reads selected fields from /proc/meminfo. Linux kB values are
// converted to bytes with overflow checks; absent selected fields stay absent.
func ReadMeminfo(r io.Reader, c Context, maxBytes int64) (Snapshot, error) {
	b, err := read(r, c, maxBytes)
	if err != nil {
		return Snapshot{}, err
	}
	seen := map[string]bool{}
	var out []Metric
	for _, line := range strings.Split(strings.TrimSpace(b), "\n") {
		f := strings.Fields(line)
		if len(f) < 2 || !strings.HasSuffix(f[0], ":") {
			return Snapshot{}, fmt.Errorf("invalid meminfo row")
		}
		key := strings.TrimSuffix(f[0], ":")
		if seen[key] {
			return Snapshot{}, fmt.Errorf("duplicate meminfo field")
		}
		seen[key] = true
		name := "node.memory." + key
		if _, ok := Unit(name); !ok {
			continue
		}
		if len(f) != 3 || f[2] != "kB" {
			return Snapshot{}, fmt.Errorf("invalid meminfo unit")
		}
		v, e := integer(f[1])
		if e != nil || v > math.MaxUint64/1024 {
			return Snapshot{}, fmt.Errorf("invalid meminfo value")
		}
		v *= 1024
		out = append(out, Metric{name, "bytes", "memory", float64(v), true, v})
	}
	if !seen["MemTotal"] {
		return Snapshot{}, fmt.Errorf("missing MemTotal")
	}
	return result(c, out)
}

// ReadCounter reads one explicitly selected network or RDMA sysfs counter.
// RDMA data units are four-byte words, not bytes. The caller selects and opens
// the path; this function never enumerates interfaces or devices.
func ReadCounter(r io.Reader, c Context, name, resource string, maxBytes int64) (Snapshot, error) {
	if !strings.HasPrefix(name, "node.net.") && !strings.HasPrefix(name, "node.rdma.") {
		return Snapshot{}, fmt.Errorf("unsupported scalar counter")
	}
	unit, ok := Unit(name)
	if !ok {
		return Snapshot{}, fmt.Errorf("unsupported scalar counter")
	}
	b, err := read(r, c, maxBytes)
	if err != nil {
		return Snapshot{}, err
	}
	v, err := integer(strings.TrimSpace(b))
	if err != nil {
		return Snapshot{}, fmt.Errorf("invalid scalar counter: %w", err)
	}
	return result(c, []Metric{{name, unit, resource, float64(v), true, v}})
}

// ReadBlockStat reads /sys/block/<selected-device>/stat. It accepts Linux's
// documented 11, 15 or 17 fields and preserves sectors as 512-byte sectors.
func ReadBlockStat(r io.Reader, c Context, resource string, maxBytes int64) (Snapshot, error) {
	b, err := read(r, c, maxBytes)
	if err != nil {
		return Snapshot{}, err
	}
	f := strings.Fields(b)
	if len(f) != 11 && len(f) != 15 && len(f) != 17 {
		return Snapshot{}, fmt.Errorf("invalid block statistic field count")
	}
	values := make([]uint64, len(f))
	for i, s := range f {
		values[i], err = integer(s)
		if err != nil {
			return Snapshot{}, fmt.Errorf("invalid block statistic")
		}
	}
	var out []Metric
	for _, x := range []struct {
		name  string
		index int
	}{{"reads_completed", 0}, {"sectors_read", 2}, {"writes_completed", 4}, {"sectors_written", 6}} {
		name := "node.block." + x.name
		unit, _ := Unit(name)
		v := values[x.index]
		out = append(out, Metric{name, unit, resource, float64(v), true, v})
	}
	return result(c, out)
}
