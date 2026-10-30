package nvidiasmi

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"regexp"
	"strconv"
	"strings"
)

// A Link is one cell of the nvidia-smi topo -m matrix: how traffic
// between two devices travels. Cells other than the constants below
// and NV# are kept as printed.
type Link string

// Link types from the nvidia-smi topo -m legend.
const (
	LinkSelf Link = "X"    // the device itself
	LinkSYS  Link = "SYS"  // PCIe and the SMP interconnect between NUMA nodes (QPI/UPI)
	LinkNode Link = "NODE" // PCIe and the interconnect between PCIe host bridges within a NUMA node
	LinkPHB  Link = "PHB"  // PCIe and a PCIe host bridge (typically the CPU)
	LinkPXB  Link = "PXB"  // multiple PCIe bridges, without the PCIe host bridge
	LinkPIX  Link = "PIX"  // at most a single PCIe bridge
)

// NVLinks returns the number of bonded NVLinks in an NV# link,
// such as 12 for "NV12". It reports ok=false for any other link.
func (l Link) NVLinks() (n int, ok bool) {
	s, ok := strings.CutPrefix(string(l), "NV")
	if !ok || s == "" || s[0] < '0' || s[0] > '9' {
		return 0, false
	}
	n, err := strconv.Atoi(s)
	if err != nil {
		return 0, false
	}
	return n, true
}

// A Topology is the output of nvidia-smi topo -m: a matrix of links
// between GPUs and NICs, the CPU and memory affinity of each GPU, and
// the legends printed below the matrix.
type Topology struct {
	// Devices lists the matrix rows, in order.
	Devices []TopologyDevice

	// Links[i][j] is the link between Devices[i] and Devices[j].
	Links [][]Link

	// Columns lists the affinity column headings, such as
	// "CPU Affinity", "NUMA Affinity", and "GPU NUMA ID".
	Columns []string

	// Legend maps each link type in the legend, such as "PIX" or
	// "NV#", to its description.
	Legend map[string]string

	// NICs maps a NIC row name, such as "NIC0", to the device name
	// in the NIC legend, such as "mlx5_0". Older drivers print the
	// device name as the row name and no NIC legend.
	NICs map[string]string
}

// A TopologyDevice is one row of a [Topology].
type TopologyDevice struct {
	// Name is the row name, such as "GPU0", "NIC1", or "mlx5_0".
	Name string

	// CPUAffinity, NUMAAffinity, and GPUNUMAID are the columns of
	// the same names. NICs leave them empty. See [Value.IntList].
	CPUAffinity  Value
	NUMAAffinity Value
	GPUNUMAID    Value

	// Other holds affinity columns with no field above,
	// keyed by heading.
	Other map[string]Value
}

// Index returns the index of the device named name in t.Devices,
// or -1 if there is none.
func (t *Topology) Index(name string) int {
	for i, d := range t.Devices {
		if d.Name == name {
			return i
		}
	}
	return -1
}

// Link returns the link between the devices named a and b.
// It reports ok=false if either is not in t.
func (t *Topology) Link(a, b string) (l Link, ok bool) {
	i, j := t.Index(a), t.Index(b)
	if i < 0 || j < 0 {
		return "", false
	}
	return t.Links[i][j], true
}

var (
	ansiEscape = regexp.MustCompile("\x1b\\[[0-9;]*[A-Za-z]")
	wideSpace  = regexp.MustCompile(`\s{2,}|\t`)
)

// ParseTopology reads the output of nvidia-smi topo -m from r.
//
// It accepts the tab-separated text nvidia-smi writes, with or
// without the terminal escape codes it adds to the heading, and copies
// in which tabs became runs of spaces. It returns an error if the
// matrix is not square, a row name does not match its column heading,
// or a device's link to itself is not X.
func ParseTopology(r io.Reader) (*Topology, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("parse nvidia-smi topology: %w", err)
	}
	t, err := parseTopology(ansiEscape.ReplaceAll(data, nil))
	if err != nil {
		return nil, fmt.Errorf("parse nvidia-smi topology: %w", err)
	}
	return t, nil
}

func parseTopology(data []byte) (*Topology, error) {
	var lines []string
	s := bufio.NewScanner(bytes.NewReader(data))
	for s.Scan() {
		lines = append(lines, strings.TrimRight(s.Text(), "\r"))
	}
	if err := s.Err(); err != nil {
		return nil, err
	}
	for len(lines) > 0 && strings.TrimSpace(lines[0]) == "" {
		lines = lines[1:]
	}
	if len(lines) == 0 {
		return nil, errors.New("no matrix")
	}

	// The heading names each device column and then each affinity
	// column. Device names have no spaces; affinity headings do,
	// so split on tabs or runs of spaces.
	var heading []string
	for _, h := range wideSpace.Split(strings.TrimSpace(lines[0]), -1) {
		if h = strings.TrimSpace(h); h != "" {
			heading = append(heading, h)
		}
	}
	lines = lines[1:]

	// Rows run to the first blank line. Cells have no spaces.
	var rows [][]string
	for len(lines) > 0 && strings.TrimSpace(lines[0]) != "" {
		rows = append(rows, strings.Fields(lines[0]))
		lines = lines[1:]
	}
	n := len(rows)
	if n == 0 {
		return nil, errors.New("no matrix rows")
	}
	if len(heading) < n {
		return nil, fmt.Errorf("heading has %d columns, want at least %d for %d rows", len(heading), n, n)
	}
	t := &Topology{
		Devices: make([]TopologyDevice, n),
		Links:   make([][]Link, n),
		Columns: heading[n:],
	}
	for i, row := range rows {
		if row[0] != heading[i] {
			return nil, fmt.Errorf("row %d is %s, but column %d is %s", i, row[0], i, heading[i])
		}
		if len(row) < 1+n {
			return nil, fmt.Errorf("row %s has %d links, want %d", row[0], len(row)-1, n)
		}
		affinity := row[1+n:]
		if len(affinity) > len(t.Columns) {
			return nil, fmt.Errorf("row %s has %d affinity values, want at most %d", row[0], len(affinity), len(t.Columns))
		}
		d := TopologyDevice{Name: row[0]}
		for k, v := range affinity {
			switch col := t.Columns[k]; col {
			case "CPU Affinity":
				d.CPUAffinity = Value(v)
			case "NUMA Affinity":
				d.NUMAAffinity = Value(v)
			case "GPU NUMA ID":
				d.GPUNUMAID = Value(v)
			default:
				if d.Other == nil {
					d.Other = make(map[string]Value)
				}
				d.Other[col] = Value(v)
			}
		}
		links := make([]Link, n)
		for j, c := range row[1 : 1+n] {
			links[j] = Link(c)
		}
		if links[i] != LinkSelf {
			return nil, fmt.Errorf("row %s links to itself by %s, want X", row[0], links[i])
		}
		t.Devices[i] = d
		t.Links[i] = links
	}

	// The legends follow, each a heading line ending in a colon and
	// then indented "KEY = text" or "KEY: name" lines. Other text is
	// ignored.
	var section string
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		switch {
		case trimmed == "":
		case trimmed == "Legend:", trimmed == "NIC Legend:":
			section = trimmed
		case section == "Legend:":
			if k, v, ok := strings.Cut(trimmed, "="); ok {
				if t.Legend == nil {
					t.Legend = make(map[string]string)
				}
				t.Legend[strings.TrimSpace(k)] = strings.TrimSpace(v)
			}
		case section == "NIC Legend:":
			if k, v, ok := strings.Cut(trimmed, ":"); ok {
				if t.NICs == nil {
					t.NICs = make(map[string]string)
				}
				t.NICs[strings.TrimSpace(k)] = strings.TrimSpace(v)
			}
		}
	}
	return t, nil
}

// QueryTopology runs nvidia-smi topo -m, found in $PATH, and parses
// its output. nvidia-smi supports topo only on Linux.
func QueryTopology(ctx context.Context) (*Topology, error) {
	out, err := run(ctx, "topo", "-m")
	if err != nil {
		return nil, err
	}
	return ParseTopology(bytes.NewReader(out))
}
