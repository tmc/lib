package nvidiasmi

import (
	"maps"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"
)

// topoFixtures lists every file in testdata/topo, with values copied
// from the text by hand.
var topoFixtures = []struct {
	file     string
	devices  []string
	columns  []string
	link     [3]string // a, b, Link(a, b)
	affinity [4]string // device, CPU, NUMA, GPU NUMA ID
	nics     map[string]string
}{
	{
		file:     "dgx1.txt",
		devices:  []string{"GPU0", "GPU1", "GPU2", "GPU3", "GPU4", "GPU5", "GPU6", "GPU7", "mlx5_0", "mlx5_2", "mlx5_1", "mlx5_3"},
		columns:  []string{"CPU Affinity"},
		link:     [3]string{"GPU2", "mlx5_1", "PIX"},
		affinity: [4]string{"GPU4", "20-39,60-79", "", ""},
	},
	{
		file:     "nv6x4-numa.txt",
		devices:  []string{"GPU0", "GPU1", "GPU2", "GPU3"},
		columns:  []string{"CPU Affinity", "NUMA Affinity", "GPU NUMA ID"},
		link:     [3]string{"GPU3", "GPU0", "NV6"},
		affinity: [4]string{"GPU1", "72-143", "1,13-19", "12"},
	},
	{
		file:     "h200x1.txt",
		devices:  []string{"GPU0"},
		columns:  []string{"CPU Affinity", "NUMA Affinity", "GPU NUMA ID"},
		link:     [3]string{"GPU0", "GPU0", "X"},
		affinity: [4]string{"GPU0", "0-23", "0", "N/A"},
	},
	{
		file: "h200x8-nic.txt",
		devices: []string{"GPU0", "GPU1", "GPU2", "GPU3", "GPU4", "GPU5", "GPU6", "GPU7",
			"NIC0", "NIC1", "NIC2", "NIC3", "NIC4", "NIC5", "NIC6", "NIC7", "NIC8"},
		columns:  []string{"CPU Affinity", "NUMA Affinity", "GPU NUMA ID"},
		link:     [3]string{"GPU3", "NIC8", "NODE"},
		affinity: [4]string{"GPU7", "84-111,196-223", "3", "N/A"},
		nics: map[string]string{
			"NIC0": "mlx5_0", "NIC1": "mlx5_1", "NIC2": "mlx5_2", "NIC3": "mlx5_5", "NIC4": "mlx5_6",
			"NIC5": "mlx5_7", "NIC6": "mlx5_8", "NIC7": "mlx5_9", "NIC8": "mlx5_bond_0",
		},
	},
	{
		file:     "nv2x4-nic.txt",
		devices:  []string{"GPU0", "GPU1", "GPU2", "GPU3", "NIC0", "NIC1"},
		columns:  []string{"CPU Affinity", "NUMA Affinity", "GPU NUMA ID"},
		link:     [3]string{"NIC1", "GPU2", "PIX"},
		affinity: [4]string{"GPU3", "0,2,4,6,8,10", "0", "N/A"},
		nics:     map[string]string{"NIC0": "mlx5_0", "NIC1": "mlx5_1"},
	},
}

// legend is the legend every fixture prints, apart from older wording
// of PXB and PIX in dgx1.txt.
var legend = map[string]string{
	"X":    "Self",
	"SYS":  "Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)",
	"NODE": "Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node",
	"PHB":  "Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)",
	"PXB":  "Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)",
	"PIX":  "Connection traversing at most a single PCIe bridge",
	"NV#":  "Connection traversing a bonded set of # NVLinks",
}

func parseTopologyFile(t *testing.T, name string) *Topology {
	t.Helper()
	f, err := os.Open(filepath.Join("testdata", "topo", name))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	topo, err := ParseTopology(f)
	if err != nil {
		t.Fatal(err)
	}
	return topo
}

func TestTopologyFixtures(t *testing.T) {
	for _, tt := range topoFixtures {
		t.Run(tt.file, func(t *testing.T) {
			topo := parseTopologyFile(t, tt.file)
			var names []string
			for _, d := range topo.Devices {
				names = append(names, d.Name)
			}
			if !slices.Equal(names, tt.devices) {
				t.Errorf("Devices = %q, want %q", names, tt.devices)
			}
			if !slices.Equal(topo.Columns, tt.columns) {
				t.Errorf("Columns = %q, want %q", topo.Columns, tt.columns)
			}
			if l, ok := topo.Link(tt.link[0], tt.link[1]); !ok || string(l) != tt.link[2] {
				t.Errorf("Link(%s, %s) = %q, %v, want %q", tt.link[0], tt.link[1], l, ok, tt.link[2])
			}
			i := topo.Index(tt.affinity[0])
			if i < 0 {
				t.Fatalf("no device %s", tt.affinity[0])
			}
			d := topo.Devices[i]
			if got := [4]string{d.Name, string(d.CPUAffinity), string(d.NUMAAffinity), string(d.GPUNUMAID)}; got != tt.affinity {
				t.Errorf("Devices[%d] = %q, want %q", i, got, tt.affinity)
			}
			if !reflect.DeepEqual(topo.NICs, tt.nics) {
				t.Errorf("NICs = %v, want %v", topo.NICs, tt.nics)
			}
			want := legend
			if tt.file == "dgx1.txt" {
				want = maps.Clone(legend)
				want["PXB"] = "Connection traversing multiple PCIe switches (without traversing the PCIe Host Bridge)"
				want["PIX"] = "Connection traversing a single PCIe switch"
			}
			if !reflect.DeepEqual(topo.Legend, want) {
				t.Errorf("Legend = %v, want %v", topo.Legend, want)
			}
		})
	}
}

func TestTopologyFixturesListed(t *testing.T) {
	files, err := filepath.Glob(filepath.Join("testdata", "topo", "*.txt"))
	if err != nil {
		t.Fatal(err)
	}
	listed := make(map[string]bool)
	for _, f := range topoFixtures {
		listed[f.file] = true
	}
	for _, f := range files {
		if !listed[filepath.Base(f)] {
			t.Errorf("%s is not in topoFixtures", f)
		}
	}
	if len(files) != len(topoFixtures) {
		t.Errorf("testdata/topo has %d files, topoFixtures lists %d", len(files), len(topoFixtures))
	}
}

// TestTopologyInvariants checks every fixture's matrix: symmetric,
// X exactly on the diagonal, and every cell a legend link type.
func TestTopologyInvariants(t *testing.T) {
	for _, tt := range topoFixtures {
		t.Run(tt.file, func(t *testing.T) {
			topo := parseTopologyFile(t, tt.file)
			for i, row := range topo.Links {
				if len(row) != len(topo.Devices) {
					t.Fatalf("row %d has %d links", i, len(row))
				}
				for j, l := range row {
					if l != topo.Links[j][i] {
						t.Errorf("Links[%d][%d] = %s, Links[%d][%d] = %s", i, j, l, j, i, topo.Links[j][i])
					}
					if (l == LinkSelf) != (i == j) {
						t.Errorf("Links[%d][%d] = %s", i, j, l)
					}
					key := string(l)
					if _, ok := l.NVLinks(); ok {
						key = "NV#"
					}
					if _, ok := topo.Legend[key]; !ok {
						t.Errorf("Links[%d][%d] = %s, not in legend", i, j, l)
					}
				}
				d := topo.Devices[i]
				isGPU := strings.HasPrefix(d.Name, "GPU")
				if _, ok := d.CPUAffinity.IntList(); ok != isGPU {
					t.Errorf("%s: CPUAffinity %q IntList ok = %v", d.Name, d.CPUAffinity, ok)
				}
			}
		})
	}
}

func TestTopologyNVLinks(t *testing.T) {
	topo := parseTopologyFile(t, "dgx1.txt")
	var counts []int
	for _, l := range topo.Links[0][:8] {
		n, _ := l.NVLinks()
		counts = append(counts, n)
	}
	if want := []int{0, 1, 1, 2, 2, 0, 0, 0}; !slices.Equal(counts, want) {
		t.Errorf("GPU0 NVLinks = %v, want %v", counts, want)
	}
}

func TestLinkNVLinks(t *testing.T) {
	tests := []struct {
		l  Link
		n  int
		ok bool
	}{
		{"NV1", 1, true},
		{"NV18", 18, true},
		{"NV", 0, false},
		{"NV#", 0, false},
		{"NV-1", 0, false},
		{"NV+1", 0, false},
		{"NODE", 0, false},
		{"X", 0, false},
		{"nv4", 0, false},
	}
	for _, tt := range tests {
		n, ok := tt.l.NVLinks()
		if n != tt.n || ok != tt.ok {
			t.Errorf("Link(%q).NVLinks() = %d, %v, want %d, %v", tt.l, n, ok, tt.n, tt.ok)
		}
	}
}

func TestTopologyLinkMissing(t *testing.T) {
	topo := parseTopologyFile(t, "nv6x4-numa.txt")
	if l, ok := topo.Link("GPU0", "GPU4"); ok {
		t.Errorf("Link(GPU0, GPU4) = %q, true", l)
	}
	if i := topo.Index("NIC0"); i != -1 {
		t.Errorf("Index(NIC0) = %d", i)
	}
}

func TestTopologyOtherColumn(t *testing.T) {
	const in = "\tGPU0\tGPU1\tCPU Affinity\tFuture Column\n" +
		"GPU0\t X \tNV4\t0-7\tyes\n" +
		"GPU1\tNV4\t X \t8-15\tno\n"
	topo, err := ParseTopology(strings.NewReader(in))
	if err != nil {
		t.Fatal(err)
	}
	d := topo.Devices[1]
	if d.CPUAffinity != "8-15" || d.Other["Future Column"] != "no" || len(d.Other) != 1 {
		t.Errorf("Devices[1] = %+v", d)
	}
	if topo.Legend != nil || topo.NICs != nil {
		t.Errorf("Legend = %v, NICs = %v, want nil", topo.Legend, topo.NICs)
	}
}

func TestParseTopologyErrors(t *testing.T) {
	tests := []struct {
		name, in, want string
	}{
		{"empty", "", "no matrix"},
		{"blank", "\n\n", "no matrix"},
		{"no rows", "\tGPU0\tCPU Affinity\n\nLegend:\n", "no matrix rows"},
		{"short heading", "\tGPU0\nGPU0\t X \nGPU1\tSYS\n", "heading has 1 columns"},
		{"row name", "\tGPU0\tGPU1\nGPU0\t X \tSYS\nGPU2\tSYS\t X \n", "row 1 is GPU2, but column 1 is GPU1"},
		{"short row", "\tGPU0\tGPU1\nGPU0\t X \nGPU1\tSYS\t X \n", "row GPU0 has 1 links, want 2"},
		{"self", "\tGPU0\tGPU1\nGPU0\tSYS\tSYS\nGPU1\tSYS\t X \n", "row GPU0 links to itself by SYS"},
		{"extra values", "\tGPU0\tCPU Affinity\nGPU0\t X \t0-3\t7\n", "row GPU0 has 2 affinity values, want at most 1"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			topo, err := ParseTopology(strings.NewReader(tt.in))
			if err == nil {
				t.Fatalf("ParseTopology succeeded: %+v", topo)
			}
			if !strings.HasPrefix(err.Error(), "parse nvidia-smi topology: ") || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("err = %q, want it to contain %q", err, tt.want)
			}
		})
	}
}

// TestTopologySpaces checks that a copy whose tabs became spaces
// parses like the original.
func TestTopologySpaces(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("testdata", "topo", "h200x8-nic.txt"))
	if err != nil {
		t.Fatal(err)
	}
	tabs, err := ParseTopology(strings.NewReader(string(data)))
	if err != nil {
		t.Fatal(err)
	}
	spaced := strings.ReplaceAll(ansiEscape.ReplaceAllString(string(data), ""), "\t", "    ")
	spaces, err := ParseTopology(strings.NewReader(spaced))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(tabs, spaces) {
		t.Errorf("tab and space forms differ:\n%+v\n%+v", tabs, spaces)
	}
}

func TestQueryTopologyFake(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("testdata", "topo", "nv2x4-nic.txt"))
	if err != nil {
		t.Fatal(err)
	}
	args := fake(t, string(data), "", 0)
	topo, err := QueryTopology(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(args(), " "); got != "topo -m" {
		t.Errorf("ran nvidia-smi %s, want topo -m", got)
	}
	if len(topo.Devices) != 6 {
		t.Errorf("len(Devices) = %d, want 6", len(topo.Devices))
	}
}
