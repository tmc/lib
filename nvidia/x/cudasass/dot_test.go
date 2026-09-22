package cudasass

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// sampleDOT is written here, not by nvdisasm: it has the shape a caller that
// emits its own DOT would, one digraph per function. See
// TestClustersAreTheGraphs for what the disassembler actually prints.
const sampleDOT = `// one digraph per function
digraph _Z7computev {
	bb0 [shape=box, label="[0000] MOV R1, c[0x0][0x28];\n[0010] RET;"];
}

digraph "vector_add" {
	bb0 [shape=box, label="[0000] LDC R1, 0x10;\n[0010] BRA bb1;"];
	bb1 [shape=box, label="[0020] EXIT;"];
	bb0 -> bb1;
}
`

func TestListDOTGraphs(t *testing.T) {
	graphs := ListDOTGraphs(sampleDOT)
	want := []string{"_Z7computev", "vector_add"}
	if len(graphs) != len(want) {
		t.Fatalf("ListDOTGraphs got %v, want %v", graphs, want)
	}
	for i := range want {
		if graphs[i] != want[i] {
			t.Errorf("graph[%d] = %q, want %q", i, graphs[i], want[i])
		}
	}
}

func TestExtractDOTGraph(t *testing.T) {
	tests := []struct {
		name      string
		kernel    string
		wantSub   string
		wantError string
	}{
		{
			name:    "all graphs when empty",
			kernel:  "",
			wantSub: "digraph _Z7computev",
		},
		{
			name:    "exact match",
			kernel:  "vector_add",
			wantSub: "digraph \"vector_add\"",
		},
		{
			name:    "mangled substring match",
			kernel:  "compute",
			wantSub: "digraph _Z7computev",
		},
		{
			name:      "not found",
			kernel:    "missing_kernel",
			wantError: `kernel "missing_kernel" not found in control flow graph`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ExtractDOTGraph(sampleDOT, tt.kernel)
			if tt.wantError != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantError) {
					t.Fatalf("ExtractDOTGraph(%q) error = %v, want %q", tt.kernel, err, tt.wantError)
				}
				return
			}
			if err != nil {
				t.Fatalf("ExtractDOTGraph(%q) unexpected error: %v", tt.kernel, err)
			}
			if !strings.Contains(got, tt.wantSub) {
				t.Errorf("ExtractDOTGraph(%q) = %q, want substring %q", tt.kernel, got, tt.wantSub)
			}
		})
	}
}

func TestExtractDOTGraphAmbiguous(t *testing.T) {
	multi := `digraph compute_1 { bb0; }
digraph compute_2 { bb0; }`
	_, err := ExtractDOTGraph(multi, "compute")
	if err == nil || !strings.Contains(err.Error(), "ambiguous") {
		t.Fatalf("expected ambiguous error, got %v", err)
	}
}

func TestExtractDOTGraphEmpty(t *testing.T) {
	_, err := ExtractDOTGraph("", "")
	if err == nil {
		t.Fatal("expected error on empty DOT text")
	}
}

// TestClustersAreTheGraphs reads nvdisasm's own output. Every graph nvdisasm
// prints, for one kernel or for a whole module, is a single digraph named "f"
// holding a "cluster_<function>" subgraph per function, so the top-level name
// identifies nothing a caller can ask for. Treating it as the graph name makes
// ListDOTGraphs report "f" for every real capture and ExtractDOTGraph refuse
// every kernel in it, which is what cudaprof inspect --cfg did.
func TestClustersAreTheGraphs(t *testing.T) {
	b, err := os.ReadFile(filepath.Join("testdata", "roundtrip", "sm121a-roundtrip.bbcfg.dot"))
	if err != nil {
		t.Fatalf("read bbcfg fixture: %v", err)
	}
	dot := string(b)
	if !strings.Contains(dot, "digraph f {") {
		t.Fatal("fixture is not nvdisasm output: no module-level digraph f")
	}
	want := []string{"intmix", "addk"}
	got := ListDOTGraphs(dot)
	if len(got) != len(want) {
		t.Fatalf("ListDOTGraphs = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("graph[%d] = %q, want %q", i, got[i], want[i])
		}
	}

	g, err := ExtractDOTGraph(dot, "addk")
	if err != nil {
		t.Fatalf("ExtractDOTGraph(addk): %v", err)
	}
	// The cluster is rewrapped as a digraph of its own, because a bare
	// subgraph is not a document any DOT renderer accepts.
	if !strings.HasPrefix(g, "digraph addk {") {
		t.Errorf("extracted graph starts %q, want a digraph of its own", strings.SplitN(g, "\n", 2)[0])
	}
	if strings.Contains(g, "subgraph") {
		t.Error("extracted graph still carries a subgraph wrapper")
	}
	// Neighbouring kernels must not come along: nvdisasm puts every node and
	// edge of a function inside its own cluster, so the cut is exact.
	if strings.Contains(g, "intmix") {
		t.Error("extracted graph carries the other kernel's nodes")
	}
	if !strings.Contains(g, ".text.addk") {
		t.Error("extracted graph does not carry the kernel's own text")
	}
}

// FuzzDOTGraphs checks the two halves of the DOT reader against each other.
//
// The seeds are the disassembler's own output, read from the fixture rather
// than copied into a corpus directory, so refreshing the fixture refreshes the
// seed and the two cannot drift. testdata/fuzz holds only the inputs a failure
// minimised to.
func FuzzDOTGraphs(f *testing.F) {
	b, err := os.ReadFile(filepath.Join("testdata", "roundtrip", "sm121a-roundtrip.bbcfg.dot"))
	if err != nil {
		f.Fatalf("read bbcfg fixture: %v", err)
	}
	f.Add(string(b), "addk")
	f.Add(sampleDOT, "vector_add")
	f.Add("", "")

	f.Fuzz(func(t *testing.T, dot, name string) {
		// A name ListDOTGraphs reports must be one ExtractDOTGraph returns,
		// and it must return that graph alone. The two disagreeing is how
		// cudaprof came to refuse every real capture: nvdisasm names the
		// module "f" and puts the functions in clusters, so what was listed
		// named nothing a caller could ask for.
		for _, listed := range ListDOTGraphs(dot) {
			g, err := ExtractDOTGraph(dot, listed)
			if err != nil {
				t.Fatalf("listed %q but could not extract it: %v", listed, err)
			}
			assertReadsBack(t, listed, g)
			if got := ListDOTGraphs(g); (len(got) != 1 || got[0] != listed) && !hasCluster(g) {
				// A rewrapped body holding clusters of its own is named by
				// them and not by the graph around them, which is the
				// convention ListDOTGraphs documents. nvdisasm emits one
				// flat list of clusters and never nests them.
				t.Fatalf("extracted %q reads back as %v, want that one graph", listed, got)
			}
		}
		// An arbitrary name either fails or yields a graph of its own. A
		// name that is empty once unquoted asks for the whole document
		// instead, which is not required to hold a named graph at all.
		if dotUnquote(name) == "" {
			return
		}
		if g, err := ExtractDOTGraph(dot, name); err == nil {
			assertReadsBack(t, name, g)
		}
	})
}

// hasCluster reports whether dot holds a cluster subgraph.
func hasCluster(dot string) bool {
	for _, s := range parseDOTGraphs(dot) {
		if s.cluster {
			return true
		}
	}
	return false
}

// assertReadsBack reports whether g is a DOT document the reader accepts on
// its own. A cluster is rewrapped under the function's name, and a name
// needing quotes would otherwise produce a header nothing reads back. Whether
// that name is one a caller could ask for again is a separate question, asked
// of the listed graphs only.
func assertReadsBack(t *testing.T, name, g string) {
	t.Helper()
	head := strings.SplitN(g, "\n", 2)[0]
	if !strings.HasPrefix(g, "digraph") {
		t.Fatalf("extracted %q is not a digraph: %q", name, head)
	}
	if len(parseDOTGraphs(g)) == 0 {
		t.Fatalf("extracted %q does not read back as a graph: %q", name, head)
	}
}
