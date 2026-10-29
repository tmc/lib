package hopper

import (
	"bufio"
	"bytes"
	"cmp"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"
)

var updateCensus = flag.Bool("update-census", false, "recompile goldens with $CUDACOMPILECHECK_NVCC and rewrite testdata/golden/*.census.json")

// A ptxCensus counts one kernel's global memory instructions in its PTX:
// ld.global and st.global by their other qualifiers ("nc.v4.u32"), and
// loads and stores through generic addresses, which may be global, as
// "generic." and theirs.
type ptxCensus struct {
	Loads  map[string]int `json:"loads"`
	Stores map[string]int `json:"stores"`
	// LateLoads counts ld.global (other than .cg, which reads other
	// blocks' results) after the kernel's first bar.sync, in program order.
	LateLoads int `json:"late_loads"`
}

// A goldenCensus is the census of each kernel in a golden translation unit
// and the sha256 of the source it was taken from.
type goldenCensus struct {
	Source  string               `json:"source_sha256"`
	Kernels map[string]ptxCensus `json:"kernels"`
}

var (
	ptxEntry = regexp.MustCompile(`^\.visible \.entry (\w+)\(`)
	ptxMem   = regexp.MustCompile(`^\s*(?:@!?%\w+\s+)?(ld|st)((?:\.[\w:]+)*)\s`)
	ptxBar   = regexp.MustCompile(`\bbar\.sync\b|\bbarrier\.sync\b`)
)

// censusPTX counts the global loads and stores of each .entry in ptx.
func censusPTX(ptx []byte) map[string]ptxCensus {
	out := make(map[string]ptxCensus)
	var name string
	var c ptxCensus
	barrier := false
	flush := func() {
		if name != "" {
			out[name] = c
		}
	}
	sc := bufio.NewScanner(bytes.NewReader(ptx))
	sc.Buffer(nil, 1<<20)
	for sc.Scan() {
		line := sc.Text()
		if m := ptxEntry.FindStringSubmatch(line); m != nil {
			flush()
			name, c, barrier = m[1], ptxCensus{Loads: map[string]int{}, Stores: map[string]int{}}, false
			continue
		}
		if name == "" {
			continue
		}
		if ptxBar.MatchString(line) {
			barrier = true
		}
		m := ptxMem.FindStringSubmatch(line)
		if m == nil {
			continue
		}
		q, ok := strings.CutPrefix(m[2], ".global.")
		if !ok {
			space := strings.SplitN(m[2], ".", 3)[1]
			if space == "shared" || space == "local" || space == "param" || space == "const" {
				continue
			}
			q = "generic" + m[2]
		}
		if m[1] == "st" {
			c.Stores[q]++
			continue
		}
		c.Loads[q]++
		if barrier && !strings.HasPrefix(q, "cg") {
			c.LateLoads++
		}
	}
	flush()
	return out
}

// censusRules are the memory-access rules every golden kernel must keep,
// with the goldens exempt from each and why. The census comes from PTX, so
// it sees what the compiler did with the emitted code: a load it sank past
// a barrier, a vector store it split.
var censusRules = []struct {
	name   string
	check  func(ptxCensus) bool
	exempt map[string]string // golden: reason
}{
	{
		// A load issued after a barrier waits a full memory latency the
		// barrier could have hidden: issue every load before the first
		// reduction.
		name:  "no ld.global after the first bar.sync",
		check: func(c ptxCensus) bool { return c.LateLoads == 0 },
		exempt: map[string]string{
			"moe8_align.cu":        "the scatter reads offsets the block computes",
			"moe8_attn_sliding.cu": "the main loop reads the page table a step ahead of its copies; the last block reads the gate after the merge",
			"moe8_expert_sum.cu":   "routes are read after the block sorts them",
		},
	},
	{
		// A thread holding adjacent 2-byte values stores them with one
		// wide store; one 16-bit store is a thread's whole share (an FP8
		// pair, a skinny GEMM's output), two or more are a split.
		name: "at most one 16-bit global store",
		check: func(c ptxCensus) bool {
			n := 0
			for q, k := range c.Stores {
				if strings.HasSuffix(q, "16") && !strings.Contains(q, ".v") {
					n += k
				}
			}
			return n <= 1
		},
		exempt: map[string]string{},
	},
}

func censusPath(golden string) string {
	return strings.TrimSuffix(golden, ".cu") + ".census.json"
}

// TestGoldenCensus checks censusRules against each golden's committed
// census, and that the census was taken from the golden as it is now. With
// -update-census it recompiles each golden to PTX and rewrites the census:
// $CUDACOMPILECHECK_NVCC names nvcc (default nvcc), and
// $CUDACOMPILECHECK_EXEC, if set, is a command prefix that runs it
// elsewhere, such as in a VM: the source goes on standard input and the
// PTX comes back on standard output.
func TestGoldenCensus(t *testing.T) {
	files, err := filepath.Glob(filepath.Join("testdata", "golden", "*.cu"))
	if err != nil {
		t.Fatal(err)
	}
	if len(files) == 0 {
		t.Fatal("no goldens")
	}
	for _, f := range files {
		t.Run(filepath.Base(f), func(t *testing.T) {
			src, err := os.ReadFile(f)
			if err != nil {
				t.Fatal(err)
			}
			sum := sha256.Sum256(src)
			if *updateCensus {
				writeCensus(t, f, hex.EncodeToString(sum[:]))
			}
			data, err := os.ReadFile(censusPath(f))
			if err != nil {
				t.Fatalf("%v (run go test -run GoldenCensus -update-census)", err)
			}
			var gc goldenCensus
			if err := json.Unmarshal(data, &gc); err != nil {
				t.Fatal(err)
			}
			if gc.Source != hex.EncodeToString(sum[:]) {
				t.Fatalf("census is stale (run go test -run GoldenCensus -update-census)")
			}
			if len(gc.Kernels) == 0 {
				t.Fatal("census has no kernels")
			}
			for _, r := range censusRules {
				if why, ok := r.exempt[filepath.Base(f)]; ok {
					t.Logf("%s: exempt: %s", r.name, why)
					continue
				}
				for _, k := range sortedKeys(gc.Kernels) {
					if c := gc.Kernels[k]; !r.check(c) {
						t.Errorf("%s: %s: loads %v, stores %v, %d late loads", k, r.name, c.Loads, c.Stores, c.LateLoads)
					}
				}
			}
		})
	}
}

func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// writeCensus compiles golden (with its launchers' addresses taken, so
// templated kernels are instantiated) to PTX and writes its census.
func writeCensus(t *testing.T, golden, sum string) {
	nvcc := cmp.Or(os.Getenv("CUDACOMPILECHECK_NVCC"), "nvcc")
	src, err := os.ReadFile(golden)
	if err != nil {
		t.Fatal(err)
	}
	var wrap strings.Builder
	wrap.Write(src)
	for i, m := range regexp.MustCompile(`(?m)^inline void (\w+)`).FindAllStringSubmatch(string(src), -1) {
		fmt.Fprintf(&wrap, "void* eg_keep_%d = (void*)&%s;\n", i, m[1])
	}
	// nvcc reads no standard input: stage the source in a temporary
	// directory on the side that runs it.
	script := `d=$(mktemp -d) && cat > "$d/k.cu" && "$0" -std=c++20 -O3 -arch=sm_90a -ptx -diag-suppress 177 -o "$d/k.ptx" "$d/k.cu" && cat "$d/k.ptx"; rc=$?; rm -rf "$d"; exit $rc`
	args := append(strings.Fields(os.Getenv("CUDACOMPILECHECK_EXEC")), "sh", "-c", script, nvcc)
	cmd := exec.Command(args[0], args[1:]...)
	cmd.Stdin = strings.NewReader(wrap.String())
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	ptx, err := cmd.Output()
	if err != nil {
		t.Fatalf("nvcc: %v\n%s", err, stderr.Bytes())
	}
	data, err := json.MarshalIndent(goldenCensus{Source: sum, Kernels: censusPTX(ptx)}, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(censusPath(golden), append(data, '\n'), 0o644); err != nil {
		t.Fatal(err)
	}
}
