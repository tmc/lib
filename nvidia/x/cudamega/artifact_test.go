package cudamega_test

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudamega"
	"github.com/tmc/lib/nvidia/x/cudataskgraph"
)

// The synchronization instructions this package's generator is allowed to
// emit, keyed by the domain they serve.
//
// This table is the test's own. It deliberately does not call
// cudamega.acquireFence or read the Go-side scope constant that requested the
// instruction: that constant is the thing under test, and a test that asked
// the generator what it emitted would agree with any answer.
var (
	releaseInstruction = map[cudataskgraph.Scope]string{
		cudataskgraph.ScopeCTA:    "mbarrier.arrive.release.cta",
		cudataskgraph.ScopeDevice: "fence.sc.gpu",
		cudataskgraph.ScopeSystem: "fence.sc.sys",
	}
	acquireInstruction = map[cudataskgraph.Scope]string{
		cudataskgraph.ScopeCTA:    "mbarrier.try_wait.parity.acquire.cta",
		cudataskgraph.ScopeDevice: "fence.sc.gpu",
		cudataskgraph.ScopeSystem: "fence.sc.sys",
	}
	// The acquiring load of a completion word, per domain. This replaced a
	// volatile load, and the replacement is a semantic change rather than a
	// rename: volatile forbids the compiler from caching a load and orders
	// nothing whatever against the producer's stores, so a reader that saw
	// the completion count could still read the data it announces from a
	// stale line. ld.acquire orders them, and it carries the scope it orders
	// at, which is why this table is keyed by domain at all.
	acquireLoad = map[cudataskgraph.Scope]string{
		cudataskgraph.ScopeCTA:    "cta",
		cudataskgraph.ScopeDevice: "gpu",
		cudataskgraph.ScopeSystem: "sys",
	}
)

// goldenPTX is the committed PTX for one fixture and target.
func goldenPTX(name string, target cudamega.Target) string {
	return filepath.Join("testdata", "golden", name+"."+string(target)+".ptx")
}

// syncKind classifies one PTX instruction as a synchronization instruction,
// or returns "" if it is not one. mbarrier.init is deliberately not a
// synchronization instruction: it publishes nothing and orders nothing, and
// counting it as an arrival is precisely the vacuous pass this test exists to
// prevent.
func syncKind(instruction string) string {
	for _, k := range []string{
		"mbarrier.arrive.release.cta",
		"mbarrier.try_wait.parity.acquire.cta",
		"fence.sc.gpu",
		"fence.sc.sys",
		"fence.sc.cta",
		"fence.acq_rel.gpu",
		"fence.acq_rel.sys",
		"membar.gl",
		"membar.sys",
		"membar.cta",
	} {
		if strings.Contains(instruction, k) {
			return k
		}
	}
	return ""
}

var (
	publishRE = regexp.MustCompile(`^atom\.global\.add\.u32\s+%\w+,\s*\[(%\w+)(?:\+(\d+))?\],\s*1;$`)
	// pollRE matches the acquiring load of a completion word. The scope
	// qualifier is captured rather than fixed, because it is the task's
	// synchronization domain and the test checks it against the domain the
	// task declares. Nothing else in the generated program emits ld.acquire,
	// so matching one identifies this primitive and no other -- which the
	// volatile form it replaced could not claim.
	pollRE = regexp.MustCompile(`^ld\.acquire\.(cta|gpu|sys)\.u32\s+%\w+,\s*\[(%\w+)(?:\+(\d+))?\];$`)
)

// entryBody returns the instruction lines of one PTX .entry, in order.
func entryBody(t *testing.T, ptx, entry string) []string {
	t.Helper()
	start := strings.Index(ptx, ".visible .entry "+entry+"(")
	if start < 0 {
		t.Fatalf("artifact has no .entry %s", entry)
	}
	rest := ptx[start+1:]
	if i := strings.Index(rest, ".visible .entry "); i >= 0 {
		rest = rest[:i]
	}
	var out []string
	for _, line := range strings.Split(rest, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "//") {
			continue
		}
		out = append(out, line)
	}
	return out
}

// memoryRef returns the base register and byte offset of the operand captured
// by the last two groups of m.
func memoryRef(m []string) (base string, off int) {
	base, text := m[len(m)-2], m[len(m)-1]
	if text != "" {
		off, _ = strconv.Atoi(text)
	}
	return base, off
}

// TestEmittedFenceAppearsInArtifact asserts on the committed PTX goldens that
// each task's release fence precedes its completion publication and each
// dependent task's acquire fence follows the poll of the dependency's
// completion word, with the instruction the task's synchronization domain
// declares.
//
// The anchor is position relative to the publishing store, not a substring
// search: a fence anywhere in the artifact would satisfy a search and satisfy
// nothing about ordering. The publications are identified structurally --
// every atom.global.add.u32 of 1 on one base register at ascending word
// offsets -- so the test does not have to be told which register holds the
// completion array.
func TestEmittedFenceAppearsInArtifact(t *testing.T) {
	for _, f := range goldenFixtures(t) {
		for _, target := range ptxTargets {
			t.Run(f.name+"/"+string(target), func(t *testing.T) {
				checkEmittedFences(t, f, target)
			})
		}
	}
}

// ptxPositional records, per target, whether the committed PTX still carries
// the generated control flow in text order.
//
// It does on sm_121a. It does not on sm_80 or sm_90 with CUDA 13.4: nvcc
// rotates the dependency poll out of line and lays its body down after the
// consumer's publication, so the entry's text order there is layout order and
// not execution order -- the same distinction [TestArtifactSurvivesToSASS]
// records for SASS.
//
// This is a property of the target's code generation, not of what this
// package emits. The sm_80 PTX built from the pre-S0.5 source, whose poll was
// a volatile load rather than an acquiring one, carries the same rotation:
// its mbarrier wait is at line 149 and the publication it must follow is at
// line 185. Asserting position on those targets would assert something about
// the listing rather than about the program.
//
// Everything else is asserted on every target: the publications are one array
// at ascending word offsets, the polls are one object at the offset of the
// dependency they name, each acquiring load carries the scope its task
// declares, and the synchronization census is exactly what the task list
// accounts for.
var ptxPositional = map[cudamega.Target]bool{
	cudamega.SM80:   false,
	cudamega.SM90:   false,
	cudamega.SM121a: true,
}

func checkEmittedFences(t *testing.T, f programFixture, target cudamega.Target) {
	t.Helper()
	path := goldenPTX(f.name, target)
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(raw), ".target "+string(target)) {
		t.Fatalf("%s: golden was not built for %s", path, target)
	}
	positional, ok := ptxPositional[target]
	if !ok {
		t.Fatalf("no record of whether %s PTX is in execution order", target)
	}
	body := entryBody(t, string(raw), "dag_worker")
	tasks := f.graph.Tasks()

	// One forward pass resolves the address arithmetic and records every
	// publication and every poll with the object and offset it names.
	addr := addresses{}
	pubAt := map[int]address{}
	pollAt := map[int]address{}
	pollScope := map[int]string{}
	for i, ins := range body {
		switch {
		case publishRE.MatchString(ins):
			m := publishRE.FindStringSubmatch(ins)
			b, off := memoryRef(m)
			pubAt[i] = addr.resolve(b, off)
		case pollRE.MatchString(ins):
			m := pollRE.FindStringSubmatch(ins)
			b, off := memoryRef(m)
			pollAt[i] = addr.resolve(b, off)
			pollScope[i] = m[1]
		default:
			addr.observe(ins)
		}
	}

	// The publications, checked to be one array indexed by task.
	var pubs []int
	var base string
	for i := range body {
		a, ok := pubAt[i]
		if !ok {
			continue
		}
		if base == "" {
			base = a.base
		}
		if a.base != base {
			t.Fatalf("publication %d stores to %s, not the completion array %s", len(pubs), a.base, base)
		}
		if want := 4 * len(pubs); a.off != want {
			t.Fatalf("publication %d is at byte offset %d, want %d", len(pubs), a.off, want)
		}
		pubs = append(pubs, i)
	}
	if len(pubs) != len(tasks) {
		t.Fatalf("artifact publishes %d completions for %d tasks", len(pubs), len(tasks))
	}

	// The polls go through one object too, so that a task polling some other
	// array is caught even where its offset happens to look right.
	//
	// That object is not the register the publications store through. nvcc
	// rematerializes the completion pointer for the inline asm operand and no
	// arithmetic in the entry connects the two, so the artifact cannot be made
	// to say they are the same parameter. What ties a poll to its dependency
	// is the byte offset, which must be the offset of that dependency's own
	// publication.
	pollBase := ""
	for i := range body {
		a, ok := pollAt[i]
		if !ok {
			continue
		}
		if pollBase == "" {
			pollBase = a.base
		}
		if a.base != pollBase {
			t.Fatalf("acquiring loads go through %s and %s; the completion array is one object", pollBase, a.base)
		}
	}

	index := map[string]int{}
	for i, task := range tasks {
		index[task.ID] = i
	}

	// Every dependency is polled, at its own offset, with an acquiring load
	// scoped to the domain the consumer declares. This is what the volatile
	// load could not carry: volatile has no scope and orders nothing.
	for _, task := range tasks {
		if len(task.Deps) == 0 {
			continue
		}
		dep := index[task.Deps[0]]
		found := false
		for i := range body {
			a, ok := pollAt[i]
			if !ok || a.off != 4*dep {
				continue
			}
			if got := pollScope[i]; got != acquireLoad[task.Scope] {
				t.Fatalf("task %q at %s scope: its acquiring load is scoped %q, want %q",
					task.ID, scopeLabel(task.Scope), got, acquireLoad[task.Scope])
			}
			found = true
		}
		if !found {
			t.Fatalf("task %q never polls the completion word of its dependency %q", task.ID, task.Deps[0])
		}
	}

	// Every synchronization instruction in the worker must be one the task
	// list accounts for. An extra one is as much a defect as a missing one:
	// it means the artifact does something the progress argument does not
	// describe.
	wantSyncs := 0
	for _, task := range tasks {
		wantSyncs++
		if len(task.Deps) > 0 {
			wantSyncs++
		}
	}
	var gotSyncs int
	for _, ins := range body {
		if syncKind(ins) != "" {
			gotSyncs++
		}
	}
	if gotSyncs != wantSyncs {
		t.Fatalf("worker has %d synchronization instructions, the %d tasks account for %d", gotSyncs, len(tasks), wantSyncs)
	}

	if !positional {
		return
	}
	for i, task := range tasks {
		lo := 0
		if i > 0 {
			lo = pubs[i-1] + 1
		}
		window := body[lo:pubs[i]]

		// The release: the last synchronization instruction before the
		// publication, and the one this scope declares.
		last, lastAt := "", -1
		for j, ins := range window {
			if k := syncKind(ins); k != "" {
				last, lastAt = k, j
			}
		}
		want := releaseInstruction[task.Scope]
		if want == "" {
			t.Fatalf("task %q: scope %d has no declared release instruction", task.ID, task.Scope)
		}
		if last == "" {
			t.Fatalf("task %q: nothing is released before its completion is published", task.ID)
		}
		if last != want {
			t.Fatalf("task %q at %s scope: the instruction before its publication is %q, want %q",
				task.ID, scopeLabel(task.Scope), last, want)
		}

		if len(task.Deps) == 0 {
			if n := countSyncs(window); n != 1 {
				t.Fatalf("task %q has no dependencies but %d synchronization instructions precede its publication", task.ID, n)
			}
			continue
		}

		// The acquire: after the poll of the dependency's completion word,
		// and before the release.
		dep := index[task.Deps[0]]
		poll := -1
		for j := range window[:lastAt] {
			if a, ok := pollAt[lo+j]; ok && a.off == 4*dep {
				poll = j
			}
		}
		if poll < 0 {
			t.Fatalf("task %q does not poll %q's completion word before publishing its own", task.ID, task.Deps[0])
		}
		acquire := -1
		for j := poll + 1; j < lastAt; j++ {
			if syncKind(window[j]) != "" {
				acquire = j
				break
			}
		}
		if acquire < 0 {
			t.Fatalf("task %q: no acquire between polling %q's completion and releasing its own", task.ID, task.Deps[0])
		}
		if got := syncKind(window[acquire]); got != acquireInstruction[task.Scope] {
			t.Fatalf("task %q at %s scope: acquires with %q, want %q",
				task.ID, scopeLabel(task.Scope), got, acquireInstruction[task.Scope])
		}
	}
}

// An address is a PTX memory operand reduced to the register holding the
// object and a byte offset into it.
type address struct {
	base string
	off  int
}

// addresses resolves the address arithmetic ptxas leaves in front of a memory
// operand.
//
// One emitted store reaches PTX in two shapes depending on the target:
// [%rd2+4] on sm_121a, and "add.s64 %rd38, %rd2, 4" followed by [%rd38] on
// sm_80 and sm_90. They are the same object at the same offset, and a test
// that recognized only the first would report that the sm_80 artifact
// publishes to some other array.
type addresses map[string]address

var (
	addRE = regexp.MustCompile(`^add\.s64\s+(%\w+),\s*(%\w+),\s*(\d+);$`)
	movRE = regexp.MustCompile(`^mov\.u64\s+(%\w+),\s*(%\w+);$`)
)

// observe folds one instruction into the map. A destination register is
// always reassigned, so a reused register cannot keep a stale meaning.
func (a addresses) observe(ins string) {
	if m := addRE.FindStringSubmatch(ins); m != nil {
		off, _ := strconv.Atoi(m[3])
		a[m[1]] = a.resolve(m[2], off)
		return
	}
	if m := movRE.FindStringSubmatch(ins); m != nil {
		a[m[1]] = a.resolve(m[2], 0)
		return
	}
	// Any other definition of a register makes what this map knows about it
	// wrong, so forget it rather than carry it forward.
	if i := strings.IndexByte(ins, ' '); i > 0 && strings.HasSuffix(ins[:i], "64") {
		rest := strings.TrimSpace(ins[i:])
		if j := strings.IndexAny(rest, ", "); j > 0 {
			delete(a, rest[:j])
		}
	}
}

// resolve reduces a register plus a literal offset to an address.
func (a addresses) resolve(reg string, off int) address {
	if r, ok := a[reg]; ok {
		return address{r.base, r.off + off}
	}
	return address{reg, off}
}

func countSyncs(lines []string) int {
	n := 0
	for _, l := range lines {
		if syncKind(l) != "" {
			n++
		}
	}
	return n
}

func scopeLabel(s cudataskgraph.Scope) string {
	switch s {
	case cudataskgraph.ScopeCTA:
		return "cta"
	case cudataskgraph.ScopeCluster:
		return "cluster"
	case cudataskgraph.ScopeDevice:
		return "device"
	case cudataskgraph.ScopeSystem:
		return "system"
	}
	return fmt.Sprintf("scope(%d)", s)
}

// TestArtifactDoesNotCarryAnotherDomainsInstruction is the other half of the
// positional test. Each golden must contain the instructions its own scopes
// declare and no instruction belonging to a domain it does not use, so that a
// generator which emitted a device fence for a CTA-scope task -- which a
// search for "is there a fence" would accept -- fails here.
func TestArtifactDoesNotCarryAnotherDomainsInstruction(t *testing.T) {
	for _, f := range goldenFixtures(t) {
		t.Run(f.name, func(t *testing.T) {
			raw, err := os.ReadFile(goldenPTX(f.name, goldenTarget))
			if err != nil {
				t.Fatal(err)
			}
			body := entryBody(t, string(raw), "dag_worker")
			present := map[string]int{}
			for _, ins := range body {
				if k := syncKind(ins); k != "" {
					present[k]++
				}
			}
			used := map[string]bool{}
			for _, task := range f.graph.Tasks() {
				used[releaseInstruction[task.Scope]] = true
				if len(task.Deps) > 0 {
					used[acquireInstruction[task.Scope]] = true
				}
			}
			for k, n := range present {
				if !used[k] {
					t.Fatalf("worker emits %s %d times, but no task in this graph declares that domain", k, n)
				}
			}
			for k := range used {
				if present[k] == 0 {
					t.Fatalf("a task declares %s and the artifact contains none", k)
				}
			}
		})
	}
}

// The sm_121a machine instructions the PTX synchronization forms lower to,
// observed in the committed SASS goldens. They are recorded per target, not
// generalized: nothing here is evidence about any other SM version.
var sassInstruction = map[string]string{
	"mbarrier.arrive.release.cta":          "SYNCS.ARRIVE.TRANS64",
	"mbarrier.try_wait.parity.acquire.cta": "SYNCS.PHASECHK.TRANS64.TRYWAIT",
	"fence.sc.gpu":                         "MEMBAR.SC.GPU",
	"fence.sc.sys":                         "MEMBAR.SC.SYS",
}

// sassFunction returns the disassembled body of one function from a
// cuobjdump -sass listing.
func sassFunction(t *testing.T, sass, name string) []string {
	t.Helper()
	start := strings.Index(sass, "Function : "+name)
	if start < 0 {
		t.Fatalf("SASS listing has no function %s", name)
	}
	rest := sass[start:]
	if i := strings.Index(rest[1:], "Function : "); i >= 0 {
		rest = rest[:i+1]
	}
	return strings.Split(rest, "\n")
}

// TestArtifactSurvivesToSASS checks the step the PTX golden cannot: that
// ptxas still emits each synchronization instruction for the authorized
// target, and emits none belonging to a domain the graph does not use.
//
// This is a survival and exclusivity check, not a positional one, and the
// distinction is load-bearing. SASS text order is layout order, not execution
// order: in cta-1cta.sass ptxas peeled the try_wait loop and placed the
// spinning copy after the function's fall-through path, at 0x1660, so the
// second SYNCS.PHASECHK appears textually after the last publication while
// executing before it. Asserting position here would assert something about
// the listing rather than about the program. Position is asserted on the PTX,
// where the control flow is still structured; see
// [TestEmittedFenceAppearsInArtifact].
//
// The SASS golden is also evidence about one of two copies. The CUDA 13.4
// cubin these fixtures compile to carries .text.dag_worker (0x2f00 bytes) and
// a second, compressed .nv.capmerc.text.dag_worker (0x107c bytes) with its
// own .nv.merc.rela.text.dag_worker relocations; cuobjdump -sass reads the
// former. Which copy the driver executes is not established here, so this
// test proves that ptxas emitted the instruction into .text and nothing
// about the mirror. Lane 24 owns that surface.
func TestArtifactSurvivesToSASS(t *testing.T) {
	for _, f := range goldenFixtures(t) {
		t.Run(f.name, func(t *testing.T) {
			raw, err := os.ReadFile(filepath.Join("testdata", "golden", f.name+"."+goldenArch+".sass"))
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(string(raw), "code for "+goldenArch) {
				t.Fatalf("SASS golden was not built for %s", goldenArch)
			}
			body := strings.Join(sassFunction(t, string(raw), "dag_worker"), "\n")

			want := map[string]int{}
			for _, task := range f.graph.Tasks() {
				want[sassInstruction[releaseInstruction[task.Scope]]]++
				if len(task.Deps) > 0 {
					want[sassInstruction[acquireInstruction[task.Scope]]]++
				}
			}
			for _, ins := range sassInstruction {
				got := strings.Count(body, ins)
				switch n := want[ins]; {
				case n == 0 && got != 0:
					t.Fatalf("worker contains %s %d times, but no task in this graph declares that domain", ins, got)
				case got < n:
					t.Fatalf("worker contains %s %d times, the tasks require at least %d", ins, got, n)
				}
			}
			// The publications survive too, as a global reduction or atomic
			// depending on whether ptxas kept the returned value.
			pubs := strings.Count(body, "REDG.E.ADD") + strings.Count(body, "ATOMG.E.ADD")
			if n := len(f.graph.Tasks()); pubs < n {
				t.Fatalf("worker publishes %d completions in SASS, want at least %d", pubs, n)
			}
		})
	}
}

// casRE matches a compare-and-swap in PTX, capturing the value it installs.
// The generated worker raises every abort code this way, so the set of
// captured values is the set of abort codes that survived to the artifact.
var casRE = regexp.MustCompile(`atom\.[\w.]*cas\.b32\s+%\w+,\s*\[%\w+\],\s*0,\s*(\d+);`)

// wantAbortCodes and wantSpinBound are this test's own copies of the two
// values the artifact is checked against, for the same reason the
// synchronization table above is: the Go constants are what asked for these,
// and a test that reads them cannot tell the difference between a bound that
// reached the hardware and a bound that was folded away on the way there.
// Changing DefaultSpinLimit or an abort code must fail this test until the
// goldens are regenerated -- that failure is the point.
var wantAbortCodes = []string{"2", "3", "4"} // bad launch, bad extent, spin exhausted

const wantSpinBound = 16777216 // DefaultSpinLimit as the fixtures were generated

// TestAbortCodesReachTheArtifact checks that every abort code the generated
// worker can raise is still in the compiled artifact, and that no code it
// cannot raise appears there.
//
// The poll bound is the reason this test exists. A bound the compiler folded
// away -- or one never emitted for one of the two poll loops -- leaves a grid
// that spins until the device is reset, and on a device that drives a display
// that costs the desktop. The Go constant is no evidence the hardware will
// ever write the code.
//
// Asserting the exact set, rather than the presence of 4 alone, is what makes
// the test non-vacuous in both directions: dropping the bad-extent guard
// fails it just as a dropped bound does.
func TestAbortCodesReachTheArtifact(t *testing.T) {
	for _, f := range []programFixture{ctaFixture(t), deviceFixture(t)} {
		t.Run(f.name, func(t *testing.T) {
			raw, err := os.ReadFile(goldenPTX(f.name, goldenTarget))
			if err != nil {
				t.Fatal(err)
			}
			ptx := string(raw)

			seen := map[string]bool{}
			for _, m := range casRE.FindAllStringSubmatch(ptx, -1) {
				seen[m[1]] = true
			}
			var got []string
			for c := range seen {
				got = append(got, c)
			}
			sort.Strings(got)
			if strings.Join(got, ",") != strings.Join(wantAbortCodes, ",") {
				t.Errorf("abort codes in %s.ptx = %v, want %v", f.name, got, wantAbortCodes)
			}

			// The bound itself. Without this, a CAS installing 4 could sit
			// on a path no counter ever reaches.
			if !strings.Contains(ptx, strconv.Itoa(wantSpinBound)) {
				t.Errorf("%s.ptx compares no counter against the poll bound %d", f.name, wantSpinBound)
			}
		})
	}
}

// TestSpinBoundSurvivesToSASS carries the bound one layer further down.
//
// PTX is not what runs. The bound is a loop-carried counter compared against
// a constant, which is exactly the shape a machine-code optimizer is entitled
// to rewrite, so surviving to PTX does not imply surviving to SASS. Recorded
// for sm_121a and CUDA 13.4 only: ptxas keeps the comparison as an immediate
// ISETP against the limit and the abort store as ATOMG.E.CAS.
func TestSpinBoundSurvivesToSASS(t *testing.T) {
	for _, f := range []programFixture{ctaFixture(t), deviceFixture(t)} {
		t.Run(f.name, func(t *testing.T) {
			raw, err := os.ReadFile(filepath.Join("testdata", "golden", f.name+"."+goldenArch+".sass"))
			if err != nil {
				t.Fatal(err)
			}
			sass := string(raw)
			hex := fmt.Sprintf("0x%x", wantSpinBound)
			if !strings.Contains(sass, hex) {
				t.Errorf("%s.sass has no immediate %s; the poll bound did not survive ptxas", f.name, hex)
			}
			if !strings.Contains(sass, "ATOMG.E.CAS") {
				t.Errorf("%s.sass has no ATOMG.E.CAS; no abort code can be installed", f.name)
			}
		})
	}
}
