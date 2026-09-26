package main

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"path"
	"regexp"
	"sort"
	"strings"
	"text/tabwriter"
)

// opCombine is an ops-file combine line, "name = op[:glob,...] ...": a row
// that adds up op name and the kernels each term names, so work fused into
// one op on one side and spread over several on the other can be read as
// one line. A term without globs takes every kernel of its op; with globs,
// the kernels whose functor or match key a glob matches, ignoring case.
type opCombine struct {
	Line  int           `json:"line"`
	Name  string        `json:"name"`
	Terms []combineTerm `json:"terms"`
}

type combineTerm struct {
	Op    string   `json:"op"`
	Globs []string `json:"globs,omitempty"`
}

func (t combineTerm) String() string {
	if len(t.Globs) == 0 {
		return t.Op
	}
	return t.Op + ":" + strings.Join(t.Globs, ",")
}

// takes reports whether t takes a kernel of op whose functor and match key
// are functor and key.
func (t combineTerm) takes(op, functor, key string) bool {
	if t.Op != op {
		return false
	}
	if len(t.Globs) == 0 {
		return true
	}
	functor, key = strings.ToLower(functor), strings.ToLower(key)
	for _, g := range t.Globs {
		if ok, _ := path.Match(g, functor); ok && functor != "" {
			return true
		}
		if ok, _ := path.Match(g, key); ok {
			return true
		}
	}
	return false
}

// readOpsFile reads an ops file, or returns nil rules for an empty path.
func readOpsFile(name string) (opRules, []opCombine, error) {
	if name == "" {
		return nil, nil, nil
	}
	if name == defaultOps {
		return parseOpsFile(strings.NewReader(defaultOpsText))
	}
	f, err := os.Open(name)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	rules, combines, err := parseOpsFile(f)
	if err != nil {
		return nil, nil, fmt.Errorf("%s: %w", name, err)
	}
	if rules == nil {
		return nil, nil, fmt.Errorf("%s: no rules", name)
	}
	return rules, combines, nil
}

// parseOpsFile reads the rules and combine lines of an ops file. A combine
// line is one whose second field is "="; its terms must name ops the rules
// name.
func parseOpsFile(r io.Reader) (opRules, []opCombine, error) {
	var rules opRules
	var combines []opCombine
	sc := bufio.NewScanner(r)
	for n := 1; sc.Scan(); n++ {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if f := strings.Fields(line); len(f) >= 2 && f[1] == "=" && line[0] != '"' && line[0] != '`' {
			c, err := parseCombine(n, f[0], f[2:])
			if err != nil {
				return nil, nil, err
			}
			combines = append(combines, c)
			continue
		}
		pattern, op, err := splitOpRule(line)
		if err != nil {
			return nil, nil, fmt.Errorf("line %d: %w", n, err)
		}
		if op == unmatchedOp {
			return nil, nil, fmt.Errorf("line %d: op name %s is reserved", n, unmatchedOp)
		}
		re, err := regexp.Compile(pattern)
		if err != nil {
			return nil, nil, fmt.Errorf("line %d: %w", n, err)
		}
		rules = append(rules, opRule{Line: n, Pattern: pattern, Op: op, re: re})
	}
	if err := sc.Err(); err != nil {
		return nil, nil, err
	}
	ops := map[string]bool{unmatchedOp: true}
	for _, r := range rules {
		ops[r.Op] = true
	}
	names := make(map[string]int)
	for _, c := range combines {
		if prev, ok := names[c.Name]; ok {
			return nil, nil, fmt.Errorf("line %d: combine %s already declared on line %d", c.Line, c.Name, prev)
		}
		names[c.Name] = c.Line
		for _, t := range c.Terms {
			if !ops[t.Op] {
				return nil, nil, fmt.Errorf("line %d: no rule names op %s", c.Line, t.Op)
			}
		}
	}
	return rules, combines, nil
}

func parseCombine(line int, name string, fields []string) (opCombine, error) {
	c := opCombine{Line: line, Name: name}
	if len(fields) == 0 {
		return c, fmt.Errorf("line %d: combine %s has no terms", line, name)
	}
	for _, f := range fields {
		op, globs, _ := strings.Cut(f, ":")
		t := combineTerm{Op: op}
		if globs != "" {
			for _, g := range strings.Split(globs, ",") {
				g = strings.ToLower(g)
				if _, err := path.Match(g, ""); g == "" || err != nil {
					return c, fmt.Errorf("line %d: bad glob %q", line, g)
				}
				t.Globs = append(t.Globs, g)
			}
		}
		if op == name && len(t.Globs) == 0 {
			return c, fmt.Errorf("line %d: combine %s includes op %s already", line, name, name)
		}
		c.Terms = append(c.Terms, t)
	}
	return c, nil
}

// functorWrappers are the templates a PyTorch elementwise kernel wraps its
// functor in.
var functorWrappers = map[string]bool{
	"gpu_kernel_impl":        true,
	"gpu_kernel_impl_nocast": true,
	"gpu_index_kernel":       true,
	"BinaryFunctor":          true,
	"AUnaryFunctor":          true,
	"BUnaryFunctor":          true,
}

var (
	identRE   = regexp.MustCompile(`[A-Za-z_][A-Za-z0-9_]*`)
	functorRE = regexp.MustCompile(`(Functor|Functor(OnSelf)?_\w+|Impl|_cuda|_impl|_kernel|_scalar|_out|_transform)$`)
)

// functorOf returns the functor of a PyTorch elementwise kernel, given its
// name with template arguments, or "" for any other kernel: in
//
//	at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, ...>
//
// it is MulFunctor. It is the first identifier of the template arguments
// that ends like a functor or kernel implementation and is not a wrapper.
func functorOf(name string) string {
	i := strings.IndexByte(name, '<')
	if i < 0 || !strings.Contains(name[:i], "elementwise") {
		return ""
	}
	for _, id := range identRE.FindAllString(name[i:], -1) {
		if !functorWrappers[id] && functorRE.MatchString(id) {
			return id
		}
	}
	return ""
}

// combineSet assigns kernels to combine rows.
type combineSet struct {
	combines []opCombine
	functors map[string]string // full name to functor
}

// functor returns the functor of the kernel named name, as functorOf does.
func (cs *combineSet) functor(name string) string {
	f, ok := cs.functors[name]
	if !ok {
		f = functorOf(opKey(name, true))
		cs.functors[name] = f
	}
	return f
}

// claim returns the index of the combine that takes a kernel of op with the
// given functor and match key, or -1 when none does, and an error when two do.
func (cs *combineSet) claim(op, functor, key string) (int, error) {
	found := -1
	for i, c := range cs.combines {
		takes := op == c.Name
		for _, t := range c.Terms {
			takes = takes || t.takes(op, functor, key)
		}
		if !takes {
			continue
		}
		if found >= 0 {
			return -1, fmt.Errorf("ops lines %d and %d both combine kernel %s of op %s", cs.combines[found].Line, c.Line, key, op)
		}
		found = i
	}
	return found, nil
}

// termHits counts, for each term of c, the launches of kernels of op it
// takes.
func (c opCombine) termHits(hits []int, op, functor, key string) {
	for i, t := range c.Terms {
		if t.takes(op, functor, key) {
			hits[i]++
		}
	}
}

// combineTotals is one combine row on one side.
type combineTotals struct {
	opTotals
	members  map[combineMember]*opTotals
	termHits []int // launches each term took
}

type combineMember struct{ op, kernel string }

// CompareCombine is a combine line of the ops file in both captures. Its
// kernels may belong to several ops, and a kernel of an op may be in it while
// the op's other kernels are not, so its rows sit outside the identity.
type CompareCombine struct {
	Line                 int             `json:"line"`
	Name                 string          `json:"name"`
	Terms                []string        `json:"terms"`
	BaseLaunches         int             `json:"base_launches"`
	TargetLaunches       int             `json:"target_launches"`
	BaseExclusiveNanos   int64           `json:"base_exclusive_ns"`
	TargetExclusiveNanos int64           `json:"target_exclusive_ns"`
	DeltaExclusiveNanos  int64           `json:"delta_exclusive_ns"`
	BaseSummedNanos      int64           `json:"base_summed_ns"`
	TargetSummedNanos    int64           `json:"target_summed_ns"`
	Members              []CombineMember `json:"members"`
}

// CombineMember is the part of a combine row one kernel, or one functor of
// a PyTorch elementwise kernel, contributed.
type CombineMember struct {
	Op                   string `json:"op"`
	Kernel               string `json:"kernel"`
	BaseLaunches         int    `json:"base_launches"`
	TargetLaunches       int    `json:"target_launches"`
	BaseExclusiveNanos   int64  `json:"base_exclusive_ns"`
	TargetExclusiveNanos int64  `json:"target_exclusive_ns"`
	DeltaExclusiveNanos  int64  `json:"delta_exclusive_ns"`
}

// joinCombines joins the combine rows of two sides, in ops-file order.
func joinCombines(combines []opCombine, base, target []*combineTotals) []CompareCombine {
	var rows []CompareCombine
	for i, c := range combines {
		b, t := base[i], target[i]
		row := CompareCombine{
			Line:                 c.Line,
			Name:                 c.Name,
			BaseLaunches:         b.launches,
			TargetLaunches:       t.launches,
			BaseExclusiveNanos:   b.exclusive,
			TargetExclusiveNanos: t.exclusive,
			DeltaExclusiveNanos:  t.exclusive - b.exclusive,
			BaseSummedNanos:      b.summed,
			TargetSummedNanos:    t.summed,
			Members:              []CombineMember{},
		}
		for _, term := range c.Terms {
			row.Terms = append(row.Terms, term.String())
		}
		members := make(map[combineMember]bool)
		for m := range b.members {
			members[m] = true
		}
		for m := range t.members {
			members[m] = true
		}
		for m := range members {
			bm, tm := b.members[m], t.members[m]
			if bm == nil {
				bm = &opTotals{}
			}
			if tm == nil {
				tm = &opTotals{}
			}
			row.Members = append(row.Members, CombineMember{
				Op:                   m.op,
				Kernel:               m.kernel,
				BaseLaunches:         bm.launches,
				TargetLaunches:       tm.launches,
				BaseExclusiveNanos:   bm.exclusive,
				TargetExclusiveNanos: tm.exclusive,
				DeltaExclusiveNanos:  tm.exclusive - bm.exclusive,
			})
		}
		sort.Slice(row.Members, func(i, j int) bool {
			a, b := row.Members[i], row.Members[j]
			if x, y := max(a.BaseExclusiveNanos, a.TargetExclusiveNanos), max(b.BaseExclusiveNanos, b.TargetExclusiveNanos); x != y {
				return x > y
			}
			if a.Op != b.Op {
				return a.Op < b.Op
			}
			return a.Kernel < b.Kernel
		})
		rows = append(rows, row)
	}
	return rows
}

// compareCombineMembers is how many members of each combine row the text
// lists. JSON gives every member.
const compareCombineMembers = 8

func (c *Comparison) writeCombines(w io.Writer) error {
	if len(c.Combines) == 0 {
		return nil
	}
	fmt.Fprintln(w, "\ncombined, ns: the ops file's combine lines. exclusive as above; a row may take part of an op, so rows")
	fmt.Fprintln(w, "sit outside the identity. members: kernels, or elementwise functors, the most exclusive first.")
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', tabwriter.AlignRight)
	fmt.Fprintln(tw, "launches\t\texclusive\t\t\tsummed\t\t")
	fmt.Fprintln(tw, "base\ttarget\tbase\ttarget\tΔ\tbase\ttarget\t  combine")
	for _, r := range c.Combines {
		fmt.Fprintf(tw, "%d\t%d\t%d\t%d\t%+d\t%d\t%d\t  %s = %s\n",
			r.BaseLaunches, r.TargetLaunches,
			r.BaseExclusiveNanos, r.TargetExclusiveNanos, r.DeltaExclusiveNanos,
			r.BaseSummedNanos, r.TargetSummedNanos, r.Name, strings.Join(r.Terms, " "))
		n := min(len(r.Members), compareCombineMembers)
		for _, m := range r.Members[:n] {
			fmt.Fprintf(tw, "%d\t%d\t%d\t%d\t%+d\t\t\t    %s: %s\n",
				m.BaseLaunches, m.TargetLaunches,
				m.BaseExclusiveNanos, m.TargetExclusiveNanos, m.DeltaExclusiveNanos, m.Op, m.Kernel)
		}
		if n < len(r.Members) {
			fmt.Fprintf(tw, "\t\t\t\t\t\t\t    and %d more\n", len(r.Members)-n)
		}
	}
	return tw.Flush()
}
