package ptxasinfo

import (
	"bufio"
	"fmt"
	"io"
	"regexp"
	"strconv"
	"strings"
)

// A Function is one function's resources for one target architecture.
type Function struct {
	Name  string // mangled name
	Arch  string // target, such as "sm_90a"; empty if ptxas never named it
	Entry bool   // a kernel (entry function) rather than a device function

	StackFrame  int // bytes per thread
	SpillStores int // bytes
	SpillLoads  int // bytes

	// The fields below come from the "Used" line, which ptxas prints for
	// entry functions only.
	Registers       int         // registers per thread
	Barriers        int         // named barriers
	CumulativeStack int         // bytes, including callees
	SharedMem       int         // static shared memory, bytes per block
	LocalMem        int         // bytes; older toolkits only
	ConstMem        map[int]int // constant bank -> bytes
	Textures        int
	Samplers        int
	Surfaces        int

	// MaxRegisters is the register limit ptxas reported overriding with
	// -maxrregcount, or zero.
	MaxRegisters int
}

// Spills reports whether the function spills registers to local memory.
func (f *Function) Spills() bool { return f.SpillStores > 0 || f.SpillLoads > 0 }

// A Target holds the module-level resources for one architecture.
type Target struct {
	Arch      string      // empty for a module with no Compiling line, which ptxas never names
	GlobalMem int         // bytes of __device__ data
	ConstMem  map[int]int // constant bank -> bytes
	Functions []*Function // in report order
}

// A Report is everything Parse found.
type Report struct {
	Targets  []*Target
	Warnings []string // "ptxas warning" lines, prefix removed
	Errors   []string // "ptxas error" and "ptxas fatal" lines, prefix removed
	Notes    []string // other "ptxas info" lines, prefix removed
}

// Functions returns every function in report order.
func (r *Report) Functions() []*Function {
	var fns []*Function
	for _, t := range r.Targets {
		fns = append(fns, t.Functions...)
	}
	return fns
}

// Lookup returns the function with the given mangled name and architecture.
func (r *Report) Lookup(name, arch string) *Function {
	for _, t := range r.Targets {
		if t.Arch != arch {
			continue
		}
		for _, f := range t.Functions {
			if f.Name == name {
				return f
			}
		}
	}
	return nil
}

var (
	prefixRE    = regexp.MustCompile(`^ptxas\s+(info|warning|error|fatal)\s*:\s*(.*)$`)
	compilingRE = regexp.MustCompile(`^Compiling (entry )?function '([^']+)' for '([^']+)'$`)
	propsRE     = regexp.MustCompile(`^Function properties for (\S+)$`)
	frameRE     = regexp.MustCompile(`^(\d{1,18}) bytes stack frame, (\d{1,18}) bytes spill stores, (\d{1,18}) bytes spill loads$`)
	overrideRE  = regexp.MustCompile(`^Overriding maximum register limit \d{1,18} for '([^']+)' with\s+(\d{1,18}) of maxrregcount option$`)
	countRE     = regexp.MustCompile(`^(?:[Uu]sed )?(\d{1,18}) (registers|barriers|textures|samplers|surfaces)$`)
	bytesRE     = regexp.MustCompile(`^(\d{1,18}) bytes (smem|lmem|gmem|cumulative stack size|cmem\[(\d{1,18})\])$`)
)

// Parse reads ptxas -v output, possibly mixed with other compiler output.
// It returns an error for a malformed or unknown resource field in a line
// it recognizes.
func Parse(r io.Reader) (*Report, error) {
	rep := &Report{}
	var (
		target    *Target
		current   *Function // function whose properties or Used line come next
		wantFrame bool
		maxRegs   = map[string]int{}
		lineNum   int
	)
	targetFor := func(arch string) *Target {
		if target != nil && target.Arch == arch {
			return target
		}
		for _, t := range rep.Targets {
			if t.Arch == arch {
				target = t
				return t
			}
		}
		target = &Target{Arch: arch}
		rep.Targets = append(rep.Targets, target)
		return target
	}
	// A module-level line (gmem) precedes the first Compiling line of its
	// architecture, so it is held until the architecture is known. It also
	// starts a new module: device functions reported before that module's
	// first Compiling line (as with nvcc -G) are held too, rather than
	// credited to the previous architecture.
	var (
		pendingModule []string
		pendingFns    []*Function
	)
	errorf := func(format string, args ...any) error {
		return fmt.Errorf("ptxasinfo: line %d: %s", lineNum, fmt.Sprintf(format, args...))
	}

	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 64*1024), 1<<20)
	for sc.Scan() {
		lineNum++
		line := strings.TrimSpace(sc.Text())
		if wantFrame {
			wantFrame = false
			m := frameRE.FindStringSubmatch(line)
			if m == nil {
				return nil, errorf("want stack frame line after function properties, got %q", line)
			}
			current.StackFrame, current.SpillStores, current.SpillLoads = atoi(m[1]), atoi(m[2]), atoi(m[3])
			continue
		}
		m := prefixRE.FindStringSubmatch(line)
		if m == nil {
			continue
		}
		kind, msg := m[1], strings.TrimSpace(m[2])
		switch kind {
		case "warning":
			rep.Warnings = append(rep.Warnings, msg)
			continue
		case "error", "fatal":
			rep.Errors = append(rep.Errors, msg)
			continue
		}
		switch {
		case compilingRE.MatchString(msg):
			c := compilingRE.FindStringSubmatch(msg)
			t := targetFor(c[3])
			for _, p := range pendingModule {
				if err := parseModule(t, p); err != nil {
					return nil, errorf("%v", err)
				}
			}
			pendingModule = nil
			for _, f := range pendingFns {
				f.Arch = t.Arch
			}
			t.Functions = append(t.Functions, pendingFns...)
			pendingFns = nil
			current = &Function{Name: c[2], Arch: c[3], Entry: c[1] != "", MaxRegisters: maxRegs[c[2]]}
			delete(maxRegs, c[2])
			t.Functions = append(t.Functions, current)
		case propsRE.MatchString(msg):
			name := propsRE.FindStringSubmatch(msg)[1]
			if current == nil || current.Name != name {
				// A device function reported without its own Compiling
				// line belongs to the architecture being compiled.
				current = &Function{Name: name}
				if target == nil {
					pendingFns = append(pendingFns, current)
				} else {
					current.Arch = target.Arch
					target.Functions = append(target.Functions, current)
				}
			}
			wantFrame = true
		case strings.HasPrefix(msg, "Used "):
			if current == nil {
				return nil, errorf("resource usage before any function")
			}
			if err := parseUsed(current, msg); err != nil {
				return nil, errorf("%v", err)
			}
		case overrideRE.MatchString(msg):
			o := overrideRE.FindStringSubmatch(msg)
			maxRegs[o[1]] = atoi(o[2])
		case strings.HasSuffix(msg, "bytes gmem") || strings.Contains(msg, "bytes gmem,"):
			pendingModule = append(pendingModule, msg)
			target, current = nil, nil
		case strings.HasPrefix(msg, "Compile time"):
		default:
			rep.Notes = append(rep.Notes, msg)
		}
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("ptxasinfo: %w", err)
	}
	if wantFrame {
		return nil, fmt.Errorf("ptxasinfo: input ends before stack frame line of %s", current.Name)
	}
	if len(pendingFns) > 0 || len(pendingModule) > 0 {
		// ptxas never named this module's architecture: it had no
		// Compiling line, which happens when it has no entry function.
		t := &Target{Functions: pendingFns}
		for _, p := range pendingModule {
			if err := parseModule(t, p); err != nil {
				return nil, fmt.Errorf("ptxasinfo: %v", err)
			}
		}
		rep.Targets = append(rep.Targets, t)
	}
	return rep, nil
}

func parseUsed(f *Function, msg string) error {
	for _, field := range strings.Split(msg, ",") {
		field = strings.TrimSpace(field)
		if m := countRE.FindStringSubmatch(field); m != nil {
			n := atoi(m[1])
			switch m[2] {
			case "registers":
				f.Registers = n
			case "barriers":
				f.Barriers = n
			case "textures":
				f.Textures = n
			case "samplers":
				f.Samplers = n
			case "surfaces":
				f.Surfaces = n
			}
			continue
		}
		m := bytesRE.FindStringSubmatch(field)
		if m == nil {
			return fmt.Errorf("unknown resource %q in %q", field, msg)
		}
		n := atoi(m[1])
		switch {
		case m[2] == "smem":
			f.SharedMem = n
		case m[2] == "lmem":
			f.LocalMem = n
		case m[2] == "cumulative stack size":
			f.CumulativeStack = n
		case m[3] != "":
			if f.ConstMem == nil {
				f.ConstMem = map[int]int{}
			}
			f.ConstMem[atoi(m[3])] = n
		default:
			return fmt.Errorf("unexpected resource %q in function usage", field)
		}
	}
	return nil
}

func parseModule(t *Target, msg string) error {
	for _, field := range strings.Split(msg, ",") {
		field = strings.TrimSpace(field)
		m := bytesRE.FindStringSubmatch(field)
		switch {
		case m == nil:
			return fmt.Errorf("unknown module resource %q in %q", field, msg)
		case m[2] == "gmem":
			t.GlobalMem = atoi(m[1])
		case m[3] != "":
			if t.ConstMem == nil {
				t.ConstMem = map[int]int{}
			}
			t.ConstMem[atoi(m[3])] = atoi(m[1])
		default:
			return fmt.Errorf("unexpected module resource %q", field)
		}
	}
	return nil
}

// atoi converts a string the regexps have already restricted to digits.
func atoi(s string) int {
	n, err := strconv.Atoi(s)
	if err != nil {
		panic("ptxasinfo: digits did not parse: " + s)
	}
	return n
}
