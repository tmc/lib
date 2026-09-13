package cudatool

import (
	"bufio"
	"bytes"
	"fmt"
	"regexp"
	"sort"
	"strings"
)

// Arity says whether an option takes an argument. It is a property of the
// option as the tool spells it, not of the Go type a caller holds: passing an
// argument to a Bare option is an error the tool reports rather than ignores,
// and compute-sanitizer in particular rejects the whole command line.
type Arity int

const (
	// Bare options stand alone, as in --kill.
	Bare Arity = iota
	// Value options are followed by an argument, as in --tool memcheck.
	Value
)

func (a Arity) String() string {
	if a == Value {
		return "value"
	}
	return "bare"
}

// Option is one command-line option a tool accepts.
type Option struct {
	// Name is the long form, including the leading dashes.
	Name string
	// Short is the alternate spelling the help output pairs with Name, empty
	// when the tool documents none.
	Short string
	// Arity says whether an argument follows.
	Arity Arity
	// Default is the argument used when the option is absent, as documented.
	// Empty means the help output names none.
	Default string
	// Values are the arguments the help output enumerates, in the order
	// given. Nil means the help output constrains the argument no further,
	// which is not a promise that any value is accepted.
	Values []string
}

// Options is a tool's option set, sorted by name.
type Options []Option

// Find returns the option with the given long name.
func (o Options) Find(name string) (Option, bool) {
	i := sort.Search(len(o), func(i int) bool { return o[i].Name >= name })
	if i < len(o) && o[i].Name == name {
		return o[i], true
	}
	return Option{}, false
}

// Names returns the long names, sorted.
func (o Options) Names() []string {
	names := make([]string, len(o))
	for i, opt := range o {
		names[i] = opt.Name
	}
	return names
}

// Text returns the canonical one-line-per-option form, sorted by name:
//
//	--binary value short=-b values=SM100,SM100a
//	--kill bare
//	--tool value default=memcheck values=memcheck,racecheck
//
// It is what this package stores and compares, rather than the help output it
// came from: it holds the facts a command line depends on, it is a fifth the
// size, and a diff of two versions' text is the flag difference itself.
func (o Options) Text() string {
	var b strings.Builder
	for _, opt := range o {
		fmt.Fprintf(&b, "%s %s", opt.Name, opt.Arity)
		if opt.Short != "" {
			fmt.Fprintf(&b, " short=%s", opt.Short)
		}
		if opt.Default != "" {
			fmt.Fprintf(&b, " default=%s", opt.Default)
		}
		if len(opt.Values) > 0 {
			fmt.Fprintf(&b, " values=%s", strings.Join(opt.Values, ","))
		}
		b.WriteByte('\n')
	}
	return b.String()
}

// ParseText reads back what [Options.Text] wrote. Blank lines and lines
// beginning with # are ignored.
func ParseText(text string) (Options, error) {
	var opts Options
	for n, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 || !strings.HasPrefix(fields[0], "-") {
			return nil, fmt.Errorf("line %d: %q is not an option line", n+1, line)
		}
		opt := Option{Name: fields[0]}
		switch fields[1] {
		case "bare":
			opt.Arity = Bare
		case "value":
			opt.Arity = Value
		default:
			return nil, fmt.Errorf("line %d: arity %q is neither bare nor value", n+1, fields[1])
		}
		for _, f := range fields[2:] {
			key, val, ok := strings.Cut(f, "=")
			if !ok {
				return nil, fmt.Errorf("line %d: %q is not key=value", n+1, f)
			}
			switch key {
			case "short":
				opt.Short = val
			case "default":
				opt.Default = val
			case "values":
				opt.Values = strings.Split(val, ",")
			default:
				return nil, fmt.Errorf("line %d: unknown key %q", n+1, key)
			}
		}
		opts = append(opts, opt)
	}
	sort.Slice(opts, func(i, j int) bool { return opts[i].Name < opts[j].Name })
	return opts, nil
}

// ParseHelp extracts the option set from a tool's --help output. The CUDA
// binary utilities and compute-sanitizer print different grammars, so the
// grammar is detected rather than configured.
func ParseHelp(out []byte) (Options, error) {
	var opts Options
	if bytes.Contains(out, []byte("Compute Sanitizer")) {
		opts = parseProgramOptions(out)
	} else {
		opts = parseToolOptions(out)
	}
	if len(opts) == 0 {
		return nil, fmt.Errorf("cudatool: no options found in %d bytes of help output", len(out))
	}
	sort.Slice(opts, func(i, j int) bool { return opts[i].Name < opts[j].Name })
	return opts, nil
}

// optionHeading matches a line that introduces an option in the CUDA binary
// utilities' help output: the long name at column zero, then an optional
// argument spec, then a column holding the alternate spelling. The spec and the
// alternate spelling are separated in splitHeading rather than here, because
// both may contain single spaces and only their column separates them.
var optionHeading = regexp.MustCompile(`^--[a-zA-Z0-9_.-]+( |$)`)

// aliasColumn matches the start of the alternate-spelling column: two or more
// spaces followed by a parenthesized short form or a second long form.
var aliasColumn = regexp.MustCompile(` {2,}(\(-[^)]*\)|--[a-zA-Z0-9_.-]+)`)

// enumeratedArg matches an argument spec that lists its own values, as in
// {c|c++|cu} or [default,size,speed].
var enumeratedArg = regexp.MustCompile(`^[{\[](.*)[}\]]$`)

// splitHeading divides an option heading into the long name, the argument
// spec, and the alternate spelling. A heading with no alternate-spelling column
// yields an empty alias.
func splitHeading(line string) (name, arg, alias string) {
	head := line
	if loc := aliasColumn.FindStringIndex(line); loc != nil {
		head = line[:loc[0]]
		alias = strings.TrimSpace(line[loc[0]:])
		alias = strings.TrimSuffix(strings.TrimPrefix(alias, "("), ")")
		if i := strings.IndexAny(alias, " \t)"); i >= 0 {
			alias = alias[:i] // the column itself may wrap
		}
	}
	name, arg, _ = strings.Cut(strings.TrimSpace(head), " ")
	return name, strings.TrimSpace(arg), alias
}

// toolDefaultLine matches the documented default inside a description.
var toolDefaultLine = regexp.MustCompile(`^Default value: *(.*?)\.?$`)

// parseToolOptions reads the grammar nvcc, ptxas, nvlink, nvdisasm, cuobjdump
// and nvprune print: an option heading at column zero followed by an indented
// description that may name a default and enumerate allowed values.
func parseToolOptions(out []byte) Options {
	var opts Options
	var cur *Option
	collecting := false // inside an "Allowed values" run
	s := bufio.NewScanner(bytes.NewReader(out))
	s.Buffer(make([]byte, 0, 64<<10), 1<<20)
	for s.Scan() {
		line := s.Text()
		if optionHeading.MatchString(line) {
			name, arg, alias := splitHeading(line)
			opt := Option{Name: name, Short: alias}
			if arg != "" {
				opt.Arity = Value
				if m := enumeratedArg.FindStringSubmatch(arg); m != nil {
					opt.Values = splitEnumerated(m[1])
				}
			}
			if opt.Short == opt.Name {
				opt.Short = "" // the column repeats the name when there is no short form
			}
			opts = append(opts, opt)
			cur, collecting = &opts[len(opts)-1], false
			continue
		}
		if cur == nil {
			continue
		}
		body := strings.TrimSpace(line)
		switch {
		case collecting:
			cur.Values = append(cur.Values, quotedValues(body)...)
			collecting = !strings.HasSuffix(body, ".")
		case strings.HasPrefix(body, "Allowed values for this option:"):
			_, rest, _ := strings.Cut(body, ":")
			// The heading's argument spec may already have enumerated the
			// values; this list is the fuller one, so it replaces them.
			cur.Values = quotedValues(rest)
			cur.Arity = Value
			collecting = !strings.HasSuffix(body, ".")
		default:
			if m := toolDefaultLine.FindStringSubmatch(body); m != nil && cur.Default == "" {
				cur.Default = strings.Trim(m[1], "'")
				cur.Arity = Value
			}
		}
	}
	return opts
}

// splitEnumerated returns the values inside a {a|b} or [a,b] argument spec.
func splitEnumerated(s string) []string {
	sep := ","
	if strings.Contains(s, "|") {
		sep = "|"
	}
	var vals []string
	for _, f := range strings.Split(s, sep) {
		if f = strings.TrimSpace(f); f != "" {
			vals = append(vals, f)
		}
	}
	return vals
}

// quotedValues returns the single-quoted values in a fragment of an allowed
// values list.
func quotedValues(s string) []string {
	var vals []string
	for _, f := range strings.Split(s, ",") {
		if f = strings.Trim(strings.TrimSpace(f), "'.`"); f != "" {
			vals = append(vals, f)
		}
	}
	return vals
}

// programOptionLine matches compute-sanitizer's Boost program_options help:
// an indented option, optionally paired with a short form, optionally taking
// "arg" with a parenthesized default.
var programOptionLine = regexp.MustCompile(`^  (?:(-[a-zA-Z0-9]) \[ (--[a-zA-Z0-9_.-]+) \]|(--[a-zA-Z0-9_.-]+))( +arg)?(?: +\(=([^)]*)\))? +(\S.*)?$`)

// programValueLine matches an enumerated argument value in a continuation
// line, as in "memcheck  : Memory access checking".
var programValueLine = regexp.MustCompile(`^([a-zA-Z0-9_.-]+) +: `)

// parseProgramOptions reads compute-sanitizer's help output.
func parseProgramOptions(out []byte) Options {
	var opts Options
	var cur *Option
	s := bufio.NewScanner(bytes.NewReader(out))
	s.Buffer(make([]byte, 0, 64<<10), 1<<20)
	for s.Scan() {
		line := s.Text()
		if m := programOptionLine.FindStringSubmatch(line); m != nil {
			name, short := m[3], ""
			if name == "" {
				name, short = m[2], m[1]
			}
			arity := Bare
			if strings.TrimSpace(m[4]) != "" {
				arity = Value
			}
			opts = append(opts, Option{Name: name, Short: short, Arity: arity, Default: m[5]})
			cur = &opts[len(opts)-1]
			continue
		}
		if cur == nil {
			continue
		}
		if m := programValueLine.FindStringSubmatch(strings.TrimSpace(line)); m != nil {
			cur.Values = append(cur.Values, m[1])
		}
	}
	return opts
}
