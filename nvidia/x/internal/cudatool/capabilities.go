package cudatool

import (
	"context"
	"embed"
	"fmt"
	"path/filepath"
	"sort"
	"strings"
	"sync"
)

// optiondata holds one distilled option set per tool per toolkit version,
// measured from that toolkit's --help output. It is the canonical answer to
// what a given toolkit's tool accepts: [Capabilities] consults it first and
// falls back to asking the tool only for a version it does not cover.
//
// Refresh it with the steps in doc.go; the measurement is a diff of these files.
//
//go:embed optiondata
var optiondata embed.FS

// corpus is the parsed optiondata, keyed by tool name.
var corpus = sync.OnceValues(loadCorpus)

// toolCorpus holds the option sets for one tool, sorted by version.
type toolCorpus struct {
	versions []Version
	options  map[Version]Options
}

func loadCorpus() (map[string]*toolCorpus, error) {
	entries, err := optiondata.ReadDir("optiondata")
	if err != nil {
		return nil, err
	}
	byTool := make(map[string]*toolCorpus)
	for _, e := range entries {
		name := strings.TrimSuffix(e.Name(), ".txt")
		i := strings.LastIndex(name, "-")
		if i < 0 {
			return nil, fmt.Errorf("cudatool: %s is not named <tool>-<version>.txt", e.Name())
		}
		tool, verText := name[:i], name[i+1:]
		major, minor, ok := strings.Cut(verText, ".")
		if !ok {
			return nil, fmt.Errorf("cudatool: %s does not name a major.minor version", e.Name())
		}
		var v Version
		if _, err := fmt.Sscanf(major+" "+minor, "%d %d", &v.Major, &v.Minor); err != nil {
			return nil, fmt.Errorf("cudatool: %s: %v", e.Name(), err)
		}
		b, err := optiondata.ReadFile("optiondata/" + e.Name())
		if err != nil {
			return nil, err
		}
		opts, err := ParseText(string(b))
		if err != nil {
			return nil, fmt.Errorf("cudatool: %s: %w", e.Name(), err)
		}
		tc := byTool[tool]
		if tc == nil {
			tc = &toolCorpus{options: make(map[Version]Options)}
			byTool[tool] = tc
		}
		tc.versions = append(tc.versions, v)
		tc.options[v] = opts
	}
	for _, tc := range byTool {
		sort.Slice(tc.versions, func(i, j int) bool { return tc.versions[i].Compare(tc.versions[j]) < 0 })
	}
	return byTool, nil
}

// Tools returns the tool names the corpus covers, sorted.
func Tools() []string {
	c, err := corpus()
	if err != nil {
		return nil
	}
	names := make([]string, 0, len(c))
	for name := range c {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// KnownVersions returns the toolkit versions the corpus covers for a tool, in
// increasing order.
func KnownVersions(tool string) []Version {
	c, err := corpus()
	if err != nil || c[tool] == nil {
		return nil
	}
	return append([]Version(nil), c[tool].versions...)
}

// KnownOptions returns the option set the corpus records for a tool at a
// toolkit version. An exact match is used when present; otherwise the highest
// recorded version below v answers, since a version we did not measure is
// described by the last one we did. A version above everything recorded has no
// answer here and reports false, which is what sends [Capabilities] to the tool
// itself.
func KnownOptions(tool string, v Version) (Options, Version, bool) {
	c, err := corpus()
	if err != nil || c[tool] == nil || len(c[tool].versions) == 0 {
		return nil, Version{}, false
	}
	tc := c[tool]
	if opts, ok := tc.options[v]; ok {
		return opts, v, true
	}
	best := Version{}
	for _, known := range tc.versions {
		if known.Compare(v) <= 0 {
			best = known
		}
	}
	if !best.Valid() {
		// v predates everything measured; the oldest record is the closest
		// description available, and saying so is better than silence.
		oldest := tc.versions[0]
		return tc.options[oldest], oldest, true
	}
	if tc.versions[len(tc.versions)-1].Compare(v) < 0 {
		return nil, Version{}, false // newer than anything measured
	}
	return tc.options[best], best, true
}

// Caps is what one resolved executable accepts.
type Caps struct {
	// Path is the executable the capabilities describe.
	Path string
	// Tool is its base name, as the corpus keys it.
	Tool string
	// Version is the toolkit version the tool reports, or the one its install
	// prefix names for a tool that reports none.
	Version Version
	// Source is the version whose recorded option set Options came from. It
	// differs from Version when the corpus covers no exact match.
	Source Version
	// Options is the option set.
	Options Options
	// Probed reports that Options came from running the tool's --help rather
	// than from the shipped corpus, which happens for a toolkit newer than
	// anything measured.
	Probed bool
}

// Accepts reports whether the tool accepts an option.
func (c Caps) Accepts(name string) bool {
	_, ok := c.Options.Find(name)
	return ok
}

// Arity returns how an option is spelled. It reports false for an option the
// tool does not accept, which a caller must not emit at all.
func (c Caps) Arity(name string) (Arity, bool) {
	opt, ok := c.Options.Find(name)
	return opt.Arity, ok
}

// capsCache holds one entry per resolved executable.
var capsCache sync.Map // string -> Caps

// Capabilities returns what the named tool accepts. An empty explicit path
// resolves the tool the way [LookTool] does.
//
// The shipped corpus answers whenever it covers the resolved toolkit version,
// so the usual call runs the tool once, for --version. A toolkit newer than
// anything measured is asked directly, with --help, and the answer records
// Probed so a caller can report that it is working beyond what was measured.
func Capabilities(ctx context.Context, tool, path string) (Caps, error) {
	found, err := LookTool(tool, path)
	if err != nil {
		return Caps{}, err
	}
	if c, ok := capsCache.Load(found); ok {
		return c.(Caps), nil
	}
	c := Caps{Path: found, Tool: filepath.Base(found)}
	out, verr := Run(ctx, found, "", nil, "--version")
	if verr == nil {
		c.Version, _ = ParseVersion(append(out.Stdout, out.Stderr...))
	}
	if !c.Version.Valid() {
		// compute-sanitizer reports a product version, not a toolkit one.
		c.Version, _ = VersionFromPath(found)
	}
	if opts, source, ok := KnownOptions(c.Tool, c.Version); ok && c.Version.Valid() {
		c.Options, c.Source = opts, source
		capsCache.Store(found, c)
		return c, nil
	}
	help, err := Run(ctx, found, "", nil, "--help")
	if err != nil && len(help.Stdout)+len(help.Stderr) == 0 {
		return Caps{}, fmt.Errorf("cudatool: %s --help: %w", found, err)
	}
	opts, err := ParseHelp(append(help.Stdout, help.Stderr...))
	if err != nil {
		return Caps{}, err
	}
	c.Options, c.Source, c.Probed = opts, c.Version, true
	capsCache.Store(found, c)
	return c, nil
}
