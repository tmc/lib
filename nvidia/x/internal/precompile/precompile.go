// Package precompile builds device code for cudaprecompile and for
// cudaprecompiled's run-time fallback, so both compile a set of sources the
// same way.
package precompile

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io/fs"
	"path"
	"slices"
	"strings"
	"sync"

	"github.com/tmc/lib/nvidia/nvjitlink"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/nvfatbinx"
	"github.com/tmc/lib/nvidia/x/nvjitlinkx"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// Config describes CUDA C++ sources and how to compile them.
type Config struct {
	Files   fs.FS    // sources and headers
	Sources []string // translation units, as paths in Files
	Headers []string // headers, as paths in Files
	Include []string // include directories, as paths in Files
	Options []string // other NVRTC options, such as -DN=4 or -std=c++17
	Names   []string // name expressions; only with one source
	NoLTO   bool     // link several sources without link-time optimization
}

// Sum returns a digest of everything that determines the compiled code
// except the target architectures.
func (c *Config) Sum() (string, error) {
	h := sha256.New()
	field := func(s string) { fmt.Fprintf(h, "%d:%s", len(s), s) }
	list := func(name string, v []string) {
		field(name)
		field(fmt.Sprint(len(v)))
		for _, s := range v {
			field(s)
		}
	}
	list("options", c.Options)
	list("names", c.Names)
	list("include", c.Include)
	field(fmt.Sprint(c.NoLTO))
	for _, group := range [][]string{c.Sources, c.Headers} {
		field(fmt.Sprint(len(group)))
		for _, name := range group {
			data, err := fs.ReadFile(c.Files, name)
			if err != nil {
				return "", err
			}
			field(name)
			field(string(data))
		}
	}
	return "sha256:" + hex.EncodeToString(h.Sum(nil)), nil
}

func (c *Config) check() error {
	if len(c.Sources) == 0 {
		return fmt.Errorf("no sources")
	}
	if len(c.Names) > 0 && len(c.Sources) > 1 {
		return fmt.Errorf("name expressions need a single source")
	}
	return nil
}

// headerOptions registers every header under each name an #include could
// use for it: its path in Files, its path relative to each include
// directory, and its path relative to each source's directory.
func (c *Config) headerOptions() ([]nvrtcx.Option, error) {
	byName := make(map[string]string)
	var names []string
	add := func(name, content string) error {
		if old, ok := byName[name]; ok {
			if old != content {
				return fmt.Errorf("two headers named %q", name)
			}
			return nil
		}
		byName[name] = content
		names = append(names, name)
		return nil
	}
	dirs := slices.Clone(c.Include)
	for _, src := range c.Sources {
		dirs = append(dirs, path.Dir(src))
	}
	for _, h := range c.Headers {
		data, err := fs.ReadFile(c.Files, h)
		if err != nil {
			return nil, err
		}
		if err := add(h, string(data)); err != nil {
			return nil, err
		}
		for _, dir := range dirs {
			if dir == "." {
				continue
			}
			if rel, ok := strings.CutPrefix(h, dir+"/"); ok {
				if err := add(rel, string(data)); err != nil {
					return nil, err
				}
			}
		}
	}
	opts := make([]nvrtcx.Option, 0, len(names))
	for _, name := range names {
		opts = append(opts, nvrtcx.WithHeader(name, byName[name]))
	}
	return opts, nil
}

func (c *Config) compile(src string, opts ...nvrtcx.Option) (*nvrtcx.CompiledModule, error) {
	code, err := fs.ReadFile(c.Files, src)
	if err != nil {
		return nil, err
	}
	headers, err := c.headerOptions()
	if err != nil {
		return nil, err
	}
	all := append([]nvrtcx.Option{nvrtcx.WithName(src)}, headers...)
	for _, o := range c.Options {
		all = append(all, nvrtcx.WithFlag(o))
	}
	if len(c.Names) > 0 {
		all = append(all, nvrtcx.WithNameExpressions(c.Names...))
	}
	all = append(all, opts...)
	cm, err := nvrtcx.Compile(string(code), all...)
	if err != nil {
		return nil, fmt.Errorf("compile %s: %w", src, err)
	}
	return cm, nil
}

// lowered returns the lowered name of each name expression.
func (c *Config) lowered(cm *nvrtcx.CompiledModule) (map[string]string, error) {
	if len(c.Names) == 0 {
		return nil, nil
	}
	m := make(map[string]string, len(c.Names))
	for _, expr := range c.Names {
		name, ok := cm.LoweredName(expr)
		if !ok {
			return nil, fmt.Errorf("no lowered name for %q", expr)
		}
		m[expr] = name
	}
	return m, nil
}

// Native compiles the sources to machine code for target, such as sm_90,
// linking them when there are several. It also returns the lowered name of
// each name expression.
func (c *Config) Native(target string) ([]byte, map[string]string, error) {
	if err := c.check(); err != nil {
		return nil, nil, err
	}
	if !strings.HasPrefix(target, "sm_") {
		return nil, nil, fmt.Errorf("target %q is not sm_*", target)
	}
	if len(c.Sources) == 1 {
		cm, err := c.compile(c.Sources[0], nvrtcx.WithPhysicalArchTarget(target))
		if err != nil {
			return nil, nil, err
		}
		names, err := c.lowered(cm)
		return cm.CUBIN(), names, err
	}
	res, err := c.link(target, false)
	if err != nil {
		return nil, nil, err
	}
	return res.CUBIN, nil, nil
}

// Portable compiles the sources to PTX for the virtual architecture of
// target, such as compute_80. Several sources need link-time optimization,
// because only nvJitLink's LTO path emits PTX.
func (c *Config) Portable(target string) ([]byte, error) {
	if err := c.check(); err != nil {
		return nil, err
	}
	if !strings.HasPrefix(target, "compute_") {
		return nil, fmt.Errorf("target %q is not compute_*", target)
	}
	if len(c.Sources) == 1 {
		cm, err := c.compile(c.Sources[0], nvrtcx.WithFlag(nvrtc.FlagGpuArchitecture+"="+target))
		if err != nil {
			return nil, err
		}
		return []byte(cm.PTX()), nil
	}
	if c.NoLTO {
		return nil, fmt.Errorf("portable code for several sources needs link-time optimization")
	}
	res, err := c.link("sm_"+strings.TrimPrefix(target, "compute_"), true)
	if err != nil {
		return nil, err
	}
	return res.PTX, nil
}

// link compiles each source as relocatable device code and links them for
// the physical target, emitting PTX instead of machine code when ptx is set.
func (c *Config) link(target string, ptx bool) (*nvjitlinkx.Result, error) {
	virtual := "compute_" + strings.TrimPrefix(target, "sm_")
	inputs := make([]nvjitlinkx.Input, len(c.Sources))
	errs := make([]error, len(c.Sources))
	var wg sync.WaitGroup
	for i, src := range c.Sources {
		wg.Go(func() {
			opts := []nvrtcx.Option{nvrtcx.WithRelocatableDeviceCode(true)}
			if c.NoLTO {
				opts = append(opts, nvrtcx.WithPhysicalArchTarget(target))
			} else {
				opts = append(opts, nvrtcx.WithFlag(nvrtc.FlagGpuArchitecture+"="+virtual), nvrtcx.WithOutput(nvrtcx.OutputLTOIR))
			}
			cm, err := c.compile(src, opts...)
			if err != nil {
				errs[i] = err
				return
			}
			inputs[i], errs[i] = nvjitlinkx.InputFromArtifact(src, cm)
		})
	}
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			return nil, err
		}
	}
	flags := []string{nvjitlink.FlagArch + "=" + target}
	if !c.NoLTO {
		flags = append(flags, nvjitlink.FlagLTO)
	}
	if ptx {
		flags = append(flags, nvjitlink.FlagPTX)
	}
	res, err := nvjitlinkx.Link(inputs, flags...)
	if err != nil {
		err = fmt.Errorf("link %s: %w", target, err)
		if res != nil {
			if res.ErrorLog != "" {
				err = fmt.Errorf("%w\n%s", err, res.ErrorLog)
			}
			err = errors.Join(err, res.Close())
		}
		return nil, err
	}
	return res, nil
}

// Image builds machine code for each native target and, unless portable is
// empty, PTX for the portable target, and packages them as one image the
// driver loads directly. It also returns the lowered name of each name
// expression.
func (c *Config) Image(native []string, portable string) ([]byte, map[string]string, error) {
	type part struct {
		in    nvfatbinx.Input
		names map[string]string
		err   error
	}
	parts := make([]part, len(native))
	var wg sync.WaitGroup
	for i, target := range native {
		wg.Go(func() {
			code, names, err := c.Native(target)
			parts[i] = part{nvfatbinx.Input{Kind: nvfatbinx.CUBIN, Name: target, Arch: strings.TrimPrefix(target, "sm_"), Data: code}, names, err}
		})
	}
	var ptx []byte
	var ptxErr error
	if portable != "" {
		wg.Go(func() { ptx, ptxErr = c.Portable(portable) })
	}
	wg.Wait()
	var inputs []nvfatbinx.Input
	var names map[string]string
	for _, p := range parts {
		if p.err != nil {
			return nil, nil, fmt.Errorf("%s: %w", p.in.Name, p.err)
		}
		inputs = append(inputs, p.in)
		names = p.names
	}
	if ptxErr != nil {
		return nil, nil, fmt.Errorf("%s: %w", portable, ptxErr)
	}
	if portable != "" {
		inputs = append(inputs, nvfatbinx.Input{Kind: nvfatbinx.PTX, Name: portable, Arch: strings.TrimPrefix(portable, "compute_"), Data: ptx})
	}
	res, err := nvfatbinx.Build(inputs)
	if res != nil {
		err = errors.Join(err, res.Close())
	}
	if err != nil {
		return nil, nil, err
	}
	return res.Data, names, nil
}
