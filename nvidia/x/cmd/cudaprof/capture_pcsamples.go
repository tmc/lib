package main

import (
	"context"
	"sort"
	"strconv"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

// pcSampleType is the dimension the agent's program-counter samples are
// counted in. It is the name the in-process sampler uses for the same
// quantity, so one capture reads the same way whichever collector produced
// it.
const pcSampleType = "pc_samples"

// projectPCSamples adds the agent's program-counter samples to the capture's
// profile as a sample type of their own, with the stack shape the profiler
// writes: a leaf frame named <kernel>.SASS whose line is the instruction's
// 1-based position in the kernel's disassembly, then the kernel's own frame
// carrying the program file and line that instruction came from. There are no
// host frames above it, because the agent samples the device from a drain
// thread and has no host stack to attribute the sample to.
//
// A sample CUPTI reported against a function the capture cannot disassemble
// is still projected, with no ordinal and no source line. Such a sample reads
// as unresolved rather than disappearing: the device did the work either way,
// and a total that silently omits it is wrong in the direction that matters.
func projectPCSamples(c *cudacapture.Capture, samples []capturePCSample, attachments map[string][]byte) {
	counts := make([]deviceCount, 0, len(samples))
	for _, s := range samples {
		counts = append(counts, deviceCount{ModuleID: s.ModuleID, Function: s.Function, Offset: s.Offset, Value: int64(s.Samples)})
	}
	projectCounts(c, pcSampleType, "samples", counts, attachments)
}

// A deviceCount is a quantity one instruction of one module accounted for:
// samples the hardware took there, or instructions it executed there. Both
// are measured per program counter and read the same way once projected, so
// they are projected by the same code.
type deviceCount struct {
	ModuleID uint32
	Function string
	Offset   uint64
	Value    int64
}

// projectCounts adds per-instruction counts to the capture's profile as a
// sample type of their own, in the stack shape described above.
func projectCounts(c *cudacapture.Capture, dimension, unit string, samples []deviceCount, attachments map[string][]byte) {
	if len(samples) == 0 || c.Profile == nil {
		return
	}
	// Every existing sample gains a zero in the new dimension: a kernel's
	// wall time is not a PC sample count.
	c.Profile.SampleType = append(c.Profile.SampleType, &profile.ValueType{Type: dimension, Unit: unit})
	width := len(c.Profile.SampleType)
	for _, s := range c.Profile.Sample {
		for len(s.Value) < width {
			s.Value = append(s.Value, 0)
		}
	}
	b := &pcBuilder{p: c.Profile, width: width, functions: map[string]*profile.Function{}, locations: map[string]*profile.Location{}}
	for _, f := range c.Profile.Function {
		b.nextFunction = max(b.nextFunction, f.ID)
	}
	for _, l := range c.Profile.Location {
		b.nextLocation = max(b.nextLocation, l.ID)
	}
	// Disassembly is per module and every sample of a module needs it, so it
	// is done once per module rather than once per sample.
	code := map[uint32]*moduleCode{}
	for _, m := range c.Modules {
		if m.DriverID == 0 {
			continue
		}
		code[m.DriverID] = newModuleCode(m, attachments[m.Code.CUBIN])
	}
	sort.Slice(samples, func(i, j int) bool {
		if samples[i].ModuleID != samples[j].ModuleID {
			return samples[i].ModuleID < samples[j].ModuleID
		}
		if samples[i].Function != samples[j].Function {
			return samples[i].Function < samples[j].Function
		}
		return samples[i].Offset < samples[j].Offset
	})
	for _, s := range samples {
		if s.Value == 0 {
			continue
		}
		b.add(s, code[s.ModuleID])
	}
}

// moduleCode is one module's disassembly, or the absence of one. A module
// whose image the capture did not keep, and a module nvdisasm could not be
// run on, are the same thing here: the offsets cannot be named, and the
// samples are projected without them.
type moduleCode struct {
	name string
	dis  *cudasass.Disassembly
}

func newModuleCode(m cudacapture.Module, cubin []byte) *moduleCode {
	c := &moduleCode{name: m.ID}
	if len(cubin) == 0 {
		return c
	}
	out, err := cudasass.Disassembler{}.Disassemble(context.Background(), cubin, cudasass.Options{PrintCode: true, SourceLines: true})
	if err != nil {
		return c
	}
	dis, err := cudasass.ParseString(string(out.Stdout))
	if err != nil {
		return c
	}
	c.dis = dis
	return c
}

// resolve names the sampled PC in the module's disassembly: the function as
// the disassembly spells it, the instruction's 1-based ordinal, and the
// program site it was compiled from. CUPTI reports the mangled name, which is
// what nvdisasm prints, so the match is normally exact.
func (c *moduleCode) resolve(function string, offset uint64) (name string, ordinal int, file string, line int) {
	if c == nil || c.dis == nil {
		return function, 0, "", 0
	}
	fn := findKernel(c.dis, function)
	if fn == nil {
		return function, 0, "", 0
	}
	for i, in := range fn.Instructions {
		if in.Address == offset {
			return fn.Name, i + 1, in.SourceFile, in.SourceLine
		}
	}
	return fn.Name, 0, "", 0
}

// pcBuilder appends samples to a profile, reusing a function and a location
// per distinct frame so a profile with many samples of one instruction stays
// the size of its distinct instructions.
type pcBuilder struct {
	p            *profile.Profile
	width        int
	functions    map[string]*profile.Function
	locations    map[string]*profile.Location
	nextFunction uint64
	nextLocation uint64
}

func (b *pcBuilder) add(s deviceCount, code *moduleCode) {
	name, ordinal, file, line := code.resolve(s.Function, s.Offset)
	unit := "module"
	if code != nil && code.name != "" {
		unit = code.name
	}
	locs := []*profile.Location{b.location(name+".SASS", unit+".sass", int64(ordinal))}
	locs = append(locs, b.location(name, file, int64(line)))
	sample := &profile.Sample{Location: locs, Value: make([]int64, b.width)}
	sample.Value[b.width-1] = s.Value
	if s.ModuleID != 0 {
		sample.Label = map[string][]string{"module_id": {strconv.FormatUint(uint64(s.ModuleID), 10)}}
	}
	b.p.Sample = append(b.p.Sample, sample)
}

func (b *pcBuilder) location(name, file string, line int64) *profile.Location {
	key := name + "\x00" + file + "\x00" + strconv.FormatInt(line, 10)
	if loc, ok := b.locations[key]; ok {
		return loc
	}
	fn, ok := b.functions[name+"\x00"+file]
	if !ok {
		b.nextFunction++
		fn = &profile.Function{ID: b.nextFunction, Name: name, SystemName: name, Filename: file}
		b.functions[name+"\x00"+file] = fn
		b.p.Function = append(b.p.Function, fn)
	}
	b.nextLocation++
	loc := &profile.Location{ID: b.nextLocation, Line: []profile.Line{{Function: fn, Line: line}}}
	b.p.Location = append(b.p.Location, loc)
	b.locations[key] = loc
	return loc
}
