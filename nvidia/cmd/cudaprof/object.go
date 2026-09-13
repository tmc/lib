package main

import (
	"crypto/sha256"
	"fmt"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/google/pprof/driver"
	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/tileirdisasm"
)

// PTX addresses are byte offsets in the recorded text, with a separate logical
// image base. They are neither native instruction addresses nor sampled PCs.
// The base and the offsets come from x/cudaprof so that an address synthesized
// here is the one a capture would have written.
const ptxBase = cudaprof.PTXMappingStart

type objectTool struct {
	images map[string]*objectImage
	ptx    *objectImage
	tileir *objectImage
}

type objectImage struct {
	file, buildID string
	size          uint64
	functions     []objectFunction
}

type objectFunction struct {
	name         string
	start, end   uint64
	instructions []driver.Inst
}

func newObjectTool(c *cudacapture.Capture, code *cudacapture.Code, dir string) (*objectTool, error) {
	cubinData := c.Files[c.Code.CUBIN]
	cubinSum := fmt.Sprintf("%x", sha256.Sum256(cubinData))
	cubinPath := filepath.Join(dir, c.Code.CUBIN)
	sassPath := filepath.Join(dir, c.Code.SASS)

	sass := &objectImage{file: cubinPath, buildID: cubinSum}
	for _, k := range code.Kernels {
		fn := objectFunction{name: k.Name + ".SASS", start: k.Offset, end: k.Offset + k.Size}
		for _, in := range k.Instructions {
			fn.instructions = append(fn.instructions, driver.Inst{
				Addr:     k.Offset + in.Offset,
				Text:     in.Text,
				Function: fn.name,
				File:     sassPath,
				Line:     in.SASSLine,
			})
		}
		sass.functions = append(sass.functions, fn)
	}

	ptxData := c.Files[c.Code.PTX]
	ptxSum := fmt.Sprintf("%x", sha256.Sum256(ptxData))
	ptxPath := filepath.Join(dir, c.Code.PTX)

	ptx, err := ptxImage(ptxPath, ptxSum, string(ptxData))
	if err != nil {
		return nil, err
	}

	tool := &objectTool{
		images: map[string]*objectImage{sass.file: sass, ptx.file: ptx},
		ptx:    ptx,
	}

	if c.Code.TileIR != "" && len(c.Files[c.Code.TileIR]) > 0 {
		tileData := c.Files[c.Code.TileIR]
		tileSum := fmt.Sprintf("%x", sha256.Sum256(tileData))
		tilePath := filepath.Join(dir, c.Code.TileIR)
		tileImg, err := tileIRImage(tilePath, tileSum, string(tileData))
		if err == nil && tileImg != nil {
			tool.images[tileImg.file] = tileImg
			tool.tileir = tileImg
		}
	}

	return tool, nil
}

func tileIRImage(file, buildID, text string) (*objectImage, error) {
	mod, err := tileirdisasm.ParseString(text)
	if err != nil {
		return nil, err
	}
	image := &objectImage{file: file, buildID: buildID, size: uint64(len(text))}
	for _, fn := range mod.Functions {
		objFn := objectFunction{name: fn.Name + ".IR"}
		for _, in := range fn.Operations {
			objFn.instructions = append(objFn.instructions, driver.Inst{
				Addr:     uint64(in.Line),
				Text:     strings.TrimSpace(in.Raw),
				Function: objFn.name,
				File:     file,
				Line:     in.Line,
			})
		}
		image.functions = append(image.functions, objFn)
	}
	return image, nil
}

func ptxImage(file, buildID, text string) (*objectImage, error) {
	mod, err := cudaptx.ParseString(text)
	if err != nil {
		return nil, err
	}
	offsets := cudaprof.PTXLineOffsets(text)
	image := &objectImage{file: file, buildID: buildID, size: uint64(len(text))}
	for _, k := range mod.Kernels {
		if k.StartLine < 1 || k.EndLine < k.StartLine || k.EndLine+1 >= len(offsets) || len(k.Instructions) == 0 {
			return nil, fmt.Errorf("%s: incomplete PTX function", k.Name)
		}
		fn := objectFunction{name: k.Name + ".PTX", start: offsets[k.StartLine], end: offsets[k.EndLine+1]}
		for _, in := range k.Instructions {
			fn.instructions = append(fn.instructions, driver.Inst{Addr: offsets[in.Line], Text: strings.TrimSpace(in.Raw), Function: fn.name, File: file, Line: in.Line})
		}
		image.functions = append(image.functions, fn)
	}
	return image, nil
}

// mapPTX gives existing PTX source frames file coordinates for pprof's assembly
// view. It does not add samples or assign instruction-level costs.
func (t *objectTool) mapPTX(p *profile.Profile) {
	var id uint64
	var mapping *profile.Mapping
	for _, m := range p.Mapping {
		id = max(id, m.ID)
		if filepath.Base(m.File) == filepath.Base(t.ptx.file) {
			mapping = m
		}
	}
	if mapping == nil {
		mapping = &profile.Mapping{ID: id + 1}
		p.Mapping = append(p.Mapping, mapping)
	}
	mapping.Start, mapping.Limit, mapping.Offset = ptxBase, ptxBase+t.ptx.size, 0
	mapping.File, mapping.BuildID = t.ptx.file, t.ptx.buildID
	// Not HasInlineFrames: this view gives each location one line, exactly as
	// the captured file does. Claiming inline frames here would contradict the
	// file the viewer was handed.
	mapping.HasFunctions, mapping.HasFilenames, mapping.HasLineNumbers = true, true, true
	mapping.HasInlineFrames = false
	for _, loc := range p.Location {
		if len(loc.Line) != 1 {
			continue
		}
		line := loc.Line[0]
		if line.Function == nil {
			continue
		}
		for _, fn := range t.ptx.functions {
			if line.Function.Name != fn.name {
				continue
			}
			loc.Mapping, loc.Address = mapping, ptxBase+fn.instructions[0].Addr
			for _, in := range fn.instructions {
				if int64(in.Line) == line.Line {
					loc.Address = ptxBase + in.Addr
					break
				}
			}
		}
	}
}

func (t *objectTool) Open(file string, start, limit, offset uint64, relocationSymbol string) (driver.ObjFile, error) {
	image := t.images[filepath.Clean(file)]
	if image == nil {
		return nil, fmt.Errorf("unsupported object %q", file)
	}
	return &objectFile{image: image, start: start, limit: limit, offset: offset}, nil
}

func (t *objectTool) Disasm(file string, start, end uint64, intelSyntax bool) ([]driver.Inst, error) {
	image := t.images[filepath.Clean(file)]
	if image == nil {
		return nil, fmt.Errorf("unsupported object %q", file)
	}
	var out []driver.Inst
	for _, fn := range image.functions {
		for _, in := range fn.instructions {
			if in.Addr >= start && in.Addr < end {
				out = append(out, in)
			}
		}
	}
	return out, nil
}

type objectFile struct {
	image                *objectImage
	start, limit, offset uint64
}

func (f *objectFile) Name() string    { return f.image.file }
func (f *objectFile) BuildID() string { return f.image.buildID }
func (f *objectFile) Close() error    { return nil }

func (f *objectFile) ObjAddr(addr uint64) (uint64, error) {
	if addr < f.start || addr >= f.limit {
		return 0, fmt.Errorf("address %#x outside mapping", addr)
	}
	return addr - f.start + f.offset, nil
}

func (f *objectFile) SourceLine(addr uint64) ([]driver.Frame, error) {
	a, err := f.ObjAddr(addr)
	if err != nil {
		return nil, err
	}
	for _, fn := range f.image.functions {
		if a < fn.start || a >= fn.end {
			continue
		}
		i := sort.Search(len(fn.instructions), func(i int) bool { return fn.instructions[i].Addr > a }) - 1
		if i < 0 {
			return nil, fmt.Errorf("no instruction at %#x", a)
		}
		in := fn.instructions[i]
		return []driver.Frame{{Func: fn.name, File: in.File, Line: in.Line, StartLine: fn.instructions[0].Line}}, nil
	}
	return nil, fmt.Errorf("no kernel at %#x", a)
}

func (f *objectFile) Symbols(re *regexp.Regexp, addr uint64) ([]*driver.Sym, error) {
	var syms []*driver.Sym
	for _, fn := range f.image.functions {
		if re != nil && !re.MatchString(fn.name) {
			continue
		}
		if addr != 0 && (addr < fn.start || addr >= fn.end) {
			continue
		}
		syms = append(syms, &driver.Sym{Name: []string{fn.name}, File: f.Name(), Start: fn.start, End: fn.end - 1})
	}
	return syms, nil
}
