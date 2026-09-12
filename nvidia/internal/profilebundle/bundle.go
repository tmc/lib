// Package profilebundle reads portable CUDA profiling captures.
package profilebundle

import (
	"bytes"
	"crypto/sha256"
	"debug/elf"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

// Base is the profile's logical image base. Addresses are Base plus CUBIN
// file offsets; they are not sampled device program counters.
const Base = uint64(0x7f000000)

var Files = []string{"gpu.pprof", "go.trace", "main.go", "capture.go", "workload.go", "kernels.cu", "kernels.ptx", "kernels.cubin", "kernels.sass", "nvdisasm.txt", "README.md"}

type Manifest struct {
	Version      int                `json:"version"`
	Created      string             `json:"created"`
	Device       string             `json:"device"`
	Arch         string             `json:"architecture"`
	GoVersion    string             `json:"go_version"`
	NVRTCVersion string             `json:"nvrtc_version"`
	Elements     int                `json:"elements"`
	Passes       int                `json:"passes"`
	Rounds       int                `json:"fma_rounds"`
	Intervals    map[string][]int64 `json:"cuda_event_nanoseconds"`
	Verified     int                `json:"verified_results"`
	MaxError     float64            `json:"max_absolute_error"`
	SHA256       map[string]string  `json:"sha256"`
}

func Write(dir string, m Manifest) error {
	m.Version = 1
	m.SHA256 = make(map[string]string)
	for _, name := range Files {
		data, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			return err
		}
		m.SHA256[name] = fmt.Sprintf("%x", sha256.Sum256(data))
	}
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "manifest.json"), append(data, '\n'), 0644)
}

func Read(dir string) (*Manifest, error) {
	data, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		return nil, err
	}
	var m Manifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("read manifest: %w", err)
	}
	if m.Version != 1 {
		return nil, fmt.Errorf("unsupported bundle version %d", m.Version)
	}
	for _, name := range Files {
		data, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			return nil, err
		}
		if got := fmt.Sprintf("%x", sha256.Sum256(data)); got != m.SHA256[name] {
			return nil, fmt.Errorf("%s: sha256 mismatch", name)
		}
	}
	return &m, nil
}

type Code struct{ Kernels []*Kernel }

type Kernel struct {
	Name         string
	Offset       uint64
	Size         uint64
	SourceLine   int
	PTXLine      int
	Instructions []Instruction
}

type Instruction struct {
	Offset     uint64 // Offset within the kernel's .text section.
	Text       string
	SASSLine   int
	SourceLine int
}

func LoadCode(dir string) (*Code, error) {
	data := make(map[string][]byte)
	for _, name := range []string{"kernels.cubin", "kernels.ptx", "kernels.sass", "kernels.cu"} {
		b, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			return nil, err
		}
		data[name] = b
	}
	return ParseCode(data["kernels.cubin"], string(data["kernels.ptx"]), string(data["kernels.sass"]), string(data["kernels.cu"]))
}

var sourceLocation = regexp.MustCompile(`//## File "([^"]+)", line ([0-9]+)`)
var textSection = regexp.MustCompile(`^\s*\.section\s+\.text\.([^, \t]+)`)

func ParseCode(cubin []byte, ptx, sass, source string) (*Code, error) {
	e, err := elf.NewFile(bytes.NewReader(cubin))
	if err != nil {
		return nil, fmt.Errorf("read cubin: %w", err)
	}
	defer e.Close()
	if e.Machine != elf.EM_CUDA {
		return nil, fmt.Errorf("cubin architecture is %s, want CUDA", e.Machine)
	}
	p, err := cudaptx.ParseString(ptx)
	if err != nil {
		return nil, err
	}
	d, err := cudasass.ParseString(sass)
	if err != nil {
		return nil, err
	}
	lineNumbers := make(map[string]int)
	cuLines := make(map[int]int)
	cuLine := 0
	currentKernel := ""
	for i, line := range strings.Split(sass, "\n") {
		if m := textSection.FindStringSubmatch(line); m != nil {
			currentKernel, cuLine = m[1], 0
		}
		if m := sourceLocation.FindStringSubmatch(line); m != nil {
			cuLine = 0
			if filepath.Base(m[1]) == "kernels.cu" {
				cuLine, _ = strconv.Atoi(m[2])
			}
		}
		lineNumbers[currentKernel+"\x00"+line] = i + 1
		cuLines[i+1] = cuLine
	}
	var code Code
	for _, fn := range d.Functions {
		section := e.Section(".text." + fn.Name)
		pk := p.Kernel(fn.Name)
		if section == nil || pk == nil {
			return nil, fmt.Errorf("%s: missing CUBIN section or PTX entry", fn.Name)
		}
		if section.Size == 0 || section.Offset > uint64(len(cubin)) || section.Size > uint64(len(cubin))-section.Offset {
			return nil, fmt.Errorf("%s: invalid CUBIN section", fn.Name)
		}
		k := &Kernel{Name: fn.Name, Offset: section.Offset, Size: section.Size, PTXLine: pk.StartLine}
		for i, line := range strings.Split(source, "\n") {
			if strings.Contains(line, "void "+fn.Name+"(") {
				k.SourceLine = i + 1
				break
			}
		}
		if k.SourceLine == 0 {
			return nil, fmt.Errorf("%s: missing CUDA definition", fn.Name)
		}
		for _, inst := range fn.Instructions {
			if inst.Address >= section.Size || inst.Address%16 != 0 {
				return nil, fmt.Errorf("%s: invalid SASS offset %#x", fn.Name, inst.Address)
			}
			if len(k.Instructions) > 0 && inst.Address <= k.Instructions[len(k.Instructions)-1].Offset {
				return nil, fmt.Errorf("%s: unordered SASS offsets", fn.Name)
			}
			line := lineNumbers[fn.Name+"\x00"+inst.Raw]
			if line == 0 {
				return nil, fmt.Errorf("%s: missing SASS source line", fn.Name)
			}
			text := strings.TrimSpace(strings.TrimSpace(inst.Predicate+" "+inst.Opcode)+" "+inst.Operands) + ";"
			k.Instructions = append(k.Instructions, Instruction{inst.Address, text, line, cuLines[line]})
		}
		if len(k.Instructions) == 0 || k.Instructions[0].Offset != 0 {
			return nil, fmt.Errorf("%s: missing SASS entry instruction", fn.Name)
		}
		code.Kernels = append(code.Kernels, k)
	}
	if len(code.Kernels) == 0 {
		return nil, fmt.Errorf("no disassembled kernels")
	}
	slices.SortFunc(code.Kernels, func(a, b *Kernel) int {
		if a.Offset < b.Offset {
			return -1
		}
		if a.Offset > b.Offset {
			return 1
		}
		return 0
	})
	for i := 1; i < len(code.Kernels); i++ {
		if code.Kernels[i].Offset < code.Kernels[i-1].Offset+code.Kernels[i-1].Size {
			return nil, fmt.Errorf("overlapping CUBIN sections")
		}
	}
	return &code, nil
}

func (c *Code) Kernel(name string) *Kernel {
	for _, k := range c.Kernels {
		if k.Name == name {
			return k
		}
	}
	return nil
}
