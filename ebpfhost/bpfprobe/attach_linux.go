//go:build linux && (amd64 || arm64)

package bpfprobe

import (
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"unsafe"

	"github.com/tmc/lib/ebpfhost/bpfload"
)

func disablePerf(file *os.File) error {
	_, _, errno := syscall.Syscall(syscall.SYS_IOCTL, file.Fd(), 0x2401, 0)
	if errno != 0 {
		return fmt.Errorf("bpf probe: disable: %w", errno)
	}
	return nil
}

func outputMaps(p *Probe) (int, int, []*os.File, error) {
	var files []*os.File
	for i, keySize := range []uint32{40, 32} {
		o, err := bpfload.CreateMap(bpfload.MapSpec{Type: 1, KeySize: keySize, ValueSize: 8, MaxEntries: 4096, Name: fmt.Sprintf("lib_output_%d", i)})
		if err != nil {
			for _, f := range files {
				f.Close()
			}
			return -1, -1, nil, err
		}
		p.extras = append(p.extras, o)
		f, err := o.File()
		if err != nil {
			for _, f := range files {
				f.Close()
			}
			return -1, -1, nil, err
		}
		files = append(files, f)
	}
	return int(files[0].Fd()), int(files[1].Fd()), files, nil
}

func stackMap(p *Probe) (int, *os.File, error) {
	o, err := bpfload.CreateStackMap(1024, 8)
	if err != nil {
		return -1, nil, err
	}
	p.stacks = o
	file, err := o.File()
	if err != nil {
		return -1, nil, err
	}
	return int(file.Fd()), file, nil
}

// AttachGroup creates one kernel ring for all configured attachments.
func AttachGroup(configs []Config) (_ *Group, err error) {
	if len(configs) == 0 || len(configs) > 256 {
		return nil, errors.New("bpf probe: attachment count outside 1..256")
	}
	validated := make([]Config, len(configs))
	for i, c := range configs {
		validated[i], err = validateConfig(c)
		if err != nil {
			return nil, err
		}
		if validated[i].Capacity != validated[0].Capacity {
			return nil, errors.New("bpf probe: inconsistent group ring capacity")
		}
	}
	layout, err := taskGenerationLayout()
	if err != nil {
		return nil, err
	}
	for i := range validated {
		validated[i].generationOffset = layout.start
		validated[i].execOffset = layout.exec
		validated[i].execSize = layout.execSize
	}
	g := &Group{Probe: &Probe{KernelBTFHash: layout.hash, TaskStartOffset: layout.start, TaskExecOffset: layout.exec, TaskExecSize: layout.execSize}}
	defer func() {
		if err != nil {
			err = errors.Join(err, g.Close())
		}
	}()
	g.events, err = bpfload.CreateRingMap(validated[0].Capacity)
	if err != nil {
		return nil, err
	}
	g.counts, err = bpfload.CreateMap(bpfload.MapSpec{Type: 2, KeySize: 4, ValueSize: 8, MaxEntries: 9, Name: "lib_probe_count"})
	if err != nil {
		return nil, err
	}
	events, err := g.events.File()
	if err != nil {
		return nil, err
	}
	defer func() { err = errors.Join(err, events.Close()) }()
	counts, err := g.counts.File()
	if err != nil {
		return nil, err
	}
	defer func() { err = errors.Join(err, counts.Close()) }()
	g.ring, err = OpenRing(int(events.Fd()), uint32(validated[0].Capacity))
	if err != nil {
		return nil, err
	}
	stackFD := -1
	outputFD, depthFD := -1, -1
	for _, c := range validated {
		if c.OutputPointer {
			var files []*os.File
			outputFD, depthFD, files, err = outputMaps(g.Probe)
			if err != nil {
				return nil, err
			}
			for _, f := range files {
				defer func() { err = errors.Join(err, f.Close()) }()
			}
			break
		}
	}
	for _, c := range validated {
		if c.UserStack {
			var file *os.File
			stackFD, file, err = stackMap(g.Probe)
			if err != nil {
				return nil, err
			}
			defer func() { err = errors.Join(err, file.Close()) }()
			break
		}
	}
	for _, c := range validated {
		p := &Probe{}
		g.probes = append(g.probes, p)
		if err = attachTo(c, p, int(events.Fd()), int(counts.Fd()), outputFD, depthFD, stackFD); err != nil {
			return nil, err
		}
	}
	return g, nil
}

// Attach loads and attaches an upstream Linux BPF uprobe on amd64 or arm64. It does
// not modify tracefs, attach GPU driver policies, or retry kernel refusals.
func Attach(config Config) (_ *Probe, err error) {
	c, err := validateConfig(config)
	if err != nil {
		return nil, err
	}
	layout, err := taskGenerationLayout()
	if err != nil {
		return nil, err
	}
	c.generationOffset = layout.start
	c.execOffset = layout.exec
	c.execSize = layout.execSize
	p := &Probe{KernelBTFHash: layout.hash, TaskStartOffset: layout.start, TaskExecOffset: layout.exec, TaskExecSize: layout.execSize}
	defer func() {
		if err != nil {
			err = errors.Join(err, p.Close())
		}
	}()
	p.events, err = bpfload.CreateRingMap(c.Capacity)
	if err != nil {
		return nil, err
	}
	p.counts, err = bpfload.CreateMap(bpfload.MapSpec{Type: 2, KeySize: 4, ValueSize: 8, MaxEntries: 9, Name: "lib_probe_count"})
	if err != nil {
		return nil, err
	}
	events, err := p.events.File()
	if err != nil {
		return nil, err
	}
	defer func() { err = errors.Join(err, events.Close()) }()
	counts, err := p.counts.File()
	if err != nil {
		return nil, err
	}
	defer func() { err = errors.Join(err, counts.Close()) }()
	p.ring, err = OpenRing(int(events.Fd()), uint32(c.Capacity))
	if err != nil {
		return nil, err
	}
	outputFD, depthFD := -1, -1
	if c.OutputPointer {
		var files []*os.File
		outputFD, depthFD, files, err = outputMaps(p)
		if err != nil {
			return nil, err
		}
		for _, f := range files {
			defer func() { err = errors.Join(err, f.Close()) }()
		}
	}
	stackFD := -1
	if c.UserStack {
		var file *os.File
		stackFD, file, err = stackMap(p)
		if err != nil {
			return nil, err
		}
		defer func() { err = errors.Join(err, file.Close()) }()
	}
	if err = attachTo(c, p, int(events.Fd()), int(counts.Fd()), outputFD, depthFD, stackFD); err != nil {
		return nil, err
	}
	return p, nil
}

func attachTo(c Config, p *Probe, ringFD, countFD, outputFD, depthFD, stackFD int) (err error) {
	p.program, _, err = bpfload.Load(bpfload.ProgramSpec{Type: 2, Instructions: program(ringFD, countFD, c.Tag, c.Return, c.OutputPointer, outputFD, depthFD, c.CorrelationTag, c.TGID, c.generationOffset, c.execOffset, c.execSize, c.UserStack, stackFD)})
	if err != nil {
		return err
	}
	programFile, err := p.program.File()
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, programFile.Close()) }()
	raw, err := os.ReadFile("/sys/bus/event_source/devices/uprobe/type")
	if err != nil {
		return fmt.Errorf("bpf probe: uprobe pmu: %w", err)
	}
	pmu, err := strconv.ParseUint(strings.TrimSpace(string(raw)), 10, 32)
	if err != nil {
		return fmt.Errorf("bpf probe: uprobe pmu type: %w", err)
	}
	configBits := uint64(0)
	if c.Return {
		raw, err = os.ReadFile("/sys/bus/event_source/devices/uprobe/format/retprobe")
		if err != nil {
			return err
		}
		s := strings.TrimSpace(string(raw))
		if !strings.HasPrefix(s, "config:") {
			return fmt.Errorf("bpf probe: unsupported retprobe format %q", s)
		}
		bit, e := strconv.ParseUint(strings.TrimPrefix(s, "config:"), 10, 6)
		if e != nil {
			return e
		}
		configBits = 1 << bit
	}
	path, err := syscall.ByteSliceFromString(c.Path)
	if err != nil {
		return err
	}
	// perf_event_attr through config2; bit zero is disabled. The attribute
	// stays disabled until SET_BPF succeeds, so setup cannot silently lose events.
	attr := struct {
		Type, Size                                         uint32
		Config, SamplePeriod, SampleType, ReadFormat, Bits uint64
		Wakeup, BPType                                     uint32
		Config1, Config2                                   uint64
	}{Type: uint32(pmu), Size: 72, Config: configBits, Bits: 1, Config1: uint64(uintptr(unsafe.Pointer(&path[0]))), Config2: c.Offset}
	fd, _, errno := syscall.Syscall6(sysPerfEventOpen, uintptr(unsafe.Pointer(&attr)), uintptr(c.PID), ^uintptr(0), ^uintptr(0), 8, 0)
	runtime.KeepAlive(path)
	if errno != 0 {
		return fmt.Errorf("bpf probe: perf open: %w", errno)
	}
	p.perf = os.NewFile(fd, "bpf-uprobe")
	if _, _, errno = syscall.Syscall(syscall.SYS_IOCTL, fd, 0x40042408, programFile.Fd()); errno != 0 {
		return fmt.Errorf("bpf probe: attach program: %w", errno)
	}
	if _, _, errno = syscall.Syscall(syscall.SYS_IOCTL, fd, 0x2400, 0); errno != 0 {
		return fmt.Errorf("bpf probe: enable: %w", errno)
	}
	return nil
}

func program(ringFD, countFD int, tag uint64, returning, outputPointer bool, outputFD, depthFD int, pair uint64, tgid uint32, generationOffset, execOffset, execSize uint32, userStack bool, stackFD int) []byte {
	var out []byte
	labels := make(map[string]int)
	type fixup struct {
		index int
		label string
	}
	var fixups []fixup
	emit := func(code, dst, src byte, offset int16, imm int32) {
		b := make([]byte, 8)
		b[0] = code
		b[1] = dst | src<<4
		binary.LittleEndian.PutUint16(b[2:], uint16(offset))
		binary.LittleEndian.PutUint32(b[4:], uint32(imm))
		out = append(out, b...)
	}
	imm64 := func(dst byte, src byte, value uint64) {
		emit(0x18, dst, src, 0, int32(value))
		emit(0, 0, 0, 0, int32(value>>32))
	}
	label := func(name string) { labels[name] = len(out) / 8 }
	jump := func(code, dst, src byte, imm int32, name string) {
		fixups = append(fixups, fixup{len(out) / 8, name})
		emit(code, dst, src, 0, imm)
	}
	emit(0xb7, 9, 0, 0, 0) // output acquisition flags survive helper calls
	emit(0xbf, 6, 1, 0, 0) // preserve pt_regs
	emit(0x85, 0, 0, 0, 5)
	emit(0x7b, 10, 0, -168, 0) // ktime
	emit(0x62, 10, 0, -24, 0)
	emit(0x62, 10, 0, -20, 0)
	emit(0x85, 0, 0, 0, 35)
	emit(0xbf, 3, 0, 0, 0)
	emit(0x07, 3, 0, 0, int32(generationOffset))
	emit(0xbf, 1, 10, 0, 0)
	emit(0x07, 1, 0, 0, -24)
	emit(0xb7, 2, 0, 0, 8)
	emit(0x85, 0, 0, 0, 113)
	jump(0x55, 0, 0, 0, "generationmissing")
	emit(0x79, 0, 10, -24, 0)
	jump(0x55, 0, 0, 0, "generationdone")
	label("generationmissing")
	emit(0xb7, 9, 0, 0, 1<<22)
	label("generationdone")
	emit(0x62, 10, 0, -16, 0)
	emit(0x62, 10, 0, -12, 0)
	emit(0x85, 0, 0, 0, 35)
	emit(0xbf, 3, 0, 0, 0)
	emit(0x07, 3, 0, 0, int32(execOffset))
	emit(0xbf, 1, 10, 0, 0)
	emit(0x07, 1, 0, 0, -16)
	emit(0xb7, 2, 0, 0, int32(execSize))
	emit(0x85, 0, 0, 0, 113)
	jump(0x15, 0, 0, 0, "execdone")
	emit(0x47, 9, 0, 0, 1<<23)
	label("execdone")

	emit(0x85, 0, 0, 0, 14)
	if tgid != 0 {
		emit(0xbf, 1, 0, 0, 0)
		emit(0x77, 1, 0, 0, 32)
		jump(0x55, 1, 0, int32(tgid), "exit")
	}
	emit(0x7b, 10, 0, -160, 0) // pid/tgid
	imm64(1, 0, tag)
	emit(0x7b, 10, 1, -152, 0)
	// Initialize the complete argument area before optional user-memory reads.
	emit(0xb7, 1, 0, 0, 0)
	for i := 0; i < 11; i++ {
		emit(0x7b, 10, 1, int16(-144+8*i), 0)
	}

	if outputPointer {
		jump(0x55, 9, 0, 0, "outputdone")
		lookup := func(fd int, key int32) {
			imm64(1, 1, uint64(fd))
			emit(0xbf, 2, 10, 0, 0)
			emit(0x07, 2, 0, 0, key)
			emit(0x85, 0, 0, 0, 1)
		}
		update := func(fd int, key, value int32) {
			imm64(1, 1, uint64(fd))
			emit(0xbf, 2, 10, 0, 0)
			emit(0x07, 2, 0, 0, key)
			emit(0xbf, 3, 10, 0, 0)
			emit(0x07, 3, 0, 0, value)
			emit(0xb7, 4, 0, 0, 0)
			emit(0x85, 0, 0, 0, 2)
		}
		remove := func(fd int, key int32) {
			imm64(1, 1, uint64(fd))
			emit(0xbf, 2, 10, 0, 0)
			emit(0x07, 2, 0, 0, key)
			emit(0x85, 0, 0, 0, 3)
		}
		// PID/TGID and explicit entry/return pair key, followed by nesting depth.
		emit(0x79, 1, 10, -160, 0)
		emit(0x7b, 10, 1, -208, 0)
		emit(0x79, 1, 10, -24, 0)
		emit(0x7b, 10, 1, -200, 0)
		emit(0x79, 1, 10, -16, 0)
		emit(0x7b, 10, 1, -192, 0)
		imm64(1, 0, pair)
		emit(0x7b, 10, 1, -184, 0)
		lookup(depthFD, -208)
		if !returning {
			emit(0xb7, 8, 0, 0, 1)
			jump(0x15, 0, 0, 0, "newdepth")
			emit(0x79, 8, 0, 0, 0)
			emit(0x07, 8, 0, 0, 1)
			label("newdepth")
			emit(0x7b, 10, 8, -216, 0)
			update(depthFD, -208, -216)
			jump(0x55, 0, 0, 0, "maperror")
			jump(0x25, 8, 0, 8, "overflow")
			emit(0x7b, 10, 8, -176, 0)
			emit(0x79, 1, 6, argumentRegisterOffsets[0], 0)
			emit(0x7b, 10, 1, -224, 0)
			update(outputFD, -208, -224)
			jump(0x55, 0, 0, 0, "maperror")
			jump(0x05, 0, 0, 0, "outputdone")
		} else {
			jump(0x15, 0, 0, 0, "missing")
			emit(0x79, 8, 0, 0, 0)
			jump(0x15, 8, 0, 0, "missing")
			emit(0xbf, 7, 8, 0, 0)
			emit(0x07, 7, 0, 0, -1)
			emit(0x7b, 10, 7, -216, 0)
			jump(0x15, 7, 0, 0, "deletedepth")
			update(depthFD, -208, -216)
			jump(0x55, 0, 0, 0, "maperror")
			jump(0x05, 0, 0, 0, "depthdone")
			label("deletedepth")
			remove(depthFD, -208)
			jump(0x55, 0, 0, 0, "maperror")
			label("depthdone")
			jump(0x25, 8, 0, 8, "overflow")
			emit(0x7b, 10, 8, -176, 0)
			lookup(outputFD, -208)
			jump(0x15, 0, 0, 0, "missing")
			emit(0x79, 7, 0, 0, 0)
			remove(outputFD, -208)
			jump(0x55, 0, 0, 0, "maperror")
			emit(0x79, 0, 6, returnRegisterOffset, 0)
			emit(0xbc, 0, 0, 0, 0)
			jump(0x55, 0, 0, 0, "outputdone") // CUresult/int low32
			emit(0xbf, 1, 10, 0, 0)
			emit(0x07, 1, 0, 0, -144)
			emit(0xb7, 2, 0, 0, 8)
			emit(0xbf, 3, 7, 0, 0)
			emit(0x85, 0, 0, 0, 112)
			jump(0x55, 0, 0, 0, "unreadable")
			emit(0xb7, 9, 0, 0, 1<<17)
			jump(0x05, 0, 0, 0, "outputdone")
		}
		if returning {
			label("missing")
			emit(0xb7, 9, 0, 0, 1<<18)
			jump(0x05, 0, 0, 0, "outputdone")
		}
		label("overflow")
		emit(0xb7, 9, 0, 0, 1<<19)
		jump(0x05, 0, 0, 0, "outputdone")
		if returning {
			label("unreadable")
			emit(0xb7, 9, 0, 0, 1<<20)
			jump(0x05, 0, 0, 0, "outputdone")
		}
		label("maperror")
		emit(0xb7, 9, 0, 0, 1<<21)
		label("outputdone")
	}
	if !returning {
		offsets := argumentRegisterOffsets
		for i, offset := range offsets {
			emit(0x79, 1, 6, offset, 0)
			emit(0x7b, 10, 1, int16(-144+8*i), 0)
		}
		emit(0xbf, 1, 10, 0, 0)
		emit(0x07, 1, 0, 0, int32(-144+8*len(offsets)))
		emit(0xb7, 2, 0, 0, int32(8*(11-len(offsets))))
		emit(0x79, 3, 6, stackPointerOffset, 0)
		emit(0x07, 3, 0, 0, stackArgumentOffset)
		emit(0x85, 0, 0, 0, 112)
		emit(0xb7, 1, 0, 0, int32((1<<len(offsets))-1))
		emit(0x55, 0, 0, 1, 0)
		emit(0xb7, 1, 0, 0, 0x7ff)
	} else {
		emit(0xb7, 1, 0, 0, 1<<16)
	}
	emit(0x4f, 1, 9, 0, 0)
	emit(0x7b, 10, 1, -32, 0) // argument and output presence flags
	for i, offset := range []int16{returnRegisterOffset, stackPointerOffset, instructionPointerOffset} {
		emit(0x79, 1, 6, offset, 0)
		emit(0x7b, 10, 1, int16(-56+8*i), 0)
	}

	// An explicit requested bit distinguishes omitted collection from failure.
	emit(0xb7, 0, 0, 0, -1)
	emit(0x7b, 10, 0, -8, 0)
	if userStack {
		emit(0xbf, 1, 6, 0, 0)
		imm64(2, 1, uint64(stackFD))
		emit(0xb7, 3, 0, 0, 256)
		emit(0x85, 0, 0, 0, 27)
		emit(0x7b, 10, 0, -8, 0)
		jump(0x75, 0, 0, 0, "stackok")
		emit(0x47, 9, 0, 0, 1<<25)
		label("stackok")
		emit(0x79, 1, 10, -32, 0)
		emit(0x47, 1, 0, 0, 1<<24)
		emit(0x4f, 1, 9, 0, 0)
		emit(0x7b, 10, 1, -32, 0)
	}
	imm64(1, 1, uint64(ringFD))
	emit(0xbf, 2, 10, 0, 0)
	emit(0x07, 2, 0, 0, -168)
	emit(0xb7, 3, 0, 0, 168)
	emit(0xb7, 4, 0, 0, 0)
	emit(0x85, 0, 0, 0, 130)
	// Store index 0 for successful submission, 1 for a refused/full ring.
	emit(0xb7, 1, 0, 0, 0)
	emit(0x55, 0, 0, 1, 0)
	emit(0x05, 0, 0, 1, 0)
	emit(0xb7, 1, 0, 0, 1)
	emit(0x63, 10, 1, -168, 0)
	imm64(1, 1, uint64(countFD))
	emit(0xbf, 2, 10, 0, 0)
	emit(0x07, 2, 0, 0, -168)
	emit(0x85, 0, 0, 0, 1)
	emit(0x15, 0, 0, 2, 0)
	emit(0xb7, 1, 0, 0, 1)
	emit(0xdb, 0, 1, 0, 0)

	if outputPointer {
		// Every state failure increments a separate bounded counter, even if the
		// corresponding event could not be submitted to a full ring.
		for bit := 18; bit <= 21; bit++ {
			emit(0xbf, 1, 9, 0, 0)
			emit(0x57, 1, 0, 0, 1<<bit)
			jump(0x15, 1, 0, 0, fmt.Sprintf("state%d", bit))
			emit(0x62, 10, 0, -168, int32(bit-16))
			imm64(1, 1, uint64(countFD))
			emit(0xbf, 2, 10, 0, 0)
			emit(0x07, 2, 0, 0, -168)
			emit(0x85, 0, 0, 0, 1)
			jump(0x15, 0, 0, 0, fmt.Sprintf("state%d", bit))
			emit(0xb7, 1, 0, 0, 1)
			emit(0xdb, 0, 1, 0, 0)
			label(fmt.Sprintf("state%d", bit))
		}
	}
	emit(0xbf, 1, 9, 0, 0)
	emit(0x57, 1, 0, 0, 1<<22)
	jump(0x15, 1, 0, 0, "generationcounterdone")
	emit(0x62, 10, 0, -168, 6)
	imm64(1, 1, uint64(countFD))
	emit(0xbf, 2, 10, 0, 0)
	emit(0x07, 2, 0, 0, -168)
	emit(0x85, 0, 0, 0, 1)
	jump(0x15, 0, 0, 0, "generationcounterdone")
	emit(0xb7, 1, 0, 0, 1)
	emit(0xdb, 0, 1, 0, 0)
	label("generationcounterdone")
	emit(0xbf, 1, 9, 0, 0)
	emit(0x57, 1, 0, 0, 1<<23)
	jump(0x15, 1, 0, 0, "execcounterdone")
	emit(0x62, 10, 0, -168, 7)
	imm64(1, 1, uint64(countFD))
	emit(0xbf, 2, 10, 0, 0)
	emit(0x07, 2, 0, 0, -168)
	emit(0x85, 0, 0, 0, 1)
	jump(0x15, 0, 0, 0, "execcounterdone")
	emit(0xb7, 1, 0, 0, 1)
	emit(0xdb, 0, 1, 0, 0)
	label("execcounterdone")

	emit(0xbf, 1, 9, 0, 0)
	emit(0x57, 1, 0, 0, 1<<25)
	jump(0x15, 1, 0, 0, "stackcounterdone")
	emit(0x62, 10, 0, -168, 8)
	imm64(1, 1, uint64(countFD))
	emit(0xbf, 2, 10, 0, 0)
	emit(0x07, 2, 0, 0, -168)
	emit(0x85, 0, 0, 0, 1)
	jump(0x15, 0, 0, 0, "stackcounterdone")
	emit(0xb7, 1, 0, 0, 1)
	emit(0xdb, 0, 1, 0, 0)
	label("stackcounterdone")
	label("exit")
	emit(0xb7, 0, 0, 0, 0)
	emit(0x95, 0, 0, 0, 0)
	for _, f := range fixups {
		offset := labels[f.label] - f.index - 1
		binary.LittleEndian.PutUint16(out[f.index*8+2:], uint16(int16(offset)))
	}
	return out
}
