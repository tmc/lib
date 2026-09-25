package main

// An opt is a value a device or process may not report.
// The zero opt is unset.
type opt[T any] struct {
	v  T
	ok bool
}

func some[T any](v T) opt[T] { return opt[T]{v, true} }

// A processKind says which NVML process lists named a process.
type processKind int

const (
	graphic processKind = iota
	compute
	graphicCompute
)

// A process is one process holding a GPU context.
type process struct {
	pid     int
	kind    processKind
	user    opt[string]
	cmdline opt[string]
	gpuMem  opt[uint64] // bytes
	memPct  opt[uint32] // gpuMem as a percentage of the device's memory
	gpuUtil opt[uint32]
	encUtil opt[uint32]
	decUtil opt[uint32]
	cpuUtil opt[uint32]
	hostMem opt[uint64] // resident bytes
}

// A device holds what is known about one GPU.
// The static fields are read once; the rest at every refresh.
type device struct {
	index int

	name         opt[string]
	arch         opt[string]
	maxPcieGen   opt[uint32]
	maxPcieWidth opt[uint32]
	tempSlowdown opt[uint32]
	tempShutdown opt[uint32]
	ecc          bool // reports memory ECC error counts
	nvlink       nvlink

	gpuClock    opt[uint32] // MHz
	gpuClockMax opt[uint32]
	memClock    opt[uint32]
	memClockMax opt[uint32]
	gpuUtil     opt[uint32] // percent
	effLoad     opt[uint32] // gpuUtil scaled by power/powerMax
	memUtil     opt[uint32]
	encUtil     opt[uint32]
	decUtil     opt[uint32]
	memTotal    opt[uint64] // bytes
	memUsed     opt[uint64]
	memFree     opt[uint64]
	pcieGen     opt[uint32]
	pcieWidth   opt[uint32]
	pcieRx      opt[uint32] // KB/s
	pcieTx      opt[uint32]
	fan         opt[uint32] // percent
	temp        opt[uint32] // °C
	power       opt[uint32] // mW
	powerMax    opt[uint32]
	mig         opt[bool]
	eccCorr     opt[uint64] // volatile corrected ECC errors
	eccUncorr   opt[uint64]
	nvlinkRate  opt[uint64] // NVLink TX+RX, KB/s

	procs []process
}

// An nvlink describes a device's NVLink hardware. A device has NVLink
// if it reports a version or an active link.
type nvlink struct {
	version uint32 // marketing version, 0 if unknown
	links   uint32 // active links
}

func (n nvlink) present() bool { return n.version > 0 || n.links > 0 }

// resetDynamic clears the fields read at every refresh.
func (d *device) resetDynamic() {
	*d = device{
		index:        d.index,
		name:         d.name,
		arch:         d.arch,
		maxPcieGen:   d.maxPcieGen,
		maxPcieWidth: d.maxPcieWidth,
		tempSlowdown: d.tempSlowdown,
		tempShutdown: d.tempShutdown,
		ecc:          d.ecc,
		nvlink:       d.nvlink,
		procs:        d.procs,
	}
}

// A source reads devices. The NVML source is the real one;
// tests use fakes.
type source interface {
	// devices returns the devices with their static fields set.
	devices() []*device
	// refresh rereads the dynamic fields of d.
	refresh(d *device)
	// processes rereads the processes on d, with their GPU memory
	// and utilization.
	processes(d *device)
	close() error
}

// gather refreshes every device from src. When withProcs is set it also
// rereads the processes and fills in their host information from h.
// Either way it then derives the device figures the processes imply.
func gather(src source, h *host, devs []*device, withProcs bool) {
	for _, d := range devs {
		src.refresh(d)
		if withProcs {
			src.processes(d)
			if h != nil {
				h.fill(d)
			}
		}
		fixup(d)
	}
	if withProcs && h != nil {
		h.forget()
	}
}
