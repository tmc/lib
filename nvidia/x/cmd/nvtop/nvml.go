package main

import (
	"errors"
	"fmt"
	"time"

	"github.com/tmc/lib/nvidia/nvml"
)

// An nvmlSource reads devices through NVML.
type nvmlSource struct {
	host    *host // for the memory of unified-memory devices
	handles []nvml.Device
	devs    []*device

	// nvlinkLast is, per device, the NVLink byte counters at the
	// previous refresh, from which refresh derives a rate.
	nvlinkLast []nvlinkSample

	// lastUtil is, per device, the newest process utilization sample
	// seen, so that each refresh reads only newer samples.
	lastUtil []uint64

	procBuf []nvml.ProcessInfo
	utilBuf []nvml.ProcessUtilizationSample
}

// openNVML initializes NVML and reads the static fields of every device.
func openNVML(h *host) (*nvmlSource, error) {
	if err := nvml.Init(); err != nil {
		return nil, fmt.Errorf("initialize nvml: %w", err)
	}
	n, err := nvml.DeviceGetCount()
	if err != nil {
		nvml.Shutdown()
		return nil, fmt.Errorf("count devices: %w", err)
	}
	s := &nvmlSource{host: h, lastUtil: make([]uint64, n), nvlinkLast: make([]nvlinkSample, n)}
	for i := range n {
		hd, err := nvml.NewDeviceByIndex(i)
		if err != nil {
			nvml.Shutdown()
			return nil, fmt.Errorf("open device %d: %w", i, err)
		}
		d := &device{index: int(i)}
		d.name = get(hd.Name())
		if a, err := hd.Architecture(); err == nil {
			d.arch = archName(a)
		}
		d.maxPcieGen = get(hd.MaxPcieLinkGeneration())
		d.maxPcieWidth = get(hd.MaxPcieLinkWidth())
		d.tempShutdown = get(hd.TemperatureThreshold(nvml.TemperatureThresholdShutdown))
		d.tempSlowdown = get(hd.TemperatureThreshold(nvml.TemperatureThresholdSlowdown))
		// Consumer devices do not count ECC errors; nvtop then leaves
		// the ECC field out of the header.
		_, err = hd.TotalEccErrors(nvml.MemoryErrorTypeCorrected, nvml.VolatileEcc)
		d.ecc = err == nil
		d.nvlink = probeNVLink(hd)
		s.handles = append(s.handles, hd)
		s.devs = append(s.devs, d)
	}
	return s, nil
}

// get converts an NVML result to an opt. Any error, most often
// nvml.ErrNotSupported, means the device does not report the value.
func get[T any](v T, err error) opt[T] {
	if err != nil {
		return opt[T]{}
	}
	return some(v)
}

func (s *nvmlSource) devices() []*device { return s.devs }

func (s *nvmlSource) close() error { return nvml.Shutdown() }

// refresh follows gpuinfo_nvidia_refresh_dynamic_info.
func (s *nvmlSource) refresh(d *device) {
	hd := s.handles[d.index]
	d.resetDynamic()

	// The GPU clock is the larger of the graphics and SM clocks,
	// and its maximum is that of the same domain.
	gfx := get(hd.ClockInfo(nvml.ClockGraphics))
	sm := get(hd.ClockInfo(nvml.ClockSm))
	domain := nvml.ClockGraphics
	if sm.ok && (!gfx.ok || gfx.v < sm.v) {
		domain = nvml.ClockSm
		d.gpuClock = sm
	} else {
		d.gpuClock = gfx
	}
	d.gpuClockMax = get(hd.MaxClockInfo(domain))
	d.memClock = get(hd.ClockInfo(nvml.ClockMem))
	d.memClockMax = get(hd.MaxClockInfo(nvml.ClockMem))

	if u, err := hd.UtilizationRates(); err == nil {
		d.gpuUtil = some(u.Gpu)
	}
	var period uint32
	d.encUtil = get(hd.EncoderUtilization(&period))
	d.decUtil = get(hd.DecoderUtilization(&period))

	// A device without its own memory, such as GB10, reports
	// nvml.ErrNotSupported or a total of 0. It shares the host's memory,
	// so its figures are the host's: the total and available memory.
	m, err := hd.MemoryInfo()
	switch {
	case err == nil && m.Total > 0:
		d.memTotal = some(m.Total)
		d.memUsed = some(m.Used)
		d.memFree = some(m.Free)
		d.memUtil = some(uint32(m.Used * 100 / m.Total))
	case err == nil || errors.Is(err, nvml.ErrNotSupported):
		if s.host == nil {
			break
		}
		if total, avail, ok := s.host.systemMemory(); ok {
			d.memTotal = some(total)
			d.memFree = some(avail)
			d.memUsed = some(total - avail)
			d.memUtil = some(uint32((total - avail) * 100 / total))
		}
	}

	// A device not on a PCIe bus reports a placeholder link.
	// Drivers without nvmlDeviceGetBusType are taken to be on one.
	onPCIe := true
	if nvml.HasDeviceGetBusType() {
		if bus, err := hd.BusType(); err == nil {
			onPCIe = bus == nvml.BusTypePcie
		}
	}
	if onPCIe {
		d.pcieGen = get(hd.CurrPcieLinkGeneration())
		d.pcieWidth = get(hd.CurrPcieLinkWidth())
	}
	d.pcieRx = get(hd.PcieThroughput(nvml.PcieUtilRxBytes))
	d.pcieTx = get(hd.PcieThroughput(nvml.PcieUtilTxBytes))
	d.fan = get(hd.FanSpeed(0))
	d.temp = get(hd.Temperature(nvml.TemperatureGpu))
	d.power = get(hd.PowerUsage())
	d.powerMax = get(hd.EnforcedPowerLimit())
	if d.ecc {
		d.eccCorr = get(hd.TotalEccErrors(nvml.MemoryErrorTypeCorrected, nvml.VolatileEcc))
		d.eccUncorr = get(hd.TotalEccErrors(nvml.MemoryErrorTypeUncorrected, nvml.VolatileEcc))
	}
	var pending uint32
	if mode, err := hd.MigMode(&pending); err == nil {
		d.mig = some(mode == 1) // NVML_DEVICE_MIG_ENABLE
	}
	if d.nvlink.links > 0 {
		d.nvlinkRate = s.nvlinkLast[d.index].next(hd, time.Now())
	}
}

// archName follows the architecture names of nvtop's
// gpuinfo_nvidia_populate_static_info.
func archName(a nvml.DeviceArchitecture) opt[string] {
	names := map[nvml.DeviceArchitecture]string{
		nvml.DeviceArchKepler:    "Kepler",
		nvml.DeviceArchMaxwell:   "Maxwell",
		nvml.DeviceArchPascal:    "Pascal",
		nvml.DeviceArchVolta:     "Volta",
		nvml.DeviceArchTuring:    "Turing",
		nvml.DeviceArchAmpere:    "Ampere",
		nvml.DeviceArchAda:       "Ada",
		nvml.DeviceArchHopper:    "Hopper",
		nvml.DeviceArchBlackwell: "Blackwell",
		// Newer than package nvml's nvml.h.
		11: "DLA", 12: "DLA2", 13: "Rubin", 15: "NPU3",
	}
	name, ok := names[a]
	return opt[string]{name, ok}
}

// maxNVLinks is NVML_NVLINK_MAX_LINKS_INTERNAL.
const maxNVLinks = 36

// probeNVLink follows nvlink_probe_and_cache: the version of link 0,
// and the number of links active from link 0 on.
func probeNVLink(hd nvml.Device) nvlink {
	var n nvlink
	if v, err := hd.NvLinkVersion(0); err == nil {
		n.version = nvlinkVersion(v)
	}
	for link := uint32(0); link < maxNVLinks; link++ {
		st, err := hd.NvLinkState(link)
		if err != nil || st != nvml.FeatureEnabled {
			break
		}
		n.links = link + 1
	}
	return n
}

// nvlinkVersion follows nvlink_marketing_version: it rounds NVML's
// version enumeration to the major version sold.
func nvlinkVersion(raw uint32) uint32 {
	switch raw {
	case 3:
		return 2 // 2.2
	case 4, 5:
		return 3 // 3.0, 3.1
	case 6:
		return 4
	case 7:
		return 5
	case 8:
		return 6
	}
	return raw
}

// An nvlinkSample is a device's cumulative NVLink data counters, in KB,
// and when they were read.
type nvlinkSample struct {
	tx, rx uint64
	at     time.Time
}

// next follows the throughput part of nvlink_refresh_cached_info. It
// reads the counters summed over all links and returns the rate since
// the previous read, if there was one. A counter that went backwards
// counts as no traffic. As in nvtop, a failed read keeps the counters
// but not their time.
func (last *nvlinkSample) next(hd nvml.Device, now time.Time) opt[uint64] {
	all := ^uint32(0)
	v := []nvml.FieldValue{
		{FieldId: nvml.FieldDevNvlinkThroughputDataTx, ScopeId: all},
		{FieldId: nvml.FieldDevNvlinkThroughputDataRx, ScopeId: all},
	}
	err := hd.FieldValues(int32(len(v)), &v[0])
	txOK := err == nil && v[0].NvmlReturn == nvml.Success
	rxOK := err == nil && v[1].NvmlReturn == nvml.Success
	if !txOK && !rxOK {
		last.at = now
		return opt[uint64]{}
	}
	var cur nvlinkSample
	cur.at = now
	if txOK {
		cur.tx = v[0].Value.Uint64()
	}
	if rxOK {
		cur.rx = v[1].Value.Uint64()
	}
	rate := nvlinkRate(*last, cur)
	*last = cur
	return rate
}

// nvlinkRate returns the rate between samples a and b in KB/s.
func nvlinkRate(a, b nvlinkSample) opt[uint64] {
	dt := b.at.Sub(a.at).Seconds()
	if a.at.IsZero() || dt <= 0 {
		return opt[uint64]{}
	}
	delta := func(x, y uint64) uint64 {
		if y < x {
			return 0
		}
		return y - x
	}
	return some(uint64(float64(delta(a.tx, b.tx))/dt) + uint64(float64(delta(a.rx, b.rx))/dt))
}

// processes follows gpuinfo_nvidia_get_running_processes: the graphics
// processes, then the compute and MPS compute processes. Unlike nvtop,
// a process in more than one list is listed once.
func (s *nvmlSource) processes(d *device) {
	hd := s.handles[d.index]
	var procs []process
	index := make(map[int]int)
	add := func(infos []nvml.ProcessInfo, kind processKind) {
		for _, info := range infos {
			pid := int(info.Pid)
			if i, ok := index[pid]; ok {
				p := &procs[i]
				if p.kind != kind {
					p.kind = graphicCompute
				}
				p.gpuMem.v = max(p.gpuMem.v, info.UsedGpuMemory)
				continue
			}
			index[pid] = len(procs)
			procs = append(procs, process{pid: pid, kind: kind, gpuMem: some(info.UsedGpuMemory)})
		}
	}
	gathered := false
	var err error
	if s.procBuf, err = hd.GraphicsRunningProcesses(s.procBuf); err == nil {
		gathered = true
		add(s.procBuf, graphic)
	}
	if s.procBuf, err = hd.ComputeRunningProcesses(s.procBuf); err == nil {
		gathered = true
		add(s.procBuf, compute)
	}
	if s.procBuf, err = hd.MPSComputeRunningProcesses(s.procBuf); err == nil {
		gathered = true
		add(s.procBuf, compute)
	}
	if !gathered {
		procs = nil
	}
	d.procs = procs

	// NVML reports no per-process utilization on a MIG-enabled device.
	if !(d.mig.ok && d.mig.v) {
		s.utilization(d)
	}
	for i := range d.procs {
		p := &d.procs[i]
		if !p.gpuUtil.ok {
			p.gpuUtil = some(uint32(0))
		}
		if !p.encUtil.ok {
			p.encUtil = some(uint32(0))
		}
		if !p.decUtil.ok {
			p.decUtil = some(uint32(0))
		}
	}
}

// utilization follows gpuinfo_nvidia_get_process_utilization. It keeps
// only the samples newer than the last refresh that belong to a listed
// process and hold no rate above 100%, which NVML sometimes reports
// (nvtop issue #110).
func (s *nvmlSource) utilization(d *device) {
	if len(d.procs) == 0 {
		return
	}
	last := s.lastUtil[d.index]
	samples, err := s.handles[d.index].ProcessUtilization(last, s.utilBuf)
	if err != nil {
		return
	}
	s.utilBuf = samples
	newest := last
	for _, x := range samples {
		if x.SmUtil > 100 || x.EncUtil > 100 || x.DecUtil > 100 || x.TimeStamp <= last {
			continue
		}
		for i := range d.procs {
			p := &d.procs[i]
			if p.pid != int(x.Pid) {
				continue
			}
			newest = max(newest, x.TimeStamp)
			p.gpuUtil = some(x.SmUtil)
			p.encUtil = some(x.EncUtil)
			p.decUtil = some(x.DecUtil)
			break
		}
	}
	s.lastUtil[d.index] = newest
}
