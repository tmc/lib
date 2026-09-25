package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"time"
)

// A recording is a sequence of frames, one JSON object per line, each
// holding every device as one refresh left it and the events since the
// frame before.
type frame struct {
	Time    time.Time    `json:"time"`
	Devices []wireDevice `json:"devices"`
	Events  []wireEvent  `json:"events,omitempty"`
}

// A wireDevice is a device as recorded. Values a device does not report
// are null or left out.
type wireDevice struct {
	Name         opt[string]             `json:"name,omitzero"`
	Arch         opt[string]             `json:"arch,omitzero"`
	MaxPcieGen   opt[uint32]             `json:"max_pcie_gen,omitzero"`
	MaxPcieWidth opt[uint32]             `json:"max_pcie_width,omitzero"`
	TempSlowdown opt[uint32]             `json:"temp_slowdown,omitzero"`
	TempShutdown opt[uint32]             `json:"temp_shutdown,omitzero"`
	ECC          bool                    `json:"ecc,omitzero"`
	NVLinkVer    uint32                  `json:"nvlink_version,omitzero"`
	NVLinks      uint32                  `json:"nvlinks,omitzero"`
	Cores        opt[uint32]             `json:"cores,omitzero"`
	GPUClock     opt[uint32]             `json:"gpu_clock_mhz,omitzero"`
	GPUClockMax  opt[uint32]             `json:"gpu_clock_max_mhz,omitzero"`
	MemClock     opt[uint32]             `json:"mem_clock_mhz,omitzero"`
	MemClockMax  opt[uint32]             `json:"mem_clock_max_mhz,omitzero"`
	GPUUtil      opt[uint32]             `json:"gpu_util,omitzero"`
	EffLoad      opt[uint32]             `json:"eff_load,omitzero"`
	MemUtil      opt[uint32]             `json:"mem_util,omitzero"`
	EncUtil      opt[uint32]             `json:"enc_util,omitzero"`
	DecUtil      opt[uint32]             `json:"dec_util,omitzero"`
	MemTotal     opt[uint64]             `json:"mem_total,omitzero"`
	MemUsed      opt[uint64]             `json:"mem_used,omitzero"`
	MemFree      opt[uint64]             `json:"mem_free,omitzero"`
	PcieGen      opt[uint32]             `json:"pcie_gen,omitzero"`
	PcieWidth    opt[uint32]             `json:"pcie_width,omitzero"`
	PcieRx       opt[uint32]             `json:"pcie_rx_kbps,omitzero"`
	PcieTx       opt[uint32]             `json:"pcie_tx_kbps,omitzero"`
	Fan          opt[uint32]             `json:"fan,omitzero"`
	Temp         opt[uint32]             `json:"temp_c,omitzero"`
	Power        opt[uint32]             `json:"power_mw,omitzero"`
	PowerMax     opt[uint32]             `json:"power_max_mw,omitzero"`
	MIG          opt[bool]               `json:"mig,omitzero"`
	ECCCorr      opt[uint64]             `json:"ecc_corrected,omitzero"`
	ECCUncorr    opt[uint64]             `json:"ecc_uncorrected,omitzero"`
	NVLinkRate   opt[uint64]             `json:"nvlink_kbps,omitzero"`
	NVLinkErrs   opt[[3]uint64]          `json:"nvlink_errors,omitzero"` // flit CRC, data CRC, ECC
	Reasons      opt[uint64]             `json:"clock_event_reasons,omitzero"`
	Throttle     opt[[numReasons]uint32] `json:"throttle_pct,omitzero"`
	Energy       opt[uint64]             `json:"energy_mj,omitzero"`
	Procs        []wireProcess           `json:"processes,omitempty"`
}

type wireProcess struct {
	PID     int         `json:"pid"`
	Kind    processKind `json:"kind"`
	User    opt[string] `json:"user,omitzero"`
	Cmdline opt[string] `json:"cmdline,omitzero"`
	GPUMem  opt[uint64] `json:"gpu_mem,omitzero"`
	MemPct  opt[uint32] `json:"mem_pct,omitzero"`
	GPUUtil opt[uint32] `json:"gpu_util,omitzero"`
	EncUtil opt[uint32] `json:"enc_util,omitzero"`
	DecUtil opt[uint32] `json:"dec_util,omitzero"`
	CPUUtil opt[uint32] `json:"cpu_util,omitzero"`
	HostMem opt[uint64] `json:"host_mem,omitzero"`
}

type wireEvent struct {
	Time time.Time `json:"time"`
	Dev  int       `json:"dev"`
	Kind uint64    `json:"kind"`
	Data uint64    `json:"data,omitzero"`
}

func (o opt[T]) IsZero() bool { return !o.ok }

func (o opt[T]) MarshalJSON() ([]byte, error) {
	if !o.ok {
		return []byte("null"), nil
	}
	return json.Marshal(o.v)
}

func (o *opt[T]) UnmarshalJSON(b []byte) error {
	if string(b) == "null" {
		*o = opt[T]{}
		return nil
	}
	var v T
	if err := json.Unmarshal(b, &v); err != nil {
		return err
	}
	*o = some(v)
	return nil
}

func toWire(d *device) wireDevice {
	w := wireDevice{
		Name: d.name, Arch: d.arch,
		MaxPcieGen: d.maxPcieGen, MaxPcieWidth: d.maxPcieWidth,
		TempSlowdown: d.tempSlowdown, TempShutdown: d.tempShutdown,
		ECC: d.ecc, NVLinkVer: d.nvlink.version, NVLinks: d.nvlink.links,
		Cores:    d.cores,
		GPUClock: d.gpuClock, GPUClockMax: d.gpuClockMax,
		MemClock: d.memClock, MemClockMax: d.memClockMax,
		GPUUtil: d.gpuUtil, EffLoad: d.effLoad, MemUtil: d.memUtil,
		EncUtil: d.encUtil, DecUtil: d.decUtil,
		MemTotal: d.memTotal, MemUsed: d.memUsed, MemFree: d.memFree,
		PcieGen: d.pcieGen, PcieWidth: d.pcieWidth, PcieRx: d.pcieRx, PcieTx: d.pcieTx,
		Fan: d.fan, Temp: d.temp, Power: d.power, PowerMax: d.powerMax,
		MIG: d.mig, ECCCorr: d.eccCorr, ECCUncorr: d.eccUncorr,
		NVLinkRate: d.nvlinkRate,
		Reasons:    d.reasons, Throttle: d.throttle, Energy: d.energy,
	}
	if e := d.nvlinkErrs; e.ok {
		w.NVLinkErrs = some([3]uint64{e.v.flit, e.v.crc, e.v.ecc})
	}
	for _, p := range d.procs {
		w.Procs = append(w.Procs, wireProcess{
			PID: p.pid, Kind: p.kind, User: p.user, Cmdline: p.cmdline,
			GPUMem: p.gpuMem, MemPct: p.memPct,
			GPUUtil: p.gpuUtil, EncUtil: p.encUtil, DecUtil: p.decUtil,
			CPUUtil: p.cpuUtil, HostMem: p.hostMem,
		})
	}
	return w
}

// fromWire returns the device w records, with the given index.
func fromWire(w *wireDevice, index int) device {
	d := device{
		index: index,
		name:  w.Name, arch: w.Arch,
		maxPcieGen: w.MaxPcieGen, maxPcieWidth: w.MaxPcieWidth,
		tempSlowdown: w.TempSlowdown, tempShutdown: w.TempShutdown,
		ecc: w.ECC, nvlink: nvlink{version: w.NVLinkVer, links: w.NVLinks},
		cores:    w.Cores,
		gpuClock: w.GPUClock, gpuClockMax: w.GPUClockMax,
		memClock: w.MemClock, memClockMax: w.MemClockMax,
		gpuUtil: w.GPUUtil, effLoad: w.EffLoad, memUtil: w.MemUtil,
		encUtil: w.EncUtil, decUtil: w.DecUtil,
		memTotal: w.MemTotal, memUsed: w.MemUsed, memFree: w.MemFree,
		pcieGen: w.PcieGen, pcieWidth: w.PcieWidth, pcieRx: w.PcieRx, pcieTx: w.PcieTx,
		fan: w.Fan, temp: w.Temp, power: w.Power, powerMax: w.PowerMax,
		mig: w.MIG, eccCorr: w.ECCCorr, eccUncorr: w.ECCUncorr,
		nvlinkRate: w.NVLinkRate,
		reasons:    w.Reasons, throttle: w.Throttle, energy: w.Energy,
	}
	if e := w.NVLinkErrs; e.ok {
		d.nvlinkErrs = some(nvlinkErrors{flit: e.v[0], crc: e.v[1], ecc: e.v[2]})
	}
	for _, p := range w.Procs {
		d.procs = append(d.procs, process{
			pid: p.PID, kind: p.Kind, user: p.User, cmdline: p.Cmdline,
			gpuMem: p.GPUMem, memPct: p.MemPct,
			gpuUtil: p.GPUUtil, encUtil: p.EncUtil, decUtil: p.DecUtil,
			cpuUtil: p.CPUUtil, hostMem: p.HostMem,
		})
	}
	return d
}

// A recorder writes frames.
type recorder struct {
	w   *bufio.Writer
	enc *json.Encoder
}

func newRecorder(w io.Writer) *recorder {
	b := bufio.NewWriter(w)
	return &recorder{w: b, enc: json.NewEncoder(b)}
}

// write writes a frame of devs and evs taken at t, and flushes it so
// that a reader at the other end of a pipe sees it at once.
func (r *recorder) write(t time.Time, devs []*device, evs []event) error {
	f := frame{Time: t}
	for _, d := range devs {
		f.Devices = append(f.Devices, toWire(d))
	}
	for _, e := range evs {
		f.Events = append(f.Events, wireEvent{Time: e.at, Dev: e.dev, Kind: e.kind, Data: e.data})
	}
	if err := r.enc.Encode(&f); err != nil {
		return fmt.Errorf("record: %w", err)
	}
	if err := r.w.Flush(); err != nil {
		return fmt.Errorf("record: %w", err)
	}
	return nil
}

// record writes a frame to w every refresh interval until ctx is done,
// without the interface: nvtop -x -record -.
func record(ctx context.Context, w io.Writer, src source, h *host, opts options) error {
	delay := opts.delay
	if delay == 0 {
		delay = time.Second
	}
	rec := newRecorder(w)
	devs := src.devices()
	t := time.NewTicker(delay)
	defer t.Stop()
	for {
		gather(src, h, devs, true)
		var evs []event
		if es, ok := src.(eventSource); ok {
			evs = es.events()
		}
		if err := rec.write(time.Now(), devs, evs); err != nil {
			return err
		}
		select {
		case <-ctx.Done():
			return nil
		case <-t.C:
		}
	}
}

// A player is a source that replays a recording frame by frame.
type player interface {
	// step moves to the next frame and reports whether there was one.
	step() bool
	// at returns the time of the current frame.
	at() time.Time
	// gap returns the time between the current frame and the next,
	// or 0 at the end.
	gap() time.Duration
}

// A replaySource replays a recording.
type replaySource struct {
	file  io.Closer
	lines *bufio.Scanner
	devs  []*device
	cur   frame
	next  *frame // read ahead to know the gap; nil at the end
	fresh bool   // the current frame's events are not yet taken
	err   error
}

// openReplay opens the recording in the named file.
func openReplay(name string) (*replaySource, error) {
	f, err := os.Open(name)
	if err != nil {
		return nil, err
	}
	s, err := newReplay(f)
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("replay %s: %w", name, err)
	}
	s.file = f
	return s, nil
}

func newReplay(r io.Reader) (*replaySource, error) {
	s := &replaySource{lines: bufio.NewScanner(r)}
	s.lines.Buffer(nil, 64<<20)
	first, err := s.read()
	if err != nil {
		return nil, err
	}
	if first == nil {
		return nil, errors.New("no frames")
	}
	s.cur, s.fresh = *first, true
	for i := range first.Devices {
		d := fromWire(&first.Devices[i], i)
		s.devs = append(s.devs, &d)
	}
	s.next, s.err = s.read()
	return s, nil
}

// read reads the next frame, or returns nil at the end.
func (s *replaySource) read() (*frame, error) {
	for s.lines.Scan() {
		line := s.lines.Bytes()
		if len(line) == 0 {
			continue
		}
		var f frame
		if err := json.Unmarshal(line, &f); err != nil {
			return nil, err
		}
		return &f, nil
	}
	return nil, s.lines.Err()
}

func (s *replaySource) devices() []*device { return s.devs }

// refresh sets d as the current frame recorded it. A device the frame
// lacks reports nothing.
func (s *replaySource) refresh(d *device) {
	if d.index < len(s.cur.Devices) {
		*d = fromWire(&s.cur.Devices[d.index], d.index)
		return
	}
	d.resetDynamic()
	d.procs = nil
}

// processes does nothing: refresh sets the processes too.
func (s *replaySource) processes(d *device) {}

func (s *replaySource) events() []event {
	if !s.fresh {
		return nil
	}
	s.fresh = false
	var evs []event
	for _, e := range s.cur.Events {
		evs = append(evs, event{at: e.Time, dev: e.Dev, kind: e.Kind, data: e.Data})
	}
	return evs
}

func (s *replaySource) step() bool {
	if s.next == nil {
		return false
	}
	s.cur, s.fresh = *s.next, true
	s.next, s.err = s.read()
	return true
}

func (s *replaySource) at() time.Time { return s.cur.Time }

func (s *replaySource) gap() time.Duration {
	if s.next == nil {
		return 0
	}
	return s.next.Time.Sub(s.cur.Time)
}

func (s *replaySource) close() error {
	if s.file != nil {
		s.file.Close()
	}
	return s.err
}
