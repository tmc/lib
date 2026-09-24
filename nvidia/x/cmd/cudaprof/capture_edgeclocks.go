package main

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
	"github.com/tmc/lib/nvidia/nvml"
)

// A phase is timed on the device at whatever clocks the GPU ran, and two
// arms at different SM clocks differ by that ratio before any code does. So
// each capture-range edge carries the GPU's SM and memory clocks, temperature
// and power as NVML reported them.
//
// The controller reads them, not the agent. The agent writes each edge from
// inside the target's cudaProfilerStart and cudaProfilerStop, and an NVML
// call there would add its cost to the range the edge bounds. Instead the
// controller follows the target's journal while the target runs and reads
// NVML as soon as an edge record appears. The reading therefore comes after
// the edge, never at it, and LagNanos says how long after, on the
// CLOCK_MONOTONIC the agent stamped the edge with. A start edge is written as
// the range opens, so its reading lies inside the range. A stop edge is
// stamped before the agent's flush and written after it, so a stop reading
// lags by at least the edge's FlushNanos.
//
// LagNanos bounds when the controller asked, not how old the answer is. The
// driver refreshes these values on its own schedule: on a GB10 nvidia-smi at
// 200 ms intervals showed power holding one value for 400-600 ms, and a
// start edge read 208 MHz, the idle clock, some 600 ms before the clock
// was seen at 2522 MHz under load. A reading is the driver's most recent
// value at SampledAt; a range much shorter than the refresh interval can
// have both its edges read the same one.

// gpuEdgeSample is the GPU state read for one capture-range edge.
type gpuEdgeSample struct {
	// SampledAt is CLOCK_MONOTONIC in the controller when the reads began,
	// LagNanos is SampledAt minus the edge's timestamp, and ReadNanos is how
	// long the reads of every device took.
	SampledAt uint64       `json:"sampled_at,omitempty"`
	LagNanos  int64        `json:"lag_nanos,omitempty"`
	ReadNanos uint64       `json:"read_nanos,omitempty"`
	Devices   []gpuReading `json:"devices,omitempty"`

	// Unavailable says why there is no reading. The fields above are then
	// absent, never zeros standing in for one.
	Unavailable string `json:"unavailable,omitempty"`
}

// gpuReading is one device's state. Index is NVML's enumeration order,
// which need not be CUDA's device ordinal; UUID names the device. A reading
// NVML refused is absent, and Errors says why.
type gpuReading struct {
	Index           uint32            `json:"index"`
	UUID            string            `json:"uuid,omitempty"`
	Name            string            `json:"name,omitempty"`
	SMClockMHz      *uint32           `json:"sm_clock_mhz,omitempty"`
	MemClockMHz     *uint32           `json:"mem_clock_mhz,omitempty"`
	TemperatureC    *uint32           `json:"temperature_c,omitempty"`
	PowerMilliwatts *uint32           `json:"power_mw,omitempty"`
	Errors          map[string]string `json:"errors,omitempty"`
}

// A gpuOpener opens the GPUs for reading. read returns every device's
// current state; closeGPU releases them.
type gpuOpener func() (read func() []gpuReading, closeGPU func(), err error)

// openNVML reads the GPUs through NVML.
func openNVML() (func() []gpuReading, func(), error) {
	if err := nvml.Init(); err != nil {
		return nil, nil, fmt.Errorf("nvml init: %w", err)
	}
	n, err := nvml.DeviceGetCount()
	if err != nil {
		nvml.Shutdown()
		return nil, nil, fmt.Errorf("nvml device count: %w", err)
	}
	type device struct {
		dev nvml.Device
		ok  bool
		id  gpuReading
	}
	devices := make([]device, 0, n)
	for i := range n {
		d := device{id: gpuReading{Index: i}}
		dev, err := nvml.NewDeviceByIndex(i)
		if err != nil {
			d.id.Errors = map[string]string{"device": err.Error()}
			devices = append(devices, d)
			continue
		}
		d.dev, d.ok = dev, true
		if d.id.UUID, err = dev.UUID(); err != nil {
			d.id.Errors = map[string]string{"uuid": err.Error()}
		}
		d.id.Name, _ = dev.Name()
		devices = append(devices, d)
	}
	read := func() []gpuReading {
		out := make([]gpuReading, 0, len(devices))
		for _, d := range devices {
			r := d.id
			r.Errors = nil
			for k, v := range d.id.Errors {
				setReadingError(&r, k, errors.New(v))
			}
			if !d.ok {
				out = append(out, r)
				continue
			}
			r.SMClockMHz = reading(&r, "sm_clock_mhz", func() (uint32, error) { return d.dev.ClockInfo(nvml.ClockSm) })
			r.MemClockMHz = reading(&r, "mem_clock_mhz", func() (uint32, error) { return d.dev.ClockInfo(nvml.ClockMem) })
			r.TemperatureC = reading(&r, "temperature_c", func() (uint32, error) { return d.dev.Temperature(nvml.TemperatureGpu) })
			r.PowerMilliwatts = reading(&r, "power_mw", d.dev.PowerUsage)
			out = append(out, r)
		}
		return out
	}
	return read, func() { nvml.Shutdown() }, nil
}

// reading returns get's value, or records its error in r and returns nil.
func reading(r *gpuReading, field string, get func() (uint32, error)) *uint32 {
	v, err := get()
	if err != nil {
		setReadingError(r, field, err)
		return nil
	}
	return &v
}

func setReadingError(r *gpuReading, field string, err error) {
	if r.Errors == nil {
		r.Errors = make(map[string]string)
	}
	r.Errors[field] = err.Error()
}

// edgeKey names an edge record: sequences restart in each process
// generation, so the timestamp tells two generations' edges apart.
type edgeKey struct{ sequence, timestamp uint64 }

// edgeClocks samples the GPU at each capture-range edge the target writes.
type edgeClocks struct {
	read        func() []gpuReading
	closeGPU    func()
	unavailable string
	watching    bool
	stop        chan struct{}
	done        chan map[edgeKey]*gpuEdgeSample
}

// openEdgeClocks opens the GPU before the target starts, so the cost of
// opening it is not paid while the target runs. A GPU that cannot be opened
// is recorded as the reason every edge goes unread.
func openEdgeClocks(open gpuOpener) *edgeClocks {
	if open == nil {
		open = openNVML
	}
	c := &edgeClocks{stop: make(chan struct{}), done: make(chan map[edgeKey]*gpuEdgeSample, 1)}
	read, closeGPU, err := open()
	if err != nil {
		c.unavailable = err.Error()
		return c
	}
	c.read, c.closeGPU = read, closeGPU
	// NVML's first reads cost milliseconds -- 4.6 ms on a GB10, against
	// microseconds after -- and that cost would land on the first edge.
	read()
	return c
}

// watch follows the journals of process pid in the capture session
// directory dir until finish is called.
func (c *edgeClocks) watch(dir string, pid int) {
	c.watching = true
	go c.run(filepath.Join(dir, fmt.Sprintf("process-%d-*", pid), "capture.journal"))
}

// finish stops following, samples any edge written since the last poll,
// and returns every sample by edge. It releases the GPU.
func (c *edgeClocks) finish() map[edgeKey]*gpuEdgeSample {
	var samples map[edgeKey]*gpuEdgeSample
	if c.watching {
		close(c.stop)
		samples = <-c.done
	}
	if c.closeGPU != nil {
		c.closeGPU()
	}
	return samples
}

// edgePollInterval bounds how long an edge waits in the journal before it
// is seen, and so most of the sample's lag.
const edgePollInterval = time.Millisecond

func (c *edgeClocks) run(pattern string) {
	samples := make(map[edgeKey]*gpuEdgeSample)
	journals := make(map[string]*journalFollower)
	defer func() {
		for _, j := range journals {
			j.f.Close()
		}
	}()
	tick := time.NewTicker(edgePollInterval)
	defer tick.Stop()
	for {
		stopping := false
		select {
		case <-c.stop:
			stopping = true
		case <-tick.C:
		}
		if paths, err := filepath.Glob(pattern); err == nil {
			for _, path := range paths {
				if journals[path] != nil {
					continue
				}
				if f, err := os.Open(path); err == nil {
					journals[path] = &journalFollower{f: f}
				}
			}
		}
		for _, j := range journals {
			for _, r := range j.edges() {
				k := edgeKey{r.Sequence, r.Timestamp}
				if samples[k] == nil {
					samples[k] = c.sample(r.Timestamp)
				}
			}
		}
		if stopping {
			c.done <- samples
			return
		}
	}
}

func (c *edgeClocks) sample(edge uint64) *gpuEdgeSample {
	if c.unavailable != "" {
		return &gpuEdgeSample{Unavailable: c.unavailable}
	}
	t0, ok := monotonicNanos()
	if !ok {
		return &gpuEdgeSample{Unavailable: "no CLOCK_MONOTONIC to place the reading against the edge"}
	}
	devices := c.read()
	t1, _ := monotonicNanos()
	return &gpuEdgeSample{SampledAt: t0, LagNanos: int64(t0 - edge), ReadNanos: t1 - t0, Devices: devices}
}

// journalFollower reads a journal that is still being written, returning
// each capture-range edge as its frame completes.
type journalFollower struct {
	f    *os.File
	buf  []byte
	dead bool
}

var captureRangeSubtype = []byte(`"subtype":"capture_range"`)

func (j *journalFollower) edges() []wire.Record {
	if j.dead {
		return nil
	}
	var chunk [64 << 10]byte
	for {
		n, err := j.f.Read(chunk[:])
		j.buf = append(j.buf, chunk[:n]...)
		if err != nil || n == 0 {
			if err != nil && err != io.EOF {
				j.dead = true
			}
			break
		}
	}
	var edges []wire.Record
	b := j.buf
	for len(b) >= wire.HeaderSize {
		n := binary.BigEndian.Uint32(b[8:12])
		if n > wire.DefaultMaxPayload {
			// A frame this decoder would refuse ends what can be followed;
			// the journal is judged after the target exits, not here.
			j.dead = true
			break
		}
		size := wire.HeaderSize + int(n)
		if len(b) < size {
			break
		}
		frame := b[:size]
		b = b[size:]
		if binary.BigEndian.Uint16(frame[6:8]) != recordActivity || !bytes.Contains(frame[wire.HeaderSize:], captureRangeSubtype) {
			continue
		}
		if r, err := wire.NewDecoder(bytes.NewReader(frame), wire.Limits{}).Next(); err == nil {
			edges = append(edges, r)
		}
	}
	j.buf = append(j.buf[:0], b...)
	return edges
}

// attachEdgeClocks gives each edge in r its sample. An edge the controller
// never saw while following the journal says so.
func attachEdgeClocks(r *capturePhaseReceipt, samples map[edgeKey]*gpuEdgeSample) {
	if r == nil {
		return
	}
	for i := range r.Edges {
		e := &r.Edges[i]
		e.GPU = samples[edgeKey{e.Sequence, e.Timestamp}]
		if e.GPU == nil {
			e.GPU = &gpuEdgeSample{Unavailable: "edge not seen in the journal while the capture ran"}
		}
	}
}
