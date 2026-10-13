//go:build native && cuda_tma_multicast_qualification

package cudatma

import (
	"context"
	"encoding/hex"
	"errors"
	"fmt"
	"runtime"
	"sync"
	"time"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// qualificationOwner is test-only. No public callback can assert successful
// retirement. A native error/cancel retains the whole bundle until verified
// drain; no unconditional deferred Free/Close bypasses this state.
type qualificationOwner struct {
	admission      qualificationAdmission
	caseSpec       qualificationCase
	trace          func(string, any) error
	device         *cuda.Device
	stream         *cuda.Stream
	module         *cuda.Module
	rawModule      qualificationPending[driver.Module]
	input          [2]cudamem.Slice[byte]
	output, status cudamem.Slice[byte]
	slots          [2]*DescriptorSlot
	maps           [2]*driver.TensorMap
	handles        [2]TransferHandle
	mirrors        [2]Multicast
	releases       []func()
	pin            runtime.Pinner
	safe, borrowed bool
}

var qualificationRetained struct {
	sync.Mutex
	owners []*qualificationOwner
}

func (o *qualificationOwner) retainFailure() {
	qualificationRetained.Lock()
	qualificationRetained.owners = append(qualificationRetained.owners, o)
	qualificationRetained.Unlock()
}
func (o *qualificationOwner) log(operation string, value any) error { return o.trace(operation, value) }

func openQualificationOwner(a qualificationAdmission, c qualificationCase, cubin []byte, trace func(string, any) error) (*qualificationOwner, error) {
	o := &qualificationOwner{admission: a, caseSpec: c, trace: trace, safe: true}
	if e := o.log("driver-load-attempt", a.DriverFile); e != nil {
		return o, e
	}
	if e := driver.Load(a.DriverFile); e != nil {
		return o, e
	}
	if driver.Path() != a.DriverFile {
		return o, fmt.Errorf("actual driver path differs from admission")
	}
	version, e := driver.DriverGetVersion()
	if e != nil {
		return o, e
	}
	if version != a.DriverVersion {
		return o, fmt.Errorf("driver version%d differs from admitted%d", version, a.DriverVersion)
	}
	// Verify UUID/family on the raw device before retaining a primary context.
	if e = driver.Init(0); e != nil {
		return o, e
	}
	rawDevice, e := driver.DeviceGet(a.Ordinal)
	if e != nil {
		return o, e
	}
	uuid, e := rawDevice.Uuid()
	if e != nil {
		return o, e
	}
	major, e := rawDevice.Attribute(driver.DevAttrComputeCapabilityMajor)
	if e != nil {
		return o, e
	}
	minor, e := rawDevice.Attribute(driver.DevAttrComputeCapabilityMinor)
	if e != nil {
		return o, e
	}
	name, e := rawDevice.Name()
	if e != nil {
		return o, e
	}
	if e = o.log("actual-device-before-primary-context", struct {
		Name, UUID    string
		Major, Minor  int32
		DriverVersion int
	}{name, hex.EncodeToString(uuid.Bytes[:]), major, minor, version}); e != nil {
		return o, e
	}
	if hex.EncodeToString(uuid.Bytes[:]) != a.DeviceUUID || major != 10 || minor != 0 {
		return o, fmt.Errorf("actual device UUID/family differs from admitted sm_100a device")
	}
	dev, e := cuda.NewDevice(a.Ordinal)
	o.device = dev
	if e != nil {
		return o, e
	}
	if dev.Raw() != rawDevice {
		return o, fmt.Errorf("owned device differs from identity checked raw handle")
	}
	cluster, e := dev.Attribute(driver.DevAttrClusterLaunch)
	if e != nil {
		return o, e
	}
	if cluster != 1 {
		return o, fmt.Errorf("actual device refuses cluster launch")
	}
	stream, e := dev.NewStream()
	o.stream = stream
	if e != nil {
		return o, e
	}
	var raw driver.Module
	if e = o.rawModule.load(dev.WithContext, func() (driver.Module, error) {
		return driver.ModuleLoadData(cubin)
	}); e != nil {
		return o, fmt.Errorf("module load scope failed; partial owner retained: %w", e)
	}
	raw = o.rawModule.value
	mod, e := cuda.OwnModule(dev, raw)
	if e != nil {
		// Raw module ownership remains here on failure. No GPU work is queued;
		// unload is checked under its context before returning.
		cleanup := o.rawModule.closeThen(func(raw driver.Module) error { return dev.WithContext(raw.Close) }, func() error { return nil })
		return o, errors.Join(e, cleanup)
	}
	o.rawModule.held = false // OwnModule now holds the loaded module.
	o.module = mod
	d := c.config().Descriptor
	input, e := qualificationInput(d, 0)
	if e != nil {
		return o, e
	}
	p, e := NewMulticastPlan(c.config())
	if e != nil {
		return o, e
	}
	r, _ := p.Requirements()
	phases := 1
	if c.Parity {
		phases = 2
	}
	total := uint64(len(input)*phases) + uint64(phases)*r.OutputBytes + uint64(phases)*uint64(r.CTAs)*32 + 256
	if total > a.AllocationBytes {
		return o, fmt.Errorf("fixture allocations exceed admitted byte budget")
	}
	for i := 0; i < phases; i++ {
		o.input[i], e = cudamem.Alloc[byte](dev, len(input))
		if e != nil {
			return o, e
		}
	}
	o.output, e = cudamem.Alloc[byte](dev, phases*int(r.OutputBytes)+128)
	if e != nil {
		return o, e
	}
	o.status, e = cudamem.Alloc[byte](dev, phases*int(r.CTAs)*32+128)
	if e != nil {
		return o, e
	}
	if e = o.log("allocations", struct {
		Input0, Input1, Output, Status uint64
		Bytes                          uint64
	}{uint64(o.input[0].Ptr()), uint64(o.input[1].Ptr()), uint64(o.output.Ptr()), uint64(o.status.Ptr()), total}); e != nil {
		return o, e
	}
	spans := []struct{ ptr, size uint64 }{{uint64(o.input[0].Ptr()), uint64(o.input[0].SizeBytes())}, {uint64(o.output.Ptr()), uint64(o.output.SizeBytes())}, {uint64(o.status.Ptr()), uint64(o.status.SizeBytes())}}
	if phases == 2 {
		spans = append(spans, struct{ ptr, size uint64 }{uint64(o.input[1].Ptr()), uint64(o.input[1].SizeBytes())})
	}
	for i, x := range spans {
		if x.ptr == 0 || x.ptr > ^uint64(0)-x.size {
			return o, fmt.Errorf("invalid actual allocation span")
		}
		for j := 0; j < i; j++ {
			y := spans[j]
			if x.ptr < y.ptr+y.size && y.ptr < x.ptr+x.size {
				return o, fmt.Errorf("actual native allocations overlap")
			}
		}
	}
	return o, nil
}

func (o *qualificationOwner) kernel(name string) (*cuda.Kernel, error) {
	fn, e := o.module.Function(name)
	if e != nil {
		return nil, e
	}
	attrs := []driver.FunctionAttribute{driver.FuncAttributeRequiredClusterWidth, driver.FuncAttributeRequiredClusterHeight, driver.FuncAttributeRequiredClusterDepth}
	for i, attr := range attrs {
		v, e := fn.Attribute(attr)
		if e != nil {
			return nil, e
		}
		want := []int{2, 1, 1}[i]
		if v != want {
			return nil, fmt.Errorf("loaded kernel required cluster axis%d=%d want%d", i, v, want)
		}
	}
	for _, attr := range []driver.DeviceAttribute{driver.DevAttrMaxThreadsPerBlock, driver.DevAttrMaxBlockDimX} {
		v, e := o.device.Attribute(attr)
		if e != nil {
			return nil, e
		}
		if v < 128 {
			return nil, fmt.Errorf("actual device block limit%d below requested128", v)
		}
	}
	v, e := fn.Attribute(driver.FuncAttributeMaxThreadsPerBlock)
	if e != nil {
		return nil, e
	}
	if v < 128 {
		return nil, fmt.Errorf("actual kernel block limit%d below requested128", v)
	}
	p, e := NewMulticastPlan(o.caseSpec.config())
	if e != nil {
		return nil, e
	}
	r, _ := p.Requirements()
	launch := cudalaunch.OwnedKernel(fn).Grid(2, 1, 1).Block(128, 1, 1).SharedMem(r.SharedBytes).On(o.stream)
	if e = launch.PrepareSharedMem(); e != nil {
		return nil, e
	}
	var max, active int32
	e = o.stream.WithHandle(func(stream driver.Stream) error {
		cfg := driver.LaunchConfig{GridDim: driver.Dims(2, 1, 1), BlockDim: driver.Dims(128, 1, 1), SharedMemBytes: r.SharedBytes, HStream: stream, Attrs: []driver.LaunchAttribute{cudalaunch.ClusterDimAttr(driver.Dims(2, 1, 1))}}
		return o.device.WithContext(func() error {
			var e error
			max, e = fn.Raw().MaxPotentialClusterSize(&cfg)
			if e != nil {
				return e
			}
			active, e = fn.Raw().MaxActiveClusters(&cfg)
			return e
		})
	})
	if e != nil {
		return nil, e
	}
	if max < 2 || active <= 0 {
		return nil, fmt.Errorf("actual queried cluster occupancy refuses requested geometry: max%d active%d", max, active)
	}
	if e = o.log("kernel-resource-admission", struct {
		Name               string
		Shared             uint32
		MaxCluster, Active int32
	}{name, r.SharedBytes, max, active}); e != nil {
		return nil, e
	}
	return fn, nil
}

func (o *qualificationOwner) releaseBorrows() {
	for _, release := range o.releases {
		release()
	}
	o.releases = nil
	o.pin.Unpin()
	o.borrowed = false
}
func (o *qualificationOwner) borrow(phases int) error {
	allocations := []cudamem.Slice[byte]{o.input[0], o.output, o.status}
	if phases == 2 {
		allocations = append(allocations, o.input[1])
	}
	for _, allocation := range allocations {
		release, e := allocation.Borrow(o.device)
		if e != nil {
			o.releaseBorrows()
			return e
		}
		o.releases = append(o.releases, release)
	}
	o.borrowed = true
	return nil
}
func (o *qualificationOwner) drain() error {
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(o.admission.DrainSeconds)*time.Second)
	defer cancel()
	e := o.stream.Synchronize(ctx)
	if e == nil {
		o.safe = true
	}
	if log := o.log("actual-drain", struct {
		Error              string
		PhysicalCompletion bool
	}{errorText(e), e == nil}); log != nil {
		return errors.Join(e, log)
	}
	return e
}
func errorText(e error) string {
	if e == nil {
		return ""
	}
	return e.Error()
}

type qualificationReadback struct {
	Kernel                                                                                  string
	Coordinates                                                                             [MaxRank]int32
	Generations                                                                             []uint64
	Input, InputAfter                                                                       [][]byte
	BeforeOutput, ActualOutput, BeforeStatus, ActualStatus, Expected                        []byte
	LaunchReturned, Drained, OwnershipRefusals, CancelledWaitRetained, SuccessfulRetirement bool
	Comparison                                                                              string
}

func (o *qualificationOwner) run(name string, epoch uint32, coords [MaxRank]int32, parity bool) (qualificationReadback, error) {
	result := qualificationReadback{Kernel: name, Coordinates: coords}
	phases := 1
	if parity {
		phases = 2
	}
	c := o.caseSpec
	p, e := NewMulticastPlan(c.config())
	if e != nil {
		return result, e
	}
	r, _ := p.Requirements()
	fn, e := o.kernel(name)
	if e != nil {
		return result, e
	}
	if e = o.borrow(phases); e != nil {
		return result, e
	}
	// Input/output mutation is synchronous and is forbidden while prior work
	// remains unverified. This owner serializes every run and drain.
	for phase := 0; phase < phases; phase++ {
		input, e := qualificationInput(c.config().Descriptor, epoch+uint32(phase))
		if e != nil {
			return result, e
		}
		result.Input = append(result.Input, input)
		if e = o.input[phase].CopyFromHost(input); e != nil {
			o.safe = false
			return result, e
		}
		d := c.config().Descriptor
		d.Base = uint64(o.input[phase].Ptr())
		if d.Base%128 != 0 {
			return result, fmt.Errorf("actual tensor base is not128B aligned")
		}
		if o.slots[phase] == nil {
			o.slots[phase], e = NewDescriptorSlot(d)
			if e != nil {
				return result, e
			}
			if phase == 1 {
				if e = o.slots[phase].Update(d); e != nil {
					return result, e
				}
			}
		} else {
			if e = o.slots[phase].Update(d); e != nil {
				return result, e
			}
		}
		snapshot, g := o.slots[phase].Current()
		o.handles[phase], e = o.slots[phase].Issue(g, Transfer{Descriptor: snapshot, Direction: GlobalToShared, Bytes: uint64(r.TileBytes)})
		if e != nil {
			return result, e
		}
		result.Generations = append(result.Generations, g)
		encoded := snapshot
		if c.Control == "wrong-swizzle" {
			encoded.Swizzle = 0
		} // Declared negative only; same valid opaque driver API.
		if e = o.log("tensor-map-encode-attempt", encoded); e != nil {
			return result, e
		}
		e = o.device.WithContext(func() error { var err error; o.maps[phase], err = encoded.EncodeTiled(); return err })
		if e != nil {
			return result, e
		}
		if o.maps[phase] == nil || uintptr(unsafe.Pointer(o.maps[phase]))%128 != 0 || unsafe.Sizeof(*o.maps[phase]) != 128 {
			return result, fmt.Errorf("actual opaque tensor map storage/ABI invalid")
		}
		o.pin.Pin(o.maps[phase])
		o.mirrors[phase], e = NewMulticast(c.config().SourceRank, c.config().Destinations, uint64(r.TileBytes))
		if e != nil {
			return result, e
		}
		want, e := qualificationOracle(snapshot, coords, input)
		if e != nil {
			return result, e
		}
		result.Expected = append(result.Expected, want...)
	}
	result.BeforeOutput = bytesOf(0xa5, o.output.Len())
	result.BeforeStatus = bytesOf(0x6b, o.status.Len())
	if e = o.output.CopyFromHost(result.BeforeOutput); e != nil {
		o.safe = false
		return result, e
	}
	if e = o.status.CopyFromHost(result.BeforeStatus); e != nil {
		o.safe = false
		return result, e
	}
	output := o.output.Slice(64, 64+phases*int(r.OutputBytes))
	status := o.status.Slice(64, 64+phases*int(r.CTAs)*32)
	args := []any{o.maps[0]}
	if parity {
		args = append(args, o.maps[1])
	}
	args = append(args, output, status)
	for _, generation := range result.Generations {
		args = append(args, generation)
	}
	for i := 0; i < c.Rank; i++ {
		args = append(args, coords[i])
	}
	packed, e := cudalaunch.PackArgs(args...)
	if e != nil {
		return result, e
	}
	for phase := 0; phase < phases; phase++ {
		if packed[phase] != unsafe.Pointer(o.maps[phase]) {
			return result, fmt.Errorf("tensor-map parameter was copied or packed as pointer value")
		}
	}
	if e = o.log("owned-launch-attempt", struct {
		Kernel      string
		Generations []uint64
		Coordinates [5]int32
	}{name, result.Generations, coords}); e != nil {
		return result, e
	}
	o.safe = false
	// Compile-time required cluster dimensions are matched above. Do not set
	// their FuncAttribute values at runtime or silently enable nonportable sizes.
	e = cudalaunch.OwnedKernel(fn).Grid(2, 1, 1).Block(128, 1, 1).SharedMem(r.SharedBytes).On(o.stream).Launch(args...)
	runtime.KeepAlive(packed)
	runtime.KeepAlive(o.maps)
	result.LaunchReturned = e == nil
	if log := o.log("owned-launch-result", errorText(e)); log != nil {
		return result, errors.Join(e, log)
	}
	if e != nil {
		return result, errors.Join(e, o.drain())
	}
	// These checks prove retained ownership before verified drain/readback;
	// they do not assert the GPU is still physically busy at this instant.
	for phase := 0; phase < phases; phase++ {
		next := o.slots[phase].Descriptor()
		next.Base += 128
		if !errors.Is(o.slots[phase].Update(next), ErrInFlight) {
			return result, fmt.Errorf("in-flight descriptor replacement escaped")
		}
		if !errors.Is(o.input[phase].Slice(0, 1).Free(), cudamem.ErrInUse) {
			return result, fmt.Errorf("in-flight allocation alias free escaped")
		}
	}
	if !errors.Is(output.Free(), cudamem.ErrInUse) {
		return result, fmt.Errorf("output alias free escaped")
	}
	result.OwnershipRefusals = true
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	e = o.stream.Synchronize(cancelled)
	if !errors.Is(e, context.Canceled) {
		return result, fmt.Errorf("cancelled wait did not return cancellation: %w", e)
	}
	if !errors.Is(o.input[0].Free(), cudamem.ErrInUse) || o.slots[0].InFlight() != 1 {
		return result, fmt.Errorf("cancelled wait released allocation/descriptor ownership")
	}
	result.CancelledWaitRetained = true
	if e = o.log("cancelled-wait-retained", true); e != nil {
		return result, e
	}
	if e = o.drain(); e != nil {
		return result, e
	}
	result.Drained = true
	result.ActualOutput = make([]byte, o.output.Len())
	result.ActualStatus = make([]byte, o.status.Len())
	if e = o.output.CopyToHost(result.ActualOutput); e != nil {
		return result, e
	}
	if e = o.status.CopyToHost(result.ActualStatus); e != nil {
		return result, e
	}
	for phase := 0; phase < phases; phase++ {
		after := make([]byte, o.input[phase].Len())
		if e = o.input[phase].CopyToHost(after); e != nil {
			return result, e
		}
		result.InputAfter = append(result.InputAfter, after)
		if !bytesEqual(after, result.Input[phase]) {
			return result, fmt.Errorf("native source/padding mutated")
		}
	}
	for _, guard := range []struct {
		before, after []byte
		active        int
	}{{result.BeforeOutput, result.ActualOutput, phases * int(r.OutputBytes)}, {result.BeforeStatus, result.ActualStatus, phases * int(r.CTAs) * 32}} {
		if !bytesEqual(guard.before[:64], guard.after[:64]) || !bytesEqual(guard.before[64+guard.active:], guard.after[64+guard.active:]) {
			return result, fmt.Errorf("native allocation guard/inactive reservation changed")
		}
	}
	checked := c
	checked.Parity = parity
	comparison := qualificationCompare(checked, result.ActualOutput[64:64+phases*int(r.OutputBytes)], result.ActualStatus[64:64+phases*int(r.CTAs)*32], result.Expected, result.Generations)
	result.Comparison = errorText(comparison)
	if comparison != nil {
		return result, nil
	} // Caller can recognize a declared completed negative; no success retirement.
	for phase := 0; phase < phases; phase++ {
		for _, rank := range c.config().Destinations {
			if e = o.mirrors[phase].Complete(rank, uint64(r.TileBytes)); e != nil {
				return result, e
			}
		}
		if !o.mirrors[phase].CompleteAll() {
			return result, fmt.Errorf("missing actual destination completion")
		}
		if e = o.slots[phase].Complete(o.handles[phase]); e != nil {
			return result, e
		}
	}
	result.SuccessfulRetirement = true
	o.releaseBorrows()
	return result, nil
}

func (o *qualificationOwner) close() error {
	if o == nil {
		return nil
	}
	if !o.safe {
		o.retainFailure()
		return fmt.Errorf("native completion unknown; owner retained, external quarantine required")
	}
	e := o.rawModule.closeThen(func(raw driver.Module) error {
		return o.device.WithContext(raw.Close)
	}, o.closeOwned)
	if e != nil && (o.rawModule.held || o.rawModule.scopeUncertain) {
		o.retainFailure()
	}
	return e
}

func (o *qualificationOwner) closeOwned() error {
	o.releaseBorrows()
	// Checked cleanup only after physical completion; failure preserves the
	// remaining objects. Module/stream Close are not timeout recovery APIs.
	if o.stream != nil {
		if e := o.stream.Close(); e != nil {
			return e
		}
		o.stream = nil
	}
	if o.module != nil {
		if e := o.module.Close(); e != nil {
			return e
		}
		o.module = nil
	}
	for i := range o.input {
		if e := o.input[i].Free(); e != nil {
			return e
		}
	}
	if e := o.output.Free(); e != nil {
		return e
	}
	if e := o.status.Free(); e != nil {
		return e
	}
	if o.device != nil {
		if e := o.device.Close(); e != nil {
			return e
		}
		o.device = nil
	}
	return o.log("checked-cleanup-complete", true)
}
