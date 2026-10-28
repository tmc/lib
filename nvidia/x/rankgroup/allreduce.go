package rankgroup

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nccl"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// An AllReducer sums float32 buffers across a [Group]'s ranks.
type AllReducer interface {
	// AllReduce enqueues, on each rank r's stream, recv[r][i] = the sum
	// over ranks q of send[q][i] for i < n. Each recv[r] lies on rank r's
	// device and overlaps no send buffer. Work enqueued on a rank's stream
	// after AllReduce sees the sum. It must run inside [Group.Do].
	AllReduce(send, recv []driver.DevicePtr, n int) error
	Close() error
}

// deviceAllReduceSource is the one-shot all-reduce kernel. Every rank sums
// all ranks' inputs in rank order, so all ranks compute the same bits.
const deviceAllReduceSource = `
struct rankgroup_ptrs { const float* p[8]; };

extern "C" __global__ void rankgroup_allreduce_f32(rankgroup_ptrs in, float* __restrict__ out, int ranks, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float s = in.p[0][i];
        for (int r = 1; r < ranks; ++r) s += in.p[r][i];
        out[i] = s;
    }
}
`

// DeviceAllReduce is an [AllReducer] whose kernel reads every rank's
// input directly: a one-shot all-reduce over one address space.
type DeviceAllReduce struct {
	g    *Group
	mods map[*cuda.Device]driver.Module
	fns  []driver.Function // by rank
}

// NewDeviceAllReduce compiles the all-reduce kernel for g's devices. Every
// rank's buffers must be addressable from every rank: g must be on one
// device or on devices with peer access enabled. It must run inside
// [Group.Do].
func NewDeviceAllReduce(g *Group) (_ *DeviceAllReduce, err error) {
	d := &DeviceAllReduce{g: g, mods: make(map[*cuda.Device]driver.Module)}
	defer func() {
		if err != nil {
			err = errors.Join(err, d.Close())
		}
	}()
	for r := range g.Size() {
		dev := g.Device(r)
		if _, ok := d.mods[dev]; !ok {
			if err := g.use(r); err != nil {
				return nil, err
			}
			cm, err := nvrtcx.Compile(deviceAllReduceSource, nvrtcx.WithName("rankgroup_allreduce.cu"), nvrtcx.WithDeviceArch(dev))
			if err != nil {
				return nil, fmt.Errorf("rankgroup: compile all-reduce: %w", err)
			}
			mod, err := cm.Load(dev)
			if err != nil {
				return nil, fmt.Errorf("rankgroup: load all-reduce: %w", err)
			}
			d.mods[dev] = mod
		}
		fn, err := d.mods[dev].Function("rankgroup_allreduce_f32")
		if err != nil {
			return nil, fmt.Errorf("rankgroup: all-reduce kernel: %w", err)
		}
		d.fns = append(d.fns, fn)
	}
	return d, nil
}

// AllReduce implements [AllReducer]. It is a barrier, one kernel per rank,
// and a barrier: the second keeps a rank from overwriting its input while
// a peer still reads it.
func (d *DeviceAllReduce) AllReduce(send, recv []driver.DevicePtr, n int) error {
	g := d.g
	if len(send) != g.Size() || len(recv) != g.Size() || n < 0 {
		return fmt.Errorf("rankgroup: all-reduce of %d sends and %d receives over %d ranks", len(send), len(recv), g.Size())
	}
	if n == 0 {
		return nil
	}
	var in Pointers
	copy(in[:], send)
	if err := g.Barrier(); err != nil {
		return err
	}
	const threads = 256
	blocks := uint32(min((n+threads-1)/threads, 4*g.SMs(0)))
	for r := range g.Size() {
		if err := g.Launch(r, d.fns[r], [3]uint32{blocks, 1, 1}, [3]uint32{threads, 1, 1}, 0, in, recv[r], int32(g.Size()), int32(n)); err != nil {
			return err
		}
	}
	return g.Barrier()
}

// Close unloads the kernel modules.
func (d *DeviceAllReduce) Close() error {
	var errs []error
	for dev, m := range d.mods {
		errs = append(errs, dev.WithContext(m.Close))
		delete(d.mods, dev)
	}
	return errors.Join(errs...)
}

// NCCLAllReduce is an [AllReducer] backed by NCCL, one communicator per
// rank.
type NCCLAllReduce struct {
	g     *Group
	comms []nccl.Comm
}

// NewNCCLAllReduce creates NCCL communicators for g's ranks with
// ncclCommInitAll. NCCL needs one distinct GPU per rank: g must come from
// [OnDevices], or have one rank. It must run inside [Group.Do].
func NewNCCLAllReduce(g *Group) (*NCCLAllReduce, error) {
	seen := make(map[int]bool)
	var devs []int32
	for r := range g.Size() {
		o := g.Device(r).Ordinal()
		if seen[o] {
			return nil, fmt.Errorf("rankgroup: nccl needs one gpu per rank; ranks share device %d", o)
		}
		seen[o] = true
		devs = append(devs, int32(o))
	}
	comms := make([]nccl.Comm, g.Size())
	if err := nccl.CommInitAll(&comms[0], int32(len(comms)), &devs[0]); err != nil {
		return nil, fmt.Errorf("rankgroup: nccl init: %w", err)
	}
	return &NCCLAllReduce{g: g, comms: comms}, nil
}

// AllReduce implements [AllReducer] with one ncclAllReduce per rank in an
// NCCL group.
func (a *NCCLAllReduce) AllReduce(send, recv []driver.DevicePtr, n int) error {
	g := a.g
	if len(send) != g.Size() || len(recv) != g.Size() || n < 0 {
		return fmt.Errorf("rankgroup: all-reduce of %d sends and %d receives over %d ranks", len(send), len(recv), g.Size())
	}
	if err := nccl.GroupStart(); err != nil {
		return fmt.Errorf("rankgroup: nccl group: %w", err)
	}
	var errs []error
	for r, c := range a.comms {
		if err := c.AllReduce(send[r], recv[r], uint(n), nccl.Float32, nccl.Sum, g.Stream(r)); err != nil {
			errs = append(errs, fmt.Errorf("rankgroup: rank %d nccl all-reduce: %w", r, err))
		}
	}
	if err := nccl.GroupEnd(); err != nil {
		errs = append(errs, fmt.Errorf("rankgroup: nccl group: %w", err))
	}
	return errors.Join(errs...)
}

// Close destroys the communicators.
func (a *NCCLAllReduce) Close() error {
	var errs []error
	for _, c := range a.comms {
		errs = append(errs, c.Destroy())
	}
	a.comms = nil
	return errors.Join(errs...)
}
