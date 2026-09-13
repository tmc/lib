package cudagraph_test

import (
	"encoding/binary"
	"strings"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

const updateKernelSource = `
extern "C" __global__ void addOffset(unsigned int *p, int n, unsigned int offset) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		p[i] += offset;
	}
}
`

func TestDeviceUpdatableAttr(t *testing.T) {
	tests := []struct {
		name     string
		enable   bool
		wantVal  uint64
		wantAttr driver.LaunchAttributeID
	}{
		{
			name:     "enabled",
			enable:   true,
			wantVal:  1,
			wantAttr: driver.LaunchAttributeDeviceUpdatableKernelNode,
		},
		{
			name:     "disabled",
			enable:   false,
			wantVal:  0,
			wantAttr: driver.LaunchAttributeDeviceUpdatableKernelNode,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			attr := cudalaunch.DeviceUpdatableAttr(tt.enable)
			if attr.Id != tt.wantAttr {
				t.Errorf("attr.Id = %v, want %v", attr.Id, tt.wantAttr)
			}
			if attr.Value[0] != tt.wantVal {
				t.Errorf("attr.Value[0] = %d, want %d", attr.Value[0], tt.wantVal)
			}
		})
	}
}

func TestLauncherWithDeviceUpdatable(t *testing.T) {
	l := cudalaunch.Kernel(0).WithDeviceUpdatable(true)
	attrs := l.Attrs()
	if len(attrs) != 1 {
		t.Fatalf("len(attrs) = %d, want 1", len(attrs))
	}
	if attrs[0].Id != driver.LaunchAttributeDeviceUpdatableKernelNode {
		t.Errorf("attrs[0].Id = %v, want %v", attrs[0].Id, driver.LaunchAttributeDeviceUpdatableKernelNode)
	}
	if attrs[0].Value[0] != 1 {
		t.Errorf("attrs[0].Value[0] = %d, want 1", attrs[0].Value[0])
	}

	// DeviceNode before launch or when zero
	if _, ok := l.DeviceNode(); ok {
		t.Error("DeviceNode() returned true when handle was 0")
	}

	// Fake driver writing devNode handle into attribute Value[1]
	l = cudalaunch.Kernel(0).WithAttrs(driver.LaunchAttribute{
		Id:    driver.LaunchAttributeDeviceUpdatableKernelNode,
		Value: [8]uint64{1, 0x12345678},
	})
	node, ok := l.DeviceNode()
	if !ok {
		t.Fatal("DeviceNode() returned false, want true")
	}
	if node != 0x12345678 {
		t.Errorf("DeviceNode() = %#x, want 0x12345678", node)
	}
}

func TestKernelNodeFieldString(t *testing.T) {
	tests := []struct {
		field cudagraph.KernelNodeField
		want  string
	}{
		{cudagraph.KernelNodeFieldInvalid, "KernelNodeFieldInvalid"},
		{cudagraph.KernelNodeFieldGridDim, "KernelNodeFieldGridDim"},
		{cudagraph.KernelNodeFieldParam, "KernelNodeFieldParam"},
		{cudagraph.KernelNodeFieldEnabled, "KernelNodeFieldEnabled"},
		{cudagraph.KernelNodeField(99), "KernelNodeField(99)"},
	}

	for _, tt := range tests {
		t.Run(tt.want, func(t *testing.T) {
			if got := tt.field.String(); got != tt.want {
				t.Errorf("String() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestDeviceNodeMethods(t *testing.T) {
	dn := cudagraph.DeviceNode(0xdeadbeef)
	if dn.Uint64() != 0xdeadbeef {
		t.Errorf("Uint64() = %#x, want 0xdeadbeef", dn.Uint64())
	}
	if !strings.Contains(dn.String(), "deadbeef") {
		t.Errorf("String() = %q, want containing deadbeef", dn.String())
	}
}

func TestDeviceKernelNodeUpdateLayout(t *testing.T) {
	// CUDA C ABI specifies cudaGraphKernelNodeUpdate is 40 bytes:
	// 4 bytes field + 4 bytes pad + 8 bytes node + 24 bytes updateData union
	const wantSize = 40
	gotSize := unsafe.Sizeof(cudagraph.DeviceKernelNodeUpdate{})
	if gotSize != wantSize {
		t.Fatalf("sizeof(DeviceKernelNodeUpdate) = %d, want %d", gotSize, wantSize)
	}

	var d cudagraph.DeviceKernelNodeUpdate
	if offset := unsafe.Offsetof(d.Field); offset != 0 {
		t.Errorf("Field offset = %d, want 0", offset)
	}
	if offset := unsafe.Offsetof(d.Node); offset != 8 {
		t.Errorf("Node offset = %d, want 8", offset)
	}
	if offset := unsafe.Offsetof(d.UpdateData); offset != 16 {
		t.Errorf("UpdateData offset = %d, want 16", offset)
	}
}

func TestKernelNodeUpdateEncoding(t *testing.T) {
	devNode := cudagraph.DeviceNode(0x4000)

	t.Run("grid dim", func(t *testing.T) {
		u := cudagraph.NewGridDimUpdate(devNode, cudalaunch.Dims(4, 2, 1))
		dev := u.DeviceUpdate()

		if dev.Field != uint32(cudagraph.KernelNodeFieldGridDim) {
			t.Errorf("dev.Field = %d, want %d", dev.Field, cudagraph.KernelNodeFieldGridDim)
		}
		if dev.Node != uint64(devNode) {
			t.Errorf("dev.Node = %#x, want %#x", dev.Node, devNode)
		}

		gx := binary.LittleEndian.Uint32(dev.UpdateData[0:4])
		gy := binary.LittleEndian.Uint32(dev.UpdateData[4:8])
		gz := binary.LittleEndian.Uint32(dev.UpdateData[8:12])
		if gx != 4 || gy != 2 || gz != 1 {
			t.Errorf("grid = (%d,%d,%d), want (4,2,1)", gx, gy, gz)
		}

		b := dev.Bytes()
		if len(b) != 40 {
			t.Errorf("len(dev.Bytes()) = %d, want 40", len(b))
		}
	})

	t.Run("param", func(t *testing.T) {
		val := uint32(42)
		u := cudagraph.NewParamUpdate(devNode, 16, unsafe.Pointer(&val), 4)
		dev := u.DeviceUpdate()

		if dev.Field != uint32(cudagraph.KernelNodeFieldParam) {
			t.Errorf("dev.Field = %d, want %d", dev.Field, cudagraph.KernelNodeFieldParam)
		}
		offset := binary.LittleEndian.Uint64(dev.UpdateData[0:8])
		ptr := binary.LittleEndian.Uint64(dev.UpdateData[8:16])
		size := binary.LittleEndian.Uint64(dev.UpdateData[16:24])

		if offset != 16 {
			t.Errorf("offset = %d, want 16", offset)
		}
		if ptr != uint64(uintptr(unsafe.Pointer(&val))) {
			t.Errorf("ptr = %#x, want %#x", ptr, uintptr(unsafe.Pointer(&val)))
		}
		if size != 4 {
			t.Errorf("size = %d, want 4", size)
		}
	})

	t.Run("enabled", func(t *testing.T) {
		uOn := cudagraph.NewEnabledUpdate(devNode, true)
		devOn := uOn.DeviceUpdate()
		if devOn.Field != uint32(cudagraph.KernelNodeFieldEnabled) {
			t.Errorf("devOn.Field = %d, want %d", devOn.Field, cudagraph.KernelNodeFieldEnabled)
		}
		if val := binary.LittleEndian.Uint32(devOn.UpdateData[0:4]); val != 1 {
			t.Errorf("enabled = %d, want 1", val)
		}

		uOff := cudagraph.NewEnabledUpdate(devNode, false)
		devOff := uOff.DeviceUpdate()
		if val := binary.LittleEndian.Uint32(devOff.UpdateData[0:4]); val != 0 {
			t.Errorf("enabled = %d, want 0", val)
		}
	})

	t.Run("batch encoding", func(t *testing.T) {
		updates := []cudagraph.KernelNodeUpdate{
			cudagraph.NewGridDimUpdate(devNode, cudalaunch.Dims(2)),
			cudagraph.NewEnabledUpdate(devNode, true),
		}
		encoded := cudagraph.EncodeDeviceUpdates(updates)
		if len(encoded) != 2 {
			t.Fatalf("len(encoded) = %d, want 2", len(encoded))
		}
		if encoded[0].Field != uint32(cudagraph.KernelNodeFieldGridDim) {
			t.Errorf("encoded[0].Field = %d, want GridDim", encoded[0].Field)
		}
		if encoded[1].Field != uint32(cudagraph.KernelNodeFieldEnabled) {
			t.Errorf("encoded[1].Field = %d, want Enabled", encoded[1].Field)
		}

		if empty := cudagraph.EncodeDeviceUpdates(nil); empty != nil {
			t.Errorf("EncodeDeviceUpdates(nil) = %v, want nil", empty)
		}
	})
}

func TestDeviceUpdateErrorHandling(t *testing.T) {
	tests := []struct {
		name string
		fn   func() error
	}{
		{
			name: "nil node SetDeviceUpdatable",
			fn: func() error {
				var n *cudagraph.Node
				return n.SetDeviceUpdatable(true)
			},
		},
		{
			name: "closed node SetDeviceUpdatable",
			fn: func() error {
				n := &cudagraph.Node{}
				return n.SetDeviceUpdatable(true)
			},
		},
		{
			name: "nil node IsDeviceUpdatable",
			fn: func() error {
				var n *cudagraph.Node
				_, err := n.IsDeviceUpdatable()
				return err
			},
		},
		{
			name: "closed node IsDeviceUpdatable",
			fn: func() error {
				n := &cudagraph.Node{}
				_, err := n.IsDeviceUpdatable()
				return err
			},
		},
		{
			name: "nil node DeviceNode",
			fn: func() error {
				var n *cudagraph.Node
				_, err := n.DeviceNode()
				return err
			},
		},
		{
			name: "closed node DeviceNode",
			fn: func() error {
				n := &cudagraph.Node{}
				_, err := n.DeviceNode()
				return err
			},
		},
		{
			name: "nil node Params",
			fn: func() error {
				var n *cudagraph.Node
				_, err := n.Params()
				return err
			},
		},
		{
			name: "closed node Params",
			fn: func() error {
				n := &cudagraph.Node{}
				_, err := n.Params()
				return err
			},
		},
		{
			name: "nil node SetParams",
			fn: func() error {
				var n *cudagraph.Node
				return n.SetParams(&driver.KernelNodeParams{})
			},
		},
		{
			name: "closed node SetParams",
			fn: func() error {
				n := &cudagraph.Node{}
				return n.SetParams(&driver.KernelNodeParams{})
			},
		},
		{
			name: "nil node SetGrid",
			fn: func() error {
				var n *cudagraph.Node
				return n.SetGrid(cudalaunch.Dims(1))
			},
		},
		{
			name: "closed node SetGrid",
			fn: func() error {
				n := &cudagraph.Node{}
				return n.SetGrid(cudalaunch.Dims(1))
			},
		},
		{
			name: "node SetGrid zero dim",
			fn: func() error {
				n := &cudagraph.Node{}
				return n.SetGrid(cudalaunch.Dim3{X: 0, Y: 1, Z: 1})
			},
		},
		{
			name: "nil graph AddDeviceUpdatableKernelNode",
			fn: func() error {
				var g *cudagraph.Graph
				_, err := g.AddDeviceUpdatableKernelNode(&driver.KernelNodeParams{})
				return err
			},
		},
		{
			name: "nil graph Nodes",
			fn: func() error {
				var g *cudagraph.Graph
				_, err := g.Nodes()
				return err
			},
		},
		{
			name: "nil exec SetKernelNodeParams",
			fn: func() error {
				var e *cudagraph.Exec
				return e.SetKernelNodeParams(&cudagraph.Node{}, &driver.KernelNodeParams{})
			},
		},
		{
			name: "closed exec SetKernelNodeParams",
			fn: func() error {
				e := &cudagraph.Exec{}
				return e.SetKernelNodeParams(&cudagraph.Node{}, &driver.KernelNodeParams{})
			},
		},
		{
			name: "nil exec SetKernelNodeGrid",
			fn: func() error {
				var e *cudagraph.Exec
				return e.SetKernelNodeGrid(&cudagraph.Node{}, cudalaunch.Dims(1))
			},
		},
		{
			name: "closed exec SetKernelNodeGrid",
			fn: func() error {
				e := &cudagraph.Exec{}
				return e.SetKernelNodeGrid(&cudagraph.Node{}, cudalaunch.Dims(1))
			},
		},
		{
			name: "nil exec UpdateKernelNode",
			fn: func() error {
				var e *cudagraph.Exec
				return e.UpdateKernelNode(&cudagraph.Node{}, cudalaunch.Dims(1))
			},
		},
		{
			name: "closed exec UpdateKernelNode",
			fn: func() error {
				e := &cudagraph.Exec{}
				return e.UpdateKernelNode(&cudagraph.Node{}, cudalaunch.Dims(1))
			},
		},
		{
			name: "nil exec ApplyKernelNodeUpdate",
			fn: func() error {
				var e *cudagraph.Exec
				return e.ApplyKernelNodeUpdate(&cudagraph.Node{}, cudagraph.NewGridDimUpdate(0, cudalaunch.Dims(1)))
			},
		},
		{
			name: "closed exec ApplyKernelNodeUpdate",
			fn: func() error {
				e := &cudagraph.Exec{}
				return e.ApplyKernelNodeUpdate(&cudagraph.Node{}, cudagraph.NewGridDimUpdate(0, cudalaunch.Dims(1)))
			},
		},
		{
			name: "exec ApplyKernelNodeUpdate unsupported field",
			fn: func() error {
				e := &cudagraph.Exec{}
				return e.ApplyKernelNodeUpdate(&cudagraph.Node{}, cudagraph.KernelNodeUpdate{Field: cudagraph.KernelNodeField(99)})
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.fn(); err == nil {
				t.Errorf("%s: expected error, got nil", tt.name)
			}
		})
	}
}

func TestDeviceUpdatableNodeIntegration(t *testing.T) {
	dev := cudatest.Require(t)

	compiled, err := nvrtcx.Compile(updateKernelSource, nvrtcx.WithDeviceArch(dev), nvrtcx.WithNoCache())
	if err != nil {
		t.Fatal(err)
	}
	mod, err := compiled.Load(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(mod.Unload)

	fn, err := mod.GetFunction("addOffset")
	if err != nil {
		t.Fatal(err)
	}

	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(stream.Destroy)

	const (
		n     = 256
		block = 128
	)

	buf, err := cudamem.Alloc[uint32](dev, n)
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(buf.Free)

	t.Run("created node device updatable", func(t *testing.T) {
		g, err := cudagraph.New(dev)
		if err != nil {
			t.Fatal(err)
		}
		defer g.Destroy()

		dptr := buf.Ptr()
		count := int32(n)
		offsetVal := uint32(5)
		kernelArgs := []unsafe.Pointer{
			unsafe.Pointer(&dptr),
			unsafe.Pointer(&count),
			unsafe.Pointer(&offsetVal),
		}

		kp := driver.KernelNodeParams{
			Func:         fn,
			GridDimX:     n / block,
			GridDimY:     1,
			GridDimZ:     1,
			BlockDimX:    block,
			BlockDimY:    1,
			BlockDimZ:    1,
			KernelParams: uintptr(unsafe.Pointer(&kernelArgs[0])),
		}

		node, err := g.AddDeviceUpdatableKernelNode(&kp)
		if err != nil {
			t.Fatalf("AddDeviceUpdatableKernelNode: %v", err)
		}

		isUpdatable, err := node.IsDeviceUpdatable()
		if err != nil {
			t.Fatalf("IsDeviceUpdatable: %v", err)
		}
		if !isUpdatable {
			t.Error("IsDeviceUpdatable = false, want true")
		}

		nodes, err := g.Nodes()
		if err != nil {
			t.Fatalf("g.Nodes(): %v", err)
		}
		if len(nodes) != 1 {
			t.Fatalf("len(g.Nodes()) = %d, want 1", len(nodes))
		}

		exec, err := g.Instantiate()
		if err != nil {
			t.Fatalf("Instantiate: %v", err)
		}
		defer exec.Destroy()

		devNode, err := node.DeviceNode()
		if err != nil {
			t.Logf("DeviceNode() query: %v", err)
		} else {
			t.Logf("DeviceNode handle: %#x", devNode)
		}

		// Initialize buffer to 0
		zeros := make([]uint32, n)
		if err := buf.CopyFromHost(zeros); err != nil {
			t.Fatal(err)
		}

		// Launch graph: buffer values should become 5
		if err := exec.Launch(stream); err != nil {
			t.Fatal(err)
		}
		if err := stream.Synchronize(nil); err != nil {
			t.Fatal(err)
		}

		got := make([]uint32, n)
		if err := buf.CopyToHost(got); err != nil {
			t.Fatal(err)
		}
		for i, v := range got {
			if v != 5 {
				t.Fatalf("before update: buf[%d] = %d, want 5", i, v)
			}
		}

		// Update kernel node arguments without whole-graph Update: change offset to 20
		if err := exec.UpdateKernelNode(node, cudalaunch.Dims(n/block), buf, int32(n), uint32(20)); err != nil {
			t.Fatalf("UpdateKernelNode: %v", err)
		}

		// Launch again: values should become 5 + 20 = 25
		if err := exec.Launch(stream); err != nil {
			t.Fatal(err)
		}
		if err := stream.Synchronize(nil); err != nil {
			t.Fatal(err)
		}

		if err := buf.CopyToHost(got); err != nil {
			t.Fatal(err)
		}
		for i, v := range got {
			if v != 25 {
				t.Fatalf("after UpdateKernelNode: buf[%d] = %d, want 25", i, v)
			}
		}
	})

	t.Run("captured node device updatable", func(t *testing.T) {
		launcher := cudalaunch.Kernel(fn).
			Grid(n / block).
			Block(block).
			WithDeviceUpdatable(true).
			On(stream)

		g, err := cudagraph.Capture(stream, func() {
			_ = launcher.Launch(buf, int32(n), uint32(10))
		})
		if err != nil {
			t.Fatalf("Capture: %v", err)
		}
		defer g.Destroy()

		nodes, err := g.Nodes()
		if err != nil {
			t.Fatalf("g.Nodes(): %v", err)
		}
		if len(nodes) != 1 {
			t.Fatalf("captured graph nodes = %d, want 1", len(nodes))
		}

		isUpdatable, err := nodes[0].IsDeviceUpdatable()
		if err != nil {
			t.Fatalf("IsDeviceUpdatable on captured node: %v", err)
		}
		if !isUpdatable {
			t.Error("IsDeviceUpdatable on captured node = false, want true")
		}

		exec, err := g.Instantiate()
		if err != nil {
			t.Fatalf("Instantiate captured graph: %v", err)
		}
		defer exec.Destroy()

		// Test SetKernelNodeGrid
		if err := exec.SetKernelNodeGrid(nodes[0], cudalaunch.Dims(n/block)); err != nil {
			t.Fatalf("SetKernelNodeGrid: %v", err)
		}

		zeros := make([]uint32, n)
		if err := buf.CopyFromHost(zeros); err != nil {
			t.Fatal(err)
		}

		if err := exec.Launch(stream); err != nil {
			t.Fatal(err)
		}
		if err := stream.Synchronize(nil); err != nil {
			t.Fatal(err)
		}

		got := make([]uint32, n)
		if err := buf.CopyToHost(got); err != nil {
			t.Fatal(err)
		}
		for i, v := range got {
			if v != 10 {
				t.Fatalf("captured node launch: buf[%d] = %d, want 10", i, v)
			}
		}
	})

	t.Run("param update for func", func(t *testing.T) {
		offsetVal := uint32(100)
		update, err := cudagraph.NewParamUpdateForFunc(0x1000, fn, 2, unsafe.Pointer(&offsetVal))
		if err != nil {
			t.Logf("NewParamUpdateForFunc: %v (GetParamInfo may not be supported by driver)", err)
		} else {
			if update.Field != cudagraph.KernelNodeFieldParam {
				t.Errorf("update.Field = %v, want Param", update.Field)
			}
			if update.Size == 0 {
				t.Error("update.Size is 0, want > 0")
			}
		}
	})
}
