package rankgroup

import (
	"fmt"
	"os"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

func nativeDevice(t *testing.T) *cuda.Device {
	t.Helper()
	if os.Getenv("CUDA_RANKGROUP_NATIVE") != "1" {
		t.Skip("set CUDA_RANKGROUP_NATIVE=1 to run ranks on device 0")
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Fatalf("native gate open without inherited gpu lock: %v", err)
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { dev.Close() })
	return dev
}

// checkAllReduce sums n floats over g's ranks with a and checks every
// rank's result, twice, so the second call reuses buffers the first read.
func checkAllReduce(t *testing.T, g *Group, a AllReducer, n int) {
	t.Helper()
	ranks := g.Size()
	var send, recv []cudamem.Slice[float32]
	defer func() {
		for _, s := range append(send, recv...) {
			s.Free()
		}
	}()
	for r := range ranks {
		s, err := cudamem.Alloc[float32](g.Device(r), n)
		if err != nil {
			t.Fatal(err)
		}
		send = append(send, s)
		d, err := cudamem.Alloc[float32](g.Device(r), n)
		if err != nil {
			t.Fatal(err)
		}
		recv = append(recv, d)
	}
	for iter := range 2 {
		var sp, rp []driver.DevicePtr
		for r := range ranks {
			v := make([]float32, n)
			for i := range v {
				v[i] = float32((r+1)*(iter+1)) + float32(i%7)*0.25
			}
			if err := send[r].CopyFromHost(v); err != nil {
				t.Fatal(err)
			}
			if err := recv[r].CopyFromHost(make([]float32, n)); err != nil {
				t.Fatal(err)
			}
			sp, rp = append(sp, send[r].Ptr()), append(rp, recv[r].Ptr())
		}
		err := g.Do(func() error {
			if err := a.AllReduce(sp, rp, n); err != nil {
				return err
			}
			return g.Synchronize()
		})
		if err != nil {
			t.Fatal(err)
		}
		for r := range ranks {
			got := make([]float32, n)
			if err := recv[r].CopyToHost(got); err != nil {
				t.Fatal(err)
			}
			for i, v := range got {
				var want float32
				for q := range ranks {
					want += float32((q+1)*(iter+1)) + float32(i%7)*0.25
				}
				if v != want {
					t.Fatalf("iteration %d rank %d: recv[%d] = %g, want %g", iter, r, i, v, want)
				}
			}
		}
	}
}

func TestNativeDeviceAllReduce(t *testing.T) {
	dev := nativeDevice(t)
	for _, mode := range []Mode{Serial, Streams, Green} {
		for _, n := range []int{1, 2, 4} {
			t.Run(fmt.Sprintf("%v/tp%d", mode, n), func(t *testing.T) {
				g, err := OnDevice(dev, n, mode)
				if err != nil {
					t.Fatal(err)
				}
				defer func() {
					if err := g.Close(); err != nil {
						t.Error(err)
					}
				}()
				var sms []int
				for r := range n {
					sms = append(sms, g.SMs(r))
				}
				t.Logf("SMs per rank %v", sms)
				var a *DeviceAllReduce
				if err := g.Do(func() (err error) { a, err = NewDeviceAllReduce(g); return err }); err != nil {
					t.Fatal(err)
				}
				defer a.Close()
				checkAllReduce(t, g, a, 4096+3)
			})
		}
	}
}

// TestNativeNCCLAllReduce runs NCCL on as many ranks as there are GPUs,
// up to 4. NCCL refuses two ranks on one GPU, so a single-GPU machine runs
// one rank and skips the rest.
func TestNativeNCCLAllReduce(t *testing.T) {
	dev := nativeDevice(t)
	count, err := cuda.DeviceCount()
	if err != nil {
		t.Fatal(err)
	}
	for _, n := range []int{1, 2, 4} {
		t.Run(fmt.Sprintf("tp%d", n), func(t *testing.T) {
			if n > count {
				t.Skipf("nccl needs %d gpus, %d visible: needs a multi-gpu node", n, count)
			}
			devs := []*cuda.Device{dev}
			for i := 1; i < n; i++ {
				d, err := cuda.NewDevice(i)
				if err != nil {
					t.Fatal(err)
				}
				defer d.Close()
				devs = append(devs, d)
			}
			g, err := OnDevices(devs)
			if err != nil {
				t.Fatal(err)
			}
			defer g.Close()
			var a *NCCLAllReduce
			if err := g.Do(func() (err error) { a, err = NewNCCLAllReduce(g); return err }); err != nil {
				t.Fatal(err)
			}
			defer a.Close()
			checkAllReduce(t, g, a, 4096+3)
		})
	}
	t.Run("shared-device", func(t *testing.T) {
		g, err := OnDevice(dev, 2, Streams)
		if err != nil {
			t.Fatal(err)
		}
		defer g.Close()
		if err := g.Do(func() error { _, err := NewNCCLAllReduce(g); return err }); err == nil {
			t.Fatal("nccl accepted two ranks on one gpu")
		}
	})
}
