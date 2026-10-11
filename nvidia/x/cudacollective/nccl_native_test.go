package cudacollective

import (
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nccl"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudair"
	"os"
	"reflect"
	"testing"
)

// This explicit opt-in executes a one-rank native collective. It establishes
// binding, stream completion and guarded numerical behavior, not multi-GPU or
// network qualification. The hardware coordinator supplies a leased GPU UUID.
func TestNCCLNativeOneRank(t *testing.T) {
	uuid := os.Getenv("LIB_NCCL_NATIVE_UUID")
	if uuid == "" {
		t.Skip("requires an explicitly leased LIB_NCCL_NATIVE_UUID")
	}
	if err := cuda.Init(); err != nil {
		t.Fatal(err)
	}
	n, err := cuda.DeviceCount()
	if err != nil {
		t.Fatal(err)
	}
	ordinal := -1
	for i := 0; i < n; i++ {
		d, err := driver.DeviceGet(i)
		if err != nil {
			t.Fatal(err)
		}
		u, err := d.Uuid()
		if err != nil {
			t.Fatal(err)
		}
		id := fmt.Sprintf("GPU-%x-%x-%x-%x-%x", u.Bytes[:4], u.Bytes[4:6], u.Bytes[6:8], u.Bytes[8:10], u.Bytes[10:])
		if id == uuid {
			ordinal = i
		}
	}
	if ordinal < 0 {
		t.Fatalf("requested physical GPU UUID %q not visible", uuid)
	}
	dev, err := cuda.NewDevice(ordinal)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := dev.Close(); err != nil {
			t.Error(err)
		}
	}()
	err = dev.WithContext(func() error {
		id, err := nccl.GetUniqueId()
		if err != nil {
			return err
		}
		handle, err := nccl.CommInitRank(1, id, 0)
		if err != nil {
			return err
		}
		defer func() {
			if err := handle.Destroy(); err != nil {
				t.Error(err)
			}
		}()
		stream, err := driver.NewStream(0)
		if err != nil {
			return err
		}
		defer func() {
			if err := stream.Close(); err != nil {
				t.Error(err)
			}
		}()
		send, err := driver.MemAlloc[int32](6)
		if err != nil {
			return err
		}
		defer func() {
			if err := send.Free(); err != nil {
				t.Error(err)
			}
		}()
		recv, err := driver.MemAlloc[int32](6)
		if err != nil {
			return err
		}
		defer func() {
			if err := recv.Free(); err != nil {
				t.Error(err)
			}
		}()
		input := []int32{101, 2, 3, 5, 7, 103}
		output := []int32{107, 0, 0, 0, 0, 109}
		if err := send.CopyFromHost(input); err != nil {
			return err
		}
		if err := recv.CopyFromHost(output); err != nil {
			return err
		}
		m, err := NewMembership([]Rank{{ID: 0, DeviceID: ordinal}})
		if err != nil {
			return err
		}
		tr := &NCCLTransport{Comms: []nccl.Comm{handle}, Streams: []driver.Stream{stream}}
		c, err := NewCommunicator(CommunicatorConfig{Rank: 0, Membership: m, Generation: 1, Transport: tr})
		if err != nil {
			return err
		}
		defer c.Close()
		op := OperationDescriptor{Kind: KindAllReduce, DataType: cudair.Int32, Count: 4, Reduction: ReductionSum}
		if err := tr.Execute(context.Background(), c, op, send+4, recv+4); err != nil {
			return err
		}
		if err := recv.CopyToHost(output); err != nil {
			return err
		}
		if !reflect.DeepEqual(output, []int32{107, 2, 3, 5, 7, 109}) {
			return fmt.Errorf("numerical output or redzone changed: %v", output)
		}
		if err := send.CopyToHost(output); err != nil {
			return err
		}
		if !reflect.DeepEqual(output, input) {
			return fmt.Errorf("send buffer changed: %v", output)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}
