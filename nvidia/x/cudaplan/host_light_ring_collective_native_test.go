//go:build native

package cudaplan_test

import (
	"context"
	"errors"
	"fmt"
	"os"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nccl"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudacollective"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudalock"
)

// TestHostLightNativeRingNCCL is a two-device NCCL fixture driven through the
// host ring consumer. Each rank's command computes a distinct Int32 payload
// with the finite CPU program, uploads it, and runs an NCCL all-reduce through
// cudacollective.NCCLTransport; the result is read back and compared with
// direct arithmetic. It is not a CPU reduction, GIN, direct NIC, NVLink or
// GPU-resident ingress claim, and nothing here measures timing.
//
// It builds only with -tags native. The inherited GPU lock is checked first,
// before any vendor or device call; without it the test skips. Native calls can block; the admitted caller
// retains the external lease and process watchdog. After a native failure,
// contexts, streams, allocations and communicators are left to the process
// rather than freed under possibly in-flight work.
func TestHostLightNativeRingNCCL(t *testing.T) {
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Skipf("NOT EXECUTED: native NCCL fixture requires inherited gpu-lock: %v", err)
	}
	if os.Getenv("LIB_HOST_LIGHT_NATIVE") != "1" {
		t.Skip("NOT EXECUTED: requires LIB_HOST_LIGHT_NATIVE=1 and two leased GPU UUIDs")
	}
	uuids, err := hostLightNativeAdmission(os.Getenv)
	if err != nil {
		t.Fatal(err)
	}

	var devs [2]*cuda.Device
	var streams [2]driver.Stream
	var send, recv [2]driver.DevicePtr
	var handles []nccl.Comm
	quarantine := false
	fail := func(err error) {
		t.Helper()
		quarantine = true
		t.Fatal(err)
	}
	defer func() {
		if quarantine {
			t.Log("native failure: device resources retained, not freed; no retry")
			return
		}
		for rank := range devs {
			if devs[rank] == nil {
				continue
			}
			err := devs[rank].WithContext(func() error {
				var errs []error
				for _, p := range []driver.DevicePtr{recv[rank], send[rank]} {
					if p != 0 {
						errs = append(errs, p.Free())
					}
				}
				if streams[rank] != 0 {
					errs = append(errs, streams[rank].Close())
				}
				if rank < len(handles) && handles[rank] != 0 {
					errs = append(errs, handles[rank].Destroy())
				}
				return errors.Join(errs...)
			})
			if err != nil {
				t.Errorf("rank %d cleanup: %v", rank, err)
				return
			}
			if err := devs[rank].Close(); err != nil {
				t.Errorf("rank %d context: %v", rank, err)
				return
			}
		}
	}()

	if err := cuda.Init(); err != nil {
		fail(err)
	}
	count, err := cuda.DeviceCount()
	if err != nil {
		fail(err)
	}
	ordinals := []int32{-1, -1}
	for i := 0; i < count; i++ {
		d, err := driver.DeviceGet(i)
		if err != nil {
			fail(err)
		}
		u, err := d.Uuid()
		if err != nil {
			fail(err)
		}
		id := fmt.Sprintf("GPU-%x-%x-%x-%x-%x", u.Bytes[:4], u.Bytes[4:6], u.Bytes[6:8], u.Bytes[8:10], u.Bytes[10:])
		for rank, want := range uuids {
			if id == want {
				ordinals[rank] = int32(i)
			}
		}
	}
	if ordinals[0] < 0 || ordinals[1] < 0 {
		fail(fmt.Errorf("admitted physical GPU UUIDs %v not visible", uuids))
	}
	for rank, ordinal := range ordinals {
		devs[rank], err = cuda.NewDevice(int(ordinal))
		if err != nil {
			fail(err)
		}
		err = devs[rank].WithContext(func() error {
			var err error
			if streams[rank], err = driver.NewStream(0); err != nil {
				return err
			}
			if send[rank], err = driver.MemAlloc[int32](hostLightElements); err != nil {
				return err
			}
			recv[rank], err = driver.MemAlloc[int32](hostLightElements)
			return err
		})
		if err != nil {
			fail(err)
		}
	}
	if handles, err = nccl.InitAll(ordinals); err != nil {
		fail(err)
	}
	transport := &cudacollective.NCCLTransport{Comms: handles, Streams: streams[:]}
	membership, err := cudacollective.NewMembership([]cudacollective.Rank{{ID: 0, DeviceID: int(ordinals[0])}, {ID: 1, DeviceID: int(ordinals[1])}})
	if err != nil {
		fail(err)
	}
	if !transport.Available(membership) {
		fail(errors.New("concrete NCCL transport unavailable"))
	}
	var comms [2]*cudacollective.Communicator
	var plans [2]*cudacollective.Plan
	op := hostLightAllReduce(hostLightElements, cudair.Int32)
	for rank := range plans {
		comms[rank], err = cudacollective.NewCommunicator(cudacollective.CommunicatorConfig{Rank: rank, Membership: membership, Generation: hostLightGeneration, Transport: transport})
		if err != nil {
			fail(err)
		}
		if plans[rank], err = cudacollective.NewPlan(comms[rank], nil); err != nil {
			fail(err)
		}
		if err := plans[rank].AddCollectiveStep("ring-allreduce", op, send[rank], recv[rank]); err != nil {
			fail(err)
		}
	}

	// Distinct per-rank inputs make a no-op, local-only or wrong-rank result
	// differ from the sum. Values are small enough that no Int32 sum overflows.
	inputs := [2][]uint32{{1, 2, 3, 4}, {11, 12, 13, 14}}
	const token = 2
	var rings [2]*cudakv.DeviceCommandRing
	var consumers [2]*cudakv.HostCommandConsumer
	for rank := range rings {
		rings[rank], consumers[rank] = hostLightConsumer(t, 1, cudakv.HostCommandInput{SeqID: 1, Words: inputs[rank]})
		if err := rings[rank].Publish(cudakv.Descriptor{SeqID: 1, Generation: hostLightGeneration, PromptToken: token}); err != nil {
			fail(err)
		}
	}

	// Both ranks prepare before either enqueues: one rank failing after its
	// peer entered NCCL would strand the peer.
	ready := make(chan struct{}, 2)
	start := make(chan struct{})
	done := make(chan struct{}, 2)
	var batchErr, prepErr [2]error
	proceed := false
	var output [2][]int32
	for rank := range rings {
		go func() {
			defer func() { done <- struct{}{} }()
			_, batchErr[rank] = consumers[rank].ConsumeBatch(context.Background(), 1, func(_ context.Context, d cudakv.Descriptor, input []uint32) error {
				tasks, ops := hostLightChain(d.PromptToken)
				out, err := hostLightRun(tasks, ops, input)
				if err == nil {
					err = newHostLightBinding(rank, len(plans)).check(comms[rank], op, len(out["bias"]))
				}
				if err == nil {
					values := make([]int32, len(out["bias"]))
					for i, w := range out["bias"] {
						values[i] = int32(w)
					}
					err = devs[rank].WithContext(func() error {
						if err := send[rank].CopyFromHost(values); err != nil {
							return err
						}
						return recv[rank].CopyFromHost([]int32{-1, -1, -1, -1})
					})
				}
				prepErr[rank] = err
				ready <- struct{}{}
				<-start
				if !proceed {
					return errors.New("all-rank preparation refused")
				}
				return devs[rank].WithContext(func() error {
					if err := plans[rank].ExecuteSteps(context.Background()); err != nil {
						return err
					}
					output[rank] = make([]int32, hostLightElements)
					return recv[rank].CopyToHost(output[rank])
				})
			})
		}()
	}
	<-ready
	<-ready
	proceed = prepErr[0] == nil && prepErr[1] == nil
	close(start)
	<-done
	<-done
	if !proceed {
		fail(fmt.Errorf("payload preparation failed: %v, %v", prepErr[0], prepErr[1]))
	}
	if batchErr[0] != nil || batchErr[1] != nil {
		fail(fmt.Errorf("NCCL rank failure: %v, %v", batchErr[0], batchErr[1]))
	}
	want := make([]int32, hostLightElements)
	for i := range want {
		for rank := range inputs {
			want[i] += int32(3*(inputs[rank][i]+token) + 5)
		}
	}
	for rank := range output {
		if !reflect.DeepEqual(output[rank], want) {
			fail(fmt.Errorf("rank %d NCCL output %v, want %v", rank, output[rank], want))
		}
	}
	for rank := range plans {
		if err := plans[rank].Close(); err != nil {
			fail(err)
		}
		if err := comms[rank].Close(); err != nil {
			fail(err)
		}
	}
	t.Logf("two-rank NCCL output verified on %v; no topology, device compute or timing claim", uuids)
}
