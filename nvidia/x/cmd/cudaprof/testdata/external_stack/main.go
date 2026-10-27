// This target is run only by the separately authorized physical stack fixture.
package main

import (
	"context"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strconv"
	"time"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/cudaprofagent"
)

type contract struct {
	PID          int    `json:"pid"`
	ExternalID   uint64 `json:"external_id"`
	ExternalKind uint32 `json:"external_kind"`
	UUID         string `json:"uuid"`
	Target       string `json:"target"`
	Artifact     string `json:"artifact"`
	SleepNs      int64  `json:"sleep_ns"`
	Output       uint32 `json:"output"`
}

//go:noinline
func sleptKernel(stream *cuda.Stream, kernel *cuda.Kernel, out cudamem.Slice[uint32]) (id uint64, err error) {
	id = cudaprof.DoID(context.Background(), cudaprof.Labels("fixture", "external-stack"), func(context.Context) {
		time.Sleep(10 * time.Millisecond)
		err = cudalaunch.OwnedKernel(kernel).Grid(1, 1, 1).Block(1, 1, 1).On(stream).Launch(out)
	})
	return id, err
}

func run() (c contract, retErr error) {
	if os.Getenv("CUDAPROF_EXTERNAL_STACK_NATIVE") != "1" {
		return c, errors.New("native stack target not admitted")
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		return c, err
	}
	ordinal, err := strconv.Atoi(os.Getenv("CUDAPROF_EXTERNAL_STACK_DEVICE"))
	if err != nil || ordinal < 0 {
		return c, errors.New("explicit device ordinal required")
	}
	target := os.Getenv("CUDAPROF_EXTERNAL_STACK_SM")
	if target != "sm_100a" && target != "sm_121a" {
		return c, errors.New("explicit sm_100a or sm_121a target required")
	}
	expectedUUID := os.Getenv("CUDAPROF_EXTERNAL_STACK_UUID")
	uuidBytes, err := hex.DecodeString(expectedUUID)
	if err != nil || len(uuidBytes) != 16 || hex.EncodeToString(uuidBytes) != expectedUUID {
		return c, errors.New("device UUID must be 32 lowercase hex digits")
	}
	var dev *cuda.Device
	var stream *cuda.Stream
	var raw driver.Module
	var module *cuda.Module
	var out cudamem.Slice[uint32]
	// Failed cleanup leaves the remaining handles intact until process exit;
	// it fails the target and never reports successful reclamation.
	cleanup := func() error {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if stream != nil {
			if err := stream.Synchronize(ctx); err != nil {
				return err
			}
		}
		if err := out.Free(); err != nil {
			return err
		}
		if module != nil {
			if err := module.Close(); err != nil {
				return err
			}
			module = nil
		}
		if raw != 0 {
			if err := dev.WithContext(func() error {
				if err := raw.Close(); err != nil {
					return err
				}
				raw = 0
				return nil
			}); err != nil {
				return err
			}
		}
		if stream != nil {
			if err := stream.Close(); err != nil {
				return err
			}
			stream = nil
		}
		if dev != nil {
			return dev.Close()
		}
		return nil
	}
	defer func() { retErr = errors.Join(retErr, cleanup()) }()
	dev, err = cuda.NewDevice(ordinal)
	if err != nil {
		return c, err
	}
	uuid, err := dev.Raw().Uuid()
	if err != nil {
		return c, err
	}
	if hex.EncodeToString(uuid.Bytes[:]) != expectedUUID {
		return c, errors.New("device UUID mismatch")
	}
	major, minor, err := dev.ComputeCapability()
	if err != nil || fmt.Sprintf("sm_%d%da", major, minor) != target {
		return c, errors.Join(errors.New("device architecture mismatch"), err)
	}
	stream, err = dev.NewStream()
	if err != nil {
		return c, err
	}
	out, err = cudamem.Alloc[uint32](dev, 1)
	if err != nil {
		return c, err
	}
	if err := out.CopyFromHost([]uint32{0xdeadbeef}); err != nil {
		return c, err
	}
	schema, err := cudakernel.NewSchema(cudakernel.Param{Name: "out", Type: "ptr", IsPointer: true})
	if err != nil {
		return c, err
	}
	ptx := fmt.Sprintf(`.version 9.0
.target %s
.address_size 64
.visible .entry external_stack_short(.param .u64 out) {
 .reg .b64 p;
 .reg .b32 value;
 ld.param.u64 p, [out];
 mov.u32 value, 42;
 st.global.u32 [p], value;
 ret;
}
`, target)
	artifact, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte(ptx), "external_stack_short", schema, target)
	if err != nil {
		return c, err
	}
	raw, err = artifact.Load(dev)
	if err != nil {
		return c, err
	}
	module, err = cuda.OwnModule(dev, raw)
	if err != nil {
		return c, err
	}
	raw = 0
	kernel, err := module.Function(artifact.EntryPoint())
	if err != nil {
		return c, err
	}
	cudaprof.Correlate(cupti.ExternalCorrelationKindCustom0)
	defer cudaprof.Correlate(cupti.ExternalCorrelationKindInvalid)
	id, err := sleptKernel(stream, kernel, out)
	if err != nil || id == 0 {
		return c, errors.Join(errors.New("kernel submission or external correlation unavailable"), err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := stream.Synchronize(ctx); err != nil {
		return c, err
	}
	got := make([]uint32, 1)
	if err := out.CopyToHost(got); err != nil {
		return c, err
	}
	if got[0] != 42 {
		return c, fmt.Errorf("output = %d, want 42", got[0])
	}
	return contract{os.Getpid(), id, uint32(cupti.ExternalCorrelationKindCustom0), expectedUUID, target, artifact.Digest(), int64(10 * time.Millisecond), got[0]}, nil
}

func main() {
	// Do not even resolve the optional agent finalizer after refused admission.
	if os.Getenv("CUDAPROF_EXTERNAL_STACK_NATIVE") != "1" {
		fmt.Fprintln(os.Stderr, "native stack target not admitted")
		os.Exit(1)
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	c, err := run()
	if !cudaprofagent.Finalize() {
		err = errors.Join(err, errors.New("injected agent did not finalize"))
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if err := json.NewEncoder(os.Stdout).Encode(c); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
