package cudakernel_test

import (
	"os"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudakernel"
)

// affinePTX is a valid, hand-crafted PTX module implementing y[i] = a * x[i] + b.
// PTX ISA 8.8 is the first to accept .target sm_121a; ptxas 13.0 and 13.4
// reject 8.7 and earlier with "PTX .version 8.7 does not support .target sm_121a".
const affinePTX = `//
// Affine kernel: y[i] = a * x[i] + b
//
.version 8.8
.target sm_121a
.address_size 64

.visible .entry affine_kernel(
	.param .u64 param_y,
	.param .u64 param_x,
	.param .f32 param_a,
	.param .f32 param_b,
	.param .u32 param_n
)
{
	.reg .b32 	%r<5>;
	.reg .f32 	%f<5>;
	.reg .b64 	%rd<7>;
	.reg .pred 	%p<2>;

	ld.param.u32 	%r1, [param_n];
	mov.u32 	%r2, %ctaid.x;
	mov.u32 	%r3, %ntid.x;
	mov.u32 	%r4, %tid.x;
	mad.lo.s32 	%r0, %r2, %r3, %r4;

	setp.ge.s32 	%p1, %r0, %r1;
	@%p1 bra 	BB0_2;

	ld.param.u64 	%rd1, [param_x];
	ld.param.u64 	%rd2, [param_y];
	ld.param.f32 	%f1, [param_a];
	ld.param.f32 	%f2, [param_b];

	mul.wide.s32 	%rd3, %r0, 4;
	add.s64 	%rd4, %rd1, %rd3;
	add.s64 	%rd5, %rd2, %rd3;

	ld.global.f32 	%f3, [%rd4];
	fma.rn.f32 	%f4, %f3, %f1, %f2;
	st.global.f32 	[%rd5], %f4;

BB0_2:
	ret;
}
`

func TestNativeAffineArtifact(t *testing.T) {
	// 1. Define explicit ABI parameter schema
	schema, err := cudakernel.NewSchema(
		cudakernel.Param{Name: "param_y", Type: "ptr", IsPointer: true},
		cudakernel.Param{Name: "param_x", Type: "ptr", IsPointer: true},
		cudakernel.Param{Name: "param_a", Type: "f32"},
		cudakernel.Param{Name: "param_b", Type: "f32"},
		cudakernel.Param{Name: "param_n", Type: "u32"},
	)
	if err != nil {
		t.Fatalf("NewSchema: %v", err)
	}

	// Verify parameter layout:
	// param_y: offset 0 (8 bytes)
	// param_x: offset 8 (8 bytes)
	// param_a: offset 16 (4 bytes)
	// param_b: offset 20 (4 bytes)
	// param_n: offset 24 (4 bytes)
	// Total: 32 bytes (aligned to 8)
	if schema.TotalParamBytes != 32 {
		t.Fatalf("TotalParamBytes = %d, want 32", schema.TotalParamBytes)
	}

	// 2. Build immutable artifact descriptor for SM121a (GB10)
	art, err := cudakernel.NewArtifact(
		cudakernel.KindPTX,
		[]byte(affinePTX),
		"affine_kernel",
		schema,
		"sm_121a",
		cudakernel.WithResources(cudakernel.ResourceRequirements{
			AllocatedRegisters:  32,
			PeakLiveGPR:         16, // Register liveness peak is distinct from 32 allocated registers
			RegisterQuantum:     256,
			ExecutionGroup:      "cta",
			StaticSharedMemory:  0,
			DynamicSharedMemory: 0,
		}),
		cudakernel.WithJIT(cudakernel.JITOptions{
			MaxRegisters:    32,
			ThreadsPerBlock: 256,
			OptLevel:        4,
		}),
		cudakernel.WithToolchain("nvrtc 13.0"),
		cudakernel.WithCompilerOptions([]string{"--std=c++20", "--gpu-architecture=sm_121a"}),
	)
	if err != nil {
		t.Fatalf("NewArtifact: %v", err)
	}

	// 3. Host-only validation of ABI packing
	packed, err := art.Schema.Pack(
		driver.DevicePtr(0x1000), // y
		driver.DevicePtr(0x2000), // x
		float32(2.5),             // a
		float32(1.0),             // b
		uint32(1024),             // n
	)
	if err != nil {
		t.Fatalf("Pack failed: %v", err)
	}
	if len(packed) != 32 {
		t.Fatalf("packed len = %d, want 32", len(packed))
	}

	// 4. Target compatibility check against GB10 (sm_121a)
	if err := art.ValidateTarget("sm_121a"); err != nil {
		t.Fatalf("ValidateTarget sm_121a: %v", err)
	}

	// 5. Gated native execution: requires explicit hardware test authorization
	if os.Getenv("CUDA_COMPILER_TEST") != "1" {
		t.Skip("native qualification pending: CUDA_COMPILER_TEST=1 required for hardware load")
	}

	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Skipf("CUDA device unavailable: %v", err)
	}
	defer dev.Close()

	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	if major != 12 || minor != 1 {
		t.Skipf("sm_121a fixture requires compute capability 12.1; device is %d.%d", major, minor)
	}

	mod, err := art.Load(dev)
	if err != nil {
		t.Fatalf("art.Load on device: %v", err)
	}
	defer func() {
		_ = dev.WithContext(mod.Close)
	}()
}
