//go:build gopccl_pccl

package gopccl

import (
	"errors"
	"testing"
	"unsafe"

	pccl "github.com/tmc/lib/pccl"
)

func TestGeneratedPCCLDataTypeMapping(t *testing.T) {
	tests := []struct {
		in   DType
		want pccl.DataType
	}{
		{Int8, pccl.Int8},
		{Int16, pccl.Int16},
		{Int32, pccl.Int32},
		{Int64, pccl.Int64},
		{Uint8, pccl.Uint8},
		{Uint16, pccl.Uint16},
		{Uint32, pccl.Uint32},
		{Uint64, pccl.Uint64},
		{Float32, pccl.Float},
		{Float64, pccl.Double},
	}
	for _, tt := range tests {
		got, err := pcclDataType(tt.in)
		if err != nil {
			t.Fatalf("pcclDataType(%v): %v", tt.in, err)
		}
		if got != tt.want {
			t.Fatalf("pcclDataType(%v) = %v, want %v", tt.in, got, tt.want)
		}
	}
	if _, err := pcclDataType(DTypeInvalid); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("pcclDataType(DTypeInvalid) error = %v, want ErrInvalidArgument", err)
	}
}

func TestGeneratedPCCLReduceOpMapping(t *testing.T) {
	tests := []struct {
		in   ReduceOp
		want pccl.RedOp
	}{
		{Sum, pccl.Sum},
		{Max, pccl.Max},
		{Min, pccl.Min},
		{Prod, pccl.Prod},
	}
	for _, tt := range tests {
		got, err := pcclReduceOp(tt.in)
		if err != nil {
			t.Fatalf("pcclReduceOp(%v): %v", tt.in, err)
		}
		if got != tt.want {
			t.Fatalf("pcclReduceOp(%v) = %v, want %v", tt.in, got, tt.want)
		}
	}
	if _, err := pcclReduceOp(ReduceOp(-1)); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("pcclReduceOp(invalid) error = %v, want ErrInvalidArgument", err)
	}
}

func TestGeneratedPCCLSyncStrategyMapping(t *testing.T) {
	tests := []struct {
		in   SyncStrategy
		want pccl.SharedStateSyncStrategy
	}{
		{EnforcePopular, pccl.SharedStateSyncStrategyEnforcePopular},
		{ReceiveOnly, pccl.SharedStateSyncStrategyReceiveOnly},
		{SendOnly, pccl.SharedStateSyncStrategySendOnly},
	}
	for _, tt := range tests {
		got, err := pcclSyncStrategy(tt.in)
		if err != nil {
			t.Fatalf("pcclSyncStrategy(%v): %v", tt.in, err)
		}
		if got != tt.want {
			t.Fatalf("pcclSyncStrategy(%v) = %v, want %v", tt.in, got, tt.want)
		}
	}
	if _, err := pcclSyncStrategy(SyncStrategy(-1)); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("pcclSyncStrategy(invalid) error = %v, want ErrInvalidArgument", err)
	}
}

func TestGeneratedPCCLParseAddress(t *testing.T) {
	addr, err := pcclParseAddress("127.0.0.1:48148")
	if err != nil {
		t.Fatal(err)
	}
	if addr.Inet.Protocol != pccl.IPv4 || addr.Port != 48148 || addr.Inet.Ipv4.Data != [4]uint8{127, 0, 0, 1} {
		t.Fatalf("IPv4 address = %#v", addr)
	}
	addr, err = pcclParseAddress("[::1]:48148")
	if err != nil {
		t.Fatal(err)
	}
	if addr.Inet.Protocol != pccl.IPv6 || addr.Port != 48148 {
		t.Fatalf("IPv6 address = %#v", addr)
	}
	for _, value := range []string{"", "localhost", "127.0.0.1", "127.0.0.1:not-a-port", "127.0.0.1:70000"} {
		if _, err := pcclParseAddress(value); !errors.Is(err, ErrInvalidArgument) {
			t.Fatalf("pcclParseAddress(%q) error = %v, want ErrInvalidArgument", value, err)
		}
	}
}

func TestGeneratedPCCLValidateConfig(t *testing.T) {
	if err := pcclValidateConfig(Config{Rank: 0, Size: 1, Coordinator: "127.0.0.1:48148"}); err != nil {
		t.Fatalf("pcclValidateConfig(valid): %v", err)
	}
	for _, cfg := range []Config{
		{Rank: 0, Size: 0, Coordinator: "127.0.0.1:48148"},
		{Rank: -1, Size: 1, Coordinator: "127.0.0.1:48148"},
		{Rank: 1, Size: 1, Coordinator: "127.0.0.1:48148"},
		{Rank: 0, Size: 1},
		{Rank: 0, Size: 1, Coordinator: "bad"},
	} {
		if err := pcclValidateConfig(cfg); !errors.Is(err, ErrInvalidArgument) {
			t.Fatalf("pcclValidateConfig(%+v) error = %v, want ErrInvalidArgument", cfg, err)
		}
	}
}

func TestGeneratedPCCLConfigFromEnv(t *testing.T) {
	t.Setenv("GOPCCL_COORDINATOR", "127.0.0.1:48148")
	t.Setenv("GOPCCL_PEER_GROUP", "7")
	t.Setenv("GOPCCL_P2P_POOL_SIZE", "9")
	t.Setenv("GOPCCL_START_MASTER", "true")
	cfg, err := ConfigFromEnv()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Coordinator != "127.0.0.1:48148" || cfg.PeerGroup != 7 || cfg.P2PPoolSize != 9 || !cfg.StartMaster {
		t.Fatalf("ConfigFromEnv() = %+v", cfg)
	}

	t.Setenv("GOPCCL_PEER_GROUP", "4294967296")
	if _, err := ConfigFromEnv(); err == nil {
		t.Fatal("ConfigFromEnv accepted overflowing GOPCCL_PEER_GROUP")
	}
}

func TestGeneratedPCCLSharedState(t *testing.T) {
	var x float32
	owner := &x
	state, pins, err := pcclSharedState(42, []StateTensor{{
		Name: "weight",
		Buffer: Buffer{
			Ptr:    unsafe.Pointer(&x),
			Owner:  owner,
			Count:  1,
			DType:  Float32,
			Device: Device{Kind: DeviceHost},
		},
		AllowInequality: true,
	}})
	if err != nil {
		t.Fatal(err)
	}
	if state.Revision != 42 || state.Count != 1 || state.Infos == 0 {
		t.Fatalf("state = %#v", state)
	}
	if len(pins.infos) != 1 || len(pins.names) != 1 || pins.owners[0] != owner {
		t.Fatalf("pins = %#v", pins)
	}
	info := unsafe.Slice((*pccl.TensorInfo)(unsafe.Pointer(state.Infos)), int(state.Count))[0]
	if got := string(pins.names[0]); got != "weight\x00" {
		t.Fatalf("name backing = %q", got)
	}
	if info.Name != uintptr(unsafe.Pointer(&pins.names[0][0])) || info.Data == 0 || info.Count != 1 || info.Datatype != pccl.Float || info.DeviceType != pccl.DeviceCpu || !info.AllowContentInequality {
		t.Fatalf("tensor info = %#v", info)
	}
}

func TestGeneratedPCCLSharedStateValidation(t *testing.T) {
	var x int32
	validBuffer := Buffer{Ptr: unsafe.Pointer(&x), Count: 1, DType: Int32, Device: Device{Kind: DeviceCUDA}}
	for _, tensors := range [][]StateTensor{
		nil,
		{{Name: "", Buffer: validBuffer}},
		{{Name: "bad\x00name", Buffer: validBuffer}},
		{{Name: "bad", Buffer: Buffer{Count: 1, DType: Int32, Device: Device{Kind: DeviceCUDA}}}},
		{{Name: "bad", Buffer: Buffer{Ptr: unsafe.Pointer(&x), Count: 1, DType: DTypeInvalid, Device: Device{Kind: DeviceCUDA}}}},
		{{Name: "bad", Buffer: Buffer{Ptr: unsafe.Pointer(&x), Count: 1, DType: Int32, Device: Device{Kind: DeviceROCM}}}},
	} {
		if _, _, err := pcclSharedState(1, tensors); !errors.Is(err, ErrInvalidArgument) {
			t.Fatalf("pcclSharedState(%+v) error = %v, want ErrInvalidArgument", tensors, err)
		}
	}
}

func TestGeneratedPCCLBufferPtrValidation(t *testing.T) {
	var x int
	for _, kind := range []DeviceKind{DeviceHost, DeviceCUDA} {
		ptr, err := pcclBufferPtr(Buffer{Ptr: unsafe.Pointer(&x), Count: 1, Device: Device{Kind: kind}})
		if err != nil {
			t.Fatalf("pcclBufferPtr(%v): %v", kind, err)
		}
		if ptr == 0 {
			t.Fatal("valid pointer mapped to zero")
		}
	}
	if _, err := pcclBufferPtr(Buffer{Ptr: unsafe.Pointer(&x), Count: 1, Device: Device{Kind: DeviceROCM}}); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("pcclBufferPtr(DeviceROCM) error = %v, want ErrInvalidArgument", err)
	}
	if _, err := pcclBufferPtr(Buffer{Count: 1, Device: Device{Kind: DeviceHost}}); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("pcclBufferPtr(nil) error = %v, want ErrInvalidArgument", err)
	}
	if _, err := pcclBufferPtr(Buffer{Ptr: unsafe.Pointer(&x), Device: Device{Kind: DeviceHost}}); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("pcclBufferPtr(zero count) error = %v, want ErrInvalidArgument", err)
	}
}

func TestGeneratedPCCLCapabilities(t *testing.T) {
	caps := (&pcclBackend{}).Capabilities()
	if len(caps.Collectives) != 1 || caps.Collectives[0] != "all_reduce" {
		t.Fatalf("collectives = %#v, want all_reduce only", caps.Collectives)
	}
	if caps.PointToPoint {
		t.Fatal("pccl capabilities unexpectedly advertise point-to-point")
	}
}
