package cuptiactivity_test

import (
	"bytes"
	"errors"
	"fmt"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
	"github.com/tmc/lib/nvidia/x/perfetto"
)

func TestDecodeKernelVersions(t *testing.T) {
	k10 := cupti.ActivityKernel10{
		Kind:                 cupti.ActivityKindConcurrentKernel,
		Start:                1000,
		End:                  2000,
		Completed:            2050,
		DeviceId:             1,
		ContextId:            2,
		StreamId:             3,
		GridX:                16,
		GridY:                8,
		GridZ:                4,
		BlockX:               32,
		BlockY:               2,
		BlockZ:               1,
		StaticSharedMemory:   1024,
		DynamicSharedMemory:  2048,
		LocalMemoryPerThread: 64,
		LocalMemoryTotalV2:   4096,
		CorrelationId:        42,
		GridId:               99,
		Name:                 101,
		Queued:               800,
		Submitted:            900,
		LaunchType:           1,
		GraphId:              11,
		GraphNodeId:          12,
		IsDeviceLaunched:     1,
	}
	k11 := cupti.ActivityKernel11{
		Kind:                 cupti.ActivityKindConcurrentKernel,
		Start:                1000,
		End:                  2000,
		Completed:            2050,
		DeviceId:             1,
		ContextId:            2,
		StreamId:             3,
		GridX:                16,
		GridY:                8,
		GridZ:                4,
		BlockX:               32,
		BlockY:               2,
		BlockZ:               1,
		StaticSharedMemory:   1024,
		DynamicSharedMemory:  2048,
		LocalMemoryPerThread: 64,
		LocalMemoryTotalV2:   4096,
		CorrelationId:        42,
		GridId:               99,
		Name:                 101,
		Queued:               800,
		Submitted:            900,
		LaunchType:           1,
		GraphId:              11,
		GraphNodeId:          12,
		IsDeviceLaunched:     1,
		Priority:             -2,
	}
	k12 := cupti.ActivityKernel12{
		Kind:                 cupti.ActivityKindConcurrentKernel,
		Start:                1000,
		End:                  2000,
		Completed:            2050,
		DeviceId:             1,
		ContextId:            2,
		StreamId:             3,
		GridX:                16,
		GridY:                8,
		GridZ:                4,
		BlockX:               32,
		BlockY:               2,
		BlockZ:               1,
		StaticSharedMemory:   1024,
		DynamicSharedMemory:  2048,
		LocalMemoryPerThread: 64,
		LocalMemoryTotalV2:   4096,
		CorrelationId:        42,
		GridId:               99,
		Name:                 101,
		Queued:               800,
		Submitted:            900,
		LaunchType:           1,
		GraphId:              11,
		GraphNodeId:          12,
		IsDeviceLaunched:     1,
		Priority:             -2,
		ExecutionModel:       3,
	}
	k13 := cupti.ActivityKernel13{
		Kind:                 cupti.ActivityKindConcurrentKernel,
		Start:                1000,
		End:                  2000,
		Completed:            2050,
		DeviceId:             1,
		ContextId:            2,
		StreamId:             3,
		GridX:                16,
		GridY:                8,
		GridZ:                4,
		BlockX:               32,
		BlockY:               2,
		BlockZ:               1,
		StaticSharedMemory:   1024,
		DynamicSharedMemory:  2048,
		LocalMemoryPerThread: 64,
		LocalMemoryTotalV2:   4096,
		CorrelationId:        42,
		GridId:               99,
		Name:                 101,
		Queued:               800,
		Submitted:            900,
		LaunchType:           1,
		GraphId:              11,
		GraphNodeId:          12,
		IsDeviceLaunched:     1,
		Priority:             -2,
		ExecutionModel:       3,
		SourceGraphId:        77,
		SourceGraphNodeId:    88,
	}

	nameCopier := func(ptr uintptr) (string, error) {
		if ptr == 101 {
			return "gemm_kernel", nil
		}
		return "", errors.New("unknown pointer")
	}

	tests := []struct {
		name           string
		data           []byte
		wantSize       int
		wantPriority   int32
		wantExecModel  uint32
		wantSrcGraphID uint32
		wantSrcNodeID  uint64
		customDecoder  func([]byte, func(uintptr) (string, error)) (cuptiactivity.KernelRecord, error)
	}{
		{
			name:          "Kernel10",
			data:          unsafe.Slice((*byte)(unsafe.Pointer(&k10)), int(unsafe.Sizeof(k10))),
			wantSize:      216,
			customDecoder: cuptiactivity.DecodeKernel10,
		},
		{
			name:          "Kernel11",
			data:          unsafe.Slice((*byte)(unsafe.Pointer(&k11)), int(unsafe.Sizeof(k11))),
			wantSize:      224,
			wantPriority:  -2,
			customDecoder: cuptiactivity.DecodeKernel11,
		},
		{
			name:          "Kernel12",
			data:          unsafe.Slice((*byte)(unsafe.Pointer(&k12)), int(unsafe.Sizeof(k12))),
			wantSize:      232,
			wantPriority:  -2,
			wantExecModel: 3,
			customDecoder: cuptiactivity.DecodeKernel12,
		},
		{
			name:           "Kernel13",
			data:           unsafe.Slice((*byte)(unsafe.Pointer(&k13)), int(unsafe.Sizeof(k13))),
			wantSize:       248,
			wantPriority:   -2,
			wantExecModel:  3,
			wantSrcGraphID: 77,
			wantSrcNodeID:  88,
			customDecoder:  cuptiactivity.DecodeKernel13,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if len(tt.data) != tt.wantSize {
				t.Fatalf("test payload len = %d, want %d", len(tt.data), tt.wantSize)
			}

			// Test explicit version decoder
			gotExplicit, err := tt.customDecoder(tt.data, nameCopier)
			if err != nil {
				t.Fatalf("customDecoder failed: %v", err)
			}
			verifyKernelFields(t, gotExplicit, tt.name, tt.wantPriority, tt.wantExecModel, tt.wantSrcGraphID, tt.wantSrcNodeID)

			// Test unified DecodeKernel
			gotUnified, err := cuptiactivity.DecodeKernel(tt.data, nameCopier)
			if err != nil {
				t.Fatalf("DecodeKernel failed: %v", err)
			}
			verifyKernelFields(t, gotUnified, tt.name, tt.wantPriority, tt.wantExecModel, tt.wantSrcGraphID, tt.wantSrcNodeID)

			// Test conversion to Record
			r := gotUnified.Record()
			if r.Name != "gemm_kernel" || r.CorrelationID != 42 || r.Start != 1000 || r.End != 2000 {
				t.Errorf("Record identity fields mismatch: %+v", r)
			}
			if !r.Complete {
				t.Errorf("Record should be Complete")
			}
			if r.DeviceID != 1 || r.ContextID != 2 || r.StreamID != 3 {
				t.Errorf("Record device coordinates mismatch: %+v", r)
			}
			if r.Grid != [3]int32{16, 8, 4} || r.Block != [3]int32{32, 2, 1} {
				t.Errorf("Record grid/block mismatch: %+v", r)
			}
			if r.StaticSharedMemory != 1024 || r.DynamicSharedMemory != 2048 {
				t.Errorf("Record shared memory mismatch: %+v", r)
			}
			if r.LocalMemoryPerThread != 64 || r.LocalMemoryTotal != 4096 {
				t.Errorf("Record local memory mismatch: %+v", r)
			}

			// Test Perfetto export
			var buf bytes.Buffer
			w := perfetto.NewWriter(&buf)
			if err := cuptiactivity.WriteRecord(w, 1, 0, r); err != nil {
				t.Errorf("WriteRecord failed for %s: %v", tt.name, err)
			}
			if buf.Len() == 0 {
				t.Errorf("Perfetto trace output is empty for %s", tt.name)
			}
		})
	}
}

func verifyKernelFields(t *testing.T, k cuptiactivity.KernelRecord, name string, wantPriority int32, wantExecModel uint32, wantSrcGraphID uint32, wantSrcNodeID uint64) {
	t.Helper()
	if k.Name != "gemm_kernel" {
		t.Errorf("%s: Name = %q, want 'gemm_kernel'", name, k.Name)
	}
	if k.CorrelationID != 42 {
		t.Errorf("%s: CorrelationID = %d, want 42", name, k.CorrelationID)
	}
	if k.Start != 1000 || k.End != 2000 || k.Completed != 2050 {
		t.Errorf("%s: timestamps (Start=%d, End=%d, Completed=%d)", name, k.Start, k.End, k.Completed)
	}
	if k.DeviceID != 1 || k.ContextID != 2 || k.StreamID != 3 {
		t.Errorf("%s: coordinates (Device=%d, Context=%d, Stream=%d)", name, k.DeviceID, k.ContextID, k.StreamID)
	}
	if k.GridX != 16 || k.GridY != 8 || k.GridZ != 4 {
		t.Errorf("%s: Grid (%d, %d, %d)", name, k.GridX, k.GridY, k.GridZ)
	}
	if k.BlockX != 32 || k.BlockY != 2 || k.BlockZ != 1 {
		t.Errorf("%s: Block (%d, %d, %d)", name, k.BlockX, k.BlockY, k.BlockZ)
	}
	if k.StaticSharedMemory != 1024 || k.DynamicSharedMemory != 2048 {
		t.Errorf("%s: SharedMemory (static=%d, dynamic=%d)", name, k.StaticSharedMemory, k.DynamicSharedMemory)
	}
	if k.LocalMemoryPerThread != 64 || k.LocalMemoryTotal != 4096 {
		t.Errorf("%s: LocalMemory (perThread=%d, total=%d)", name, k.LocalMemoryPerThread, k.LocalMemoryTotal)
	}
	if k.Queued != 800 || k.Submitted != 900 {
		t.Errorf("%s: Latency timestamps (Queued=%d, Submitted=%d)", name, k.Queued, k.Submitted)
	}
	if k.GraphID != 11 || k.GraphNodeID != 12 || !k.DeviceLaunched {
		t.Errorf("%s: Graph info (id=%d, node=%d, deviceLaunched=%v)", name, k.GraphID, k.GraphNodeID, k.DeviceLaunched)
	}
	if k.Priority != wantPriority {
		t.Errorf("%s: Priority = %d, want %d", name, k.Priority, wantPriority)
	}
	if k.ExecutionModel != wantExecModel {
		t.Errorf("%s: ExecutionModel = %d, want %d", name, k.ExecutionModel, wantExecModel)
	}
	if k.SourceGraphID != wantSrcGraphID {
		t.Errorf("%s: SourceGraphID = %d, want %d", name, k.SourceGraphID, wantSrcGraphID)
	}
	if k.SourceGraphNodeID != wantSrcNodeID {
		t.Errorf("%s: SourceGraphNodeID = %d, want %d", name, k.SourceGraphNodeID, wantSrcNodeID)
	}
}

func TestDecodeKernelBoundedOverrunProtection(t *testing.T) {
	shortTests := []struct {
		name    string
		bufSize int
		decoder func([]byte, func(uintptr) (string, error)) (cuptiactivity.KernelRecord, error)
	}{
		{"empty buffer", 0, cuptiactivity.DecodeKernel},
		{"too small (10 bytes)", 10, cuptiactivity.DecodeKernel},
		{"short of K10 (215 bytes)", 215, cuptiactivity.DecodeKernel},
		{"short of K10 for DecodeKernel10", 215, cuptiactivity.DecodeKernel10},
		{"short of K11 for DecodeKernel11", 216, cuptiactivity.DecodeKernel11},
		{"short of K12 for DecodeKernel12", 224, cuptiactivity.DecodeKernel12},
		{"short of K13 for DecodeKernel13", 232, cuptiactivity.DecodeKernel13},
	}
	for _, tt := range shortTests {
		t.Run(tt.name, func(t *testing.T) {
			buf := make([]byte, tt.bufSize)
			if _, err := tt.decoder(buf, nil); err == nil {
				t.Fatalf("decoder accepted truncated buffer of %d bytes", tt.bufSize)
			}
		})
	}
}

func TestDecodeKernelSizeSelection(t *testing.T) {
	// Verify DecodeKernel routes to the correct decoder based on size
	k10Bytes := make([]byte, cupti.ActivityKernel10Size)
	(*cupti.ActivityKernel10)(unsafe.Pointer(&k10Bytes[0])).Kind = cupti.ActivityKindConcurrentKernel
	(*cupti.ActivityKernel10)(unsafe.Pointer(&k10Bytes[0])).CorrelationId = 10

	k11Bytes := make([]byte, cupti.ActivityKernel11Size)
	(*cupti.ActivityKernel11)(unsafe.Pointer(&k11Bytes[0])).Kind = cupti.ActivityKindConcurrentKernel
	(*cupti.ActivityKernel11)(unsafe.Pointer(&k11Bytes[0])).CorrelationId = 11
	(*cupti.ActivityKernel11)(unsafe.Pointer(&k11Bytes[0])).Priority = 5

	k12Bytes := make([]byte, cupti.ActivityKernel12Size)
	(*cupti.ActivityKernel12)(unsafe.Pointer(&k12Bytes[0])).Kind = cupti.ActivityKindConcurrentKernel
	(*cupti.ActivityKernel12)(unsafe.Pointer(&k12Bytes[0])).CorrelationId = 12
	(*cupti.ActivityKernel12)(unsafe.Pointer(&k12Bytes[0])).ExecutionModel = 9

	k13Bytes := make([]byte, cupti.ActivityKernel13Size)
	(*cupti.ActivityKernel13)(unsafe.Pointer(&k13Bytes[0])).Kind = cupti.ActivityKindConcurrentKernel
	(*cupti.ActivityKernel13)(unsafe.Pointer(&k13Bytes[0])).CorrelationId = 13
	(*cupti.ActivityKernel13)(unsafe.Pointer(&k13Bytes[0])).SourceGraphId = 77

	tests := []struct {
		name            string
		data            []byte
		wantCorrelation uint32
		wantPriority    int32
		wantExecModel   uint32
		wantSrcGraphID  uint32
	}{
		{"exact K10", k10Bytes, 10, 0, 0, 0},
		{"exact K11", k11Bytes, 11, 5, 0, 0},
		{"exact K12", k12Bytes, 12, 0, 9, 0},
		{"exact K13", k13Bytes, 13, 0, 0, 77},
		{"oversized K10 (220 bytes)", append(k10Bytes, make([]byte, 4)...), 10, 0, 0, 0},
		{"oversized K12 (240 bytes)", append(k12Bytes, make([]byte, 8)...), 12, 0, 9, 0},
		{"oversized K13 (300 bytes)", append(k13Bytes, make([]byte, 52)...), 13, 0, 0, 77},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			k, err := cuptiactivity.DecodeKernel(tt.data, nil)
			if err != nil {
				t.Fatalf("DecodeKernel failed: %v", err)
			}
			if k.CorrelationID != tt.wantCorrelation {
				t.Errorf("CorrelationID = %d, want %d", k.CorrelationID, tt.wantCorrelation)
			}
			if k.Priority != tt.wantPriority {
				t.Errorf("Priority = %d, want %d", k.Priority, tt.wantPriority)
			}
			if k.ExecutionModel != tt.wantExecModel {
				t.Errorf("ExecutionModel = %d, want %d", k.ExecutionModel, tt.wantExecModel)
			}
			if k.SourceGraphID != tt.wantSrcGraphID {
				t.Errorf("SourceGraphID = %d, want %d", k.SourceGraphID, tt.wantSrcGraphID)
			}
		})
	}
}

func TestDecodeKernelKindValidation(t *testing.T) {
	buf := make([]byte, cupti.ActivityKernel10Size)
	(*cupti.ActivityKernel10)(unsafe.Pointer(&buf[0])).Kind = cupti.ActivityKindDriver
	if _, err := cuptiactivity.DecodeKernel(buf, nil); err == nil {
		t.Fatal("accepted ActivityKindDriver as kernel record")
	}

	// ActivityKindKernel (non-concurrent) should also be accepted
	(*cupti.ActivityKernel10)(unsafe.Pointer(&buf[0])).Kind = cupti.ActivityKindKernel
	k, err := cuptiactivity.DecodeKernel(buf, nil)
	if err != nil {
		t.Fatalf("rejected ActivityKindKernel: %v", err)
	}
	if k.Kind != cupti.ActivityKindKernel {
		t.Errorf("Kind = %s, want ActivityKindKernel", k.Kind)
	}
}

func TestDecodeKernelNameCopier(t *testing.T) {
	k := cupti.ActivityKernel10{
		Kind: cupti.ActivityKindConcurrentKernel,
		Name: 42,
	}
	data := unsafe.Slice((*byte)(unsafe.Pointer(&k)), int(unsafe.Sizeof(k)))

	// Nil copier with non-zero pointer fails
	if _, err := cuptiactivity.DecodeKernel(data, nil); err == nil {
		t.Fatal("expected error with nil name copier")
	}

	// Failing copier returns wrapped error
	wantErr := errors.New("copy failed")
	failingCopier := func(uintptr) (string, error) { return "", wantErr }
	if _, err := cuptiactivity.DecodeKernel(data, failingCopier); !errors.Is(err, wantErr) {
		t.Fatalf("expected wantErr, got %v", err)
	}

	// Zero pointer returns empty string without calling copier
	kZero := cupti.ActivityKernel10{Kind: cupti.ActivityKindConcurrentKernel, Name: 0}
	dataZero := unsafe.Slice((*byte)(unsafe.Pointer(&kZero)), int(unsafe.Sizeof(kZero)))
	rec, err := cuptiactivity.DecodeKernel(dataZero, nil)
	if err != nil || rec.Name != "" {
		t.Fatalf("failed zero name pointer: rec=%+v, err=%v", rec, err)
	}
}

func TestKernelRecordRoundtrip(t *testing.T) {
	r := cuptiactivity.Record{
		Kind:                 cupti.ActivityKindConcurrentKernel,
		Name:                 "vectorAdd",
		Start:                500,
		End:                  1500,
		Complete:             true,
		CorrelationID:        100,
		DeviceID:             0,
		ContextID:            1,
		StreamID:             2,
		Grid:                 [3]int32{256, 1, 1},
		Block:                [3]int32{64, 1, 1},
		RegistersPerThread:   32,
		StaticSharedMemory:   512,
		DynamicSharedMemory:  1024,
		LocalMemoryPerThread: 0,
		LocalMemoryTotal:     0,
		Queued:               400,
		Submitted:            450,
	}

	k := r.Kernel()
	if k.Name != "vectorAdd" || k.GridX != 256 || k.BlockX != 64 || k.RegistersPerThread != 32 {
		t.Fatalf("Kernel() conversion lost fields: %+v", k)
	}

	r2 := k.Record()
	if r2.Name != r.Name || r2.Grid != r.Grid || r2.Block != r.Block || r2.RegistersPerThread != r.RegistersPerThread {
		t.Fatalf("Roundtrip mismatch: %+v vs %+v", r2, r)
	}
	if !r2.Complete {
		t.Errorf("r2 should be complete")
	}

	// Non-kernel record returns zero KernelRecord
	rNonKernel := cuptiactivity.Record{Kind: cupti.ActivityKindDriver}
	if kZero := rNonKernel.Kernel(); kZero != (cuptiactivity.KernelRecord{}) {
		t.Errorf("non-kernel record gave non-zero KernelRecord: %+v", kZero)
	}
}

func ExampleDecodeKernel() {
	k := cupti.ActivityKernel11{
		Kind:          cupti.ActivityKindConcurrentKernel,
		Start:         1000,
		End:           2000,
		CorrelationId: 42,
		GridX:         16,
		BlockX:        32,
	}
	data := unsafe.Slice((*byte)(unsafe.Pointer(&k)), int(unsafe.Sizeof(k)))

	record, err := cuptiactivity.DecodeKernel(data, nil)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Printf("CorrelationID: %d, GridX: %d, BlockX: %d\n", record.CorrelationID, record.GridX, record.BlockX)
	// Output:
	// CorrelationID: 42, GridX: 16, BlockX: 32
}
