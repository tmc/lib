package cudamem_test

import (
	"bytes"
	"io"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

func TestAllocAndCopies(t *testing.T) {
	if err := cuda.Init(); err != nil {
		t.Skipf("CUDA driver not available: %v", err)
	}
	count, err := cuda.DeviceCount()
	if err != nil || count == 0 {
		t.Skip("no CUDA devices available")
	}

	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatalf("NewDevice failed: %v", err)
	}

	n := 1024
	dslice, err := cudamem.Alloc[float32](dev, n)
	if err != nil {
		t.Fatalf("Alloc failed: %v", err)
	}
	defer dslice.Free()

	if dslice.Len() != n {
		t.Errorf("expected len %d, got %d", n, dslice.Len())
	}

	hostSrc := make([]float32, n)
	for i := range hostSrc {
		hostSrc[i] = float32(i) * 1.5
	}

	if err := dslice.CopyFromHost(hostSrc); err != nil {
		t.Fatalf("CopyFromHost failed: %v", err)
	}

	hostDst := make([]float32, n)
	if err := dslice.CopyToHost(hostDst); err != nil {
		t.Fatalf("CopyToHost failed: %v", err)
	}

	for i := range hostSrc {
		if hostDst[i] != hostSrc[i] {
			t.Fatalf("mismatch at index %d: want %v, got %v", i, hostSrc[i], hostDst[i])
		}
	}
}

func TestTrackedAllocation(t *testing.T) {
	if err := cuda.Init(); err != nil {
		t.Skipf("CUDA driver not available: %v", err)
	}
	count, err := cuda.DeviceCount()
	if err != nil || count == 0 {
		t.Skip("no CUDA devices available")
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()

	var tracker cudamem.Tracker
	a, err := cudamem.AllocTracked[float32](dev, &tracker, 1024)
	if err != nil {
		t.Fatal(err)
	}
	if current, peak := tracker.Snapshot(); current != 4096 || peak != 4096 {
		t.Fatalf("after first allocation = %d, %d, want 4096, 4096", current, peak)
	}
	tracker.Reset()
	b, err := cudamem.AllocTracked[byte](dev, &tracker, 17)
	if err != nil {
		t.Fatal(err)
	}
	if current, peak := tracker.Snapshot(); current != 4113 || peak != 4113 {
		t.Fatalf("after second allocation = %d, %d, want 4113, 4113", current, peak)
	}
	if err := b.Free(); err != nil {
		t.Fatal(err)
	}
	if err := b.Free(); err != nil {
		t.Fatalf("second Free: %v", err)
	}
	if current, peak := tracker.Snapshot(); current != 4096 || peak != 4113 {
		t.Fatalf("after free = %d, %d, want 4096, 4113", current, peak)
	}
	if err := a.Free(); err != nil {
		t.Fatal(err)
	}
	if current, peak := tracker.Snapshot(); current != 0 || peak != 4113 {
		t.Fatalf("after final free = %d, %d, want 0, 4113", current, peak)
	}
}

func TestHostAllocation(t *testing.T) {
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Skipf("CUDA driver not available: %v", err)
	}
	defer dev.Close()

	n := 512
	var ha cudamem.HostAllocation[int32]
	err = dev.WithContext(func() error {
		var err error
		ha, err = cudamem.AllocHost[int32](n)
		return err
	})
	if err != nil {
		t.Fatalf("AllocHost failed: %v", err)
	}
	defer dev.WithContext(ha.Free)

	slice := ha.Slice()
	for i := 0; i < n; i++ {
		slice[i] = int32(i * 10)
	}

	for i := 0; i < n; i++ {
		if slice[i] != int32(i*10) {
			t.Fatalf("mismatch in host pinned slice at %d", i)
		}
	}
}

func TestStreamingIO(t *testing.T) {
	if err := cuda.Init(); err != nil {
		t.Skipf("CUDA driver not available: %v", err)
	}
	count, err := cuda.DeviceCount()
	if err != nil || count == 0 {
		t.Skip("no CUDA devices available")
	}

	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatalf("NewDevice failed: %v", err)
	}

	data := []byte("Hello, CUDA streaming IO through Go standard interfaces!")
	dslice, err := cudamem.Alloc[byte](dev, len(data))
	if err != nil {
		t.Fatalf("Alloc failed: %v", err)
	}
	defer dslice.Free()

	// Write via ReaderFrom
	w := cudamem.NewWriter(dslice)
	n, err := w.ReadFrom(bytes.NewReader(data))
	if err != nil {
		t.Fatalf("ReadFrom failed: %v", err)
	}
	if n != int64(len(data)) {
		t.Errorf("expected %d bytes written, got %d", len(data), n)
	}

	// Read via WriterTo
	r := cudamem.NewReader(dslice)
	var out bytes.Buffer
	readN, err := r.WriteTo(&out)
	if err != nil && err != io.EOF {
		t.Fatalf("WriteTo failed: %v", err)
	}
	if readN != int64(len(data)) {
		t.Errorf("expected %d bytes read, got %d", len(data), readN)
	}

	if !bytes.Equal(out.Bytes(), data) {
		t.Errorf("payload mismatch: want %q, got %q", string(data), out.String())
	}
}
