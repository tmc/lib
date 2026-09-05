//go:build linux

package nvshmem

import (
	"bytes"
	"encoding/binary"
	"errors"
	"os"
	"strconv"
	"testing"
	"time"
)

// The tests in this file touch the real library, and one of them touches real
// GPUs. They are off by default and are enabled by NVSHMEM_PHYSICAL_TEST=1,
// because a machine without the prerequisites cannot distinguish "the binding
// is correct" from "nothing ran".
//
// When they are off they skip with the missing prerequisite named. When they
// are on they never skip: a missing library, a failed initialization, or a
// wrong result is a failure. A test that turns a hardware problem into a pass
// is worse than no test, since it reports coverage that does not exist.

// physical reports whether the physical tests were asked for, and skips with
// the specific prerequisite if not.
func physical(t *testing.T, needs string) {
	t.Helper()
	if os.Getenv("NVSHMEM_PHYSICAL_TEST") != "1" {
		t.Skipf("NVSHMEM_PHYSICAL_TEST is not 1: not run. Requires %s", needs)
	}
}

// TestPhysicalSymbolsResolve checks that every symbol this package binds is
// present in the library it loads. It needs libnvshmem_host.so.3 and nothing
// else: no GPU, no bootstrap, no second rank. A name this package got wrong
// would otherwise surface only when a caller reached that one function.
func TestPhysicalSymbolsResolve(t *testing.T) {
	physical(t, "libnvshmem_host.so.3 on the loader path or at NVSHMEM_LIB_PATH")

	if err := Load(""); err != nil {
		t.Fatalf("loading libnvshmem_host.so.3: %v", err)
	}
	t.Logf("loaded %s", Path())

	var missing []string
	for _, sym := range Symbols() {
		if !Has(sym) {
			missing = append(missing, sym)
		}
	}
	if len(missing) > 0 {
		t.Errorf("%d of %d bound symbols are absent from the loaded library: %v",
			len(missing), len(Symbols()), missing)
	}
}

// TestPhysicalVendorVersion checks the loaded library against the release the
// bindings were generated from. A mismatch is not automatically a failure -
// the package is built to tolerate an older library - so this reports rather
// than fails, except when the query itself does not work.
func TestPhysicalVendorVersion(t *testing.T) {
	physical(t, "libnvshmem_host.so.3 on the loader path or at NVSHMEM_LIB_PATH")

	if err := Load(""); err != nil {
		t.Fatalf("loading libnvshmem_host.so.3: %v", err)
	}
	major, minor, patch := VendorVersion()
	if major == 0 && minor == 0 && patch == 0 {
		t.Fatal("VendorVersion returned 0.0.0; nvshmemx_vendor_get_version_info did not answer")
	}
	t.Logf("library reports NVSHMEM %d.%d.%d; bindings generated from 3.7.2", major, minor, patch)
	if major != 3 || minor != 7 || patch != 2 {
		t.Logf("this is not the pinned release; layouts are checked by TestGeneratedLayouts against 3.7.2 headers")
	}
	if spec, rev := InfoGetVersion(); spec == 0 {
		t.Errorf("InfoGetVersion returned %d.%d; want the OpenSHMEM specification version", spec, rev)
	}
}

// TestPhysicalSymmetricHeap runs a real NVSHMEM job: bootstrap, allocate,
// exchange, finalize. It cannot run standalone. The launcher must start one
// process per PE, give each its rank and the job size, and give them all a
// path to a shared file for the unique ID.
//
//	NVSHMEM_PHYSICAL_TEST=1 \
//	NVSHMEM_TEST_PE=$rank NVSHMEM_TEST_NPES=$size \
//	NVSHMEM_TEST_UID_FILE=/shared/path/uid.bin \
//	go test -run TestPhysicalSymmetricHeap ./nvshmem
//
// Every PE must also have selected a CUDA device before this runs: NVSHMEM
// binds the symmetric heap to the current context's device.
func TestPhysicalSymmetricHeap(t *testing.T) {
	physical(t, "2 or more GPUs, a CUDA driver, libnvshmem_host.so.3, "+
		"an interconnect NVSHMEM supports between them, and a launcher setting "+
		"NVSHMEM_TEST_PE, NVSHMEM_TEST_NPES and NVSHMEM_TEST_UID_FILE")

	pe := mustEnvInt(t, "NVSHMEM_TEST_PE")
	npes := mustEnvInt(t, "NVSHMEM_TEST_NPES")
	uidFile := os.Getenv("NVSHMEM_TEST_UID_FILE")
	if uidFile == "" {
		t.Fatal("NVSHMEM_TEST_UID_FILE is unset; PE 0 publishes the unique ID there and the others read it")
	}
	if npes < 2 {
		t.Fatalf("NVSHMEM_TEST_NPES = %d; this test exercises communication and needs at least 2 PEs on 2 GPUs", npes)
	}

	id := rendezvous(t, pe, uidFile)
	if err := InitWithUniqueID(pe, npes, &id); err != nil {
		t.Fatalf("InitWithUniqueID(pe=%d, npes=%d): %v", pe, npes, err)
	}
	defer Finalize()

	if got := int(MyPE()); got != pe {
		t.Fatalf("MyPE = %d, want %d: the bootstrap did not assign the rank it was given", got, pe)
	}
	if got := int(NPEs()); got != npes {
		t.Fatalf("NPEs = %d, want %d", got, npes)
	}

	// The allocation is collective and must be identical on every PE.
	const size = 4096
	buf := Malloc(size)
	if buf == 0 {
		t.Fatal("Malloc(4096) returned 0; the symmetric heap allocation failed")
	}
	defer Free(buf)

	// Every PE writes to its right neighbour and reads what its left
	// neighbour wrote. The barrier before the exchange makes the remote
	// buffer allocated; the barrier after makes the write visible.
	BarrierAll()
	next := (pe + 1) % npes
	PutMem(buf, buf, size, int32(next))
	Quiet()
	BarrierAll()

	// Ptr returns a directly addressable pointer to a peer's symmetric memory
	// where the transport allows it, and 0 where it does not. Either answer is
	// correct; a panic or a wild address is not.
	t.Logf("nvshmem_ptr(buf, pe %d) = %#x", next, Ptr(buf, int32(next)))
}

// rendezvous publishes or reads the unique ID through the shared file. It is
// the "distribute the ID yourself" step: NVSHMEM has no opinion about how the
// PEs exchange it.
func rendezvous(t *testing.T, pe int, path string) UniqueID {
	t.Helper()
	if pe == 0 {
		id, err := NewUniqueID()
		if err != nil {
			t.Fatalf("NewUniqueID: %v", err)
		}
		if err := os.WriteFile(path, encodeUniqueID(t, id), 0600); err != nil {
			t.Fatalf("publishing the unique ID to %s: %v", path, err)
		}
		return id
	}
	deadline := time.Now().Add(60 * time.Second)
	for {
		data, err := os.ReadFile(path)
		if err == nil && len(data) == uniqueIDSize {
			return decodeUniqueID(t, data)
		}
		if err != nil && !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("reading the unique ID from %s: %v", path, err)
		}
		if time.Now().After(deadline) {
			t.Fatalf("PE %d waited 60s for PE 0 to publish the unique ID at %s", pe, path)
		}
		time.Sleep(50 * time.Millisecond)
	}
}

func mustEnvInt(t *testing.T, name string) int {
	t.Helper()
	s := os.Getenv(name)
	if s == "" {
		t.Fatalf("%s is unset; the launcher must set it", name)
	}
	n, err := strconv.Atoi(s)
	if err != nil {
		t.Fatalf("%s = %q: %v", name, s, err)
	}
	return n
}

// A UniqueID is an opaque blob with exported fields, so encoding/binary moves
// it between processes without unsafe. Its size is fixed by the ABI, which
// TestGeneratedLayouts pins at 128 bytes.
const uniqueIDSize = 128

func encodeUniqueID(t *testing.T, id UniqueID) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := binary.Write(&buf, binary.LittleEndian, id); err != nil {
		t.Fatalf("encoding the unique ID: %v", err)
	}
	if buf.Len() != uniqueIDSize {
		t.Fatalf("encoded unique ID is %d bytes, want %d", buf.Len(), uniqueIDSize)
	}
	return buf.Bytes()
}

func decodeUniqueID(t *testing.T, data []byte) UniqueID {
	t.Helper()
	var id UniqueID
	if err := binary.Read(bytes.NewReader(data), binary.LittleEndian, &id); err != nil {
		t.Fatalf("decoding the unique ID: %v", err)
	}
	return id
}
