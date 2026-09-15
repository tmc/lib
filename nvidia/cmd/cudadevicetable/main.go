// Cudadevicetable reads GPU device facts from an attached device and writes
// them into x/cudadevicetable/devices.json.
//
// It exists so that device specifics are generated rather than hand-typed.
// A hand-maintained architecture table fails silently: a wrong shared-memory
// capacity yields a plausible occupancy number, not an error.
//
// Usage:
//
//	cudadevicetable                                  print the record for device 0
//	cudadevicetable -update x/cudadevicetable/devices.json
//	cudadevicetable -device 1 -host dgx -update path
//
// With -update the device's architecture record is replaced and every other
// architecture is left alone. Values that no instrument can read -- the
// register and shared-memory allocation granularities -- are carried across
// from the existing record rather than dropped, since a probe cannot produce
// them and losing them on every regeneration would defeat the point.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudadevicetable"
)

// probes maps a table attribute name to the driver attribute that reads it.
// Adding a row here is the only thing needed to start recording a new fact.
var probes = []struct {
	name string
	attr driver.DeviceAttribute
}{
	{"multiprocessorCount", driver.DevAttrMultiprocessorCount},
	{"warpSize", driver.DevAttrWarpSize},
	{"maxThreadsPerBlock", driver.DevAttrMaxThreadsPerBlock},
	{"maxThreadsPerMultiprocessor", driver.DevAttrMaxThreadsPerMultiprocessor},
	{"maxBlocksPerMultiprocessor", driver.DevAttrMaxBlocksPerMultiprocessor},
	{"maxRegistersPerMultiprocessor", driver.DevAttrMaxRegistersPerMultiprocessor},
	{"maxRegistersPerBlock", driver.DevAttrMaxRegistersPerBlock},
	{"maxSharedMemoryPerMultiprocessor", driver.DevAttrMaxSharedMemoryPerMultiprocessor},
	{"maxSharedMemoryPerBlock", driver.DevAttrMaxSharedMemoryPerBlock},
	{"maxSharedMemoryPerBlockOptin", driver.DevAttrMaxSharedMemoryPerBlockOptin},
	{"reservedSharedMemoryPerBlock", driver.DevAttrReservedSharedMemoryPerBlock},
	{"globalMemoryBusWidth", driver.DevAttrGlobalMemoryBusWidth},
	{"memoryClockRate", driver.DevAttrMemoryClockRate},
	{"clockRate", driver.DevAttrClockRate},
	{"totalConstantMemory", driver.DevAttrTotalConstantMemory},
	{"pageableMemoryAccess", driver.DevAttrPageableMemoryAccess},
	{"pageableMemoryAccessUsesHostPageTables", driver.DevAttrPageableMemoryAccessUsesHostPageTables},
}

// owned names the attributes this tool reads, so update can tell a value it
// failed to read from one it was never responsible for.
var owned = func() map[string]bool {
	m := make(map[string]bool, len(probes))
	for _, p := range probes {
		m[p.name] = true
	}
	return m
}()

func main() {
	ordinal := flag.Int("device", 0, "CUDA device ordinal to probe")
	update := flag.String("update", "", "devices.json to update in place; empty prints to stdout")
	host := flag.String("host", "", "host name to record in the probe (default: the system hostname)")
	flag.Parse()

	if err := run(*ordinal, *update, *host); err != nil {
		fmt.Fprintln(os.Stderr, "cudadevicetable:", err)
		os.Exit(1)
	}
}

func run(ordinal int, update, host string) error {
	arch, err := probe(ordinal, host)
	if err != nil {
		return err
	}
	if update == "" {
		return writeJSON(os.Stdout, cudadevicetable.Table{
			Schema:        1,
			Architectures: map[string]cudadevicetable.Architecture{arch.Architecture: arch},
		})
	}
	return updateFile(update, arch)
}

func probe(ordinal int, host string) (cudadevicetable.Architecture, error) {
	var zero cudadevicetable.Architecture
	dev, err := cuda.NewDevice(ordinal)
	if err != nil {
		return zero, fmt.Errorf("open device %d: %w", ordinal, err)
	}
	defer dev.Close()

	major, minor, err := dev.ComputeCapability()
	if err != nil {
		return zero, fmt.Errorf("compute capability: %w", err)
	}
	name, err := dev.Name()
	if err != nil {
		return zero, fmt.Errorf("device name: %w", err)
	}
	if host == "" {
		host, _ = os.Hostname()
	}
	if host == "" {
		host = "unknown"
	}

	attrs := make(map[string]cudadevicetable.Value, len(probes))
	for _, p := range probes {
		v, err := dev.Attribute(p.attr)
		if err != nil {
			// A driver that does not know an attribute is a fact about the
			// driver, not a reason to write a plausible number. Skip it;
			// the absence is visible in the output.
			fmt.Fprintf(os.Stderr, "cudadevicetable: %s unavailable: %v\n", p.name, err)
			continue
		}
		attrs[p.name] = cudadevicetable.Value{
			Value:      int64(v),
			Source:     cudadevicetable.Probed,
			Instrument: attrName(p.attr),
		}
	}
	if len(attrs) == 0 {
		return zero, fmt.Errorf("device %d returned no attributes", ordinal)
	}

	arch := fmt.Sprintf("sm_%d%d", major, minor)
	return cudadevicetable.Architecture{
		Architecture: cudadevicetable.Normalize(arch),
		Probe: &cudadevicetable.Probe{
			Device:            name,
			ComputeCapability: fmt.Sprintf("%d.%d", major, minor),
			DriverVersion:     driverVersion(),
			Host:              host,
			ProbedAt:          time.Now().UTC().Format("2006-01-02"),
			Tool:              "cmd/cudadevicetable",
		},
		Attributes: attrs,
	}, nil
}

// attrName renders a driver.DeviceAttribute as the "DevAttrX" constant name
// the table records as the instrument. The String method is the source of
// truth; a bare number would not be reproducible by a reader.
func attrName(a driver.DeviceAttribute) string {
	s := fmt.Sprintf("%v", a)
	if !strings.HasPrefix(s, "DevAttr") {
		return "DevAttr" + s
	}
	return s
}

// driverVersion reports the installed driver, or "" when it cannot be read.
// An empty string records that it is unknown; a guess would not.
func driverVersion() string {
	out, err := exec.Command("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader").Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

// updateFile replaces arch's record in path, preserving values that no
// instrument can read.
func updateFile(path string, arch cudadevicetable.Architecture) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var t cudadevicetable.Table
	if err := json.Unmarshal(b, &t); err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	if t.Architectures == nil {
		t.Architectures = map[string]cudadevicetable.Architecture{}
	}
	if t.Schema == 0 {
		t.Schema = 1
	}

	// Carry across what this tool does not produce. Dropping it on every
	// regeneration would make the generator lossy, which is how people end
	// up back at hand-editing.
	//
	// The distinction is ownership, not source. This tool reads the
	// attributes in probes and nothing else, so a value recorded through
	// some other instrument -- a runtime device property, say -- is not
	// stale merely because this run did not produce it. Only an attribute
	// this tool owns and failed to read is dropped, because that one really
	// has gone unconfirmed.
	if prev, ok := t.Architectures[arch.Architecture]; ok {
		for name, v := range prev.Attributes {
			if _, fresh := arch.Attributes[name]; fresh {
				continue
			}
			if owned[name] {
				fmt.Fprintf(os.Stderr, "cudadevicetable: dropping %q: this tool reads it and the device did not report it\n", name)
				continue
			}
			arch.Attributes[name] = v
		}
	}

	t.Architectures[arch.Architecture] = arch
	if err := t.Validate(); err != nil {
		return fmt.Errorf("%s: refusing to write: %w", path, err)
	}
	f, err := os.CreateTemp(filepath.Dir(path), "devices-*.json")
	if err != nil {
		return err
	}
	defer os.Remove(f.Name())
	if err := writeJSON(f, t); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(f.Name(), path)
}

func writeJSON(w *os.File, t cudadevicetable.Table) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(t)
}
