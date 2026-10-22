package cudacluster

import (
	"cmp"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
)

// DeviceTopology describes the host PCIe, NUMA, and InfiniBand HCA affinity of
// an NVIDIA GPU discovered from Linux procfs and sysfs without calling cuInit.
type DeviceTopology struct {
	DeviceIndex  int
	PCIBusID     string
	NUMANode     int
	NearestIBDev string
	IBNUMANode   int
}

// NormalizeUnsetNUMA normalizes negative kernel NUMA node sentinels (such as -1
// on single-socket or UMA hosts) to 0, leaving non-negative NUMA node indices
// unchanged.
func NormalizeUnsetNUMA(node int) int {
	if node < 0 {
		return 0
	}
	return node
}

func resolveFSRoot(fsRoot string) string {
	if strings.TrimSpace(fsRoot) == "" {
		return "/"
	}
	return fsRoot
}

func validateTopologyIdent(kind, ident string) error {
	if ident == "" {
		return fmt.Errorf("cudacluster: empty %s identifier", kind)
	}
	if ident == "." || ident == ".." || strings.Contains(ident, "..") || strings.ContainsAny(ident, `/\`) {
		return fmt.Errorf("cudacluster: invalid %s identifier %q", kind, ident)
	}
	return nil
}

func parseNUMAInt(raw string, source string) (int, error) {
	s := strings.TrimSpace(raw)
	if s == "" {
		return 0, fmt.Errorf("cudacluster: empty NUMA node in %s", source)
	}
	val, err := strconv.ParseInt(s, 0, 32)
	if err != nil {
		return 0, fmt.Errorf("cudacluster: parse NUMA node from %s (%q): %w", source, s, err)
	}
	if val < -1 {
		return 0, fmt.Errorf("cudacluster: invalid NUMA node %d in %s: must be >= -1", val, source)
	}
	return int(val), nil
}

// ReadProcNvidiaGPUNUMANode reads the NUMA node for pciBusID from
// <fsRoot>/proc/driver/nvidia/gpus/<pciBusID>/information, falling back to
// [ReadSysfsPCINUMANode] when the procfs information file omits a numeric NUMA
// Node entry.
func ReadProcNvidiaGPUNUMANode(fsRoot string, pciBusID string) (int, error) {
	if err := validateTopologyIdent("PCI bus ID", pciBusID); err != nil {
		return 0, err
	}
	root := resolveFSRoot(fsRoot)
	infoPath := filepath.Join(root, "proc", "driver", "nvidia", "gpus", pciBusID, "information")
	data, err := os.ReadFile(infoPath)
	if err != nil {
		return 0, fmt.Errorf("cudacluster: read procfs GPU information %s: %w", infoPath, err)
	}
	content := strings.TrimSpace(string(data))
	if content == "" {
		return 0, fmt.Errorf("cudacluster: empty procfs GPU information %s", infoPath)
	}

	// Support either key-value procfs format ("NUMA Node: 0") or bare integer fixtures.
	if !strings.Contains(content, ":") && !strings.Contains(content, "\n") {
		return parseNUMAInt(content, infoPath)
	}

	for _, line := range strings.Split(content, "\n") {
		k, v, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		key := strings.ToLower(strings.TrimSpace(k))
		if key != "numa node" && key != "gpu numa node" {
			continue
		}
		valStr := strings.TrimSpace(v)
		if valStr == "" || strings.EqualFold(valStr, "n/a") || strings.EqualFold(valStr, "none") {
			break
		}
		return parseNUMAInt(valStr, infoPath)
	}

	sysNode, sysErr := ReadSysfsPCINUMANode(fsRoot, pciBusID)
	if sysErr == nil {
		return sysNode, nil
	}
	return 0, fmt.Errorf("cudacluster: no NUMA node in %s and sysfs fallback failed: %w", infoPath, sysErr)
}

// ReadSysfsPCINUMANode reads the raw NUMA node for pciBusID from
// <fsRoot>/sys/bus/pci/devices/<pciBusID>/numa_node (or the nvidia driver
// sysfs fallback <fsRoot>/sys/bus/pci/drivers/nvidia/<pciBusID>/numa_node).
// A kernel sentinel of -1 is returned as -1; pass the result to
// [NormalizeUnsetNUMA] when a non-negative node index is required.
func ReadSysfsPCINUMANode(fsRoot string, pciBusID string) (int, error) {
	if err := validateTopologyIdent("PCI bus ID", pciBusID); err != nil {
		return 0, err
	}
	root := resolveFSRoot(fsRoot)
	candidates := []string{
		filepath.Join(root, "sys", "bus", "pci", "devices", pciBusID, "numa_node"),
		filepath.Join(root, "sys", "bus", "pci", "drivers", "nvidia", pciBusID, "numa_node"),
	}
	var lastErr error
	for _, p := range candidates {
		data, err := os.ReadFile(p)
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				lastErr = err
				continue
			}
			return 0, fmt.Errorf("cudacluster: read PCI NUMA node %s: %w", p, err)
		}
		return parseNUMAInt(string(data), p)
	}
	return 0, fmt.Errorf("cudacluster: PCI NUMA node unavailable for %s: %w", pciBusID, lastErr)
}

// ReadSysfsIBNUMANode reads the raw NUMA node for an InfiniBand HCA from
// <fsRoot>/sys/class/infiniband/<ibDevName>/device/numa_node.
func ReadSysfsIBNUMANode(fsRoot string, ibDevName string) (int, error) {
	if err := validateTopologyIdent("InfiniBand device", ibDevName); err != nil {
		return 0, err
	}
	root := resolveFSRoot(fsRoot)
	p := filepath.Join(root, "sys", "class", "infiniband", ibDevName, "device", "numa_node")
	data, err := os.ReadFile(p)
	if err != nil {
		return 0, fmt.Errorf("cudacluster: read InfiniBand NUMA node %s: %w", p, err)
	}
	return parseNUMAInt(string(data), p)
}

type ibEndpoint struct {
	name     string
	order    int
	numaNode int
}

func parseIBIndex(name string) int {
	if idx := strings.LastIndexByte(name, '_'); idx >= 0 && idx+1 < len(name) {
		if n, err := strconv.Atoi(name[idx+1:]); err == nil && n >= 0 {
			return n
		}
	}
	return int(^uint(0) >> 1)
}

func discoverIBEndpoints(fsRoot string) ([]ibEndpoint, error) {
	root := resolveFSRoot(fsRoot)
	ibDir := filepath.Join(root, "sys", "class", "infiniband")
	entries, err := os.ReadDir(ibDir)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, fmt.Errorf("cudacluster: read infiniband sysfs %s: %w", ibDir, err)
	}

	var mlxDevs []string
	var otherDevs []string
	for _, e := range entries {
		name := e.Name()
		if strings.HasPrefix(name, ".") {
			continue
		}
		if strings.HasPrefix(name, "mlx5_") {
			mlxDevs = append(mlxDevs, name)
		} else {
			otherDevs = append(otherDevs, name)
		}
	}
	names := mlxDevs
	if len(names) == 0 {
		names = otherDevs
	}
	slices.SortFunc(names, func(a, b string) int {
		ia, ib := parseIBIndex(a), parseIBIndex(b)
		if ia != ib {
			return cmp.Compare(ia, ib)
		}
		return cmp.Compare(a, b)
	})

	out := make([]ibEndpoint, 0, len(names))
	for _, name := range names {
		node, err := ReadSysfsIBNUMANode(fsRoot, name)
		if err != nil {
			return nil, err
		}
		out = append(out, ibEndpoint{
			name:     name,
			order:    parseIBIndex(name),
			numaNode: NormalizeUnsetNUMA(node),
		})
	}
	return out, nil
}

func isSysfsNvidiaGPU(pciDevDir string) bool {
	vendorBytes, err := os.ReadFile(filepath.Join(pciDevDir, "vendor"))
	if err != nil {
		return false
	}
	vendor := strings.ToLower(strings.TrimSpace(string(vendorBytes)))
	if vendor != "0x10de" {
		return false
	}
	classBytes, err := os.ReadFile(filepath.Join(pciDevDir, "class"))
	if err != nil {
		return false
	}
	class := strings.ToLower(strings.TrimSpace(string(classBytes)))
	return class == "0x030000" || class == "0x030200" ||
		strings.HasPrefix(class, "0x0300") || strings.HasPrefix(class, "0x0302")
}

// DiscoverHostTopology discovers NVIDIA GPUs and InfiniBand HCAs under fsRoot
// using procfs (<fsRoot>/proc/driver/nvidia/gpus/*) and sysfs
// (<fsRoot>/sys/bus/pci/devices/* and <fsRoot>/sys/class/infiniband/mlx5_*),
// pairing each GPU with the InfiniBand HCA on the same NUMA node (or the
// lowest-index HCA as fallback) without initializing CUDA.
func DiscoverHostTopology(fsRoot string) ([]DeviceTopology, error) {
	root := resolveFSRoot(fsRoot)
	seenPCI := make(map[string]struct{})
	var pciBusIDs []string

	procGPUDir := filepath.Join(root, "proc", "driver", "nvidia", "gpus")
	if entries, err := os.ReadDir(procGPUDir); err == nil {
		for _, e := range entries {
			if !e.IsDir() || strings.HasPrefix(e.Name(), ".") {
				continue
			}
			id := e.Name()
			if err := validateTopologyIdent("PCI bus ID", id); err != nil {
				return nil, err
			}
			if _, ok := seenPCI[id]; !ok {
				seenPCI[id] = struct{}{}
				pciBusIDs = append(pciBusIDs, id)
			}
		}
	}

	sysPCIDir := filepath.Join(root, "sys", "bus", "pci", "devices")
	if entries, err := os.ReadDir(sysPCIDir); err == nil {
		for _, e := range entries {
			id := e.Name()
			if strings.HasPrefix(id, ".") {
				continue
			}
			if _, ok := seenPCI[id]; ok {
				continue
			}
			if isSysfsNvidiaGPU(filepath.Join(sysPCIDir, id)) {
				if err := validateTopologyIdent("PCI bus ID", id); err != nil {
					return nil, err
				}
				seenPCI[id] = struct{}{}
				pciBusIDs = append(pciBusIDs, id)
			}
		}
	}

	slices.Sort(pciBusIDs)
	if len(pciBusIDs) == 0 {
		return nil, fmt.Errorf("cudacluster: no nvidia gpus discovered under %q: %w", root, ErrHardwarePending)
	}

	ibs, err := discoverIBEndpoints(fsRoot)
	if err != nil {
		return nil, err
	}

	topologies := make([]DeviceTopology, len(pciBusIDs))
	for i, busID := range pciBusIDs {
		numaNode, err := resolveGPUNUMANode(fsRoot, busID)
		if err != nil {
			return nil, err
		}
		nearestIB := ""
		ibNUMA := -1
		if len(ibs) > 0 {
			chosen := ibs[0]
			for _, candidate := range ibs {
				if candidate.numaNode == numaNode {
					chosen = candidate
					break
				}
			}
			nearestIB = chosen.name
			ibNUMA = chosen.numaNode
		}
		topologies[i] = DeviceTopology{
			DeviceIndex:  i,
			PCIBusID:     busID,
			NUMANode:     numaNode,
			NearestIBDev: nearestIB,
			IBNUMANode:   ibNUMA,
		}
	}
	return topologies, nil
}

func resolveGPUNUMANode(fsRoot, pciBusID string) (int, error) {
	procNode, procErr := ReadProcNvidiaGPUNUMANode(fsRoot, pciBusID)
	if procErr == nil && procNode >= 0 {
		return procNode, nil
	}
	sysNode, sysErr := ReadSysfsPCINUMANode(fsRoot, pciBusID)
	if sysErr == nil {
		return NormalizeUnsetNUMA(sysNode), nil
	}
	if procErr == nil {
		return NormalizeUnsetNUMA(procNode), nil
	}
	return 0, fmt.Errorf("cudacluster: resolve NUMA node for GPU %s: %w", pciBusID, procErr)
}
