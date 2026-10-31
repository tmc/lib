package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"time"
	"unsafe"

	"github.com/tmc/lib/nvidia/fabricmanager"
	"github.com/tmc/lib/nvidia/nvml"
	"github.com/tmc/lib/nvidia/x/imex"
	"github.com/tmc/lib/nvidia/x/internal/cudatool"
)

// probeNode fills the facts that come from the driver, the toolkit and the
// IMEX and Fabric Manager services. It creates no CUDA context.
func probeNode(ctx context.Context, f *Facts, nvccPath, root string) {
	f.Host, _ = os.Hostname()
	f.OS = runtime.GOOS + "/" + runtime.GOARCH
	probeNVML(f)
	f.NVCC = probeNVCC(ctx, nvccPath)
	f.IMEXChannels, f.IMEXChannelsError = imexChannels(root)
	f.IMEXDomain, f.IMEXDomainError = imexDomain(ctx)
	f.FabricManager = probeFabricManager()
}

func probeNVML(f *Facts) {
	if err := nvml.Init(); err != nil {
		f.NVMLError = err.Error()
		return
	}
	defer nvml.Shutdown()
	f.NVMLVersion, _ = nvml.SystemGetNVMLVersion()
	f.Driver, _ = nvml.SystemGetDriverVersion()
	if v, err := nvml.SystemGetCudaDriverVersion(); err == nil {
		f.CUDADriverVersion = int(v)
	}
	n, err := nvml.DeviceGetCount()
	if err != nil {
		f.NVMLError = fmt.Sprintf("device count: %v", err)
		return
	}
	for i := range n {
		g := GPU{Index: int(i)}
		d, err := nvml.NewDeviceByIndex(i)
		if err != nil {
			g.Error = err.Error()
			f.GPUs = append(f.GPUs, g)
			continue
		}
		g.UUID, _ = d.UUID()
		g.Name, _ = d.Name()
		if pci, err := d.PciInfo(); err == nil {
			g.BusID = cString(pci.BusId[:])
		}
		if major, minor, err := d.CudaComputeCapability(); err == nil {
			g.Major, g.Minor = int(major), int(minor)
		} else {
			g.Error = fmt.Sprintf("compute capability: %v", err)
		}
		g.Links, g.LinkError = probeLinks(d)
		if info, err := d.FabricInfo(); err != nil {
			g.FabricError = err.Error()
		} else {
			g.Fabric = &Fabric{
				ClusterUUID:   formatUUID(info.ClusterUuid),
				CliqueID:      info.CliqueId,
				State:         int(info.State),
				Status:        int(info.Status),
				HealthMask:    info.HealthMask,
				HealthSummary: info.HealthSummary,
			}
		}
		f.GPUs = append(f.GPUs, g)
	}
}

// probeLinks reads every NVLink port NVML will answer for. Ports past the
// device's last return an error; a device with no NVLink fails on port 0.
func probeLinks(d nvml.Device) ([]Link, string) {
	var links []Link
	for i := range uint32(nvml.NvlinkMaxLinks) {
		state, err := d.NvLinkState(i)
		if err != nil {
			if i == 0 {
				return nil, err.Error()
			}
			break
		}
		l := Link{Link: int(i), Active: state == nvml.FeatureEnabled}
		if l.Active {
			v, verr := d.NvLinkVersion(i)
			var t nvml.IntNvLinkDeviceType
			terr := d.NvLinkRemoteDeviceType(i, &t)
			l.Version, l.Remote = int(v), linkRemote(t)
			if err := errors.Join(verr, terr); err != nil {
				l.Error = err.Error()
			}
		}
		links = append(links, l)
	}
	return links, ""
}

func linkRemote(t nvml.IntNvLinkDeviceType) string {
	switch t {
	case nvml.NvlinkDeviceTypeGpu:
		return "gpu"
	case nvml.NvlinkDeviceTypeIbmnpu:
		return "ibmnpu"
	case nvml.NvlinkDeviceTypeSwitch:
		return "switch"
	}
	return "unknown"
}

func probeNVCC(ctx context.Context, path string) Tool {
	p, err := cudatool.LookTool("nvcc", path)
	if err != nil {
		return Tool{Error: err.Error()}
	}
	t := Tool{Path: p}
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	out, err := cudatool.Run(ctx, p, "", nil, "--version")
	if err != nil {
		t.Error = err.Error()
		return t
	}
	if v, ok := cudatool.ParseVersion(out.Stdout); ok {
		t.Version = v.String()
	} else {
		t.Error = "no release in nvcc --version output"
	}
	return t
}

// imexChannels lists the IMEX channel device nodes under root.
func imexChannels(root string) ([]string, string) {
	dir := filepath.Join(root, "/dev/nvidia-caps-imex-channels")
	ents, err := os.ReadDir(dir)
	if err != nil {
		return nil, err.Error()
	}
	var names []string
	for _, e := range ents {
		if strings.HasPrefix(e.Name(), "channel") {
			names = append(names, e.Name())
		}
	}
	if names == nil {
		return nil, dir + " has no channel nodes"
	}
	slices.Sort(names)
	return names, ""
}

// imexDomain runs nvidia-imex-ctl -N -j and parses its snapshot.
func imexDomain(ctx context.Context) (*IMEXDomain, string) {
	p, err := cudatool.LookTool("nvidia-imex-ctl", "")
	if err != nil {
		return nil, "nvidia-imex-ctl not found"
	}
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	out, err := cudatool.Run(ctx, p, "", nil, "-N", "-j")
	if err != nil {
		return nil, fmt.Sprintf("nvidia-imex-ctl -N -j: %v: %s", err, bytes.TrimSpace(out.Stderr))
	}
	return parseIMEX(out.Stdout)
}

func parseIMEX(data []byte) (*IMEXDomain, string) {
	s, err := imex.Read(bytes.NewReader(data), imex.DCGMSpec58030, imex.Limits{})
	if err != nil {
		return nil, err.Error()
	}
	d := &IMEXDomain{Status: s.Status, Nodes: map[string]string{}}
	for _, id := range keys(s.Nodes) {
		n := s.Nodes[id]
		d.Nodes[id] = n.Status
		for _, to := range keys(n.Connections) {
			if st := n.Connections[to].Status; st != "CONNECTED" {
				d.Unconnected = append(d.Unconnected, id+"->"+to+" "+st)
			}
		}
	}
	return d, ""
}

// probeFabricManager connects to a local Fabric Manager instance, if any,
// and counts its partitions and NVLink training failures.
func probeFabricManager() (fm FabricManager) {
	if err := fabricmanager.LibInit(); err != nil {
		fm.Error = err.Error()
		return fm
	}
	defer fabricmanager.LibShutdown()
	fm.Library = fabricmanager.Path()
	params := fabricmanager.ConnectParams{TimeoutMs: 1000}
	copy(params.AddressInfo[:], "127.0.0.1:6666")
	params.Version = uint32(unsafe.Sizeof(params)) | 1<<24
	h, err := fabricmanager.Connect(&params)
	if err != nil {
		fm.Error = fmt.Sprintf("connect 127.0.0.1:6666: %v", err)
		return fm
	}
	defer h.Disconnect()
	var list fabricmanager.FabricPartitionList
	list.Version = uint32(unsafe.Sizeof(list)) | 1<<24
	if err := h.SupportedFabricPartitions(&list); err != nil {
		fm.Error = fmt.Sprintf("partitions: %v", err)
		return fm
	}
	fm.Partitions = int(list.NumPartitions)
	var failed fabricmanager.NvlinkFailedDevices
	failed.Version = uint32(unsafe.Sizeof(failed)) | 1<<24
	if err := h.NvlinkFailedDevices(&failed); err != nil {
		fm.Error = fmt.Sprintf("failed devices: %v", err)
		return fm
	}
	fm.FailedGPUs, fm.FailedSwitches = int(failed.NumGpus), int(failed.NumSwitches)
	return fm
}

func formatUUID(b [16]uint8) string {
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:16])
}

func cString(b []byte) string {
	if i := bytes.IndexByte(b, 0); i >= 0 {
		b = b[:i]
	}
	return string(b)
}
