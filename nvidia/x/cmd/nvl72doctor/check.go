package main

import (
	"fmt"
	"slices"
	"strings"

	"github.com/tmc/lib/nvidia/x/internal/cudatool"
)

// A Profile is what one rack product requires of each compute node.
type Profile struct {
	Name        string
	Major       int // compute capability of every GPU
	Minor       int
	GPUsPerNode int
	Links       int // active NVLink ports per GPU, each to an NVSwitch
	LinkVersion int // nvmlNvlinkVersion_t; 7 is NVLink 5.0
	MinCUDA     int // minimum driver CUDA version, 12090 is 12.9
	MinToolkit  cudatool.Version
}

var profiles = map[string]Profile{
	// sm_103 first appears in CUDA 12.9; sm_100 in 12.8.
	"gb300-nvl72": {Name: "gb300-nvl72", Major: 10, Minor: 3, GPUsPerNode: 4, Links: 18, LinkVersion: 7, MinCUDA: 12090, MinToolkit: cudatool.V(12, 9)},
	"gb200-nvl72": {Name: "gb200-nvl72", Major: 10, Minor: 0, GPUsPerNode: 4, Links: 18, LinkVersion: 7, MinCUDA: 12080, MinToolkit: cudatool.V(12, 8)},
}

// Arch returns the profile's architecture-specific target, such as sm_103a.
func (p Profile) Arch() string { return smName(p.Major, p.Minor) + "a" }

// sharedPerBlock is the largest shared memory one block may use, static
// plus dynamic, in bytes (cudaDevAttrMaxSharedMemoryPerBlockOptin).
var sharedPerBlock = map[string]int{
	"sm_90":  227 << 10,
	"sm_100": 227 << 10,
	"sm_103": 227 << 10,
	"sm_120": 99 << 10,
	"sm_121": 99 << 10,
}

// A Check is one readiness verdict. Status is pass, fail, warn, skip or
// info; only fail makes the command exit nonzero. Evidence says how the
// fact was obtained: observed (queried from the driver or a daemon),
// executed (ran on this node's GPUs) or compiled (nvcc only, no GPU).
type Check struct {
	Name     string `json:"name"`
	Status   string `json:"status"`
	Evidence string `json:"evidence,omitempty"`
	Detail   string `json:"detail"`
}

// Evaluate turns facts into checks against p. A node whose GPUs are not
// p's architecture is not a rack node: checks that only a rack node can
// pass are skipped with the reason, instead of failed.
func Evaluate(f Facts, p Profile) []Check {
	var cs []Check
	add := func(name, status, evidence, format string, args ...any) {
		cs = append(cs, Check{Name: name, Status: status, Evidence: evidence, Detail: fmt.Sprintf(format, args...)})
	}

	switch {
	case f.NVMLError != "":
		add("driver", "fail", "observed", "%s", f.NVMLError)
	case f.CUDADriverVersion < p.MinCUDA:
		add("driver", "fail", "observed", "driver %s supports CUDA %s; %s needs %s", f.Driver, cudaVersion(f.CUDADriverVersion), p.Name, cudaVersion(p.MinCUDA))
	default:
		add("driver", "pass", "observed", "driver %s, CUDA %s, NVML %s", f.Driver, cudaVersion(f.CUDADriverVersion), f.NVMLVersion)
	}

	if f.NVCC.Error != "" {
		add("toolkit", "fail", "observed", "nvcc: %s", f.NVCC.Error)
	} else if v, ok := cudatool.ParseVersion([]byte("release " + f.NVCC.Version + ",")); !ok || v.Compare(p.MinToolkit) < 0 {
		add("toolkit", "fail", "observed", "nvcc %s at %s; %s needs %s or later", f.NVCC.Version, f.NVCC.Path, p.Arch(), p.MinToolkit)
	} else {
		add("toolkit", "pass", "observed", "nvcc %s at %s", f.NVCC.Version, f.NVCC.Path)
	}

	skip := notRackNode(f, p)
	switch {
	case len(f.GPUs) == 0:
		add("gpus", "fail", "observed", "no GPUs reported")
	case skip != "":
		add("gpus", "warn", "observed", "%s; checks that need a %s node are skipped", skip, p.Name)
	default:
		var bad []string
		for _, g := range f.GPUs {
			if g.Error != "" {
				bad = append(bad, fmt.Sprintf("gpu%d: %s", g.Index, g.Error))
			}
		}
		if len(f.GPUs) != p.GPUsPerNode {
			bad = append(bad, fmt.Sprintf("%d GPUs, want %d per compute tray", len(f.GPUs), p.GPUsPerNode))
		}
		if bad != nil {
			add("gpus", "fail", "observed", "%s", strings.Join(bad, "; "))
		} else {
			add("gpus", "pass", "observed", "%d x %s (%s)", len(f.GPUs), f.GPUs[0].Name, f.GPUs[0].Arch())
		}
	}

	for _, g := range f.GPUs {
		name := fmt.Sprintf("nvlink gpu%d", g.Index)
		active, problems := linkSummary(g, p)
		switch {
		case skip != "" && active == 0:
			add(name, "skip", "observed", "%s; %s", noLinks(g), skip)
		case skip != "":
			add(name, "info", "observed", "%d active links; not checked against %s", active, p.Name)
		case problems != nil:
			add(name, "fail", "observed", "%s", strings.Join(problems, "; "))
		default:
			add(name, "pass", "observed", "%d active NVLink 5 links, each to an NVSwitch", active)
		}
	}

	for _, g := range f.GPUs {
		name := fmt.Sprintf("fabric gpu%d", g.Index)
		switch fb := g.Fabric; {
		case skip != "":
			add(name, "skip", "observed", "%s; %s", orText(g.FabricError, "fabric probe not run"), skip)
		case fb == nil:
			add(name, "fail", "observed", "nvmlDeviceGetGpuFabricInfoV: %s", g.FabricError)
		case fb.State != fabricStateCompleted:
			add(name, "fail", "observed", "fabric probe state %s, want completed", fabricState(fb.State))
		case fb.Status != 0:
			add(name, "fail", "observed", "fabric probe finished with nvmlReturn_t %d", fb.Status)
		case fb.HealthMask != 0:
			add(name, "warn", "observed", "fabric health mask %#x (summary %d); see nvmlGpuFabricInfo_v3 health bits", fb.HealthMask, fb.HealthSummary)
		default:
			add(name, "pass", "observed", "probe completed; cluster %s clique %d", fb.ClusterUUID, fb.CliqueID)
		}
	}

	if skip != "" {
		add("mnnvl clique", "skip", "observed", "%s", skip)
	} else {
		clusters, cliques := map[string]bool{}, map[uint32]bool{}
		missing := 0
		for _, g := range f.GPUs {
			if g.Fabric == nil || g.Fabric.State != fabricStateCompleted {
				missing++
				continue
			}
			clusters[g.Fabric.ClusterUUID] = true
			cliques[g.Fabric.CliqueID] = true
		}
		switch {
		case missing > 0:
			add("mnnvl clique", "fail", "observed", "%d of %d GPUs report no completed fabric probe", missing, len(f.GPUs))
		case len(clusters) != 1 || len(cliques) != 1:
			add("mnnvl clique", "fail", "observed", "local GPUs disagree: clusters %v, cliques %v", keys(clusters), keys(cliques))
		case clusters[zeroUUID]:
			add("mnnvl clique", "fail", "observed", "cluster UUID is zero: GPUs are not in an NVLink domain")
		default:
			add("mnnvl clique", "pass", "observed", "all %d local GPUs in cluster %s clique %d; rack-wide membership needs every tray's report", len(f.GPUs), keys(clusters)[0], keys(cliques)[0])
		}
	}

	switch {
	case len(f.IMEXChannels) > 0:
		add("imex channels", "pass", "observed", "%s", strings.Join(f.IMEXChannels, ", "))
	case skip != "":
		add("imex channels", "skip", "observed", "%s; %s", f.IMEXChannelsError, skip)
	default:
		add("imex channels", "fail", "observed", "%s; cross-tray fabric memory needs channel0 (driver parameter NVreg_CreateImexChannel0=1, or mknod from /proc/devices nvidia-caps-imex-channels)", f.IMEXChannelsError)
	}

	switch d := f.IMEXDomain; {
	case d == nil && skip != "":
		add("imex domain", "skip", "observed", "%s; %s", f.IMEXDomainError, skip)
	case d == nil:
		add("imex domain", "fail", "observed", "%s", f.IMEXDomainError)
	default:
		var notReady []string
		for _, id := range keys(d.Nodes) {
			if d.Nodes[id] != "READY" {
				notReady = append(notReady, "node "+id+" "+d.Nodes[id])
			}
		}
		notReady = append(notReady, d.Unconnected...)
		if d.Status != "UP" || notReady != nil {
			add("imex domain", "fail", "observed", "domain %s, %d nodes; %s", d.Status, len(d.Nodes), orText(strings.Join(notReady, "; "), "all nodes ready"))
		} else {
			add("imex domain", "pass", "observed", "domain UP, %d nodes READY, all connections CONNECTED", len(d.Nodes))
		}
	}

	switch fm := f.FabricManager; {
	case fm.Error != "" && skip != "":
		add("fabric manager", "skip", "observed", "%s; %s", fm.Error, skip)
	case fm.Error != "":
		add("fabric manager", "info", "observed", "not reachable on this node (%s); on NVL72 the switch trays manage the fabric, and the fabric gpuN checks read its result through NVML", fm.Error)
	case fm.FailedGPUs > 0 || fm.FailedSwitches > 0:
		add("fabric manager", "fail", "observed", "%d GPUs and %d switches have NVLinks that failed to train", fm.FailedGPUs, fm.FailedSwitches)
	default:
		add("fabric manager", "pass", "observed", "%d supported partitions, no failed NVLink devices", fm.Partitions)
	}

	cs = append(cs, ncclChecks(f, p, skip)...)

	for _, c := range f.Compile {
		name := "compile " + c.Kernel + " " + c.Arch
		limit := sharedPerBlock[strings.TrimSuffix(c.Arch, "a")]
		switch {
		case c.Error != "":
			add(name, "fail", "compiled", "%s", c.Error)
		case c.Entries == 0:
			add(name, "fail", "compiled", "ptxas reported no entry functions")
		case c.SpillBytes > 0:
			add(name, "fail", "compiled", "%d bytes spilled", c.SpillBytes)
		case limit > 0 && c.StaticShared+c.DynamicShared > limit:
			add(name, "fail", "compiled", "shared memory %d static + %d dynamic exceeds %d per block", c.StaticShared, c.DynamicShared, limit)
		default:
			add(name, "pass", "compiled", "%d entries, max %d registers, no spills, shared %d static + %d dynamic of %d, %.1fs", c.Entries, c.MaxRegisters, c.StaticShared, c.DynamicShared, limit, c.Seconds)
		}
	}
	return cs
}

func ncclChecks(f Facts, p Profile, skip string) []Check {
	var cs []Check
	add := func(name, status, evidence, format string, args ...any) {
		cs = append(cs, Check{Name: name, Status: status, Evidence: evidence, Detail: fmt.Sprintf(format, args...)})
	}
	if f.NCCLError != "" {
		add("nccl", "fail", "executed", "%s", f.NCCLError)
	}
	for _, op := range []string{"all_reduce", "alltoall"} {
		var rs []Collective
		for _, r := range f.NCCL {
			if r.Op == op {
				rs = append(rs, r)
			}
		}
		name := "nccl " + op
		if len(rs) == 0 {
			if f.NCCLError == "" {
				add(name, "skip", "", "not run")
			}
			continue
		}
		var bad []string
		for _, r := range rs {
			switch {
			case r.Error != "":
				bad = append(bad, fmt.Sprintf("%d bytes: %s", r.Bytes, r.Error))
			case r.Wrong > 0 || r.Comparisons == 0:
				bad = append(bad, fmt.Sprintf("%d bytes: %d of %d values wrong", r.Bytes, r.Wrong, r.Comparisons))
			}
		}
		last := rs[len(rs)-1]
		switch {
		case bad != nil:
			add(name, "fail", "executed", "%s", strings.Join(bad, "; "))
		case skip == "" && last.Ranks < p.GPUsPerNode:
			add(name, "fail", "executed", "%d ranks, want all %d local GPUs", last.Ranks, p.GPUsPerNode)
		default:
			note := ""
			if last.Ranks == 1 {
				note = "; one rank exercises NCCL and CUDA, not NVLink"
			}
			add(name, "pass", "executed", "%d ranks, NCCL %d, %d sizes %d..%d bytes checked against a host oracle; at %d bytes %.1f GB/s bus, %.1f GB/s algorithm%s",
				last.Ranks, last.NCCLVersion, len(rs), rs[0].Bytes, last.Bytes, last.Bytes, last.BusGBs, last.AlgGBs, note)
		}
	}
	add("nccl cross-tray", "skip", "", "a 72-rank MNNVL collective needs one process per GPU on every tray (srun or mpirun with an NCCL unique ID); this command runs one single-process communicator over the local GPUs")
	return cs
}

const (
	fabricStateCompleted = 3
	zeroUUID             = "00000000-0000-0000-0000-000000000000"
)

// notRackNode returns why f is not a node of p's rack, or "".
func notRackNode(f Facts, p Profile) string {
	if len(f.GPUs) == 0 {
		return "no GPUs reported"
	}
	for _, g := range f.GPUs {
		if g.Major != p.Major || g.Minor != p.Minor {
			return fmt.Sprintf("gpu%d %s is %s, not %s", g.Index, g.Name, orText(g.Arch(), "of unknown capability"), smName(p.Major, p.Minor))
		}
	}
	return ""
}

// linkSummary counts g's active links and lists how they fall short of p.
func linkSummary(g GPU, p Profile) (active int, problems []string) {
	if g.LinkError != "" {
		return 0, []string{"nvlink: " + g.LinkError}
	}
	var down, version, remote []string
	for _, l := range g.Links {
		if !l.Active {
			down = append(down, fmt.Sprint(l.Link))
			continue
		}
		active++
		if l.Version != p.LinkVersion {
			version = append(version, fmt.Sprintf("%d:v%d", l.Link, l.Version))
		}
		if l.Remote != "switch" {
			remote = append(remote, fmt.Sprintf("%d:%s", l.Link, l.Remote))
		}
	}
	if active != p.Links {
		problems = append(problems, fmt.Sprintf("%d active links, want %d", active, p.Links))
	}
	if down != nil {
		problems = append(problems, "inactive links "+strings.Join(down, ","))
	}
	if version != nil {
		problems = append(problems, fmt.Sprintf("links not NVLink version code %d: %s", p.LinkVersion, strings.Join(version, ",")))
	}
	if remote != nil {
		problems = append(problems, "links not to an NVSwitch: "+strings.Join(remote, ","))
	}
	return active, problems
}

func noLinks(g GPU) string {
	if g.LinkError != "" {
		return "nvlink: " + g.LinkError
	}
	return "no active NVLink ports"
}

func smName(major, minor int) string { return fmt.Sprintf("sm_%d%d", major, minor) }

func cudaVersion(v int) string { return fmt.Sprintf("%d.%d", v/1000, v%1000/10) }

func fabricState(s int) string {
	switch s {
	case 0:
		return "not supported"
	case 1:
		return "not started"
	case 2:
		return "in progress"
	case 3:
		return "completed"
	}
	return fmt.Sprintf("unknown (%d)", s)
}

func orText(s, alt string) string {
	if s == "" {
		return alt
	}
	return s
}

func keys[K interface{ ~string | ~uint32 }, V any](m map[K]V) []K {
	ks := make([]K, 0, len(m))
	for k := range m {
		ks = append(ks, k)
	}
	slices.Sort(ks)
	return ks
}
