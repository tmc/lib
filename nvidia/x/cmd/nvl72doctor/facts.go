package main

// Facts is everything nvl72doctor observed on one node. It is written by
// -save-facts and read by -facts, so a capture from a rack node can be
// re-evaluated elsewhere. Error fields hold the reason a fact is missing;
// an empty error with a zero value is a reported zero.
type Facts struct {
	Schema int    `json:"schema"`
	Host   string `json:"host,omitempty"`
	OS     string `json:"os"`

	NVMLError         string `json:"nvml_error,omitempty"`
	NVMLVersion       string `json:"nvml_version,omitempty"`
	Driver            string `json:"driver,omitempty"`
	CUDADriverVersion int    `json:"cuda_driver_version,omitempty"` // 13000 is CUDA 13.0

	NVCC Tool  `json:"nvcc"`
	GPUs []GPU `json:"gpus,omitempty"`

	IMEXChannels      []string    `json:"imex_channels,omitempty"`
	IMEXChannelsError string      `json:"imex_channels_error,omitempty"`
	IMEXDomain        *IMEXDomain `json:"imex_domain,omitempty"`
	IMEXDomainError   string      `json:"imex_domain_error,omitempty"`

	FabricManager FabricManager `json:"fabric_manager"`

	NCCL      []Collective `json:"nccl,omitempty"`
	NCCLError string       `json:"nccl_error,omitempty"`
	Compile   []Compile    `json:"compile,omitempty"`
}

// A Tool is an executable found and asked for its version.
type Tool struct {
	Path    string `json:"path,omitempty"`
	Version string `json:"version,omitempty"`
	Error   string `json:"error,omitempty"`
}

// A GPU is one device as NVML reports it.
type GPU struct {
	Index       int     `json:"index"`
	UUID        string  `json:"uuid,omitempty"`
	Name        string  `json:"name,omitempty"`
	BusID       string  `json:"bus_id,omitempty"`
	Major       int     `json:"major"`
	Minor       int     `json:"minor"`
	Error       string  `json:"error,omitempty"`
	Links       []Link  `json:"links,omitempty"`
	LinkError   string  `json:"link_error,omitempty"`
	Fabric      *Fabric `json:"fabric,omitempty"`
	FabricError string  `json:"fabric_error,omitempty"`
}

// Arch returns the GPU's sm_XY name, or "" if its capability is unknown.
func (g GPU) Arch() string {
	if g.Major == 0 {
		return ""
	}
	return smName(g.Major, g.Minor)
}

// A Link is one NVLink port. Version is NVML's nvmlNvlinkVersion_t code
// (7 is NVLink 5.0); Remote is "gpu", "switch", "ibmnpu" or "unknown".
type Link struct {
	Link    int    `json:"link"`
	Active  bool   `json:"active"`
	Version int    `json:"version,omitempty"`
	Remote  string `json:"remote,omitempty"`
	Error   string `json:"error,omitempty"`
}

// Fabric is the GPU's nvmlGpuFabricInfo_v3: its multi-node NVLink
// (MNNVL) cluster, clique and probe state.
type Fabric struct {
	ClusterUUID   string `json:"cluster_uuid"`
	CliqueID      uint32 `json:"clique_id"`
	State         int    `json:"state"`  // nvmlGpuFabricState_t; 3 is completed
	Status        int    `json:"status"` // nvmlReturn_t of the fabric probe
	HealthMask    uint32 `json:"health_mask"`
	HealthSummary uint8  `json:"health_summary"`
}

// IMEXDomain is the parsed nvidia-imex-ctl -N -j snapshot. Unconnected
// lists each directed connection whose status is not CONNECTED, as
// "from->to STATUS".
type IMEXDomain struct {
	Status      string            `json:"status"`
	Nodes       map[string]string `json:"nodes"` // node ID -> reported status
	Unconnected []string          `json:"unconnected,omitempty"`
}

// FabricManager records an attempt to reach a Fabric Manager instance.
// On GB200 and GB300 NVL72 compute trays Fabric Manager normally does not
// run locally, and fabric state is read through NVML instead.
type FabricManager struct {
	Library        string `json:"library,omitempty"`
	Error          string `json:"error,omitempty"`
	Partitions     int    `json:"partitions"`
	FailedGPUs     int    `json:"failed_gpus"`
	FailedSwitches int    `json:"failed_switches"`
}

// A Collective is one size of one NCCL sweep, checked against a host oracle.
type Collective struct {
	Op          string  `json:"op"`
	Ranks       int     `json:"ranks"`
	Bytes       int     `json:"bytes"`
	TimeNanos   int64   `json:"time_ns"`
	AlgGBs      float64 `json:"alg_gbs"`
	BusGBs      float64 `json:"bus_gbs"`
	Comparisons int     `json:"comparisons"`
	Wrong       int     `json:"wrong"`
	NCCLVersion int32   `json:"nccl_version"`
	Error       string  `json:"error,omitempty"`
}

// A Compile is one emitted kernel compiled by nvcc for one architecture.
type Compile struct {
	Kernel        string  `json:"kernel"`
	Arch          string  `json:"arch"`
	Error         string  `json:"error,omitempty"`
	Entries       int     `json:"entries"`
	MaxRegisters  int     `json:"max_registers"`
	SpillBytes    int     `json:"spill_bytes"`
	StaticShared  int     `json:"static_shared"`
	DynamicShared int     `json:"dynamic_shared"`
	Seconds       float64 `json:"seconds"`
}
