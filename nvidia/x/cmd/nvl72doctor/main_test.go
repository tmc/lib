package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// rackNode returns constructed facts for a healthy GB300 NVL72 compute
// tray. They are not a capture from hardware.
func rackNode() Facts {
	f := Facts{
		Schema: 1, Host: "tray01", OS: "linux/arm64",
		NVMLVersion: "13.580.0", Driver: "580.0", CUDADriverVersion: 13000,
		NVCC:         Tool{Path: "/usr/local/cuda/bin/nvcc", Version: "13.4"},
		IMEXChannels: []string{"channel0"},
		IMEXDomain:   &IMEXDomain{Status: "UP", Nodes: map[string]string{"0": "READY", "1": "READY"}},
		FabricManager: FabricManager{
			Error: "connect 127.0.0.1:6666: connection refused",
		},
	}
	for i := range 4 {
		g := GPU{Index: i, Name: "NVIDIA GB300", Major: 10, Minor: 3,
			Fabric: &Fabric{ClusterUUID: "0a0b0c0d-0000-0000-0000-000000000001", CliqueID: 7, State: 3}}
		for l := range 18 {
			g.Links = append(g.Links, Link{Link: l, Active: true, Version: 7, Remote: "switch"})
		}
		f.GPUs = append(f.GPUs, g)
	}
	for _, op := range []string{"all_reduce", "alltoall"} {
		f.NCCL = append(f.NCCL,
			Collective{Op: op, Ranks: 4, Bytes: 1 << 10, Comparisons: 1024, NCCLVersion: 22903},
			Collective{Op: op, Ranks: 4, Bytes: 1 << 28, Comparisons: 1024, NCCLVersion: 22903, BusGBs: 700})
	}
	f.Compile = []Compile{{Kernel: "tile_bf16", Arch: "sm_103a", Entries: 1, MaxRegisters: 95, DynamicShared: 110592}}
	return f
}

// statuses maps each check name to its status.
func statuses(cs []Check) map[string]string {
	m := map[string]string{}
	for _, c := range cs {
		m[c.Name] = c.Status
	}
	return m
}

func TestEvaluateRackNode(t *testing.T) {
	for _, c := range Evaluate(rackNode(), profiles["gb300-nvl72"]) {
		want := "pass"
		switch c.Name {
		case "fabric manager":
			want = "info"
		case "nccl cross-tray":
			want = "skip"
		}
		if c.Status != want {
			t.Errorf("%s: %s, want %s (%s)", c.Name, c.Status, want, c.Detail)
		}
	}
}

// TestEvaluateFaults breaks one fact of a healthy rack node at a time and
// checks that the checks that read it, and only those, fail. A GPU without
// a completed fabric probe also fails the clique check.
func TestEvaluateFaults(t *testing.T) {
	tests := []struct {
		name  string
		check string
		edit  func(*Facts)
	}{
		{"old driver", "driver", func(f *Facts) { f.CUDADriverVersion = 12080 }},
		{"no nvml", "driver", func(f *Facts) { f.NVMLError = "cannot open libnvidia-ml" }},
		{"old toolkit", "toolkit", func(f *Facts) { f.NVCC.Version = "12.8" }},
		{"three gpus", "gpus", func(f *Facts) { f.GPUs = f.GPUs[:3] }},
		{"link down", "nvlink gpu2", func(f *Facts) { f.GPUs[2].Links[5].Active = false }},
		{"link to gpu", "nvlink gpu1", func(f *Facts) { f.GPUs[1].Links[0].Remote = "gpu" }},
		{"no nvlink", "nvlink gpu0", func(f *Facts) { f.GPUs[0].Links, f.GPUs[0].LinkError = nil, "Not Supported" }},
		{"nvlink 4", "nvlink gpu0", func(f *Facts) { f.GPUs[0].Links[3].Version = 6 }},
		{"fabric in progress", "fabric gpu3,mnnvl clique", func(f *Facts) { f.GPUs[3].Fabric.State = 2 }},
		{"fabric status", "fabric gpu0", func(f *Facts) { f.GPUs[0].Fabric.Status = 999 }},
		{"fabric missing", "fabric gpu1,mnnvl clique", func(f *Facts) { f.GPUs[1].Fabric, f.GPUs[1].FabricError = nil, "Not Supported" }},
		{"clique split", "mnnvl clique", func(f *Facts) { f.GPUs[2].Fabric.CliqueID = 8 }},
		{"zero cluster", "mnnvl clique", func(f *Facts) {
			for i := range f.GPUs {
				f.GPUs[i].Fabric.ClusterUUID = zeroUUID
			}
		}},
		{"no channels", "imex channels", func(f *Facts) { f.IMEXChannels, f.IMEXChannelsError = nil, "no such file" }},
		{"imex degraded", "imex domain", func(f *Facts) { f.IMEXDomain.Status = "DEGRADED" }},
		{"imex node", "imex domain", func(f *Facts) { f.IMEXDomain.Nodes["1"] = "UNAVAILABLE" }},
		{"imex link", "imex domain", func(f *Facts) { f.IMEXDomain.Unconnected = []string{"0->1 DISCONNECTED"} }},
		{"no imex", "imex domain", func(f *Facts) { f.IMEXDomain, f.IMEXDomainError = nil, "nvidia-imex-ctl not found" }},
		{"fm failures", "fabric manager", func(f *Facts) { f.FabricManager = FabricManager{FailedSwitches: 1} }},
		{"nccl wrong", "nccl all_reduce", func(f *Facts) { f.NCCL[1].Wrong = 3 }},
		{"nccl unchecked", "nccl alltoall", func(f *Facts) { f.NCCL[3].Comparisons = 0 }},
		{"nccl one rank", "nccl alltoall", func(f *Facts) { f.NCCL[2].Ranks, f.NCCL[3].Ranks = 1, 1 }},
		{"nccl error", "nccl", func(f *Facts) { f.NCCL, f.NCCLError = nil, "timed out" }},
		{"spill", "compile tile_bf16 sm_103a", func(f *Facts) { f.Compile[0].SpillBytes = 8 }},
		{"smem", "compile tile_bf16 sm_103a", func(f *Facts) { f.Compile[0].DynamicShared = 240 << 10 }},
		{"no entries", "compile tile_bf16 sm_103a", func(f *Facts) { f.Compile[0].Entries = 0 }},
		{"ptxas error", "compile tile_bf16 sm_103a", func(f *Facts) { f.Compile[0].Error = "ptxas error" }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := rackNode()
			tt.edit(&f)
			var failed []string
			for _, c := range Evaluate(f, profiles["gb300-nvl72"]) {
				if c.Status == "fail" {
					failed = append(failed, c.Name)
				}
			}
			if got := strings.Join(failed, ","); got != tt.check {
				t.Errorf("failed checks %q, want %q", got, tt.check)
			}
		})
	}
}

// TestEvaluateGB10 checks that a GB10 skips the rack checks instead of
// failing them, and that a GB200 profile rejects a GB300 node.
func TestEvaluateGB10(t *testing.T) {
	f := Facts{
		NVMLVersion: "13.580.173.02", Driver: "580.173.02", CUDADriverVersion: 13000,
		NVCC: Tool{Path: "/usr/local/cuda/bin/nvcc", Version: "13.4"},
		GPUs: []GPU{{Index: 0, Name: "NVIDIA GB10", Major: 12, Minor: 1, LinkError: "Not Supported", FabricError: "Not Supported"}},
		NCCL: []Collective{
			{Op: "all_reduce", Ranks: 1, Bytes: 1024, Comparisons: 256},
			{Op: "alltoall", Ranks: 1, Bytes: 1024, Comparisons: 256},
		},
		IMEXChannelsError: "no such file", IMEXDomainError: "nvidia-imex-ctl not found",
		FabricManager: FabricManager{Error: "cannot open libnvfm"},
	}
	got := statuses(Evaluate(f, profiles["gb300-nvl72"]))
	want := map[string]string{
		"driver": "pass", "toolkit": "pass", "gpus": "warn",
		"nvlink gpu0": "skip", "fabric gpu0": "skip", "mnnvl clique": "skip",
		"imex channels": "skip", "imex domain": "skip", "fabric manager": "skip",
		"nccl all_reduce": "pass", "nccl alltoall": "pass", "nccl cross-tray": "skip",
	}
	for name, s := range want {
		if got[name] != s {
			t.Errorf("%s: %s, want %s", name, got[name], s)
		}
	}
	if s := statuses(Evaluate(rackNode(), profiles["gb200-nvl72"]))["gpus"]; s != "warn" {
		t.Errorf("gb300 node under gb200 profile: gpus %s, want warn", s)
	}
}

func TestParseIMEX(t *testing.T) {
	data, err := os.ReadFile("../../imex/testdata/dcgm-spec-580.30.json")
	if err != nil {
		t.Fatal(err)
	}
	d, msg := parseIMEX(data)
	if d == nil {
		t.Fatal(msg)
	}
	if d.Status != "DEGRADED" || d.Nodes["1"] != "READY" || d.Nodes["3"] != "READY" || len(d.Unconnected) != 0 {
		t.Errorf("parseIMEX = %+v", d)
	}
	if d, _ := parseIMEX([]byte("{}")); d != nil {
		t.Errorf("parseIMEX accepted an empty object: %+v", d)
	}
}

func TestIMEXChannels(t *testing.T) {
	root := t.TempDir()
	if names, msg := imexChannels(root); names != nil || msg == "" {
		t.Errorf("missing dir: %q, %q", names, msg)
	}
	dir := filepath.Join(root, "dev", "nvidia-caps-imex-channels")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, n := range []string{"channel1", "channel0", "other"} {
		if err := os.WriteFile(filepath.Join(dir, n), nil, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	names, msg := imexChannels(root)
	if strings.Join(names, ",") != "channel0,channel1" || msg != "" {
		t.Errorf("imexChannels = %q, %q", names, msg)
	}
}

// TestKernelsEmit checks that every kernel in the compile suite emits for
// the archs it targets and fits their shared memory, without a compiler.
func TestKernelsEmit(t *testing.T) {
	for _, arch := range []string{"sm_100a", "sm_103a", "sm_121a"} {
		n := 0
		for _, k := range kernels() {
			src, shared, ok, err := k.emit(arch)
			if !ok {
				continue
			}
			n++
			if err != nil || !strings.Contains(src, "__global__") {
				t.Errorf("%s %s: err %v, %d bytes", k.name, arch, err, len(src))
			}
			if limit := sharedPerBlock[strings.TrimSuffix(arch, "a")]; shared > limit {
				t.Errorf("%s %s: %d bytes shared exceeds %d", k.name, arch, shared, limit)
			}
		}
		if n < 4 {
			t.Errorf("%s: %d kernels, want at least 4", arch, n)
		}
	}
}

func TestCompileArchs(t *testing.T) {
	p := profiles["gb300-nvl72"]
	tests := []struct {
		list string
		gpus []GPU
		want string
	}{
		{"", []GPU{{Major: 12, Minor: 1}}, "sm_121a,sm_103a"},
		{"", []GPU{{Major: 10, Minor: 3}, {Major: 10, Minor: 3}}, "sm_103a"},
		{"", nil, "sm_103a"},
		{"sm_100a,sm_90a", []GPU{{Major: 12, Minor: 1}}, "sm_100a,sm_90a"},
	}
	for _, tt := range tests {
		if got := strings.Join(compileArchs(tt.list, Facts{GPUs: tt.gpus}, p), ","); got != tt.want {
			t.Errorf("compileArchs(%q, %v) = %s, want %s", tt.list, tt.gpus, got, tt.want)
		}
	}
}

// TestFacts evaluates saved facts: the constructed rack node and the same
// node with a link down. testdata/script/gb10.txt evaluates a GB10 capture.
func TestFacts(t *testing.T) {
	dir := t.TempDir()
	broken := rackNode()
	broken.GPUs[0].Links[0].Active = false
	tests := []struct {
		name string
		f    Facts
		code int
	}{
		{"rack", rackNode(), 0},
		{"broken", broken, 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			file := filepath.Join(dir, tt.name+".json")
			data, _ := json.Marshal(tt.f)
			if err := os.WriteFile(file, data, 0o644); err != nil {
				t.Fatal(err)
			}
			var out, errout bytes.Buffer
			if code := run([]string{"-facts", file}, &out, &errout); code != tt.code {
				t.Errorf("exit %d, want %d\n%s%s", code, tt.code, &out, &errout)
			}
		})
	}
}

func TestArguments(t *testing.T) {
	for _, args := range [][]string{{"-profile=unknown"}, {"-timeout=0"}, {"extra"}, {"-probe=unknown"}, {"-nccl-max-bytes=10"}} {
		var out, errout bytes.Buffer
		if code := run(args, &out, &errout); code != 2 {
			t.Errorf("%q: exit %d, want 2", args, code)
		}
	}
}
