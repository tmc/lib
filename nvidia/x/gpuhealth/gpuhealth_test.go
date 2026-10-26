package gpuhealth

import (
	"context"
	"io/fs"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"syscall"
	"testing"
)

func smiXML(gpus ...string) string {
	return `<?xml version="1.0" ?>
<!DOCTYPE nvidia_smi_log SYSTEM "nvsmi_device_v12.dtd">
<nvidia_smi_log>
	<driver_version>580.95.05</driver_version>
	<attached_gpus>1</attached_gpus>
` + strings.Join(gpus, "\n") + `
</nvidia_smi_log>
`
}

func gpu(id, name, uuid, minor string) string {
	return `<gpu id="` + id + `"><product_name>` + name + `</product_name><uuid>` + uuid +
		`</uuid><minor_number>` + minor + `</minor_number></gpu>`
}

func TestParseSMIXML(t *testing.T) {
	tests := []struct {
		name    string
		in      string
		want    []GPU
		wantErr bool
	}{
		{"nonzero minor", smiXML(gpu("00000000:19:00.0", "NVIDIA H200", "GPU-1234-abcd", "7")),
			[]GPU{{"00000000:19:00.0", "NVIDIA H200", "GPU-1234-abcd", 7}}, false},
		{"two gpus", smiXML(gpu("a", "A", "GPU-a", "0"), gpu("b", "B", "GPU-b", "1")),
			[]GPU{{"a", "A", "GPU-a", 0}, {"b", "B", "GPU-b", 1}}, false},
		{"no minor", smiXML(gpu("a", "NVIDIA GB10", "GPU-a", "N/A")), []GPU{{"a", "NVIDIA GB10", "GPU-a", -1}}, false},
		{"minor 255", smiXML(gpu("a", "A", "GPU-a", "255")), []GPU{{"a", "A", "GPU-a", -1}}, false},
		{"minor path", smiXML(gpu("a", "A", "GPU-a", "../7")), []GPU{{"a", "A", "GPU-a", -1}}, false},
		{"minor fraction", smiXML(gpu("a", "A", "GPU-a", "1.0")), []GPU{{"a", "A", "GPU-a", -1}}, false},
		{"minor negative", smiXML(gpu("a", "A", "GPU-a", "-1")), []GPU{{"a", "A", "GPU-a", -1}}, false},
		{"not xml", "Failed to initialize NVML: Unknown Error\n", nil, true},
		{"wrong root", "<other/>", nil, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ParseSMIXML([]byte(tt.in))
			if (err != nil) != tt.wantErr {
				t.Fatalf("ParseSMIXML() error = %v, wantErr %v", err, tt.wantErr)
			}
			if !tt.wantErr && !reflect.DeepEqual(got, tt.want) {
				t.Errorf("ParseSMIXML() = %+v, want %+v", got, tt.want)
			}
		})
	}
}

func TestSelect(t *testing.T) {
	gpus := []GPU{{UUID: "GPU-a", Minor: 3}, {UUID: "GPU-b", Minor: -1}, {UUID: "GPU-c", Minor: 4}, {UUID: "GPU-c", Minor: 5}}
	if g, err := Select(gpus, "GPU-a"); err != nil || g.DevicePath() != "/dev/nvidia3" {
		t.Errorf("Select(GPU-a) = %+v, %v", g, err)
	}
	for _, uuid := range []string{"GPU-b", "GPU-c", "GPU-x"} {
		if _, err := Select(gpus, uuid); err == nil {
			t.Errorf("Select(%s) succeeded", uuid)
		}
	}
}

func TestDockerArgs(t *testing.T) {
	got, err := DockerArgs(GPU{UUID: "GPU-1", Minor: 7})
	if err != nil {
		t.Fatal(err)
	}
	want := []string{
		"--device", "/dev/nvidia7:/dev/nvidia7:rw",
		"--device", "/dev/nvidiactl:/dev/nvidiactl:rw",
		"--device", "/dev/nvidia-uvm:/dev/nvidia-uvm:rw",
		"--device", "/dev/nvidia-uvm-tools:/dev/nvidia-uvm-tools:rw",
		"-e", "NVIDIA_VISIBLE_DEVICES=none",
		"-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
		"-e", "CUDA_VISIBLE_DEVICES=GPU-1",
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("DockerArgs() = %q", got)
	}
	if _, err := DockerArgs(GPU{UUID: "GPU-1", Minor: -1}); err == nil {
		t.Error("DockerArgs(no minor) succeeded")
	}
}

func TestExplainNode(t *testing.T) {
	tests := []struct {
		n    Node
		want string
	}{
		{Node{Errno: int(syscall.ENOENT), Error: "no such file"}, "not mapped into this container"},
		{Node{Exists: true, Mode: 0o644}, "not a character device"},
		{Node{Exists: true, CharDevice: true, Mode: fs.ModeDevice | fs.ModeCharDevice | 0o666, Errno: int(syscall.EPERM), Error: "operation not permitted"},
			"device cgroup no longer admits this node"},
		{Node{Exists: true, CharDevice: true, Mode: fs.ModeDevice | fs.ModeCharDevice | 0o600, Errno: int(syscall.EACCES), Error: "permission denied"},
			"denied (permission denied) by mode"},
		{Node{Exists: true, CharDevice: true, Mode: fs.ModeDevice | fs.ModeCharDevice | 0o666, Errno: int(syscall.ENXIO), Error: "no such device or address"},
			"fell off the bus"},
	}
	for _, tt := range tests {
		if got := explainNode(tt.n); !strings.Contains(got, tt.want) {
			t.Errorf("explainNode(%+v) = %q, want %q", tt.n, got, tt.want)
		}
	}
}

// fakeRoot returns a root whose /dev nodes are symlinks to /dev/null, which
// is a character device anyone can open read-write.
func fakeRoot(t *testing.T, nodes ...string) string {
	t.Helper()
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "dev"), 0o755); err != nil {
		t.Fatal(err)
	}
	for _, n := range nodes {
		if err := os.Symlink("/dev/null", filepath.Join(root, n)); err != nil {
			t.Fatal(err)
		}
	}
	os.MkdirAll(filepath.Join(root, "proc/driver/nvidia"), 0o755)
	os.MkdirAll(filepath.Join(root, "proc/self"), 0o755)
	os.WriteFile(filepath.Join(root, "proc/driver/nvidia/version"), []byte("NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  580.95.05\nGCC version: gcc 13\n"), 0o644)
	os.WriteFile(filepath.Join(root, "proc/self/cgroup"), []byte("0::/system.slice/docker-abc.scope\n"), 0o644)
	return root
}

func fakeSMI(t *testing.T, script string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "nvidia-smi")
	if err := os.WriteFile(p, []byte("#!/bin/sh\n"+script), 0o755); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestProbe(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("needs /bin/sh and /dev/null")
	}
	xmlOut := smiXML(gpu("00000000:19:00.0", "NVIDIA H200", "GPU-1234-abcd", "7"))
	okSMI := fakeSMI(t, "cat <<'EOF'\n"+xmlOut+"EOF\n")
	all := []string{"dev/nvidia7", "dev/nvidiactl", "dev/nvidia-uvm", "dev/nvidia-uvm-tools"}
	tests := []struct {
		name  string
		smi   string
		uuid  string
		nodes []string
		want  []string // substrings of Problems, in order; none means OK
	}{
		{"healthy", okSMI, "GPU-1234-abcd", all, nil},
		{"healthy without uuid", okSMI, "", all, nil},
		{"node missing", okSMI, "GPU-1234-abcd", all[1:], []string{"/dev/nvidia7: absent"}},
		{"wrong uuid", okSMI, "GPU-ffff", all[1:], []string{"no GPU with UUID GPU-ffff"}},
		{"nvml lost", fakeSMI(t, "echo 'Failed to initialize NVML: Unknown Error'; exit 255\n"), "GPU-1234-abcd", all,
			[]string{"NVML failed through nvidia-smi (exit 255): Failed to initialize NVML: Unknown Error"}},
		{"nvidia-smi missing", "/nonexistent/nvidia-smi", "", all[1:], []string{"NVML failed through nvidia-smi (exit -1)"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			root := fakeRoot(t, tt.nodes...)
			r := Probe(context.Background(), Options{UUID: tt.uuid, SMI: tt.smi, Root: root})
			if r.OK != (len(tt.want) == 0) || len(r.Problems) != len(tt.want) {
				t.Fatalf("Probe() OK=%v problems=%q, want %q", r.OK, r.Problems, tt.want)
			}
			for i, w := range tt.want {
				if !strings.Contains(r.Problems[i], w) {
					t.Errorf("problem %d = %q, want %q", i, r.Problems[i], w)
				}
			}
			if !strings.HasPrefix(r.Driver, "NVRM version") || !strings.Contains(r.Cgroup, "docker") {
				t.Errorf("driver %q cgroup %q", r.Driver, r.Cgroup)
			}
		})
	}
}
