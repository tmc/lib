package gpuhealth

import (
	"bytes"
	"context"
	"encoding/xml"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"syscall"
	"time"
)

// A Node is what opening one device node showed.
type Node struct {
	Path       string      `json:"path"`
	Exists     bool        `json:"exists"`
	CharDevice bool        `json:"char_device"`
	Mode       fs.FileMode `json:"mode"`
	Major      uint32      `json:"major,omitempty"`
	Minor      uint32      `json:"minor,omitempty"`
	OpenOK     bool        `json:"open_ok"`
	Errno      int         `json:"errno,omitempty"` // errno of stat or open, when known
	Error      string      `json:"error,omitempty"`
}

// ControlNodes are the nodes every CUDA process needs besides its GPU's own.
var ControlNodes = []string{"/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools"}

// ProbeNodes stats and opens each path read-write and closes it again.
// root, if not empty, is prefixed to every path, for tests.
func ProbeNodes(root string, paths []string) []Node {
	nodes := make([]Node, len(paths))
	for i, p := range paths {
		n := Node{Path: p}
		full := filepath.Join(root, p)
		fi, err := os.Stat(full)
		if err != nil {
			n.Errno, n.Error = errnoOf(err), err.Error()
			nodes[i] = n
			continue
		}
		n.Exists = true
		n.Mode = fi.Mode()
		n.CharDevice = fi.Mode()&fs.ModeCharDevice != 0
		n.Major, n.Minor = deviceNumbers(fi)
		if !n.CharDevice {
			n.Errno, n.Error = int(syscall.ENODEV), "not a character device"
			nodes[i] = n
			continue
		}
		f, err := os.OpenFile(full, os.O_RDWR, 0)
		if err != nil {
			n.Errno, n.Error = errnoOf(err), err.Error()
		} else {
			f.Close()
			n.OpenOK = true
		}
		nodes[i] = n
	}
	return nodes
}

func errnoOf(err error) int {
	var e syscall.Errno
	if errors.As(err, &e) {
		return int(e)
	}
	return 0
}

// A GPU is one <gpu> element of nvidia-smi -q -x.
type GPU struct {
	PCIBusID string `json:"pci_bus_id"`
	Name     string `json:"name"`
	UUID     string `json:"uuid"`
	Minor    int    `json:"minor"` // -1 when the XML has none, as on some platforms
}

// DevicePath returns /dev/nvidia<minor>, or "" if the minor is unknown.
func (g GPU) DevicePath() string {
	if g.Minor < 0 {
		return ""
	}
	return "/dev/nvidia" + strconv.Itoa(g.Minor)
}

var minorRE = regexp.MustCompile(`^[0-9]{1,3}$`)

// ParseSMIXML parses the output of nvidia-smi -q -x. A minor_number that is
// not a decimal in 0..254 (such as "N/A") gives Minor -1.
func ParseSMIXML(data []byte) ([]GPU, error) {
	var log struct {
		XMLName xml.Name `xml:"nvidia_smi_log"`
		GPUs    []struct {
			ID    string `xml:"id,attr"`
			Name  string `xml:"product_name"`
			UUID  string `xml:"uuid"`
			Minor string `xml:"minor_number"`
		} `xml:"gpu"`
	}
	// nvidia-smi declares a DTD the decoder cannot fetch and does not need.
	d := xml.NewDecoder(bytes.NewReader(data))
	d.Strict = true
	if err := d.Decode(&log); err != nil {
		return nil, fmt.Errorf("gpuhealth: parse nvidia-smi xml: %w", err)
	}
	gpus := make([]GPU, len(log.GPUs))
	for i, g := range log.GPUs {
		gpus[i] = GPU{PCIBusID: g.ID, Name: strings.TrimSpace(g.Name), UUID: strings.TrimSpace(g.UUID), Minor: -1}
		m := strings.TrimSpace(g.Minor)
		if minorRE.MatchString(m) {
			if n, _ := strconv.Atoi(m); n < 255 {
				gpus[i].Minor = n
			}
		}
	}
	return gpus, nil
}

// Select returns the one GPU with the given UUID.
func Select(gpus []GPU, uuid string) (GPU, error) {
	var found []GPU
	for _, g := range gpus {
		if g.UUID == uuid {
			found = append(found, g)
		}
	}
	switch {
	case len(found) == 0:
		return GPU{}, fmt.Errorf("gpuhealth: no GPU with UUID %s among %d", uuid, len(gpus))
	case len(found) > 1:
		return GPU{}, fmt.Errorf("gpuhealth: %d GPUs with UUID %s", len(found), uuid)
	case found[0].Minor < 0:
		return GPU{}, fmt.Errorf("gpuhealth: GPU %s has no device minor number", uuid)
	}
	return found[0], nil
}

// DockerArgs returns docker run arguments that map exactly g's device node
// and the control nodes read-write and make g the only CUDA device. It sets
// NVIDIA_VISIBLE_DEVICES=none so the NVIDIA runtime injects driver libraries
// but no devices of its own.
func DockerArgs(g GPU) ([]string, error) {
	dev := g.DevicePath()
	if dev == "" {
		return nil, fmt.Errorf("gpuhealth: GPU %s has no device minor number", g.UUID)
	}
	var args []string
	for _, n := range append([]string{dev}, ControlNodes...) {
		args = append(args, "--device", n+":"+n+":rw")
	}
	return append(args,
		"-e", "NVIDIA_VISIBLE_DEVICES=none",
		"-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
		"-e", "CUDA_VISIBLE_DEVICES="+g.UUID,
	), nil
}

// Options configures Probe.
type Options struct {
	UUID    string        // GPU expected; empty accepts exactly one GPU
	SMI     string        // nvidia-smi path; default "nvidia-smi"
	Timeout time.Duration // limit for nvidia-smi; default 5s
	Root    string        // prefix for /dev and /proc paths, for tests
}

// SMIResult is one nvidia-smi run.
type SMIResult struct {
	ExitCode int    `json:"exit_code"` // -1 when it could not run or timed out
	Error    string `json:"error,omitempty"`
	Output   string `json:"output,omitempty"` // first 4 KiB of stdout and stderr when it failed
	GPUs     []GPU  `json:"gpus,omitempty"`
}

// A Report is one probe's evidence and verdict.
type Report struct {
	Time     time.Time `json:"time"`
	OK       bool      `json:"ok"`
	Problems []string  `json:"problems,omitempty"`
	GPU      *GPU      `json:"gpu,omitempty"`
	Nodes    []Node    `json:"nodes"`
	SMI      SMIResult `json:"nvidia_smi"`
	Driver   string    `json:"driver,omitempty"` // first line of /proc/driver/nvidia/version
	Cgroup   string    `json:"cgroup,omitempty"` // /proc/self/cgroup
	Env      []string  `json:"env,omitempty"`    // CUDA_VISIBLE_DEVICES and NVIDIA_VISIBLE_DEVICES
}

// Probe runs nvidia-smi -q -x, selects the GPU, opens its node and the
// control nodes, and reads the driver version and cgroup. OK reports whether
// every check passed; Problems explains each failure.
func Probe(ctx context.Context, opt Options) *Report {
	r := &Report{Time: time.Now().UTC()}
	smi := opt.SMI
	if smi == "" {
		smi = "nvidia-smi"
	}
	timeout := opt.Timeout
	if timeout <= 0 {
		timeout = 5 * time.Second
	}
	r.SMI = runSMI(ctx, smi, timeout)
	paths := append([]string(nil), ControlNodes...)
	switch {
	case r.SMI.ExitCode != 0:
		r.problem("NVML failed through nvidia-smi (exit %d): %s; the driver is unreachable from this process, so CUDA and CUPTI calls will fail too (CUPTI as CUPTI_ERROR_UNKNOWN)",
			r.SMI.ExitCode, firstLine(r.SMI.Output+r.SMI.Error))
	case opt.UUID != "":
		g, err := Select(r.SMI.GPUs, opt.UUID)
		if err != nil {
			r.problem("%v", err)
			break
		}
		r.GPU = &g
	case len(r.SMI.GPUs) != 1:
		r.problem("nvidia-smi lists %d GPUs; give a UUID to choose one", len(r.SMI.GPUs))
	case r.SMI.GPUs[0].Minor < 0:
		r.problem("GPU %s has no device minor number", r.SMI.GPUs[0].UUID)
	default:
		r.GPU = &r.SMI.GPUs[0]
	}
	if r.GPU != nil {
		paths = append([]string{r.GPU.DevicePath()}, paths...)
	}
	r.Nodes = ProbeNodes(opt.Root, paths)
	for _, n := range r.Nodes {
		if !n.OpenOK {
			r.problem("%s: %s", n.Path, explainNode(n))
		}
	}
	if b, err := os.ReadFile(filepath.Join(opt.Root, "/proc/driver/nvidia/version")); err == nil {
		r.Driver = firstLine(string(b))
	}
	if b, err := os.ReadFile(filepath.Join(opt.Root, "/proc/self/cgroup")); err == nil {
		r.Cgroup = strings.TrimSpace(string(b))
	}
	for _, k := range []string{"CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"} {
		if v, ok := os.LookupEnv(k); ok {
			r.Env = append(r.Env, k+"="+v)
		}
	}
	r.OK = len(r.Problems) == 0
	return r
}

func (r *Report) problem(format string, args ...any) {
	r.Problems = append(r.Problems, fmt.Sprintf(format, args...))
}

func runSMI(ctx context.Context, smi string, timeout time.Duration) SMIResult {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, smi, "-q", "-x")
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	err := cmd.Run()
	res := SMIResult{}
	if err != nil {
		res.ExitCode = -1
		var ee *exec.ExitError
		if errors.As(err, &ee) && ee.Exited() {
			res.ExitCode = ee.ExitCode()
		}
		if ctx.Err() != nil {
			res.Error = fmt.Sprintf("timed out after %v", timeout)
		} else if res.ExitCode == -1 {
			res.Error = err.Error()
		}
		// nvidia-smi prints its errors on standard output.
		res.Output = clip(stdout.String()+stderr.String(), 4096)
		return res
	}
	gpus, err := ParseSMIXML(stdout.Bytes())
	if err != nil {
		res.ExitCode, res.Error = -1, err.Error()
		return res
	}
	res.GPUs = gpus
	return res
}

// explainNode names the likely cause of a node that did not open.
func explainNode(n Node) string {
	switch {
	case !n.Exists && n.Errno == int(syscall.ENOENT):
		return "absent; the node is not mapped into this container or the driver is not loaded"
	case !n.Exists:
		return "stat failed: " + n.Error
	case !n.CharDevice:
		return fmt.Sprintf("mode %v is not a character device", n.Mode)
	case n.Errno == int(syscall.EACCES) || n.Errno == int(syscall.EPERM):
		if n.Mode.Perm()&0o006 == 0o006 {
			return fmt.Sprintf("open denied (%s) although mode %v allows it; the device cgroup no longer admits this node, as after a systemd daemon-reload with docker --gpus; map nodes with --device", n.Error, n.Mode)
		}
		return fmt.Sprintf("open denied (%s) by mode %v", n.Error, n.Mode)
	case n.Errno == int(syscall.ENXIO) || n.Errno == int(syscall.ENODEV):
		return fmt.Sprintf("open failed (%s); the driver has no such device: it fell off the bus, was reset, or the driver is unloading", n.Error)
	}
	return "open failed: " + n.Error
}

func firstLine(s string) string {
	s = strings.TrimSpace(s)
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		s = s[:i]
	}
	return s
}

func clip(s string, n int) string {
	if len(s) > n {
		return s[:n]
	}
	return s
}
