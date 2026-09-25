package main

import (
	"bufio"
	"bytes"
	"fmt"
	"io/fs"
	"strconv"
	"strings"
	"time"
)

// A procKey names a process on a device.
type procKey struct{ dev, pid int }

// procDetail is what the process detail view shows beyond the process
// list's columns, read from /proc.
type procDetail struct {
	state     opt[string] // as in /proc/pid/status: "S (sleeping)"
	ppid      opt[int]
	threads   opt[int]
	started   opt[time.Time]
	cgroup    opt[string]
	container opt[string] // runtime and short ID
	pod       opt[string] // Kubernetes pod UID
	env       opt[[]string]
}

// envPrefixes select the environment variables the detail view shows:
// those that decide which GPUs a process sees and how it uses them.
var envPrefixes = []string{"CUDA_", "NVIDIA_", "NCCL_", "PYTORCH_CUDA_", "TF_GPU_", "XLA_"}

// detail reads the detail of pid.
func (h *host) detail(pid int) procDetail {
	var d procDetail
	dir := strconv.Itoa(pid) + "/"
	if data, err := fs.ReadFile(h.fsys, dir+"status"); err == nil {
		s := bufio.NewScanner(bytes.NewReader(data))
		for s.Scan() {
			key, val, _ := strings.Cut(s.Text(), ":")
			val = strings.TrimSpace(val)
			switch key {
			case "State":
				d.state = some(val)
			case "PPid":
				if n, err := strconv.Atoi(val); err == nil {
					d.ppid = some(n)
				}
			case "Threads":
				if n, err := strconv.Atoi(val); err == nil {
					d.threads = some(n)
				}
			}
		}
	}
	d.started = h.started(pid)
	if data, err := fs.ReadFile(h.fsys, dir+"cgroup"); err == nil {
		if cg, ok := cgroupPath(string(data)); ok {
			d.cgroup = some(cg)
			d.container, d.pod = containerOf(cg)
		}
	}
	if data, err := fs.ReadFile(h.fsys, dir+"environ"); err == nil {
		var env []string
		for _, kv := range strings.Split(string(data), "\x00") {
			for _, p := range envPrefixes {
				if strings.HasPrefix(kv, p) {
					env = append(env, kv)
					break
				}
			}
		}
		d.env = some(env)
	}
	return d
}

// started returns when pid started: field 22 of /proc/pid/stat, in
// clock ticks after the boot time in /proc/stat.
func (h *host) started(pid int) opt[time.Time] {
	data, err := fs.ReadFile(h.fsys, strconv.Itoa(pid)+"/stat")
	if err != nil {
		return opt[time.Time]{}
	}
	i := bytes.LastIndexByte(data, ')')
	if i < 0 {
		return opt[time.Time]{}
	}
	f := strings.Fields(string(data[i+1:]))
	if len(f) < 20 {
		return opt[time.Time]{}
	}
	ticks, err := strconv.ParseUint(f[19], 10, 64)
	if err != nil {
		return opt[time.Time]{}
	}
	data, err = fs.ReadFile(h.fsys, "stat")
	if err != nil {
		return opt[time.Time]{}
	}
	s := bufio.NewScanner(bytes.NewReader(data))
	for s.Scan() {
		if v, ok := strings.CutPrefix(s.Text(), "btime "); ok {
			boot, err := strconv.ParseInt(strings.TrimSpace(v), 10, 64)
			if err != nil {
				break
			}
			off := time.Duration(float64(ticks) / h.hz * float64(time.Second))
			return some(time.Unix(boot, 0).Add(off))
		}
	}
	return opt[time.Time]{}
}

// cgroupPath returns the cgroup of a process from /proc/pid/cgroup: the
// unified hierarchy's, or failing that the first listed.
func cgroupPath(data string) (string, bool) {
	var first string
	for _, line := range strings.Split(strings.TrimSpace(data), "\n") {
		f := strings.SplitN(line, ":", 3)
		if len(f) != 3 {
			continue
		}
		if f[0] == "0" && f[1] == "" {
			return f[2], true
		}
		if first == "" {
			first = f[2]
		}
	}
	return first, first != ""
}

// containerOf returns the container runtime and short container ID, and
// the Kubernetes pod UID, that a cgroup path names, if any. It knows the
// systemd scope names of Docker, containerd, CRI-O and Podman, and the
// cgroupfs layouts of Docker and Kubernetes.
func containerOf(cg string) (container, pod opt[string]) {
	short := func(id string) string { return id[:min(len(id), 12)] }
	segs := strings.Split(cg, "/")
	for i, seg := range segs {
		for _, rt := range []struct{ prefix, name string }{
			{"docker-", "docker"},
			{"cri-containerd-", "containerd"},
			{"crio-", "cri-o"},
			{"libpod-", "podman"},
		} {
			if id, ok := strings.CutPrefix(seg, rt.prefix); ok {
				if id, ok := strings.CutSuffix(id, ".scope"); ok && isHex(id) {
					container = some(rt.name + " " + short(id))
				}
			}
		}
		if i > 0 && isHex(seg) && len(seg) == 64 {
			switch parent := segs[i-1]; {
			case parent == "docker":
				container = some("docker " + short(seg))
			case strings.HasPrefix(parent, "pod"):
				container = some("container " + short(seg))
			}
		}
		// kubepods-burstable-pod<uid>.slice, with _ for -, or
		// kubepods/burstable/pod<uid>.
		if j := strings.LastIndex(seg, "pod"); j >= 0 && strings.Contains(cg, "kubepods") {
			uid := strings.TrimSuffix(seg[j+3:], ".slice")
			if len(uid) == 36 {
				pod = some(strings.ReplaceAll(uid, "_", "-"))
			}
		}
	}
	return container, pod
}

func isHex(s string) bool {
	if s == "" {
		return false
	}
	for _, c := range s {
		if !('0' <= c && c <= '9' || 'a' <= c && c <= 'f') {
			return false
		}
	}
	return true
}

// recordProcs appends each process's GPU and GPU memory use to its
// history, and drops the history of processes gone, but for the one
// shown in detail.
func (u *ui) recordProcs() {
	seen := make(map[procKey]bool)
	for i, d := range u.devs {
		for _, p := range d.procs {
			k := procKey{i, p.pid}
			h := u.procHistory[k]
			if h == nil {
				h = new([2]series)
				u.procHistory[k] = h
			}
			h[0].push(p.gpuUtil.v)
			h[1].push(p.memPct.v)
			seen[k] = true
		}
	}
	for k := range u.procHistory {
		if !seen[k] && !(u.view == viewDetail && k == u.detailKey) {
			delete(u.procHistory, k)
		}
	}
}

// openDetail shows the selected process in detail.
func (u *ui) openDetail() {
	if u.selected >= len(u.rows) {
		return
	}
	r := u.rows[u.selected]
	u.view = viewDetail
	u.detailKey = procKey{r.dev, r.p.pid}
	u.detailProc = *r.p
}

// drawDetail draws the process shown in detail in place of the process
// list: what the list shows about it, what /proc adds, and a plot of its
// GPU and GPU memory use.
func (u *ui) drawDetail(s *screen, r rect) {
	w := s.window(r.x, r.y, r.w, r.h)
	k := u.detailKey
	p := &u.detailProc
	gone := true
	if k.dev < len(u.devs) {
		for i := range u.devs[k.dev].procs {
			if q := &u.devs[k.dev].procs[i]; q.pid == k.pid {
				u.detailProc, gone = *q, false
				break
			}
		}
	}
	title := fmt.Sprintf("PID %d  %s  %s", k.pid, orNA(p.user), orNA(p.cmdline))
	if gone {
		title += "  (exited)"
	}
	w.print(0, 0, title, plain)
	w.chgat(0, 0, -1, standout(green))

	y := 1
	line := func(parts ...string) {
		x := 0
		for i := 0; i+1 < len(parts); i += 2 {
			if parts[i+1] == "" {
				continue
			}
			x = w.print(x, y, parts[i]+" ", fg(cyan))
			x = w.print(x, y, parts[i+1]+"  ", plain)
		}
		y++
	}
	mem := "N/A"
	if p.gpuMem.ok {
		mem = fmt.Sprintf("%dMiB", p.gpuMem.v>>20)
		if p.memPct.ok {
			mem += fmt.Sprintf(" (%d%%)", p.memPct.v)
		}
	}
	host := "N/A"
	if p.hostMem.ok {
		host = fmt.Sprintf("%dMiB", p.hostMem.v>>20)
	}
	pctOrNA := func(v opt[uint32]) string {
		if v.ok {
			return fmt.Sprintf("%d%%", v.v)
		}
		return "N/A"
	}
	kind := map[processKind]string{graphic: "Graphic", compute: "Compute", graphicCompute: "Both G+C"}[p.kind]
	line("DEV", strconv.Itoa(k.dev), "TYPE", kind, "GPU", pctOrNA(p.gpuUtil), "GPU MEM", mem,
		"CPU", pctOrNA(p.cpuUtil), "HOST MEM", host)

	if u.host != nil {
		d := u.host.detail(k.pid)
		str := func(v opt[int]) string {
			if v.ok {
				return strconv.Itoa(v.v)
			}
			return ""
		}
		started := ""
		if d.started.ok {
			started = d.started.v.In(u.now().Location()).Format(time.DateTime)
			if age := u.now().Sub(d.started.v); age >= 0 {
				started += " (" + age.Round(time.Second).String() + " ago)"
			}
		}
		line("STATE", d.state.v, "PPID", str(d.ppid), "THREADS", str(d.threads), "STARTED", started)
		line("CGROUP", d.cgroup.v)
		if d.container.ok || d.pod.ok {
			line("CONTAINER", d.container.v, "POD", d.pod.v)
		}
		switch {
		case !d.env.ok:
			line("ENV", "not readable")
		case len(d.env.v) == 0:
			line("ENV", "no GPU variables set")
		default:
			for _, kv := range d.env.v {
				line("ENV", kv)
			}
		}
	}

}

// drawDetailPlot draws the GPU and GPU memory use of the process shown
// in detail in the plot at r.
func (u *ui) drawDetailPlot(s *screen, r rect) {
	k := u.detailKey
	h := u.procHistory[k]
	if h == nil {
		h = new([2]series)
	}
	u.drawPlot(s, r, []*series{&h[0], &h[1]},
		[]string{fmt.Sprintf("PID%d %%", k.pid), fmt.Sprintf("PID%d mem%%", k.pid)})
}
