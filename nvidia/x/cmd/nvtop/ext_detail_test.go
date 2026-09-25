package main

import (
	"slices"
	"testing"
	"testing/fstest"
	"time"
)

func TestContainerOf(t *testing.T) {
	const id = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	const uid = "1f2e3d4c-5b6a-7980-a1b2-c3d4e5f60718"
	for _, tt := range []struct {
		cg             string
		container, pod string
	}{
		{"/user.slice/user-1000.slice/session-3.scope", "", ""},
		{"/system.slice/docker-" + id + ".scope", "docker 0123456789ab", ""},
		{"/docker/" + id, "docker 0123456789ab", ""},
		{"/machine.slice/libpod-" + id + ".scope/container", "podman 0123456789ab", ""},
		{"/kubepods.slice/kubepods-burstable.slice/kubepods-burstable-pod" + "1f2e3d4c_5b6a_7980_a1b2_c3d4e5f60718" + ".slice/cri-containerd-" + id + ".scope",
			"containerd 0123456789ab", uid},
		{"/kubepods.slice/kubepods-pod" + "1f2e3d4c_5b6a_7980_a1b2_c3d4e5f60718" + ".slice/crio-" + id + ".scope", "cri-o 0123456789ab", uid},
		{"/kubepods/besteffort/pod" + uid + "/" + id, "container 0123456789ab", uid},
		{"/system.slice/docker-notanid.scope", "", ""},
	} {
		c, p := containerOf(tt.cg)
		if c.v != tt.container || p.v != tt.pod || c.ok != (tt.container != "") || p.ok != (tt.pod != "") {
			t.Errorf("containerOf(%q) = %v, %v; want %q, %q", tt.cg, c, p, tt.container, tt.pod)
		}
	}
}

func TestCgroupPath(t *testing.T) {
	for _, tt := range []struct{ data, want string }{
		{"0::/system.slice/foo.service\n", "/system.slice/foo.service"},
		{"12:memory:/docker/abc\n0::/\n", "/"},
		{"12:memory:/docker/abc\n11:cpu:/other\n", "/docker/abc"},
		{"", ""},
	} {
		got, ok := cgroupPath(tt.data)
		if got != tt.want || ok != (tt.want != "") {
			t.Errorf("cgroupPath(%q) = %q, %v; want %q", tt.data, got, ok, tt.want)
		}
	}
}

// detailFS is a /proc holding pid 1234, which started an hour after
// boot, in a Docker container.
func detailFS() fstest.MapFS {
	return fstest.MapFS{
		"stat":         {Data: []byte("cpu  1 2 3\nbtime 1790240400\n")},
		"1234/status":  {Data: []byte("Name:\tpython\nState:\tS (sleeping)\nPPid:\t77\nThreads:\t42\n")},
		"1234/stat":    {Data: []byte("1234 (python) S 77 1234 1234 0 -1 4194304 0 0 0 0 5 5 0 0 20 0 42 0 360000 1000000 256 0\n")},
		"1234/cgroup":  {Data: []byte("0::/system.slice/docker-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef.scope\n")},
		"1234/environ": {Data: []byte("HOME=/root\x00CUDA_VISIBLE_DEVICES=0\x00NCCL_DEBUG=INFO\x00PATH=/bin\x00")},
	}
}

func TestHostDetail(t *testing.T) {
	now := testNow
	h := testHost(detailFS(), &now)
	d := h.detail(1234)
	boot := time.Unix(1790240400, 0)
	if d.state.v != "S (sleeping)" || d.ppid.v != 77 || d.threads.v != 42 {
		t.Errorf("status: state %v, ppid %v, threads %v", d.state, d.ppid, d.threads)
	}
	if !d.started.ok || !d.started.v.Equal(boot.Add(time.Hour)) {
		t.Errorf("started %v, want %v", d.started, boot.Add(time.Hour))
	}
	if d.container.v != "docker 0123456789ab" || d.pod.ok {
		t.Errorf("container %v, pod %v", d.container, d.pod)
	}
	if !slices.Equal(d.env.v, []string{"CUDA_VISIBLE_DEVICES=0", "NCCL_DEBUG=INFO"}) {
		t.Errorf("env %v", d.env)
	}

	// A process that is gone, or whose files are not readable.
	d = h.detail(5)
	if d.state.ok || d.started.ok || d.cgroup.ok || d.env.ok {
		t.Errorf("missing pid: %+v", d)
	}
}

func TestProcessDetail(t *testing.T) {
	now := time.Unix(1790240400, 0).UTC().Add(3 * time.Hour)
	u := newUI(testDevices(), options{ext: true, encodeHide: 30 * time.Second}, 1, func() time.Time { return now })
	u.host = testHost(detailFS(), &now)
	for range 50 {
		u.record()
	}
	s := newScreen(120, 40, true)
	u.render(s) // sort: pid 1234 first
	u.handle(keyEnter)
	if u.view != viewDetail || u.detailKey != (procKey{0, 1234}) {
		t.Fatalf("Enter: view %d, detail %v; want 1234 in detail", u.view, u.detailKey)
	}
	u.render(s)
	golden(t, "render/detail.golden", []byte(s.String()))
	checkColors(t, s, []colorCheck{
		{"THREADS 42", 0, cyan},
		{"THREADS 42", 8, noColor},
		{"PID1234 %", 0, cyan},
		{"PID1234 mem%", 0, yellow},
	})

	// The history of a process shown in detail outlives the process.
	u.devs[0].procs = u.devs[0].procs[1:]
	u.record()
	if u.procHistory[procKey{0, 1234}] == nil {
		t.Errorf("history of the process in detail dropped")
	}
	u.render(s)
	if _, _, ok := find(s, "(exited)"); !ok {
		t.Errorf("exited process not marked:\n%s", s)
	}
	u.handle(keyEsc)
	if u.view != viewProcesses || u.quit {
		t.Errorf("Esc: view %d, quit %v", u.view, u.quit)
	}
	u.record()
	if u.procHistory[procKey{0, 1234}] != nil {
		t.Errorf("history of an exited process kept after leaving the detail")
	}
}
