//go:build linux

package cudaschedule

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

func TestRecordedPinnedFile(t *testing.T) {
	dir := t.TempDir()
	regular := filepath.Join(dir, "body")
	body := []byte("actual pinned file\n")
	if err := os.WriteFile(regular, body, 0600); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(body)
	digest := hex.EncodeToString(sum[:])
	symlink := filepath.Join(dir, "link")
	if err := os.Symlink(regular, symlink); err != nil {
		t.Fatal(err)
	}
	fifo := filepath.Join(dir, "fifo")
	if err := syscall.Mkfifo(fifo, 0600); err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		name, path, digest string
		limit              int64
		want               bool
	}{
		{"actual", regular, digest, 1024, true},
		{"wrong digest", regular, strings.Repeat("0", 64), 1024, false},
		{"oversize", regular, digest, 1, false},
		{"symlink", symlink, digest, 1024, false},
		{"fifo", fifo, digest, 1024, false},
		{"relative", "body", digest, 1024, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			start := time.Now()
			data, err := recordedPinFile(tt.path, tt.digest, tt.limit)
			if (err == nil) != tt.want {
				t.Fatalf("accepted=%v want=%v: %v", err == nil, tt.want, err)
			}
			if time.Since(start) > time.Second {
				t.Fatal("pinned-file refusal exceeded one second")
			}
			if err == nil && string(data) != string(body) {
				t.Fatalf("substituted body: %q", data)
			}
		})
	}
}

func TestRecordedPeerCredentials(t *testing.T) {
	self, err := recordedIdentify(os.Getpid())
	if err != nil {
		t.Fatal(err)
	}
	listener, err := net.ListenUnix("unixpacket", &net.UnixAddr{Name: "@recorded-peer-" + strconv.Itoa(os.Getpid()) + "-" + strconv.FormatInt(time.Now().UnixNano(), 10), Net: "unixpacket"})
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	client, err := net.DialUnix("unixpacket", nil, listener.Addr().(*net.UnixAddr))
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()
	if err := listener.SetDeadline(time.Now().Add(time.Second)); err != nil {
		t.Fatal(err)
	}
	server, err := listener.AcceptUnix()
	if err != nil {
		t.Fatal(err)
	}
	defer server.Close()
	for _, tt := range []struct {
		name          string
		parent, group int
		image         string
		want          bool
	}{
		{"actual", self.Parent, self.Group, self.Executable, true},
		{"wrong parent", self.Parent + 1, self.Group, self.Executable, false},
		{"wrong group", self.Parent, self.Group + 1, self.Executable, false},
		{"wrong image", self.Parent, self.Group, strings.Repeat("0", 64), false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			proc, err := recordedPeer(server, tt.parent, tt.group, tt.image)
			if (err == nil) != tt.want {
				t.Fatalf("accepted=%v want=%v: %v", err == nil, tt.want, err)
			}
			if err == nil && proc.PID != self.PID {
				t.Fatalf("invented peer pid: %+v", proc)
			}
		})
	}
}

func TestRecordedProcessIdentity(t *testing.T) {
	proc, err := recordedIdentify(os.Getpid())
	if err != nil {
		t.Fatal(err)
	}
	if proc.PID != os.Getpid() || proc.Parent != os.Getppid() || !validPTXDigest(proc.Executable) || !recordedAlive(proc) {
		t.Fatalf("invalid identity: %+v", proc)
	}
	proc.Start++
	if recordedAlive(proc) {
		t.Fatal("accepted reused pid with different start")
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := recordedRetire(ctx, []int{syscall.Getpgrp()}); err == nil {
		t.Fatal("accepted caller's group")
	}
}

func TestRecordedRetireAfterLeaderExit(t *testing.T) {
	dir := t.TempDir()
	pidFile := filepath.Join(dir, "child")
	cmd := exec.Command("/bin/sh", "-c", `sh -c 'trap "" TERM; echo $$ > "$1"; while :; do sleep 1; done' child "$1" &`, "leader", pidFile)
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	group := cmd.Process.Pid
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		if err := recordedRetire(ctx, []int{group}); err != nil {
			t.Errorf("descendant cleanup: %v", err)
		}
	})
	if err := cmd.Wait(); err != nil {
		t.Fatal(err)
	}
	var child int
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		data, err := os.ReadFile(pidFile)
		if err == nil {
			child, _ = strconv.Atoi(strings.TrimSpace(string(data)))
			if child > 1 {
				break
			}
		}
		time.Sleep(10 * time.Millisecond)
	}
	if child <= 1 {
		t.Fatal("child did not publish readiness")
	}
	members, err := recordedGroupMembers(group)
	if err != nil || len(members) == 0 {
		t.Fatalf("child not retained after leader exit: %v %v", members, err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := recordedRetire(ctx, []int{group}); err != nil {
		t.Fatal(err)
	}
	members, err = recordedGroupMembers(group)
	if err != nil || len(members) != 0 {
		t.Fatalf("live descendants after cleanup: %v %v", members, err)
	}
}
