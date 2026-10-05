package cudaschedule

import (
	"os"
	"strings"
	"testing"
)

func TestRecordedSignalTargetAfterProcessRelease(t *testing.T) {
	process, err := os.FindProcess(os.Getpid())
	if err != nil {
		t.Fatal(err)
	}
	retained := recordedProcess{PID: process.Pid, Group: 2, Start: 42}
	if err := process.Release(); err != nil {
		t.Fatal(err)
	}
	if process.Pid != -1 {
		t.Fatalf("release did not invalidate process pid: %d", process.Pid)
	}
	target, err := recordedSignalTarget(retained)
	if err != nil || target != os.Getpid() {
		t.Fatalf("immutable identity changed after release: %d %v", target, err)
	}
	for _, pid := range []int{-1, 0, 1} {
		bad := retained
		bad.PID = pid
		if _, err := recordedSignalTarget(bad); err == nil {
			t.Fatalf("accepted signal pid %d", pid)
		}
		bad = retained
		bad.Group = pid
		if _, err := recordedSignalTarget(bad); err == nil {
			t.Fatalf("accepted signal group %d", pid)
		}
	}
}

func TestRecordedMessage(t *testing.T) {
	nonce := strings.Repeat("a", 64)
	base := `{"version":1,"nonce":"` + nonce + `","kind":"ready"}`
	for _, tt := range []struct {
		name   string
		packet string
		want   bool
	}{
		{"valid", base, true},
		{"duplicate", strings.Replace(base, `"version":1`, `"version":1,"version":1`, 1), false},
		{"escaped duplicate", strings.Replace(base, `"version":1`, `"version":1,"vers\u0069on":1`, 1), false},
		{"case alias", strings.Replace(base, `"version"`, `"Version"`, 1), false},
		{"unknown", strings.Replace(base, `"version":1`, `"version":1,"pid":7`, 1), false},
		{"trailing", base + `{}`, false},
		{"null", "null", false},
		{"wrong version", strings.Replace(base, `"version":1`, `"version":2`, 1), false},
		{"missing version", strings.Replace(base, `"version":1,`, "", 1), false},
		{"wrong nonce", strings.Replace(base, nonce, strings.Repeat("b", 64), 1), false},
		{"unknown kind", strings.Replace(base, "ready", "cuda", 1), false},
		{"bad digest", strings.Replace(base, `"version":1`, `"version":1,"digest":"bad"`, 1), false},
		{"oversize", base + strings.Repeat(" ", recordedMessageLimit), false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			_, err := readRecordedMessage([]byte(tt.packet), nonce)
			if (err == nil) != tt.want {
				t.Fatalf("accepted=%v, want %v: %v", err == nil, tt.want, err)
			}
		})
	}
}

func TestRecordedHandshake(t *testing.T) {
	order := []string{"ready", "start", "admit", "retained", "exec", "worker-ready", "verified", "permit", "done"}
	for skip := range order {
		var h recordedHandshake
		for i, kind := range order {
			if i == skip {
				continue
			}
			if err := h.accept(kind); err != nil {
				break
			}
		}
		if skip < 8 && h.permitted() {
			t.Fatalf("missing step %s admitted CUDA", order[skip])
		}
	}
	var h recordedHandshake
	for i, kind := range order {
		if err := h.accept(kind); err != nil {
			t.Fatal(err)
		}
		if h.permitted() != (i >= 7) {
			t.Fatalf("permission at step %s", kind)
		}
	}
	if err := h.accept("permit"); err == nil {
		t.Fatal("accepted replayed permit")
	}
}

func FuzzRecordedMessage(f *testing.F) {
	nonce := strings.Repeat("a", 64)
	f.Add([]byte(`{"version":1,"nonce":"` + nonce + `","kind":"ready"}`))
	f.Add([]byte(`{"version":1,"version":1}`))
	f.Fuzz(func(t *testing.T, packet []byte) {
		msg, err := readRecordedMessage(packet, nonce)
		if err != nil {
			return
		}
		encoded, err := writeRecordedMessage(msg)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := readRecordedMessage(encoded, nonce); err != nil {
			t.Fatal(err)
		}
	})
}
