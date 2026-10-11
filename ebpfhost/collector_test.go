package ebpfhost

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestReadCancellation(t *testing.T) {
	r, w, e := os.Pipe()
	if e != nil {
		t.Fatal(e)
	}
	defer r.Close()
	defer w.Close()
	c := &Collector{}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, e = c.Read(ctx); !errors.Is(e, context.Canceled) {
		t.Fatal(e)
	}
}
func TestReadBuffered(t *testing.T) {
	c := &Collector{pending: []byte("one\ntwo\n")}
	r, e := c.Read(context.Background())
	if e != nil || r.Line != "one\n" {
		t.Fatal(r, e)
	}
	c.closed = true
	if _, e = c.Read(context.Background()); !errors.Is(e, os.ErrClosed) {
		t.Fatal(e)
	}
}
func TestProcessStartTime(t *testing.T) {
	stat := "123 (name with ) paren) S " + strings.Repeat("0 ", 18) + "987 0"
	n, e := parseStartTime(stat)
	if e != nil || n != 987 {
		t.Fatal(n, e)
	}
}
func TestResolveImage(t *testing.T) {
	cc, e := exec.LookPath("cc")
	if e != nil {
		t.Skip(e)
	}
	dir := t.TempDir()
	src := filepath.Join(dir, "probe.c")
	bin := filepath.Join(dir, "probe.so")
	os.WriteFile(src, []byte("int known(void) { return 42; }"), 0600)
	cmd := exec.Command(cc, "-shared", "-fPIC", "-Wl,--build-id", "-o", bin, src)
	if out, e := cmd.CombinedOutput(); e != nil {
		t.Skipf("ELF compiler unavailable: %s: %v", out, e)
	}
	image, e := ResolveImage(bin, "known")
	if e != nil {
		t.Fatal(e)
	}
	if image.Offset == 0 || image.Size == 0 || len(image.SHA256) != 64 || image.BuildID == "" {
		t.Fatal(image)
	}
	if _, e = ResolveImage(bin, "absent"); e == nil {
		t.Fatal("missing symbol accepted")
	}
}
func ExampleCollector_TraceClock() {
	c := new(Collector)
	fmt.Println(c.TraceClock())
	// Output: mono
}

func TestCloseRetainsFailedCleanup(t *testing.T) {
	root := t.TempDir()
	instance := filepath.Join(root, "instance")
	if e := os.Mkdir(instance, 0700); e != nil {
		t.Fatal(e)
	}
	if e := os.WriteFile(filepath.Join(instance, "tracing_on"), []byte("1"), 0600); e != nil {
		t.Fatal(e)
	}
	c := &Collector{root: root, instance: instance, group: "owned", probes: []string{"entry"}}
	if e := c.Close(); e == nil {
		t.Fatal("missing cleanup failure")
	}
	if c.cleanupDone || len(c.probes) != 1 {
		t.Fatal("failed cleanup forgotten")
	}
	if e := os.Mkdir(filepath.Join(instance, "events"), 0700); e != nil {
		t.Fatal(e)
	}
	if e := os.WriteFile(filepath.Join(root, "uprobe_events"), nil, 0600); e != nil {
		t.Fatal(e)
	}
	if e := c.Close(); e == nil {
		t.Fatal("first error not retained")
	}
	if len(c.probes) != 0 || !c.eventsDisabled {
		t.Fatal("retry did not remove probes")
	}
	data, e := os.ReadFile(filepath.Join(root, "uprobe_events"))
	if e != nil || string(data) != "-:owned/entry\n" {
		t.Fatal(string(data), e)
	}
	if e = os.Remove(filepath.Join(instance, "tracing_on")); e != nil {
		t.Fatal(e)
	}
	if e = os.Remove(filepath.Join(instance, "events/enable")); e != nil {
		t.Fatal(e)
	}
	if e = os.Remove(filepath.Join(instance, "events")); e != nil {
		t.Fatal(e)
	}
	if e = c.Close(); e == nil {
		t.Fatal("historical cleanup error vanished")
	}
	if !c.cleanupDone {
		t.Fatal("cleanup incomplete")
	}
	data, _ = os.ReadFile(filepath.Join(root, "uprobe_events"))
	if string(data) != "-:owned/entry\n" {
		t.Fatal("successful probe removal repeated")
	}
}
