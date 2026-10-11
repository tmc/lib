package bpfprobe

import (
	"errors"
	"github.com/tmc/lib/ebpfhost/bpfload"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"testing"
)

func updateConfig(t *testing.T) Config {
	t.Helper()
	path := filepath.Join(t.TempDir(), "binary")
	if err := os.WriteFile(path, []byte{1}, 0600); err != nil {
		t.Fatal(err)
	}
	return Config{Path: path, PID: 1, Capacity: 1 << 20}
}
func updateGroup() *Group {
	return &Group{Probe: &Probe{}, capacity: 1 << 20, probes: []*Probe{{}}, attach: func(Config, *Probe) error { return nil }}
}
func updateVerify() error { return nil }

func TestGroupAddRetainsOwnership(t *testing.T) {
	g := updateGroup()
	old, root := g.probes[0], g.Probe
	c := updateConfig(t)
	g.TaskStartOffset, g.TaskExecOffset, g.TaskExecSize = 8, 16, 8
	var seen Config
	verified := 0
	g.attach = func(c Config, p *Probe) error { seen = c; return nil }
	if err := g.Add([]Config{c, c}, func() error { verified++; return nil }); err != nil {
		t.Fatal(err)
	}
	if len(g.probes) != 3 || g.probes[0] != old || g.Probe != root || old.closed || root.closed {
		t.Fatal("existing resources changed")
	}
	if seen.generationOffset != 8 || seen.execOffset != 16 || seen.execSize != 8 || verified != 2 {
		t.Fatal("layout/verification not retained")
	}
	if err := g.Close(); err != nil {
		t.Fatal(err)
	}
	for _, p := range g.probes {
		if !p.closed {
			t.Fatal("attachment leaked")
		}
	}
}

func TestGroupAddRollback(t *testing.T) {
	for _, failure := range []string{"attachment", "postverify", "preverify"} {
		t.Run(failure, func(t *testing.T) {
			g := updateGroup()
			old := g.probes[0]
			c := updateConfig(t)
			sentinel := errors.New("refused")
			var added []*Probe
			verifications := 0
			g.attach = func(c Config, p *Probe) error {
				f, err := os.CreateTemp(t.TempDir(), "perf")
				if err != nil {
					return err
				}
				p.perf = f
				added = append(added, p)
				if failure == "attachment" && len(added) == 2 {
					return sentinel
				}
				return nil
			}
			verify := func() error {
				verifications++
				if failure == "preverify" || failure == "postverify" && verifications == 2 {
					return sentinel
				}
				return nil
			}
			if err := g.Add([]Config{c, c, c}, verify); !errors.Is(err, sentinel) {
				t.Fatal("wrong failure", err)
			}
			if len(g.probes) != 1 || old.closed || g.Probe.closed {
				t.Fatal("old group changed")
			}
			for _, p := range added {
				if !p.closed {
					t.Fatal("partial attachment leaked")
				}
				if _, err := p.perf.Stat(); !errors.Is(err, os.ErrClosed) {
					t.Fatal("perf leaked", err)
				}
			}
			before := len(added)
			if err := g.Add([]Config{c}, updateVerify); !errors.Is(err, sentinel) {
				t.Fatal("failure not terminal", err)
			}
			if len(added) != before {
				t.Fatal("automatic retry")
			}
			if err := g.Close(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestGroupAddValidation(t *testing.T) {
	c := updateConfig(t)
	tests := []struct {
		name string
		edit func(*Group, *Config)
	}{
		{"capacity", func(g *Group, c *Config) { c.Capacity = 2 << 20 }},
		{"stack", func(g *Group, c *Config) { c.UserStack = true }},
		{"output", func(g *Group, c *Config) { c.OutputPointer = true; c.CorrelationTag = 1 }},
		{"layout", func(g *Group, c *Config) { c.execOffset = 9 }},
		{"inherit", func(g *Group, c *Config) { c.InheritThreads = true }},
		{"limit", func(g *Group, c *Config) { g.probes = make([]*Probe, 256) }},
		{"disabled", func(g *Group, c *Config) { g.disabled = true }},
		{"closed", func(g *Group, c *Config) { g.closed = true }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := updateGroup()
			v := c
			tt.edit(g, &v)
			calls := 0
			g.attach = func(Config, *Probe) error { calls++; return nil }
			if err := g.Add([]Config{v}, updateVerify); err == nil {
				t.Fatal("accepted invalid update")
			}
			if calls != 0 || g.updateErr != nil {
				t.Fatal("validation attached or poisoned group")
			}
		})
	}
	g := updateGroup()
	if err := g.Add(nil, updateVerify); err == nil {
		t.Fatal("accepted empty batch")
	}
	if err := g.Add([]Config{c}, nil); err == nil {
		t.Fatal("accepted nil verifier")
	}
	if err := g.Add([]Config{c}, updateVerify); err != nil {
		t.Fatal("validation blocked valid addition", err)
	}
	var nilGroup *Group
	if err := nilGroup.Add([]Config{c}, updateVerify); err == nil {
		t.Fatal("accepted nil group")
	}
	if err := (&Group{}).Add([]Config{c}, updateVerify); err == nil {
		t.Fatal("accepted zero group")
	}
}

func TestGroupAddSerializesClose(t *testing.T) {
	g := updateGroup()
	c := updateConfig(t)
	entered, release := make(chan struct{}), make(chan struct{})
	g.attach = func(Config, *Probe) error { close(entered); <-release; return nil }
	var wg sync.WaitGroup
	wg.Add(2)
	errs := make(chan error, 2)
	go func() { defer wg.Done(); errs <- g.Add([]Config{c}, updateVerify) }()
	<-entered
	go func() { defer wg.Done(); errs <- g.Close() }()
	close(release)
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
	if len(g.probes) != 2 || !g.probes[1].closed {
		t.Fatal("close missed new attachment")
	}
	if err := g.Add([]Config{c}, updateVerify); err == nil {
		t.Fatal("accepted after close")
	}
	if err := g.Close(); err != nil {
		t.Fatal("close not idempotent", err)
	}
}
func TestGroupAddAfterDisable(t *testing.T) {
	g := updateGroup()
	c := updateConfig(t)
	if err := g.Disable(); err != nil {
		t.Fatal(err)
	}
	if err := g.Add([]Config{c}, updateVerify); err == nil {
		t.Fatal("accepted after disable")
	}
	if g.probes[0].closed || g.Probe.closed {
		t.Fatal("disable closed drain resources")
	}
	if err := g.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestGroupAddUnsupported(t *testing.T) {
	if runtime.GOOS == "linux" && (runtime.GOARCH == "amd64" || runtime.GOARCH == "arm64") {
		t.Skip("supported kernel platform")
	}
	g := updateGroup()
	g.attach = nil
	c := updateConfig(t)
	if err := g.Add([]Config{c}, updateVerify); !errors.Is(err, bpfload.ErrUnsupported) {
		t.Fatal("platform refusal missing", err)
	}
	if len(g.probes) != 1 || g.Probe.closed || g.probes[0].closed {
		t.Fatal("platform refusal changed original ownership")
	}
	if err := g.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestGroupAddSerializesDisable(t *testing.T) {
	g := updateGroup()
	c := updateConfig(t)
	entered, release := make(chan struct{}), make(chan struct{})
	g.attach = func(Config, *Probe) error { close(entered); <-release; return nil }
	var wg sync.WaitGroup
	wg.Add(2)
	errs := make(chan error, 2)
	go func() { defer wg.Done(); errs <- g.Add([]Config{c}, updateVerify) }()
	<-entered
	go func() { defer wg.Done(); errs <- g.Disable() }()
	close(release)
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
	if len(g.probes) != 2 || !g.disabled || g.Probe.closed {
		t.Fatal("disable/update not serialized")
	}
	if err := g.Add([]Config{c}, updateVerify); err == nil {
		t.Fatal("accepted after disable")
	}
	if err := g.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestGroupAddLifetimeBoundary(t *testing.T) {
	g := updateGroup()
	g.probes = make([]*Probe, 254)
	c := updateConfig(t)
	calls := 0
	g.attach = func(Config, *Probe) error { calls++; return nil }
	if err := g.Add([]Config{c, c}, updateVerify); err != nil {
		t.Fatal(err)
	}
	if len(g.probes) != 256 || calls != 2 {
		t.Fatal("boundary not admitted")
	}
	if err := g.Add([]Config{c}, updateVerify); err == nil {
		t.Fatal("lifetime limit exceeded")
	}
	if calls != 2 {
		t.Fatal("overlimit attachment attempted")
	}
	if err := g.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestGroupAddPreservesOptionalMaps(t *testing.T) {
	g := updateGroup()
	g.events = &bpfload.Object{}
	g.counts = &bpfload.Object{}
	g.stacks = &bpfload.Object{}
	g.extras = []*bpfload.Object{{}, {}}
	events, counts, stacks, output, depth := g.events, g.counts, g.stacks, g.extras[0], g.extras[1]
	c := updateConfig(t)
	c.UserStack = true
	c.OutputPointer = true
	c.CorrelationTag = 7
	if err := g.Add([]Config{c}, updateVerify); err != nil {
		t.Fatal(err)
	}
	if g.events != events || g.counts != counts || g.stacks != stacks || g.extras[0] != output || g.extras[1] != depth {
		t.Fatal("shared maps replaced")
	}
	p := g.probes[1]
	if p.ring != nil || p.events != nil || p.counts != nil || p.stacks != nil || len(p.extras) != 0 {
		t.Fatal("new probe claimed shared map ownership")
	}
	if err := g.Close(); err != nil {
		t.Fatal(err)
	}
}
