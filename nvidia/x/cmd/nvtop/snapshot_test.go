package main

import (
	"bytes"
	"flag"
	"os"
	"path/filepath"
	"testing"
)

var update = flag.Bool("update", false, "rewrite the golden files in testdata")

// golden compares got with testdata/name, or rewrites it with -update.
func golden(t *testing.T, name string, got []byte) {
	t.Helper()
	path := filepath.Join("testdata", name)
	if *update {
		if err := os.WriteFile(path, got, 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Errorf("%s differs; got:\n%s\nwant:\n%s", name, got, want)
	}
}

// testDevices returns two devices: a discrete GPU reporting everything,
// with a graphics, a compute, and a mixed process, and one reporting
// nothing but its name.
func testDevices() []*device {
	const mib = 1 << 20
	return []*device{
		{
			index:        0,
			name:         some("NVIDIA GeForce RTX 4090"),
			arch:         some("Ada"),
			maxPcieGen:   some[uint32](4),
			maxPcieWidth: some[uint32](16),
			gpuClock:     some[uint32](2520),
			gpuClockMax:  some[uint32](3105),
			memClock:     some[uint32](10501),
			memClockMax:  some[uint32](10501),
			gpuUtil:      some[uint32](87),
			effLoad:      some[uint32](52),
			memUtil:      some[uint32](45),
			encUtil:      some[uint32](12),
			decUtil:      some[uint32](0),
			memTotal:     some[uint64](24564 * mib),
			memUsed:      some[uint64](11000 * mib),
			memFree:      some[uint64](13564 * mib),
			pcieGen:      some[uint32](4),
			pcieWidth:    some[uint32](16),
			pcieRx:       some[uint32](2048),
			pcieTx:       some[uint32](512),
			fan:          some[uint32](65),
			temp:         some[uint32](71),
			power:        some[uint32](270000),
			powerMax:     some[uint32](450000),
			procs: []process{
				{pid: 1234, kind: compute, user: some("alice"), cmdline: some(`python "train.py"` + "\t--epochs 3"), gpuMem: some[uint64](9000 * mib), memPct: some[uint32](37), gpuUtil: some[uint32](80), encUtil: some[uint32](0), decUtil: some[uint32](0), cpuUtil: some[uint32](104), hostMem: some[uint64](3000 * mib)},
				{pid: 999, kind: graphic, user: some("root"), cmdline: some("/usr/lib/xorg/Xorg vt2"), gpuMem: some[uint64](512 * mib), memPct: some[uint32](2), cpuUtil: some[uint32](1), hostMem: some[uint64](80 * mib)},
				{pid: 4321, kind: graphicCompute, user: some("bob"), cmdline: some("blender"), gpuMem: some[uint64](1488 * mib), memPct: some[uint32](6), gpuUtil: some[uint32](7), cpuUtil: some[uint32](12), hostMem: some[uint64](900 * mib)},
			},
		},
		{
			index: 1,
			name:  some("NVIDIA GB10"),
		},
	}
}

func TestWriteSnapshot(t *testing.T) {
	for _, tt := range []struct {
		name                  string
		fahrenheit, hideProcs bool
	}{
		{"snapshot.golden", false, false},
		{"snapshot-fahrenheit.golden", true, false},
		{"snapshot-no-processes.golden", false, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var b bytes.Buffer
			if err := writeSnapshot(&b, testDevices(), tt.fahrenheit, tt.hideProcs); err != nil {
				t.Fatal(err)
			}
			golden(t, tt.name, b.Bytes())
		})
	}
}

func TestWriteSnapshotNoCmdline(t *testing.T) {
	d := &device{procs: []process{{pid: 7, kind: compute}}}
	var b bytes.Buffer
	if err := writeSnapshot(&b, []*device{d}, false, false); err != nil {
		t.Fatal(err)
	}
	if want := `"cmdline": null,`; !bytes.Contains(b.Bytes(), []byte(want)) {
		t.Errorf("snapshot lacks %s:\n%s", want, b.Bytes())
	}
}

func TestToFahrenheit(t *testing.T) {
	tests := []struct{ c, f uint32 }{{0, 32}, {100, 212}, {71, 160}, {37, 99}}
	for _, tt := range tests {
		if got := toFahrenheit(tt.c); got != tt.f {
			t.Errorf("toFahrenheit(%d) = %d, want %d", tt.c, got, tt.f)
		}
	}
}
