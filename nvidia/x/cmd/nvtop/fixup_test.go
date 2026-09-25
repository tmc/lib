package main

import "testing"

func TestFixup(t *testing.T) {
	const gib = 1 << 30
	tests := []struct {
		name string
		dev  device
		want device // only the derived fields are compared
	}{
		{
			name: "process utilization sums and caps",
			dev: device{procs: []process{
				{gpuUtil: some[uint32](70)},
				{gpuUtil: some[uint32](50)},
			}},
			want: device{gpuUtil: some[uint32](100)},
		},
		{
			name: "reported utilization wins when larger",
			dev: device{
				gpuUtil: some[uint32](40),
				procs:   []process{{gpuUtil: some[uint32](10)}},
			},
			want: device{gpuUtil: some[uint32](40)},
		},
		{
			name: "process sum wins when larger",
			dev: device{
				gpuUtil: some[uint32](5),
				procs:   []process{{gpuUtil: some[uint32](10)}, {gpuUtil: some[uint32](15)}},
			},
			want: device{gpuUtil: some[uint32](25)},
		},
		{
			name: "encoder and decoder only when unreported",
			dev: device{
				encUtil: some[uint32](3),
				procs:   []process{{encUtil: some[uint32](20), decUtil: some[uint32](30)}},
			},
			want: device{encUtil: some[uint32](3), decUtil: some[uint32](30)},
		},
		{
			name: "unified memory use is the process sum",
			dev: device{
				memTotal: some[uint64](8 * gib),
				procs:    []process{{gpuMem: some[uint64](1 * gib)}, {gpuMem: some[uint64](2 * gib)}},
			},
			want: device{
				memTotal: some[uint64](8 * gib),
				memUsed:  some[uint64](3 * gib),
				memFree:  some[uint64](5 * gib),
			},
		},
		{
			name: "reported memory use is kept",
			dev: device{
				memTotal: some[uint64](8 * gib),
				memUsed:  some[uint64](4 * gib),
				procs:    []process{{gpuMem: some[uint64](1 * gib)}},
			},
			want: device{memTotal: some[uint64](8 * gib), memUsed: some[uint64](4 * gib)},
		},
		{
			name: "process memory over the total is dropped",
			dev: device{
				memTotal: some[uint64](1 * gib),
				procs:    []process{{gpuMem: some[uint64](2 * gib)}},
			},
			want: device{memTotal: some[uint64](1 * gib)},
		},
		{
			name: "effective load",
			dev: device{
				gpuUtil:  some[uint32](80),
				power:    some[uint32](50000),
				powerMax: some[uint32](100000),
			},
			want: device{gpuUtil: some[uint32](80), effLoad: some[uint32](40)},
		},
		{
			name: "effective load capped",
			dev: device{
				gpuUtil:  some[uint32](100),
				power:    some[uint32](150000),
				powerMax: some[uint32](100000),
			},
			want: device{gpuUtil: some[uint32](100), effLoad: some[uint32](100)},
		},
		{
			name: "no effective load without a power limit",
			dev: device{
				gpuUtil:  some[uint32](100),
				power:    some[uint32](150000),
				powerMax: some[uint32](0),
			},
			want: device{gpuUtil: some[uint32](100)},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d := tt.dev
			fixup(&d)
			type derived struct {
				gpu, eff, enc, dec opt[uint32]
				total, used, free  opt[uint64]
			}
			got := derived{d.gpuUtil, d.effLoad, d.encUtil, d.decUtil, d.memTotal, d.memUsed, d.memFree}
			want := derived{tt.want.gpuUtil, tt.want.effLoad, tt.want.encUtil, tt.want.decUtil, tt.want.memTotal, tt.want.memUsed, tt.want.memFree}
			if got != want {
				t.Errorf("fixup:\ngot  %+v\nwant %+v", got, want)
			}
		})
	}
}

func TestFixupProcessShare(t *testing.T) {
	d := device{
		memTotal: some[uint64](1000),
		procs: []process{
			{gpuMem: some[uint64](125)},
			{gpuMem: some[uint64](1000)},
			{},
		},
	}
	fixup(&d)
	want := []opt[uint32]{some[uint32](13), some[uint32](100), {}}
	for i, p := range d.procs {
		if p.memPct != want[i] {
			t.Errorf("process %d: memPct = %+v, want %+v", i, p.memPct, want[i])
		}
	}
}
