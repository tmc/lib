package train

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

const (
	gb  = int64(1e9)
	gib = int64(1) << 30
)

// llama70BLayer is a Llama-3-70B decoder layer at 32K tokens per microbatch
// on a GB200 (8 TB/s HBM3e, 2.25 PFLOP/s dense BF16).
func llama70BLayer(t *testing.T) ([]ActivationNode, RematConstraints) {
	t.Helper()
	s := BlockShape{
		Tokens: 32768, Hidden: 8192, FFN: 28672, Heads: 64, KVHeads: 8, HeadDim: 128,
		ElemBytes: 2, HBMGBps: 8000, TensorTFLOPS: 2250,
	}
	nodes, err := TransformerBlockActivations(s)
	if err != nil {
		t.Fatal(err)
	}
	params := s.Hidden*(s.Heads*s.HeadDim+2*s.KVHeads*s.HeadDim) + s.Heads*s.HeadDim*s.Hidden + 3*s.Hidden*s.FFN
	flops := 2*float64(s.Tokens)*float64(params) + 2*float64(s.Tokens)*float64(s.Tokens)*float64(s.Heads*s.HeadDim)
	fwdUs := flops / (s.TensorTFLOPS * 1e6)
	return nodes, RematConstraints{
		MaxHBMBytes:       48 * gb,
		MaxC2CHostBytes:   512 * gib,
		C2CBandwidthGBps:  450,
		LayerFwdComputeUs: fwdUs,
		LayerBwdComputeUs: 2 * fwdUs,
		NumLayers:         64,
	}
}

func TestSolveJointRematC2CLlama70B(t *testing.T) {
	nodes, c := llama70BLayer(t)
	var full int64
	for _, n := range nodes {
		full += n.Bytes
	}
	full *= int64(c.NumLayers)
	if full <= 192*gb {
		t.Fatalf("full activation storage %d bytes fits in 192 GB; test needs a config that does not", full)
	}
	p, err := SolveJointRematC2C(nodes, c)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("full=%.0f GB peakHBM=%.1f GB host=%.0f GB offload/layer=%.2f GB recompute=%.0f us (%.2f%%) stall=%.0f us",
		float64(full)/1e9, float64(p.PeakHBMBytes)/1e9, float64(p.PeakC2CHostBytes)/1e9,
		float64(p.OffloadBytesPerLayer)/1e9, p.RecomputeOverheadUs, 100*p.RecomputeFraction(c), p.ExposedC2CStallUs)
	if p.PeakHBMBytes > c.MaxHBMBytes {
		t.Errorf("PeakHBMBytes = %d, want <= %d", p.PeakHBMBytes, c.MaxHBMBytes)
	}
	if p.PeakC2CHostBytes > c.MaxC2CHostBytes {
		t.Errorf("PeakC2CHostBytes = %d, want <= %d", p.PeakC2CHostBytes, c.MaxC2CHostBytes)
	}
	if p.ExposedC2CStallUs != 0 {
		t.Errorf("ExposedC2CStallUs = %v, want 0", p.ExposedC2CStallUs)
	}
	// Block checkpointing recomputes the whole forward: 1/3 of fwd+bwd.
	if f := p.RecomputeFraction(c); f >= 0.04 {
		t.Errorf("recompute fraction = %.3f, want < 0.04", f)
	}
	for _, n := range nodes {
		if p.Decisions[n.Name] == PolicyRematInTMEM && !n.CanRematTMEM {
			t.Errorf("%s rematerialized but CanRematTMEM is false", n.Name)
		}
	}
	// 512 GiB of host memory cannot hold every activation of 64 layers,
	// so some cheap activation must be recomputed.
	if p.RecomputeOverheadUs == 0 {
		t.Errorf("no activation rematerialized, but offloading all needs %d host bytes", full)
	}
}

func TestSolveJointRematC2CBudgets(t *testing.T) {
	nodes, base := llama70BLayer(t)
	tests := []struct {
		name    string
		mutate  func(*RematConstraints)
		wantErr error
		check   func(t *testing.T, p RematPlan)
	}{
		{
			name:    "no_c2c_48GB",
			mutate:  func(c *RematConstraints) { c.C2CBandwidthGBps = 0 },
			wantErr: ErrHBMBudgetExceeded,
		},
		{
			name: "no_c2c_large_hbm",
			mutate: func(c *RematConstraints) {
				c.C2CBandwidthGBps = 0
				c.MaxHBMBytes = 1 << 50
			},
			check: func(t *testing.T, p RematPlan) {
				for name, pol := range p.Decisions {
					if pol == PolicyOffloadC2CAsync {
						t.Errorf("%s offloaded with zero C2C bandwidth", name)
					}
				}
				if p.PeakC2CHostBytes != 0 {
					t.Errorf("PeakC2CHostBytes = %d, want 0", p.PeakC2CHostBytes)
				}
			},
		},
		{
			name:    "tiny_hbm",
			mutate:  func(c *RematConstraints) { c.MaxHBMBytes = 1 * gb },
			wantErr: ErrHBMBudgetExceeded,
		},
		{
			name: "no_host_memory",
			mutate: func(c *RematConstraints) {
				c.MaxC2CHostBytes = 0
			},
			wantErr: ErrC2CHostBudgetExceeded,
		},
		{
			name: "short_window_stalls",
			mutate: func(c *RematConstraints) {
				// The two staging layers alone exceed this budget unless
				// offload is reduced, so a stall or remat must appear.
				c.LayerFwdComputeUs = 1000
				c.LayerBwdComputeUs = 2000
			},
			check: func(t *testing.T, p RematPlan) {
				if p.ExposedC2CStallUs == 0 && p.OffloadBytesPerLayer > int64(1000*450e3) {
					t.Errorf("offload %d bytes in 1 ms at 450 GB/s reported no stall", p.OffloadBytesPerLayer)
				}
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			c := base
			tc.mutate(&c)
			p, err := SolveJointRematC2C(nodes, c)
			if tc.wantErr != nil {
				if !errors.Is(err, tc.wantErr) {
					t.Fatalf("err = %v, want %v", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if p.PeakHBMBytes > c.MaxHBMBytes {
				t.Errorf("PeakHBMBytes = %d > %d", p.PeakHBMBytes, c.MaxHBMBytes)
			}
			tc.check(t, p)
		})
	}
}

// TestSolveJointRematC2COptimal compares the branch-and-bound solver with
// exhaustive enumeration on random small layers.
func TestSolveJointRematC2COptimal(t *testing.T) {
	r := rand.New(rand.NewPCG(10, 10))
	for trial := range 300 {
		n := 1 + r.IntN(7)
		nodes := make([]ActivationNode, n)
		for i := range nodes {
			nodes[i] = ActivationNode{
				Name:         fmt.Sprintf("a%d", i),
				Bytes:        1 + r.Int64N(100),
				RecomputeUs:  r.Float64() * 10,
				CanRematTMEM: r.IntN(2) == 0,
			}
		}
		c := RematConstraints{
			MaxHBMBytes:       r.Int64N(800),
			MaxC2CHostBytes:   r.Int64N(1500),
			C2CBandwidthGBps:  float64(r.IntN(3)) * 1e-3, // 0, 1 or 2 bytes/us
			LayerFwdComputeUs: r.Float64() * 100,
			LayerBwdComputeUs: r.Float64() * 200,
			NumLayers:         1 + r.IntN(8),
		}
		got, err := SolveJointRematC2C(nodes, c)

		bestCost := math.Inf(1)
		pol := make([]ActivationPolicy, n)
		for code := range int(math.Pow(3, float64(n))) {
			x := code
			ok := true
			for i := range pol {
				pol[i] = ActivationPolicy(x % 3)
				x /= 3
				if pol[i] == PolicyRematInTMEM && !nodes[i].CanRematTMEM ||
					pol[i] == PolicyOffloadC2CAsync && c.C2CBandwidthGBps == 0 {
					ok = false
				}
			}
			if !ok {
				continue
			}
			p := evalRemat(nodes, pol, c)
			if p.PeakHBMBytes > c.MaxHBMBytes || p.PeakC2CHostBytes > c.MaxC2CHostBytes {
				continue
			}
			bestCost = min(bestCost, p.RecomputeOverheadUs+p.ExposedC2CStallUs)
		}
		if math.IsInf(bestCost, 1) {
			if err == nil {
				t.Fatalf("trial %d: solver found plan %+v, exhaustive search found none", trial, got)
			}
			continue
		}
		if err != nil {
			t.Fatalf("trial %d: %v, exhaustive best cost %v", trial, err, bestCost)
		}
		if cost := got.RecomputeOverheadUs + got.ExposedC2CStallUs; math.Abs(cost-bestCost) > 1e-9 {
			t.Fatalf("trial %d: cost %v, exhaustive best %v", trial, cost, bestCost)
		}
	}
}

func TestSolveJointRematC2CInvalid(t *testing.T) {
	ok := []ActivationNode{{Name: "x", Bytes: 1}}
	c := RematConstraints{MaxHBMBytes: 10, NumLayers: 1}
	tests := []struct {
		name  string
		nodes []ActivationNode
		c     RematConstraints
	}{
		{"empty", nil, c},
		{"zero_layers", ok, RematConstraints{MaxHBMBytes: 10}},
		{"dup", []ActivationNode{{Name: "x", Bytes: 1}, {Name: "x", Bytes: 1}}, c},
		{"zero_bytes", []ActivationNode{{Name: "x"}}, c},
		{"too_many", make([]ActivationNode, maxRematNodes+1), c},
		{"negative_bw", ok, RematConstraints{MaxHBMBytes: 10, NumLayers: 1, C2CBandwidthGBps: -1}},
	}
	for _, tc := range tests {
		if _, err := SolveJointRematC2C(tc.nodes, tc.c); err == nil {
			t.Errorf("%s: no error", tc.name)
		}
	}
}

func TestPrefetchSchedule(t *testing.T) {
	plan := RematPlan{OffloadBytesPerLayer: 1}
	for _, layers := range []int{1, 2, 3, 8} {
		ops, err := plan.PrefetchSchedule(layers)
		if err != nil {
			t.Fatal(err)
		}
		if err := VerifyPrefetchSchedule(ops, layers); err != nil {
			t.Fatalf("layers=%d: %v", layers, err)
		}
	}
	ops, err := RematPlan{}.PrefetchSchedule(4)
	if err != nil {
		t.Fatal(err)
	}
	for _, op := range ops {
		if op.Stream != "compute" {
			t.Errorf("no-offload schedule has %s op on stream %s", op.Kind, op.Stream)
		}
	}
	if err := VerifyPrefetchSchedule(ops, 4); err != nil {
		t.Fatal(err)
	}
}

func TestVerifyPrefetchScheduleMutants(t *testing.T) {
	const layers = 6
	dropWait := func(kind string, layer int, event string) func([]C2CStreamOp) []C2CStreamOp {
		return func(ops []C2CStreamOp) []C2CStreamOp {
			for i, op := range ops {
				if op.Kind == kind && op.Layer == layer {
					ops[i].Wait = slices.DeleteFunc(slices.Clone(op.Wait), func(w string) bool { return w == event })
				}
			}
			return ops
		}
	}
	tests := []struct {
		name   string
		mutate func([]C2CStreamOp) []C2CStreamOp
	}{
		{"fwd_overwrites_unoffloaded_slot", dropWait("fwd", 3, "offload1")},
		{"prefetch_overwrites_unconsumed_slot", dropWait("prefetch", 1, "bwd3")},
		{"bwd_without_prefetch", dropWait("bwd", 2, "prefetch2")},
		{"prefetch_before_offload", func(ops []C2CStreamOp) []C2CStreamOp {
			// Move the first prefetch ahead of every op on the c2c stream.
			i := slices.IndexFunc(ops, func(op C2CStreamOp) bool { return op.Kind == "prefetch" })
			p := ops[i]
			p.Wait = nil
			ops = slices.Delete(ops, i, i+1)
			return slices.Insert(ops, 0, p)
		}},
		{"deadlock", func(ops []C2CStreamOp) []C2CStreamOp {
			ops[0].Wait = append(ops[0].Wait, "bwd0")
			return ops
		}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ops, err := RematPlan{OffloadBytesPerLayer: 1}.PrefetchSchedule(layers)
			if err != nil {
				t.Fatal(err)
			}
			if err := VerifyPrefetchSchedule(tc.mutate(ops), layers); err == nil {
				t.Fatal("mutated schedule verified")
			} else {
				t.Log(err)
			}
		})
	}
}

func ExampleSolveJointRematC2C() {
	nodes := []ActivationNode{
		{Name: "norm", OpKind: "RMSNorm", Bytes: 1 << 30, RecomputeUs: 250, CanRematTMEM: true},
		{Name: "qkv", OpKind: "QKVProj", Bytes: 3 << 30},
		{Name: "lse", OpKind: "AttnLSE", Bytes: 16 << 20},
	}
	p, err := SolveJointRematC2C(nodes, RematConstraints{
		MaxHBMBytes:       16 << 30,
		MaxC2CHostBytes:   512 << 30,
		C2CBandwidthGBps:  450,
		LayerFwdComputeUs: 10000,
		LayerBwdComputeUs: 20000,
		NumLayers:         32,
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	for _, n := range nodes {
		fmt.Println(n.Name, p.Decisions[n.Name])
	}
	fmt.Printf("peak HBM %d MiB, stall %.0f us\n", p.PeakHBMBytes>>20, p.ExposedC2CStallUs)
	// Output:
	// norm offload_c2c
	// qkv offload_c2c
	// lse store_hbm
	// peak HBM 8704 MiB, stall 0 us
}
