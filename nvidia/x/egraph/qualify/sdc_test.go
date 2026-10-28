package qualify

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
	"github.com/tmc/lib/nvidia/x/egraph/tilebridge"
)

func TestFaultRecordEncoding(t *testing.T) {
	r := FaultRecord{
		Kind: FaultMBarrierTimeout, Corrected: true, Rank: 0x0102, PeerRank: 0x0304, LayerID: 0x0506,
		TileMIdx: 0x0708090a, TileNIdx: 0x0b0c0d0e, CorruptRow: -1, CorruptCol: 300,
		ObservedDiff: 1.5, ElapsedNs: 0x1122334455667788,
	}
	b, err := r.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	want := []byte{
		3, 1, 0x02, 0x01, 0x04, 0x03, 0x06, 0x05,
		0x0a, 0x09, 0x08, 0x07, 0x0e, 0x0d, 0x0c, 0x0b,
		0xff, 0xff, 0x2c, 0x01, 0x00, 0x00, 0xc0, 0x3f,
		0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11,
	}
	if !bytes.Equal(b, want) {
		t.Fatalf("MarshalBinary = % x\nwant            % x", b, want)
	}
	var got FaultRecord
	if err := got.UnmarshalBinary(b); err != nil {
		t.Fatal(err)
	}
	if got != r {
		t.Errorf("round trip = %+v, want %+v", got, r)
	}
	if _, err := (FaultRecord{CorruptRow: 40000}).MarshalBinary(); err == nil {
		t.Error("MarshalBinary accepted a row outside int16")
	}
	if err := got.UnmarshalBinary(b[:31]); err == nil {
		t.Error("UnmarshalBinary accepted 31 bytes")
	}
}

// tileGEMM computes d = a·b in FP32 in one of several accumulation orders,
// all within the γ_K bound VerifyTileABFT allows.
func tileGEMM(order string, m, n, k int, a, b, d []float32) {
	for i := range m {
		for j := range n {
			p := make([]float32, k)
			for kk := range k {
				p[kk] = float32(a[i*k+kk] * b[kk*n+j])
			}
			var acc float32
			switch order {
			case "sequential":
				for _, v := range p {
					acc = float32(acc + v)
				}
			case "reversed":
				for _, v := range slices.Backward(p) {
					acc = float32(acc + v)
				}
			case "chunk16-tree":
				var parts []float32
				for c := 0; c < k; c += 16 {
					var s float32
					for _, v := range p[c:min(k, c+16)] {
						s = float32(s + v)
					}
					parts = append(parts, s)
				}
				acc = FixedTreeSum(parts)
			case "fma":
				for kk := range k {
					acc = float32(math.FMA(float64(a[i*k+kk]), float64(b[kk*n+j]), float64(acc)))
				}
			default:
				panic(order)
			}
			d[i*n+j] = acc
		}
	}
}

// randomTile returns a and b with entries drawn from format.
func randomTile(r *rand.Rand, format string, m, n, k int) (a, b []float32) {
	draw := func() float32 {
		switch format {
		case "f32":
			return r.Float32()*2 - 1
		case "bf16":
			return tilebridge.BF16.Round(float32(r.NormFloat64()))
		case "e4m3":
			return tilebridge.E4M3.Round(float32(r.NormFloat64() * 8))
		case "nvfp4":
			return DecodeE2M1(uint8(r.IntN(16))) * float32(math.Ldexp(1, r.IntN(5)-2))
		case "cancel":
			// Large terms of both signs whose sums cancel.
			return float32(r.NormFloat64()) * float32(math.Ldexp(1, 12*r.IntN(2)))
		}
		panic(format)
	}
	a = make([]float32, m*k)
	b = make([]float32, k*n)
	for i := range a {
		a[i] = draw()
	}
	for i := range b {
		b[i] = draw()
	}
	return a, b
}

func TestVerifyTileABFTCleanTiles(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 14))
	formats := []string{"f32", "bf16", "e4m3", "nvfp4", "cancel"}
	orders := []string{"sequential", "reversed", "chunk16-tree", "fma"}
	outputs := []struct {
		name string
		rel  float32
		dt   tilebridge.DType
	}{
		{"f32", 0, tilebridge.F32},
		{"bf16", 1.0 / 256, tilebridge.BF16},
	}
	tiles := 0
	for i := range 1000 {
		m, n, k := 8+r.IntN(25), 16*(1+r.IntN(4)), 16*(1+r.IntN(8))
		if i%100 == 0 {
			m, n, k = 128, 256, 64
		}
		format := formats[i%len(formats)]
		order := orders[(i/len(formats))%len(orders)]
		out := outputs[i%len(outputs)]
		a, b := randomTile(r, format, m, n, k)
		d := make([]float32, m*n)
		tileGEMM(order, m, n, k, a, b, d)
		for j := range d {
			d[j] = out.dt.Round(d[j])
		}
		cfg := ABFTConfig{TileM: m, TileN: n, TileK: k, RelTolerance: out.rel}
		rec, err := VerifyTileABFT(a, b, d, cfg)
		if err != nil {
			t.Fatal(err)
		}
		if rec.Kind != FaultNone {
			t.Errorf("tile %d (%dx%dx%d %s inputs, %s order, %s output): false alarm %+v", i, m, n, k, format, order, out.name, rec)
		}
		tiles++
	}
	t.Logf("%d clean tiles, no false alarms", tiles)
}

// injectionOracle flips one random bit of one element of n clean tiles and
// returns how many flips changed the element by more than 2.1τ in both its
// row and column (or made it non-finite), and how many of those
// VerifyTileABFT failed to report at the exact coordinates.
func injectionOracle(t *testing.T, cfg ABFTConfig, n int, seed uint64) (detectable, missed int, byBit [32][3]int) {
	t.Helper()
	r := rand.New(rand.NewPCG(seed, 99))
	m, nn, k := cfg.TileM, cfg.TileN, cfg.TileK
	for range n {
		a, b := randomTile(r, "bf16", m, nn, k)
		d := make([]float32, m*nn)
		tileGEMM("chunk16-tree", m, nn, k, a, b, d)
		row, col, bit := r.IntN(m), r.IntN(nn), r.IntN(32)
		byBit[bit][2]++
		clean := d[row*nn+col]
		d[row*nn+col] = math.Float32frombits(math.Float32bits(clean) ^ 1<<bit)
		delta := math.Abs(float64(d[row*nn+col]) - float64(clean))
		rec, err := VerifyTileABFT(a, b, d, cfg)
		if err != nil {
			t.Fatal(err)
		}
		v := float64(d[row*nn+col])
		nonFinite := math.IsNaN(v) || math.IsInf(v, 0)
		_, _, rowTol, _, _, colTol := abftBounds(a, b, d, ABFTConfig{TileM: m, TileN: nn, TileK: k})
		if !nonFinite && delta <= 2.1*max(rowTol[row], colTol[col]) {
			continue
		}
		detectable++
		byBit[bit][0]++
		wantKind := FaultTensorCoreSDC
		if nonFinite {
			wantKind = FaultNaNInfGenerated
		}
		if rec.Kind != wantKind || rec.CorruptRow != int32(row) || rec.CorruptCol != int32(col) {
			missed++
			byBit[bit][1]++
		}
	}
	return detectable, missed, byBit
}

func TestVerifyTileABFTFaultInjection(t *testing.T) {
	cfg := ABFTConfig{TileM: 32, TileN: 64, TileK: 64}
	const n = 1000
	detectable, missed, byBit := injectionOracle(t, cfg, n, 7)
	if missed != 0 {
		t.Errorf("%d of %d detectable flips missed or mislocated", missed, detectable)
	}
	// Exponent and sign flips change an element by far more than the
	// rounding tolerance, except on elements near zero.
	var tried, found int
	var lines []string
	for bit, c := range byBit {
		if bit >= 23 {
			tried += c[2]
			found += c[0]
		}
		lines = append(lines, fmt.Sprintf("bit %d: %d/%d", bit, c[0], c[2]))
	}
	if found < tried*95/100 {
		t.Errorf("%d of %d exponent and sign flips detectable, want at least 95%%", found, tried)
	}
	t.Logf("%d/%d flips detectable and all located; per bit: %s", detectable, n, strings.Join(lines, ", "))

	// A tolerance of 1e30 hides every finite flip.
	loose := cfg
	loose.RelTolerance = 1e30
	if _, missed, _ := injectionOracle(t, loose, 200, 7); missed == 0 {
		t.Error("RelTolerance 1e30 still detected every injected fault")
	}
}

func TestVerifyTileABFTCases(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 3))
	const m, n, k = 16, 32, 32
	cfg := ABFTConfig{TileM: m, TileN: n, TileK: k}
	a, b := randomTile(r, "f32", m, n, k)
	clean := make([]float32, m*n)
	tileGEMM("sequential", m, n, k, a, b, clean)
	tests := []struct {
		name     string
		at       int
		set      func(float32) float32
		kind     FaultKind
		row, col int32
	}{
		{"clean", 0, func(v float32) float32 { return v }, FaultNone, -1, -1},
		{"exponent msb", 5*n + 7, func(v float32) float32 { return math.Float32frombits(math.Float32bits(v) ^ 1<<30) }, FaultNaNInfGenerated, 5, 7},
		{"sign", 9*n + 31, func(v float32) float32 { return -v - 1 }, FaultTensorCoreSDC, 9, 31},
		{"plus one", 0, func(v float32) float32 { return v + 1 }, FaultTensorCoreSDC, 0, 0},
		{"nan", 15*n + 3, func(float32) float32 { return float32(math.NaN()) }, FaultNaNInfGenerated, 15, 3},
		{"one ulp", 4*n + 4, func(v float32) float32 { return math.Nextafter32(v, 10) }, FaultNone, -1, -1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d := slices.Clone(clean)
			d[tt.at] = tt.set(d[tt.at])
			rec, err := VerifyTileABFT(a, b, d, cfg)
			if err != nil {
				t.Fatal(err)
			}
			if rec.Kind != tt.kind || rec.CorruptRow != tt.row || rec.CorruptCol != tt.col {
				t.Errorf("VerifyTileABFT = %v at (%d, %d), want %v at (%d, %d)", rec.Kind, rec.CorruptRow, rec.CorruptCol, tt.kind, tt.row, tt.col)
			}
		})
	}
	if _, err := VerifyTileABFT(a, b, clean[:10], cfg); err == nil {
		t.Error("VerifyTileABFT accepted a short tile")
	}
	bad := slices.Clone(a)
	bad[0] = float32(math.Inf(1))
	if _, err := VerifyTileABFT(bad, b, clean, cfg); err == nil {
		t.Error("VerifyTileABFT accepted a non-finite input")
	}
}

func TestRunTileWithABFT(t *testing.T) {
	r := rand.New(rand.NewPCG(4, 4))
	const m, n, k = 8, 16, 16
	a, b := randomTile(r, "f32", m, n, k)
	calls := 0
	transient := func(a, b, d []float32) {
		tileGEMM("sequential", m, n, k, a, b, d)
		if calls++; calls == 1 {
			d[3*n+2] += 100
		}
	}
	persistent := func(a, b, d []float32) {
		tileGEMM("sequential", m, n, k, a, b, d)
		d[3*n+2] += 100
	}
	tests := []struct {
		name      string
		gemm      func(a, b, d []float32)
		replay    bool
		kind      FaultKind
		corrected bool
	}{
		{"transient, no replay", transient, false, FaultTensorCoreSDC, false},
		{"transient, replay", transient, true, FaultTensorCoreSDC, true},
		{"persistent, replay", persistent, true, FaultTensorCoreSDC, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			calls = 0
			cfg := ABFTConfig{TileM: m, TileN: n, TileK: k, AutoRecomputeOnSDC: tt.replay}
			d, rec, err := RunTileWithABFT(a, b, cfg, tt.gemm)
			if err != nil {
				t.Fatal(err)
			}
			if rec.Kind != tt.kind || rec.Corrected != tt.corrected || rec.CorruptRow != 3 || rec.CorruptCol != 2 {
				t.Errorf("record = %+v, want kind %v corrected %v at (3, 2)", rec, tt.kind, tt.corrected)
			}
			if again, _ := VerifyTileABFT(a, b, d, cfg); (again.Kind == FaultNone) != tt.corrected {
				t.Errorf("returned tile check = %v, corrected = %v", again.Kind, tt.corrected)
			}
		})
	}
}

func TestEmitABFTAndWatchdogPTX(t *testing.T) {
	r := rand.New(rand.NewPCG(5, 5))
	const m, n, k = 4, 64, 32
	cfg := ABFTConfig{TileM: m, TileN: n, TileK: k, WatchdogTimeoutNs: 5000}
	ptx, err := EmitABFTAndWatchdogPTX(cfg, "qs")
	if err != nil {
		t.Fatal(err)
	}
	if vs, err := AuditPTXDeterminism(ptx, DeterministicConfig{}); err != nil || len(vs) != 0 {
		t.Errorf("AuditPTXDeterminism(emitted) = %v, %v; want no violations", vs, err)
	}
	a, b := randomTile(r, "f32", m, n, k)
	d := make([]float32, m*n)
	tileGEMM("chunk16-tree", m, n, k, a, b, d)
	// Column n of TMEM holds the checksum A·(B·1), as an MMA with B
	// extended by its row sums writes it.
	bsum := make([]float32, k)
	for kk := range k {
		for j := range n {
			bsum[kk] += b[kk*n+j]
		}
	}
	tmem := map[uint32]uint32{}
	for i := range m {
		var ref float32
		for kk := range k {
			ref = float32(ref + float32(a[i*k+kk]*bsum[kk]))
		}
		for j := range n {
			tmem[uint32(i)<<16|uint32(j)] = math.Float32bits(d[i*n+j])
		}
		tmem[uint32(i)<<16|uint32(n)] = math.Float32bits(ref)
	}
	corrupt := func(row, col int, f func(float32) float32) map[uint32]uint32 {
		c := maps(tmem)
		ad := uint32(row)<<16 | uint32(col)
		c[ad] = math.Float32bits(f(math.Float32frombits(c[ad])))
		return c
	}
	abftRow := funcBody(t, ptx, "qs_abft_row")
	mutant := strings.Replace(abftRow, "@%p mov.u32 %code, 1;", "", 1)
	tests := []struct {
		name string
		body string
		tmem map[uint32]uint32
		row  int
		want uint64
	}{
		{"clean", abftRow, tmem, 2, 0},
		{"exponent flip", abftRow, corrupt(2, 17, func(v float32) float32 { return math.Float32frombits(math.Float32bits(v) ^ 1<<27) }), 2, 1},
		{"plus one", abftRow, corrupt(1, 63, func(v float32) float32 { return v + 1 }), 1, 1},
		{"other row clean", abftRow, corrupt(1, 63, func(v float32) float32 { return v + 1 }), 0, 0},
		{"inf", abftRow, corrupt(3, 0, func(float32) float32 { return float32(math.Inf(-1)) }), 3, 2},
		{"mutant without compare", mutant, corrupt(1, 63, func(v float32) float32 { return v + 1 }), 1, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := &ptxSim{params: map[string]uint64{"p_taddr": uint64(tt.row) << 16}, tmem: tt.tmem}
			s.run(t, tt.body)
			if got := s.params["r_out"]; got != tt.want {
				t.Errorf("qs_abft_row(row %d) = %d, want %d", tt.row, got, tt.want)
			}
		})
	}

	wait := funcBody(t, ptx, "qs_wait")
	for _, tt := range []struct {
		name    string
		polls   int // polls before the barrier completes; -1 never
		timeout uint64
		late    bool
	}{
		{"completes", 3, 5000, false},
		{"never", -1, 5000, true},
		{"completes after timeout", 100, 5000, true},
	} {
		t.Run("wait "+tt.name, func(t *testing.T) {
			var now uint64 = 1_000_000
			polls := 0
			s := &ptxSim{
				params:  map[string]uint64{"p_bar": 0x100, "p_parity": 1, "p_timeout": tt.timeout},
				timer:   func() uint64 { now += 100; return now },
				tryWait: func() bool { polls++; return tt.polls >= 0 && polls > tt.polls },
			}
			s.run(t, wait)
			el := s.params["r_out"]
			if (el != 0) != tt.late || tt.late && el <= tt.timeout {
				t.Errorf("qs_wait returned %d after %d polls, want timed out = %v", el, polls, tt.late)
			}
		})
	}

	// Report three faults into a ring with room for two.
	report := funcBody(t, ptx, "qs_report")
	s := &ptxSim{mem: map[uint64]uint32{0x1000: 0, 0x1004: 2}}
	recs := []FaultRecord{
		{Kind: FaultMBarrierTimeout, Rank: 3, PeerRank: 4, LayerID: 7, TileMIdx: 1, TileNIdx: 2, CorruptRow: -1, CorruptCol: -1, ElapsedNs: 6100},
		{Kind: FaultTensorCoreSDC, Corrected: true, Rank: 3, TileMIdx: 9, CorruptRow: 5, CorruptCol: 77, ObservedDiff: -0.25},
		{Kind: FaultNaNInfGenerated},
	}
	for _, rec := range recs {
		enc, err := rec.MarshalBinary()
		if err != nil {
			t.Fatal(err)
		}
		w := func(i int) uint64 { return uint64(binary.LittleEndian.Uint32(enc[4*i:])) }
		s.regs = nil
		s.params = map[string]uint64{"p_ring": 0x1000, "p_kind_flags_rank": w(0), "p_peer_layer": w(1), "p_tile_m": w(2),
			"p_tile_n": w(3), "p_row_col": w(4), "p_diff": w(5), "p_elapsed": rec.ElapsedNs}
		s.run(t, report)
	}
	if s.mem[0x1000] != 3 {
		t.Errorf("ring counter = %d, want 3", s.mem[0x1000])
	}
	for i, want := range recs {
		buf := make([]byte, FaultRecordSize)
		written := false
		for j := range 8 {
			v, ok := s.mem[0x1000+32+uint64(32*i+4*j)]
			written = written || ok
			binary.LittleEndian.PutUint32(buf[4*j:], v)
		}
		if i >= 2 {
			if written {
				t.Errorf("record %d written past the ring capacity", i)
			}
			continue
		}
		var got FaultRecord
		if err := got.UnmarshalBinary(buf); err != nil {
			t.Fatal(err)
		}
		if got != want {
			t.Errorf("ring record %d = %+v, want %+v", i, got, want)
		}
	}

	if _, err := EmitABFTAndWatchdogPTX(ABFTConfig{TileM: 128, TileN: 504, TileK: 64}, "x"); err == nil {
		t.Error("EmitABFTAndWatchdogPTX accepted TileN 504 (no room for the checksum column)")
	}
	wd, err := EmitWatchdogPTX(cfg, "qs")
	if err != nil || !strings.Contains(wd, ".target sm_90") || strings.Contains(wd, "tcgen05") {
		t.Errorf("EmitWatchdogPTX: %v\n%s", err, wd)
	}
}

func maps(m map[uint32]uint32) map[uint32]uint32 {
	c := make(map[uint32]uint32, len(m))
	for k, v := range m {
		c[k] = v
	}
	return c
}

func TestABFTCUDAHostEmu(t *testing.T) {
	if _, err := cudahostemu.Compiler(); err != nil {
		t.Skip(err)
	}
	r := rand.New(rand.NewPCG(6, 6))
	const m, n, k = 8, 32, 32
	cfg := ABFTConfig{TileM: m, TileN: n, TileK: k}
	src, err := ABFTCUDA(cfg)
	if err != nil {
		t.Fatal(err)
	}
	a, b := randomTile(r, "f32", m, n, k)
	tests := []struct {
		name     string
		row, col int
		mask     uint32
		kind     FaultKind
	}{
		{"clean", -1, 0, 0, FaultNone},
		{"exponent flip", 5, 9, 1 << 28, FaultTensorCoreSDC},
		{"to inf", 2, 31, 0, FaultNaNInfGenerated},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mask := tt.mask
			if tt.kind == FaultNaNInfGenerated {
				// Overwrite the exponent with all ones and the mantissa
				// with zeros: xor against the clean value gives ±Inf.
				d := make([]float32, m*n)
				tileGEMM("sequential", m, n, k, a, b, d)
				v := math.Float32bits(d[tt.row*n+tt.col])
				mask = v ^ (v&0x80000000 | 0x7f800000)
			}
			ring := make([]byte, 32+4*32)
			binary.LittleEndian.PutUint32(ring[4:], 4)
			out, err := cudahostemu.Run(t.Context(), t.TempDir(), cudahostemu.Program{
				Source: src,
				Launch: fmt.Sprintf("eg_emu_launch(dim3(1), dim3(%d), 0, qs_abft_gemm, a, b, d, ring, %d, %d, %d, %d, %d, %#xu, 11u)", m, m, n, k, tt.row, tt.col, mask),
				Buffers: []cudahostemu.Buffer{
					{Name: "a", CType: "float", Data: f32Bytes(a)},
					{Name: "b", CType: "float", Data: f32Bytes(b)},
					{Name: "d", CType: "float", Data: cudahostemu.Poison(4 * m * n)},
					{Name: "ring", CType: "uint32_t", Data: ring},
				},
			})
			if err != nil {
				t.Fatal(err)
			}
			got := out["ring"]
			count := binary.LittleEndian.Uint32(got)
			if tt.kind == FaultNone {
				if count != 0 {
					t.Fatalf("clean tile wrote %d fault records", count)
				}
				return
			}
			if count != 1 {
				t.Fatalf("ring holds %d records, want 1", count)
			}
			var rec FaultRecord
			if err := rec.UnmarshalBinary(got[32:64]); err != nil {
				t.Fatal(err)
			}
			if rec.Kind != tt.kind || rec.CorruptRow != int32(tt.row) || rec.CorruptCol != -1 || rec.TileMIdx != 11 {
				t.Errorf("record = %+v, want kind %v row %d", rec, tt.kind, tt.row)
			}
		})
	}
}

func ExampleVerifyTileABFT() {
	// D = A·B for a 2x2x2 tile, with D[1][0] corrupted.
	a := []float32{1, 2, 3, 4}
	b := []float32{5, 6, 7, 8}
	d := []float32{19, 22, 43 + 1, 50}
	rec, err := VerifyTileABFT(a, b, d, ABFTConfig{TileM: 2, TileN: 2, TileK: 2})
	if err != nil {
		panic(err)
	}
	fmt.Println(rec.Kind, rec.CorruptRow, rec.CorruptCol, rec.ObservedDiff)
	// Output: tensor-core-sdc 1 0 1
}
