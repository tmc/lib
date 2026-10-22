package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"hash/fnv"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/examples/ds4-min/dsv4shape"
	"github.com/tmc/lib/gguf"
)

// shapeSeedVersion names the seeded tensor directory and byte recipe. The
// independent C generator and oracle in testdata/proseed emit the same value.
const shapeSeedVersion = "pro-role-alias-v3-header1894"

// The retained 8MiB header of antirez/deepseek-v4-gguf revision
// f71f23d552d664e523b422157b2befbf74040380,
// DeepSeek-V4-Pro-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-Instruct-imatrix-0813.gguf.
const (
	proHeaderSHA256    = "f9789d879496430297ec4e9d1390d308dfcd96e99a0dae5c148a25b637287721"
	proHeaderFileBytes = 464627334560
	proHeaderTensors   = 1894
)

// Repeated layers borrow identical immutable operands by role/type/shape.
// Every backing still has its complete admitted physical byte extent,
// including all384 experts. No expert IDs, dimensions or consumers are shrunk.
// The tensor directory is the real Pro header's: hash layers carry
// ffn_gate_tid2eid and no exp_probs_b.bias. The representation qualifies only
// a seeded61-layer graph, not a checkpoint.
type shapeSeedPlan struct {
	config        dsv4shape.Config
	binding       *dsv4shape.Binding
	specs         []dsv4shape.TensorSpec
	kinds         map[string]uint32
	roles         map[string]string
	physicalBytes uint64
}

// shapeSeedOwnedOperand reports whether name is a typed operand the CUDA
// checkpoint uploads and frees independently. Such operands never alias.
func shapeSeedOwnedOperand(name string) bool {
	return strings.Contains(name, ".hc_") || strings.HasPrefix(name, "output_hc_") || strings.HasSuffix(name, "_norm.weight") || strings.HasSuffix(name, ".attn_sinks.weight") || strings.HasSuffix(name, "_ape.weight")
}

func shapeSeedKey(name string, kind uint32, dimensions []uint64) string {
	role := name
	if strings.HasPrefix(role, "blk.") {
		parts := strings.SplitN(role, ".", 3)
		role = parts[2]
	}
	if shapeSeedOwnedOperand(name) {
		role = name
	}
	return fmt.Sprintf("%s/%d/%v", role, kind, dimensions)
}

// shapeSeedHashBias reports whether spec is an exp_probs_b.bias on a hash
// layer. The real Pro header has none there, and upstream and the CUDA FFN
// route hash layers by ffn_gate_tid2eid without reading the bias.
func shapeSeedHashBias(c dsv4shape.Config, name string) bool {
	var layer int
	if _, err := fmt.Sscanf(name, "blk.%d.exp_probs_b.bias", &layer); err != nil || name != fmt.Sprintf("blk.%d.exp_probs_b.bias", layer) {
		return false
	}
	l, err := c.Layer(layer)
	return err == nil && l.HashRouter
}

func newShapeSeedPlan() (shapeSeedPlan, error) {
	c, err := dsv4shape.For(dsv4shape.Pro)
	if err != nil {
		return shapeSeedPlan{}, err
	}
	specs, err := c.Weights()
	if err != nil {
		return shapeSeedPlan{}, err
	}
	p := shapeSeedPlan{config: c, kinds: make(map[string]uint32), roles: make(map[string]string)}
	g := new(gguf.File)
	offsets := make(map[string]uint64)
	for _, spec := range specs {
		if shapeSeedHashBias(c, spec.Name) {
			continue
		}
		kind := spec.Types[0]
		switch {
		case strings.HasSuffix(spec.Name, "ffn_gate_exps.weight"), strings.HasSuffix(spec.Name, "ffn_up_exps.weight"):
			kind = 16
		case strings.HasSuffix(spec.Name, "ffn_down_exps.weight"):
			kind = 10
		}
		tensor := gguf.Tensor{Name: spec.Name, TypeID: kind, Dimensions: spec.Dimensions}
		n, err := tensor.Bytes()
		if err != nil {
			return p, err
		}
		key := shapeSeedKey(spec.Name, kind, spec.Dimensions)
		offset, ok := offsets[key]
		if !ok {
			if p.physicalBytes > math.MaxUint64-31 {
				return p, fmt.Errorf("seed alignment overflow")
			}
			offset = (p.physicalBytes + 31) &^ uint64(31)
			p.physicalBytes = offset
			if n > math.MaxUint64-p.physicalBytes {
				return p, fmt.Errorf("seed extent overflow")
			}
			p.physicalBytes += n
			offsets[key] = offset
		}
		tensor.Offset = offset
		p.specs = append(p.specs, spec)
		p.kinds[spec.Name] = kind
		p.roles[spec.Name] = key
		g.Tensors = append(g.Tensors, tensor)
	}
	g.Size = p.physicalBytes
	p.binding, err = c.Bind(g)
	return p, err
}

// Byte generation is independent of the production kernels. A small bounded
// staging buffer fills complete backings; it is never a shortened tensor.
// The same generator specification must feed the independently executed C
// oracle. Its bytes, binary and outputs require separate sealed receipts.
func shapeSeedChunk(key string, kind uint32, offset uint64, dst []byte) error {
	hash := fnv.New32a()
	_, _ = hash.Write([]byte(key))
	seed := uint64(hash.Sum32())
	switch kind {
	case 0:
		if offset%4 != 0 || len(dst)%4 != 0 {
			return fmt.Errorf("unaligned F32 seed")
		}
		for i := 0; i < len(dst); i += 4 {
			index := (offset + uint64(i)) / 4
			v := float32(1)
			if strings.Contains(key, "base.weight") {
				v = 0
			}
			if strings.Contains(key, "attn_sinks.weight") || strings.Contains(key, "exp_probs_b.bias") {
				v = float32(index%13) / 64
			}
			binary.LittleEndian.PutUint32(dst[i:], math.Float32bits(v))
		}
	case 1:
		if offset%2 != 0 || len(dst)%2 != 0 {
			return fmt.Errorf("unaligned F16 seed")
		}
		for i := 0; i < len(dst); i += 2 {
			index := (offset + uint64(i)) / 2
			bits := uint16(0x0800 + ((index+seed)%31)*8)
			if strings.HasPrefix(key, "token_embd.weight/") {
				bits = 0x2800 + uint16((index+seed)%31)*16
			}
			if (index+seed)%3 == 0 {
				bits |= 0x8000
			}
			binary.LittleEndian.PutUint16(dst[i:], bits)
		}
	case 26:
		if offset%4 != 0 || len(dst)%4 != 0 {
			return fmt.Errorf("unaligned I32 seed")
		}
		for i := 0; i < len(dst); i += 4 {
			binary.LittleEndian.PutUint32(dst[i:], uint32((offset/4+uint64(i/4))%384))
		}
	case 8, 16, 10:
		block := 34
		if kind == 16 {
			block = 66
		}
		if kind == 10 {
			block = 84
		}
		if offset%uint64(block) != 0 || len(dst)%block != 0 {
			return fmt.Errorf("unaligned quant seed")
		}
		for i := 0; i < len(dst); i += block {
			index := (offset + uint64(i)) / uint64(block)
			b := dst[i : i+block]
			for j := range b {
				b[j] = byte(seed + index*37 + uint64(j*13+7))
			}
			if kind == 8 {
				binary.LittleEndian.PutUint16(b, 0x0800)
				for j := 2; j < block; j++ {
					b[j] = byte((seed+index+uint64(j))%7 + 1)
				}
			} else if kind == 16 {
				binary.LittleEndian.PutUint16(b, 0x0c00)
			} else {
				binary.LittleEndian.PutUint16(b[80:], 0x0c00)
				binary.LittleEndian.PutUint16(b[82:], 0x0800)
			}
		}
	default:
		return fmt.Errorf("unsupported seed type %d", kind)
	}
	return nil
}

func TestShapeSeededStorageRepresentation(t *testing.T) {
	p, err := newShapeSeedPlan()
	if err != nil {
		t.Fatal(err)
	}
	if p.config.Layers != 61 || p.physicalBytes == 0 || p.physicalBytes > 16<<30 {
		t.Fatalf("invalid seeded physical extent %d", p.physicalBytes)
	}
	if len(p.specs) != proHeaderTensors || len(p.binding.Weights) != proHeaderTensors {
		t.Fatalf("seeded directory has %d specs and %d views, want %d", len(p.specs), len(p.binding.Weights), proHeaderTensors)
	}
	views := make(map[string]dsv4shape.WeightView)
	iq2, q2 := 0, 0
	for _, v := range p.binding.Weights {
		views[v.Name] = v
		if v.Offset%32 != 0 {
			t.Fatalf("unaligned seeded tensor %s offset=%d", v.Name, v.Offset)
		}
		if v.Offset > p.physicalBytes || v.Bytes > p.physicalBytes-v.Offset {
			t.Fatalf("truncated seeded tensor %s", v.Name)
		}
		if v.Type == 16 {
			iq2++
		}
		if v.Type == 10 {
			q2++
		}
	}
	if iq2 != 122 || q2 != 61 {
		t.Fatalf("missing full routed shapes: IQ2=%d Q2=%d", iq2, q2)
	}
	for layer := range p.config.Layers {
		l, _ := p.config.Layer(layer)
		_, bias := views[fmt.Sprintf("blk.%d.exp_probs_b.bias", layer)]
		hash, tid2eid := views[fmt.Sprintf("blk.%d.ffn_gate_tid2eid.weight", layer)]
		if l.HashRouter == bias || l.HashRouter != tid2eid {
			t.Fatalf("layer%d router tensors: hash=%v bias=%v tid2eid=%v", layer, l.HashRouter, bias, tid2eid)
		}
		if tid2eid && (hash.Type != 26 || hash.Bytes != 4*uint64(p.config.ExpertsUsed)*uint64(p.config.Vocab)) {
			t.Fatalf("layer%d tid2eid type=%d bytes=%d", layer, hash.Type, hash.Bytes)
		}
	}
	if views["blk.0.ffn_gate_exps.weight"].Offset != views["blk.60.ffn_gate_exps.weight"].Offset {
		t.Fatal("identical immutable role backing not shared")
	}
	if views["blk.0.ffn_gate_exps.weight"].Offset == views["blk.0.ffn_up_exps.weight"].Offset {
		t.Fatal("simultaneous gate/up operands alias")
	}
	for layer := range p.config.Layers {
		var ranges []mxfp4DeviceRange
		for _, name := range []string{"ffn_gate_inp.weight", "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"} {
			v := views[fmt.Sprintf("blk.%d.%s", layer, name)]
			ranges = append(ranges, mxfp4DeviceRange{address: 4096 + v.Offset, bytes: v.Bytes})
		}
		if err := checkMXFP4DeviceRanges(ranges...); err != nil {
			t.Fatalf("layer%d seeded FFN operand overlap: %v", layer, err)
		}
	}
	for _, kind := range []uint32{0, 1, 8, 16, 10, 26} {
		block := 4
		switch kind {
		case 1:
			block = 2
		case 8:
			block = 34
		case 16:
			block = 66
		case 10:
			block = 84
		}
		all := make([]byte, block*7)
		if err := shapeSeedChunk("seed-chunk-control", kind, 0, all); err != nil {
			t.Fatal(err)
		}
		split := make([]byte, len(all))
		if err := shapeSeedChunk("seed-chunk-control", kind, 0, split[:block*3]); err != nil {
			t.Fatal(err)
		}
		if err := shapeSeedChunk("seed-chunk-control", kind, uint64(block*3), split[block*3:]); err != nil {
			t.Fatal(err)
		}
		for i, v := range all {
			if v != split[i] {
				t.Fatalf("seed chunk seam kind=%d byte=%d", kind, i)
			}
		}
		if err := shapeSeedChunk("seed-chunk-control", kind, 1, all); err == nil {
			t.Fatalf("seed accepts unaligned kind=%d", kind)
		}
	}
	t.Logf("PRO_SEEDED_STORAGE full_layers=61 tensors=%d IQ2_tensors=122 Q2_tensors=61 physical_bytes=%d", len(p.specs), p.physicalBytes)
}

// proSeedGenerator returns the sealed independent C generator named by
// DS4_PRO_SEED_GENERATOR, rechecking its seal when the test ends.
func proSeedGenerator(t *testing.T) string {
	t.Helper()
	path := os.Getenv("DS4_PRO_SEED_GENERATOR")
	if path == "" {
		t.Skip("sealed independent C generator not requested")
	}
	seal := os.Getenv("DS4_PRO_SEED_GENERATOR_SHA256")
	check := func() {
		t.Helper()
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if len(seal) != 64 || fmt.Sprintf("%x", sha256.Sum256(data)) != seal {
			t.Fatal("independent C generator binary seal differs")
		}
	}
	check()
	t.Cleanup(check)
	return path
}

func proSeedRun(t *testing.T, generator string, args ...string) []byte {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, generator, args...).Output()
	if err != nil {
		t.Fatalf("independent C %v: %v", args, err)
	}
	return out
}

// This CPU-only gate invokes the independent C executable's bounded inspection
// modes. It never materializes tensor backings or opens a device.
func TestShapeSeededCEquivalence(t *testing.T) {
	generator := proSeedGenerator(t)
	p, err := newShapeSeedPlan()
	if err != nil {
		t.Fatal(err)
	}
	views := make(map[string]dsv4shape.WeightView)
	for _, v := range p.binding.Weights {
		views[v.Name] = v
	}
	var layout, samples strings.Builder
	seen := make(map[string]bool)
	for _, spec := range p.specs {
		view, ok := views[spec.Name]
		if !ok {
			t.Fatalf("missing admitted seed view %s", spec.Name)
		}
		dims := [3]uint64{}
		copy(dims[:], spec.Dimensions)
		key := p.roles[spec.Name]
		fmt.Fprintf(&layout, "%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%s\n", spec.Name, view.Type, len(spec.Dimensions), dims[0], dims[1], dims[2], view.Offset, view.Bytes, key)
		if seen[key] {
			continue
		}
		seen[key] = true
		block := uint64(4)
		switch view.Type {
		case 1:
			block = 2
		case 8:
			block = 34
		case 16:
			block = 66
		case 10:
			block = 84
		}
		n := min(view.Bytes, block*7)
		for _, offset := range []uint64{0, view.Bytes - n} {
			data := make([]byte, n)
			if err := shapeSeedChunk(key, view.Type, offset, data); err != nil {
				t.Fatal(err)
			}
			fmt.Fprintf(&samples, "%s\t%d\t%x\n", key, offset, data)
		}
	}
	for _, test := range []struct {
		mode, want string
	}{{"--inspect", layout.String()}, {"--samples", samples.String()}} {
		t.Run(test.mode, func(t *testing.T) {
			got := string(proSeedRun(t, generator, test.mode))
			if got != test.want {
				t.Fatalf("independent C %s differs from admitted Go plan/bytes at line %d", test.mode, firstLineDiff(got, test.want))
			}
		})
	}
	t.Logf("PRO_SEED_C_EQUIVALENCE layers=61 tensors=%d unique_backings=%d physical_bytes=%d", len(p.specs), len(seen), p.physicalBytes)
}

func firstLineDiff(a, b string) int {
	x, y := strings.Split(a, "\n"), strings.Split(b, "\n")
	for i := range min(len(x), len(y)) {
		if x[i] != y[i] {
			return i + 1
		}
	}
	return min(len(x), len(y)) + 1
}

// proHeader reads the sealed real Pro header named by DS4_PRO_HEADER. The
// header is the file's first 8MiB, so it is parsed at the full declared size;
// gguf.Read never reads tensor data.
func proHeader(t *testing.T) ([]byte, *gguf.File) {
	t.Helper()
	path := os.Getenv("DS4_PRO_HEADER")
	if path == "" {
		t.Skip("sealed real Pro header not requested")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprintf("%x", sha256.Sum256(data)) != proHeaderSHA256 {
		t.Fatal("real Pro header seal differs")
	}
	g, err := gguf.Read(bytes.NewReader(data), proHeaderFileBytes)
	if err != nil {
		t.Fatal(err)
	}
	if len(g.Tensors) != proHeaderTensors {
		t.Fatalf("real Pro header has %d tensors, want %d", len(g.Tensors), proHeaderTensors)
	}
	if _, ok := g.Metadata["general.alignment"]; ok {
		t.Fatal("real Pro header declares general.alignment; seeded data start assumes 32")
	}
	return data, g
}

// proDirectory renders a tensor directory as sorted name/type/dims rows.
func proDirectory(tensors []gguf.Tensor) []string {
	var rows []string
	for _, t := range tensors {
		rows = append(rows, fmt.Sprintf("%s\t%d\t%v", t.Name, t.TypeID, t.Dimensions))
	}
	slices.Sort(rows)
	return rows
}

func diffDirectory(t *testing.T, name string, got, want []string) {
	t.Helper()
	if slices.Equal(got, want) {
		return
	}
	gotSet, wantSet := make(map[string]bool), make(map[string]bool)
	for _, r := range got {
		gotSet[r] = true
	}
	for _, r := range want {
		wantSet[r] = true
	}
	var extra, missing []string
	for _, r := range got {
		if !wantSet[r] {
			extra = append(extra, r)
		}
	}
	for _, r := range want {
		if !gotSet[r] {
			missing = append(missing, r)
		}
	}
	t.Fatalf("%s tensor directory differs from real Pro header: %d rows, want %d; extra %d %q; missing %d %q", name, len(got), len(want), len(extra), head(extra, 4), len(missing), head(missing, 4))
}

func head(s []string, n int) []string { return s[:min(len(s), n)] }

// TestShapeSeededProHeader compares the seeded tensor directory, from the Go
// plan and from the independent C generator, against the real Pro header.
func TestShapeSeededProHeader(t *testing.T) {
	_, g := proHeader(t)
	want := proDirectory(g.Tensors)
	p, err := newShapeSeedPlan()
	if err != nil {
		t.Fatal(err)
	}
	var plan []gguf.Tensor
	for _, spec := range p.specs {
		plan = append(plan, gguf.Tensor{Name: spec.Name, TypeID: p.kinds[spec.Name], Dimensions: spec.Dimensions})
	}
	diffDirectory(t, "Go seed plan", proDirectory(plan), want)
	if os.Getenv("DS4_PRO_SEED_GENERATOR") == "" {
		t.Logf("PRO_SEED_HEADER go_tensors=%d header_tensors=%d c=not-requested", len(plan), len(g.Tensors))
		return
	}
	generator := proSeedGenerator(t)
	var c []gguf.Tensor
	for i, line := range strings.Split(strings.TrimSuffix(string(proSeedRun(t, generator, "--inspect")), "\n"), "\n") {
		f := strings.Split(line, "\t")
		if len(f) != 9 {
			t.Fatalf("C --inspect line %d has %d fields", i+1, len(f))
		}
		var n [5]uint64
		for j := range n {
			if n[j], err = strconv.ParseUint(f[1+j], 10, 64); err != nil {
				t.Fatalf("C --inspect line %d field %d: %v", i+1, 2+j, err)
			}
		}
		if n[1] < 1 || n[1] > 3 {
			t.Fatalf("C --inspect line %d rank %d", i+1, n[1])
		}
		c = append(c, gguf.Tensor{Name: f[0], TypeID: uint32(n[0]), Dimensions: n[2 : 2+n[1]]})
	}
	diffDirectory(t, "C --inspect", proDirectory(c), want)
	t.Logf("PRO_SEED_HEADER go_tensors=%d c_tensors=%d header_tensors=%d", len(plan), len(c), len(g.Tensors))
}

// TestShapeSeededCDirectoryBytes checks the C generator's GGUF prefix byte for
// byte: the sealed header's metadata records verbatim, the Go plan's tensor
// directory, and zero padding to the 32-byte data start upstream computes.
func TestShapeSeededCDirectoryBytes(t *testing.T) {
	data, g := proHeader(t)
	generator := proSeedGenerator(t)
	p, err := newShapeSeedPlan()
	if err != nil {
		t.Fatal(err)
	}
	// Metadata ends where the real header's first tensor record begins.
	first := g.Tensors[0].Name
	record := binary.LittleEndian.AppendUint64(nil, uint64(len(first)))
	record = append(record, first...)
	metadataEnd := bytes.Index(data[24:], record)
	if metadataEnd < 0 || bytes.Count(data[24:], record) != 1 {
		t.Fatal("real Pro header first tensor record not unique")
	}
	metadataEnd += 24
	views := make(map[string]dsv4shape.WeightView)
	for _, v := range p.binding.Weights {
		views[v.Name] = v
	}
	want := []byte("GGUF")
	want = binary.LittleEndian.AppendUint32(want, 3)
	want = binary.LittleEndian.AppendUint64(want, uint64(len(p.specs)))
	want = append(want, data[16:metadataEnd]...)
	for _, spec := range p.specs {
		want = binary.LittleEndian.AppendUint64(want, uint64(len(spec.Name)))
		want = append(want, spec.Name...)
		want = binary.LittleEndian.AppendUint32(want, uint32(len(spec.Dimensions)))
		for _, d := range spec.Dimensions {
			want = binary.LittleEndian.AppendUint64(want, d)
		}
		want = binary.LittleEndian.AppendUint32(want, p.kinds[spec.Name])
		want = binary.LittleEndian.AppendUint64(want, views[spec.Name].Offset)
	}
	for len(want)%32 != 0 {
		want = append(want, 0)
	}
	out := filepath.Join(t.TempDir(), "directory.gguf")
	proSeedRun(t, generator, "--directory", os.Getenv("DS4_PRO_HEADER"), out)
	got, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	if len(got)%32 != 0 {
		t.Fatalf("C directory ends at %d, not on the 32-byte data start", len(got))
	}
	if !bytes.Equal(got, want) {
		i := 0
		for i < min(len(got), len(want)) && got[i] == want[i] {
			i++
		}
		t.Fatalf("C directory bytes differ at %d: got %d bytes, want %d", i, len(got), len(want))
	}
	t.Logf("PRO_SEED_C_DIRECTORY bytes=%d metadata_end=%d tensors=%d sha256=%x", len(got), metadataEnd, len(p.specs), sha256.Sum256(got))
}
