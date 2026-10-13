package cudatma

import (
	"bytes"
	"crypto/sha256"
	"debug/elf"
	"embed"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"io"
	"math"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"unsafe"
)

// These bytes identify the fixture implementation in the executing test
// binary. Native admission must match this digest as well as the executable.
//
//go:embed tma.go lifetime.go multicast_source.go multicast_native_test.go multicast_qualification_test.go multicast_qualification_source_test.go multicast_qualification_owner_test.go multicast_qualification_native_test.go
var qualificationImplementation embed.FS

func qualificationDigest() string {
	h := sha256.New()
	for _, name := range []string{"tma.go", "lifetime.go", "multicast_source.go", "multicast_native_test.go", "multicast_qualification_test.go", "multicast_qualification_source_test.go", "multicast_qualification_owner_test.go", "multicast_qualification_native_test.go"} {
		b, _ := qualificationImplementation.ReadFile(name)
		fmt.Fprintf(h, "%s:%d:", name, len(b))
		h.Write(b)
	}
	return hex.EncodeToString(h.Sum(nil))
}

type qualificationCase struct {
	ID           string
	Rank         int
	Type         ElementType
	ElementBytes uint32
	Width        uint32
	Swizzle      uint32
	Mask         uint16
	Parity       bool
	Control      string
}

func qualificationCases() []qualificationCase {
	var cases []qualificationCase
	types := []struct {
		typ   ElementType
		bytes uint32
	}{{ElementTypeDefault, 2}, {ElementTypeDefault, 4}, {ElementTypeFloat16, 2}, {ElementTypeBfloat16, 2}, {ElementTypeFloat32, 4}}
	for rank := 1; rank <= 5; rank++ {
		for _, dt := range types {
			for sw := uint32(0); sw <= 128; {
				widths := []uint32{sw}
				if sw == 0 {
					widths = []uint32{16, 32, 48, 64, 80, 96, 112, 128}
				}
				for _, width := range widths {
					for _, mask := range []uint16{3, 2} {
						c := qualificationCase{Rank: rank, Type: dt.typ, ElementBytes: dt.bytes, Width: width, Swizzle: sw, Mask: mask}
						c.ID = fmt.Sprintf("rank%d-type%d-bytes%d-sw%d-width%d-mask%d", rank, dt.typ, dt.bytes, sw, width, mask)
						cases = append(cases, c)
						if width == 128 && sw == 0 || width == sw && sw != 0 {
							p := c
							p.Parity = true
							p.ID += "-parity"
							cases = append(cases, p)
						}
					}
				}
				if sw == 0 {
					sw = 32
				} else {
					sw *= 2
				}
			}
		}
	}
	for _, control := range []string{"wrong-swizzle", "single-destination"} {
		cases = append(cases, qualificationCase{ID: "rank2-f32-sw128-mask3-" + control, Rank: 2, Type: ElementTypeFloat32, ElementBytes: 4, Width: 128, Swizzle: 128, Mask: 3, Control: control})
	}
	return cases
}

func (c qualificationCase) config() MulticastConfig {
	d := Descriptor{Target: TargetSM100a, Rank: c.Rank, Base: 0x1000, ElementBytes: c.ElementBytes, ElementType: c.Type, Swizzle: c.Swizzle}
	d.Box[0] = c.Width / c.ElementBytes
	d.Dims[0] = uint64(d.Box[0])*2 + 1
	outer := [4]uint32{3, 2, 1, 2}
	stride := (d.Dims[0]*uint64(d.ElementBytes) + 31) &^ uint64(15)
	bytes := uint64(c.Width)
	for i := 1; i < c.Rank; i++ {
		d.Box[i] = outer[i-1]
		d.Dims[i] = uint64(d.Box[i]) + 1
		d.Strides[i-1] = stride
		stride = (stride*d.Dims[i] + 31) &^ uint64(15)
		bytes *= uint64(d.Box[i])
	}
	cfg := MulticastConfig{Descriptor: d, Cluster: [3]uint32{2, 1, 1}, SourceRank: 0, AvailableSharedBytes: bytes + 8, SharedLimitProvenance: "fixture allocation request; runtime device/kernel queries mandatory"}
	for rank := uint32(0); rank < 2; rank++ {
		if c.Mask&(1<<rank) != 0 {
			cfg.Destinations = append(cfg.Destinations, rank)
		}
		cfg.Windows = append(cfg.Windows, SharedWindow{Rank: rank, Descriptor: d, BarrierOffset: bytes, ReservedBytes: bytes + 8})
	}
	return cfg
}

func (c qualificationCase) coordinates(scenario int) [MaxRank]int32 {
	d := c.config().Descriptor
	var coords [MaxRank]int32
	for i := 0; i < c.Rank; i++ {
		switch scenario {
		case 1:
			coords[i] = int32(d.Dims[i] - 1)
		case 2:
			coords[i] = -1
		}
	}
	// Keep the contiguous coordinate on a 16-byte element boundary as well.
	if scenario == 1 {
		coords[0] = int32(d.Box[0]) * 2
	}
	if scenario == 2 {
		coords[0] = -int32(16 / d.ElementBytes)
	}
	return coords
}

func qualificationInput(d Descriptor, epoch uint32) ([]byte, error) {
	span := d.Dims[0] * uint64(d.ElementBytes)
	if d.Rank > 1 {
		span = d.Strides[d.Rank-2] * d.Dims[d.Rank-1]
	}
	if span > 8<<20 {
		return nil, fmt.Errorf("fixture input exceeds fixed corpus bound")
	}
	b := bytes.Repeat([]byte{0xad}, int(span))
	var visit func(int, uint64, uint64)
	visit = func(axis int, offset, tag uint64) {
		if axis == d.Rank {
			// Finite, format-specific bits with disjoint epoch exponent ranges.
			switch d.ElementBytes {
			case 2:
				base := uint16(0x3c00 + epoch*0x400)
				if d.ElementType == ElementTypeBfloat16 {
					base = uint16(0x3f00 + epoch*0x80)
				}
				binary.LittleEndian.PutUint16(b[offset:], base|uint16(tag%0x70))
			case 4:
				binary.LittleEndian.PutUint32(b[offset:], uint32(0x3f000000+epoch*0x800000)|uint32(tag%0x700000))
			}
			return
		}
		stride := uint64(d.ElementBytes)
		if axis > 0 {
			stride = d.Strides[axis-1]
		}
		for i := uint64(0); i < d.Dims[axis]; i++ {
			visit(axis+1, offset+i*stride, tag*131+i+1)
		}
	}
	visit(0, 0, 0)
	return b, nil
}

// Independent mixed-radix global oracle. It does not call any CUDA source
// reader, physical layout helper or actual-output-derived reference.
func qualificationOracle(d Descriptor, coords [MaxRank]int32, input []byte) ([]byte, error) {
	elements := uint64(1)
	for i := 0; i < d.Rank; i++ {
		elements *= uint64(d.Box[i])
	}
	if elements*uint64(d.ElementBytes) > maxMulticastTransactionBytes {
		return nil, fmt.Errorf("oracle transaction bound exceeded")
	}
	out := make([]byte, elements*uint64(d.ElementBytes))
	for index := uint64(0); index < elements; index++ {
		rem, offset, oob := index, uint64(0), false
		for axis := 0; axis < d.Rank; axis++ {
			q := rem % uint64(d.Box[axis])
			rem /= uint64(d.Box[axis])
			g := int64(coords[axis]) + int64(q)
			if g < 0 || uint64(g) >= d.Dims[axis] {
				oob = true
				continue
			}
			stride := uint64(d.ElementBytes)
			if axis > 0 {
				stride = d.Strides[axis-1]
			}
			if uint64(g) > math.MaxUint64/stride || offset > math.MaxUint64-uint64(g)*stride {
				return nil, fmt.Errorf("oracle offset overflow")
			}
			offset += uint64(g) * stride
		}
		if oob {
			continue
		}
		if offset+uint64(d.ElementBytes) > uint64(len(input)) {
			return nil, fmt.Errorf("oracle input span too small")
		}
		copy(out[index*uint64(d.ElementBytes):], input[offset:offset+uint64(d.ElementBytes)])
	}
	return out, nil
}

type qualificationArtifact struct{ CaseID, SourceFile, SourceSHA256, PTXFile, PTXSHA256, CubinFile, CubinSHA256, CompilerReceiptFile, CompilerReceiptSHA256 string }
type qualificationAdmission struct {
	Version                                                              int
	AuthorizationID, CandidateCommit, SourceSHA256, ExecutableSHA256     string
	Host, DeviceUUID, Target                                             string
	Ordinal                                                              int
	LockDir                                                              string
	LockPID                                                              int
	LockEpoch                                                            int64
	DriverFile, DriverSHA256                                             string
	DriverVersion                                                        int
	NVCCFile, NVCCSHA256, NVCCVersion                                    string
	OutputDir, QuarantineContractFile, QuarantineContractSHA256          string
	MaxChildren, MaxLaunches, ChildSeconds, DrainSeconds, OverallSeconds int
	AllocationBytes, ReceiptBytes                                        uint64
	Cases                                                                []qualificationArtifact
}

func digestString(s string, n int) bool {
	b, e := hex.DecodeString(s)
	return e == nil && len(b) == n && s == strings.ToLower(s)
}
func (a qualificationAdmission) validate() error {
	if a.Version != 1 || a.AuthorizationID == "" {
		return fmt.Errorf("explicit versioned native authorization required")
	}
	if !digestString(a.CandidateCommit, 20) || a.SourceSHA256 != qualificationDigest() || !digestString(a.ExecutableSHA256, 32) {
		return fmt.Errorf("candidate/source/executable pins required")
	}
	if a.Host == "" || !digestString(a.DeviceUUID, 16) || a.Target != TargetSM100a || a.Ordinal < 0 {
		return fmt.Errorf("exact host/UUID/ordinal/supported target required")
	}
	if !filepath.IsAbs(a.LockDir) || a.LockPID <= 1 || a.LockEpoch <= 0 {
		return fmt.Errorf("exact inherited lock owner binding required")
	}
	if !filepath.IsAbs(a.DriverFile) || !digestString(a.DriverSHA256, 32) || a.DriverVersion <= 0 || !filepath.IsAbs(a.NVCCFile) || !digestString(a.NVCCSHA256, 32) || a.NVCCVersion == "" {
		return fmt.Errorf("explicit driver/toolchain identity required")
	}
	if !filepath.IsAbs(a.OutputDir) || !filepath.IsAbs(a.QuarantineContractFile) || !digestString(a.QuarantineContractSHA256, 32) {
		return fmt.Errorf("explicit output and external quarantine contract required")
	}
	if a.MaxChildren <= 0 || a.MaxLaunches <= 0 || a.ChildSeconds <= 0 || a.DrainSeconds <= 0 || a.OverallSeconds <= 0 || a.ChildSeconds > 86400 || a.OverallSeconds > 86400 || a.DrainSeconds > a.ChildSeconds || a.ChildSeconds > a.OverallSeconds || a.AllocationBytes == 0 || a.ReceiptBytes == 0 {
		return fmt.Errorf("explicit finite reviewed execution budgets required")
	}
	if len(a.Cases) == 0 || len(a.Cases) > a.MaxChildren {
		return fmt.Errorf("case list exceeds reviewed child budget")
	}
	catalog := map[string]qualificationCase{}
	for _, c := range qualificationCases() {
		catalog[c.ID] = c
	}
	seen := map[string]bool{}
	launches := 0
	for _, artifact := range a.Cases {
		c, ok := catalog[artifact.CaseID]
		if !ok {
			return fmt.Errorf("unknown or unsafe non-dispatchable HOLD case %q", artifact.CaseID)
		}
		if seen[c.ID] {
			return fmt.Errorf("duplicate case %q", c.ID)
		}
		seen[c.ID] = true
		source, e := qualificationSource(c)
		if e != nil {
			return e
		}
		hash := sha256.Sum256([]byte(source))
		if artifact.SourceSHA256 != hex.EncodeToString(hash[:]) || !filepath.IsAbs(artifact.SourceFile) || !filepath.IsAbs(artifact.CompilerReceiptFile) || !digestString(artifact.CompilerReceiptSHA256, 32) || !filepath.IsAbs(artifact.PTXFile) || !digestString(artifact.PTXSHA256, 32) || !filepath.IsAbs(artifact.CubinFile) || !digestString(artifact.CubinSHA256, 32) {
			return fmt.Errorf("case %s requires exact source/PTX/cubin pins", c.ID)
		}
		if c.Parity {
			launches += 3
		} else if c.Control != "" {
			launches++
		} else {
			launches += 12
		}
	}
	if launches > a.MaxLaunches {
		return fmt.Errorf("case list exceeds reviewed launch budget")
	}
	return nil
}

func decodeQualificationAdmission(b []byte) (qualificationAdmission, error) {
	var a qualificationAdmission
	if len(b) > 1<<20 {
		return a, fmt.Errorf("admission exceeds one MiB")
	}
	// Require every explicit field, including zero-valued ordinal. Null is not
	// a supplied identity or budget; case aliases/unknown keys are refused.
	var raw map[string]json.RawMessage
	var rawErr error
	raw, rawErr = qualificationJSONObject(b)
	if e := rawErr; e != nil {
		return a, e
	}
	if len(raw) != 29 {
		return a, fmt.Errorf("admission keys must match the exact schema")
	}
	for _, key := range []string{"Version", "AuthorizationID", "CandidateCommit", "SourceSHA256", "ExecutableSHA256", "Host", "DeviceUUID", "Target", "Ordinal", "LockDir", "LockPID", "LockEpoch", "DriverFile", "DriverSHA256", "DriverVersion", "NVCCFile", "NVCCSHA256", "NVCCVersion", "OutputDir", "QuarantineContractFile", "QuarantineContractSHA256", "MaxChildren", "MaxLaunches", "ChildSeconds", "DrainSeconds", "OverallSeconds", "AllocationBytes", "ReceiptBytes", "Cases"} {
		v, ok := raw[key]
		if !ok || bytes.Equal(bytes.TrimSpace(v), []byte("null")) {
			return a, fmt.Errorf("required admission field %s missing or null", key)
		}
	}
	var rawArtifacts []json.RawMessage
	if e := json.Unmarshal(raw["Cases"], &rawArtifacts); e != nil {
		return a, e
	}
	var artifacts []map[string]json.RawMessage
	for _, item := range rawArtifacts {
		m, e := qualificationJSONObject(item)
		if e != nil {
			return a, e
		}
		artifacts = append(artifacts, m)
	}

	for _, artifact := range artifacts {
		if len(artifact) != 9 {
			return a, fmt.Errorf("artifact keys must match exact schema")
		}
		for _, key := range []string{"CaseID", "SourceFile", "SourceSHA256", "PTXFile", "PTXSHA256", "CubinFile", "CubinSHA256", "CompilerReceiptFile", "CompilerReceiptSHA256"} {
			v, ok := artifact[key]
			if !ok || bytes.Equal(bytes.TrimSpace(v), []byte("null")) {
				return a, fmt.Errorf("artifact field %s missing or null", key)
			}
		}
	}
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	if e := dec.Decode(&a); e != nil {
		return a, e
	}
	if e := dec.Decode(new(any)); e != io.EOF {
		return a, fmt.Errorf("trailing admission content")
	}
	return a, a.validate()
}

type qualificationStatus struct {
	Generation                                         uint64
	Rank, Selected, Phase, Bytes, Retired, TileAddress uint32
}

func decodeQualificationStatus(b []byte) (qualificationStatus, error) {
	var s qualificationStatus
	if len(b) != 32 {
		return s, fmt.Errorf("status storage must be 32 bytes")
	}
	s.Generation = binary.LittleEndian.Uint64(b)
	fields := []*uint32{&s.Rank, &s.Selected, &s.Phase, &s.Bytes, &s.Retired, &s.TileAddress}
	for i, p := range fields {
		*p = binary.LittleEndian.Uint32(b[8+4*i:])
	}
	return s, nil
}
func qualificationCompare(c qualificationCase, output, status, want []byte, generations []uint64) error {
	p, e := NewMulticastPlan(c.config())
	if e != nil {
		return e
	}
	r, _ := p.Requirements()
	phases := 1
	if c.Parity {
		phases = 2
	}
	if len(output) != phases*int(r.OutputBytes) || len(status) != phases*int(r.CTAs)*32 || len(want) != phases*int(r.TileBytes) || len(generations) != phases {
		return fmt.Errorf("readback extents differ from manifest")
	}
	for phase := 0; phase < phases; phase++ {
		for rank := uint32(0); rank < r.CTAs; rank++ {
			s, e := decodeQualificationStatus(status[(phase*int(r.CTAs)+int(rank))*32:][:32])
			if e != nil {
				return e
			}
			selected := uint32(0)
			if r.Mask&(1<<rank) != 0 {
				selected = 1
			}
			bytes := uint32(0)
			if selected != 0 {
				bytes = r.TileBytes
			}
			alignment := uint32(128)
			if c.Swizzle != 0 {
				alignment = c.Swizzle * 8
			}
			if s.Generation != generations[phase] || s.Rank != rank || s.Selected != selected || s.Phase != uint32(phase) || s.Bytes != bytes || s.Retired != 1 || s.TileAddress%alignment != 0 {
				return fmt.Errorf("CTA%d phase%d completion/layout receipt mismatch: %+v", rank, phase, s)
			}
			got := output[(phase*int(r.CTAs)+int(rank))*int(r.TileBytes):][:r.TileBytes]
			expected := want[phase*int(r.TileBytes):][:r.TileBytes]
			if selected == 0 {
				expected = bytesOf(0xa5, int(r.TileBytes))
			}
			if !bytesEqual(got, expected) {
				return fmt.Errorf("CTA%d phase%d numerical/canary mismatch", rank, phase)
			}
		}
	}
	return nil
}
func bytesOf(value byte, n int) []byte { return bytes.Repeat([]byte{value}, n) }
func bytesEqual(a, b []byte) bool      { return bytes.Equal(a, b) }

func TestQualificationCorpusAndOracle(t *testing.T) {
	normal, parity, controls := 0, 0, 0
	for _, c := range qualificationCases() {
		if c.Control != "" {
			controls++
			continue
		}
		if c.Parity {
			parity++
		} else {
			normal++
		}
		cfg := c.config()
		if _, e := NewMulticastPlan(cfg); e != nil {
			t.Fatalf("%s: %v", c.ID, e)
		}
		for epoch := uint32(0); epoch < 2; epoch++ {
			input, e := qualificationInput(cfg.Descriptor, epoch)
			if e != nil {
				t.Fatal(e)
			}
			for scenario := 0; scenario < 3; scenario++ {
				want, e := qualificationOracle(cfg.Descriptor, c.coordinates(scenario), input)
				if e != nil {
					t.Fatal(e)
				}
				if len(want) == 0 {
					t.Fatal("empty oracle")
				}
				if scenario == 0 && bytes.Equal(want, bytesOf(0, len(want))) {
					t.Fatal("vacuous interior oracle")
				}
			}
		}
	}
	if normal != 550 || parity != 200 || controls != 2 {
		t.Fatalf("corpus %d/%d/%d", normal, parity, controls)
	}
	d := Descriptor{Target: TargetSM100a, Rank: 2, Base: 0x1000, Dims: [5]uint64{4, 2}, Box: [5]uint32{4, 2}, Strides: [5]uint64{32}, ElementBytes: 4}
	input := make([]byte, 64)
	for i := range input {
		input[i] = byte(i + 1)
	}
	want, e := qualificationOracle(d, [5]int32{0, 0}, input)
	if e != nil {
		t.Fatal(e)
	}
	if !bytes.Equal(want, append(append([]byte(nil), input[:16]...), input[32:48]...)) {
		t.Fatal("independent padded-stride oracle")
	}
	tail, e := qualificationOracle(d, [5]int32{-2, 1}, input)
	if e != nil {
		t.Fatal(e)
	}
	expected := make([]byte, 32)
	copy(expected[8:16], input[32:40])
	if !bytes.Equal(tail, expected) {
		t.Fatal("independent signed/OOB oracle")
	}
}

func qualificationTestAdmission(t *testing.T) qualificationAdmission {
	t.Helper()
	c := qualificationCases()[0]
	source, e := qualificationSource(c)
	if e != nil {
		t.Fatal(e)
	}
	h := sha256.Sum256([]byte(source))
	digest := strings.Repeat("1", 64)
	return qualificationAdmission{Version: 1, AuthorizationID: "explicit-coordinator-test-admission", CandidateCommit: strings.Repeat("1", 40), SourceSHA256: qualificationDigest(), ExecutableSHA256: digest, Host: "host", DeviceUUID: strings.Repeat("1", 32), Target: TargetSM100a, Ordinal: 0, LockDir: "/task/lock", LockPID: 42, LockEpoch: 1, DriverFile: "/task/libcuda.so", DriverSHA256: digest, DriverVersion: 13000, NVCCFile: "/task/nvcc", NVCCSHA256: digest, NVCCVersion: "pinned", OutputDir: "/task/output", QuarantineContractFile: "/task/quarantine-contract", QuarantineContractSHA256: digest, MaxChildren: 1, MaxLaunches: 12, ChildSeconds: 30, DrainSeconds: 15, OverallSeconds: 30, AllocationBytes: 8 << 20, ReceiptBytes: 16 << 20, Cases: []qualificationArtifact{{CaseID: c.ID, SourceFile: "/task/f.cu", CompilerReceiptFile: "/task/compiler.json", CompilerReceiptSHA256: digest, SourceSHA256: hex.EncodeToString(h[:]), PTXFile: "/task/f.ptx", PTXSHA256: digest, CubinFile: "/task/f.cubin", CubinSHA256: digest}}}
}
func TestQualificationAdmissionRefusals(t *testing.T) {
	a := qualificationTestAdmission(t)
	if e := a.validate(); e != nil {
		t.Fatal(e)
	}
	mutations := []struct {
		name string
		f    func(*qualificationAdmission)
	}{{"authorization", func(a *qualificationAdmission) { a.AuthorizationID = "" }}, {"source", func(a *qualificationAdmission) { a.SourceSHA256 = strings.Repeat("2", 64) }}, {"UUID", func(a *qualificationAdmission) { a.DeviceUUID = "" }}, {"owner", func(a *qualificationAdmission) { a.LockPID = 1 }}, {"lock epoch", func(a *qualificationAdmission) { a.LockEpoch = 0 }}, {"target", func(a *qualificationAdmission) { a.Target = TargetSM121a }}, {"driver pin", func(a *qualificationAdmission) { a.DriverSHA256 = "" }}, {"tool pin", func(a *qualificationAdmission) { a.NVCCSHA256 = "" }}, {"budget", func(a *qualificationAdmission) { a.MaxLaunches = 0 }}, {"insufficient budget", func(a *qualificationAdmission) { a.MaxLaunches = 11 }}, {"quarantine", func(a *qualificationAdmission) { a.QuarantineContractSHA256 = "" }}, {"unsafe control", func(a *qualificationAdmission) { a.Cases[0].CaseID = "drop-expect" }}}
	for _, tt := range mutations {
		t.Run(tt.name, func(t *testing.T) {
			a := qualificationTestAdmission(t)
			tt.f(&a)
			if a.validate() == nil {
				t.Fatal("invalid admission accepted")
			}
		})
	}
	b, _ := json.Marshal(a)
	if _, e := decodeQualificationAdmission(b); e != nil {
		t.Fatal("valid exact admission refused", e)
	}
	var raw map[string]json.RawMessage
	json.Unmarshal(b, &raw)
	for key := range raw {
		t.Run("missing "+key, func(t *testing.T) {
			m := map[string]json.RawMessage{}
			for k, v := range raw {
				m[k] = v
			}
			delete(m, key)
			data, _ := json.Marshal(m)
			if _, e := decodeQualificationAdmission(data); e == nil {
				t.Fatal("missing key accepted")
			}
		})
	}
}

func TestQualificationStatusCheckerControls(t *testing.T) {
	c := qualificationCase{Rank: 2, Type: ElementTypeFloat32, ElementBytes: 4, Width: 128, Swizzle: 128, Mask: 2}
	p, _ := NewMulticastPlan(c.config())
	r, _ := p.Requirements()
	input, _ := qualificationInput(c.config().Descriptor, 0)
	want, _ := qualificationOracle(c.config().Descriptor, [5]int32{}, input)
	output := bytesOf(0xa5, int(r.OutputBytes))
	copy(output[r.TileBytes:], want)
	status := make([]byte, 64)
	for rank := 0; rank < 2; rank++ {
		b := status[rank*32:]
		binary.LittleEndian.PutUint64(b, 1)
		binary.LittleEndian.PutUint32(b[8:], uint32(rank))
		binary.LittleEndian.PutUint32(b[24:], 1)
		binary.LittleEndian.PutUint32(b[28:], 1024)
		if rank == 1 {
			binary.LittleEndian.PutUint32(b[12:], 1)
			binary.LittleEndian.PutUint32(b[20:], r.TileBytes)
		}
	}
	if e := qualificationCompare(c, output, status, want, []uint64{1}); e != nil {
		t.Fatal(e)
	}
	for _, offset := range []int{0, int(r.TileBytes), len(output) - 1} {
		bad := append([]byte(nil), output...)
		bad[offset] ^= 1
		if qualificationCompare(c, bad, status, want, []uint64{1}) == nil {
			t.Fatal("output/canary corruption escaped")
		}
	}
	for _, offset := range []int{0, 8, 12, 16, 20, 24, 28} {
		bad := append([]byte(nil), status...)
		bad[32+offset] ^= 1
		if qualificationCompare(c, output, bad, want, []uint64{1}) == nil {
			t.Fatal("status corruption escaped", offset)
		}
	}
}

type qualificationCompilerReceipt struct {
	CaseID, Target, CandidateCommit, ImplementationSHA256, NVCCFile, NVCCSHA256, NVCCVersion, SourceSHA256, PTXSHA256, CubinSHA256 string
	PTXCommand, CubinCommand                                                                                                       []string
	PTXExit, CubinExit                                                                                                             int
	VersionOutput, PTXOutput, CubinOutput                                                                                          string
}

func (r qualificationCompilerReceipt) validate(a qualificationAdmission, artifact qualificationArtifact) error {
	if r.CaseID != artifact.CaseID || r.Target != TargetSM100a || r.CandidateCommit != a.CandidateCommit || r.ImplementationSHA256 != a.SourceSHA256 || r.NVCCFile != a.NVCCFile || r.NVCCSHA256 != a.NVCCSHA256 || r.NVCCVersion != a.NVCCVersion || r.SourceSHA256 != artifact.SourceSHA256 || r.PTXSHA256 != artifact.PTXSHA256 || r.CubinSHA256 != artifact.CubinSHA256 || r.PTXExit != 0 || r.CubinExit != 0 || strings.TrimSpace(r.VersionOutput) != r.NVCCVersion {
		return fmt.Errorf("compiler provenance differs from exact native admission")
	}
	wantPTX := []string{a.NVCCFile, "-std=c++17", "-arch=sm_100a", "-ptx", artifact.SourceFile, "-o", artifact.PTXFile}
	wantCubin := []string{a.NVCCFile, "-std=c++17", "-arch=sm_100a", "-cubin", artifact.SourceFile, "-o", artifact.CubinFile}
	if strings.Join(r.PTXCommand, "\x00") != strings.Join(wantPTX, "\x00") || strings.Join(r.CubinCommand, "\x00") != strings.Join(wantCubin, "\x00") {
		return fmt.Errorf("compiler command differs from admitted source/target/artifacts")
	}
	return nil
}

func TestQualificationAlignedTensorMapABI(t *testing.T) {
	maps := []*driver.TensorMap{driver.NewTensorMap(), driver.NewTensorMap()}
	var pin runtime.Pinner
	defer pin.Unpin()
	for _, m := range maps {
		if unsafe.Sizeof(*m) != 128 || uintptr(unsafe.Pointer(m))%128 != 0 {
			t.Fatal("opaque map ABI/alignment")
		}
		pin.Pin(m)
	}
	args := []any{maps[0], maps[1], driver.DevicePtr(0x2000), driver.DevicePtr(0x3000), uint64(1), uint64(2), int32(-1), int32(2)}
	packed, e := cudalaunch.PackArgs(args...)
	if e != nil {
		t.Fatal(e)
	}
	for i, m := range maps {
		if packed[i] != unsafe.Pointer(m) {
			t.Fatal("map pointer-storage argument was copied or reduced to pointer value")
		}
	}
	// This demonstrates the existing public signature mismatch rather than
	// claiming it accepts an eight-byte pointer as a128-byte map value.
	sig := cudalaunch.Signature{Params: []cudalaunch.Param{{Type: "b8", Size: 128}}}
	if cudalaunch.Validate(sig, maps[0]) == nil {
		t.Fatal("existing pointer-vs-map signature mismatch disappeared; re-audit seam")
	}
	tail := cudalaunch.Signature{Params: []cudalaunch.Param{{Type: "u64", Size: 8, IsPointer: true}, {Type: "u64", Size: 8, IsPointer: true}, {Type: "u64", Size: 8}, {Type: "u64", Size: 8}, {Type: "s32", Size: 4}, {Type: "s32", Size: 4}}}
	if e := cudalaunch.Validate(tail, args[2:]...); e != nil {
		t.Fatal(e)
	}
	runtime.KeepAlive(maps)
}

func TestQualificationAdmissionExactKeys(t *testing.T) {
	a := qualificationTestAdmission(t)
	b, _ := json.Marshal(a)
	for _, key := range []string{"DeviceUUID", "LockPID", "SourceSHA256", "MaxLaunches"} {
		needle := []byte(`"` + key + `":`)
		alias := bytes.Replace(b, needle, []byte(`"`+strings.ToLower(key)+`":`), 1)
		if _, e := decodeQualificationAdmission(alias); e == nil {
			t.Fatal("case alias admitted", key)
		}
		var m map[string]json.RawMessage
		json.Unmarshal(b, &m)
		m[key] = json.RawMessage("null")
		null, _ := json.Marshal(m)
		if _, e := decodeQualificationAdmission(null); e == nil {
			t.Fatal("null identity admitted", key)
		}
	}
}

func TestQualificationCompilerReceiptBinding(t *testing.T) {
	a := qualificationTestAdmission(t)
	artifact := a.Cases[0]
	receipt := qualificationCompilerReceipt{CaseID: artifact.CaseID, Target: a.Target, CandidateCommit: a.CandidateCommit, ImplementationSHA256: a.SourceSHA256, NVCCFile: a.NVCCFile, NVCCSHA256: a.NVCCSHA256, NVCCVersion: a.NVCCVersion, VersionOutput: a.NVCCVersion, SourceSHA256: artifact.SourceSHA256, PTXSHA256: artifact.PTXSHA256, CubinSHA256: artifact.CubinSHA256, PTXCommand: []string{a.NVCCFile, "-std=c++17", "-arch=sm_100a", "-ptx", artifact.SourceFile, "-o", artifact.PTXFile}, CubinCommand: []string{a.NVCCFile, "-std=c++17", "-arch=sm_100a", "-cubin", artifact.SourceFile, "-o", artifact.CubinFile}}
	if e := receipt.validate(a, artifact); e != nil {
		t.Fatal(e)
	}
	bad := receipt
	bad.PTXExit = 1
	if bad.validate(a, artifact) == nil {
		t.Fatal("failed compiler receipt accepted")
	}
	bad = receipt
	bad.CubinSHA256 = strings.Repeat("2", 64)
	if bad.validate(a, artifact) == nil {
		t.Fatal("unbound native image accepted")
	}
	bad = receipt
	bad.PTXCommand = append([]string(nil), receipt.PTXCommand...)
	bad.PTXCommand[2] = "-arch=sm_121a"
	if bad.validate(a, artifact) == nil {
		t.Fatal("different compiler target accepted")
	}
}

func qualificationJSONObject(data []byte) (map[string]json.RawMessage, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	token, e := dec.Token()
	if e != nil {
		return nil, e
	}
	if token != json.Delim('{') {
		return nil, fmt.Errorf("JSON object required")
	}
	out := map[string]json.RawMessage{}
	for dec.More() {
		token, e = dec.Token()
		if e != nil {
			return nil, e
		}
		key, ok := token.(string)
		if !ok {
			return nil, fmt.Errorf("string object key required")
		}
		if _, ok = out[key]; ok {
			return nil, fmt.Errorf("duplicate object key %s", key)
		}
		var raw json.RawMessage
		if e = dec.Decode(&raw); e != nil {
			return nil, e
		}
		out[key] = raw
	}
	if token, e = dec.Token(); e != nil || token != json.Delim('}') {
		return nil, fmt.Errorf("object terminator required")
	}
	if e = dec.Decode(new(any)); e != io.EOF {
		return nil, fmt.Errorf("trailing object content")
	}
	return out, nil
}

func TestQualificationDuplicateKeys(t *testing.T) {
	b, _ := json.Marshal(qualificationTestAdmission(t))
	duplicate := bytes.Replace(b, []byte(`"Ordinal":0`), []byte(`"Ordinal":0,"Ordinal":1`), 1)
	if _, e := decodeQualificationAdmission(duplicate); e == nil {
		t.Fatal("duplicate identity admitted")
	}
}

func TestQualificationParityReceiptControls(t *testing.T) {
	c := qualificationCase{Rank: 1, Type: ElementTypeFloat32, ElementBytes: 4, Width: 128, Swizzle: 128, Mask: 3, Parity: true}
	p, _ := NewMulticastPlan(c.config())
	r, _ := p.Requirements()
	var want []byte
	for epoch := uint32(0); epoch < 2; epoch++ {
		input, _ := qualificationInput(c.config().Descriptor, epoch)
		expected, _ := qualificationOracle(c.config().Descriptor, [5]int32{}, input)
		want = append(want, expected...)
	}
	output := make([]byte, 2*int(r.OutputBytes))
	status := make([]byte, 128)
	for phase := 0; phase < 2; phase++ {
		for rank := 0; rank < 2; rank++ {
			copy(output[(phase*2+rank)*int(r.TileBytes):], want[phase*int(r.TileBytes):][:r.TileBytes])
			b := status[(phase*2+rank)*32:]
			binary.LittleEndian.PutUint64(b, uint64(phase+1))
			binary.LittleEndian.PutUint32(b[8:], uint32(rank))
			binary.LittleEndian.PutUint32(b[12:], 1)
			binary.LittleEndian.PutUint32(b[16:], uint32(phase))
			binary.LittleEndian.PutUint32(b[20:], r.TileBytes)
			binary.LittleEndian.PutUint32(b[24:], 1)
			binary.LittleEndian.PutUint32(b[28:], 1024)
		}
	}
	if e := qualificationCompare(c, output, status, want, []uint64{1, 2}); e != nil {
		t.Fatal(e)
	}
	stale := append([]byte(nil), status...)
	binary.LittleEndian.PutUint32(stale[64+16:], 0)
	if qualificationCompare(c, output, stale, want, []uint64{1, 2}) == nil {
		t.Fatal("stale consumed parity escaped checker")
	}
	reused := append([]byte(nil), output...)
	copy(reused[int(r.OutputBytes):], output[:r.OutputBytes])
	if qualificationCompare(c, reused, status, want, []uint64{1, 2}) == nil {
		t.Fatal("prior epoch data escaped checker")
	}
}

func qualificationCubin(data []byte) error {
	// ModuleLoadData also accepts PTX and could JIT it. Only a reviewed native
	// CUDA ELF image is permitted here; a filename alone proves nothing.
	if len(data) < 64 || !bytes.Equal(data[:4], []byte{0x7f, 'E', 'L', 'F'}) {
		return fmt.Errorf("prebuilt native CUDA ELF required; PTX/JIT input refused")
	}
	image, e := elf.NewFile(bytes.NewReader(data))
	if e != nil {
		return e
	}
	defer image.Close()
	if image.Class != elf.ELFCLASS64 || image.Data != elf.ELFDATA2LSB || image.Machine != elf.EM_CUDA || image.Type != elf.ET_EXEC {
		return fmt.Errorf("artifact is not a native64-bit CUDA executable image")
	}
	return nil
}
func TestQualificationCubinOnly(t *testing.T) {
	if qualificationCubin([]byte(".version 9.4\n.target sm_100a\n")) == nil {
		t.Fatal("runtime PTX/JIT fallback admitted")
	}
	header := make([]byte, 64)
	copy(header, []byte{0x7f, 'E', 'L', 'F', 2, 1, 1})
	binary.LittleEndian.PutUint16(header[16:], uint16(elf.ET_EXEC))
	binary.LittleEndian.PutUint16(header[18:], uint16(elf.EM_CUDA))
	binary.LittleEndian.PutUint32(header[20:], 1)
	binary.LittleEndian.PutUint16(header[52:], 64)
	// A minimal synthetic ELF tests format admission only. It is not a CUDA
	// module, compiled fixture, executable code, or positive native evidence.
	if e := qualificationCubin(header); e != nil {
		t.Fatal(e)
	}
	binary.LittleEndian.PutUint16(header[18:], uint16(elf.EM_X86_64))
	if qualificationCubin(header) == nil {
		t.Fatal("wrong machine ELF admitted")
	}
}

// qualificationPending keeps a loaded raw handle until checked adoption or
// close. Its callbacks are actual context-bound operations in the native owner;
// CPU fault tests exercise this same retention and cleanup ordering only.
type qualificationPending[T comparable] struct {
	value          T
	held           bool
	scopeUncertain bool
}

// Capture nonzero ownership before the context callback returns: the enclosing
// scope can fail while restoring the prior thread context after a successful
// load. Any failed scope conservatively blocks automatic cleanup; this fixture
// has no verified recovery operation for an unrestored context.
func (p *qualificationPending[T]) load(scope func(func() error) error, load func() (T, error)) error {
	e := scope(func() error {
		raw, e := load()
		var zero T
		if raw != zero {
			p.value, p.held = raw, true
		}
		return e
	})
	if e != nil {
		p.scopeUncertain = true
	}
	return e
}

func (p *qualificationPending[T]) closeThen(closeRaw func(T) error, remaining func() error) error {
	if p.scopeUncertain {
		return fmt.Errorf("module load context scope uncertain; owner retained, external recovery required")
	}
	if p.held {
		if e := closeRaw(p.value); e != nil {
			return fmt.Errorf("raw module retained after failed close: %w", e)
		}
		p.held = false
		var zero T
		p.value = zero
	}
	return remaining()
}

// Every uncertain post-child outcome attempts quarantine, including a failed
// supervisor report. Neither a successful exit nor a kill proves CUDA drain.
func qualificationFinishChild(id, source string, launches int, runErr error, report func() error, read func() ([]byte, error), quarantine func(error) error) error {
	e := errors.Join(runErr, report())
	if e == nil {
		var b []byte
		b, e = read()
		if e == nil {
			var completed struct {
				CaseID           string
				Launches         int
				CleanupCompleted bool
				SourceSHA256     string
			}
			fields, err := qualificationJSONObject(b)
			e = err
			keys := []string{"CaseID", "Launches", "CleanupCompleted", "SourceSHA256"}
			if e == nil && len(fields) != len(keys) {
				e = fmt.Errorf("exact completed receipt fields required")
			}
			for _, key := range keys {
				if e == nil && (len(fields[key]) == 0 || bytes.Equal(bytes.TrimSpace(fields[key]), []byte("null"))) {
					e = fmt.Errorf("completed receipt field %s missing or null", key)
				}
			}
			if e == nil {
				e = json.Unmarshal(b, &completed)
			}
			if e == nil && (completed.CaseID != id || !completed.CleanupCompleted || completed.SourceSHA256 != source || completed.Launches != launches) {
				e = fmt.Errorf("child completion receipt does not match admission")
			}
		}
	}
	if e != nil {
		return errors.Join(e, quarantine(e))
	}
	return nil
}

func qualificationLaunches(id string) int {
	for _, c := range qualificationCases() {
		if c.ID == id {
			if c.Parity {
				return 3
			}
			if c.Control != "" {
				return 1
			}
			return 12
		}
	}
	return 0
}

func TestQualificationPendingModuleFaults(t *testing.T) {
	fault := errors.New("unload failed")
	p := qualificationPending[uint64]{value: 19, held: true}
	remaining, attempts := 0, 0
	closeRaw := func(raw uint64) error {
		attempts++
		if raw != 19 {
			t.Fatalf("lost raw handle: %d", raw)
		}
		return fault
	}
	next := func() error { remaining++; return nil }
	// Failed adoption uses the first close; child preparation cleanup uses the
	// second. Neither may discard the raw module or release its context/device.
	for i := 0; i < 2; i++ {
		if !errors.Is(p.closeThen(closeRaw, next), fault) || !p.held || p.value != 19 || remaining != 0 {
			t.Fatal("failed unload lost ownership or permitted downstream cleanup")
		}
	}
	if attempts != 2 {
		t.Fatal("unexpected unload attempts")
	}
	if e := p.closeThen(func(raw uint64) error { return nil }, next); e != nil || p.held || p.value != 0 || remaining != 1 {
		t.Fatal("successful checked unload did not permit cleanup")
	}
	if e := p.closeThen(func(raw uint64) error { t.Fatal("duplicate unload"); return nil }, next); e != nil {
		t.Fatal(e)
	}
}

func TestQualificationTerminalFaults(t *testing.T) {
	valid := []byte(`{"CaseID":"case","Launches":12,"CleanupCompleted":true,"SourceSHA256":"source"}`)
	fault := errors.New("I/O fault")
	for _, tt := range []struct {
		name              string
		run, report, read error
		body              []byte
	}{
		{"healthy", nil, nil, nil, valid},
		{"nonzero", fault, nil, nil, valid},
		{"deadline", fault, nil, nil, valid},
		{"log-close", fault, nil, nil, valid},
		{"report", nil, fault, nil, valid},
		{"missing", nil, nil, fault, nil},
		{"malformed", nil, nil, nil, []byte("{")},
		{"null", nil, nil, nil, []byte("null")},
		{"alias", nil, nil, nil, bytes.ReplaceAll(valid, []byte("CaseID"), []byte("caseid"))},
		{"duplicate", nil, nil, nil, bytes.ReplaceAll(valid, []byte(`"Launches":12`), []byte(`"Launches":12,"Launches":12`))},
		{"unknown", nil, nil, nil, bytes.ReplaceAll(valid, []byte(`"Launches":12`), []byte(`"Launches":12,"extra":true`))},
		{"null-field", nil, nil, nil, bytes.ReplaceAll(valid, []byte(`"Launches":12`), []byte(`"Launches":null`))},
		{"mismatching-case", nil, nil, nil, bytes.ReplaceAll(valid, []byte("case"), []byte("other"))},
		{"mismatching-source", nil, nil, nil, bytes.ReplaceAll(valid, []byte("source"), []byte("other"))},
		{"mismatching-launches", nil, nil, nil, bytes.ReplaceAll(valid, []byte("12"), []byte("11"))},
		{"unconfirmed-cleanup", nil, nil, nil, bytes.ReplaceAll(valid, []byte("true"), []byte("false"))},
	} {
		t.Run(tt.name, func(t *testing.T) {
			quarantines, reports, reads := 0, 0, 0
			e := qualificationFinishChild("case", "source", 12, tt.run,
				func() error { reports++; return tt.report },
				func() ([]byte, error) { reads++; return tt.body, tt.read },
				func(reason error) error {
					quarantines++
					if reason == nil {
						t.Fatal("missing uncertainty")
					}
					return nil
				})
			if tt.name == "healthy" {
				if e != nil || quarantines != 0 || reads != 1 {
					t.Fatal("healthy checked receipt refused")
				}
			} else if e == nil || quarantines != 1 {
				t.Fatal("uncertain terminal evidence did not quarantine")
			}
			if reports != 1 {
				t.Fatal("missing supervisor report attempt")
			}
			if (tt.run != nil || tt.report != nil) && reads != 0 {
				t.Fatal("failed child/report read as completion")
			}
		})
	}
	// A failed quarantine write must remain an error, never a successful release.
	if e := qualificationFinishChild("case", "source", 12, fault, func() error { return nil }, func() ([]byte, error) { return valid, nil }, func(error) error { return errors.New("quarantine write failed") }); e == nil || !strings.Contains(e.Error(), "quarantine write failed") {
		t.Fatal("quarantine persistence failure lost")
	}
}

func TestQualificationLoadScopeFaults(t *testing.T) {
	loadFault, restoreFault := errors.New("load failed"), errors.New("restore failed")
	for _, tt := range []struct {
		name              string
		raw               uint64
		loadErr, scopeErr error
		call              bool
	}{
		{"healthy", 19, nil, nil, true},
		{"load-success-restore-failure", 19, nil, restoreFault, true},
		{"partial-load-error", 19, loadFault, nil, true},
		{"partial-load-and-restore-error", 19, loadFault, restoreFault, true},
		{"load-zero-error", 0, loadFault, nil, true},
		{"scope-before-callback-error", 0, nil, restoreFault, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var p qualificationPending[uint64]
			e := p.load(func(callback func() error) error {
				if !tt.call {
					return tt.scopeErr
				}
				callbackErr := callback()
				if tt.raw != 0 && (!p.held || p.value != tt.raw) {
					t.Fatal("loaded raw ownership absent before scope restoration")
				}
				return errors.Join(callbackErr, tt.scopeErr)
			}, func() (uint64, error) { return tt.raw, tt.loadErr })
			uncertain := tt.loadErr != nil || tt.scopeErr != nil
			if (e != nil) != uncertain || p.scopeUncertain != uncertain || p.held != (tt.raw != 0) || p.value != tt.raw {
				t.Fatal("partial load/context state lost")
			}
			closed, released := 0, 0
			cleanup := p.closeThen(func(uint64) error { closed++; return nil }, func() error { released++; return nil })
			if uncertain {
				if cleanup == nil || closed != 0 || released != 0 || p.value != tt.raw {
					t.Fatal("uncertain context permitted automatic cleanup")
				}
			} else if cleanup != nil || closed != 1 || released != 1 || p.held {
				t.Fatal("healthy load cleanup refused")
			}
		})
	}
}
