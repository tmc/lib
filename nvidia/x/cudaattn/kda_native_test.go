//go:build native

package cudaattn

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

type kdaPacketEntry struct {
	Name, SourceSHA256, ImageSHA256, Target, CompilerPath, CompilerSHA256 string
	CompilerMajor, CompilerMinor                                          int32
	CompilerOptions                                                       []string
}

func kdaDigest(data []byte) string { s := sha256.Sum256(data); return hex.EncodeToString(s[:]) }
func kdaJSON(path string, value any) error {
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0600)
}

func kdaReadBounded(path string, limit int64) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() || info.Size() < 0 || info.Size() > limit {
		return nil, fmt.Errorf("invalid or oversized kda packet file")
	}
	data, err := io.ReadAll(io.LimitReader(f, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > limit {
		return nil, fmt.Errorf("growing kda packet exceeds bound")
	}
	return data, nil
}

func kdaCompilerDigest(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return "", err
	}
	const limit = 512 << 20
	if !info.Mode().IsRegular() || info.Size() <= 0 || info.Size() > limit {
		return "", fmt.Errorf("invalid compiler file")
	}
	h := sha256.New()
	n, err := io.Copy(h, io.LimitReader(f, limit+1))
	if err != nil {
		return "", err
	}
	if n > limit {
		return "", fmt.Errorf("compiler exceeds bound")
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// TestBuildKDAPacket only compiles the eight fixed ordinary images, without
// loading a driver or querying a GPU. Metadata binds retained bytes and the
// selected compiler observed here; packet agreement is not compiler attestation.
func TestBuildKDAPacket(t *testing.T) {
	if os.Getenv("KDA_BUILD_ONLY") != "1" {
		t.Skip("offline packet build not requested")
	}
	dir, target, path := os.Getenv("KDA_PACKET_DIR"), os.Getenv("KDA_TARGET"), os.Getenv("KDA_NVRTC_PATH")
	if !filepath.IsAbs(dir) || !regexp.MustCompile(`^sm_[0-9]{2,3}$`).MatchString(target) || !filepath.IsAbs(path) {
		t.Fatal("absolute packet/compiler path and explicit physical target required")
	}
	if err := os.MkdirAll(dir, 0700); err != nil {
		t.Fatal(err)
	}
	if err := nvrtc.Load(path); err != nil {
		t.Fatal(err)
	}
	major, minor, err := nvrtc.Version()
	if err != nil {
		t.Fatal(err)
	}
	digest, err := kdaCompilerDigest(nvrtc.Path())
	if err != nil {
		t.Fatal(err)
	}
	sources := kdaMutants()
	sources["baseline"] = KDASource
	names := make([]string, 0, len(sources))
	for name := range sources {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		source := sources[name]
		a, err := kdaCompileArtifact(source, target)
		if err != nil {
			t.Fatalf("compile %s: %v", name, err)
		}
		entry := kdaPacketEntry{name, kdaDigest([]byte(source)), a.Digest, target, nvrtc.Path(), digest, major, minor, append([]string(nil), a.CompilerOpts...)}
		for suffix, value := range map[string]any{".artifact.json": a, ".receipt.json": entry} {
			if err := kdaJSON(filepath.Join(dir, name+suffix), value); err != nil {
				t.Fatal(err)
			}
		}
		if err := os.WriteFile(filepath.Join(dir, name+".cu"), []byte(source), 0600); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, name+".cubin"), a.Raw, 0600); err != nil {
			t.Fatal(err)
		}
		t.Logf("COMPILED %s source=%s image=%s target=%s compiler=%s sha256=%s version=%d.%d", name, entry.SourceSHA256, entry.ImageSHA256, target, entry.CompilerPath, digest, major, minor)
	}
}

func kdaReadPacket(dir, name, source, target string) (*cudakernel.Artifact, kdaPacketEntry, error) {
	var a cudakernel.Artifact
	var entry kdaPacketEntry
	for suffix, value := range map[string]any{".artifact.json": &a, ".receipt.json": &entry} {
		data, err := kdaReadBounded(filepath.Join(dir, name+suffix), 16<<20)
		if err != nil {
			return nil, entry, err
		}
		if err := json.Unmarshal(data, value); err != nil {
			return nil, entry, err
		}
	}
	data, err := kdaReadBounded(filepath.Join(dir, name+".cu"), 1<<20)
	if err != nil {
		return nil, entry, err
	}
	image, err := kdaReadBounded(filepath.Join(dir, name+".cubin"), 16<<20)
	if err != nil {
		return nil, entry, err
	}
	if !bytes.Equal(image, a.Raw) {
		return nil, entry, fmt.Errorf("kda retained image bytes mismatch")
	}
	schema, err := KDASchema()
	if err != nil {
		return nil, entry, err
	}
	opts := []string{"--gpu-architecture=" + target, "--std=c++17", "--fmad=false", "--ftz=false"}
	if string(data) != source || entry.Name != name || entry.SourceSHA256 != kdaDigest(data) || entry.ImageSHA256 != a.Digest || entry.Target != target || a.Target != target || a.Kind != cudakernel.KindCUBIN || a.EntryPoint != "kda_fp32" || !reflect.DeepEqual(a.Schema, schema) {
		return nil, entry, fmt.Errorf("kda packet identity/ABI mismatch")
	}
	if !filepath.IsAbs(entry.CompilerPath) || strings.IndexByte(entry.CompilerPath, 0) >= 0 || entry.CompilerMajor <= 0 || entry.CompilerMinor < 0 || !regexp.MustCompile(`^[0-9a-f]{64}$`).MatchString(entry.CompilerSHA256) || a.Toolchain != fmt.Sprintf("nvrtc %d.%d", entry.CompilerMajor, entry.CompilerMinor) || !reflect.DeepEqual(entry.CompilerOptions, opts) || !reflect.DeepEqual(a.CompilerOpts, opts) {
		return nil, entry, fmt.Errorf("kda retained compiler metadata mismatch")
	}
	found := false
	for _, d := range a.Dependencies {
		if d.Path == "cudaattn_kda.cu" && d.Tracked && d.Digest == entry.SourceSHA256 && d.ErrorMsg == "" {
			found = true
		}
	}
	if !found {
		return nil, entry, fmt.Errorf("kda source dependency missing")
	}
	if err := a.Validate(); err != nil {
		return nil, entry, err
	}
	return &a, entry, nil
}

type kdaNativeChain struct {
	state     *KDAState
	roots     []cudamem.Slice[float32]
	snapshots []struct {
		data cudamem.Slice[float32]
		host []float32
	}
}

type kdaNativeEvidence struct {
	Resources                                                     map[string]int
	ComputeSubmitted, ComputeCompleted, PublicationCompleted      bool
	CanceledWaitRetainedPins, StateFreeRefused, OutputFreeRefused bool
	AdmissionRefusals                                             int
	DeviceStatus                                                  int32
	Initial, Published                                            []KDAIdentity
}

func (c *kdaNativeChain) close(t *testing.T) {
	t.Helper()
	for _, snapshot := range c.snapshots {
		got := make([]float32, len(snapshot.host))
		if err := snapshot.data.CopyToHost(got); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, snapshot.host) {
			t.Fatal("retained consumer snapshot changed")
		}
	}
	for _, root := range c.roots {
		if err := root.Free(); err != nil {
			t.Error(err)
		}
	}
}

func kdaNativeRun(t *testing.T, dev *cuda.Device, a *cudakernel.Artifact, f kdaFixture, chain *kdaNativeChain, evidence *kdaNativeEvidence) ([]float32, []float32, error) {
	t.Helper()
	k, err := kdaLoadArtifact(dev, f.c, a)
	var stream *cuda.Stream
	var roots []cudamem.Slice[float32]
	defer func() {
		if k != nil {
			if err := k.Close(); err != nil {
				t.Error(err)
				return
			}
		}
		if stream != nil {
			if err := stream.Close(); err != nil {
				t.Error(err)
				return
			}
		}
		if chain != nil {
			chain.roots = append(chain.roots, roots...)
		} else {
			for _, root := range roots {
				if err := root.Free(); err != nil {
					t.Error(err)
				}
			}
		}
	}()
	if err != nil {
		return nil, nil, err
	}
	evidence.Resources = make(map[string]int)
	for _, fact := range []struct {
		name      string
		attribute driver.FunctionAttribute
	}{{"allocated_registers_per_thread", driver.FuncAttributeNumRegs}, {"local_bytes_per_thread", driver.FuncAttributeLocalSizeBytes}, {"static_shared_bytes", driver.FuncAttributeSharedSizeBytes}, {"max_threads_per_block", driver.FuncAttributeMaxThreadsPerBlock}} {
		value, err := k.kernel.Attribute(fact.attribute)
		if err != nil {
			t.Fatal(err)
		}
		evidence.Resources[fact.name] = value
	}
	stream, err = dev.NewStream()
	if err != nil {
		return nil, nil, err
	}
	guards := []func(){}
	alloc := func(data []float32) cudamem.Slice[float32] {
		if len(data) == 0 {
			return cudamem.Slice[float32]{}
		}
		root, err := cudamem.Alloc[float32](dev, len(data)+2)
		if err != nil {
			t.Fatal(err)
		}
		roots = append(roots, root)
		all := append([]float32{90123}, data...)
		all = append(all, -90123)
		if err := root.CopyFromHost(all); err != nil {
			t.Fatal(err)
		}
		guards = append(guards, func() {
			g := make([]float32, root.Len())
			if err := root.CopyToHost(g); err != nil {
				t.Fatal(err)
			}
			if g[0] != 90123 || g[len(g)-1] != -90123 {
				t.Fatal("device redzone changed")
			}
		})
		return root.Slice(1, root.Len()-1)
	}
	poison := func(n int) []float32 {
		x := make([]float32, n)
		for i := range x {
			x[i] = -12345
		}
		return x
	}
	b := KDABuffers{Q: alloc(f.q), Key: alloc(f.key), Value: alloc(f.value), Decay: alloc(f.decay), Beta: alloc(f.beta), FinalState: alloc(poison(len(f.state))), Output: alloc(poison(len(f.q)))}
	ids := make([]KDAIdentity, f.c.Sequences)
	for i := range ids {
		ids[i] = KDAIdentity{fmt.Sprintf("seq%d", i), 11, KDAFP32}
	}
	var initial *KDAState
	if chain != nil && chain.state != nil {
		initial = chain.state
		ids = initial.Identities()
	} else {
		initial, err = NewKDAState(f.c, alloc(f.state), ids)
		if err != nil {
			t.Fatal(err)
		}
		if chain != nil {
			chain.snapshots = append(chain.snapshots, struct {
				data cudamem.Slice[float32]
				host []float32
			}{initial.Data(), append([]float32(nil), f.state...)})
		}
	}
	evidence.Initial = append([]KDAIdentity(nil), ids...)
	// These controls reach the actual consumer admission path and must queue no
	// work. A returned completion would invalidate the refusal receipt.
	for _, id := range []KDAIdentity{{"wrong", ids[0].Prefix, KDAFP32}, {ids[0].Sequence, ids[0].Prefix + 1, KDAFP32}, {ids[0].Sequence, ids[0].Prefix, "bf16"}} {
		wrong := append([]KDAIdentity(nil), ids...)
		wrong[0] = id
		bad, err := k.Execute(nil, stream, initial, wrong, f.offsets, b)
		if bad != nil || err == nil {
			t.Fatal("wrong state identity queued work")
		}
		evidence.AdmissionRefusals++
	}
	for _, badOffsets := range [][]int32{append([]int32(nil), f.offsets...), append([]int32(nil), f.offsets...)} {
		if evidence.AdmissionRefusals == 3 {
			if len(badOffsets) > 2 {
				badOffsets[1] = badOffsets[len(badOffsets)-1] + 1
			} else {
				badOffsets[0] = -1
			}
		} else {
			badOffsets[len(badOffsets)-1]++
		}
		bad, err := k.Execute(nil, stream, initial, ids, badOffsets, b)
		if bad != nil || err == nil {
			t.Fatal("invalid offsets queued work")
		}
		evidence.AdmissionRefusals++
	}
	alias := b
	alias.FinalState = initial.Data()
	bad, aliasErr := k.Execute(nil, stream, initial, ids, f.offsets, alias)
	if bad != nil || aliasErr == nil {
		t.Fatal("in-place state update admitted")
	}
	evidence.AdmissionRefusals++
	r, err := k.Execute(context.Background(), stream, initial, ids, f.offsets, b)
	if r != nil {
		evidence.ComputeSubmitted = r.submitErr == nil
		evidence.StateFreeRefused = initial.Data().Free() != nil
		evidence.OutputFreeRefused = b.FinalState.Free() != nil
		if !evidence.StateFreeRefused || !evidence.OutputFreeRefused {
			t.Fatal("queued state borrow did not prevent Free")
		}
		if _, stateErr := r.State(); stateErr == nil {
			t.Fatal("state published before wait")
		}
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if cancelErr := r.Wait(ctx); !errors.Is(cancelErr, context.Canceled) || r.done.Load() {
			t.Fatalf("canceled wait lost pending state: %v", cancelErr)
		}
		evidence.CanceledWaitRetainedPins = true
		waitErr := r.Wait(nil)
		err = errors.Join(err, waitErr)
		evidence.ComputeCompleted = r.compute != nil && r.compute.Done()
		evidence.PublicationCompleted = r.publish != nil && r.publish.Done()
		status := []int32{-1}
		if evidence.ComputeCompleted {
			if err := k.status.CopyToHost(status); err != nil {
				t.Fatal(err)
			}
		}
		evidence.DeviceStatus = status[0]
		if err == nil {
			state, stateErr := r.State()
			if stateErr != nil {
				t.Fatal(stateErr)
			}
			for i, id := range state.Identities() {
				if id.Prefix != ids[i].Prefix+uint64(f.offsets[i+1]-f.offsets[i]) {
					t.Fatal("published wrong prefix")
				}
			}
			evidence.Published = state.Identities()
			if chain != nil {
				chain.state = state
			}
		}
	}
	if err != nil && !strings.Contains(err.Error(), "kda nonfinite or missing device result") {
		t.Fatalf("native submission/drain/publication failed: %v", err)
	}
	if err != nil && (r == nil || !r.done.Load()) {
		return nil, nil, err
	}
	s, o := make([]float32, len(f.state)), make([]float32, len(f.q))
	if err := b.FinalState.CopyToHost(s); err != nil {
		t.Fatal(err)
	}
	if len(o) > 0 {
		if err := b.Output.CopyToHost(o); err != nil {
			t.Fatal(err)
		}
	}
	for _, check := range guards {
		check()
	}
	if chain != nil && err == nil {
		chain.snapshots = append(chain.snapshots, struct {
			data cudamem.Slice[float32]
			host []float32
		}{b.FinalState, append([]float32(nil), s...)})
	}
	return s, o, err
}

// TestNativeKDAStateAndOutput is a finite packet consumer, with six lengths,
// single-state, four chunk splits, isolation, and seven arithmetic mutants.
// No numerical success is inferred from compilation or host source shims.
func TestNativeKDAStateAndOutput(t *testing.T) {
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Skipf("native kda requires inherited gpu-lock: %v", err)
	}
	dir, out := os.Getenv("KDA_PACKET_DIR"), os.Getenv("KDA_OUTPUT_DIR")
	if !filepath.IsAbs(dir) || !filepath.IsAbs(out) {
		t.Fatal("absolute retained packet/output paths required")
	}
	if err := os.MkdirAll(out, 0700); err != nil {
		t.Fatal(err)
	}
	dev := cudatest.Require(t)
	defer func() {
		if err := dev.Close(); err != nil {
			t.Error(err)
		}
	}()
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	target := fmt.Sprintf("sm_%d%d", major, minor)
	baseline, entry, err := kdaReadPacket(dir, "baseline", KDASource, target)
	if err != nil {
		t.Fatal(err)
	}
	launches := 0
	admissionRefusals := 0
	var chain *kdaNativeChain
	defer func() {
		if chain != nil {
			chain.close(t)
		}
	}()
	run := func(name string, a *cudakernel.Artifact, f kdaFixture) ([]float32, []float32, error) {
		launches++
		if launches > 24 {
			t.Fatal("native invocation bound exceeded")
		}
		evidence := kdaNativeEvidence{}
		s, o, err := kdaNativeRun(t, dev, a, f, chain, &evidence)
		admissionRefusals += evidence.AdmissionRefusals
		ws, wo := kdaOracle(f)
		record := struct {
			Name                      string
			Config                    KDAConfig
			Offsets                   []int32
			Packet                    kdaPacketEntry
			InputSHA256               string
			State, Output             []float32
			OracleState, OracleOutput []float64
			Error                     string
			Absolute, Relative        float64
			Evidence                  kdaNativeEvidence
		}{name, f.c, f.offsets, entry, kdaDigest(kdaInputBytes(f)), s, o, ws, wo, "", kdaAbsolute, kdaRelative, evidence}
		if err != nil {
			record.Error = err.Error()
		}
		if err := kdaJSON(filepath.Join(out, name+".json"), record); err != nil {
			t.Fatal(err)
		}
		if a == baseline && err == nil {
			if err := kdaCompare(s, ws, f, KDASource, target); err != nil {
				t.Fatal(err)
			}
			if err := kdaCompare(o, wo, f, KDASource, target); err != nil {
				t.Fatal(err)
			}
		}
		return s, o, err
	}
	check := func(name string, f kdaFixture) {
		s, o, err := run(name, baseline, f)
		if err != nil {
			t.Fatal(err)
		}
		ws, wo := kdaOracle(f)
		if err := kdaCompare(s, ws, f, KDASource, target); err != nil {
			t.Fatal(err)
		}
		if err := kdaCompare(o, wo, f, KDASource, target); err != nil {
			t.Fatal(err)
		}
	}
	for _, n := range []int{0, 1, 15, 16, 17, 33} {
		check(fmt.Sprintf("length%d", n), kdaInput(n, 2, 2))
	}
	check("single", kdaInput(17, 1, 1))
	f := kdaInput(33, 1, 1)
	ws, wo := kdaOracle(f)
	for _, split := range []int{1, 15, 16, 17} {
		chain = &kdaNativeChain{}
		state := append([]float32(nil), f.state...)
		var output []float32
		for arm, span := range [][2]int{{0, split}, {split, 33}} {
			p := f
			p.c.Tokens = span[1] - span[0]
			p.offsets = []int32{0, int32(p.c.Tokens)}
			p.state = state
			p.q = f.q[span[0]*128 : span[1]*128]
			p.key = f.key[span[0]*128 : span[1]*128]
			p.value = f.value[span[0]*128 : span[1]*128]
			p.decay = f.decay[span[0]*128 : span[1]*128]
			p.beta = f.beta[span[0]:span[1]]
			var o []float32
			state, o, err = run(fmt.Sprintf("chunk%d-arm%d", split, arm), baseline, p)
			if err != nil {
				t.Fatal(err)
			}
			output = append(output, o...)
		}
		if err := kdaCompare(state, ws, f, KDASource, target); err != nil {
			t.Fatal(err)
		}
		if err := kdaCompare(output, wo, f, KDASource, target); err != nil {
			t.Fatal(err)
		}
		chain.close(t)
		chain = nil
	}
	f = kdaInput(17, 2, 2)
	// Mixed empty/nonempty slots exercise state preservation and sequence
	// isolation without adding a length or invocation outside the fixed packet.
	f.c.Tokens = 17
	f.offsets = []int32{0, 0, 17}
	f.q = f.q[:17*2*128]
	f.key = f.key[:17*2*128]
	f.value = f.value[:17*2*128]
	f.decay = f.decay[:17*2*128]
	f.beta = f.beta[:17*2]
	s, o, err := run("isolation-a", baseline, f)
	if err != nil {
		t.Fatal(err)
	}
	other := f
	other.state = append([]float32(nil), f.state...)
	for i := 0; i < 2*128*128; i++ {
		if s[i] != f.state[i] {
			t.Fatal("mixed empty segment changed imported state")
		}
		other.state[i] *= 0.5
	}
	ss, oo, err := run("isolation-b", baseline, other)
	if err != nil {
		t.Fatal(err)
	}
	for i := 2 * 128 * 128; i < len(s); i++ {
		if s[i] != ss[i] {
			t.Fatal("sequence state contamination")
		}
	}
	for i := 0; i < len(o); i++ {
		if o[i] != oo[i] {
			t.Fatal("sequence output contamination")
		}
	}
	mutants := kdaMutants()
	names := make([]string, 0, len(mutants))
	for name := range mutants {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		source := mutants[name]
		a, mutantEntry, err := kdaReadPacket(dir, name, source, target)
		if err != nil {
			t.Fatal(err)
		}
		entry = mutantEntry
		s, o, err := run("mutant-"+name, a, f)
		ws, wo := kdaOracle(f)
		badState := kdaCompare(s, ws, f, source, target) != nil
		badOutput := kdaCompare(o, wo, f, source, target) != nil
		if err == nil && !badState && !badOutput {
			t.Fatalf("%s survived", name)
		}
		if name == "no-op" && (err == nil || !badState || !badOutput || len(s) == 0 || s[0] != -12345 || len(o) == 0 || o[0] != -12345) {
			t.Fatal("no-op must fail publication and both comparisons")
		}
	}
	if launches != 24 {
		t.Fatalf("native invocations %d want 24", launches)
	}
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	binaryBytes, err := kdaReadBounded(executable, 32<<20)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(out, "executed.test"), binaryBytes, 0600); err != nil {
		t.Fatal(err)
	}
	name, err := dev.Name()
	if err != nil {
		t.Fatal(err)
	}
	uuid, err := dev.Raw().Uuid()
	if err != nil {
		t.Fatal(err)
	}
	version, err := driver.DriverGetVersion()
	if err != nil {
		t.Fatal(err)
	}
	manifestPath := os.Getenv("KDA_SOURCE_MANIFEST")
	if !filepath.IsAbs(manifestPath) {
		t.Fatal("absolute source manifest path required")
	}
	sourceManifest, err := kdaReadBounded(manifestPath, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	producer := struct {
		GoVersion, Platform, DriverLibrary, DeviceName, DeviceUUID, Target, ExecutableSHA256, SourceManifestSHA256 string
		DriverVersion, RecurrenceInvocations, AdmissionRefusals                                                    int
		Absolute, Relative                                                                                         float64
	}{runtime.Version(), runtime.GOOS + "/" + runtime.GOARCH, driver.Path(), name, fmt.Sprintf("%x", uuid.Bytes), target, kdaDigest(binaryBytes), kdaDigest(sourceManifest), version, launches, admissionRefusals, kdaAbsolute, kdaRelative}
	if err := kdaJSON(filepath.Join(out, "producer.json"), producer); err != nil {
		t.Fatal(err)
	}
	t.Logf("COMPLETED 24 actual ordinary recurrence invocations; target=%s; numeric policy abs=%g rel=%g", target, kdaAbsolute, kdaRelative)
}

func TestKDAPacketAdmission(t *testing.T) {
	schema, err := KDASchema()
	if err != nil {
		t.Fatal(err)
	}
	// Synthetic bytes test parser agreement only; never loaded as CUDA code.
	raw := make([]byte, 64)
	copy(raw, []byte{0x7f, 'E', 'L', 'F', 2, 1, 1, 0x33, 7})
	binary.LittleEndian.PutUint16(raw[16:], 2)
	binary.LittleEndian.PutUint16(raw[18:], 190)
	binary.LittleEndian.PutUint32(raw[20:], 1)
	binary.LittleEndian.PutUint32(raw[48:], 121)
	binary.LittleEndian.PutUint16(raw[52:], 64)
	a, err := cudakernel.NewArtifact(cudakernel.KindCUBIN, raw, "kda_fp32", schema, "sm_121")
	if err != nil {
		t.Fatal(err)
	}
	a.Toolchain = "nvrtc 13.4"
	a.CompilerOpts = []string{"--gpu-architecture=sm_121", "--std=c++17", "--fmad=false", "--ftz=false"}
	a.Dependencies = []cudakernel.Dependency{{Path: "cudaattn_kda.cu", Digest: kdaDigest([]byte(KDASource)), Tracked: true}}
	entry := kdaPacketEntry{"baseline", kdaDigest([]byte(KDASource)), a.Digest, "sm_121", "/compiler/libnvrtc.so", strings.Repeat("a", 64), 13, 4, a.CompilerOpts}
	for _, name := range []string{"valid", "empty-path", "relative-path", "nul-path", "zero-major", "negative-minor", "version-mismatch", "empty-options", "empty-digest", "oversize-cu", "oversize-cubin", "oversize-artifact"} {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			e := entry
			switch name {
			case "empty-path":
				e.CompilerPath = ""
			case "relative-path":
				e.CompilerPath = "nvrtc.so"
			case "nul-path":
				e.CompilerPath = "/bad\x00path"
			case "zero-major":
				e.CompilerMajor = 0
			case "negative-minor":
				e.CompilerMinor = -1
			case "version-mismatch":
				e.CompilerMinor = 3
			case "empty-options":
				e.CompilerOptions = nil
			case "empty-digest":
				e.CompilerSHA256 = ""
			}
			for suffix, value := range map[string]any{".artifact.json": a, ".receipt.json": e} {
				if err := kdaJSON(filepath.Join(dir, "baseline"+suffix), value); err != nil {
					t.Fatal(err)
				}
			}
			for suffix, data := range map[string][]byte{".cu": []byte(KDASource), ".cubin": a.Raw} {
				if err := os.WriteFile(filepath.Join(dir, "baseline"+suffix), data, 0600); err != nil {
					t.Fatal(err)
				}
			}
			for _, oversize := range []struct {
				name, suffix string
				limit        int64
			}{{"oversize-cu", ".cu", 1 << 20}, {"oversize-cubin", ".cubin", 16 << 20}, {"oversize-artifact", ".artifact.json", 16 << 20}} {
				if name == oversize.name {
					if err := os.Truncate(filepath.Join(dir, "baseline"+oversize.suffix), oversize.limit+1); err != nil {
						t.Fatal(err)
					}
				}
			}
			_, _, err := kdaReadPacket(dir, "baseline", KDASource, "sm_121")
			if (err == nil) != (name == "valid") {
				t.Fatalf("%s: %v", name, err)
			}
		})
	}
}
