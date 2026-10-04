package cudareduce

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudablock"
	"github.com/tmc/lib/nvidia/x/cudakernel"
)

type distillPacket struct {
	Version                                                                                                 int
	FragmentDigest                                                                                          string
	SourceDigest, HeaderDigest, CompilerPath, CompilerVersion, CompilerDigest, BuiltinsPath, BuiltinsDigest string
	Artifact                                                                                                cudakernel.Artifact
}

func distillReadBounded(path string, limit int64) ([]byte, error) {
	info, err := os.Lstat(path)
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() || info.Size() < 1 || info.Size() > limit {
		return nil, fmt.Errorf("invalid distill packet file size or kind")
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	info, err = f.Stat()
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() || info.Size() < 1 || info.Size() > limit {
		return nil, fmt.Errorf("invalid opened distill packet file")
	}
	b, err := io.ReadAll(io.LimitReader(f, limit+1))
	if err != nil {
		return nil, err
	}
	if len(b) == 0 || int64(len(b)) > limit {
		return nil, fmt.Errorf("distill packet grew beyond limit")
	}
	return b, nil
}
func distillReadPacket(dir, name, source string) (distillPacket, error) {
	var p distillPacket
	if !filepath.IsAbs(dir) || name == "" || name != filepath.Base(name) || strings.ContainsAny(name, "/\\") || name == ".." {
		return p, fmt.Errorf("invalid distill packet path")
	}
	raw, err := distillReadBounded(filepath.Join(dir, name+".json"), 16<<20)
	if err != nil {
		return p, err
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&p); err != nil {
		return p, err
	}
	if err := decoder.Decode(new(any)); err != io.EOF {
		return p, fmt.Errorf("trailing distill packet JSON")
	}
	if p.Version != 1 || p.SourceDigest != distillDigest([]byte(source)) || p.HeaderDigest != distillDigest([]byte(cudablock.Header)) || p.CompilerVersion != "13.4.59" || p.Artifact.Toolchain != "nvrtc 13.4" || !filepath.IsAbs(p.CompilerPath) || filepath.Base(p.CompilerPath) != "libnvrtc.so.13.4.59" || !filepath.IsAbs(p.BuiltinsPath) || filepath.Base(p.BuiltinsPath) != "libnvrtc-builtins.so.13.4.59" || p.CompilerDigest != "7aed524a136022f5938f104223d4a205008562d29dc4f65359b2a5c089750fae" || p.BuiltinsDigest != "0fb388f79e4cb37b419cd6b7a8fdd9f954bd522fcb8e442378cba234fc169df4" {
		return p, fmt.Errorf("distill compiler/source receipt disagrees")
	}
	if err := distillAdmitArtifact(&p.Artifact, source); err != nil {
		return p, err
	}
	fragment, err := distillBlockHeader()
	if err != nil {
		return p, err
	}
	if p.FragmentDigest != distillDigest([]byte(fragment)) {
		return p, fmt.Errorf("distill fragment binding disagrees")
	}
	used, err := distillReadBounded(filepath.Join(dir, "cudablock-used.cuh"), 1<<20)
	if err != nil {
		return p, err
	}
	if !bytes.Equal(used, []byte(fragment)) {
		return p, fmt.Errorf("distill used fragment bytes disagree")
	}
	if p.Artifact.Target != "sm_121" {
		return p, fmt.Errorf("distill packet target disagrees")
	}
	for suffix, want := range map[string][]byte{".cu": []byte(source), ".cubin": p.Artifact.Raw} {
		limit := int64(16 << 20)
		if suffix == ".cu" {
			limit = 1 << 20
		}
		got, err := distillReadBounded(filepath.Join(dir, name+suffix), limit)
		if err != nil {
			return p, err
		}
		if !bytes.Equal(got, want) {
			return p, fmt.Errorf("distill retained bytes disagree")
		}
	}
	header, err := distillReadBounded(filepath.Join(dir, "cudablock.cuh"), 1<<20)
	if err != nil {
		return p, err
	}
	if !bytes.Equal(header, []byte(cudablock.Header)) {
		return p, fmt.Errorf("distill header bytes disagree")
	}
	return p, nil
}
func TestSparseTeacher_PacketBounds(t *testing.T) {
	root := t.TempDir()
	for _, name := range []string{"source.cu", "image.cubin"} {
		limit := int64(1 << 20)
		if strings.HasSuffix(name, "cubin") {
			limit = 16 << 20
		}
		path := filepath.Join(root, name)
		f, err := os.Create(path)
		if err != nil {
			t.Fatal(err)
		}
		if err := f.Truncate(limit + 1); err != nil {
			t.Fatal(err)
		}
		f.Close()
		if _, err := distillReadBounded(path, limit); err == nil {
			t.Fatal("oversized packet admitted")
		}
	}
	for _, name := range []string{"../escape", "", ".."} {
		if _, err := distillReadPacket(root, name, SparseTeacherSource); err == nil {
			t.Fatal("path admitted")
		}
	}
	empty := filepath.Join(root, "empty")
	os.WriteFile(empty, nil, 0600)
	if _, err := distillReadBounded(empty, 1<<20); err == nil {
		t.Fatal("empty admitted")
	}
	if _, err := distillReadBounded(root, 1<<20); err == nil {
		t.Fatal("directory admitted")
	}
}

func TestSparseTeacher_HeaderAndArtifactBinding(t *testing.T) {
	fragment, err := distillBlockHeader()
	if err != nil {
		t.Fatal(err)
	}
	expected := strings.Split(cudablock.Header, "// Block-level cooperative 2D tile loader")[0]
	expected = strings.ReplaceAll(expected, "#include <cuda_fp16.h>", "")
	expected = strings.ReplaceAll(expected, "#include <cuda_bf16.h>", "") + "\n#endif\n"
	if fragment != expected {
		t.Fatal("owner collective bodies changed")
	}
	for _, damaged := range []string{strings.Replace(cudablock.Header, "// Block-level cooperative 2D tile loader", "// lost", 1), cudablock.Header + "\n// Block-level cooperative 2D tile loader", strings.Replace(cudablock.Header, "#include <cuda_fp16.h>", "#include <other.h>", 1), strings.Replace(cudablock.Header, "T block_reduce_sum(", "T changed_reduce_sum(", 1), "#include <unexpected.h>\n" + cudablock.Header} {
		if _, err := distillTrimBlockHeader(damaged); err == nil {
			t.Fatal("unexpected header form admitted")
		}
	}
	schema, _ := SparseTeacherSchema()
	image := make([]byte, 64)
	copy(image, []byte{0x7f, 'E', 'L', 'F', 2, 1, 1, 0x33, 7})
	binary.LittleEndian.PutUint16(image[16:], 2)
	binary.LittleEndian.PutUint16(image[18:], 190)
	binary.LittleEndian.PutUint32(image[20:], 1)
	binary.LittleEndian.PutUint32(image[48:], 121)
	binary.LittleEndian.PutUint16(image[52:], 64)
	deps := []cudakernel.Dependency{{Path: "cudareduce_distill.cu", Digest: distillDigest([]byte(SparseTeacherSource)), Tracked: true}, {Path: "cudablock.cuh", Digest: distillDigest([]byte(fragment)), Tracked: true}, {Path: "cudablock.owner.cuh", Digest: distillDigest([]byte(cudablock.Header)), Tracked: true}}
	a, err := cudakernel.NewArtifact(cudakernel.KindCUBIN, image, "sparse_teacher", schema, "sm_121", cudakernel.WithToolchain("nvrtc 13.4"), cudakernel.WithCompilerOptions([]string{"--gpu-architecture=sm_121", "--std=c++17", "--fmad=false", "--ftz=false"}), cudakernel.WithDependencies(deps))
	if err != nil {
		t.Fatal(err)
	}
	if err := distillAdmitArtifact(a, SparseTeacherSource); err != nil {
		t.Fatal(err)
	}
	for _, edit := range []func(*cudakernel.Artifact){func(a *cudakernel.Artifact) {
		a.Dependencies[1].Digest = distillDigest([]byte("mutated actual fragment"))
	}, func(a *cudakernel.Artifact) { a.Dependencies[2].Digest = distillDigest([]byte("changed owner")) }, func(a *cudakernel.Artifact) { a.Dependencies = append(a.Dependencies, a.Dependencies[0]) }, func(a *cudakernel.Artifact) { a.CompilerOpts = append(a.CompilerOpts, "--use_fast_math") }, func(a *cudakernel.Artifact) { a.Schema.Params[0].Offset++ }} {
		damaged := distillClone(a)
		edit(damaged)
		if distillAdmitArtifact(damaged, SparseTeacherSource) == nil {
			t.Fatal("damaged artifact admitted")
		}
	}
	// Synthetic parser-only ELF. It is never loaded or called a compiler image.
	p := distillPacket{Version: 1, SourceDigest: distillDigest([]byte(SparseTeacherSource)), HeaderDigest: distillDigest([]byte(cudablock.Header)), FragmentDigest: distillDigest([]byte(fragment)), CompilerPath: "/tool/libnvrtc.so.13.4.59", CompilerVersion: "13.4.59", CompilerDigest: "7aed524a136022f5938f104223d4a205008562d29dc4f65359b2a5c089750fae", BuiltinsPath: "/tool/libnvrtc-builtins.so.13.4.59", BuiltinsDigest: "0fb388f79e4cb37b419cd6b7a8fdd9f954bd522fcb8e442378cba234fc169df4", Artifact: *a}
	dir := t.TempDir()
	write := func(p distillPacket) {
		raw, err := json.Marshal(p)
		if err != nil {
			t.Fatal(err)
		}
		for suffix, data := range map[string][]byte{"baseline.json": raw, "baseline.cu": []byte(SparseTeacherSource), "baseline.cubin": image, "cudablock.cuh": []byte(cudablock.Header), "cudablock-used.cuh": []byte(fragment)} {
			if err := os.WriteFile(filepath.Join(dir, suffix), data, 0600); err != nil {
				t.Fatal(err)
			}
		}
	}
	write(p)
	if _, err := distillReadPacket(dir, "baseline", SparseTeacherSource); err != nil {
		t.Fatal(err)
	}
	for _, edit := range []func(*distillPacket){func(p *distillPacket) { p.CompilerPath = "" }, func(p *distillPacket) { p.CompilerPath = "libnvrtc.so.13.4.59" }, func(p *distillPacket) { p.CompilerVersion = "13.0.88" }, func(p *distillPacket) { p.BuiltinsPath = "/tool/libnvrtc-builtins.so.13.0.88" }, func(p *distillPacket) { p.CompilerDigest = "unknown" }, func(p *distillPacket) { p.FragmentDigest = distillDigest([]byte("wrong compiled fragment")) }, func(p *distillPacket) { p.HeaderDigest = p.FragmentDigest }} {
		q := p
		edit(&q)
		write(q)
		if _, err := distillReadPacket(dir, "baseline", SparseTeacherSource); err == nil {
			t.Fatal("bad packet metadata admitted")
		}
	}
	write(p)
	if err := os.WriteFile(filepath.Join(dir, "cudablock-used.cuh"), []byte(fragment+"// mutated"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := distillReadPacket(dir, "baseline", SparseTeacherSource); err == nil {
		t.Fatal("changed retained fragment admitted")
	}
}

type distillHostResult struct {
	Status                   []int32
	Losses, Gradients, Stats []float32
	Total                    float32
}

var distillHostCache struct {
	sync.Mutex
	binaries map[string]string
}

func distillHostBinary(t *testing.T, source string) string {
	t.Helper()
	distillHostCache.Lock()
	defer distillHostCache.Unlock()
	if distillHostCache.binaries == nil {
		distillHostCache.binaries = make(map[string]string)
	}
	key := distillDigest([]byte(source))
	if p := distillHostCache.binaries[key]; p != "" {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	root := t.TempDir()
	path := filepath.Join(root, "control.cc")
	binary := filepath.Join(root, "control")
	// The unchanged FP32 collective definitions are executed with all 32 host
	// threads and barriers. This test-only shim is not device execution.
	header, headerErr := distillBlockHeader()
	if headerErr != nil {
		t.Fatal(headerErr)
	}
	prelude := `#include <algorithm>
#include <barrier>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>
#define __device__
#define __global__
#define __forceinline__ inline
#define __shared__ static
struct dim { int x=0; }; thread_local dim threadIdx,blockIdx; dim blockDim{32};
std::barrier syncpoint(32); long double exchange[32];
void __syncthreads(){syncpoint.arrive_and_wait();}
template<class T> T __shfl_down_sync(unsigned,T x,int offset){int i=threadIdx.x;exchange[i]=x;__syncthreads();T out=(i+offset<32)?(T)exchange[i+offset]:x;__syncthreads();return out;}
unsigned __float_as_uint(float x){return std::bit_cast<unsigned>(x);}
float __int_as_float(int x){return std::bit_cast<float>(x);}
`
	main := `
int main(){int rows,vocab,k,objective;unsigned tb;std::cin>>rows>>vocab>>k>>objective>>tb;float temperature=__int_as_float(tb);
std::vector<float> z(rows*vocab),w(rows*k),stats(rows*4,-99),loss(rows,-99),grad(rows*vocab,-99);std::vector<int> ids(rows*k),masks(rows),status(rows,-1);float total=-99;
for(auto&x:z){unsigned b;std::cin>>b;x=__int_as_float(b);}for(auto&x:w){unsigned b;std::cin>>b;x=__int_as_float(b);}for(auto&x:ids)std::cin>>x;for(auto&x:masks)std::cin>>x;
for(int stage=0;stage<3;stage++){for(int row=0;row<(stage==2?1:rows);row++){std::vector<std::thread> threads;for(int lane=0;lane<32;lane++)threads.emplace_back([&,row,lane]{threadIdx.x=lane;blockIdx.x=row;sparse_teacher(z.data(),ids.data(),w.data(),masks.data(),status.data(),stats.data(),loss.data(),grad.data(),&total,rows,vocab,k,objective,temperature,stage);});for(auto&t:threads)t.join();}bool bad=false;for(int s:status)if(s)bad=true;if(bad)break;}
for(int s:status)std::cout<<s<<' ';for(float x:loss)std::cout<<__float_as_uint(x)<<' ';for(float x:grad)std::cout<<__float_as_uint(x)<<' ';for(float x:stats)std::cout<<__float_as_uint(x)<<' ';std::cout<<__float_as_uint(total)<<'\n';}
`
	code := prelude + strings.Replace(source, "#include \"cudablock.cuh\"", header, 1) + main
	if err := os.WriteFile(path, []byte(code), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30e9)
	defer cancel()
	out, err := exec.CommandContext(ctx, "c++", "-std=c++20", "-O2", "-pthread", "-ffp-contract=off", path, "-o", binary).CombinedOutput()
	if err != nil {
		t.Fatalf("host CUDA control compile: %v\n%s", err, out)
	}
	// Cached binaries are usable only while their owning test tempdir exists.
	distillHostCache.binaries[key] = binary
	return binary
}
func distillRunHost(t *testing.T, f distillFixture, source string) distillHostResult {
	t.Helper()
	binary := distillHostBinary(t, source)
	var input bytes.Buffer
	c := f.Config
	fmt.Fprintf(&input, "%d %d %d %d %d ", c.Rows, c.Vocabulary, c.Retained, c.Objective, math.Float32bits(c.Temperature))
	for _, s := range [][]float32{f.Z, f.W} {
		for _, x := range s {
			fmt.Fprintf(&input, "%d ", math.Float32bits(x))
		}
	}
	for _, s := range [][]int32{f.IDs, f.Masks} {
		for _, x := range s {
			fmt.Fprintf(&input, "%d ", x)
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15e9)
	defer cancel()
	cmd := exec.CommandContext(ctx, binary)
	cmd.Stdin = &input
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("host CUDA control: %v\n%s", err, out)
	}
	r := bytes.NewReader(out)
	result := distillHostResult{Status: make([]int32, c.Rows), Losses: make([]float32, c.Rows), Gradients: make([]float32, len(f.Z)), Stats: make([]float32, c.Rows*4)}
	for i := range result.Status {
		if _, err := fmt.Fscan(r, &result.Status[i]); err != nil {
			t.Fatal(err)
		}
	}
	for _, s := range [][]float32{result.Losses, result.Gradients, result.Stats} {
		for i := range s {
			var bits uint32
			if _, err := fmt.Fscan(r, &bits); err != nil {
				t.Fatal(err)
			}
			s[i] = math.Float32frombits(bits)
		}
	}
	var bits uint32
	if _, err := fmt.Fscan(r, &bits); err != nil {
		t.Fatal(err)
	}
	result.Total = math.Float32frombits(bits)
	return result
}
func distillFixtureByName(name string) distillFixture {
	for _, f := range distillFixtures() {
		if f.Name == name {
			return f
		}
	}
	panic("unknown private fixture")
}

func TestSparseTeacher_NormalizedSupportDenseOracle(t *testing.T) {
	for _, f := range distillFixtures() {
		if f.Config.Objective == NormalizedSupport {
			t.Run(f.Name, func(t *testing.T) {
				r := distillRunHost(t, f, SparseTeacherSource)
				if err := CheckSparseTeacherStatus(r.Status, f.Config.Rows); err != nil {
					t.Fatal(err)
				}
				distillCheckReference(t, f, r)
			})
		}
	}
}
func TestSparseTeacher_FullLogitGradient(t *testing.T) {
	for _, name := range []string{"normalized_outside", "tail_irregular", "tail_underflow"} {
		f := distillFixtureByName(name)
		_, want, _ := distillReference(f)
		r := distillRunHost(t, f, SparseTeacherSource)
		if err := distillCompare(f, "host full gradient", r.Gradients, want); err != nil {
			t.Fatal(err)
		}
		for j := range f.Z {
			old := f.Z[j]
			h := float32(0.01)
			f.Z[j] = old + h
			_, _, plus := distillReference(f)
			f.Z[j] = old - h
			_, _, minus := distillReference(f)
			f.Z[j] = old
			derivative := (plus - minus) / float64((old+h)-(old-h))
			if math.Abs(derivative-want[j]) > 2e-5 {
				t.Fatalf("%s gradient %d finite difference %g want %g", name, j, derivative, want[j])
			}
		}
	}
}
func TestSparseTeacher_TailBucketDenseOracle(t *testing.T) {
	for _, f := range distillFixtures() {
		if f.Config.Objective == PreservedTail {
			t.Run(f.Name, func(t *testing.T) {
				r := distillRunHost(t, f, SparseTeacherSource)
				if err := CheckSparseTeacherStatus(r.Status, f.Config.Rows); err != nil {
					t.Fatal(err)
				}
				distillCheckReference(t, f, r)
			})
		}
	}
}
func TestSparseTeacher_ZeroAndFullSupport(t *testing.T) {
	for _, name := range []string{"normalized_singleton", "normalized_full", "normalized_zero_weight", "tail_full", "tail_zero_other"} {
		f := distillFixtureByName(name)
		if err := ValidateSparseTeacher(f.Config, f.Z, f.IDs, f.W, f.Masks); err != nil {
			t.Fatal(err)
		}
		distillCheckReference(t, f, distillRunHost(t, f, SparseTeacherSource))
	}
	f := distillFixtureByName("tail_full")
	f.W[0] = 0
	if ValidateSparseTeacher(f.Config, f.Z, f.IDs, f.W, f.Masks) == nil {
		t.Fatal("accepted absent bucket with mass")
	}
}

func distillInvalidFixtures() []distillFixture {
	var out []distillFixture
	for _, name := range []string{"duplicate", "range", "negative weight", "zero mass", "nan logit", "infinite logit", "negative infinite logit", "nan weight", "mask", "tail excess"} {
		f := distillFixtureByName("normalized_outside")
		f.Name = name
		f.Z = append([]float32(nil), f.Z...)
		f.W = append([]float32(nil), f.W...)
		f.IDs = append([]int32(nil), f.IDs...)
		f.Masks = append([]int32(nil), f.Masks...)
		switch name {
		case "duplicate":
			f.IDs[1] = f.IDs[0]
		case "range":
			f.IDs[0] = int32(f.Config.Vocabulary)
		case "negative weight":
			f.W[0] = -1
		case "zero mass":
			f.W[0] = 0
			f.W[1] = 0
		case "nan logit":
			f.Z[0] = float32(math.NaN())
		case "infinite logit":
			f.Z[0] = float32(math.Inf(1))
		case "negative infinite logit":
			f.Z[0] = float32(math.Inf(-1))
		case "nan weight":
			f.W[0] = float32(math.NaN())
		case "mask":
			f.Masks[0] = 2
		case "tail excess":
			f.Config.Objective = PreservedTail
			f.Config.Temperature = 1
		}
		out = append(out, f)
	}
	return out
}
func TestSparseTeacher_ValidationAndMasks(t *testing.T) {
	for _, f := range distillInvalidFixtures() {
		t.Run(f.Name, func(t *testing.T) {
			if ValidateSparseTeacher(f.Config, f.Z, f.IDs, f.W, f.Masks) == nil {
				t.Fatal("host admitted invalid input")
			}
			r := distillRunHost(t, f, SparseTeacherSource)
			if CheckSparseTeacherStatus(r.Status, f.Config.Rows) == nil {
				t.Fatal("resident validation admitted invalid input")
			}
			for _, v := range r.Losses {
				if v != -99 {
					t.Fatal("invalid input wrote math")
				}
			}
		})
	}
	f := distillFixtureByName("all_masked")
	r := distillRunHost(t, f, SparseTeacherSource)
	distillCheckReference(t, f, r)
	if r.Total != 0 {
		t.Fatal("all-masked nonzero total")
	}
	for _, mutate := range []func(*SparseTeacherConfig){func(c *SparseTeacherConfig) { c.Temperature = 0 }, func(c *SparseTeacherConfig) { c.Temperature = float32(math.NaN()) }, func(c *SparseTeacherConfig) { c.Objective = PreservedTail; c.Temperature = 2 }, func(c *SparseTeacherConfig) { c.Rows = math.MaxInt }, func(c *SparseTeacherConfig) { c.Retained = 0 }, func(c *SparseTeacherConfig) { c.Vocabulary = math.MaxInt }} {
		c := f.Config
		mutate(&c)
		if _, err := c.Extents(); err == nil {
			t.Fatal("admitted invalid config")
		}
	}
	if ValidateSparseTeacher(f.Config, f.Z[:1], f.IDs, f.W, f.Masks) == nil {
		t.Fatal("admitted wrong extents")
	}
}

type distillMutant struct{ Name, Old, New, Fixture string }

func distillMutants() []distillMutant {
	return []distillMutant{
		{"support_student", "s+=expf(x-maximum);", "if([&](){for(int i=0;i<retained;i++)if(ids[row*retained+i]==j)return true;return false;}()) s+=expf(x-maximum);", "normalized_outside"},
		{"remove_tail", "double other=objective==1 ? 1-mass : 0;", "double other=0;", "tail_underflow"},
		{"renormalize_preserved", "if(objective==0) p/=mass;", "p/=mass;", "tail_irregular"},
		{"remove_temperature", "float grad=(float)((q-p)/temperature);", "float grad=(float)(q-p);", "normalized_outside"},
		{"remove_outside_gradient", "gradients[row*vocab+j]=grad;", "if(!found) grad=0; gradients[row*vocab+j]=grad;", "normalized_outside"},
	}
}
func distillMutantSource(m distillMutant) string {
	return strings.ReplaceAll(SparseTeacherSource, m.Old, m.New)
}
func TestSparseTeacher_Mutants(t *testing.T) {
	for _, m := range distillMutants() {
		t.Run(m.Name, func(t *testing.T) {
			if !strings.Contains(SparseTeacherSource, m.Old) {
				t.Fatal("mutant anchor absent")
			}
			f := distillFixtureByName(m.Fixture)
			r := distillRunHost(t, f, distillMutantSource(m))
			if err := CheckSparseTeacherStatus(r.Status, f.Config.Rows); err != nil {
				t.Fatalf("mutant control failed operationally: %v", err)
			}
			loss, grad, total := distillReference(f)
			a := distillCompare(f, "host mutant", r.Losses, loss)
			b := distillCompare(f, "host mutant", r.Gradients, grad)
			c := distillCompare(f, "host mutant", []float32{r.Total}, []float64{total})
			if a == nil && b == nil && c == nil {
				t.Fatal("mutant survived")
			}
			t.Logf("mutant killed: loss=%v gradient=%v total=%v", a, b, c)
		})
	}
}

func TestSparseTeacher_TailAndScratchGuards(t *testing.T) {
	dev := new(cuda.Device)
	views := []distillView{{"a", 0x1004, 2, 2, dev, func(*cuda.Device) (func(), error) { return func() {}, nil }}, {"b", 0x2004, 3, 3, dev, func(*cuda.Device) (func(), error) { return func() {}, nil }}}
	if err := distillCheckViews(views, dev); err != nil {
		t.Fatal(err)
	}
	views[1].ptr = 0x1008
	if distillCheckViews(views, dev) == nil {
		t.Fatal("alias admitted")
	}
	views[1].ptr = 0x2004
	released := 0
	views[0].borrow = func(*cuda.Device) (func(), error) { return func() { released++ }, nil }
	views[1].borrow = func(*cuda.Device) (func(), error) { return nil, errors.New("borrow failure") }
	if _, err := distillBorrow(views, dev); err == nil || released != 1 {
		t.Fatal("partial borrow not unwound")
	}
	f := distillFixtureByName("tail_irregular")
	first := distillRunHost(t, f, SparseTeacherSource)
	distillCheckReference(t, f, first)
	for i := range f.W {
		f.W[i] *= 0.5
	}
	f.Masks[0] = 0
	f.Masks[1] = 1
	second := distillRunHost(t, f, SparseTeacherSource)
	distillCheckReference(t, f, second)
	if second.Losses[0] != 0 || first.Losses[0] == 0 {
		t.Fatal("stale masked result")
	}
	var o SparseTeacher
	if o.Execute(context.Background(), nil, SparseTeacherBuffers{}) == nil {
		t.Fatal("zero owner executed")
	}
	if err := o.Close(); err != nil {
		t.Fatal(err)
	}
}
func TestSparseTeacher_LaunchPointerArray(t *testing.T) {
	c := distillFixtureByName("tail_irregular").Config
	schema, _ := SparseTeacherSchema()
	var ptrs [9]driver.DevicePtr
	for i := range ptrs {
		ptrs[i] = driver.DevicePtr(0x1004 + i*0x100)
	}
	args, err := distillArguments(schema, c, ptrs, 1)
	if err != nil {
		t.Fatal(err)
	}
	for i, w := range ptrs {
		if uintptr(args[i])%8 != 0 || *(*driver.DevicePtr)(args[i]) != w {
			t.Fatal("pointer ABI mismatch")
		}
	}
	for i, w := range []int32{int32(c.Rows), int32(c.Vocabulary), int32(c.Retained), int32(c.Objective)} {
		if *(*int32)(args[i+9]) != w {
			t.Fatal("scalar ABI mismatch")
		}
	}
	if *(*float32)(args[13]) != c.Temperature || *(*int32)(args[14]) != 1 {
		t.Fatal("temperature/stage mismatch")
	}
	ptrs[0]++
	if _, err := distillArguments(schema, c, ptrs, 1); err == nil {
		t.Fatal("unaligned admitted")
	}
	schema.Params[0].Type = "s32"
	if _, err := distillArguments(schema, c, ptrs, 1); err == nil {
		t.Fatal("wrong schema admitted")
	}
	_ = unsafe.Sizeof(args)
}
func TestSparseTeacher_SourceBinding(t *testing.T) {
	schema, _ := SparseTeacherSchema()
	a, err := cudakernel.NewArtifact(cudakernel.KindCUBIN, []byte{0x7f, 'E', 'L', 'F', 1}, "sparse_teacher", schema, "sm_121")
	if err == nil && distillAdmitArtifact(a, SparseTeacherSource) == nil {
		t.Fatal("unbound artifact admitted")
	}
}
func FuzzSparseTeacherAdmission(f *testing.F) {
	f.Add(int32(1), float32(1), int32(1))
	f.Fuzz(func(t *testing.T, id int32, w float32, mask int32) {
		c := SparseTeacherConfig{1, 3, 1, NormalizedSupport, 1}
		err := ValidateSparseTeacher(c, []float32{1, 2, 3}, []int32{id}, []float32{w}, []int32{mask})
		want := id >= 0 && id < 3 && distillFinite(w) && w > 0 && (mask == 0 || mask == 1)
		if (err == nil) != want {
			t.Fatal("admission disagrees")
		}
	})
}

func distillOnlyCanceled(err error) bool {
	if err == context.Canceled {
		return true
	}
	if joined, ok := err.(interface{ Unwrap() []error }); ok {
		children := joined.Unwrap()
		if len(children) == 0 {
			return false
		}
		for _, child := range children {
			if !distillOnlyCanceled(child) {
				return false
			}
		}
		return true
	}
	if child, ok := err.(interface{ Unwrap() error }); ok {
		return distillOnlyCanceled(child.Unwrap())
	}
	return false
}
func TestSparseTeacher_CancellationClassification(t *testing.T) {
	if !distillOnlyCanceled(errors.Join(context.Canceled)) || distillOnlyCanceled(errors.Join(context.Canceled, errors.New("driver failure"))) || distillOnlyCanceled(nil) {
		t.Fatal("mixed operational error accepted as cancellation")
	}
}

func TestSparseTeacher_ArithmeticDomain(t *testing.T) {
	f := distillFixture{Name: "finite_loss_overflow", Config: SparseTeacherConfig{1, 2, 1, NormalizedSupport, 1}, Z: []float32{-math.MaxFloat32, math.MaxFloat32}, W: []float32{1}, IDs: []int32{0}, Masks: []int32{1}}
	if err := ValidateSparseTeacher(f.Config, f.Z, f.IDs, f.W, f.Masks); err != nil {
		t.Fatal(err)
	}
	r := distillRunHost(t, f, SparseTeacherSource)
	if len(r.Status) != 1 || r.Status[0] != 2 {
		t.Fatal("nonfinite loss did not refuse")
	}
	f.Config.Temperature = math.SmallestNonzeroFloat32
	if ValidateSparseTeacher(f.Config, f.Z, f.IDs, f.W, f.Masks) == nil {
		t.Fatal("scaled student overflow admitted")
	}
	r = distillRunHost(t, f, SparseTeacherSource)
	if r.Status[0] != 1 {
		t.Fatal("scaled overflow escaped resident validation")
	}
}
