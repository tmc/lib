//go:build ds4cpumodel && ds4cpuactualmodel

package dsv4cpu

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
)

func TestActualLayerZeroSourceAdmission(t *testing.T) {
	c := actualLayerZeroHCSource(t)
	dir := t.TempDir()
	path := filepath.Join(dir, "l0.c")
	if err := os.WriteFile(path, []byte(c), 0600); err != nil {
		t.Fatal(err)
	}
	arch := "-march=native"
	if runtime.GOOS == "darwin" {
		arch = "-mcpu=native"
	}
	args := []string{"-O3", "-ffast-math", "-fno-finite-math-only", arch, path, "-lm", "-o", filepath.Join(dir, "l0")}
	if out, err := exec.Command("cc", args...).CombinedOutput(); err != nil {
		t.Fatalf("compile literal L0 operator: %v %s", err, out)
	}
	t.Logf("literal L0 source sha256=%x flags=%q; no model binding or forward", sha256.Sum256([]byte(c)), args[:4])
}

// This diagnostic never evaluates the complete model. Both operators consume
// the actual token-128000 embedding and actual layer-zero HC tensors.
func TestFlashActualLayerZeroHC(t *testing.T) {
	if runtime.GOARCH != "arm64" {
		t.Fatal("owning optimized ARM CPU reference requires arm64")
	}
	path, output := os.Getenv("DS4_CPU_MODEL"), os.Getenv("DS4_CPU_L0_OUTPUT")
	if path == "" || output == "" {
		t.Fatal("DS4_CPU_MODEL and DS4_CPU_L0_OUTPUT are required")
	}
	if _, err := os.Stat(output); err == nil || !os.IsNotExist(err) {
		t.Fatal("layer-zero receipt must not exist", err)
	}
	c := actualLayerZeroHCSource(t)
	b, err := openFlashBackend(path, "ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0")
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := b.Close(); err != nil {
			t.Errorf("close model: %v", err)
		}
	}()
	m := b.model
	h := &m.layers[0].attentionHC
	if !h.nativeF16 {
		t.Fatal("actual model did not select native HC projection")
	}
	embedding, state := make([]float32, 4096), make([]float32, 16384)
	if err := m.embedding.Row(embedding, 128000); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 4; i++ {
		copy(state[i*4096:], embedding)
	}
	embeddingBytes := make([]byte, m.embedding.rowBytes)
	n, err := m.embedding.source.ReadAt(embeddingBytes, m.embedding.offset+int64(128000*m.embedding.rowBytes))
	if err != nil || n != len(embeddingBytes) {
		t.Fatalf("read actual embedding: %v bytes=%d want=%d", err, n, len(embeddingBytes))
	}
	weights := make([]byte, h.projection.rows*h.projection.rowBytes)
	n, err = h.projection.source.ReadAt(weights, h.projection.offset)
	if err != nil || n != len(weights) {
		t.Fatalf("read actual HC projection: %v bytes=%d want=%d", err, n, len(weights))
	}
	flat, mix := make([]float32, 16384), make([]float32, 24)
	plain, post, comb, norm := make([]float32, 4096), make([]float32, 4), make([]float32, 16), make([]float32, 4096)
	if err := RMSNorm(flat, state, nil, 1e-6); err != nil {
		t.Fatal(err)
	}
	if err := nativeF16Mul(mix, flat, h.projection); err != nil {
		t.Fatal(err)
	}
	if err := h.pre(plain, post, comb, state); err != nil {
		t.Fatal(err)
	}
	if err := RMSNorm(norm, plain, m.layers[0].attnNorm, 1e-6); err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	file, bin := filepath.Join(dir, "actual-l0.c"), filepath.Join(dir, "actual-l0")
	if err := os.WriteFile(file, []byte(c), 0600); err != nil {
		t.Fatal(err)
	}
	if err := writeCPUForwardOutput(output+".c", []byte(c)); err != nil {
		t.Fatal(err)
	}
	arch := "-march=native"
	if runtime.GOOS == "darwin" {
		arch = "-mcpu=native"
	}
	flags := []string{"-O3", "-ffast-math", "-g", arch, "-Wall", "-Wextra", "-std=c99", "-D_GNU_SOURCE", "-fno-finite-math-only", file, "-lm", "-o", bin}
	compileOutput, compileErr := exec.Command("cc", flags...).CombinedOutput()
	if err := writeCPUForwardOutput(output+".compile", compileOutput); err != nil {
		t.Fatal(err)
	}
	if compileErr != nil {
		t.Fatalf("compile owning optimized HC reference: %v %s", compileErr, compileOutput)
	}
	compiled, err := os.ReadFile(bin)
	if err != nil {
		t.Fatal(err)
	}
	if err := writeCPUForwardOutput(output+".oracle", compiled); err != nil {
		t.Fatal(err)
	}
	var in bytes.Buffer
	in.Write(embeddingBytes)
	in.Write(weights)
	for _, values := range [][]float32{flat, h.scale, h.base, m.layers[0].attnNorm} {
		if err := binary.Write(&in, binary.LittleEndian, values); err != nil {
			t.Fatal(err)
		}
	}
	input := append([]byte(nil), in.Bytes()...)
	if err := writeCPUForwardOutput(output+".input", input); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(bin)
	cmd.Stdin = &in
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	want, runErr := cmd.Output()
	if err := writeCPUForwardOutput(output+".C.f32", want); err != nil {
		t.Fatal(err)
	}
	if err := writeCPUForwardOutput(output+".stderr", stderr.Bytes()); err != nil {
		t.Fatal(err)
	}
	var got bytes.Buffer
	for _, values := range [][]float32{embedding, flat, mix, plain, post, comb, norm, mix} {
		if err := binary.Write(&got, binary.LittleEndian, values); err != nil {
			t.Fatal(err)
		}
	}
	if err := writeCPUForwardOutput(output+".Go.f32", got.Bytes()); err != nil {
		t.Fatal(err)
	}
	if runErr != nil || len(want) != (4096+16384+24+4096+4+16+4096+24)*4 {
		t.Fatalf("actual HC reference: %v bytes=%d stderr=%s", runErr, len(want), stderr.Bytes())
	}
	if err := writeCPUForwardOutput(output, append(want, got.Bytes()...)); err != nil {
		t.Fatal(err)
	}
	t.Logf("token=128000 position=0 layer=0 input_sha256=%x C_source_sha256=%x flags=%q qualification=NO_VERDICT_control_out_of_bound", sha256.Sum256(input), sha256.Sum256([]byte(c)), flags[:9])
	first := "none"
	offset := 0
	for _, stage := range []struct {
		name string
		n    int
	}{{"embedding", 4096}, {"hc_flat", 16384}, {"hc_mix", 24}, {"hc_attn_pre", 4096}, {"hc_post_weights", 4}, {"hc_comb", 16}, {"attn_norm", 4096}, {"matched_input_mix", 24}} {
		changed, violations := 0, 0
		maxAbs, ss, energy := 0.0, 0.0, 0.0
		for i := 0; i < stage.n; i++ {
			w := math.Float32frombits(binary.LittleEndian.Uint32(want[(offset+i)*4:]))
			v := math.Float32frombits(binary.LittleEndian.Uint32(got.Bytes()[(offset+i)*4:]))
			if math.IsNaN(float64(w)) || math.IsInf(float64(w), 0) || math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("nonfinite actual stage %s index %d", stage.name, i)
			}
			d := math.Abs(float64(v) - float64(w))
			if math.Float32bits(v) != math.Float32bits(w) {
				changed++
			}
			if d > 1e-4+1e-4*math.Abs(float64(w)) {
				violations++
			}
			maxAbs = math.Max(maxAbs, d)
			ss += d * d
			energy += float64(w) * float64(w)
		}
		t.Logf("stage=%s count=%d changed=%d violations=%d max_abs=%.9g rms=%.9g reference_rms=%.9g", stage.name, stage.n, changed, violations, maxAbs, math.Sqrt(ss/float64(stage.n)), math.Sqrt(energy/float64(stage.n)))
		if violations != 0 && first == "none" {
			first = stage.name
		}
		if (stage.name == "embedding" || stage.name == "matched_input_mix") && changed != 0 {
			t.Errorf("stage %s differs on identical actual inputs", stage.name)
		}
		offset += stage.n
	}
	t.Logf("first_local_F5_exceeding_stage=%s", first)
	if first != "none" {
		t.Error("actual L0 stage numerical bound exceeded")
	}
}

func actualLayerZeroHCSource(t *testing.T) string {
	t.Helper()
	source := string(pinnedCPUOracle(t))
	c := `#include <arm_neon.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define DS4_N_HC 4
#define DS4_N_EMBD 4096
#define DS4_RMS_EPS 1.0e-6f
#define DS4_NEG_INF (-1.0e30f)
#define DS4_N_HC_SINKHORN_ITER 20
typedef struct{int unused;}ds4_model;
typedef struct{const void *data;}ds4_tensor;
static const void *tensor_data(const ds4_model *m,const ds4_tensor *t){return t->data;}
static float saved_mix[24];
`
	for _, name := range []string{"f16_to_f32", "dot_f16_row", "rms_norm_no_weight", "rms_norm_weight", "hc_split_sinkhorn_one", "hc_weighted_sum_one"} {
		c += oracleFunction(t, source, name) + "\n"
	}
	c += `static void matvec_f16(float *out,const ds4_model *m,const ds4_tensor *w,const float *x){const uint16_t *data=w->data;for(int r=0;r<24;r++)out[r]=dot_f16_row(data+r*16384,x,16384);memcpy(saved_mix,out,sizeof(saved_mix));}
static void matvec_f16_serial(float *out,const ds4_model *m,const ds4_tensor *w,const float *x){matvec_f16(out,m,w,x);}
`
	c += oracleFunction(t, source, "hc_pre_from_state_one_scratch") + "\n"
	c += `int main(void){uint16_t embed[4096],*fn=malloc(24*16384*2);float embedding[4096],state[16384],flat[16384],gflat[16384],out[4096],post[4],comb[16],norm[4096],nw[4096],scale[3],base[24],matched[24];
if(!fn||fread(embed,2,4096,stdin)!=4096||fread(fn,2,24*16384,stdin)!=24*16384||fread(gflat,4,16384,stdin)!=16384||fread(scale,4,3,stdin)!=3||fread(base,4,24,stdin)!=24||fread(nw,4,4096,stdin)!=4096)return 2;
for(int i=0;i<4096;i++)embedding[i]=f16_to_f32(embed[i]);for(int h=0;h<4;h++)memcpy(state+h*4096,embedding,sizeof(embedding));
ds4_model m={0};ds4_tensor w={fn},sc={scale},b={base};hc_pre_from_state_one_scratch(&m,&w,&sc,&b,state,out,post,comb,flat,false);
rms_norm_weight(norm,out,nw,4096,DS4_RMS_EPS);for(int r=0;r<24;r++)matched[r]=dot_f16_row(fn+r*16384,gflat,16384);
if(fwrite(embedding,4,4096,stdout)!=4096||fwrite(flat,4,16384,stdout)!=16384||fwrite(saved_mix,4,24,stdout)!=24||fwrite(out,4,4096,stdout)!=4096||fwrite(post,4,4,stdout)!=4||fwrite(comb,4,16,stdout)!=16||fwrite(norm,4,4096,stdout)!=4096||fwrite(matched,4,24,stdout)!=24)return 2;free(fn);return 0;}`
	return c
}
