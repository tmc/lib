//go:build ds4cpumodel && ds4cpuactualmodel

package dsv4cpu

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
)

type qkvStage struct {
	name   string
	values []float32
	exact  bool
}

func qkvQuantStage(t *testing.T, name string, x []float32) ([]qkvStage, []Q80Activation) {
	t.Helper()
	a := make([]Q80Activation, len(x)/32)
	if err := QuantizeQ80(a, x); err != nil {
		t.Fatal(err)
	}
	qs, scale := make([]float32, len(x)), make([]float32, len(a))
	for b, v := range a {
		scale[b] = v.Scale
		for i, q := range v.Values {
			qs[b*32+i] = float32(q)
		}
	}
	return []qkvStage{{name + "_qs", qs, true}, {name + "_scale", scale, false}}, a
}

// This diagnostic extends the retained actual L0 state, not State.Eval.
// RoPE positions1/64 are controlled operator probes on the position0 heads.
func TestFlashActualLayerZeroQKV(t *testing.T) {
	if runtime.GOARCH != "arm64" {
		t.Fatal("owning optimized ARM reference requires arm64")
	}
	model, sourceDir, output := os.Getenv("DS4_CPU_MODEL"), os.Getenv("DS4_CPU_L0_RECEIPT"), os.Getenv("DS4_CPU_QKV_OUTPUT")
	if model == "" || sourceDir == "" || output == "" {
		t.Fatal("model, L0 receipt and QKV output are required")
	}
	for _, suffix := range []string{".c", ".input", ".compile", ".oracle", ".C.f32", ".Go.f32", ".stderr"} {
		if _, err := os.Stat(output + suffix); err == nil || !os.IsNotExist(err) {
			t.Fatal("QKV artifact must not exist", output+suffix, err)
		}
	}
	c := actualLayerZeroQKVSource(t)
	read := func(name, expected string) []float32 {
		raw, err := os.ReadFile(filepath.Join(sourceDir, name))
		if err != nil {
			t.Fatal(err)
		}
		if fmt.Sprintf("%x", sha256.Sum256(raw)) != expected {
			t.Fatal("retained L0 receipt identity differs", name)
		}
		if len(raw) != (4096+16384+24+4096+4+16+4096+24)*4 {
			t.Fatal("retained L0 shape differs")
		}
		x := make([]float32, 4096)
		offset := (4096 + 16384 + 24 + 4096 + 4 + 16) * 4
		for i := range x {
			x[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[offset+i*4:]))
		}
		return x
	}
	cnorm := read("L0.C.f32", "76f38212162a8269e5d3420404e691f0febc20bfa4719f6ee73689956aef3e49")
	gnorm := read("L0.Go.f32", "87967ecb7e85d1e7ef05ebc2e10b24be45e038fac8754226d797a7ac1a57b66b")
	b, err := openFlashBackend(model, "ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0")
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := b.Close(); err != nil {
			t.Error(err)
		}
	}()
	l := &b.model.layers[0]
	for _, want := range []struct {
		name       string
		m          *Matrix
		rows, cols int
	}{{"QA", l.qA, 1024, 4096}, {"QB", l.qB, 32768, 1024}, {"KV", l.kv, 512, 4096}} {
		if want.m == nil || want.m.encoding != Q80 || want.m.rows != want.rows || want.m.cols != want.cols || want.m.rowBytes != want.cols/32*34 {
			t.Fatal("actual Q8 tensor admission differs", want.name)
		}
	}
	stages, ga := qkvQuantStage(t, "attn_activation", gnorm)
	qa, qr, qb, qhead, kv, kvnorm := make([]float32, 1024), make([]float32, 1024), make([]float32, 32768), make([]float32, 32768), make([]float32, 512), make([]float32, 512)
	if err := cpuDense(qa, gnorm, l.qA); err != nil {
		t.Fatal(err)
	}
	if err := RMSNorm(qr, qa, l.qNorm, 1e-6); err != nil {
		t.Fatal(err)
	}
	stages = append(stages, qkvStage{"qa", qa, false}, qkvStage{"q_rank_norm", qr, false})
	rankStages, gra := qkvQuantStage(t, "rank_activation", qr)
	stages = append(stages, rankStages...)
	if err := cpuDense(qb, qr, l.qB); err != nil {
		t.Fatal(err)
	}
	copy(qhead, qb)
	for h := 0; h < 64; h++ {
		row := qhead[h*512 : (h+1)*512]
		if err := RMSNorm(row, row, nil, 1e-6); err != nil {
			t.Fatal(err)
		}
	}
	if err := cpuDense(kv, gnorm, l.kv); err != nil {
		t.Fatal(err)
	}
	if err := RMSNorm(kvnorm, kv, l.kvNorm, 1e-6); err != nil {
		t.Fatal(err)
	}
	stages = append(stages, qkvStage{"qb", qb, false}, qkvStage{"q_head_norm", qhead, false}, qkvStage{"kv_raw", kv, false}, qkvStage{"kv_norm", kvnorm, false})
	for _, pos := range []int{0, 1, 64} {
		q, k := append([]float32(nil), qhead...), append([]float32(nil), kvnorm...)
		if err := flashRoPE(0).Rotate(q, 512, 64, pos, false); err != nil {
			t.Fatal(err)
		}
		if err := flashRoPE(0).Rotate(k, 512, 64, pos, false); err != nil {
			t.Fatal(err)
		}
		stages = append(stages, qkvStage{fmt.Sprintf("q_rope_%d", pos), q, false}, qkvStage{fmt.Sprintf("kv_rope_%d", pos), k, false})
	}
	matchedQuant, _ := qkvQuantStage(t, "matched_activation", gnorm)
	stages = append(stages, matchedQuant...)
	stages = append(stages, qkvStage{"matched_qa_dot", qa, true}, qkvStage{"matched_qb_dot", qb, true}, qkvStage{"matched_kv_dot", kv, true})
	// Identical pre-normalization inputs isolate RMS and rotary from upstream dots.
	stages = append(stages, qkvStage{"matched_rank_norm", qr, false}, qkvStage{"matched_head_norm", qhead, false}, qkvStage{"matched_kv_norm", kvnorm, false})
	for _, pos := range []int{0, 1, 64} {
		q, k := append([]float32(nil), qhead...), append([]float32(nil), kvnorm...)
		if err := flashRoPE(0).Rotate(q, 512, 64, pos, false); err != nil {
			t.Fatal(err)
		}
		if err := flashRoPE(0).Rotate(k, 512, 64, pos, false); err != nil {
			t.Fatal(err)
		}
		stages = append(stages, qkvStage{fmt.Sprintf("matched_q_rope_%d", pos), q, false}, qkvStage{fmt.Sprintf("matched_kv_rope_%d", pos), k, false})
	}
	var input bytes.Buffer
	for _, m := range []*Matrix{l.qA, l.qB, l.kv} {
		row := make([]byte, m.rows*m.rowBytes)
		n, e := m.source.ReadAt(row, m.offset)
		if e != nil || n != len(row) {
			t.Fatalf("read actual Q8 tensor: %d %v", n, e)
		}
		input.Write(row)
	}
	for _, x := range [][]float32{l.qNorm, l.kvNorm, cnorm, gnorm, qa, qb, kv, qhead, kvnorm} {
		if err := binary.Write(&input, binary.LittleEndian, x); err != nil {
			t.Fatal(err)
		}
	}
	for _, a := range [][]Q80Activation{ga, gra} {
		for _, v := range a {
			if err := binary.Write(&input, binary.LittleEndian, v.Values); err != nil {
				t.Fatal(err)
			}
		}
		for _, v := range a {
			if err := binary.Write(&input, binary.LittleEndian, v.Scale); err != nil {
				t.Fatal(err)
			}
		}
	}
	inputBytes := append([]byte(nil), input.Bytes()...)
	dir := t.TempDir()
	file, bin := filepath.Join(dir, "qkv.c"), filepath.Join(dir, "qkv")
	if err := os.WriteFile(file, []byte(c), 0600); err != nil {
		t.Fatal(err)
	}
	for suffix, data := range map[string][]byte{".c": []byte(c), ".input": input.Bytes()} {
		if err := writeCPUForwardOutput(output+suffix, data); err != nil {
			t.Fatal(err)
		}
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
		t.Fatalf("compile QKV: %v %s", compileErr, compileOutput)
	}
	binaryBytes, err := os.ReadFile(bin)
	if err != nil {
		t.Fatal(err)
	}
	if err := writeCPUForwardOutput(output+".oracle", binaryBytes); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(bin)
	cmd.Stdin = &input
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	want, runErr := cmd.Output()
	var got bytes.Buffer
	count := 0
	for _, s := range stages {
		count += len(s.values)
		if err := binary.Write(&got, binary.LittleEndian, s.values); err != nil {
			t.Fatal(err)
		}
	}
	for suffix, data := range map[string][]byte{".C.f32": want, ".Go.f32": got.Bytes(), ".stderr": stderr.Bytes()} {
		if err := writeCPUForwardOutput(output+suffix, data); err != nil {
			t.Fatal(err)
		}
	}
	if runErr != nil || len(want) != count*4 {
		t.Fatalf("QKV oracle: %v bytes=%d want=%d stderr=%s", runErr, len(want), count*4, stderr.Bytes())
	}
	t.Logf("token=128000 layer=0 input_sha256=%x source_sha256=%x flags=%q qualification=NO_VERDICT_control_out_of_bound", sha256.Sum256(inputBytes), sha256.Sum256([]byte(c)), flags[:9])
	offset := 0
	first := "none"
	for _, s := range stages {
		changed, violations := 0, 0
		max, ss, energy := 0.0, 0.0, 0.0
		for i, v := range s.values {
			w := math.Float32frombits(binary.LittleEndian.Uint32(want[(offset+i)*4:]))
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.IsNaN(float64(w)) || math.IsInf(float64(w), 0) {
				t.Fatalf("nonfinite %s %d", s.name, i)
			}
			d := math.Abs(float64(v) - float64(w))
			if math.Float32bits(v) != math.Float32bits(w) {
				changed++
			}
			if d > 1e-4+1e-4*math.Abs(float64(w)) {
				violations++
			}
			max = math.Max(max, d)
			ss += d * d
			energy += float64(w) * float64(w)
		}
		t.Logf("stage=%s count=%d changed=%d violations=%d max_abs=%.9g rms=%.9g reference_rms=%.9g", s.name, len(s.values), changed, violations, max, math.Sqrt(ss/float64(len(s.values))), math.Sqrt(energy/float64(len(s.values))))
		if violations != 0 && first == "none" {
			first = s.name
		}
		if s.exact && changed != 0 {
			t.Errorf("same-input stage %s differs", s.name)
		}
		offset += len(s.values)
	}
	t.Logf("first_local_F5_exceeding_stage=%s", first)
	if first != "none" {
		t.Error("actual QKV numerical bound exceeded")
	}
}

func TestActualLayerZeroQKVSourceAdmission(t *testing.T) {
	c := actualLayerZeroQKVSource(t)
	dir := t.TempDir()
	file := filepath.Join(dir, "qkv.c")
	if err := os.WriteFile(file, []byte(c), 0600); err != nil {
		t.Fatal(err)
	}
	arch := "-march=native"
	if runtime.GOOS == "darwin" {
		arch = "-mcpu=native"
	}
	out, err := exec.Command("cc", "-O3", "-ffast-math", "-fno-finite-math-only", arch, file, "-lm", "-o", filepath.Join(dir, "qkv")).CombinedOutput()
	if err != nil {
		t.Fatalf("QKV source admission: %v %s", err, out)
	}
	// A zero-valued packet admits framing/dependencies only, not arithmetic.
	const inputBytes = 1024*128*34 + 32768*32*34 + 512*128*34 + 77312*4 + 4096 + 128*4 + 1024 + 32*4
	cmd := exec.Command(filepath.Join(dir, "qkv"))
	cmd.Stdin = bytes.NewReader(make([]byte, inputBytes))
	raw, err := cmd.Output()
	if err != nil || len(raw) != 346400*4 {
		t.Fatalf("QKV frame admission: %v bytes=%d", err, len(raw))
	}
	for i := 0; i < len(raw); i += 4 {
		v := math.Float32frombits(binary.LittleEndian.Uint32(raw[i:]))
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("nonfinite zero frame %d", i/4)
		}
	}
	t.Logf("QKV source_sha256=%x; no model binding or forward", sha256.Sum256([]byte(c)))
}

func actualLayerZeroQKVSource(t *testing.T) string {
	t.Helper()
	s := string(pinnedCPUOracle(t))
	c := `#include <arm_neon.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
`
	for _, name := range []string{"f16_to_f32", "dot_i8_32", "dot_q8_0_row", "quantize_q8_0_activation", "rms_norm_weight", "head_rms_norm_inplace", "rope_yarn_ramp", "rope_yarn_corr_dim", "rope_yarn_corr_dims", "rope_tail_ext_inplace"} {
		c += oracleFunction(t, s, name) + "\n"
	}
	c += `static void project(float*out,const uint8_t*w,int rows,int cols,const int8_t*q,const float*scale){for(int r=0;r<rows;r++)out[r]=dot_q8_0_row(w+(size_t)r*(cols/32)*34,q,scale,cols,cols/32);}
static void emit(const float*x,int n){if(fwrite(x,4,n,stdout)!=(size_t)n)exit(2);}
static void emit_quant(const int8_t*q,const float*sc,int n){for(int i=0;i<n;i++){float v=q[i];emit(&v,1);}emit(sc,n/32);}
static void rotate(float*x,int heads,int pos){rope_tail_ext_inplace(x,heads,512,64,pos,0,10000.0f,1.0f,0.0f,1.0f,32.0f,1.0f,false);}
int main(void){size_t asz=1024*128*34,bsz=32768*32*34,ksz=512*128*34;uint8_t*a=malloc(asz),*b=malloc(bsz),*k=malloc(ksz);
float nw[1024],kn[512],cn[4096],gn[4096],gqa[1024],gqb[32768],gkv[512],gqh[32768],gkvn[512],qa[1024],qr[1024],qb[32768],qh[32768],kv[512],kvn[512],tmpq[32768],tmpk[512];int8_t q[4096],rq[1024],gq[4096],grq[1024];float sc[128],rsc[32],gsc[128],grsc[32];
if(!a||!b||!k||fread(a,1,asz,stdin)!=asz||fread(b,1,bsz,stdin)!=bsz||fread(k,1,ksz,stdin)!=ksz||fread(nw,4,1024,stdin)!=1024||fread(kn,4,512,stdin)!=512||fread(cn,4,4096,stdin)!=4096||fread(gn,4,4096,stdin)!=4096||fread(gqa,4,1024,stdin)!=1024||fread(gqb,4,32768,stdin)!=32768||fread(gkv,4,512,stdin)!=512||fread(gqh,4,32768,stdin)!=32768||fread(gkvn,4,512,stdin)!=512||fread(gq,1,4096,stdin)!=4096||fread(gsc,4,128,stdin)!=128||fread(grq,1,1024,stdin)!=1024||fread(grsc,4,32,stdin)!=32)return 2;
quantize_q8_0_activation(cn,q,sc,4096);emit_quant(q,sc,4096);project(qa,a,1024,4096,q,sc);rms_norm_weight(qr,qa,nw,1024,1e-6f);emit(qa,1024);emit(qr,1024);
quantize_q8_0_activation(qr,rq,rsc,1024);emit_quant(rq,rsc,1024);project(qb,b,32768,1024,rq,rsc);memcpy(qh,qb,sizeof(qb));head_rms_norm_inplace(qh,64,512,1e-6f);project(kv,k,512,4096,q,sc);rms_norm_weight(kvn,kv,kn,512,1e-6f);emit(qb,32768);emit(qh,32768);emit(kv,512);emit(kvn,512);
int positions[3]={0,1,64};for(int i=0;i<3;i++){memcpy(tmpq,qh,sizeof(qh));memcpy(tmpk,kvn,sizeof(kvn));rotate(tmpq,64,positions[i]);rotate(tmpk,1,positions[i]);emit(tmpq,32768);emit(tmpk,512);}
quantize_q8_0_activation(gn,q,sc,4096);emit_quant(q,sc,4096);
project(qa,a,1024,4096,gq,gsc);emit(qa,1024);project(qb,b,32768,1024,grq,grsc);emit(qb,32768);project(kv,k,512,4096,gq,gsc);emit(kv,512);
rms_norm_weight(qr,gqa,nw,1024,1e-6f);emit(qr,1024);memcpy(qh,gqb,sizeof(qh));head_rms_norm_inplace(qh,64,512,1e-6f);emit(qh,32768);rms_norm_weight(kvn,gkv,kn,512,1e-6f);emit(kvn,512);
for(int i=0;i<3;i++){memcpy(tmpq,gqh,sizeof(gqh));memcpy(tmpk,gkvn,sizeof(gkvn));rotate(tmpq,64,positions[i]);rotate(tmpk,1,positions[i]);emit(tmpq,32768);emit(tmpk,512);}free(a);free(b);free(k);return 0;}`
	return c
}
