//go:build darwin && arm64

package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// This compiles the actual scalar MSL head body as C++, without creating a
// Metal device. The independent activation quantizer/dot is copied verbatim
// from C 11e8cad; the scalar fallback is selected explicitly. Device reduction
// order and native compilation remain separate gates.
func TestMetalDSparkHeadQ8PinnedC(t *testing.T) {
	pinned, err := os.ReadFile("testdata/dspark_q8_head_11e8cad.inc")
	if err != nil {
		t.Fatal(err)
	}
	start := strings.Index(metalDSparkAttentionMSL, "inline float dspark_head_weight")
	end := strings.Index(metalDSparkAttentionMSL, "kernel void dspark_replace_drafts")
	if start < 0 || end <= start {
		t.Fatal("head extraction boundary missing")
	}
	head := metalDSparkAttentionMSL[start:end]
	for _, change := range [][2]string{{"device ", ""}, {"constant ", ""}, {"kernel void ", "void "}} {
		head = strings.ReplaceAll(head, change[0], change[1])
	}
	head = regexp.MustCompile(`\[\[[^\]]+\]\]`).ReplaceAllString(head, "")
	for _, arm := range []struct{ name, from, to string }{
		{"base", "", ""},
		{"unquantized-activation", "if(type!=8){", "if(true){"},
		{"logits-before-decline", "if(probability<threshold)return;", "if(!isfinite(logits[row*vocab])){result[1]=1;return;}if(probability<threshold)return;"},
		{"sigmoid-in-raw-readout", "confidence_logits[row]=z;", "confidence_logits[row]=1.0f/(1.0f+exp(-z));"},
	} {
		t.Run(arm.name, func(t *testing.T) {
			body := head
			if arm.from != "" {
				if strings.Count(body, arm.from) != 1 {
					t.Fatal("mutation site is missing or duplicated")
				}
				body = strings.Replace(body, arm.from, arm.to, 1)
			}
			dir := t.TempDir()
			path, binary := filepath.Join(dir, "head.cpp"), filepath.Join(dir, "head")
			if err := os.WriteFile(path, []byte(metalDSparkHeadCPPPrelude+string(pinned)+body+metalDSparkHeadCPPControls), 0600); err != nil {
				t.Fatal(err)
			}
			if out, err := exec.Command("clang++", "-std=c++17", "-U__ARM_NEON", "-U__ARM_FEATURE_DOTPROD", path, "-o", binary).CombinedOutput(); err != nil {
				t.Fatalf("compile actual head: %v\n%s", err, out)
			}
			out, err := exec.Command(binary).CombinedOutput()
			t.Logf("arm=%s %s", arm.name, out)
			if (err != nil) != (arm.name != "base") {
				t.Fatalf("head control result: %v", err)
			}
		})
	}
}

const metalDSparkHeadCPPPrelude = `
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
using uint=unsigned int;using uchar=unsigned char;using half=_Float16;
using std::max;using std::abs;using std::clamp;using std::isfinite;
`

const metalDSparkHeadCPPControls = `
int main(){
 std::fesetround(FE_TONEAREST);
 float latent[64]={};latent[0]=1.0f;latent[1]=.001f;
 uchar weights[68]={};half one=1;memcpy(weights,&one,2);weights[3]=1;
 int8_t quantized[32]={};float scale[1]={};
 quantize_q8_0_activation(latent,quantized,scale,32);
 float oracle=dot_q8_0_row(weights,quantized,scale,32,1);
 float hidden[4096]={};
 float got=dspark_head_dot(weights,8,0,32,hidden,reinterpret_cast<uchar *>(latent),0,32,0,0);
 printf("q8_dot oracle=%.9g got=%.9g\n",oracle,got);
 if(oracle!=0||got!=oracle)return 11;
 float confidence[4128]={};float logits[2]={-.0005f,0};
 uint args[7]={1,2,32,0,0,8,0},result[2]={};int ids[1]={};float probs[1]={},raw[1]={};
 float threshold=.4f;
 dspark_proposal_head(logits,hidden,reinterpret_cast<uchar *>(latent),weights,reinterpret_cast<uchar *>(confidence),args,threshold,ids,probs,result,raw,0);
 printf("markov count=%u status=%u id=%d expected=1\n",result[0],result[1],ids[0]);
 if(result[0]!=1||result[1]!=0||ids[0]!=1)return 12;
 hidden[0]=1;confidence[0]=.25f;
 dspark_proposal_head(logits,hidden,reinterpret_cast<uchar *>(latent),weights,reinterpret_cast<uchar *>(confidence),args,threshold,ids,probs,result,raw,0);
 printf("raw_confidence logit=%.9g probability=%.9g expected_logit=.25\n",raw[0],probs[0]);
 if(raw[0]!=.25f||probs[0]<=.5f)return 15;
 hidden[0]=0;confidence[0]=0;
 uchar qconfidence[129*34]={};memcpy(qconfidence+128*34,&one,2);qconfidence[128*34+3]=1;
 args[6]=8;threshold=.5001f;
 dspark_proposal_head(logits,hidden,reinterpret_cast<uchar *>(latent),weights,qconfidence,args,threshold,ids,probs,result,raw,0);
 printf("confidence count=%u status=%u p=%.9g threshold=%.9g\n",result[0],result[1],probs[0],threshold);
 if(result[0]!=0||result[1]!=0||probs[0]!=.5f)return 13;
 args[6]=0;threshold=.6f;logits[0]=std::numeric_limits<float>::quiet_NaN();
 dspark_proposal_head(logits,hidden,reinterpret_cast<uchar *>(latent),weights,reinterpret_cast<uchar *>(confidence),args,threshold,ids,probs,result,raw,0);
 printf("decline_nan count=%u status=%u expected=0/0\n",result[0],result[1]);
 if(result[0]!=0||result[1]!=0)return 14;
 logits[0]=-.0005f;hidden[0]=1;confidence[0]=-100;threshold=0;
 dspark_proposal_head(logits,hidden,reinterpret_cast<uchar *>(latent),weights,reinterpret_cast<uchar *>(confidence),args,threshold,ids,probs,result,raw,0);
 printf("threshold_zero count=%u status=%u probability=%.9g raw_logit=%.9g\n",result[0],result[1],probs[0],raw[0]);
 if(result[0]!=1||result[1]!=0||probs[0]>.01f||raw[0]!=-100)return 16;
 return 0;
}
`
