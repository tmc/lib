package dsv41

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"os/exec"
	"strings"
	"testing"
)

// TestPayloadPinnedC executes the unchanged source-owned save function and
// state-span walker with synthetic GPU tensor storage backed by host arrays.
// It qualifies section order/extent/value transport, not device synchronization
// or continuation of a real model checkpoint.
func TestPayloadPinnedC(t *testing.T) {
	repo := os.Getenv("DS41_ORACLE")
	if repo == "" {
		t.Skip("set DS41_ORACLE for literal source control")
	}
	b, err := exec.Command("git", "-C", repo, "show", "11e8cad4473716a41baf4d62e92ed9764a00fce6:ds4.c").Output()
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprintf("%x", sha256.Sum256(b)) != "62e458f2c416148c0882f7247c7b16f9820054b0c2ae41374a9209541fe7a4c7" {
		t.Fatal("payload source hash mismatch")
	}
	source := string(b)
	f := literalFunction(t, source, "ds41_state_spans") + literalFunction(t, source, "ds41_payload_body_bytes") + literalFunction(t, source, "ds41_save_payload") + literalFunction(t, source, "ds41_load_payload")
	const prefix = `
#include <stdlib.h>
#define DS4_N_VOCAB 129280
#define DS4_SESSION_PAYLOAD_MAGIC UINT32_C(0x34565344)
#define DS4_SESSION_PAYLOAD_VERSION 2u
#define DS4_SESSION_PAYLOAD_U32_FIELDS 13u
#define DS4_SESSION_IO_CHUNK 4096
struct ds4_gpu_tensor {float *data;};
struct ds4_tokens {int len;int *v;};
struct ds41_gpu_graph {uint32_t ctx,pos;bool valid;struct {int32_t tail[3];}history;uint32_t *token_map;ds4_gpu_tensor *window[40],*compressed[4],*index_cache[4],*previous_kv[4],*previous_score[4];};
struct ds4_session {ds41_gpu_graph ds41_graph;bool ds41_graph_ready,checkpoint_valid;ds4_tokens checkpoint;float *logits;};
struct ds41_state_span {ds4_gpu_tensor *tensor;uint64_t bytes;};
static bool ds4_gpu_synchronize(){return true;}
static void payload_set_err(char*,size_t,const char*){}
static uint8_t*xmalloc(size_t n){uint8_t*p=(uint8_t*)malloc(n?n:1);if(!p)exit(4);return p;}
static void token_vec_push(ds4_tokens*t,int x){t->v=(int*)realloc(t->v,(t->len+1)*4);t->v[t->len++]=x;}
static void ds4_tokens_copy(ds4_tokens*d,ds4_tokens*s){free(d->v);d->v=(int*)xmalloc(s->len*4);memcpy(d->v,s->v,s->len*4);d->len=s->len;}
static void ds4_tokens_free(ds4_tokens*t){free(t->v);}
static void ds41_graph_reset(ds41_gpu_graph*g){for(int i=0;i<3;i++)g->history.tail[i]=-1;g->pos=0;g->valid=true;}
static int payload_read_bytes(FILE*f,void*p,uint64_t n,uint64_t*remaining,char*,size_t){if(n>*remaining||fread(p,1,n,f)!=n)return 1;*remaining-=n;return 0;}
static int payload_read_u32(FILE*f,uint32_t*p,uint64_t*r,char*e,size_t n){return payload_read_bytes(f,p,4,r,e,n);}
static int payload_read_tensor_span(FILE*f,ds4_gpu_tensor*t,uint64_t off,uint64_t n,uint8_t*,uint32_t,uint64_t*r,char*e,size_t z){return payload_read_bytes(f,(char*)t->data+off,n,r,e,z);}
static int payload_write_u32(FILE*f,uint32_t x,char*,size_t){return fwrite(&x,4,1,f)!=1;}
static int payload_write_bytes(FILE*f,const void*p,uint64_t n,char*,size_t){return fwrite(p,1,n,f)!=n;}
static int payload_write_tensor_span(FILE*f,ds4_gpu_tensor*t,uint64_t off,uint64_t n,uint8_t*,uint32_t,char*,size_t){return fwrite((char*)t->data+off,1,n,f)!=n;}
static ds4_gpu_tensor*tensor(unsigned n,unsigned tag){ds4_gpu_tensor*t=(ds4_gpu_tensor*)xmalloc(sizeof(*t));t->data=(float*)xmalloc(n*4);for(unsigned i=0;i<n;i++)t->data[i]=(float)tag+(float)(i%31)/32;return t;}
`
	for _, pos := range []int{1, 128, 129} {
		t.Run(fmt.Sprint(pos), func(t *testing.T) {
			s := payloadFixture(t, pos)
			var wire bytes.Buffer
			if err := s.WritePayload(&wire); err != nil {
				t.Fatal(err)
			}
			want := make([]float32, wire.Len()/4)
			for i := range want {
				want[i] = math.Float32frombits(binary.LittleEndian.Uint32(wire.Bytes()[i*4:]))
			}
			main := fmt.Sprintf(`int main(){ds4_session s={};ds41_gpu_graph*g=&s.ds41_graph;g->ctx=132;g->pos=%d;g->valid=true;s.ds41_graph_ready=true;s.checkpoint.len=g->pos;s.checkpoint.v=(int*)xmalloc(g->pos*4);for(unsigned i=0;i<g->pos;i++)s.checkpoint.v[i]=i%%129280;s.logits=tensor(129280,99)->data;for(unsigned i=0;i<40;i++)g->window[i]=tensor(128*512,i);for(unsigned i=0;i<4;i++){unsigned rows=g->pos/(i<3?2:1);g->compressed[i]=tensor(rows*512,40+i*4);g->index_cache[i]=tensor(rows*128,41+i*4);g->previous_kv[i]=tensor(512,42+i*4);g->previous_score[i]=tensor(512,43+i*4);}g->token_map=(uint32_t*)xmalloc(129280*4);for(unsigned i=0;i<129280;i++)g->token_map[i]=i*3%%99092;char err[64];FILE*tmp=tmpfile();if(!tmp)return 5;if(ds41_save_payload(&s,tmp,err,sizeof(err)))return 6;uint64_t n=ftell(tmp);rewind(tmp);uint32_t h[13];if(fread(h,4,13,tmp)!=13)return 7;if(ds41_load_payload(&s,tmp,h,n-52,err,sizeof(err)))return 8;for(unsigned i=0;i<3;i++){int32_t want=i<g->pos?(int32_t)(((g->pos-1-i)%%129280)*3%%99092):-1;if(g->history.tail[i]!=want)return 9;}fclose(tmp);return ds41_save_payload(&s,stdout,err,sizeof(err));}`, pos)
			if !equalBits(runLiteral(t, operationPrefix+prefix+f+main, nil), want) {
				t.Fatal("literal C payload differs")
			}
			historyMain := strings.Replace(main, "if(g->history.tail[i]!=want)return 9;", "(void)want;", 1)
			historyMain = strings.Replace(historyMain, "return ds41_save_payload(&s,stdout,err,sizeof(err));", "return fwrite(g->history.tail,4,3,stdout)!=3;", 1)
			historyWant := make([]float32, 3)
			for i := range historyWant {
				v := int32(-1)
				if i < pos {
					v = int32((pos - 1 - i) * 3 % 99092)
				}
				historyWant[i] = math.Float32frombits(uint32(v))
			}
			if !equalBits(runLiteral(t, operationPrefix+prefix+f+historyMain, nil), historyWant) {
				t.Fatal("literal restored history differs")
			}
			if pos > 1 {
				historyMutant := strings.Replace(f, "pos - 1u - i", "i", 1)
				if historyMutant == f || equalBits(runLiteral(t, operationPrefix+prefix+historyMutant+historyMain, nil), historyWant) {
					t.Fatal("history order mutant not detected")
				}
			}
			mutant := strings.Replace(f, "g->index_cache[i]", "g->compressed[i]", 1)
			if mutant == f {
				t.Fatal("index section mutant missing")
			}
			// At pos1 the ratio-two owners have no index rows, but owner3 does.
			if equalBits(runLiteral(t, operationPrefix+prefix+mutant+main, nil), want) {
				t.Fatal("index section mutant not detected")
			}
		})
	}
}
