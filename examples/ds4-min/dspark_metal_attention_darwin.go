//go:build darwin && arm64

package main

import (
	"fmt"

	applemetal "github.com/tmc/apple/metal"
	"github.com/tmc/apple/objectivec"
	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

// A bounded noncausal support route. Its private cache is not target rawKV.
// Query/row half conversion is explicit; RoPE is dense base10000, never the
// target compressed YaRN route. Numerical parity remains separately gated.
const metalDSparkAttentionMSL = `
#include <metal_stdlib>
using namespace metal;
inline float dspark_head_weight(device const uchar *data,uint index,uint type) {
 if(type==0)return reinterpret_cast<device const float *>(data)[index];
 if(type==1)return float(reinterpret_cast<device const half *>(data)[index]);
 uint offset=(index/32)*34;return float(*reinterpret_cast<device const half *>(data+offset))*float(reinterpret_cast<device const char *>(data+offset+2)[index%32]);
}
inline float dspark_head_input(device const float *hidden,device const uchar *w1,uint previous,uint rank,uint w1type,uint hiddenCount,uint index) {
 return index<hiddenCount?hidden[index]:dspark_head_weight(w1,previous*rank+index-hiddenCount,w1type);
}
inline float dspark_head_dot(device const uchar *weights,uint type,uint first,uint count,device const float *hidden,device const uchar *w1,uint previous,uint rank,uint w1type,uint hiddenCount) {
 float sum=0.0f;
 if(type!=8){for(uint i=0;i<count;i++)sum+=dspark_head_weight(weights,first+i,type)*dspark_head_input(hidden,w1,previous,rank,w1type,hiddenCount,i);return sum;}
 // Apple Spec06 Markov/confidence use CPU matvec_any, whose Q8 route
 // quantizes activations per32 before the integer dot (C 11e8cad:8359).
 for(uint base=0;base<count;base+=32){
  float maximum=0.0f;for(uint i=0;i<32;i++){float value=dspark_head_input(hidden,w1,previous,rank,w1type,hiddenCount,base+i);if(!isfinite(value))return NAN;maximum=max(maximum,abs(value));}
  float scale=maximum/127.0f,inverse=scale!=0.0f?1.0f/scale:0.0f;int dot=0;
  uint offset=((first+base)/32)*34;
  for(uint i=0;i<32;i++){int value=clamp(int(rint(dspark_head_input(hidden,w1,previous,rank,w1type,hiddenCount,base+i)*inverse)),-128,127);dot+=int(reinterpret_cast<device const char *>(weights+offset+2)[i])*value;}
  sum+=float(*reinterpret_cast<device const half *>(weights+offset))*scale*float(dot);
 }
 return sum;
}
kernel void dspark_proposal_head(device const float *logits [[buffer(0)]],device const float *hidden [[buffer(1)]],device const uchar *w1 [[buffer(2)]],device const uchar *w2 [[buffer(3)]],device const uchar *confidence [[buffer(4)]],constant uint *args [[buffer(5)]],constant float &threshold [[buffer(6)]],device int *ids [[buffer(7)]],device float *probabilities [[buffer(8)]],device uint *result [[buffer(9)]],device float *confidence_logits [[buffer(10)]],uint tid [[thread_position_in_grid]]) {
 if(tid!=0)return;uint rows=args[0],vocab=args[1],rank=args[2],previous=args[3];result[0]=0;result[1]=0;
 for(uint row=0;row<rows;row++){
  float z=dspark_head_dot(confidence,args[6],0,4096+rank,hidden+row*4096,w1,previous,rank,args[4],4096);
  confidence_logits[row]=z;
  if(!isfinite(z)){result[1]=1;return;}float probability=1.0f/(1.0f+exp(-z));probabilities[row]=probability;
  if(probability<threshold)return;
  uint best=0;float bestValue=-INFINITY;
  for(uint token=0;token<vocab;token++){
   float bias=dspark_head_dot(w2,args[5],token*rank,rank,hidden,w1,previous,rank,args[4],0);
   float value=logits[row*vocab+token]+bias;
   if(!isfinite(value)){result[1]=1;return;}if(value>bestValue){best=token;bestValue=value;}
  }
  ids[row]=int(best);result[0]++;previous=best;
 }
}
kernel void dspark_replace_drafts(device const float *drafts [[buffer(0)]],device float *state [[buffer(1)]],constant uint &count [[buffer(2)]],uint index [[thread_position_in_grid]]) {
 if(index<count*16384)state[16384+index]=drafts[index];
}
kernel void dspark_expand_hc(device const float *main [[buffer(0)]],device const float *embeddings [[buffer(1)]],device float *state [[buffer(2)]],constant uint &drafts [[buffer(3)]],uint index [[thread_position_in_grid]]) {
 if(index>=(drafts+1)*16384)return;uint row=index/16384,dim=index%4096;state[index]=row==0?main[dim]:embeddings[(row-1)*4096+dim];
}
kernel void dspark_final_hc(device const float *state [[buffer(0)]],device const float *mix [[buffer(1)]],device const float *scale [[buffer(2)]],device const float *base [[buffer(3)]],device float *output [[buffer(4)]],constant uint &rows [[buffer(5)]],uint index [[thread_position_in_grid]]) {
 if(index>=rows*4096)return;uint row=index/4096,dim=index%4096;float value=0;for(uint lane=0;lane<4;lane++){float w=1.0f/(1.0f+exp(-(mix[row*4+lane]*scale[0]+base[lane])))+1e-6f;value+=w*state[row*16384+lane*4096+dim];}output[index]=value;
}
inline float dspark_fp8_value(uint code) {
 uint exponent=(code>>3)&15,mantissa=code&7;
 return exponent==0?float(mantissa)*0.001953125f:(1.0f+float(mantissa)*.125f)*exp2(float(exponent)-7.0f);
}
inline float dspark_fp8_round(float value) {
 float magnitude=min(abs(value),448.0f);uint lo=0,hi=126;
 while(lo<hi){uint mid=(lo+hi+1)>>1;if(dspark_fp8_value(mid)<=magnitude)lo=mid;else hi=mid-1;}
 if(lo<126){float a=abs(magnitude-dspark_fp8_value(lo)),b=abs(magnitude-dspark_fp8_value(lo+1));if(b<a||(b==a&&(lo&1)))lo++;}
 return (value<0?-1.0f:1.0f)*dspark_fp8_value(lo);
}
kernel void dspark_store_raw(device float *kv [[buffer(0)]],device float *raw [[buffer(1)]],
 constant uint &capacity [[buffer(2)]],constant uint &start [[buffer(3)]],constant uint &rows [[buffer(4)]],uint row [[thread_position_in_grid]]) {
 if(row>=rows)return;device float *x=kv+row*512;
 for(uint offset=0;offset<448;offset+=64){float maximum=1e-4f;for(uint i=0;i<64;i++)maximum=max(maximum,abs(x[offset+i]));float scale=exp2(ceil(log2(maximum/448.0f)));for(uint i=0;i<64;i++)x[offset+i]=dspark_fp8_round(x[offset+i]/scale)*scale;}
 for(uint i=0;i<512;i++)raw[((start+row)%capacity)*512+i]=float(half(x[i]));
}
kernel void dspark_rope_tail(device float *values [[buffer(0)]],constant uint &heads [[buffer(1)]],
 constant uint &position [[buffer(2)]],constant uint &inverse [[buffer(3)]],constant uint &rows [[buffer(4)]],uint index [[thread_position_in_grid]]) {
 if(index>=heads*rows*32)return;uint pair=index%32,hrow=index/32,row=hrow/heads,d=448+pair*2;
 float theta=float(position+row)*pow(10000.0f,-float(pair*2)/64.0f),c=cos(theta),s=sin(theta);if(inverse)s=-s;
 uint offset=hrow*512+d;float a=values[offset],b=values[offset+1];values[offset]=a*c-b*s;values[offset+1]=a*s+b*c;
}
kernel void dspark_noncausal(device const float *query [[buffer(0)]],device const float *raw [[buffer(1)]],
 device const float *sinks [[buffer(2)]],device float *output [[buffer(3)]],constant uint &capacity [[buffer(4)]],
 constant uint &start [[buffer(5)]],constant uint &visible [[buffer(6)]],constant uint &rows [[buffer(7)]],uint hrow [[thread_position_in_grid]]) {
 if(hrow>=rows*64)return;uint head=hrow%64;float scores[145],maximum=sinks[head];
 for(uint r=0;r<visible;r++){float score=0;uint key=((start+r)%capacity)*512;for(uint d=0;d<512;d++)score+=float(half(query[hrow*512+d]))*raw[key+d];scores[r]=score*rsqrt(512.0f);maximum=max(maximum,scores[r]);}
 float denominator=exp(sinks[head]-maximum);for(uint r=0;r<visible;r++){scores[r]=exp(scores[r]-maximum);denominator+=scores[r];}
 for(uint d=0;d<512;d++){float value=0;for(uint r=0;r<visible;r++)value+=scores[r]*raw[((start+r)%capacity)*512+d];output[hrow*512+d]=value/denominator;}
}
`

type metalDSparkAttention struct {
	weights                  *metalDSparkWeights
	library                  applemetal.MTLLibrary
	rope, store, attention   applemetal.MTLComputePipelineState
	expand, finalHC, replace applemetal.MTLComputePipelineState
	head                     applemetal.MTLComputePipelineState
	batches                  []*metalBatch
	closed                   bool
}

func newMetalDSparkAttention(dev *libmetal.Device, w *metalDSparkWeights) (a *metalDSparkAttention, err error) {
	if dev == nil || w == nil || w.device != dev {
		return nil, fmt.Errorf("metal dspark attention: owned device required")
	}
	if err := w.acquire(); err != nil {
		return nil, err
	}
	a = &metalDSparkAttention{weights: w}
	owner := a
	defer func() {
		if err != nil {
			err = finishBackendCleanup(err, owner)
		}
	}()
	a.library, err = dev.CompileLibrary(metalDSparkAttentionMSL)
	if err != nil {
		return nil, err
	}
	for _, v := range []struct {
		name string
		dst  *applemetal.MTLComputePipelineState
	}{{"dspark_rope_tail", &a.rope}, {"dspark_store_raw", &a.store}, {"dspark_noncausal", &a.attention}, {"dspark_expand_hc", &a.expand}, {"dspark_final_hc", &a.finalHC}, {"dspark_replace_drafts", &a.replace}, {"dspark_proposal_head", &a.head}} {
		*v.dst, err = dev.NewComputePipeline(a.library, v.name)
		if err != nil {
			return nil, err
		}
	}
	return a, nil
}

func (a *metalDSparkAttention) ready(b *metalBatch) error {
	if a == nil || a.closed || a.weights == nil || a.weights.model == nil || b == nil || b.ops == nil || b.ops.dev != a.weights.device {
		return fmt.Errorf("metal dspark attention: foreign command owner")
	}
	return b.ready()
}

func (a *metalDSparkAttention) EncodeRoPE(b *metalBatch, values metalmem.Slice[float32], heads, rows, position int, inverse bool) error {
	if err := a.ready(b); err != nil {
		return err
	}
	if (heads != 1 && heads != 64) || rows < 1 || rows > 17 || position < 0 || position > int(^uint32(0))-rows || values.Len() != heads*rows*512 {
		return fmt.Errorf("metal dspark rope: invalid shape")
	}
	enc := b.encoder()
	enc.SetComputePipelineState(a.rope)
	enc.SetBufferWithOffsetAtIndex(values.Buffer(), values.Offset(), 0)
	libmetal.SetScalar(enc, uint32(heads), 1)
	libmetal.SetScalar(enc, uint32(position), 2)
	libmetal.SetScalar(enc, boolScalar(inverse), 3)
	libmetal.SetScalar(enc, uint32(rows), 4)
	enc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(heads*rows*32), libmetal.Threads(256))
	b.encoded++
	a.batches = retainMetalDSparkBatch(a.batches, b)
	return nil
}

func (a *metalDSparkAttention) EncodeStore(b *metalBatch, kv, raw metalmem.Slice[float32], capacity, start, rows int) error {
	if err := a.ready(b); err != nil {
		return err
	}
	if capacity < 1 || capacity > 145 || start < 0 || start >= capacity || rows < 1 || rows > 17 || rows > capacity || kv.Len() != rows*512 || raw.Len() != capacity*512 {
		return fmt.Errorf("metal dspark cache: invalid extent")
	}
	enc := b.encoder()
	enc.SetComputePipelineState(a.store)
	enc.SetBufferWithOffsetAtIndex(kv.Buffer(), kv.Offset(), 0)
	enc.SetBufferWithOffsetAtIndex(raw.Buffer(), raw.Offset(), 1)
	libmetal.SetScalar(enc, uint32(capacity), 2)
	libmetal.SetScalar(enc, uint32(start), 3)
	libmetal.SetScalar(enc, uint32(rows), 4)
	enc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(rows), libmetal.Threads(1))
	b.encoded++
	a.batches = retainMetalDSparkBatch(a.batches, b)
	return nil
}

func (a *metalDSparkAttention) EncodeAttention(b *metalBatch, query, raw, output metalmem.Slice[float32], stage, capacity, start, visible, rows int) error {
	if err := a.ready(b); err != nil {
		return err
	}
	if capacity < 1 || capacity > 145 || start < 0 || start >= capacity || visible < 1 || visible > capacity || rows < 1 || rows > 16 || query.Len() != rows*32768 || output.Len() != query.Len() || raw.Len() != capacity*512 {
		return fmt.Errorf("metal dspark attention: invalid visible window")
	}
	sinks, err := a.weights.tensor(fmt.Sprintf("mtp.%d.attn_sinks.weight", stage))
	if err != nil {
		return err
	}
	if sinks.typeID != 0 || sinks.data.Len() != 256 {
		return fmt.Errorf("metal dspark attention: invalid sinks")
	}
	enc := b.encoder()
	enc.SetComputePipelineState(a.attention)
	for i, v := range []metalmem.Slice[float32]{query, raw, output} {
		index := uint(i)
		if i == 2 {
			index = 3
		}
		enc.SetBufferWithOffsetAtIndex(v.Buffer(), v.Offset(), index)
	}
	enc.SetBufferWithOffsetAtIndex(sinks.data.Buffer(), sinks.data.Offset(), 2)
	libmetal.SetScalar(enc, uint32(capacity), 4)
	libmetal.SetScalar(enc, uint32(start), 5)
	libmetal.SetScalar(enc, uint32(visible), 6)
	libmetal.SetScalar(enc, uint32(rows), 7)
	enc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(rows*64), libmetal.Threads(64))
	b.encoded++
	a.batches = retainMetalDSparkBatch(a.batches, b)
	return nil
}

func (a *metalDSparkAttention) Close() error {
	if a == nil || a.closed {
		return nil
	}
	for _, b := range a.batches {
		if !b.done || !metalGroupedTerminal(b) {
			return fmt.Errorf("metal dspark attention: live consumer retains owners")
		}
	}
	if err := a.weights.release(); err != nil {
		return err
	}
	for _, p := range []applemetal.MTLComputePipelineState{a.rope, a.store, a.attention, a.expand, a.finalHC, a.replace, a.head} {
		if p != nil && p.GetID() != 0 {
			objectivec.ObjectFromID(p.GetID()).Release()
		}
	}
	if a.library != nil && a.library.GetID() != 0 {
		objectivec.ObjectFromID(a.library.GetID()).Release()
	}
	a.closed = true
	return nil
}
