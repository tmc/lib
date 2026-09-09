//go:build linux

package main

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

type cudaCompressedState struct {
	cache, stateKV, stateScore, pooled cudamem.Slice[float32]
	headDim, width, ratio, cap, n      int
}

// encodeCompressor appends an update and, at a ratio boundary, a finished row.
// The caller advances state.n after the launches succeed. Later work on the
// same stream observes the update before using the new row count.
func (r *cudaFoundation) encodeCompressor(kv, gate cudamem.Slice[float32], ape, norm cudaModelTensor, state *cudaCompressedState, position int, indexer bool) (bool, error) {
	head := 512
	if indexer {
		head = 128
	}
	if state == nil || state.headDim != head || (state.ratio != 4 && state.ratio != 128) || indexer && state.ratio != 4 || position < 0 {
		return false, fmt.Errorf("cuda: compressor: invalid shape")
	}
	width, rows := head, state.ratio
	if state.ratio == 4 {
		width *= 2
		rows *= 2
	}
	if state.width != width || kv.Len() != width || gate.Len() != width || state.stateKV.Len() != rows*width || state.stateScore.Len() != rows*width || state.pooled.Len() != head || state.cap < 1 || state.cache.Len() != state.cap*head || state.n < 0 || state.n > state.cap || ape.typeID != 1 || len(ape.dimensions) != 2 || ape.dimensions[0] != uint64(width) || ape.dimensions[1] != uint64(state.ratio) || norm.typeID != 0 || len(norm.dimensions) != 1 || norm.dimensions[0] != uint64(head) {
		return false, fmt.Errorf("cuda: compressor: invalid tensor profile")
	}
	emit := (position+1)%state.ratio == 0
	if emit && state.n == state.cap {
		return false, fmt.Errorf("cuda: compressor: cache capacity exceeded")
	}
	err := r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.compressorPoolFn).Grid(1).Block(uint32(head)).On(r.device.DefaultStream()).Launch(kv, gate, state.stateKV, state.stateScore, state.pooled, ape.device, uint32(head), uint32(width), uint32(state.ratio), uint32(position)); err != nil {
			return fmt.Errorf("cuda: launch compressor pool: %w", err)
		}
		if !emit {
			return nil
		}
		fn := r.compressorAttentionFn
		if indexer {
			fn = r.compressorIndexerFn
		}
		if err := cudalaunch.Kernel(fn).Grid(1).Block(1).On(r.device.DefaultStream()).Launch(state.pooled, norm.device, state.cache, uint32(state.n), uint32(position+1-state.ratio)); err != nil {
			return fmt.Errorf("cuda: launch compressor finish: %w", err)
		}
		return nil
	})
	return emit, err
}

// encodeMixedAttention retains the memoized single-block qualification path.
// Production uses encodeMixedAttentionParallel.
func (r *cudaFoundation) encodeMixedAttention(q, kv cudamem.Slice[float32], sinks cudaModelTensor, raw, comp cudamem.Slice[float32], nComp int, allowed cudamem.Slice[uint32], scores, out cudamem.Slice[float32], position, layer int) error {
	if q.Len() != 64*512 || kv.Len() != 512 || sinks.typeID != 0 || len(sinks.dimensions) != 1 || sinks.dimensions[0] != 64 || raw.Len() == 0 || raw.Len()%512 != 0 || comp.Len()%512 != 0 || nComp < 0 || nComp > comp.Len()/512 || allowed.Len() < max(1, nComp) || position < 0 || layer < 2 || out.Len() != 32768+512+32768 {
		return fmt.Errorf("cuda: mixed attention: invalid shape")
	}
	total := min(position+1, raw.Len()/512) + nComp
	if scores.Len() < 64*total {
		return fmt.Errorf("cuda: mixed attention: invalid score capacity")
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.mixedAttentionFn).Grid(1).Block(64).On(r.device.DefaultStream()).Launch(q, kv, sinks.device, raw, comp, allowed, scores, out, uint32(position), uint32(layer), uint32(raw.Len()/512), uint32(nComp)); err != nil {
			return fmt.Errorf("cuda: launch mixed attention: %w", err)
		}
		return nil
	})
}

// encodeMixedAttentionParallel queues preparation and pairwise value evaluation.
// The single stream orders both kernels before any reuse of token scratch.
func (r *cudaFoundation) encodeMixedAttentionParallel(q, kv cudamem.Slice[float32], sinks cudaModelTensor, raw, comp cudamem.Slice[float32], nComp int, allowed cudamem.Slice[uint32], scores, denominators, out cudamem.Slice[float32], position, layer int) error {
	if q.Len() != 64*512 || kv.Len() != 512 || sinks.typeID != 0 || len(sinks.dimensions) != 1 || sinks.dimensions[0] != 64 || raw.Len() == 0 || raw.Len()%512 != 0 || comp.Len()%512 != 0 || nComp < 0 || nComp > comp.Len()/512 || allowed.Len() < max(1, nComp) || denominators.Len() != 64 || position < 0 || layer < 2 || out.Len() != 32768+512+32768 {
		return fmt.Errorf("cuda: mixed attention: invalid shape")
	}
	total := min(position+1, raw.Len()/512) + nComp
	if scores.Len() < 64*total {
		return fmt.Errorf("cuda: mixed attention: invalid score capacity")
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.mixedAttentionPrepareFn).Grid(1).Block(64).On(r.device.DefaultStream()).Launch(q, kv, sinks.device, raw, comp, allowed, scores, denominators, out, uint32(position), uint32(layer), uint32(raw.Len()/512), uint32(nComp)); err != nil {
			return fmt.Errorf("cuda: launch mixed attention preparation: %w", err)
		}
		if err := cudalaunch.Kernel(r.mixedAttentionValuesFn).Grid(64).Block(256).On(r.device.DefaultStream()).Launch(raw, comp, allowed, scores, denominators, out, uint32(position), uint32(raw.Len()/512), uint32(nComp)); err != nil {
			return fmt.Errorf("cuda: launch mixed attention values: %w", err)
		}
		return nil
	})
}

// encodeIndexer consumes projected queries and head weights. Up to 512 rows
// need no projections; all are allowed. Above that boundary it selects exactly
// 512 rows, resolving score ties in favor of the lower cache index.
func (r *cudaFoundation) encodeIndexer(q, weights, cache cudamem.Slice[float32], nComp int, qScratch, scores cudamem.Slice[float32], allowed cudamem.Slice[uint32], position int) error {
	if nComp < 0 || nComp > cache.Len()/128 || cache.Len()%128 != 0 || allowed.Len() < max(1, nComp) || position < 0 || nComp > 512 && (q.Len() != 64*128 || weights.Len() != 64 || qScratch.Len() != 64*128 || scores.Len() < nComp) {
		return fmt.Errorf("cuda: indexer: invalid shape")
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if nComp > 512 {
			if err := cudalaunch.Kernel(r.indexerQueryFn).Grid(64).Block(1).On(r.device.DefaultStream()).Launch(q, qScratch, uint32(position)); err != nil {
				return fmt.Errorf("cuda: launch indexer query: %w", err)
			}
		}
		if err := cudalaunch.Kernel(r.indexerSelectFn).Grid(1).Block(1).On(r.device.DefaultStream()).Launch(qScratch, weights, cache, scores, allowed, uint32(nComp)); err != nil {
			return fmt.Errorf("cuda: launch indexer selection: %w", err)
		}
		return nil
	})
}

func (r *cudaFoundation) encodeFillU32(dst cudamem.Slice[uint32], count int, value uint32) error {
	if count < 0 || count > dst.Len() {
		return fmt.Errorf("cuda: fill uint32: invalid length")
	}
	if count == 0 {
		return nil
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.fillU32Fn).Grid(uint32((count+255)/256)).Block(256).On(r.device.DefaultStream()).Launch(dst, uint32(count), value); err != nil {
			return fmt.Errorf("cuda: launch fill uint32: %w", err)
		}
		return nil
	})
}

const cudaCompressedSource = `
extern "C" __global__ void ds4_fill_u32(unsigned int *dst,unsigned int count,unsigned int value) {
 const unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<count)dst[i]=value;
}
__device__ float ds4_compressed_theta(unsigned int position,unsigned int r) {
 const float pi=3.14159265358979323846f;
 const float low=fmaxf(0.0f,floorf(64.0f*logf(65536.0f/(32.0f*2*pi))/(2*logf(160000.0f))));
 const float high=fminf(63.0f,ceilf(64.0f*logf(65536.0f/(2*pi))/(2*logf(160000.0f))));
 const float y=(float(r/2)-low)/fmaxf(.001f,high-low);
 const float ramp=1-fminf(1.0f,fmaxf(0.0f,y));
 return float(position)*powf(160000.0f,-float(r)/64)*(.0625f*(1-ramp)+ramp);
}
__device__ void ds4_compressed_rope(float *row,unsigned int width,unsigned int position,bool inverse) {
 for(unsigned int r=0;r<64;r+=2) {
  const unsigned int i=width-64+r;const float theta=ds4_compressed_theta(position,r),c=cosf(theta),s=(inverse?-1.0f:1.0f)*sinf(theta),a=row[i],b=row[i+1];
  row[i]=a*c-b*s;row[i+1]=a*s+b*c;
 }
}
__device__ float ds4_e2m1_round(float value) {
 const float levels[8]={0,.5f,1,1.5f,2,3,4,6};
 const float x=fminf(fabsf(value),6.0f);int best=0;
 for(int i=1;i<8;i++) { const float d=fabsf(x-levels[i]),b=fabsf(x-levels[best]);if(d<b || (d==b && i%2==0 && best%2!=0))best=i; }
 return value<0?-levels[best]:levels[best];
}
__device__ void ds4_indexer_qat(float *x) {
 for(unsigned int stride=1;stride<128;stride*=2)
  for(unsigned int base=0;base<128;base+=2*stride)
   for(unsigned int i=0;i<stride;i++) { const float a=x[base+i],b=x[base+stride+i];x[base+i]=a+b;x[base+stride+i]=a-b; }
 for(unsigned int i=0;i<128;i++)x[i]*=.08838834764831845f;
 for(unsigned int off=0;off<128;off+=32) {
  float amax=7.052966104933725e-38f;for(unsigned int i=0;i<32;i++)amax=fmaxf(amax,fabsf(x[off+i]));
  const float scale=exp2f(ceilf(log2f(amax/6)));
  for(unsigned int i=0;i<32;i++)x[off+i]=ds4_e2m1_round(x[off+i]/scale)*scale;
 }
}
extern "C" __global__ void ds4_compressor_pool(const float *kv,const float *gate,float *state_kv,float *state_score,float *pooled,const unsigned short *ape,unsigned int head,unsigned int width,unsigned int ratio,unsigned int position) {
 const unsigned int j=threadIdx.x;if(blockIdx.x || j>=head)return;
 const unsigned int lanes=ratio==4?2:1,p=position%ratio,row=ratio==4?ratio+p:p;
 for(unsigned int lane=0;lane<lanes;lane++) { unsigned int i=lane*head+j;state_kv[row*width+i]=kv[i];state_score[row*width+i]=gate[i]+ds4_half_to_float(ape[p*width+i]); }
 if((position+1)%ratio!=0)return;
 float maximum=-__int_as_float(0x7f800000);
 for(unsigned int r=0;r<ratio;r++) {
  maximum=fmaxf(maximum,state_score[r*width+j]);
  if(ratio==4)maximum=fmaxf(maximum,state_score[(ratio+r)*width+head+j]);
 }
 float denominator=0,sum=0;
 if(isfinite(maximum))for(unsigned int r=0;r<ratio;r++) {
  const float wp=expf(state_score[r*width+j]-maximum);
  if(ratio==4) { const float wc=expf(state_score[(ratio+r)*width+head+j]-maximum);denominator+=wp+wc;sum+=wp*state_kv[r*width+j];sum+=wc*state_kv[(ratio+r)*width+head+j]; }
  else { denominator+=wp;sum+=wp*state_kv[r*width+j]; }
 }
 pooled[j]=denominator>0?sum/denominator:0;
 if(ratio==4)for(unsigned int r=0;r<ratio;r++)for(unsigned int lane=0;lane<2;lane++) {
  const unsigned int i=lane*head+j;state_kv[r*width+i]=state_kv[(ratio+r)*width+i];state_score[r*width+i]=state_score[(ratio+r)*width+i];
 }
}
extern "C" __global__ void ds4_compressor_attention(float *pooled,const float *weight,float *cache,unsigned int row,unsigned int position) {
 if(blockIdx.x || threadIdx.x)return;
 double ss=0;for(unsigned int i=0;i<512;i++)ss+=double(pooled[i])*pooled[i];
 const float inv=1.0f/sqrtf(float(ss/512)+1e-6f);for(unsigned int i=0;i<512;i++)pooled[i]=pooled[i]*inv*weight[i];
 ds4_compressed_rope(pooled,512,position,false);
 for(unsigned int off=0;off<448;off+=64) {
  float amax=1e-4f;for(unsigned int i=0;i<64;i++)amax=fmaxf(amax,fabsf(pooled[off+i]));
  const float scale=exp2f(ceilf(log2f(amax/448)));
  for(unsigned int i=0;i<64;i++)pooled[off+i]=ds4_e4m3fn_round(pooled[off+i]/scale)*scale;
 }
 for(unsigned int i=0;i<512;i++)cache[row*512+i]=ds4_half_round(pooled[i]);
}
extern "C" __global__ void ds4_compressor_indexer(float *pooled,const float *weight,float *cache,unsigned int row,unsigned int position) {
 if(blockIdx.x || threadIdx.x)return;
 double ss=0;for(unsigned int i=0;i<128;i++)ss+=double(pooled[i])*pooled[i];
 const float inv=1.0f/sqrtf(float(ss/128)+1e-6f);for(unsigned int i=0;i<128;i++)pooled[i]=pooled[i]*inv*weight[i];
 ds4_compressed_rope(pooled,128,position,false);ds4_indexer_qat(pooled);
 for(unsigned int i=0;i<128;i++)cache[row*128+i]=ds4_half_round(pooled[i]);
}
extern "C" __global__ void ds4_indexer_query(const float *q,float *out,unsigned int position) {
 if(threadIdx.x)return;float *row=out+blockIdx.x*128;
 for(unsigned int i=0;i<128;i++)row[i]=q[blockIdx.x*128+i];
 ds4_compressed_rope(row,128,position,false);ds4_indexer_qat(row);
}
extern "C" __global__ void ds4_indexer_select(const float *q,const float *weights,const float *cache,float *scores,unsigned int *allowed,unsigned int count) {
 if(blockIdx.x || threadIdx.x)return;
 for(unsigned int i=0;i<count;i++)allowed[i]=count<=512?1:0;
 if(count<=512)return;
 for(unsigned int c=0;c<count;c++) {
  float score=0;
  for(unsigned int h=0;h<64;h++) { float dot=0;for(unsigned int i=0;i<128;i++)dot+=q[h*128+i]*cache[c*128+i];score+=fmaxf(0.0f,dot)*weights[h]*.011048543456039805f; }
  scores[c]=score;
 }
 for(unsigned int k=0;k<512;k++) {
  unsigned int best=0;float bestScore=-__int_as_float(0x7f800000);
  for(unsigned int i=0;i<count;i++)if(!allowed[i] && scores[i]>bestScore) { best=i;bestScore=scores[i]; }
  allowed[best]=1;
 }
}
extern "C" __global__ void ds4_attention_mixed(const float *q,const float *kv,const float *sinks,float *raw,const float *comp,const unsigned int *allowed,float *scores,float *out,unsigned int position,unsigned int layer,unsigned int raw_cap,unsigned int n_comp) {
 const unsigned int head=threadIdx.x;if(blockIdx.x || head>=64)return;
  const unsigned int qoff=head*512;
  for(unsigned int i=0;i<512;i++)out[qoff+i]=q[qoff+i];
  if(head==0)for(unsigned int i=0;i<512;i++)out[32768+i]=kv[i];
  for(unsigned int r=0;r<64;r+=2){
    const unsigned int qi=qoff+448+r;float theta=ds4_compressed_theta(position,r),c=cosf(theta),s=sinf(theta),a=out[qi],b=out[qi+1];
    out[qi]=a*c-b*s;out[qi+1]=a*s+b*c;
    if(head==0){unsigned int ki=32768+448+r;a=out[ki];b=out[ki+1];out[ki]=a*c-b*s;out[ki+1]=a*s+b*c;}
  }
  __syncthreads();
  if(head==0){
    for(unsigned int off=0;off<448;off+=64){
      float amax=1.0e-4f;for(unsigned int i=0;i<64;i++)amax=fmaxf(amax,fabsf(out[32768+off+i]));
      const float scale=exp2f(ceilf(log2f(amax/448.0f)));
      for(unsigned int i=0;i<64;i++)out[32768+off+i]=ds4_e4m3fn_round(fminf(fmaxf(out[32768+off+i]/scale,-448.0f),448.0f))*scale;
    }
    const unsigned int row=position%raw_cap;
    for(unsigned int i=0;i<512;i++)raw[row*512+i]=ds4_half_round(out[32768+i]);
  }
  __syncthreads();
  const unsigned int n_raw=min(position+1,raw_cap),first=position+1-n_raw,total=n_raw+n_comp;
  float maximum=sinks[head];
  for(unsigned int t=0;t<n_raw;t++){
    const unsigned int row=(first+t)%raw_cap;float score=0.0f;
    // Both dot operands go through f16 as the oracle does; see ds4_attention_raw.
    for(unsigned int i=0;i<512;i++)score+=ds4_half_round(out[qoff+i])*raw[row*512+i];
    score*=0.04419417382415922f;scores[head*total+t]=score;maximum=fmaxf(maximum,score);
  }
  for(unsigned int cidx=0;cidx<n_comp;cidx++){
    float score=-1.0e30f;
    if(allowed[cidx]){score=0.0f;for(unsigned int i=0;i<512;i++)score+=ds4_half_round(out[qoff+i])*comp[cidx*512+i];score*=0.04419417382415922f;maximum=fmaxf(maximum,score);}
    scores[head*total+n_raw+cidx]=score;
  }
  float denom=expf(sinks[head]-maximum);
  // Retain each softmax weight. The value loop consumes the same float that
  // the denominator did instead of evaluating expf again for every value
  // dimension.
  for(unsigned int t=0;t<n_raw;t++){
    const unsigned int i=head*total+t;const float weight=expf(scores[i]-maximum);scores[i]=weight;denom+=weight;
  }
  for(unsigned int cidx=0;cidx<n_comp;cidx++)if(allowed[cidx]){
    const unsigned int i=head*total+n_raw+cidx;const float weight=expf(scores[i]-maximum);scores[i]=weight;denom+=weight;
  }
  for(unsigned int i=0;i<512;i++){
    float value=0.0f;
    for(unsigned int t=0;t<n_raw;t++){const unsigned int row=(first+t)%raw_cap;value+=scores[head*total+t]*raw[row*512+i];}
    for(unsigned int cidx=0;cidx<n_comp;cidx++)if(allowed[cidx])value+=scores[head*total+n_raw+cidx]*comp[cidx*512+i];
    out[33280+qoff+i]=value/denom;
  }
  for(unsigned int r=0;r<64;r+=2){
    const unsigned int oi=33280+qoff+448+r;float theta=ds4_compressed_theta(position,r),c=cosf(theta),s=-sinf(theta),a=out[oi],b=out[oi+1];
    out[oi]=a*c-b*s;out[oi+1]=a*s+b*c;
  }
  (void)layer;
}

extern "C" __global__ void ds4_attention_mixed_prepare(const float *q,const float *kv,const float *sinks,float *raw,const float *comp,const unsigned int *allowed,float *scores,float *denominators,float *out,unsigned int position,unsigned int layer,unsigned int raw_cap,unsigned int n_comp) {
 const unsigned int head=threadIdx.x;if(blockIdx.x || head>=64)return;
  const unsigned int qoff=head*512;
  for(unsigned int i=0;i<512;i++)out[qoff+i]=q[qoff+i];
  if(head==0)for(unsigned int i=0;i<512;i++)out[32768+i]=kv[i];
  for(unsigned int r=0;r<64;r+=2){
    const unsigned int qi=qoff+448+r;float theta=ds4_compressed_theta(position,r),c=cosf(theta),s=sinf(theta),a=out[qi],b=out[qi+1];
    out[qi]=a*c-b*s;out[qi+1]=a*s+b*c;
    if(head==0){unsigned int ki=32768+448+r;a=out[ki];b=out[ki+1];out[ki]=a*c-b*s;out[ki+1]=a*s+b*c;}
  }
  __syncthreads();
  if(head==0){
    for(unsigned int off=0;off<448;off+=64){
      float amax=1.0e-4f;for(unsigned int i=0;i<64;i++)amax=fmaxf(amax,fabsf(out[32768+off+i]));
      const float scale=exp2f(ceilf(log2f(amax/448.0f)));
      for(unsigned int i=0;i<64;i++)out[32768+off+i]=ds4_e4m3fn_round(fminf(fmaxf(out[32768+off+i]/scale,-448.0f),448.0f))*scale;
    }
    const unsigned int row=position%raw_cap;
    for(unsigned int i=0;i<512;i++)raw[row*512+i]=ds4_half_round(out[32768+i]);
  }
  __syncthreads();
  const unsigned int n_raw=min(position+1,raw_cap),first=position+1-n_raw,total=n_raw+n_comp;
  float maximum=sinks[head];
  for(unsigned int t=0;t<n_raw;t++){
    const unsigned int row=(first+t)%raw_cap;float score=0.0f;
    // Both dot operands go through f16 as the oracle does; see ds4_attention_raw.
    for(unsigned int i=0;i<512;i++)score+=ds4_half_round(out[qoff+i])*raw[row*512+i];
    score*=0.04419417382415922f;scores[head*total+t]=score;maximum=fmaxf(maximum,score);
  }
  for(unsigned int cidx=0;cidx<n_comp;cidx++){
    float score=-1.0e30f;
    if(allowed[cidx]){score=0.0f;for(unsigned int i=0;i<512;i++)score+=ds4_half_round(out[qoff+i])*comp[cidx*512+i];score*=0.04419417382415922f;maximum=fmaxf(maximum,score);}
    scores[head*total+n_raw+cidx]=score;
  }
  float denom=expf(sinks[head]-maximum);
  // Retain each softmax weight. The value loop consumes the same float that
  // the denominator did instead of evaluating expf again for every value
  // dimension.
  for(unsigned int t=0;t<n_raw;t++){
    const unsigned int i=head*total+t;const float weight=expf(scores[i]-maximum);scores[i]=weight;denom+=weight;
  }
  for(unsigned int cidx=0;cidx<n_comp;cidx++)if(allowed[cidx]){
    const unsigned int i=head*total+n_raw+cidx;const float weight=expf(scores[i]-maximum);scores[i]=weight;denom+=weight;
  }
  denominators[head]=denom;
}

extern "C" __global__ void ds4_attention_mixed_values(const float *raw,const float *comp,const unsigned int *allowed,const float *scores,const float *denominators,float *out,unsigned int position,unsigned int raw_cap,unsigned int n_comp) {
 const unsigned int head=blockIdx.x,pair=threadIdx.x;
 if(head>=64 || pair>=256)return;
 const unsigned int i=2*pair,qoff=head*512;
 const unsigned int n_raw=min(position+1,raw_cap),first=position+1-n_raw,total=n_raw+n_comp;
 float a=0.0f,b=0.0f;
 for(unsigned int t=0;t<n_raw;t++){
  const unsigned int row=(first+t)%raw_cap;const float weight=scores[head*total+t];
  a+=weight*raw[row*512+i];b+=weight*raw[row*512+i+1];
 }
 for(unsigned int cidx=0;cidx<n_comp;cidx++)if(allowed[cidx]){
  const float weight=scores[head*total+n_raw+cidx];
  a+=weight*comp[cidx*512+i];b+=weight*comp[cidx*512+i+1];
 }
 const float denom=denominators[head];a/=denom;b/=denom;
 if(i>=448){
  const float theta=ds4_compressed_theta(position,i-448),c=cosf(theta),s=-sinf(theta);
  out[33280+qoff+i]=a*c-b*s;out[33280+qoff+i+1]=a*s+b*c;
 }else{out[33280+qoff+i]=a;out[33280+qoff+i+1]=b;}
}


`
