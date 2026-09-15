
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
// ds4_compressed_rope_pair rotates the single (i,i+1) pair at offset r. The 64
// wide rotation is 32 disjoint pairs, so a lane per pair reads and writes the
// same two floats one iteration of a serial walk would have.
__device__ void ds4_compressed_rope_pair(float *row,unsigned int width,unsigned int position,bool inverse,unsigned int r) {
 const unsigned int i=width-64+r;const float theta=ds4_compressed_theta(position,r),c=cosf(theta),s=(inverse?-1.0f:1.0f)*sinf(theta),a=row[i],b=row[i+1];
 row[i]=a*c-b*s;row[i+1]=a*s+b*c;
}
__device__ float ds4_e2m1_round(float value) {
 const float levels[8]={0,.5f,1,1.5f,2,3,4,6};
 const float x=fminf(fabsf(value),6.0f);int best=0;
 for(int i=1;i<8;i++) { const float d=fabsf(x-levels[i]),b=fabsf(x-levels[best]);if(d<b || (d==b && i%2==0 && best%2!=0))best=i; }
 return value<0?-levels[best]:levels[best];
}
// ds4_indexer_qat_block quantizes one 128-wide indexer row with 128 lanes.
// A Hadamard stage partitions the row into 64 disjoint pairs, so lane p owns
// both halves of pair p and performs the same add and subtract the serial walk
// did. The four 32-wide amax scans keep their serial fmaxf order in one lane
// each. x is block-shared; the caller supplies 128 lanes.
__device__ void ds4_indexer_qat_block(float *x,unsigned int t) {
 __shared__ float amax[4];
 for(unsigned int stride=1;stride<128;stride*=2) {
  if(t<64) { const unsigned int base=(t/stride)*2*stride,i=t%stride,lo=base+i,hi=base+stride+i;const float a=x[lo],b=x[hi];x[lo]=a+b;x[hi]=a-b; }
  __syncthreads();
 }
 x[t]*=.08838834764831845f;
 __syncthreads();
 if(t<4) { float m=7.052966104933725e-38f;for(unsigned int i=0;i<32;i++)m=fmaxf(m,fabsf(x[t*32+i]));amax[t]=m; }
 __syncthreads();
 const float scale=exp2f(ceilf(log2f(amax[t/32]/6)));
 x[t]=ds4_e2m1_round(x[t]/scale)*scale;
 __syncthreads();
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
// ds4_compressor_attention finishes one compressed attention row with one lane
// per element. The 512-term double sum of squares is order bound and stays in
// lane 0; the normalization, the rotation pairs and the seven 64-wide
// quantization groups are disjoint per lane. Requires 512 lanes.
extern "C" __global__ void ds4_compressor_attention(float *pooled,const float *weight,float *cache,unsigned int row,unsigned int position) {
 if(blockIdx.x)return;
 const unsigned int t=threadIdx.x;
 __shared__ float x[512];
 __shared__ float inv;
 __shared__ float amax[7];
 x[t]=pooled[t];
 __syncthreads();
 if(!t) { double ss=0;for(unsigned int i=0;i<512;i++)ss+=double(x[i])*x[i];inv=1.0f/sqrtf(float(ss/512)+1e-6f); }
 __syncthreads();
 x[t]=x[t]*inv*weight[t];
 __syncthreads();
 if(t<32)ds4_compressed_rope_pair(x,512,position,false,2*t);
 if(t<7) { float m=1e-4f;for(unsigned int i=0;i<64;i++)m=fmaxf(m,fabsf(x[t*64+i]));amax[t]=m; }
 __syncthreads();
 if(t<448) { const float scale=exp2f(ceilf(log2f(amax[t/64]/448)));x[t]=ds4_e4m3fn_round(x[t]/scale)*scale; }
 __syncthreads();
 pooled[t]=x[t];cache[row*512+t]=ds4_half_round(x[t]);
}
// ds4_compressor_indexer finishes one compressed indexer row. The 128-term
// double sum of squares stays in lane 0 for the same reason. Requires 128 lanes.
extern "C" __global__ void ds4_compressor_indexer(float *pooled,const float *weight,float *cache,unsigned int row,unsigned int position) {
 if(blockIdx.x)return;
 const unsigned int t=threadIdx.x;
 __shared__ float x[128];
 __shared__ float inv;
 x[t]=pooled[t];
 __syncthreads();
 if(!t) { double ss=0;for(unsigned int i=0;i<128;i++)ss+=double(x[i])*x[i];inv=1.0f/sqrtf(float(ss/128)+1e-6f); }
 __syncthreads();
 x[t]=x[t]*inv*weight[t];
 __syncthreads();
 if(t<32)ds4_compressed_rope_pair(x,128,position,false,2*t);
 __syncthreads();
 ds4_indexer_qat_block(x,t);
 pooled[t]=x[t];cache[row*128+t]=ds4_half_round(x[t]);
}
// ds4_indexer_query rotates and quantizes one query head per block. The body
// carries no accumulation, so every lane is independent. Requires 128 lanes.
extern "C" __global__ void ds4_indexer_query(const float *q,float *out,unsigned int position) {
 const unsigned int t=threadIdx.x;
 __shared__ float x[128];
 x[t]=q[blockIdx.x*128+t];
 __syncthreads();
 if(t<32)ds4_compressed_rope_pair(x,128,position,false,2*t);
 __syncthreads();
 ds4_indexer_qat_block(x,t);
 out[blockIdx.x*128+t]=x[t];
}
extern "C" __global__ void ds4_indexer_score(const float *q,const float *weights,const float *cache,float *scores,unsigned int count) {
  __shared__ float s_cache[64 * 128];
  const unsigned int block_row_start = blockIdx.x * 64;
  const unsigned int global_float_start = block_row_start * 128;
  const unsigned int max_valid_floats = count * 128;

  const float4 *cache4 = (const float4 *)(cache + global_float_start);
  float4 *s_cache4 = (float4 *)s_cache;
  const unsigned int max_valid4 = max_valid_floats / 4;
  const unsigned int global_float4_start = global_float_start / 4;

  for (unsigned int idx = threadIdx.x; idx < 2048; idx += blockDim.x) {
    unsigned int g_idx = global_float4_start + idx;
    s_cache4[idx] = (g_idx < max_valid4) ? cache4[idx] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }
  __syncthreads();

  const unsigned int c = block_row_start + threadIdx.x;
  if (c >= count) return;

  const float *row = s_cache + threadIdx.x * 128;
  float score = 0.0f;
  for (unsigned int h = 0; h < 64; h++) {
    float dot = 0.0f;
    for (unsigned int i = 0; i < 128; i++) dot += q[h * 128 + i] * row[i];
    score += fmaxf(0.0f, dot) * weights[h] * 0.011048543456039805f;
  }
  scores[c] = score;
}
__device__ __forceinline__ static unsigned int ds4_topk_float_key(float v) {
 unsigned int u = __float_as_uint(v);
 return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}
__device__ __forceinline__ static unsigned long long ds4_topk_pack_key(float v, unsigned int idx) {
 return ((unsigned long long)ds4_topk_float_key(v) << 32ULL) | (unsigned long long)(0xffffffffu - idx);
}
__device__ __forceinline__ static void ds4_bitonic_sort_2048(unsigned long long *buf, unsigned int tid) {
 for (unsigned int k = 2; k <= 2048; k <<= 1) {
  for (unsigned int j = k >> 1; j > 0; j >>= 1) {
   for (unsigned int i = tid; i < 2048; i += blockDim.x) {
    unsigned int other = i ^ j;
    if (other > i) {
     unsigned long long a = buf[i];
     unsigned long long b = buf[other];
     bool desc = (i & k) == 0;
     if (desc ? (b > a) : (a > b)) {
      buf[i] = b;
      buf[other] = a;
     }
    }
   }
   __syncthreads();
  }
 }
}
extern "C" __global__ void ds4_indexer_select(const float *scores, unsigned int *allowed, unsigned int count) {
 const unsigned int tid = threadIdx.x;
 if (count <= 512) {
  for (unsigned int i = tid; i < count; i += blockDim.x) allowed[i] = 1;
  return;
 }
 for (unsigned int i = tid; i < count; i += blockDim.x) allowed[i] = 0;
 __syncthreads();

 __shared__ unsigned long long buf[2048];
 __shared__ unsigned int s_cnt;
 __shared__ unsigned long long s_thr;

 if (tid == 0) {
  s_cnt = 0;
  s_thr = 0ULL;
 }
 __syncthreads();

 const unsigned int tile = blockDim.x;
 for (unsigned int base = 0; base < count; base += tile) {
  const unsigned int i = base + tid;
  const unsigned long long thr = s_thr;
  unsigned long long key = 0ULL;
  bool take = false;
  if (i < count) {
   float s = scores[i];
   if (s > -1.0e30f && !isnan(s)) {
    key = ds4_topk_pack_key(s, i);
    take = (key > thr);
   }
  }
  const unsigned int ballot = __ballot_sync(0xffffffffu, take);
  if (take) {
   const unsigned int lane = tid & 31u;
   const unsigned int rank = __popc(ballot & ((1u << lane) - 1u));
   unsigned int pos = 0;
   if (rank == 0) pos = atomicAdd(&s_cnt, __popc(ballot));
   pos = __shfl_sync(ballot, pos, __ffs(ballot) - 1);
   buf[pos + rank] = key;
  }
  __syncthreads();

  if (s_cnt > 2048 - tile) {
   for (unsigned int j = tid; j < 2048; j += blockDim.x) {
    if (j >= s_cnt) buf[j] = 0ULL;
   }
   __syncthreads();
   ds4_bitonic_sort_2048(buf, tid);
   if (tid == 0) {
    s_thr = buf[511];
    s_cnt = 512;
   }
   __syncthreads();
  }
 }

 if (s_cnt > 512) {
  for (unsigned int j = tid; j < 2048; j += blockDim.x) {
   if (j >= s_cnt) buf[j] = 0ULL;
  }
  __syncthreads();
  ds4_bitonic_sort_2048(buf, tid);
  if (tid == 0) s_cnt = 512;
  __syncthreads();
 }

 const unsigned int n_admit = s_cnt < 512 ? s_cnt : 512;
 for (unsigned int j = tid; j < n_admit; j += blockDim.x) {
  unsigned long long k = buf[j];
  if (k != 0ULL) {
   unsigned int idx = 0xffffffffu - (unsigned int)(k & 0xffffffffu);
   allowed[idx] = 1;
  }
 }
 __syncthreads();
 if (tid == 0 && s_cnt < 512) {
  allowed[0] = 1;
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
  const unsigned int bid = blockIdx.x;
  const unsigned int tid = threadIdx.x;

  if (gridDim.x == 1) {
    const unsigned int head = tid;
    if (head >= 64) return;
    const unsigned int qoff = head * 512;
    for (unsigned int i = 0; i < 512; i++) out[qoff + i] = q[qoff + i];
    if (head == 0) {
      for (unsigned int i = 0; i < 512; i++) out[32768 + i] = kv[i];
    }
    for (unsigned int r = 0; r < 64; r += 2) {
      const unsigned int qi = qoff + 448 + r;
      float theta = ds4_compressed_theta(position, r), c = cosf(theta), s = sinf(theta), a = out[qi], b = out[qi + 1];
      out[qi] = a * c - b * s;
      out[qi + 1] = a * s + b * c;
      if (head == 0) {
        unsigned int ki = 32768 + 448 + r;
        a = out[ki]; b = out[ki + 1];
        out[ki] = a * c - b * s;
        out[ki + 1] = a * s + b * c;
      }
    }
    __syncthreads();
    for (unsigned int b = 0; b < 7; b++) {
      const unsigned int off = b * 64;
      const float val = out[32768 + off + head];
      float amax = fabsf(val);
      __shared__ float s_amax[2];
      for (int offset = 16; offset > 0; offset /= 2) {
        amax = fmaxf(amax, __shfl_down_sync(0xffffffff, amax, offset));
      }
      if ((head & 31) == 0) s_amax[head >> 5] = amax;
      __syncthreads();
      if (head == 0) s_amax[0] = fmaxf(1.0e-4f, fmaxf(s_amax[0], s_amax[1]));
      __syncthreads();
      const float group_amax = s_amax[0];
      const float scale = exp2f(ceilf(log2f(group_amax / 448.0f)));
      out[32768 + off + head] = ds4_e4m3fn_round(fminf(fmaxf(val / scale, -448.0f), 448.0f)) * scale;
      __syncthreads();
    }
    const unsigned int row = position % raw_cap;
    for (unsigned int i = head; i < 512; i += 64) {
      raw[row * 512 + i] = ds4_half_round(out[32768 + i]);
    }
    (void)sinks;(void)comp;(void)allowed;(void)scores;(void)denominators;(void)layer;(void)n_comp;
    return;
  }

  // Multi-block parallel grid: Grid(65), Block(64)
  if (bid < 64) {
    const unsigned int head = bid;
    const unsigned int qoff = head * 512;

    for (unsigned int i = tid; i < 448; i += 64) {
      out[qoff + i] = q[qoff + i];
    }
    if (tid < 32) {
      const unsigned int r = tid * 2;
      const unsigned int qi = qoff + 448 + r;
      const float theta = ds4_compressed_theta(position, r);
      const float c = cosf(theta);
      const float s = sinf(theta);
      const float a = q[qi];
      const float b = q[qi + 1];
      out[qi]     = a * c - b * s;
      out[qi + 1] = a * s + b * c;
    }
    (void)sinks;(void)comp;(void)allowed;(void)scores;(void)denominators;(void)layer;(void)n_comp;
    return;
  }

  if (bid == 64) {
    const unsigned int koff = 32768;

    for (unsigned int i = tid; i < 448; i += 64) {
      out[koff + i] = kv[i];
    }
    if (tid < 32) {
      const unsigned int r = tid * 2;
      const unsigned int ki = koff + 448 + r;
      const float theta = ds4_compressed_theta(position, r);
      const float c = cosf(theta);
      const float s = sinf(theta);
      const float a = kv[448 + r];
      const float b = kv[448 + r + 1];
      const float bs = b * s;
      const float bc = b * c;
      out[ki]     = fmaf(a, c, -bs);
      out[ki + 1] = fmaf(a, s, bc);
    }
    __syncthreads();

    __shared__ float s_amax[2];
    for (unsigned int b = 0; b < 7; b++) {
      const unsigned int off = b * 64;
      const float val = out[koff + off + tid];
      float amax = fabsf(val);
      for (int offset = 16; offset > 0; offset /= 2) {
        amax = fmaxf(amax, __shfl_down_sync(0xffffffff, amax, offset));
      }
      if ((tid & 31) == 0) s_amax[tid >> 5] = amax;
      __syncthreads();
      if (tid == 0) s_amax[0] = fmaxf(1.0e-4f, fmaxf(s_amax[0], s_amax[1]));
      __syncthreads();
      const float group_amax = s_amax[0];
      const float scale = exp2f(ceilf(log2f(group_amax / 448.0f)));
      out[koff + off + tid] = ds4_e4m3fn_round(fminf(fmaxf(val / scale, -448.0f), 448.0f)) * scale;
      __syncthreads();
    }

    const unsigned int row = position % raw_cap;
    for (unsigned int i = tid; i < 512; i += 64) {
      raw[row * 512 + i] = ds4_half_round(out[koff + i]);
    }
    (void)sinks;(void)comp;(void)allowed;(void)scores;(void)denominators;(void)layer;(void)n_comp;
    return;
  }
}

extern "C" __global__ void ds4_attention_mixed_scores(const float *raw,const float *comp,const unsigned int *allowed,float *scores,const float *out,unsigned int position,unsigned int raw_cap,unsigned int n_comp) {
 const unsigned int head=blockIdx.x,t=blockIdx.y*blockDim.x+threadIdx.x;
 const unsigned int n_raw=min(position+1,raw_cap),first=position+1-n_raw,total=n_raw+n_comp;
 if(head>=64 || t>=total)return;
 const unsigned int qoff=head*512;
 float score=-1.0e30f;
 if(t<n_raw){
  const unsigned int row=(first+t)%raw_cap;score=0.0f;
  for(unsigned int i=0;i<512;i++)score+=ds4_half_round(out[qoff+i])*raw[row*512+i];
  score*=0.04419417382415922f;
 }else{
  const unsigned int cidx=t-n_raw;
  if(allowed[cidx]){score=0.0f;for(unsigned int i=0;i<512;i++)score+=ds4_half_round(out[qoff+i])*comp[cidx*512+i];score*=0.04419417382415922f;}
 }
 scores[head*total+t]=score;
}

extern "C" __global__ void ds4_attention_mixed_normalize(const float *sinks,const unsigned int *allowed,float *scores,float *denominators,unsigned int position,unsigned int raw_cap,unsigned int n_comp) {
 const unsigned int head=threadIdx.x;if(blockIdx.x || head>=64)return;
 const unsigned int n_raw=min(position+1,raw_cap),total=n_raw+n_comp;
 float maximum=sinks[head];
 for(unsigned int t=0;t<n_raw;t++)maximum=fmaxf(maximum,scores[head*total+t]);
 for(unsigned int cidx=0;cidx<n_comp;cidx++)if(allowed[cidx])maximum=fmaxf(maximum,scores[head*total+n_raw+cidx]);
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

extern "C" __global__ void ds4_attention_mixed_fused(const float *sinks,const float *raw,const float *comp,const unsigned int *allowed,float *denominators,float *out,unsigned int position,unsigned int raw_cap,unsigned int n_comp) {
 const unsigned int head=blockIdx.x;if(head>=64)return;
 const unsigned int tid=threadIdx.x,qoff=head*512;
 alignas(16) __shared__ float s_q[512];
 alignas(16) __shared__ float s_kv[16*512];
 __shared__ float s_max;
 __shared__ float s_denom;
 extern __shared__ float s_scores[];

 s_q[tid]=ds4_half_round(out[qoff+tid]);
 s_q[tid+256]=ds4_half_round(out[qoff+tid+256]);
 __syncthreads();

 const unsigned int n_raw=min(position+1,raw_cap),first=position+1-n_raw,total=n_raw+n_comp;

 for(unsigned int base=0;base<total;base+=16){
  const unsigned int count=(total-base<16)?(total-base):16;
  const unsigned int total_vec4=count*128;
  for(unsigned int idx=tid;idx<total_vec4;idx+=256){
   const unsigned int r=idx>>7,d4=idx&127,g=base+r;
   float4 v=make_float4(0.0f,0.0f,0.0f,0.0f);
   if(g<n_raw){
    const unsigned int row=(first+g)%raw_cap;
    v=((const float4 *)(raw+row*512))[d4];
   }else{
    const unsigned int cidx=g-n_raw;
    if(allowed[cidx]){
     v=((const float4 *)(comp+cidx*512))[d4];
    }
   }
   ((float4 *)(s_kv+r*512))[d4]=v;
  }
  __syncthreads();

  if(tid<count){
   const unsigned int g=base+tid;
   float score=-1.0e30f;
   if(g<n_raw || allowed[g-n_raw]){
    score=0.0f;
    const float *kvrow=s_kv+tid*512;
    for(unsigned int i=0;i<512;i++)score+=s_q[i]*kvrow[i];
    score*=0.04419417382415922f;
   }
   s_scores[g]=score;
  }
  __syncthreads();
 }

 if(tid==0){
  float mx=sinks[head];
  for(unsigned int t=0;t<n_raw;t++)mx=fmaxf(mx,s_scores[t]);
  for(unsigned int cidx=0;cidx<n_comp;cidx++)if(allowed[cidx])mx=fmaxf(mx,s_scores[n_raw+cidx]);
  s_max=mx;
 }
 __syncthreads();

 const float mx=s_max;
 for(unsigned int t=tid;t<n_raw;t+=256)s_scores[t]=expf(s_scores[t]-mx);
 for(unsigned int cidx=tid;cidx<n_comp;cidx+=256)if(allowed[cidx])s_scores[n_raw+cidx]=expf(s_scores[n_raw+cidx]-mx);
 __syncthreads();

 if(tid==0){
  float denom=expf(sinks[head]-mx);
  for(unsigned int t=0;t<n_raw;t++)denom+=s_scores[t];
  for(unsigned int cidx=0;cidx<n_comp;cidx++)if(allowed[cidx])denom+=s_scores[n_raw+cidx];
  s_denom=denom;
  if(denominators!=0)denominators[head]=denom;
 }
 __syncthreads();

 const float denom=s_denom;
 float val0=0.0f,val1=0.0f;
 for(unsigned int t=0;t<n_raw;t++){
  const unsigned int row=(first+t)%raw_cap;const float w=s_scores[t];
  val0+=w*raw[row*512+tid];val1+=w*raw[row*512+tid+256];
 }
 for(unsigned int cidx=0;cidx<n_comp;cidx++)if(allowed[cidx]){
  const float w=s_scores[n_raw+cidx];
  val0+=w*comp[cidx*512+tid];val1+=w*comp[cidx*512+tid+256];
 }
 out[33280+qoff+tid]=val0/denom;
 out[33280+qoff+tid+256]=val1/denom;
 __syncthreads();

 if(tid<32){
  const unsigned int r=tid*2;
  const unsigned int oi=33280+qoff+448+r;
  float theta=ds4_compressed_theta(position,r),c=cosf(theta),s=-sinf(theta),a=out[oi],b=out[oi+1];
  out[oi]=a*c-b*s;out[oi+1]=a*s+b*c;
 }
}

#define DS4_SCORE_TILE_HEADS 16u
#define DS4_SCORE_TILE_ROWS  16u
#define DS4_SCORE_TILE_STRIDE 516u

extern "C" __global__ void ds4_attention_mixed_scores_tile(
    const float *raw,
    const float *comp,
    const unsigned int *allowed,
    float *scores,
    const float *out,
    unsigned int position,
    unsigned int raw_cap,
    unsigned int n_comp) {
  const unsigned int n_raw = min(position + 1, raw_cap);
  const unsigned int first = position + 1 - n_raw;
  const unsigned int total = n_raw + n_comp;
  if (total == 0) return;

  extern __shared__ float s_tile_mem[];
  float *sh_q = s_tile_mem;
  float *sh_kv = sh_q + DS4_SCORE_TILE_HEADS * DS4_SCORE_TILE_STRIDE;
  __shared__ float sh_add[DS4_SCORE_TILE_ROWS];

  const unsigned int g_base = blockIdx.x * DS4_SCORE_TILE_ROWS;
  const unsigned int h_base = blockIdx.y * DS4_SCORE_TILE_HEADS;
  if (g_base >= total || h_base >= 64) return;

  const unsigned int tile_heads = 64 - h_base < DS4_SCORE_TILE_HEADS ? 64 - h_base : DS4_SCORE_TILE_HEADS;
  for (unsigned int idx = threadIdx.x; idx < tile_heads * 128u; idx += blockDim.x) {
    const unsigned int hh = idx >> 7u;
    const unsigned int dd = idx & 127u;
    const float4 v = ((const float4 *)(out + (h_base + hh) * 512u))[dd];
    float *dst = sh_q + hh * DS4_SCORE_TILE_STRIDE + dd * 4u;
    dst[0] = ds4_half_round(v.x);
    dst[1] = ds4_half_round(v.y);
    dst[2] = ds4_half_round(v.z);
    dst[3] = ds4_half_round(v.w);
  }

  if (threadIdx.x < DS4_SCORE_TILE_ROWS) {
    const unsigned int g = g_base + threadIdx.x;
    float add = -1.0e30f;
    if (g < total) {
      if (g < n_raw) {
        add = 0.0f;
      } else {
        const unsigned int cidx = g - n_raw;
        if (allowed[cidx]) add = 0.0f;
      }
    }
    sh_add[threadIdx.x] = add;
  }
  __syncthreads();

  const unsigned int rr0 = threadIdx.x >> 7u;
  const unsigned int dd = threadIdx.x & 127u;
  for (unsigned int r = rr0; r < DS4_SCORE_TILE_ROWS; r += 2u) {
    const unsigned int g = g_base + r;
    if (g >= total) continue;
    if (sh_add[r] <= -1.0e20f) continue;
    const float4 *src;
    if (g < n_raw) {
      const unsigned int raw_row = (first + g) % raw_cap;
      src = (const float4 *)(raw + raw_row * 512u);
    } else {
      const unsigned int cidx = g - n_raw;
      src = (const float4 *)(comp + cidx * 512u);
    }
    const float4 v = src[dd];
    float *dst = sh_kv + r * DS4_SCORE_TILE_STRIDE + dd * 4u;
    dst[0] = v.x; dst[1] = v.y; dst[2] = v.z; dst[3] = v.w;
  }
  __syncthreads();

  const unsigned int r = threadIdx.x & 15u;
  const unsigned int hh = threadIdx.x >> 4u;
  const unsigned int h = h_base + hh;
  const unsigned int g = g_base + r;
  if (h >= 64 || g >= total) return;

  float *row_scores = scores + (unsigned long long)h * total;
  const float *qh = sh_q + hh * DS4_SCORE_TILE_STRIDE;
  const float *kvrow = sh_kv + r * DS4_SCORE_TILE_STRIDE;
  float s = -1.0e30f;
  if (sh_add[r] > -1.0e20f) {
    float dot = 0.0f;
    #pragma unroll 1
    for (unsigned int d = 0; d < 512u; d++) {
      dot += qh[d] * kvrow[d];
    }
    s = dot * 0.04419417382415922f;
  }
  row_scores[g] = s;
}

extern "C" __global__ void ds4_attention_mixed_finalize(
    const float *sinks,
    const float *raw,
    const float *comp,
    const unsigned int *allowed,
    const float *scores,
    float *denominators,
    float *out,
    unsigned int position,
    unsigned int raw_cap,
    unsigned int n_comp) {
  const unsigned int h = blockIdx.x;
  if (h >= 64) return;
  const unsigned int tid = threadIdx.x;
  const unsigned int qoff = h * 512;
  const unsigned int n_raw = min(position + 1, raw_cap);
  const unsigned int first = position + 1 - n_raw;
  const unsigned int total = n_raw + n_comp;

  __shared__ float s_max;
  __shared__ float s_denom;
  extern __shared__ float s_scores[];

  const float *row_scores = scores + (unsigned long long)h * total;
  for (unsigned int i = tid; i < total; i += 512) {
    s_scores[i] = row_scores[i];
  }
  __syncthreads();

  if (tid == 0) {
    float mx = sinks[h];
    for (unsigned int t = 0; t < n_raw; t++) mx = fmaxf(mx, s_scores[t]);
    for (unsigned int cidx = 0; cidx < n_comp; cidx++) if (allowed[cidx]) mx = fmaxf(mx, s_scores[n_raw + cidx]);
    s_max = mx;
  }
  __syncthreads();

  const float mx = s_max;
  for (unsigned int i = tid; i < n_raw; i += 512) s_scores[i] = expf(s_scores[i] - mx);
  for (unsigned int i = tid; i < n_comp; i += 512) {
    if (allowed[i]) s_scores[n_raw + i] = expf(s_scores[n_raw + i] - mx);
  }
  __syncthreads();

  if (tid == 0) {
    float denom = expf(sinks[h] - mx);
    for (unsigned int t = 0; t < n_raw; t++) denom += s_scores[t];
    for (unsigned int cidx = 0; cidx < n_comp; cidx++) if (allowed[cidx]) denom += s_scores[n_raw + cidx];
    s_denom = denom;
    if (denominators != 0) denominators[h] = denom;
  }
  __syncthreads();

  const float denom = s_denom;
  const unsigned int d = tid;
  float acc = 0.0f;
  for (unsigned int r = 0; r < n_raw; r++) {
    const unsigned int raw_row = (first + r) % raw_cap;
    acc += s_scores[r] * raw[raw_row * 512 + d];
  }
  for (unsigned int c = 0; c < n_comp; c++) {
    if (allowed[c]) {
      acc += s_scores[n_raw + c] * comp[c * 512 + d];
    }
  }
  out[33280 + qoff + d] = acc / denom;
  __syncthreads();

  if (tid < 32) {
    const unsigned int r = tid * 2;
    const unsigned int oi = 33280 + qoff + 448 + r;
    float theta = ds4_compressed_theta(position, r), c = cosf(theta), s = -sinf(theta);
    float a = out[oi], b = out[oi + 1];
    out[oi] = a * c - b * s;
    out[oi + 1] = a * s + b * c;
  }
}

