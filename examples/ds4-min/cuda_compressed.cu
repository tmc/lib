
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
__device__ __forceinline__ float ds4_dense_theta(unsigned int position,unsigned int r) {
 const float scale=powf(10000.0f,-2.0f/64.0f);
 float theta=(float)position;
 for(unsigned int i=0;i<r;i+=2) theta*=scale;
 return theta;
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
// The three kernels below are ds4_compressor_pool for a whole group of tokens.
// ds4_compressor_pool keeps a rolling window and pools when the window closes,
// which reads like a scan over tokens, but the carry is of inputs and not of
// results: the window closing at Q reads the raw kv and gate of positions
// Q-2*ratio+1..Q for ratio 4 and Q-ratio+1..Q otherwise, and nothing a pool
// wrote. So the windows are independent and a group of T tokens is T/ratio
// blocks rather than T dispatches. At a group of 2048 that was two thirds of
// every kernel the prefill issued, each one block wide on a 48-SM device.
//
// The group must open at a multiple of ratio, so a token's index within the
// group is its index within its window, and the incoming rolling window -- read
// only here -- supplies the first window's older half.
// ds4_compressor_term recovers the r'th pair of the window that closes at
// block b: the score the rolling kernel would have stored and the value beside
// it. A ratio 4 window carries two, the second from the token four later; the
// first four of a group's tokens take theirs from the incoming rolling window,
// which is where the down-shift left them.
__device__ void ds4_compressor_term(const float *kv,const float *gate,const float *state_kv,const float *state_score,const unsigned short *ape,unsigned int head,unsigned int width,unsigned int ratio,unsigned int b,unsigned int r,unsigned int j,float *score,float *value) {
 if(ratio!=4) {
  const unsigned int t=b*ratio+r;
  score[0]=gate[t*width+j]+ds4_half_to_float(ape[r*width+j]);value[0]=kv[t*width+j];
  return;
 }
 const int lo=int(4*b)-4+int(r);
 if(lo<0) { score[0]=state_score[r*width+j];value[0]=state_kv[r*width+j]; }
 else { score[0]=gate[unsigned(lo)*width+j]+ds4_half_to_float(ape[r*width+j]);value[0]=kv[unsigned(lo)*width+j]; }
 const unsigned int hi=4*b+r;
 score[1]=gate[hi*width+head+j]+ds4_half_to_float(ape[r*width+head+j]);value[1]=kv[hi*width+head+j];
}
// ds4_compressor_window pools the window that closes at block b. The two scans
// below are ds4_compressor_pool's, addend for addend and in the same order, so
// the pooled row is the same float and not merely a close one.
__device__ void ds4_compressor_window(const float *kv,const float *gate,const float *state_kv,const float *state_score,float *pooled,const unsigned short *ape,unsigned int head,unsigned int width,unsigned int ratio,unsigned int b) {
 const unsigned int j=threadIdx.x;if(j>=head)return;
 float score[2],value[2],maximum=-__int_as_float(0x7f800000);
 for(unsigned int r=0;r<ratio;r++) {
  ds4_compressor_term(kv,gate,state_kv,state_score,ape,head,width,ratio,b,r,j,score,value);
  maximum=fmaxf(maximum,score[0]);
  if(ratio==4)maximum=fmaxf(maximum,score[1]);
 }
 float denominator=0,sum=0;
 if(isfinite(maximum))for(unsigned int r=0;r<ratio;r++) {
  ds4_compressor_term(kv,gate,state_kv,state_score,ape,head,width,ratio,b,r,j,score,value);
  const float wp=expf(score[0]-maximum);
  if(ratio==4) { const float wc=expf(score[1]-maximum);denominator+=wp+wc;sum+=wp*value[0];sum+=wc*value[1]; }
  else { denominator+=wp;sum+=wp*value[0]; }
 }
 pooled[b*head+j]=denominator>0?sum/denominator:0;
}
extern "C" __global__ void ds4_compressor_pool_batch(const float *kv,const float *gate,const float *state_kv,const float *state_score,float *pooled,const unsigned short *ape,unsigned int head,unsigned int width,unsigned int ratio) {
 ds4_compressor_window(kv,gate,state_kv,state_score,pooled,ape,head,width,ratio,blockIdx.x);
}
// ds4_compressor_state_batch leaves the rolling window holding what the token
// at a time kernel would have left in it. Only the last token to reach a row
// survives there, so each row names its own source and no block reads a row
// another block writes. Row r of the lower half of a ratio 4 window comes from
// the last down-shift, that is from the last closed window, not from the last
// token to carry residue r.
extern "C" __global__ void ds4_compressor_state_batch(const float *kv,const float *gate,float *state_kv,float *state_score,const unsigned short *ape,unsigned int width,unsigned int ratio,unsigned int n) {
 unsigned int r=blockIdx.x,src;
 if(ratio==4 && blockIdx.x<4) {
  if(n<4)return;
  src=4*(n/4)-4+r;                 // the last closed window, in token order
 } else {
  if(ratio==4)r=blockIdx.x-4;
  if(r>=n)return;
  src=r+ratio*((n-1-r)/ratio);     // the last token to carry residue r
 }
 for(unsigned int c=threadIdx.x;c<width;c+=blockDim.x) {
  state_kv[blockIdx.x*width+c]=kv[src*width+c];
  state_score[blockIdx.x*width+c]=gate[src*width+c]+ds4_half_to_float(ape[r*width+c]);
 }
}
// ds4_compressor_attention finishes one compressed attention row with one lane
// per element. The 512-term double sum of squares is order bound and stays in
// lane 0; the normalization, the rotation pairs and the seven 64-wide
// quantization groups are disjoint per lane. Requires 512 lanes.
__device__ void ds4_compressor_attention_row(float *pooled,const float *weight,float *cache,unsigned int row,unsigned int position) {
 const unsigned int t=threadIdx.x;
 __shared__ float x[512];
 __shared__ float inv;
 __shared__ float amax[7];
 x[t]=pooled[t];
 __syncthreads();
 if(!t) { double ss=0;for(unsigned int i=0;i<512;i++)ss+=double(x[i])*x[i];inv=rsqrtf(float(ss/512)+1e-6f); }
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
extern "C" __global__ void ds4_compressor_attention(float *pooled,const float *weight,float *cache,unsigned int row,unsigned int position) {
 if(blockIdx.x)return;
 ds4_compressor_attention_row(pooled,weight,cache,row,position);
}
// ds4_compressor_attention_batch finishes one closed window per block. Block b
// holds the pooled row b wrote, lands cache row row0+b, and rotates at the
// position that window opened at. Requires 512 lanes.
extern "C" __global__ void ds4_compressor_attention_batch(float *pooled,const float *weight,float *cache,unsigned int row0,unsigned int position0,unsigned int ratio) {
 ds4_compressor_attention_row(pooled+blockIdx.x*512,weight,cache,row0+blockIdx.x,position0+blockIdx.x*ratio);
}
// ds4_compressor_indexer finishes one compressed indexer row. The 128-term
// double sum of squares stays in lane 0 for the same reason. Requires 128 lanes.
__device__ void ds4_compressor_indexer_row(float *pooled,const float *weight,float *cache,unsigned int row,unsigned int position) {
 const unsigned int t=threadIdx.x;
 __shared__ float x[128];
 __shared__ float inv;
 x[t]=pooled[t];
 __syncthreads();
 if(!t) { double ss=0;for(unsigned int i=0;i<128;i++)ss+=double(x[i])*x[i];inv=rsqrtf(float(ss/128)+1e-6f); }
 __syncthreads();
 x[t]=x[t]*inv*weight[t];
 __syncthreads();
 if(t<32)ds4_compressed_rope_pair(x,128,position,false,2*t);
 __syncthreads();
 ds4_indexer_qat_block(x,t);
 pooled[t]=x[t];cache[row*128+t]=ds4_half_round(x[t]);
}
extern "C" __global__ void ds4_compressor_indexer(float *pooled,const float *weight,float *cache,unsigned int row,unsigned int position) {
 if(blockIdx.x)return;
 ds4_compressor_indexer_row(pooled,weight,cache,row,position);
}
// ds4_compressor_indexer_batch is ds4_compressor_attention_batch at 128 lanes.
extern "C" __global__ void ds4_compressor_indexer_batch(float *pooled,const float *weight,float *cache,unsigned int row0,unsigned int position0,unsigned int ratio) {
 ds4_compressor_indexer_row(pooled+blockIdx.x*128,weight,cache,row0+blockIdx.x,position0+blockIdx.x*ratio);
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
// DS4_INDEX_ROW_STRIDE pads each cached row by one float. Every thread walks
// its own row at the same offset i, so with a stride of 128 the whole warp
// lands on bank (i % 32) together: a 32-way conflict on each of the 8192
// shared loads below, which is the entire cost of this kernel (8192 * 32
// cycles at 1.5 GHz is ~175 us against the 138 us measured). 129 % 32 == 1
// makes the bank (threadIdx.x + i) % 32, so a warp covers all 32 banks. The
// arithmetic and its order are untouched.
#define DS4_INDEX_ROW_STRIDE 129u

// ds4_indexer_score gives each cache row a warp, not a thread.
//
// One thread per row put 8192 dependent FMAs on a single chain and put only
// count threads on the whole device: at 2048 compressed rows that is 32
// blocks of 64 threads, under 1.5% of the machine, every one of them stalled
// on the previous multiply-add. Splitting by head fixes both without moving
// an addend. Lane l owns heads l and l+32, so each of its two chains is 128
// long instead of 8192 and the two are independent; the 64 head terms are
// then gathered in h order 0, 1, ... 63 by __shfl_sync, which is the order
// the serial loop summed them in. Same products, same addends, same
// sequence: the decode digest is unaffected.
//
// q is staged because the per-head split makes lanes read different rows of
// it; DS4_INDEX_ROW_STRIDE keeps that access conflict-free the same way it
// does for the cache rows.
#define DS4_INDEX_ROWS_PER_BLOCK 8u

extern "C" __global__ void ds4_indexer_score(const float *q,const float *weights,const float *cache,float *scores,unsigned int count) {
  __shared__ float s_q[64 * DS4_INDEX_ROW_STRIDE];
  __shared__ float s_cache[DS4_INDEX_ROWS_PER_BLOCK * DS4_INDEX_ROW_STRIDE];

  const unsigned int block_row_start = blockIdx.x * DS4_INDEX_ROWS_PER_BLOCK;
  const unsigned int max_valid_floats = count * 128;

  for (unsigned int idx = threadIdx.x; idx < 64 * 128; idx += blockDim.x)
    s_q[(idx / 128) * DS4_INDEX_ROW_STRIDE + idx % 128] = q[idx];
  for (unsigned int idx = threadIdx.x; idx < DS4_INDEX_ROWS_PER_BLOCK * 128; idx += blockDim.x) {
    const unsigned int g_idx = block_row_start * 128 + idx;
    s_cache[(idx / 128) * DS4_INDEX_ROW_STRIDE + idx % 128] =
        (g_idx < max_valid_floats) ? cache[g_idx] : 0.0f;
  }
  __syncthreads();

  const unsigned int warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
  const unsigned int c = block_row_start + warp;
  const float *row = s_cache + warp * DS4_INDEX_ROW_STRIDE;
  const float *qa = s_q + lane * DS4_INDEX_ROW_STRIDE;
  const float *qb = s_q + (lane + 32u) * DS4_INDEX_ROW_STRIDE;
  float dot_a = 0.0f, dot_b = 0.0f;
  for (unsigned int i = 0; i < 128; i++) {
    dot_a += qa[i] * row[i];
    dot_b += qb[i] * row[i];
  }
  // Gather the term BEFORE its last multiply, and fold that multiply into the
  // accumulate with fmaf. The serial form is score += fmaxf(0,dot)*w[h]*k, and
  // both the reference and the original kernel contract its tail into
  // fma(fmaxf(0,dot)*w[h], k, score) -- NVRTC defaults to -fmad=true and Go on
  // arm64 fuses x*y+z as well. Gathering the finished term and adding it plainly
  // leaves no multiply for the add to fuse with, which is a different rounding
  // and cost this kernel a bit-exactness failure against the serial reference
  // in 44% of rows. Same addends in the same order was not sufficient: the
  // contraction has to survive too.
  const float k = 0.011048543456039805f;
  const float t_a = fmaxf(0.0f, dot_a) * weights[lane];
  const float t_b = fmaxf(0.0f, dot_b) * weights[lane + 32u];
  float score = 0.0f;
  for (unsigned int h = 0; h < 32; h++) score = fmaf(__shfl_sync(0xffffffff, t_a, h), k, score);
  for (unsigned int h = 0; h < 32; h++) score = fmaf(__shfl_sync(0xffffffff, t_b, h), k, score);
  if (c < count && lane == 0) scores[c] = score;
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
// DS4_SELECT_RANK_MAX is the largest count ds4_indexer_select_rank handles.
// Its work is quadratic in count, so it is the cheaper choice only while count
// stays small. At decode this kernel sees a few hundred scores, where the
// quadratic term is a few hundred thousand comparisons spread across the
// device -- against a 2048-element bitonic sort that one block has to run by
// itself, 66 stages deep with a barrier between each.
#define DS4_SELECT_RANK_MAX 4096u

// ds4_indexer_select_rank admits an element when fewer than 512 elements
// outrank it.
//
// This is the same selection the threshold-and-sort path makes, bit for bit,
// and not by a bit-exactness argument about arithmetic: ds4_topk_pack_key
// packs the score above the complemented index, so no two elements compare
// equal and "the top 512 keys" names exactly one set however it is found. A
// mask is not a sum -- there is no addend order to preserve here.
//
// The short-count fallback needs the number of valid candidates, which is a
// figure over the whole input and would otherwise need a cross-block
// reduction. It does not: every thread already scans every element to compute
// one rank, so the thread that owns element 0 counts the valid ones on the way
// past and applies the fallback alone.
__device__ static void ds4_indexer_select_rank(const float *scores, unsigned int *allowed, unsigned int count) {
 // Static, not dynamic, shared memory, and a strided loop over i rather than
 // one element per thread. Both exist so the kernel is correct under ANY
 // launch geometry: a caller that under-provisions the grid would otherwise
 // leave most of allowed[] simply unwritten -- the rank path has no zeroing
 // pass to fall back on -- and one that forgot the shared-memory argument
 // would index a zero-length array. A launch geometry is an optimization
 // here, never a correctness contract.
 __shared__ float sh_scores[DS4_SELECT_RANK_MAX];
 for (unsigned int j = threadIdx.x; j < count; j += blockDim.x) sh_scores[j] = scores[j];
 __syncthreads();

 const unsigned int stride = gridDim.x * blockDim.x;
 for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += stride) {
  const float si = sh_scores[i];
  const bool valid = (si > -1.0e30f && !isnan(si));
  const unsigned long long key = valid ? ds4_topk_pack_key(si, i) : 0ULL;

  unsigned int rank = 0, nvalid = 0;
  for (unsigned int j = 0; j < count; j++) {
   const float sj = sh_scores[j];
   if (!(sj > -1.0e30f && !isnan(sj))) continue;
   nvalid++;
   if (valid && ds4_topk_pack_key(sj, j) > key) rank++;
  }
  allowed[i] = (valid && rank < 512u) ? 1u : 0u;
  if (i == 0 && nvalid < 512u) allowed[0] = 1u;
 }
}

extern "C" __global__ void ds4_indexer_select(const float *scores, unsigned int *allowed, unsigned int count) {
 const unsigned int tid = threadIdx.x;
 if (count <= 512) {
  const unsigned int stride = gridDim.x * blockDim.x;
  for (unsigned int i = blockIdx.x * blockDim.x + tid; i < count; i += stride) allowed[i] = 1;
  return;
 }
 if (count <= DS4_SELECT_RANK_MAX) {
  ds4_indexer_select_rank(scores, allowed, count);
  return;
 }
 // The threshold-and-sort path below evolves one shared threshold across
 // tiles, so it is single-block by construction. Every other block leaves.
 if (blockIdx.x) return;
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
// 513, not 516: the dot loop has every thread at the same offset d in its own
// row, so the bank is (r * STRIDE + d) % 32. 516 % 32 == 4 puts the 16 rows on
// 8 banks and each bank under 4 threads of a warp; 513 % 32 == 1 gives (r + d)
// % 32, 16 distinct banks with the two heads of a row sharing an address, which
// broadcasts. It is also 384 bytes smaller a block.
#define DS4_SCORE_TILE_STRIDE 513u
// The query tile is stored as binary16, halving its footprint without losing a
// bit: every value written to it has already been through ds4_half_round, so
// it is exactly representable and the round trip back to float is the
// identity. That takes the block from 65,664 bytes of shared to 49,216, which
// is what decides whether two blocks fit on an SM instead of one. The q tile
// needs no bank padding -- all sixteen threads of a row read the same offset
// and broadcast -- so its stride is the bare 512.
#define DS4_SCORE_TILE_QSTRIDE 512u

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
  float *sh_kv = s_tile_mem;
  __half *sh_q = (__half *)(sh_kv + DS4_SCORE_TILE_ROWS * DS4_SCORE_TILE_STRIDE);
  __shared__ float sh_add[DS4_SCORE_TILE_ROWS];

  const unsigned int g_base = blockIdx.x * DS4_SCORE_TILE_ROWS;
  const unsigned int h_base = blockIdx.y * DS4_SCORE_TILE_HEADS;
  if (g_base >= total || h_base >= 64) return;

  const unsigned int tile_heads = 64 - h_base < DS4_SCORE_TILE_HEADS ? 64 - h_base : DS4_SCORE_TILE_HEADS;
  for (unsigned int idx = threadIdx.x; idx < tile_heads * 128u; idx += blockDim.x) {
    const unsigned int hh = idx >> 7u;
    const unsigned int dd = idx & 127u;
    const float4 v = ((const float4 *)(out + (h_base + hh) * 512u))[dd];
    __half *dst = sh_q + hh * DS4_SCORE_TILE_QSTRIDE + dd * 4u;
    dst[0] = __float2half_rn(ds4_half_round(v.x));
    dst[1] = __float2half_rn(ds4_half_round(v.y));
    dst[2] = __float2half_rn(ds4_half_round(v.z));
    dst[3] = __float2half_rn(ds4_half_round(v.w));
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
  const __half *qh = sh_q + hh * DS4_SCORE_TILE_QSTRIDE;
  const float *kvrow = sh_kv + r * DS4_SCORE_TILE_STRIDE;
  float s = -1.0e30f;
  if (sh_add[r] > -1.0e20f) {
    float dot = 0.0f;
    // Unrolled, not reassociated: one accumulator in the original order, so
    // the decode digest is unaffected. Unrolling only lets the shared loads
    // run ahead of the dependent adds, the same reasoning the finalize
    // denominator loops below are annotated with.
    #pragma unroll 8
    for (unsigned int d = 0; d < 512u; d++) {
      dot += __half2float(qh[d]) * kvrow[d];
    }
    s = dot * 0.04419417382415922f;
  }
  row_scores[g] = s;
}

// DS4_FINALIZE_HEADS_PER_BLOCK is how many attention heads one
// ds4_attention_mixed_finalize block finishes. cudaFinalizeHeadsPerBlock in
// cuda_compressed_linux.go must hold the same value, and it also scales the
// dynamic shared-memory request, which is now HEADS * total floats.
//
// The constant exists because this is latent attention: h indexes sinks,
// scores and the output slice, but never raw, comp or allowed. Every one of
// the 64 blocks therefore streamed the byte-identical latent cache, roughly
// 5 MB per layer at ctx 4096, sixty-four times over. Those are L2 hits rather
// than device reads, so they never showed up against the memory budget, but
// they are the largest single term in this kernel. Folding HEADS heads into
// one block divides that traffic by HEADS: the row is loaded once and feeds
// HEADS independent accumulators.
//
// Each head keeps its own accumulator and its own addend order, so the change
// is bit-exact by construction and the decode digest gates it rather than
// tolerating it.
#define DS4_FINALIZE_HEADS_PER_BLOCK 2u

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
  const unsigned int h0 = blockIdx.x * DS4_FINALIZE_HEADS_PER_BLOCK;
  if (h0 >= 64) return;
  const unsigned int tid = threadIdx.x;
  const unsigned int n_raw = min(position + 1, raw_cap);
  const unsigned int first = position + 1 - n_raw;
  const unsigned int total = n_raw + n_comp;

  __shared__ float s_max[DS4_FINALIZE_HEADS_PER_BLOCK];
  __shared__ float s_denom[DS4_FINALIZE_HEADS_PER_BLOCK];
  __shared__ float s_warpmax[16];
  extern __shared__ float s_all[];

  // Every loop below is written for the 512 threads that own an output
  // element. The block may be launched wider -- see the denominators after
  // this loop -- so the extra threads are held out rather than folded into a
  // stride of 512 they do not divide, which would exponentiate an element
  // twice. The __syncthreads stay outside the guards.
  for (unsigned int j = 0; j < DS4_FINALIZE_HEADS_PER_BLOCK; j++) {
    const unsigned int h = h0 + j;
    float *s_scores = s_all + (unsigned long long)j * total;
    const float *row_scores = scores + (unsigned long long)h * total;
    if (tid < 512) {
      for (unsigned int i = tid; i < total; i += 512) {
        s_scores[i] = row_scores[i];
      }
    }
    __syncthreads();

    // fmaxf is associative and commutative over the non-NaN scores, so the
    // block reduction below is bit-identical to the serial scan it replaces.
    // The sum that follows is not, and stays serial for that reason.
    if (tid < 512) {
      float mx = sinks[h];
      for (unsigned int t = tid; t < n_raw; t += 512) mx = fmaxf(mx, s_scores[t]);
      for (unsigned int cidx = tid; cidx < n_comp; cidx += 512) {
        if (allowed[cidx]) mx = fmaxf(mx, s_scores[n_raw + cidx]);
      }
      for (unsigned int off = 16; off; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off));
      if ((tid & 31u) == 0) s_warpmax[tid >> 5u] = mx;
    }
    __syncthreads();
    if (tid == 0) {
      float b = s_warpmax[0];
      for (unsigned int w = 1; w < 16; w++) b = fmaxf(b, s_warpmax[w]);
      s_max[j] = b;
    }
    __syncthreads();

    const float mx = s_max[j];
    if (tid < 512) {
      for (unsigned int i = tid; i < n_raw; i += 512) s_scores[i] = expf(s_scores[i] - mx);
      for (unsigned int i = tid; i < n_comp; i += 512) {
        if (allowed[i]) s_scores[n_raw + i] = expf(s_scores[n_raw + i] - mx);
      }
    }
    __syncthreads();
  }

  // The denominators, one thread per head, after the loop rather than inside
  // it. Each is a serial sum in a pinned order -- one dependent add after
  // another, with nothing else for that thread to issue -- and on a
  // 4096-token context that chain is long. It is independent of the output
  // pass below: the accumulation reads the exponentiated scores, and only the
  // final division needs a denominator, so the two run together and meet at
  // the __syncthreads before that division. Inside the loop they could not:
  // a thread there has to reach the next iteration's __syncthreads, so every
  // other thread would wait on the chain this is trying to hide.
  //
  // The threads are the last DS4_FINALIZE_HEADS_PER_BLOCK of the block.
  // Launched with an extra warp they own no output element and the sums
  // overlap the accumulation entirely; launched at 512 they own one each and
  // do both in turn, which is slower but still correct. A launch geometry is
  // an optimization here, never a correctness contract.
  //
  // Every addend and its order are unchanged, and a run of plain adds has no
  // a*b+c in it for the move to re-contract.
  if (tid >= blockDim.x - DS4_FINALIZE_HEADS_PER_BLOCK) {
    const unsigned int j = tid - (blockDim.x - DS4_FINALIZE_HEADS_PER_BLOCK);
    const unsigned int h = h0 + j;
    const float *s_scores = s_all + (unsigned long long)j * total;
    const float mx = s_max[j];
    float denom = expf(sinks[h] - mx);
    #pragma unroll 8
    for (unsigned int t = 0; t < n_raw; t++) denom += s_scores[t];
    #pragma unroll 8
    for (unsigned int cidx = 0; cidx < n_comp; cidx++) if (allowed[cidx]) denom += s_scores[n_raw + cidx];
    s_denom[j] = denom;
    if (denominators != 0) denominators[h] = denom;
  }

  // One pass over the latent cache feeding every head this block owns. The
  // per-head addend order is unchanged; only the loads are shared.
  const unsigned int d = tid;
  float acc[DS4_FINALIZE_HEADS_PER_BLOCK];
  #pragma unroll
  for (unsigned int j = 0; j < DS4_FINALIZE_HEADS_PER_BLOCK; j++) acc[j] = 0.0f;
  if (d < 512) for (unsigned int r = 0; r < n_raw; r++) {
    const unsigned int raw_row = (first + r) % raw_cap;
    const float v = raw[raw_row * 512 + d];
    #pragma unroll
    for (unsigned int j = 0; j < DS4_FINALIZE_HEADS_PER_BLOCK; j++)
      acc[j] += s_all[(unsigned long long)j * total + r] * v;
  }
  if (d < 512) for (unsigned int c = 0; c < n_comp; c++) {
    if (allowed[c]) {
      const float v = comp[c * 512 + d];
      #pragma unroll
      for (unsigned int j = 0; j < DS4_FINALIZE_HEADS_PER_BLOCK; j++)
        acc[j] += s_all[(unsigned long long)j * total + n_raw + c] * v;
    }
  }
  // The denominator threads are still summing until here.
  __syncthreads();
  if (d < 512) {
    #pragma unroll
    for (unsigned int j = 0; j < DS4_FINALIZE_HEADS_PER_BLOCK; j++)
      out[33280 + (h0 + j) * 512 + d] = acc[j] / s_denom[j];
  }
  __syncthreads();

  if (tid < 32) {
    const unsigned int r = tid * 2;
    float theta = ds4_compressed_theta(position, r), c = cosf(theta), s = -sinf(theta);
    #pragma unroll
    for (unsigned int j = 0; j < DS4_FINALIZE_HEADS_PER_BLOCK; j++) {
      const unsigned int oi = 33280 + (h0 + j) * 512 + 448 + r;
      float a = out[oi], b = out[oi + 1];
      out[oi] = a * c - b * s;
      out[oi + 1] = a * s + b * c;
    }
  }
}

// ds4_attention_prepare_kv_row writes one token's key/value row to
// token_out[32768:33280]: rotated, rounded to e4m3 per 64-wide block, then to
// half. Call with 64 threads.
__device__ __forceinline__ void ds4_attention_prepare_kv_row(
    const float *token_kv, float *token_out, unsigned int position, unsigned int dense_rope) {
  const unsigned int tid = threadIdx.x;
  const unsigned int koff = 32768u;

  for (unsigned int i = tid; i < 448u; i += 64u) {
    token_out[koff + i] = token_kv[i];
  }
  if (tid < 32u) {
    const unsigned int r = tid * 2u;
    const unsigned int ki = koff + 448u + r;
    const float theta = dense_rope ? ds4_dense_theta(position, r) : ds4_compressed_theta(position, r);
    const float c = cosf(theta);
    const float s = sinf(theta);
    const float a = token_kv[448u + r];
    const float b = token_kv[448u + r + 1u];
    const float bs = b * s;
    const float bc = b * c;
    token_out[ki]      = fmaf(a, c, -bs);
    token_out[ki + 1u] = fmaf(a, s, bc);
  }
  __syncthreads();

  __shared__ float s_amax[2];
  for (unsigned int b = 0; b < 7u; b++) {
    const unsigned int off = b * 64u;
    const float val = token_out[koff + off + tid];
    float amax = fabsf(val);
    for (int offset = 16; offset > 0; offset /= 2) {
      amax = fmaxf(amax, __shfl_down_sync(0xffffffff, amax, offset));
    }
    if ((tid & 31u) == 0) s_amax[tid >> 5u] = amax;
    __syncthreads();
    if (tid == 0) s_amax[0] = fmaxf(1.0e-4f, fmaxf(s_amax[0], s_amax[1]));
    __syncthreads();
    const float group_amax = s_amax[0];
    const float scale = exp2f(ceilf(log2f(group_amax / 448.0f)));
    token_out[koff + off + tid] = ds4_e4m3fn_round(fminf(fmaxf(val / scale, -448.0f), 448.0f)) * scale;
    __syncthreads();
  }

  for (unsigned int i = tid; i < 512u; i += 64u) {
    token_out[koff + i] = ds4_half_round(token_out[koff + i]);
  }
}

extern "C" __global__ void ds4_attention_mixed_prepare_batch(
    const float *q,
    const float *kv,
    float *out,
    unsigned int base_position,
    unsigned int raw_cap,
    unsigned int num_tokens,
    unsigned int dense_rope) {
  const unsigned int t = blockIdx.y;
  if (t >= num_tokens) return;
  const unsigned int bid = blockIdx.x;
  const unsigned int tid = threadIdx.x;
  const unsigned int position = base_position + t;

  const float *token_q = q + (unsigned long long)t * (64u * 512u);
  const float *token_kv = kv + (unsigned long long)t * 512u;
  float *token_out = out + (unsigned long long)t * 66048u;

  if (bid < 64u) {
    const unsigned int head = bid;
    const unsigned int qoff = head * 512u;

    for (unsigned int i = tid; i < 448u; i += 64u) {
      token_out[qoff + i] = token_q[qoff + i];
    }
    if (tid < 32u) {
      const unsigned int r = tid * 2u;
      const unsigned int qi = qoff + 448u + r;
      const float theta = dense_rope ? ds4_dense_theta(position, r) : ds4_compressed_theta(position, r);
      const float c = cosf(theta);
      const float s = sinf(theta);
      const float a = token_q[qi];
      const float b = token_q[qi + 1u];
      token_out[qi]      = a * c - b * s;
      token_out[qi + 1u] = a * s + b * c;
    }
    return;
  }

  if (bid == 64u) {
    ds4_attention_prepare_kv_row(token_kv, token_out, position, dense_rope);
  }
}

// ds4_attention_mixed_prepare_kv_batch is the key/value half of
// ds4_attention_mixed_prepare_batch, for callers whose query rows were
// rotated in place by ds4_head_rms_rope_batch. Launch one 64-thread block
// per token.
extern "C" __global__ void ds4_attention_mixed_prepare_kv_batch(
    const float *kv,
    float *out,
    unsigned int base_position,
    unsigned int num_tokens,
    unsigned int dense_rope) {
  const unsigned int t = blockIdx.x;
  if (t >= num_tokens) return;
  ds4_attention_prepare_kv_row(kv + (unsigned long long)t * 512u, out + (unsigned long long)t * 66048u,
                               base_position + t, dense_rope);
}

// ds4_head_rms_rope_batch normalizes each 512-float query head row of q in
// place and rotates its last 64 dimensions at the row's token position. It
// is bit-identical to ds4_rms (unweighted, width 512, 256 threads) followed
// by the query half of ds4_attention_mixed_prepare_batch: the norm repeats
// ds4_rms's general path statement for statement, and the rotation is the
// same expression on the same normalized floats. Launch one 256-thread block
// per head row, 64 rows per token.
extern "C" __global__ void ds4_head_rms_rope_batch(
    float *q, unsigned int base_position, unsigned int num_tokens, unsigned int dense_rope) {
  const unsigned int row = blockIdx.x;
  if (row >= num_tokens * 64u) return;
  const unsigned int width = 512u;
  const unsigned int tid = threadIdx.x;
  const unsigned int warp_id = tid >> 5;
  const unsigned int lane = tid & 31;
  const unsigned int num_warps = blockDim.x >> 5;
  float *x = q + (unsigned long long)row * width;

  float sum = 0.0f;
  for (unsigned int i = tid; i < width; i += blockDim.x) {
    const float v = x[i];
    sum += v * v;
  }

  #pragma unroll
  for (unsigned int stride = 16; stride != 0; stride >>= 1) {
    sum += __shfl_down_sync(0xffffffff, sum, stride);
  }

  __shared__ float s_warp[32];
  if (lane == 0 && warp_id < 32) {
    s_warp[warp_id] = sum;
  }
  __syncthreads();

  float block_sum = 0.0f;
  if (warp_id == 0) {
    block_sum = (lane < num_warps && lane < 32) ? s_warp[lane] : 0.0f;
    #pragma unroll
    for (unsigned int stride = 16; stride != 0; stride >>= 1) {
      block_sum += __shfl_down_sync(0xffffffff, block_sum, stride);
    }
  }

  __shared__ float s_inv;
  if (tid == 0) {
    s_inv = rsqrtf(block_sum / float(width) + 1.0e-6f);
  }
  __syncthreads();

  // Threads 192..255 hold the rotated dimensions 448..511 on their second
  // pass. The even lane of each pair takes its neighbor's value and rotates
  // the pair. ds4_attention_mixed_prepare_batch writes a*c - b*s and
  // a*s + b*c and leaves the contraction to NVRTC, which there picks the fused
  // multiply-adds below; left to itself here it contracts a*s + b*c the other
  // way round, so they are spelled out to match that kernel bit for bit.
  const float inv = s_inv;
  float tail = 0.0f;
  for (unsigned int i = tid; i < width; i += blockDim.x) {
    const float y = x[i] * inv;
    if (i < 448u) x[i] = y;
    else tail = y;
  }
  if (tid >= 192u) {
    const float next = __shfl_down_sync(0xffffffff, tail, 1);
    if ((tid & 1u) == 0) {
      const unsigned int r = tid - 192u;
      const unsigned int position = base_position + row / 64u;
      const float theta = dense_rope ? ds4_dense_theta(position, r) : ds4_compressed_theta(position, r);
      const float c = cosf(theta);
      const float s = sinf(theta);
      const float a = tail;
      const float b = next;
      x[448u + r] = fmaf(a, c, -(b * s));
      x[449u + r] = fmaf(a, s, b * c);
    }
  }
}

extern "C" __global__ void ds4_attention_mixed_scores_tile_batch(
    const float *raw,
    const float *comp,
    const unsigned int *allowed,
    float *scores,
    const float *out,
    const unsigned int *token_n_comp,
    unsigned int base_position,
    unsigned int raw_cap,
    unsigned int max_comp,
    unsigned int score_stride,
    unsigned int num_tokens) {
  const unsigned int t = blockIdx.z;
  if (t >= num_tokens) return;

  const unsigned int position = base_position + t;
  const unsigned int n_comp = token_n_comp[t];
  const unsigned int n_raw = min(position + 1u, raw_cap);
  const unsigned int first = position + 1u - n_raw;
  const unsigned int total = n_raw + n_comp;
  if (total == 0u) return;

  const unsigned int g_base = blockIdx.x * 32u;
  const unsigned int h_base = blockIdx.y * DS4_SCORE_TILE_HEADS;
  if (g_base >= total || h_base >= 64u) return;

  extern __shared__ float s_tile_mem[];
  float *sh_q = s_tile_mem;
  float *sh_kv = sh_q + DS4_SCORE_TILE_HEADS * DS4_SCORE_TILE_STRIDE;
  // The batch path uses twice as many key rows per block as the single-token
  // path.  Each thread still owns one (head,row) dot product, so the FP32
  // accumulation order is unchanged; the wider tile only reduces redundant
  // global/shared-memory staging and block scheduling overhead.
  __shared__ float sh_add[32];

  const float *token_out = out + (unsigned long long)t * 66048u;
  const unsigned int *token_allowed = allowed + (unsigned long long)t * max_comp;
  float *token_scores = scores + (unsigned long long)t * score_stride;

  const unsigned int tile_heads = 64u - h_base < DS4_SCORE_TILE_HEADS ? 64u - h_base : DS4_SCORE_TILE_HEADS;
  for (unsigned int idx = threadIdx.x; idx < tile_heads * 128u; idx += blockDim.x) {
    const unsigned int hh = idx >> 7u;
    const unsigned int dd = idx & 127u;
    const float4 v = ((const float4 *)(token_out + (h_base + hh) * 512u))[dd];
    float *dst = sh_q + hh * DS4_SCORE_TILE_STRIDE + dd * 4u;
    dst[0] = ds4_half_round(v.x);
    dst[1] = ds4_half_round(v.y);
    dst[2] = ds4_half_round(v.z);
    dst[3] = ds4_half_round(v.w);
  }

  if (threadIdx.x < 32u) {
    const unsigned int g = g_base + threadIdx.x;
    float add = -1.0e30f;
    if (g < total) {
      if (g < n_raw) {
        add = 0.0f;
      } else {
        const unsigned int cidx = g - n_raw;
        if (token_allowed[cidx]) add = 0.0f;
      }
    }
    sh_add[threadIdx.x] = add;
  }
  __syncthreads();

  const unsigned int rr0 = (threadIdx.x >> 7u) * 8u;
  const unsigned int dd = threadIdx.x & 127u;
  for (unsigned int r = rr0; r < rr0 + 8u; r++) {
    const unsigned int g = g_base + r;
    if (g >= total) continue;
    if (sh_add[r] <= -1.0e20f) continue;
    const float4 *src;
    if (g < n_raw) {
      const unsigned int pos_g = first + g;
      if (pos_g < base_position) {
        const unsigned int raw_row = pos_g % raw_cap;
        src = (const float4 *)(raw + raw_row * 512u);
      } else {
        const unsigned int t_g = pos_g - base_position;
        src = (const float4 *)(out + (unsigned long long)t_g * 66048u + 32768u);
      }
    } else {
      const unsigned int cidx = g - n_raw;
      src = (const float4 *)(comp + cidx * 512u);
    }
    const float4 v = src[dd];
    float *dst = sh_kv + r * DS4_SCORE_TILE_STRIDE + dd * 4u;
    dst[0] = v.x; dst[1] = v.y; dst[2] = v.z; dst[3] = v.w;
  }
  __syncthreads();

  const unsigned int r = threadIdx.x & 31u;
  const unsigned int hh = threadIdx.x >> 5u;
  const unsigned int h = h_base + hh;
  const unsigned int g = g_base + r;
  if (h >= 64u || g >= total) return;

  float *row_scores = token_scores + (unsigned long long)h * total;
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

extern "C" __global__ void ds4_attention_mixed_finalize_batch(
    const float *sinks,
    const float *raw,
    const float *comp,
    const unsigned int *allowed,
    const float *scores,
    float *denominators,
    float *out,
    const unsigned int *token_n_comp,
    unsigned int base_position,
    unsigned int raw_cap,
    unsigned int max_comp,
    unsigned int score_stride,
    unsigned int num_tokens) {
  const unsigned int h = blockIdx.x;
  const unsigned int t = blockIdx.y;
  if (h >= 64u || t >= num_tokens) return;

  const unsigned int tid = threadIdx.x;
  const unsigned int qoff = h * 512u;
  const unsigned int position = base_position + t;
  const unsigned int n_comp = token_n_comp[t];
  const unsigned int n_raw = min(position + 1u, raw_cap);
  const unsigned int first = position + 1u - n_raw;
  const unsigned int total = n_raw + n_comp;

  const float *token_scores = scores + (unsigned long long)t * score_stride;
  const unsigned int *token_allowed = allowed + (unsigned long long)t * max_comp;
  float *token_denominators = denominators != 0 ? denominators + (unsigned long long)t * 64u : 0;
  float *token_out = out + (unsigned long long)t * 66048u;

  __shared__ float s_max;
  __shared__ float s_denom;
  extern __shared__ float s_scores[];

  const float *row_scores = token_scores + (unsigned long long)h * total;
  for (unsigned int i = tid; i < total; i += 512u) {
    s_scores[i] = row_scores[i];
  }
  __syncthreads();

  if (tid == 0) {
    float mx = sinks[h];
    for (unsigned int p = 0; p < n_raw; p++) mx = fmaxf(mx, s_scores[p]);
    for (unsigned int cidx = 0; cidx < n_comp; cidx++) {
      if (token_allowed[cidx]) mx = fmaxf(mx, s_scores[n_raw + cidx]);
    }
    s_max = mx;
  }
  __syncthreads();

  const float mx = s_max;
  for (unsigned int i = tid; i < n_raw; i += 512u) s_scores[i] = expf(s_scores[i] - mx);
  for (unsigned int i = tid; i < n_comp; i += 512u) {
    if (token_allowed[i]) s_scores[n_raw + i] = expf(s_scores[n_raw + i] - mx);
  }
  __syncthreads();

  if (tid == 0) {
    float denom = expf(sinks[h] - mx);
    for (unsigned int p = 0; p < n_raw; p++) denom += s_scores[p];
    for (unsigned int cidx = 0; cidx < n_comp; cidx++) {
      if (token_allowed[cidx]) denom += s_scores[n_raw + cidx];
    }
    s_denom = denom;
    if (token_denominators != 0) token_denominators[h] = denom;
  }
  __syncthreads();

  const float denom = s_denom;
  const unsigned int d = tid;
  float acc = 0.0f;
  for (unsigned int r = 0; r < n_raw; r++) {
    const unsigned int pos_r = first + r;
    float kv_val;
    if (pos_r < base_position) {
      const unsigned int raw_row = pos_r % raw_cap;
      kv_val = raw[raw_row * 512u + d];
    } else {
      const unsigned int t_r = pos_r - base_position;
      kv_val = out[(unsigned long long)t_r * 66048u + 32768u + d];
    }
    acc += s_scores[r] * kv_val;
  }
  for (unsigned int c = 0; c < n_comp; c++) {
    if (token_allowed[c]) {
      acc += s_scores[n_raw + c] * comp[c * 512u + d];
    }
  }
  token_out[33280u + qoff + d] = acc / denom;
  __syncthreads();

  if (tid < 32u) {
    const unsigned int r = tid * 2u;
    const unsigned int oi = 33280u + qoff + 448u + r;
    float theta = ds4_compressed_theta(position, r), c = cosf(theta), s = -sinf(theta);
    float a = token_out[oi], b = token_out[oi + 1u];
    token_out[oi]      = a * c - b * s;
    token_out[oi + 1u] = a * s + b * c;
  }
}

extern "C" __global__ void ds4_attention_mixed_commit_raw(
    float *raw,
    const float *out,
    unsigned int base_position,
    unsigned int raw_cap,
    unsigned int num_tokens) {
  const unsigned int count = num_tokens < raw_cap ? num_tokens : raw_cap;
  const unsigned int b = blockIdx.x;
  if (b >= count) return;
  const unsigned int t = num_tokens - count + b;
  const unsigned int pos = base_position + t;
  const unsigned int raw_row = pos % raw_cap;
  const float *src = out + (unsigned long long)t * 66048u + 32768u;
  float *dst = raw + raw_row * 512u;
  for (unsigned int i = threadIdx.x; i < 512u; i += blockDim.x) {
    dst[i] = src[i];
  }
}
