package cudaattn

import "github.com/tmc/lib/nvidia/x/cudakernel"

// IndexedSource is an independently written ordinary FP32 CUDA baseline. One
// thread handles each query head. Mode 0 validates mapping, mode 1 additionally
// validates selection, mode 2 computes attention, and mode 3 computes scores.
// Compute modes are submitted only after the separate validation has completed.
// Scores are the maximum scaled Q dot K over visible tokens in each block;
// fully masked blocks have negative infinity. This is a deterministic proxy,
// not a learned selector or a claim of equivalence to full dense attention.
const IndexedSource = `
__device__ __forceinline__ bool indexed_finite(float x) {
    return (__float_as_uint(x) & 0x7f800000u) != 0x7f800000u;
}
extern "C" __global__ void indexed_attention(
    const float* q, const float* k, const float* v,
    const int* sequences, const int* positions, const int* lengths,
    const int* pages, const int* ids, const int* counts,
    int* status, float* output, float* lse,
    float* scores, int* score_lengths, int* score_offsets,
    int batch, int queries, int qheads, int kvheads, int dim,
    int block_size, int max_sequence, int physical_blocks, int selected,
    int shared_sets, int causal, float scale, int mode) {
    int row = blockIdx.x;
    if (threadIdx.x != 0 || row >= queries*qheads) return;
    int query = row/qheads, head = row%qheads;
    int kvhead = head/(qheads/kvheads);
    int set_heads = shared_sets ? kvheads : qheads;
    int set = query*set_heads + (shared_sets ? kvhead : head);
    int max_blocks = (max_sequence+block_size-1)/block_size;
    status[row] = 0;
    int seq = sequences[query];
    if (seq < 0 || seq >= batch) { status[row]=1; return; }
    int length = lengths[seq], position = positions[query];
    if (length<0 || length>max_sequence || position < -1 || position>=length) {status[row]=1;return;}
    int blocks = (length+block_size-1)/block_size;
    if (mode<=1) {
        for (int s=0;s<batch;s++) {
            int n=lengths[s];
            if(n<0 || n>max_sequence){status[row]=1;return;}
            int used=(n+block_size-1)/block_size;
            for (int b=0;b<max_blocks;b++) {
                int p=pages[s*max_blocks+b];
                if ((b<used && (p<0 || p>=physical_blocks)) || (b>=used && p!=-1)) {status[row]=1;return;}
            }
        }
        if (mode==1) {
            int count=counts[set];
            if (count<0 || count>selected) {status[row]=1;return;}
            for (int j=0;j<selected;j++) {
                int id=ids[set*selected+j];
                if (j>=count) {if(id!=-1){status[row]=1;return;}continue;}
                if(id<0 || id>=blocks){status[row]=1;return;}
                for(int a=0;a<j;a++) if(ids[set*selected+a]==id){status[row]=1;return;}
            }
        }
        return;
    }
    float u[128];
    for(int d=0;d<dim;d++) {
        u[d]=0.0f;
        if(!indexed_finite(q[row*dim+d])) {status[row]=2;return;}
    }
    float m=-__int_as_float(0x7f800000), l=0.0f;
    int visits=mode==3 ? blocks : counts[set];
    for(int j=0;j<visits;j++) {
        int logical=mode==3 ? j : ids[set*selected+j];
        int physical=pages[seq*max_blocks+logical];
        float maximum=-__int_as_float(0x7f800000);
        for(int t=0;t<block_size;t++) {
            int actual=logical*block_size+t;
            if(actual>=length || (causal && actual>position)) continue;
            int base=((physical*block_size+t)*kvheads+kvhead)*dim;
            float dot=0.0f;
            for(int d=0;d<dim;d++) {
                float key=k[base+d];
                if(!indexed_finite(key)) {status[row]=2;return;}
                dot=__fadd_rn(dot,__fmul_rn(q[row*dim+d],key));
            }
            float s=__fmul_rn(dot,scale);
            if(!indexed_finite(s)) {status[row]=2;return;}
            if(mode==3) {maximum=fmaxf(maximum,s);continue;}
            float next=l==0.0f ? s : fmaxf(m,s);
            float alpha=l==0.0f ? 0.0f : expf(m-next);
            float p=expf(s-next);
            l=__fadd_rn(__fmul_rn(alpha,l),p);
            for(int d=0;d<dim;d++) {
                float value=v[base+d];
                if(!indexed_finite(value)){status[row]=2;return;}
                u[d]=__fadd_rn(__fmul_rn(alpha,u[d]),__fmul_rn(p,value));
                if(!indexed_finite(u[d])){status[row]=2;return;}
            }
            m=next;
        }
        if(mode==3) scores[row*max_blocks+j]=maximum;
    }
    if(mode==3) {
        for(int j=blocks;j<max_blocks;j++) scores[row*max_blocks+j]=-__int_as_float(0x7f800000);
        score_lengths[row]=blocks;score_offsets[row]=0;return;
    }
    for(int d=0;d<dim;d++) {
        float result=l==0.0f ? 0.0f : u[d]/l;
        if(!indexed_finite(result)){status[row]=2;return;}
        output[row*dim+d]=result;
    }
    float logsum=l==0.0f ? -__int_as_float(0x7f800000) : m+logf(l);
    if(l!=0.0f && !indexed_finite(logsum)){status[row]=2;return;}
    lse[row]=logsum;
}
`

// IndexedSchema returns the exact pointer/scalar ABI shared by all four modes.
func IndexedSchema() (cudakernel.Schema, error) {
	var p []cudakernel.Param
	for _, name := range []string{"q", "k", "v", "sequences", "positions", "lengths", "pages", "ids", "counts", "status", "output", "lse", "scores", "score_lengths", "score_offsets"} {
		p = append(p, cudakernel.Param{Name: name, Type: "ptr", IsPointer: true})
	}
	for _, name := range []string{"batch", "queries", "qheads", "kvheads", "dim", "block_size", "max_sequence", "physical_blocks", "selected", "shared_sets", "causal"} {
		p = append(p, cudakernel.Param{Name: name, Type: "s32"})
	}
	p = append(p, cudakernel.Param{Name: "scale", Type: "f32"}, cudakernel.Param{Name: "mode", Type: "s32"})
	return cudakernel.NewSchema(p...)
}
