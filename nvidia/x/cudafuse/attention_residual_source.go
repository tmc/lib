package cudafuse

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"reflect"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudablock"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// AttentionResidualConfig specifies contiguous FP32 candidates [C,R,D], actual
// normalization multipliers g[D], and pseudo-query weights w[D]. Candidates
// include the current partial block last. The finite implementation admits
// 1<=C<=8, 1<=R<=4, 1<=D<=129 and positive finite epsilon. FP32 accumulation
// uses 32-thread reductions, increasing candidate order and no contraction.
// A sole candidate preserves its raw FP32 bits; additive multi-candidate mixing
// may change zero signs. Subnormals are not flushed; nonfinite intermediates
// produce a resident diagnostic instead of an admitted output.
type AttentionResidualConfig struct {
	Candidates, Rows, Width int
	Epsilon                 float32
}

func (c AttentionResidualConfig) validate() error {
	if c.Candidates < 1 || c.Candidates > 8 || c.Rows < 1 || c.Rows > 4 || c.Width < 1 || c.Width > 129 || !attentionResidualFinite(c.Epsilon) || c.Epsilon <= 0 {
		return errors.New("cudafuse: unsupported attention residual extent or epsilon")
	}
	return nil
}

// AttentionResidualPath selects four composed kernels or one bounded fused
// kernel. Both use one ordinary, independently synchronized CTA per row.
type AttentionResidualPath int

const (
	AttentionResidualComposed AttentionResidualPath = iota + 1
	AttentionResidualFused
)

func attentionResidualHeader(full string) (string, error) {
	const anchor = "// Block-level cooperative 2D"
	if strings.Count(full, anchor) != 1 || strings.Count(full, "#endif") != 1 || !strings.Contains(full, "#ifndef CUDABLOCK_CUH") || !strings.Contains(full, "#define CUDABLOCK_CUH") {
		return "", errors.New("cudafuse: unsupported collective header form")
	}
	h := full[:strings.Index(full, anchor)]
	for _, include := range []string{"#include <cuda_fp16.h>", "#include <cuda_bf16.h>"} {
		if strings.Count(h, include) != 1 {
			return "", errors.New("cudafuse: collective include form changed")
		}
		h = strings.Replace(h, include, "", 1)
	}
	if strings.Contains(h, "#include") || strings.Contains(h, "#endif") || !strings.Contains(h, "T block_reduce_sum(T val, T* shared_scratch)") || !strings.Contains(h, "T block_reduce_max(T val, T* shared_scratch)") {
		return "", errors.New("cudafuse: incomplete collective fragment")
	}
	return h + "\n#endif // CUDABLOCK_CUH\n", nil
}

func attentionResidualDigest(s string) string { return fmt.Sprintf("%x", sha256.Sum256([]byte(s))) }

// AttentionResidualSchema defines seven naturally aligned pointer argument
// slots. FP32 data pointers need only four-byte alignment, independently of
// the eight-byte host ABI slots. Shapes and epsilon are source-specialized.
func AttentionResidualSchema() (cudakernel.Schema, error) {
	var p []cudakernel.Param
	for _, n := range []string{"values", "multiplier", "query", "scores", "probabilities", "temporary", "status"} {
		p = append(p, cudakernel.Param{Name: n, Type: "ptr", ByteWidth: 8, Alignment: 8, IsPointer: true})
	}
	return cudakernel.NewSchema(p...)
}

// AttentionResidualSource returns the source-specialized composed or fused
// ordinary CUDA entries. Resident validation and intermediate overflow status
// precede publication. The reused FP32 collective bodies are unchanged; unused
// half/BF16 and tile helpers are excluded by a checked private header trim.
func AttentionResidualSource(c AttentionResidualConfig, path AttentionResidualPath) ([]FusedKernel, error) {
	if err := c.validate(); err != nil {
		return nil, err
	}
	h, err := attentionResidualHeader(cudablock.Header)
	if err != nil {
		return nil, err
	}
	prefix := fmt.Sprintf("#define AR_C %d\n#define AR_R %d\n#define AR_D %d\n#define AR_EPS %.9ef\n#define AR_MUTANT 0\n", c.Candidates, c.Rows, c.Width, c.Epsilon)
	entries := []string{"ar_validate", "ar_score", "ar_softmax", "ar_contract"}
	if path == AttentionResidualFused {
		entries = []string{"ar_fused"}
	} else if path != AttentionResidualComposed {
		return nil, errors.New("cudafuse: invalid attention residual path")
	}
	var out []FusedKernel
	for _, entry := range entries {
		out = append(out, FusedKernel{Name: entry, Source: prefix + h + attentionResidualCUDA, Inputs: []string{"values", "multiplier", "query"}, Outputs: []string{"scores", "probabilities", "temporary", "status"}})
	}
	return out, nil
}

// CompileAttentionResidual performs offline NVRTC compilation only. All
// supplied source/header dependencies and conservative numerical flags are
// retained in each immutable artifact. It does not initialize the CUDA driver.
func CompileAttentionResidual(c AttentionResidualConfig, path AttentionResidualPath, major, minor int) ([]*cudakernel.Artifact, error) {
	sources, err := AttentionResidualSource(c, path)
	if err != nil {
		return nil, err
	}
	return compileAttentionResidualSources(sources, major, minor)
}

func compileAttentionResidualSources(sources []FusedKernel, major, minor int) ([]*cudakernel.Artifact, error) {
	h, err := attentionResidualHeader(cudablock.Header)
	if err != nil {
		return nil, err
	}
	schema, err := AttentionResidualSchema()
	if err != nil {
		return nil, err
	}
	var out []*cudakernel.Artifact
	for _, s := range sources {
		flags := []string{"--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true"}
		opts := []nvrtcx.Option{nvrtcx.WithArch(major, minor), nvrtcx.WithNoCache()}
		for _, f := range flags {
			opts = append(opts, nvrtcx.WithFlag(f))
		}
		cm, err := nvrtcx.Compile(s.Source, opts...)
		if err != nil {
			return nil, fmt.Errorf("cudafuse: compile attention residual %s: %w", s.Name, err)
		}
		a, err := cm.ToArtifact(s.Name, schema, fmt.Sprintf("compute_%d%d", major, minor))
		if err != nil {
			return nil, err
		}
		// ToArtifact does not retain compiler flags. These are the exact
		// required numerical flags supplied above, not an invented compiler
		// report. The real compiler key binds all effective flags and tools.
		a.CompilerOpts = append([]string(nil), flags...)
		a.Toolchain = "nvrtc compiler-key=" + cm.Key()
		a.Dependencies = append(a.Dependencies, cudakernel.Dependency{Path: "attention_residual.cu", Digest: attentionResidualDigest(s.Source), Tracked: true}, cudakernel.Dependency{Path: "cudablock-owner.cuh", Digest: attentionResidualDigest(cudablock.Header), Tracked: true}, cudakernel.Dependency{Path: "cudablock-fp32.cuh", Digest: attentionResidualDigest(h), Tracked: true})
		if err := a.Validate(); err != nil {
			return nil, err
		}
		out = append(out, a)
	}
	return out, nil
}

func attentionResidualArtifact(a *cudakernel.Artifact, s FusedKernel) (*cudakernel.Artifact, error) {
	if err := a.Validate(); err != nil {
		return nil, err
	}
	schema, err := AttentionResidualSchema()
	if err != nil {
		return nil, err
	}
	if a.EntryPoint != s.Name || !reflect.DeepEqual(a.Schema, schema) || a.Kind != cudakernel.KindPTX && a.Kind != cudakernel.KindCUBIN {
		return nil, errors.New("cudafuse: attention residual artifact ABI differs")
	}
	h, err := attentionResidualHeader(cudablock.Header)
	if err != nil {
		return nil, err
	}
	for p, want := range map[string]string{"attention_residual.cu": attentionResidualDigest(s.Source), "cudablock-owner.cuh": attentionResidualDigest(cudablock.Header), "cudablock-fp32.cuh": attentionResidualDigest(h)} {
		count := 0
		for _, d := range a.Dependencies {
			if d.Path == p {
				count++
				if !d.Tracked || d.ErrorMsg != "" || d.Digest != want {
					return nil, errors.New("cudafuse: attention residual source binding differs")
				}
			}
		}
		if count != 1 {
			return nil, errors.New("cudafuse: missing or duplicate attention residual dependency")
		}
	}
	for _, flag := range []string{"--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true"} {
		count := 0
		for _, v := range a.CompilerOpts {
			key := strings.Split(strings.TrimPrefix(flag, "--"), "=")[0]
			if strings.Contains(v, key) && v != flag {
				return nil, errors.New("cudafuse: contradictory numerical compiler identity")
			}
			if v == flag {
				count++
			}
			if strings.Contains(v, "use_fast_math") {
				return nil, errors.New("cudafuse: unsupported fast math")
			}
		}
		if count != 1 {
			return nil, errors.New("cudafuse: missing numerical compiler identity")
		}
	}
	owned := *a
	owned.Raw = append([]byte(nil), a.Raw...)
	owned.Schema.Params = append([]cudakernel.Param(nil), a.Schema.Params...)
	owned.CompilerOpts = append([]string(nil), a.CompilerOpts...)
	owned.Dependencies = append([]cudakernel.Dependency(nil), a.Dependencies...)
	return &owned, nil
}

const attentionResidualCUDA = `
#define AR_ARGS const float *v,const float *g,const float *w,float *scores,float *p,float *tmp,int *status
__device__ bool ar_geometry() { return blockDim.x==32&&blockDim.y==1&&blockDim.z==1&&gridDim.x==AR_R&&gridDim.y==1&&gridDim.z==1; }
__device__ void ar_validate_row(const float *v,const float *g,const float *w,int *status,int r,float *scratch) {
    int lane=threadIdx.x;float bad=0.0f;
    for(int d=lane;d<AR_D;d+=32) { if(!isfinite(g[d])||!isfinite(w[d]))bad=1.0f; for(int c=0;c<AR_C;++c)if(!isfinite(v[(c*AR_R+r)*AR_D+d]))bad=1.0f; }
    float total=block_reduce_sum<float,32>(bad,scratch);
    if(lane==0)status[r]=total==0.0f?0:1;
    __syncthreads();
}
__device__ void ar_scores(const float *v,const float *g,const float *w,float *scores,int *status,int r,float *scratch,float *scalar) {
    int lane=threadIdx.x;
    for(int c=0;c<AR_C;++c) {
        float squares=0.0f;for(int d=lane;d<AR_D;d+=32) {float x=v[(c*AR_R+r)*AR_D+d];squares=__fadd_rn(squares,__fmul_rn(x,x));}
        float sum=block_reduce_sum<float,32>(squares,scratch);
        if(lane==0) {scalar[0]=sqrtf(__fadd_rn(sum/(float)AR_D,AR_EPS));if(!isfinite(sum)||!isfinite(scalar[0])||scalar[0]<=0.0f)status[r]=2;}
        __syncthreads();
        float dot=0.0f;for(int d=lane;d<AR_D;d+=32) {float multiplier=AR_MUTANT==3?__fadd_rn(g[d],-1.0f):g[d];float term=__fmul_rn(__fmul_rn(w[d],multiplier),v[(c*AR_R+r)*AR_D+d]/scalar[0]);dot=__fadd_rn(dot,term);}
        sum=block_reduce_sum<float,32>(dot,scratch);
        if(lane==0) {if(AR_MUTANT==4)sum/=sqrtf((float)AR_D);if(!isfinite(sum))status[r]=2;scores[c*AR_R+r]=sum;}
        __syncthreads();
    }
}
__device__ void ar_probs(const float *scores,float *p,int *status,int r) {
    if(threadIdx.x==0) {
        int count=AR_MUTANT==5&&AR_C>1?AR_C-1:AR_C;
        float mx=scores[r];for(int c=1;c<count;++c)mx=fmaxf(mx,scores[c*AR_R+r]);float sum=0.0f;
        for(int c=0;c<count;++c) {float delta=__fadd_rn(scores[c*AR_R+r],-mx);if(!isfinite(delta))status[r]=2;float x=expf(delta);p[c*AR_R+r]=x;sum=__fadd_rn(sum,x);}
        for(int c=0;c<AR_C;++c)p[c*AR_R+r]=c<count?p[c*AR_R+r]/sum:0.0f;
        if(AR_MUTANT==2) {for(int c=0;c<AR_C;++c) {float mx=scores[c*AR_R],den=0.0f;for(int q=1;q<AR_R;++q)mx=fmaxf(mx,scores[c*AR_R+q]);for(int q=0;q<AR_R;++q)den+=expf(scores[c*AR_R+q]-mx);p[c*AR_R+r]=expf(scores[c*AR_R+r]-mx)/den;}}
        if(!isfinite(sum)||sum<=0.0f)status[r]=2;
    }
    __syncthreads();
}
__device__ void ar_values(const float *v,const float *p,float *tmp,int *status,int r) {
    for(int d=threadIdx.x;d<AR_D;d+=32) {
        if(AR_C==1&&AR_MUTANT==0){tmp[r*AR_D+d]=v[r*AR_D+d];continue;}
        float result=0.0f;for(int c=0;c<AR_C;++c) {float x=v[(c*AR_R+r)*AR_D+d];if(AR_MUTANT==1) {float squares=0.0f;for(int k=0;k<AR_D;++k){float a=v[(c*AR_R+r)*AR_D+k];squares=__fadd_rn(squares,__fmul_rn(a,a));}x/=sqrtf(squares/(float)AR_D+AR_EPS);}result=__fadd_rn(result,__fmul_rn(p[c*AR_R+r],x));}
        if(!isfinite(result))atomicExch(&status[r],2);tmp[r*AR_D+d]=result;
    }
    __syncthreads();
}
extern "C" __global__ void ar_validate(AR_ARGS) {if(!ar_geometry())return;__shared__ float scratch[1];ar_validate_row(v,g,w,status,blockIdx.x,scratch);}
extern "C" __global__ void ar_score(AR_ARGS) {if(!ar_geometry())return;int r=blockIdx.x;if(status[r]!=0)return;__shared__ float scratch[1],scalar[1];ar_scores(v,g,w,scores,status,r,scratch,scalar);}
extern "C" __global__ void ar_softmax(AR_ARGS) {if(!ar_geometry())return;int r=blockIdx.x;if(status[r]!=0)return;ar_probs(scores,p,status,r);}
extern "C" __global__ void ar_contract(AR_ARGS) {if(!ar_geometry())return;int r=blockIdx.x;if(status[r]!=0)return;ar_values(v,p,tmp,status,r);}
extern "C" __global__ void ar_fused(AR_ARGS) {if(!ar_geometry())return;int r=blockIdx.x;__shared__ float scratch[1],scalar[1],s[AR_C*AR_R],prob[AR_C*AR_R];__shared__ int localstatus[AR_R];ar_validate_row(v,g,w,status,r,scratch);if(status[r]!=0)return;ar_scores(v,g,w,s,status,r,scratch,scalar);if(status[r]!=0)return;if(AR_MUTANT==2){if(threadIdx.x==0)for(int q=0;q<AR_R;++q)localstatus[q]=0;__syncthreads();for(int q=0;q<AR_R;++q)if(q!=r)ar_scores(v,g,w,s,localstatus,q,scratch,scalar);}ar_probs(s,prob,status,r);if(status[r]!=0)return;ar_values(v,prob,tmp,status,r);}
`
