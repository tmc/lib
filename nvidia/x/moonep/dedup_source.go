//go:build !windows

package moonep

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// DedupKernel is an ordinary local CUDA kernel and its pointer/count ABI.
type DedupKernel struct {
	Name, Source string
	Schema       cudakernel.Schema
}

func dedupSchema(names ...string) cudakernel.Schema {
	var params []cudakernel.Param
	for i, name := range names {
		params = append(params, cudakernel.Param{Name: name, Type: "ptr", IsPointer: true, Offset: i * 8, ByteWidth: 8, Alignment: 8})
	}
	params = append(params, cudakernel.Param{Name: "count", Type: "s32", Offset: len(names) * 8, ByteWidth: 4, Alignment: 4})
	return cudakernel.Schema{Params: params, TotalParamBytes: (len(names)*8 + 4 + 7) &^ 7}
}

func dedupArray(name string, xs []int32) string {
	var s strings.Builder
	fmt.Fprintf(&s, "__device__ __constant__ int %s[%d]={", name, len(xs))
	for i, x := range xs {
		if i > 0 {
			s.WriteByte(',')
		}
		fmt.Fprint(&s, x)
	}
	s.WriteString("};\n")
	return s.String()
}

// DedupSources validates local fixture metadata and returns expansion, weighted
// expert, reduction and token-combine sources. No remote transport is emitted.
func DedupSources(s DedupSpec) ([4]DedupKernel, error) {
	m, err := dedupAdmit(s)
	if err != nil {
		return [4]DedupKernel{}, err
	}
	head := fmt.Sprintf("#define WIDTH %d\n#define SLOTS %d\n", s.Width, s.Slots) + dedupArray("primary", m.primary) + dedupArray("token", m.token) + dedupArray("expert", m.expert)
	bodies := []struct {
		name, body string
		args       []string
	}{
		{"dedup_expand", `const float *input,float *out,int count) {
    unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(count<=0||i>=(unsigned int)count)return;
    unsigned int row=i/WIDTH,c=i%WIDTH;
    out[i]=primary[row]<0?0.0f:input[primary[row]*WIDTH+c];
}`, []string{"input", "out"}},
		{"dedup_expert", `const float *input,const float *weight,float *out,int count) {
    unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(count<=0||i>=(unsigned int)count)return;
    unsigned int row=i/WIDTH;
    out[i]=primary[row]<0?0.0f:(input[i]*(float)(expert[row]+1)+(float)expert[row]*0.125f)*weight[row];
}`, []string{"input", "weight", "out"}},
		{"dedup_reduce", `const float *input,float *out,int count) {
    unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(count<=0||i>=(unsigned int)count)return;
    unsigned int row=i/WIDTH,c=i%WIDTH;
    float v=0.0f;
    if(primary[row]==(int)row)for(int j=0;j<SLOTS;++j)if(primary[j]==(int)row)v+=input[j*WIDTH+c];
    out[i]=v;
}`, []string{"input", "out"}},
		{"dedup_combine", `const float *input,float *out,int count) {
    unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(count<=0||i>=(unsigned int)count)return;
    unsigned int t=i/WIDTH,c=i%WIDTH;
    float v=0.0f;
    for(int j=0;j<SLOTS;++j)if(primary[j]==j&&token[j]==(int)t)v+=input[j*WIDTH+c];
    out[i]=v;
}`, []string{"input", "out"}},
	}
	var out [4]DedupKernel
	for i, b := range bodies {
		out[i] = DedupKernel{b.name, head + "extern \"C\" __global__ void " + b.name + "(" + b.body, dedupSchema(b.args...)}
	}
	return out, nil
}

// CompileDedup compiles the four local FP32 stages without initializing CUDA.
// Zero artifact resource fields mean unknown, not a measured zero requirement.
func CompileDedup(s DedupSpec, major, minor int) ([4]*cudakernel.Artifact, error) {
	sources, err := DedupSources(s)
	if err != nil {
		return [4]*cudakernel.Artifact{}, err
	}
	var out [4]*cudakernel.Artifact
	for i, k := range sources {
		cm, e := nvrtcx.Compile(k.Source, nvrtcx.WithArch(major, minor), nvrtcx.WithFlag("--fmad=false"))
		if e != nil {
			return out, e
		}
		out[i], e = cm.ToArtifact(k.Name, k.Schema, fmt.Sprintf("compute_%d%d", major, minor))
		if e != nil {
			return out, e
		}
	}
	return out, nil
}
