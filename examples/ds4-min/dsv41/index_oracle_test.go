package dsv41

import (
	"math"
	"strings"
	"testing"
)

// TestIndexScoresPinnedC evaluates the unmodified kernel for lane zero. The
// launcher supplies each shuffle's neighbour from an independent lane tree.
// This qualifies scalar values, not GPU warp execution or cache residency.
func TestIndexScoresPinnedC(t *testing.T) {
	f := literalFunction(t, pinned41(t), "dsv41_index_scores_kernel")
	prefix := strings.Replace(operationPrefix, "unsigned x;", "unsigned x,y;", 1) + `
static float neighbours[32][5];static unsigned head,stage;
static float shuffle(float v,unsigned stride){float n=neighbours[head][stage++];if(stage==5){stage=0;head++;}return n;}
#define __shfl_down_sync(mask,v,stride) shuffle(v,stride)
`
	const main = `int main(){float q[4096],w[32],k[384],scores[3];if(fread(q,4,4096,stdin)!=4096||fread(w,4,32,stdin)!=32||fread(k,4,384,stdin)!=384)return 2;
for(unsigned key=0;key<3;key++){for(unsigned h=0;h<32;h++){float lanes[32];for(unsigned l=0;l<32;l++){const float*x=q+h*128+l,*y=k+key*128+l;float a=__fadd_rn(__fmul_rn(x[0],y[0]),__fmul_rn(x[64],y[64]));float b=__fadd_rn(__fmul_rn(x[32],y[32]),__fmul_rn(x[96],y[96]));lanes[l]=__fadd_rn(a,b);}unsigned s=0;for(unsigned d=16;d;d/=2){neighbours[h][s++]=lanes[d];for(unsigned l=0;l<d;l++)lanes[l]+=lanes[l+d];}}
head=stage=0;blockIdx.x=key/8;blockIdx.y=0;threadIdx.x=(key%8)*32;dsv41_index_scores_kernel(scores,q,w,k,3,5,2);}
return fwrite(scores,4,3,stdout)==3?0:3;}
`
	q, weights, keys := make([]float32, 4096), make([]float32, 32), make([]float32, 384)
	for i := range q {
		q[i] = BF16(float32(math.Sin(float64(i)*.27)) * 2)
	}
	for i := range keys {
		keys[i] = BF16(float32(math.Cos(float64(i) * .11)))
	}
	for i := range weights {
		weights[i] = float32(i+1) / 32
	}
	want, err := indexScores(q, weights, keys)
	if err != nil {
		t.Fatal(err)
	}
	input := append(append(append([]float32(nil), q...), weights...), keys...)
	if !equalBits(runLiteral(t, prefix+f+main, input), want) {
		t.Fatal("index score literal mismatch")
	}
	mutant := strings.Replace(f, "weights[(uint64_t)row * 32u + h]", "weights[(uint64_t)row * 32u]", 1)
	if mutant == f {
		t.Fatal("index head weight mutant missing")
	}
	if equalBits(runLiteral(t, prefix+mutant+main, input), want) {
		t.Fatal("index head weight mutant not detected")
	}
}
