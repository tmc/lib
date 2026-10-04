package cudaattn

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"github.com/tmc/lib/nvidia/driver"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudablock"
)

const indexedShim = `
#include <cmath>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <vector>
#define __global__
#define __device__
#define __forceinline__ inline
struct {int x;} blockIdx,threadIdx;
unsigned __float_as_uint(float x){unsigned u;std::memcpy(&u,&x,4);return u;}
float __int_as_float(int x){float f;std::memcpy(&f,&x,4);return f;}
float __fmul_rn(float a,float b){volatile float x=a*b;return x;}
float __fadd_rn(float a,float b){volatile float x=a+b;return x;}
template<class T> std::vector<T> read(int n){std::vector<T> v(n);for(T& x:v)std::cin>>x;return v;}
template<class T> void print(const std::vector<T>& v){for(T x:v)std::cout<<x<<" ";std::cout<<"\n";}
`
const indexedShimMain = `
int main(){
 int batch,queries,qheads,kvheads,dim,bs,ms,phys,selected,shared,causal,pipeline;float scale;
 std::cin>>batch>>queries>>qheads>>kvheads>>dim>>bs>>ms>>phys>>selected>>shared>>causal>>scale>>pipeline;
 int rows=queries*qheads,blocks=(ms+bs-1)/bs,sets=queries*(shared?kvheads:qheads);
 auto q=read<float>(rows*dim),k=read<float>(phys*bs*kvheads*dim),v=read<float>(k.size());
 auto seq=read<int>(queries),pos=read<int>(queries),lens=read<int>(batch),pages=read<int>(batch*blocks),ids=read<int>(sets*selected),counts=read<int>(sets);
 std::vector<int> status(rows),sl(rows),so(rows),ts(rows);std::vector<float> out(rows*dim,-987),lse(rows,-987),scores(rows*blocks);
 auto run=[&](int mode){for(int row=0;row<rows;row++){blockIdx.x=row;indexed_attention(q.data(),k.data(),v.data(),seq.data(),pos.data(),lens.data(),pages.data(),ids.data(),counts.data(),status.data(),out.data(),lse.data(),scores.data(),sl.data(),so.data(),batch,queries,qheads,kvheads,dim,bs,ms,phys,selected,shared,causal,scale,mode);}};
 auto good=[&](){for(int s:status)if(s)return false;return true;};
 run(pipeline?0:1);
 if(good() && pipeline){run(3);if(good()){for(int row=0;row<rows;row++){blockIdx.x=row;topk_fp32(scores.data(),sl.data(),so.data(),ids.data(),nullptr,counts.data(),ts.data(),blocks,blocks,selected,rows,0);}for(int row=0;row<rows;row++)if(ts[row])status[row]=3;if(good())run(1);}}
 if(good())run(2);
 std::cout<<std::setprecision(9);print(status);print(out);print(lse);print(ids);print(counts);print(scores);
}
`

type indexedEmulation struct {
	Status              []int32
	Output, LSE, Scores []float32
	IDs, Counts         []int32
}

func indexedEmulator(t *testing.T, source string) func(indexedFixture) indexedEmulation {
	t.Helper()
	compiler, err := exec.LookPath("clang++")
	if err != nil {
		t.Fatal("host CUDA arithmetic controls require clang++:", err)
	}
	dir := t.TempDir()
	src, bin := filepath.Join(dir, "indexed.cc"), filepath.Join(dir, "indexed")
	if err := os.WriteFile(src, []byte(indexedShim+source+cudablock.TopKSource+indexedShimMain), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if output, err := exec.CommandContext(ctx, compiler, "-std=c++17", "-O0", "-ffp-contract=off", src, "-o", bin).CombinedOutput(); err != nil {
		t.Fatalf("compile host arithmetic control: %v\n%s", err, output)
	}
	return func(f indexedFixture) indexedEmulation {
		t.Helper()
		c, r := f.Config, f.Request
		e, _ := c.Extents()
		var input bytes.Buffer
		shared, causal, pipeline := 0, 0, 0
		if c.Selection == IndexedKVHeads {
			shared = 1
		}
		if c.Causal {
			causal = 1
		}
		if f.Pipeline {
			pipeline = 1
		}
		fmt.Fprintln(&input, c.Batch, c.Queries, c.QueryHeads, c.KVHeads, c.HeadDim, c.BlockSize, c.MaxSequence, c.PhysicalBlocks, c.SelectedBlocks, shared, causal, c.Scale, pipeline)
		for _, v := range [][]float32{f.Q, f.K, f.V} {
			for _, x := range v {
				fmt.Fprint(&input, x, " ")
			}
		}
		for _, v := range [][]int32{r.Sequences, r.Positions, r.Lengths, r.Pages, r.IDs, r.Counts} {
			for _, x := range v {
				fmt.Fprint(&input, x, " ")
			}
		}
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		cmd := exec.CommandContext(ctx, bin)
		cmd.Stdin = &input
		output, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("run host arithmetic control: %v\n%s", err, output)
		}
		got := indexedEmulation{Status: make([]int32, e.Rows), Output: make([]float32, e.Query), LSE: make([]float32, e.Rows), IDs: make([]int32, e.IDs), Counts: make([]int32, e.Sets), Scores: make([]float32, e.Scores)}
		reader := bytes.NewReader(output)
		for _, v := range [][]int32{got.Status} {
			for i := range v {
				if _, err := fmt.Fscan(reader, &v[i]); err != nil {
					t.Fatal(err)
				}
			}
		}
		for _, v := range [][]float32{got.Output, got.LSE} {
			for i := range v {
				if _, err := fmt.Fscan(reader, &v[i]); err != nil {
					t.Fatalf("read control: %v: %s", err, output)
				}
			}
		}
		for _, v := range [][]int32{got.IDs, got.Counts} {
			for i := range v {
				if _, err := fmt.Fscan(reader, &v[i]); err != nil {
					t.Fatal(err)
				}
			}
		}
		for i := range got.Scores {
			if _, err := fmt.Fscan(reader, &got.Scores[i]); err != nil {
				t.Fatal(err)
			}
		}
		return got
	}
}

func indexedRunHost(t *testing.T, fixture int) {
	t.Helper()
	f := indexedFixtures()[fixture]
	if err := ValidateIndexedRequest(f.Config, f.Request); err != nil {
		t.Fatal(err)
	}
	got := indexedEmulator(t, IndexedSource)(f)
	e, _ := f.Config.Extents()
	if err := CheckIndexedStatus(got.Status, e.Rows); err != nil {
		t.Fatal(err)
	}
	indexedCheckResult(t, f, IndexedSource, "host arithmetic control, no GPU qualification", got.Output, got.LSE)
}
func TestIndexedAttentionRestrictedDenseReference(t *testing.T) { indexedRunHost(t, 0) }
func TestIndexedAttentionAllBlocks(t *testing.T)                { indexedRunHost(t, 1) }
func TestIndexedGQASelectionSets(t *testing.T) {
	indexedRunHost(t, 0)
	indexedRunHost(t, 3)
	c := indexedFixtures()[0].Config
	c.Selection = 0
	if _, err := c.Extents(); err == nil {
		t.Fatal("ambiguous heads admitted")
	}
	c.Selection = IndexedKVHeads
	if _, err := IndexedTopKConfig(c); err == nil {
		t.Fatal("per-head proxy silently shared")
	}
}
func TestIndexedCausalOffsetAndTail(t *testing.T) { indexedRunHost(t, 0) }
func TestIndexedEmptySelection(t *testing.T)      { indexedRunHost(t, 2); indexedRunHost(t, 4) }
func TestIndexedSelectionPipeline(t *testing.T) {
	f := indexedFixtures()[5]
	got := indexedEmulator(t, IndexedSource)(f)
	scores, ids, counts := indexedProxyOracle(f)
	if err := indexedCompare(f, IndexedSource, "host arithmetic control", got.Scores, scores, true); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.IDs, ids) || !reflect.DeepEqual(got.Counts, counts) {
		t.Fatalf("selection differs: ids %v want %v counts %v want %v", got.IDs, ids, got.Counts, counts)
	}
	f.Request.IDs, f.Request.Counts = got.IDs, got.Counts
	indexedCheckResult(t, f, IndexedSource, "host arithmetic control", got.Output, got.LSE)
}

func indexedBadRequests() []struct {
	name string
	edit func(*indexedFixture)
} {
	return []struct {
		name string
		edit func(*indexedFixture)
	}{
		{"duplicate", func(f *indexedFixture) { f.Request.IDs[1] = f.Request.IDs[0] }},
		{"negative-live", func(f *indexedFixture) { f.Request.IDs[0] = -1 }},
		{"range", func(f *indexedFixture) { f.Request.IDs[0] = 3 }},
		{"padding", func(f *indexedFixture) { f.Request.IDs[3] = 0 }},
		{"count", func(f *indexedFixture) { f.Request.Counts[0] = 5 }},
		{"page", func(f *indexedFixture) { f.Request.Pages[0] = 7 }},
		{"unused-page", func(f *indexedFixture) { f.Request.Pages[5] = 0 }},
		{"sequence", func(f *indexedFixture) { f.Request.Sequences[0] = 2 }},
		{"position", func(f *indexedFixture) { f.Request.Positions[0] = 8 }},
		{"length", func(f *indexedFixture) { f.Request.Lengths[0] = 9 }},
	}
}
func TestIndexedIDsAdmission(t *testing.T) {
	run := indexedEmulator(t, IndexedSource)
	for _, tt := range indexedBadRequests() {
		t.Run(tt.name, func(t *testing.T) {
			f := indexedFixtures()[0]
			tt.edit(&f)
			if err := ValidateIndexedRequest(f.Config, f.Request); err == nil {
				t.Fatal("bad request admitted")
			}
			got := run(f)
			if err := CheckIndexedStatus(got.Status, len(got.Status)); err == nil {
				t.Fatal("device validation admitted damage")
			}
			for row, s := range got.Status {
				if s != 0 {
					for _, x := range got.Output[row*f.Config.HeadDim : (row+1)*f.Config.HeadDim] {
						if x != -987 {
							t.Fatal("invalid validation wrote compute output")
						}
					}
				}
			}
		})
	}
	if CheckIndexedStatus(nil, 1) == nil || CheckIndexedStatus([]int32{9}, 1) == nil {
		t.Fatal("missing/unknown evidence accepted")
	}
	f := indexedFixtures()[0]
	f.Request.Sequences[1] = 0
	f.Request.Pages[3] = int32(f.Config.PhysicalBlocks)
	if ValidateIndexedRequest(f.Config, f.Request) == nil {
		t.Fatal("unused sequence page damage admitted on host")
	}
	got := run(f)
	if CheckIndexedStatus(got.Status, len(got.Status)) == nil {
		t.Fatal("unused sequence page damage admitted on device validation")
	}
}

func indexedMutants() []struct {
	name, old, new string
	fixture        int
} {
	return []struct {
		name, old, new string
		fixture        int
	}{
		{"sentinel-aliases-zero", "int visits=mode==3 ? blocks : counts[set];", "int visits=mode==3 ? blocks : selected;", 2},
		{"duplicate-double-count", "for(int a=0;a<j;a++) if(ids[set*selected+a]==id){status[row]=1;return;}", "", 0},
		{"compacted-causal-position", "int actual=logical*block_size+t;", "int actual=j*block_size+t;", 0},
		{"wrong-query-head-set", "int set = query*set_heads + (shared_sets ? kvhead : head);", "int set = query*set_heads + (shared_sets ? kvhead : kvhead);", 0},
		{"missing-normalizer", "u[d]/l", "u[d]", 0},
		{"inverse-rescaling", "expf(m-next)", "expf(next-m)", 0},
	}
}
func TestIndexedBehavioralMutants(t *testing.T) {
	for _, m := range indexedMutants() {
		t.Run(m.name, func(t *testing.T) {
			if strings.Count(IndexedSource, m.old) != 1 {
				t.Fatal("mutant anchor drift")
			}
			src := strings.Replace(IndexedSource, m.old, m.new, 1)
			f := indexedFixtures()[m.fixture]
			if m.name == "sentinel-aliases-zero" {
				src = strings.Replace(src, "int physical=pages[seq*max_blocks+logical];", "logical=logical<0?0:logical; int physical=pages[seq*max_blocks+logical];", 1)
			}
			if m.name == "duplicate-double-count" {
				f.Request.IDs[1] = f.Request.IDs[0]
			}
			got := indexedEmulator(t, src)(f)
			want, lse := indexedOracle(f)
			err := indexedCompare(f, src, "host source mutant", got.Output, want, false)
			lerr := indexedCompare(f, src, "host source mutant", got.LSE, lse, true)
			if err == nil && lerr == nil {
				t.Fatal("actual source mutant survived common numerical policy")
			}
			t.Logf("killed: output=%v lse=%v", err, lerr)
		})
	}
}

func TestIndexedBufferAdmission(t *testing.T) {
	dev := new(cuda.Device)
	released := 0
	v := indexedBuffer{name: "a", ptr: 4, length: 1, need: 1, owner: dev, borrow: func(*cuda.Device) (func(), error) { return func() { released++ }, nil }}
	if _, err := indexedBorrow([]indexedBuffer{v, {name: "b", ptr: 8, length: 1, need: 1, owner: dev, borrow: func(*cuda.Device) (func(), error) { return nil, fmt.Errorf("refused") }}}, dev); err == nil || released != 1 {
		t.Fatal("partial borrow leaked")
	}
	if indexedCheckBuffers([]indexedBuffer{v, v}, dev) == nil {
		t.Fatal("alias accepted")
	}
	v.owner = new(cuda.Device)
	if indexedCheckBuffers([]indexedBuffer{v}, dev) == nil {
		t.Fatal("foreign owner accepted")
	}
	var zero IndexedAttention
	if zero.Execute(nil, nil, IndexedBuffers{}) == nil {
		t.Fatal("zero executable admitted")
	}
	if zero.Close() != nil {
		t.Fatal("zero close failed")
	}
}

func TestIndexedPhaseFailure(t *testing.T) {
	var owner IndexedAttention
	var observed []string
	owner.observe = func(name string, start bool) error { observed = append(observed, fmt.Sprint(name, start)); return nil }
	want := fmt.Errorf("actual action failed")
	if err := owner.phase("score", func() error { return want }); !errors.Is(err, want) {
		t.Fatal("phase hid action failure")
	}
	if !reflect.DeepEqual(observed, []string{"scoretrue", "scorefalse"}) {
		t.Fatal(observed)
	}
	called := false
	owner.observe = func(string, bool) error { return want }
	if err := owner.phase("score", func() error { called = true; return nil }); !errors.Is(err, want) || called {
		t.Fatal("observer start failure still submitted")
	}
}

func TestIndexedLaunchFourByteViews(t *testing.T) {
	schema, _ := IndexedSchema()
	c := indexedFixtures()[0].Config
	var ptrs [15]driver.DevicePtr
	for i := range ptrs {
		ptrs[i] = driver.DevicePtr(0x1004 + i*0x100)
	}
	args, err := indexedLaunchArguments(schema, c, ptrs, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(args) != 28 {
		t.Fatal("wrong pointer-array ABI count")
	}
	for i, p := range ptrs {
		if *(*driver.DevicePtr)(args[i]) != p {
			t.Fatalf("pointer %d differs", i)
		}
	}
	for i, want := range []int32{2, 3, 4, 2, 7, 3, 8, 7, 4, 0, 1} {
		if *(*int32)(args[15+i]) != want {
			t.Fatalf("scalar slot %d differs", i)
		}
	}
	if *(*float32)(args[26]) != c.Scale || *(*int32)(args[27]) != 2 {
		t.Fatal("scalar slots differ")
	}
	ptrs[0]++
	if _, err := indexedLaunchArguments(schema, c, ptrs, 2); err == nil {
		t.Fatal("unaligned FP32 pointer admitted")
	}
	ptrs[0]--
	schema.Params[0].Offset++
	if _, err := indexedLaunchArguments(schema, c, ptrs, 2); err == nil {
		t.Fatal("damaged ABI admitted")
	}
}

func TestIndexedConfigAdmission(t *testing.T) {
	for _, edit := range []func(*IndexedConfig){
		func(c *IndexedConfig) { c.Batch = 0 }, func(c *IndexedConfig) { c.Queries = int(^uint(0) >> 1) },
		func(c *IndexedConfig) { c.QueryHeads = 3 }, func(c *IndexedConfig) { c.KVHeads = 0 },
		func(c *IndexedConfig) { c.HeadDim = 129 }, func(c *IndexedConfig) { c.SelectedBlocks = 0 },
		func(c *IndexedConfig) { c.BlockSize = -1 }, func(c *IndexedConfig) { c.MaxSequence = 4097 },
		func(c *IndexedConfig) { c.PhysicalBlocks = 257 }, func(c *IndexedConfig) { c.Scale = 0 },
		func(c *IndexedConfig) { c.Scale = float32(math.NaN()) }, func(c *IndexedConfig) { c.Scale = float32(math.Inf(1)) },
	} {
		c := indexedFixtures()[0].Config
		edit(&c)
		if _, err := c.Extents(); err == nil {
			t.Fatal("invalid extent/precision contract admitted", c)
		}
	}
}

func TestIndexedInputIdentity(t *testing.T) {
	f := indexedFixtures()[0]
	before := indexedInputDigest(f)
	f.Config.Scale = 0.5
	if indexedInputDigest(f) == before {
		t.Fatal("scale absent from input identity")
	}
	f = indexedFixtures()[0]
	f.Config.Causal = false
	if indexedInputDigest(f) == before {
		t.Fatal("causal policy absent from input identity")
	}
	f = indexedFixtures()[0]
	f.Config.Selection = IndexedKVHeads
	if indexedInputDigest(f) == before {
		t.Fatal("head-selection ownership absent from input identity")
	}
}

func FuzzIndexedIDsAdmission(f *testing.F) {
	f.Add(int32(0), int32(2), int32(6))
	f.Add(int32(-1), int32(-1), int32(-1))
	f.Add(int32(2147483647), int32(2147483647), int32(2147483647))
	f.Fuzz(func(t *testing.T, id, count, position int32) {
		fixture := indexedFixtures()[0]
		fixture.Request.IDs[0] = id
		fixture.Request.Counts[0] = count
		fixture.Request.Positions[0] = position
		err := ValidateIndexedRequest(fixture.Config, fixture.Request)
		if err == nil && (id < 0 || id >= 3 || id == 2 || count != 2 || position < -1 || position >= 8) {
			t.Fatal("malformed indexed request survived", id, count, position)
		}
	})
}
