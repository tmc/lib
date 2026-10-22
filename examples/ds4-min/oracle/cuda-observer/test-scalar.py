from pathlib import Path
import subprocess,re,os
root=Path.cwd();p=Path(os.environ["DS4_OBSERVER_TEST_DIR"])
c=Path('ds4_cuda.cu').read_text();h=Path('ds4_gpu.h').read_text();base=subprocess.check_output(['git','show','HEAD:ds4_cuda.cu'],text=True)
def section(s,a,b): return s[s.index(a):s.index(b,s.index(a))]
for start,end in [('__global__ static void attention_prefill_raw_kernel(', '/* Mirrored in ds4_cuda.cu:'),('extern "C" int ds4_gpu_attention_prefill_raw_heads_tensor(', 'extern "C" int ds4_gpu_attention_prefill_raw_heads_tensor_observed(')]:
    chunk=section(c,start,end)
    assert chunk in base
assert section(c,'__global__ static void add_kernel(', '__global__ static void directional_steering_project_kernel')==section(base,'__global__ static void add_kernel(', '__global__ static void directional_steering_project_kernel')
abi=section(c,'/* Mirrored in ds4_cuda.cu:', 'static_assert(sizeof(ds4_gpu_raw_observer)')
assert abi.strip() in h
old=section(base,'__global__ static void attention_prefill_raw_kernel(', '__global__ static void attention_prefill_mixed_kernel(')
new=section(c,'__global__ static void attention_prefill_raw_observed_kernel(', '__global__ static void attention_prefill_mixed_kernel(')
new=new.replace('attention_prefill_raw_observed_kernel(', 'attention_prefill_raw_kernel(').replace('uint32_t head_dim, ds4_gpu_raw_observer observer)', 'uint32_t head_dim)')
new=new.replace('''for (uint32_t d = 0; d < head_dim; d++) {
            float value = kv[d];
            if (t == observer.token && h == observer.head &&
                raw_start + r == observer.row && d == observer.dim)
                cuda_observe_raw(observer, __float_as_uint(value));
            dot += qh[d] * value;
        }''','for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kv[d];')
assert old==new
wrapper=section(c,'extern "C" int ds4_gpu_attention_prefill_raw_heads_tensor_observed(', 'extern "C" int ds4_gpu_attention_prefill_raw_heads_tensor_window_observed(')
original=section(base,'extern "C" int ds4_gpu_attention_prefill_raw_heads_tensor(', 'static int attention_decode_batch_launch(')
for start,end in [('    if (n_tokens > 1','        dim3 grid'),('    if (g_cublas_ready','        const uint32_t n_keys')]:
    predicate=section(original,start,end)
    assert predicate in wrapper
assert wrapper.count('return ds4_gpu_attention_prefill_raw_heads_tensor(')==3
helper=section(c,'__device__ static void cuda_observe_raw(', '__global__ static void attention_prefill_raw_observed_kernel(')
assert not re.search(r'cuda(?:Memcpy|DeviceSynchronize|StreamSynchronize|LaunchHostFunc)',helper+wrapper)
assert subprocess.check_output(['git','diff','--','ds4.c'])==(p/'pre-existing.patch').read_bytes()
# Compile the public ABI as C and C++ independently of CUDA headers.
header_test='#include "ds4_gpu.h"\n#include <stddef.h>\n_Static_assert(sizeof(ds4_gpu_raw_observer)==56,"size");\n_Static_assert(offsetof(ds4_gpu_raw_observer,buffer)==40,"buffer");\n_Static_assert(sizeof(ds4_gpu_raw_observer_state)==8,"state");\n_Static_assert(sizeof(ds4_gpu_raw_observer_record)==32,"record");\n'
(p/'abi.c').write_text(header_test)
subprocess.run(['cc','-std=c11','-Wall','-Wextra','-Werror','-I',str(root),'-fsyntax-only',str(p/'abi.c')],check=True)
(p/'abi.cc').write_text(header_test.replace('_Static_assert','static_assert'))
subprocess.run(['c++','-std=c++11','-Wall','-Wextra','-Werror','-I',str(root),'-fsyntax-only',str(p/'abi.cc')],check=True)
# Synthetic CPU control of the actual extracted append/validation source.
test='''#include "ds4_gpu.h"
#include <cassert>
#include <cstring>
#include <thread>
#include <vector>
#include <cstdio>
#define __device__
static uint32_t atomicAdd(uint32_t *p,uint32_t v){return __atomic_fetch_add(p,v,__ATOMIC_SEQ_CST);}
static uint32_t atomicOr(uint32_t *p,uint32_t v){return __atomic_fetch_or(p,v,__ATOMIC_SEQ_CST);}
static uint32_t atomicCAS(uint32_t *p,uint32_t a,uint32_t b){__atomic_compare_exchange_n(p,&a,b,false,__ATOMIC_SEQ_CST,__ATOMIC_SEQ_CST);return a;}
'''+helper+'''
int main(){
 alignas(8) unsigned char storage[8+8*40+16]; memset(storage,0,sizeof storage);
 memset(storage+8+8*40,0xa5,16);
 ds4_gpu_raw_observer o={1,56,1,8,40,2,3,4,5,0,(uint64_t)(uintptr_t)storage,8+8*40};
 assert(cuda_raw_observer_valid(&o));
 #define BAD(field,val) {auto bad=o;bad.field=val;assert(!cuda_raw_observer_valid(&bad));}
 BAD(version,2);BAD(size,55);BAD(reserved,1);BAD(buffer,0);BAD(buffer,o.buffer+1);
 BAD(stride,28);BAD(stride,33);BAD(bytes,7);BAD(bytes,o.bytes-1);
 BAD(capacity,UINT32_MAX);BAD(buffer,UINT64_MAX-3);
 auto *state=(ds4_gpu_raw_observer_state*)storage;
 cuda_observe_raw(o,0x3f800000);
 assert(state->write_index==1 && state->overflow==0);
 auto *r=(ds4_gpu_raw_observer_record*)(storage+8);
 assert(r->version==1 && r->size==32 && r->site==1 && r->token==2 && r->head==3 && r->row==4 && r->dim==5 && r->value_bits==0x3f800000);
 std::vector<std::thread> threads;
 for(int i=0;i<32;i++) threads.emplace_back([=]{cuda_observe_raw(o,0x40000000);});
 for(auto &t:threads)t.join();
 assert(state->write_index==8 && state->overflow==1);
 for(int i=0;i<16;i++) assert(storage[8+8*40+i]==0xa5);
 for(int i=0;i<8;i++) for(int j=32;j<40;j++) assert(storage[8+i*40+j]==0);
 state->write_index=UINT32_MAX;cuda_observe_raw(o,0);assert(state->write_index==UINT32_MAX);
 o.capacity=0;state->write_index=0;state->overflow=0;cuda_observe_raw(o,0);
 assert(state->write_index==0 && state->overflow==1);
 puts("PASS synthetic CPU append controls (not CUDA qualification)");
}
'''
(p/'append.cc').write_text(test)
cmd=['c++','-std=c++11','-Wall','-Wextra','-Werror','-pthread','-I',str(root),str(p/'append.cc'),'-o',str(p/'append')]
subprocess.run(cmd,check=True);subprocess.run([str(p/'append')],check=True)
# A capacity-validation mutant must be rejected by these same assertions.
mutant=test.replace('(uint64_t)o->capacity * o->stride <=','0 <=')
(p/'mutant.cc').write_text(mutant)
cmd[cmd.index(str(p/'append.cc'))]=str(p/'mutant.cc');cmd[-1]=str(p/'mutant')
subprocess.run(cmd,check=True)
result=subprocess.run([str(p/'mutant')],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
assert result.returncode != 0
# Source-derived selector: same-site positive, adjacent coordinate negatives.
selector=re.search(r'if \(t == observer.token && h == observer.head &&\s*raw_start \+ r == observer.row && d == observer.dim\)',c)
assert selector
expr=selector.group()[4:-1].replace('&&','and').replace('observer.token','2').replace('observer.head','3').replace('observer.row','4').replace('observer.dim','5')
expr=' '.join(expr.split())
coords=dict(t=2,h=3,raw_start=1,r=3,d=5)
assert eval(expr,{},coords)
for key in ('t','h','r','d'):
    adjacent=dict(coords);adjacent[key]+=1
    assert not eval(expr,{},adjacent)
print('PASS source-derived same-site and adjacent-coordinate selector controls; not device execution')
print('PASS C/C++ ABI, source preservation, route predicates, bounded append, overflow, guard bytes, concurrency, validation mutant')
