from pathlib import Path
import subprocess,re,os
p=Path(os.environ["DS4_OBSERVER_TEST_DIR"])
c=Path('ds4_cuda.cu').read_text();old=(p/'ds4_cuda.cu').read_text();h=Path('ds4_gpu.h').read_text()
def section(s,a,b): return s[s.index(a):s.index(b,s.index(a))]
k=section(c,'__global__ static void attention_static_mixed_heads8_online_kernel(', '__global__ static void attention_raw_window_observed_kernel(')
assert k in old
new=section(c,'__global__ static void attention_raw_window_observed_kernel(', '/* -------------------------------------------------------------------------\n * DS4_ATTN_TOKENTILE:')
new=new.replace('attention_raw_window_observed_kernel(', 'attention_static_mixed_heads8_online_kernel(').replace('uint32_t head_dim, ds4_gpu_raw_observer observer)', 'uint32_t head_dim)')
a=new.index('            const float4 value = src[c4];');b=new.index('            kv_shared[off] = value;',a)+len('            kv_shared[off] = value;')
new=new[:a]+'            kv_shared[off] = src[c4];'+new[b:]
assert new==k
scalar=section(c,'extern "C" int ds4_gpu_attention_prefill_raw_heads_tensor_observed(', 'extern "C" int ds4_gpu_attention_prefill_raw_heads_tensor_window_observed(')
assert scalar in old
wrapper=section(c,'extern "C" int ds4_gpu_attention_prefill_raw_heads_tensor_window_observed(', 'static int attention_decode_batch_launch(')
assert wrapper.count('return ds4_gpu_attention_prefill_raw_heads_tensor(')==2
assert wrapper.count('<<<')==1
predicate=section(wrapper,'    if (!(n_tokens',' {').replace('    if (!(', '').rstrip(')')
assert predicate in old
assert 'n_tokens, 0, window, 1, n_head, head_dim,' in wrapper
assert not re.search(r'cuda(?:Memcpy|DeviceSynchronize|StreamSynchronize|LaunchHostFunc)',wrapper)
# Exhaustive producer-lane enumeration for source-derived window selector.
selector=re.search(r'if \(sr < raw_count && t == observer.token &&\s*observer.head < n_head && head_group == observer.head / 8u &&\s*raw_start \+ sr == observer.row && c4 == observer.dim / 4u &&\s*observer.dim < head_dim\)',c).group()[4:-1]
expr=' '.join(selector.replace('&&','and').replace('/ 8u','// 8').replace('/ 4u','// 4').replace('observer.','obs_').split())
def count(token,head,row,dim,actual_token=129,window=128,n_head=10):
    hits=[];raw_count=min(actual_token+1,window) if window else actual_token+1;raw_start=actual_token+1-raw_count
    for group in range((n_head+7)//8):
      for row0 in range(0,raw_count,4):
       nr=min(4,raw_count-row0)
       for thread in range(256):
        for off in range(thread,nr*128,256):
         sr=row0+(off>>7);c4=off&127
         env=dict(sr=sr,raw_count=raw_count,t=actual_token,obs_token=token,obs_head=head,n_head=n_head,head_group=group,raw_start=raw_start,obs_row=row,obs_dim=dim,c4=c4,head_dim=512)
         if eval(expr,{},env): hits.append((group,raw_start+sr,c4,dim%4))
    return hits
for dim in (0,1,2,3,4,511):
    assert count(129,9,2,dim)==[(1,2,dim//4,dim%4)]
for args in ((128,9,2,0),(129,10,2,0),(129,9,1,0),(129,9,130,0),(129,9,2,512)):
    assert not count(*args)
assert count(129,8,129,3)==count(129,9,129,3)
assert count(1,0,0,0,actual_token=1,window=0)==[(0,0,0,0)]
# Test shared append implementation's explicit site=2 and bit preservation.
s=(p/'append.cc').read_text().replace('cuda_observe_raw(o,0x3f800000);','cuda_observe_raw(o,0x3f800000,2u);').replace('r->site==1','r->site==2')
(p/'window-append.cc').write_text(s)
subprocess.run(['c++','-std=c++11','-Wall','-Wextra','-Werror','-pthread','-I',str(Path.cwd()),str(p/'window-append.cc'),'-o',str(p/'window-append')],check=True)
subprocess.run([str(p/'window-append')],check=True)
print('PASS original window/scalar source preservation, kernel delta, route predicate, synthetic producer selector and site 2 append controls')
