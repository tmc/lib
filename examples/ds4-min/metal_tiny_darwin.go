//go:build darwin && arm64

package main

import (
	"context"
	"fmt"
	"math"

	apple "github.com/tmc/apple/metal"
	"github.com/tmc/apple/objectivec"
	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
	"github.com/tmc/lib/apple/x/metal/metalx"
)

const (
	metalTinyStateFloats = 396
	metalTinyTraceFloats = 55
)

type metalTinyKernels struct {
	Step apple.MTLComputePipelineState `metal:"ds4_tiny_step"`
}

type metalTinyBackend struct {
	dev     *libmetal.Device
	lib     apple.MTLLibrary
	kernels metalTinyKernels
	mem     *metalmem.Arena
	weights metalmem.Slice[float32]
	state   metalmem.Slice[float32]
	out     metalmem.Slice[float32]
	pos     int
}

func (m *metalTinyBackend) DeviceName() string {
	if m == nil || m.dev == nil {
		return ""
	}
	return m.dev.Name()
}

func newMetalTinyBackend(seed uint64) (*metalTinyBackend, error) {
	dev, err := libmetal.NewDevice()
	if err != nil {
		return nil, err
	}
	m := &metalTinyBackend{dev: dev}
	failed := true
	defer func() {
		if failed {
			_ = m.Close()
		}
	}()
	m.lib, err = dev.CompileLibrary(metalTinyMSL)
	if err != nil {
		return nil, err
	}
	if err := metalx.Bind(dev, m.lib, &m.kernels); err != nil {
		return nil, err
	}
	m.mem = metalmem.NewArena(dev)
	m.weights = metalmem.UploadIn(m.mem, "weights", packTinyWeights(makeTinyWeights(seed)))
	m.state = metalmem.AllocIn[float32](m.mem, "state", metalTinyStateFloats)
	m.out = metalmem.AllocIn[float32](m.mem, "trace", metalTinyTraceFloats)
	if err := m.mem.Err(); err != nil {
		return nil, err
	}
	m.Reset()
	failed = false
	return m, nil
}

func (m *metalTinyBackend) Reset() {
	clear(m.state.HostSlice())
	for i := 204; i < 268; i++ {
		m.state.HostSlice()[i] = -math.MaxFloat32
	}
	for i := 332; i < 396; i++ {
		m.state.HostSlice()[i] = -math.MaxFloat32
	}
	m.pos = 0
}

func (m *metalTinyBackend) Step(token int) (tinyTrace, error) {
	if token < 0 || token >= tinyVocab {
		return tinyTrace{}, errShape
	}
	cb := m.dev.Queue().CommandBuffer()
	enc := cb.ComputeCommandEncoder()
	enc.SetComputePipelineState(m.kernels.Step)
	enc.SetBufferWithOffsetAtIndex(m.weights.Buffer(), m.weights.Offset(), 0)
	enc.SetBufferWithOffsetAtIndex(m.state.Buffer(), m.state.Offset(), 1)
	enc.SetBufferWithOffsetAtIndex(m.out.Buffer(), m.out.Offset(), 2)
	libmetal.SetScalar(enc, uint32(token), 3)
	libmetal.SetScalar(enc, uint32(m.pos), 4)
	enc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(1), libmetal.Threads(1))
	enc.EndEncoding()
	cb.Commit()
	if err := libmetal.WaitCommandBuffer(context.Background(), cb); err != nil {
		return tinyTrace{}, fmt.Errorf("metal tiny step: %w", err)
	}
	o := m.out.HostSlice()
	trace := tinyTrace{
		Token: token, Position: m.pos,
		Norm: cloneFloats(o[0:4]), HCAttnMix: cloneFloats(o[4:12]), HCFFNMix: cloneFloats(o[12:20]),
		Query: cloneFloats(o[20:24]), Key: cloneFloats(o[24:28]), Attention: cloneFloats(o[28:32]),
		Compressed: decodeMetalIDs(o[32:33]), Experts: decodeMetalIDs(o[33:35]), Weights: cloneFloats(o[35:37]),
		MoE: cloneFloats(o[37:41]), HC: cloneFloats(o[41:49]), Logits: cloneFloats(o[49:55]),
	}
	m.pos++
	return trace, nil
}

func (m *metalTinyBackend) Close() error {
	if m == nil {
		return nil
	}
	if m.mem != nil {
		m.mem.Free()
		m.mem = nil
	}
	if m.kernels.Step != nil && m.kernels.Step.GetID() != 0 {
		objectivec.ObjectFromID(m.kernels.Step.GetID()).Release()
		m.kernels.Step = nil
	}
	if m.lib != nil && m.lib.GetID() != 0 {
		objectivec.ObjectFromID(m.lib.GetID()).Release()
		m.lib = nil
	}
	if m.dev != nil {
		err := m.dev.Close()
		m.dev = nil
		return err
	}
	return nil
}

func cloneFloats(x []float32) []float32 { return append([]float32(nil), x...) }

func decodeMetalIDs(x []float32) []int {
	if len(x) == 1 && x[0] < 0 {
		return []int{}
	}
	out := make([]int, len(x))
	for i := range x {
		out[i] = int(x[i])
	}
	return out
}

func packTinyWeights(w tinyWeights) []float32 {
	out := make([]float32, 0, 639)
	add := func(x []float32) { out = append(out, x...) }
	add(w.embedding)
	add(w.norm)
	add(w.q)
	add(w.kv)
	add(w.o)
	add(w.router)
	add(w.routerBias)
	add(w.hcAttnFn)
	add(w.hcAttnScale)
	add(w.hcAttnBase)
	add(w.hcFFNFn)
	add(w.hcFFNScale)
	add(w.hcFFNBase)
	add(w.indexQuery)
	add(w.indexWeight)
	add(w.outputHead)
	add(w.outputNorm)
	for _, expert := range w.experts {
		add(expert.gate)
		add(expert.up)
		add(expert.down)
	}
	for _, compressor := range []compressorWeights{w.attention, w.indexer} {
		add(compressor.kv)
		add(compressor.gate)
		add(compressor.ape)
		add(compressor.norm)
	}
	out = append(out, w.attentionSink)
	return out
}

const metalTinyMSL = `
#include <metal_stdlib>
using namespace metal;

constant int D=4, S=2, E=4, I=3, R=4;
float sigm(float x) { return 1.0f/(1.0f+exp(-x)); }
void mv(device const float *w, thread const float *x, thread float *o, int rows, int cols) {
  for (int r=0;r<rows;r++) { float v=0; for(int c=0;c<cols;c++) v+=w[r*cols+c]*x[c]; o[r]=v; }
}
void rms(thread const float *x, device const float *w, thread float *o, int n) {
  float ss=0; for(int i=0;i<n;i++) ss+=x[i]*x[i]; float z=rsqrt(ss/float(n)+1e-5f);
  for(int i=0;i<n;i++) o[i]=x[i]*z*w[i];
}
void rms1(thread const float *x, thread float *o, int n) {
  float ss=0; for(int i=0;i<n;i++) ss+=x[i]*x[i]; float z=rsqrt(ss/float(n)+1e-5f);
  for(int i=0;i<n;i++) o[i]=x[i]*z;
}
void rope(thread float *x, int pos) {
  for(int i=0;i<D;i+=2) { float a=float(pos)/pow(10000.0f,float(i)/float(D)); float c=cos(a),s=sin(a),u=x[i],v=x[i+1]; x[i]=u*c-v*s; x[i+1]=u*s+v*c; }
}
void split(device const float *w, int fn, int sc, int ba, thread const float *state, thread float *mix, thread float *pre, thread float *post, thread float *comb) {
  float flat[8]; rms1(state,flat,8); mv(w+fn,flat,mix,8,8);
  for(int i=0;i<S;i++){ pre[i]=sigm(mix[i]*w[sc]+w[ba+i])+1e-6f; post[i]=2*sigm(mix[S+i]*w[sc+1]+w[ba+S+i]); }
  for(int d=0;d<S;d++){ float mx=-3.402823466e+38f; for(int s=0;s<S;s++){int j=s+d*S;comb[j]=mix[2*S+j]*w[sc+2]+w[ba+2*S+j];mx=max(mx,comb[j]);} float z=0;for(int s=0;s<S;s++){int j=s+d*S;comb[j]=exp(comb[j]-mx);z+=comb[j];}for(int s=0;s<S;s++){int j=s+d*S;comb[j]=comb[j]/z+1e-6f;} }
  for(int it=0;it<20;it++){ for(int s=0;s<S;s++){float z=0;for(int d=0;d<S;d++)z+=comb[s+d*S];for(int d=0;d<S;d++)comb[s+d*S]/=z+1e-6f;} if(it+1==20)break; for(int d=0;d<S;d++){float z=0;for(int s=0;s<S;s++)z+=comb[s+d*S];for(int s=0;s<S;s++)comb[s+d*S]/=z+1e-6f;} }
}
void weighted(thread const float *x, thread const float *a, thread float *o){for(int d=0;d<D;d++){float v=0;for(int s=0;s<S;s++)v+=x[s*D+d]*a[s];o[d]=v;}}
void posthc(thread const float *b,thread const float *r,thread const float *p,thread const float *c,thread float *o){for(int dst=0;dst<S;dst++)for(int d=0;d<D;d++){float v=b[d]*p[dst];for(int src=0;src<S;src++)v+=c[dst+src*S]*r[src*D+d];o[dst*D+d]=v;}}
void compress(device const float *w,int wo,device float *st,int ko,int so,thread const float *x,int pos,thread float *out){
  float kv[8],sg[8];mv(w+wo,x,kv,8,4);mv(w+wo+32,x,sg,8,4);int mod=pos%4,row=4+mod;for(int j=0;j<8;j++){sg[j]+=w[wo+64+mod*8+j];st[ko+row*8+j]=kv[j];st[so+row*8+j]=sg[j];}
  if((pos+1)%4) return; for(int j=0;j<4;j++){float mx=-3.402823466e+38f;for(int r=0;r<4;r++){mx=max(mx,st[so+r*8+j]);mx=max(mx,st[so+(4+r)*8+4+j]);}float den=0,sum=0;for(int r=0;r<4;r++){float a=exp(st[so+r*8+j]-mx),b=exp(st[so+(4+r)*8+4+j]-mx);den+=a+b;sum+=a*st[ko+r*8+j]+b*st[ko+(4+r)*8+4+j];}out[j]=sum/den;}
  float n[4];rms(out,w+wo+96,n,4);for(int j=0;j<4;j++)out[j]=n[j];rope(out,pos+1-4);
  for(int r=0;r<4;r++)for(int j=0;j<8;j++){st[ko+r*8+j]=st[ko+(4+r)*8+j];st[so+r*8+j]=st[so+(4+r)*8+j];}
  for(int r=0;r<4;r++)for(int j=0;j<8;j++){st[ko+(4+r)*8+j]=st[ko+r*8+j];st[so+(4+r)*8+j]=st[so+r*8+j];}
}
kernel void ds4_tiny_step(device const float *w [[buffer(0)]],device float *st [[buffer(1)]],device float *out [[buffer(2)]],constant uint &tok [[buffer(3)]],constant uint &upos [[buffer(4)]],uint tid [[thread_position_in_grid]]){
 if(tid) return; int pos=int(upos); float emb[4],hc[8];for(int d=0;d<4;d++)emb[d]=w[int(tok)*4+d];for(int s=0;s<2;s++)for(int d=0;d<4;d++)hc[s*4+d]=emb[d];
 float am[8],pre[2],post[2],comb[4];split(w,96,160,163,hc,am,pre,post,comb);for(int i=0;i<8;i++)out[4+i]=am[i];
 float plain[4],norm[4];weighted(hc,pre,plain);rms(plain,w+24,norm,4);for(int i=0;i<4;i++)out[i]=norm[i];
 float q[4],kv[4];mv(w+28,norm,q,4,4);mv(w+44,norm,kv,4,4);rope(q,pos);rope(kv,pos);for(int i=0;i<4;i++){out[20+i]=q[i];out[24+i]=kv[i];}
 int raw=int(st[0]);if(raw<2){for(int d=0;d<4;d++)st[2+raw*4+d]=kv[d];st[10+raw]=float(pos);raw++;}else{for(int d=0;d<4;d++)st[2+d]=st[6+d];st[10]=st[11];for(int d=0;d<4;d++)st[6+d]=kv[d];st[11]=float(pos);}st[0]=float(raw);
 float ac[4],ic[4];compress(w,438,st,140,204,norm,pos,ac);compress(w,538,st,268,332,norm,pos,ic);int nc=int(st[1]);if((pos+1)%4==0){for(int d=0;d<4;d++){st[12+nc*4+d]=ac[d];st[76+nc*4+d]=ic[d];}nc++;st[1]=float(nc);}
 float iq[4];mv(w+246,norm,iq,4,4);rope(iq,pos);float iw[1];mv(w+262,norm,iw,1,4);iw[0]/=2;int oldest=int(st[10]),best=-1;float bs=-3.402823466e+38f;for(int c=0;c<nc;c++){int end=c*4+3;if(end>=oldest)continue;float z=0;for(int d=0;d<4;d++)z+=iq[d]*st[76+c*4+d];z=max(0.0f,z)*iw[0];if(z>bs){bs=z;best=c;}}out[32]=float(best);
 float mx=w[638],scores[3];int n=0;if(best>=0){float z=0;for(int d=0;d<4;d++)z+=q[d]*st[12+best*4+d];scores[n++]=z/2;mx=max(mx,scores[n-1]);}for(int r=0;r<raw;r++){float z=0;for(int d=0;d<4;d++)z+=q[d]*st[2+r*4+d];scores[n++]=z/2;mx=max(mx,scores[n-1]);}float den=exp(w[638]-mx),att[4]={0,0,0,0};for(int i=0;i<n;i++){scores[i]=exp(scores[i]-mx);den+=scores[i];}int ri=0;if(best>=0){for(int d=0;d<4;d++)att[d]+=scores[ri]/den*st[12+best*4+d];ri++;}for(int r=0;r<raw;r++){for(int d=0;d<4;d++)att[d]+=scores[ri]/den*st[2+r*4+d];ri++;}rope(att,-pos);float ao[4];mv(w+60,att,ao,4,4);for(int i=0;i<4;i++)out[28+i]=ao[i];
 float ahc[8];posthc(ao,hc,post,comb,ahc);float fm[8];split(w,171,235,238,ahc,fm,pre,post,comb);for(int i=0;i<8;i++)out[12+i]=fm[i];weighted(ahc,pre,plain);rms(plain,w+24,norm,4);
 float logits[4];mv(w+76,norm,logits,4,4);int ids[4]={0,1,2,3};for(int a=0;a<4;a++)for(int b=a+1;b<4;b++){float za=logits[ids[a]]+w[92+ids[a]],zb=logits[ids[b]]+w[92+ids[b]];if(zb>za||(zb==za&&ids[b]<ids[a])){int x=ids[a];ids[a]=ids[b];ids[b]=x;}}float rm=max(logits[ids[0]],logits[ids[1]]),rw[2]={exp(logits[ids[0]]-rm),exp(logits[ids[1]]-rm)};float rz=rw[0]+rw[1];rw[0]/=rz;rw[1]/=rz;for(int i=0;i<2;i++){out[33+i]=float(ids[i]);out[35+i]=rw[i];}
 float moe[4]={0,0,0,0};for(int slot=0;slot<2;slot++){int base=294+ids[slot]*36;float g[3],u[3],a[3],v[4];mv(w+base,norm,g,3,4);mv(w+base+12,norm,u,3,4);for(int i=0;i<3;i++)a[i]=g[i]*sigm(g[i])*u[i];mv(w+base+24,a,v,4,3);for(int d=0;d<4;d++)moe[d]+=rw[slot]*v[d];}for(int i=0;i<4;i++)out[37+i]=moe[i];posthc(moe,ahc,post,comb,hc);for(int i=0;i<8;i++)out[41+i]=hc[i];float hw[2]={.55f,.45f},head[4],on[4],final[6];weighted(hc,hw,head);rms(head,w+290,on,4);mv(w+266,on,final,6,4);for(int i=0;i<6;i++)out[49+i]=final[i];
}
`
