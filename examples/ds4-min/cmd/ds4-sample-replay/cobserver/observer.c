/* Standalone sampler observer. The owning oracle tree remains read-only. */
#include "ds4.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
typedef struct { uint32_t h[8]; uint64_t bits; unsigned char b[64]; size_t n; } sha;
static uint32_t rr(uint32_t x, int n) { return (x >> n) | (x << (32-n)); }
static void shab(sha *s, const unsigned char *p) {
    static const uint32_t k[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    uint32_t w[64]; for (int i=0;i<16;i++) w[i]=(uint32_t)p[4*i]<<24|(uint32_t)p[4*i+1]<<16|(uint32_t)p[4*i+2]<<8|p[4*i+3];
    for (int i=16;i<64;i++) { uint32_t a=w[i-15], b=w[i-2]; uint32_t x=rr(a,7)^rr(a,18)^(a>>3), y=rr(b,17)^rr(b,19)^(b>>10); w[i]=w[i-16]+x+w[i-7]+y; }
    uint32_t a=s->h[0],b=s->h[1],c=s->h[2],d=s->h[3],e=s->h[4],f=s->h[5],g=s->h[6],h=s->h[7];
    for(int i=0;i<64;i++){uint32_t S1=rr(e,6)^rr(e,11)^rr(e,25), ch=(e&f)^((~e)&g), t1=h+S1+ch+k[i]+w[i]; uint32_t S0=rr(a,2)^rr(a,13)^rr(a,22), maj=(a&b)^(a&c)^(b&c),t2=S0+maj; h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;}
    s->h[0]+=a;s->h[1]+=b;s->h[2]+=c;s->h[3]+=d;s->h[4]+=e;s->h[5]+=f;s->h[6]+=g;s->h[7]+=h;
}
static void shainit(sha *s){static const uint32_t h[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};memcpy(s->h,h,sizeof h);s->bits=0;s->n=0;}
static void shaadd(sha *s,const unsigned char *p,size_t n){s->bits+=8*n;while(n){size_t m=64-s->n;if(m>n)m=n;memcpy(s->b+s->n,p,m);s->n+=m;p+=m;n-=m;if(s->n==64){shab(s,s->b);s->n=0;}}}
static void shaout(sha *s,unsigned char out[32]){size_t n=s->n;s->b[n++]=128;while(n!=56){if(n==64){shab(s,s->b);n=0;}s->b[n++]=0;}for(int i=0;i<8;i++)s->b[56+i]=(unsigned char)(s->bits>>(56-8*i));shab(s,s->b);for(int i=0;i<8;i++)for(int j=0;j<4;j++)out[4*i+j]=(unsigned char)(s->h[i]>>(24-8*j));}
static void digest(const void *data, size_t n, unsigned char out[32]) { sha s; shainit(&s); shaadd(&s, (const unsigned char *)data, n); shaout(&s, out); }
static void print_digest(const char *name, const unsigned char d[32]) { printf("%s_sha256=", name); for (int i=0;i<32;i++) printf("%02x", d[i]); putchar('\n'); }

static void fail(const char *s) { fprintf(stderr,"sample-observer: %s\n",s); exit(2); }
static void hexsum(const void *data,size_t n,char out[65]) {
    unsigned char d[32]; digest(data,n,d);
    for(int i=0;i<32;i++) sprintf(out+2*i,"%02x",d[i]);
    out[64]=0;
}
static void seal_head_stage(const char *from, const char *to, size_t bytes, char hash[65]) {
    FILE *in = fopen(from, "rb");
    unsigned char *data = malloc(bytes);
    if (!in || !data || fread(data,1,bytes,in)!=bytes || fgetc(in)!=EOF || fclose(in)) fail("head stage extent");
    FILE *out = fopen(to,"wbx");
    if (!out || fwrite(data,1,bytes,out)!=bytes || fclose(out)) fail("seal head stage");
    hexsum(data,bytes,hash); free(data);
}
static void hash_file(const char *path, char hash[65]) {
    FILE *f = fopen(path,"rb"); if (!f) fail("provenance file open");
    sha s; shainit(&s); unsigned char data[8192]; size_t n;
    while ((n=fread(data,1,sizeof data,f))) shaadd(&s,data,n);
    if(ferror(f)||fclose(f))fail("provenance file read");
    unsigned char digest[32];shaout(&s,digest);
    for(int i=0;i<32;i++)sprintf(hash+2*i,"%02x",digest[i]);hash[64]=0;
}
static void seal_head(const int *history, int count, int ctx) {
    const char *prefix=getenv("DS4_HEAD_OBSERVER_PREFIX");
    if(!prefix||!*prefix)return;
    const char *dump=getenv("DS4_METAL_GRAPH_DUMP_PREFIX");
    const char *model=getenv("DS4_HEAD_OBSERVER_MODEL_SHA256");
    const char *source=getenv("DS4_HEAD_OBSERVER_SOURCE_MANIFEST");
    if(!dump||!source||!model||strlen(model)!=64)fail("head provenance or existing C dump prefix missing");
    for(int i=0;i<64;i++)if(!((model[i]>='0'&&model[i]<='9')||(model[i]>='a'&&model[i]<='f')))fail("model checksum assertion");
    if(strlen(prefix)>3500||strlen(dump)>3500)fail("head path length");
    const char *base=strrchr(prefix,'/');base=base?base+1:prefix;
    for(const char *p=base;*p;p++)if(!((*p>='a'&&*p<='z')||(*p>='A'&&*p<='Z')||(*p>='0'&&*p<='9')||*p=='-'||*p=='_'))fail("head basename must be an identifier");
    char from[4096],to[4096],hc[65],logits[65],norm[65],q81[65],hs[65],src[65],binary[65],route_hash[65];
    if(snprintf(from,sizeof from,"%s_result_hc-43_pos0.bin",dump)>=(int)sizeof from||
       snprintf(to,sizeof to,"%s.hc.f32",prefix)>=(int)sizeof to)fail("head path");
    seal_head_stage(from,to,4096*4,hc);
    if(snprintf(from,sizeof from,"%s_result_output-43_pos0.bin",dump)>=(int)sizeof from||
       snprintf(to,sizeof to,"%s.logits.f32",prefix)>=(int)sizeof to)fail("head path");
    seal_head_stage(from,to,129280*4,logits);
    snprintf(from,sizeof from,"%s.norm.f32",prefix);hash_file(from,norm);
    snprintf(from,sizeof from,"%s_result_norm-43_pos0.bin",dump);
    char norm_hook[65];hash_file(from,norm_hook);
    if(strcmp(norm,norm_hook))fail("live head input disagrees with existing result_norm hook");
    snprintf(from,sizeof from,"%s.q81.bin",prefix);hash_file(from,q81);
    snprintf(from,sizeof from,"%s.route",prefix);FILE *r=fopen(from,"rb");char route[100]={0};
    if(!r||!fgets(route,sizeof route,r)||fgetc(r)!=EOF||fclose(r))fail("head route unread");
    route[strcspn(route,"\n")]=0;
    if(strcmp(route,"aligned-q8_0-folded-q8_1")&&strcmp(route,"aligned-q8_0-scratch-q8_1"))fail("head route unsupported");
    hash_file(from,route_hash);
    hexsum(history,(size_t)count*4,hs);hash_file(source,src);hash_file("/proc/self/exe",binary);
    snprintf(to,sizeof to,"%s.json",prefix);FILE *out=fopen(to,"wx");if(!out)fail("head manifest create");
    if(fprintf(out,
        "{\"version\":1,\"model_sha256\":\"%s\",\"history_sha256\":\"%s\","
        "\"source_sha256\":\"%s\",\"binary_sha256\":\"%s\",\"position\":%d,\"context\":%d,\"route\":\"%s\","
        "\"model_sha256_basis\":\"external-receipt-assertion\","
        "\"source_sha256_basis\":\"actual-source-manifest-file\","
        "\"binary_sha256_basis\":\"actual-executable-file\","
        "\"route_record\":{\"file\":\"%s.route\",\"sha256\":\"%s\"},"
        "\"stages\":{\"hc\":{\"file\":\"%s.hc.f32\",\"sha256\":\"%s\"},"
        "\"norm\":{\"file\":\"%s.norm.f32\",\"sha256\":\"%s\"},"
        "\"q81\":{\"file\":\"%s.q81.bin\",\"sha256\":\"%s\"},"
        "\"logits\":{\"file\":\"%s.logits.f32\",\"sha256\":\"%s\"}}}\n",
        model,hs,src,binary,count-1,ctx,route,base,route_hash,base,hc,base,norm,base,q81,base,logits)<0||fclose(out))fail("head manifest write");
}
int main(int argc,char **argv) {
    /* Raw int32 LE prompt avoids a second tokenizer or permissive JSON parser. */
    if(argc!=7) fail("usage: MODEL PROMPT.i32 CTX DECISIONS SEED OUTDIR-or--");
    if(sizeof(int)!=4 || sizeof(float)!=4) fail("unsupported scalar width");
    uint32_t endian=1; if(*(unsigned char *)&endian!=1) fail("requires little endian");
    char *end; long ctx=strtol(argv[3],&end,10); if(*end||ctx<=0||ctx>2147483647) fail("context");
    long limit=strtol(argv[4],&end,10); if(*end||(limit!=16&&limit!=64)) fail("decisions must be 16 or 64");
    uint64_t rng=strtoull(argv[5],&end,10); if(*end||!rng) fail("seed");
    FILE *f=fopen(argv[2],"rb"); if(!f) fail("prompt open");
    if(fseek(f,0,SEEK_END)) fail("prompt seek"); long bytes=ftell(f);
    if(bytes<=0||bytes%4||bytes/4+limit>ctx) fail("prompt length");
    rewind(f); int count=bytes/4;
    int *history=calloc((size_t)count+limit,sizeof(int)); if(!history) fail("history allocation");
    if(fread(history,1,bytes,f)!=(size_t)bytes||fclose(f)) fail("prompt read");
    int observe=strcmp(argv[6],"-")!=0;
    if(observe && mkdir(argv[6],0700)) fail("output directory must be fresh");
    setenv("DS4_MTP_SPEC_DISABLE","1",1);
    ds4_engine_options opt={0}; opt.model_path=argv[1]; opt.backend=DS4_BACKEND_CUDA; opt.context_size=ctx; opt.n_threads=1;
    ds4_engine *e=NULL; if(ds4_engine_open(&e,&opt)||!e) fail("engine open");
    int vocab=ds4_engine_vocab_size(e);
    for(int i=0;i<count;i++) if(history[i]<0||history[i]>=vocab) fail("prompt token");
    ds4_session *s=NULL; if(ds4_session_create(&s,e,ctx)||!s) fail("session create");
    ds4_tokens prompt={.v=history,.len=count,.cap=count}; char err[256]={0};
    if(ds4_session_sync(s,&prompt,err,sizeof err)) fail(err);
    /* Seal before any subsequent output head overwrites the literal pos0 hook. */
    seal_head(history,count,ctx);
    float *logits=malloc((size_t)vocab*4); if(!logits) fail("logit allocation");
    int stopped=0, decisions=0;
    for(int step=0;step<limit;step++) {
        if(observe && ds4_session_copy_logits(s,logits,vocab)!=vocab) fail("copy logits");
        uint64_t before=rng;
        int token=ds4_session_sample(s,1,0,1,0.05f,&rng);
        if(token<0||token>=vocab) fail("sample token");
        int stop=ds4_token_is_stop_for_think_mode(e,token,DS4_THINK_NONE);
        printf("%d %d %llu %llu %d\n",step,token,(unsigned long long)before,(unsigned long long)rng,stop);
        if(observe) {
            for(int i=0;i<vocab;i++) if(!isfinite(logits[i])) fail("non-finite logits");
            char hs[65],ls[65],path[4096];
            hexsum(history,(size_t)count*4,hs); hexsum(logits,(size_t)vocab*4,ls);
            if(snprintf(path,sizeof path,"%s/step-%04d.f32",argv[6],step)>=(int)sizeof path) fail("path");
            f=fopen(path,"wbx"); if(!f||fwrite(logits,4,vocab,f)!=(size_t)vocab||fclose(f)) fail("write logits");
            snprintf(path,sizeof path,"%s/step-%04d.json",argv[6],step);
            f=fopen(path,"wx"); if(!f) fail("write decision");
            if(fprintf(f,"{\"step\":%d,\"token\":%d,\"rng_before\":%llu,\"rng_after\":%llu,\"stop\":%s,\"history_sha256\":\"%s\",\"logits_sha256\":\"%s\"}\n",step,token,(unsigned long long)before,(unsigned long long)rng,stop?"true":"false",hs,ls)<0||fclose(f)) fail("write decision");
        }
        decisions++; if(stop) { stopped=1; break; }
        if(ds4_session_eval(s,token,err,sizeof err)) fail(err);
        history[count++]=token;
    }
    if(ds4_session_copy_logits(s,logits,vocab)!=vocab) fail("copy final logits");
    if(observe) {
        char path[4096];
        if(snprintf(path,sizeof path,"%s.final.f32",argv[6])>=(int)sizeof path)fail("final logits path");
        f=fopen(path,"wbx");
        if(!f||fwrite(logits,4,vocab,f)!=(size_t)vocab||fclose(f))fail("write final logits");
    }
    char final[65]; hexsum(logits,(size_t)vocab*4,final);
    printf("terminal decisions=%d reason=%s final_logits_sha256=%s\n",decisions,stopped?"stop":"limit",final);
    free(logits); ds4_session_free(s); ds4_engine_close(e); free(history); return 0;
}
