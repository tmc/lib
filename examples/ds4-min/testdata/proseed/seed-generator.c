/* Independent seeded GGUF generator. Not checkpoint parity.
 * Input is the sealed 8MiB Pro header, not a Go-created tensor description.
 * The tensor directory matches the real Pro header (1894 tensors): hash
 * layers carry ffn_gate_tid2eid [6,129280] and no exp_probs_b.bias. */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
typedef struct {
    char name[128], key[256];
    uint32_t kind, rank;
    uint64_t dims[3], bytes, offset;
    int first;
} tensor;
static tensor tensors[2048];
static size_t count;
static uint64_t physical;
static void die(const char *s) { fprintf(stderr,"%s\n",s); exit(1); }
static void put(FILE *f,uint64_t n,unsigned width) {
    unsigned char b[8];
    for (unsigned i=0;i<width;i++) b[i]=(unsigned char)(n>>(8*i));
    if (fwrite(b,1,width,f)!=width) die("write failed");
}
static void string(FILE *f,const char *s) {
    put(f,strlen(s),8);
    if (fwrite(s,1,strlen(s),f)!=strlen(s)) die("string write failed");
}
static int suffix(const char *s,const char *end) {
    size_t n=strlen(s), m=strlen(end);
    return n>=m && !strcmp(s+n-m,end);
}
static int uploaded(const char *s) {
    return strstr(s,".hc_") || !strncmp(s,"output_hc_",10) ||
        suffix(s,"_norm.weight") || suffix(s,".attn_sinks.weight") || suffix(s,"_ape.weight");
}
static void add(const char *name,uint32_t kind,uint64_t a,uint64_t b,uint64_t c) {
    if (count==2048 || strlen(name)>=128) die("tensor admission exceeded");
    tensor *t=&tensors[count]; strcpy(t->name,name); t->kind=kind;
    t->rank=c?3:b?2:1; t->dims[0]=a; t->dims[1]=b; t->dims[2]=c;
    const char *role=name;
    if (!strncmp(name,"blk.",4) && !uploaded(name)) {
        role=strchr(name+4,'.'); if (!role) die("invalid layer name"); role++;
    }
    int n;
    if (c) n=snprintf(t->key,256,"%s/%u/[%" PRIu64 " %" PRIu64 " %" PRIu64 "]",role,kind,a,b,c);
    else if (b) n=snprintf(t->key,256,"%s/%u/[%" PRIu64 " %" PRIu64 "]",role,kind,a,b);
    else n=snprintf(t->key,256,"%s/%u/[%" PRIu64 "]",role,kind,a);
    if (n<0 || n>=256) die("key overflow");
    uint64_t block=kind==8?32:(kind==16 || kind==10)?256:1;
    uint64_t size=kind==0 || kind==26?4:kind==1?2:kind==8?34:kind==16?66:kind==10?84:0;
    if (!size || a%block) die("invalid row/type");
    t->bytes=a/block*size*(b?b:1)*(c?c:1); t->first=1; t->offset=physical;
    for (size_t i=0;i<count;i++) if (!strcmp(t->key,tensors[i].key)) {
        if (t->bytes!=tensors[i].bytes) die("alias extent differs");
        t->offset=tensors[i].offset; t->first=0; break;
    }
    if (t->first) {
        if (t->bytes>(UINT64_C(16)<<30)-physical) die("physical admission exceeded");
		uint64_t padding=(32-physical%32)%32;
		if (padding>(UINT64_C(16)<<30)-physical || t->bytes>(UINT64_C(16)<<30)-physical-padding) die("aligned physical admission exceeded");
		physical+=padding; t->offset=physical; physical+=t->bytes;
    }
    count++;
}
static void layer(int l,const char *role,uint32_t kind,uint64_t a,uint64_t b,uint64_t c) {
    char name[128];
    if (snprintf(name,sizeof(name),"blk.%d.%s",l,role)>=(int)sizeof(name)) die("layer name overflow");
    add(name,kind,a,b,c);
}
static void plan(void) {
    add("token_embd.weight",1,7168,129280,0);
    add("output_hc_base.weight",0,4,0,0); add("output_hc_fn.weight",1,28672,4,0);
    add("output_hc_scale.weight",0,1,0,0); add("output_norm.weight",0,7168,0,0);
    add("output.weight",8,7168,129280,0);
    for (int l=0;l<61;l++) {
        const char *phases[]={"attn","ffn"};
        for (int p=0;p<2;p++) {
            char role[64];
            snprintf(role,64,"hc_%s_fn.weight",phases[p]); layer(l,role,1,28672,24,0);
            snprintf(role,64,"hc_%s_base.weight",phases[p]); layer(l,role,0,24,0,0);
            snprintf(role,64,"hc_%s_scale.weight",phases[p]); layer(l,role,0,3,0,0);
        }
        layer(l,"attn_norm.weight",0,7168,0,0);
        layer(l,"attn_q_a.weight",8,7168,1536,0); layer(l,"attn_q_a_norm.weight",0,1536,0,0);
        layer(l,"attn_q_b.weight",8,1536,65536,0); layer(l,"attn_kv.weight",8,7168,512,0);
        layer(l,"attn_kv_a_norm.weight",0,512,0,0); layer(l,"attn_sinks.weight",0,128,0,0);
        layer(l,"attn_output_a.weight",8,4096,16384,0); layer(l,"attn_output_b.weight",8,16384,7168,0);
        int ratio=l<2 || l%2?128:4, width=ratio==4?1024:512;
        layer(l,"attn_compressor_ape.weight",1,width,ratio,0);
        layer(l,"attn_compressor_kv.weight",1,7168,width,0);
        layer(l,"attn_compressor_gate.weight",1,7168,width,0);
        layer(l,"attn_compressor_norm.weight",0,512,0,0);
        if (ratio==4) {
            layer(l,"indexer.attn_q_b.weight",1,1536,8192,0); layer(l,"indexer.proj.weight",1,7168,64,0);
            layer(l,"indexer_compressor_ape.weight",1,256,ratio,0);
            layer(l,"indexer_compressor_kv.weight",1,7168,256,0);
            layer(l,"indexer_compressor_gate.weight",1,7168,256,0);
            layer(l,"indexer_compressor_norm.weight",0,128,0,0);
        }
        layer(l,"ffn_norm.weight",0,7168,0,0); layer(l,"ffn_gate_inp.weight",1,7168,384,0);
        if (l>=3) layer(l,"exp_probs_b.bias",0,384,0,0);
        layer(l,"ffn_gate_exps.weight",16,7168,3072,384); layer(l,"ffn_up_exps.weight",16,7168,3072,384);
        layer(l,"ffn_down_exps.weight",10,3072,7168,384);
        layer(l,"ffn_gate_shexp.weight",8,7168,3072,0); layer(l,"ffn_up_shexp.weight",8,7168,3072,0);
        layer(l,"ffn_down_shexp.weight",8,3072,7168,0);
        if (l<3) layer(l,"ffn_gate_tid2eid.weight",26,6,129280,0);
    }
}
static uint32_t hash(const char *s) {
    uint32_t h=2166136261u; for (;*s;s++) h=(h^(unsigned char)*s)*16777619u; return h;
}
static void le(unsigned char *p,uint64_t n,int width) {
    for (int i=0;i<width;i++) p[i]=(unsigned char)(n>>(8*i));
}
static void seed(const tensor *t,uint64_t offset,unsigned char *p,size_t n) {
    uint64_t h=hash(t->key);
    size_t block=t->kind==8?34:t->kind==16?66:t->kind==10?84:t->kind==1?2:4;
    if (offset%block || n%block) die("unaligned seed extent");
    for (size_t i=0;i<n;i+=block) {
        uint64_t index=(offset+i)/block;
        if (t->kind==0) {
            uint32_t bits=0x3f800000u;
            if (strstr(t->key,"base.weight")) bits=0;
            if (strstr(t->key,"attn_sinks.weight") || strstr(t->key,"exp_probs_b.bias")) {
                float v=(float)(index%13)/64; memcpy(&bits,&v,4);
            }
            le(p+i,bits,4);
        } else if (t->kind==26) le(p+i,index%384,4);
        else if (t->kind==1) {
            uint16_t bits=0x0800+((index+h)%31)*8;
            if (!strncmp(t->key,"token_embd.weight/",18)) bits=0x2800+((index+h)%31)*16;
            if ((index+h)%3==0) bits|=0x8000;
            le(p+i,bits,2);
        } else {
            for (size_t j=0;j<block;j++) p[i+j]=(unsigned char)(h+index*37+j*13+7);
            if (t->kind==8) {
                le(p+i,0x0800,2); for (size_t j=2;j<block;j++) p[i+j]=(unsigned char)((h+index+j)%7+1);
            } else if (t->kind==16) le(p+i,0x0c00,2);
            else { le(p+i+80,0x0c00,2); le(p+i+82,0x0800,2); }
        }
    }
}
static unsigned char header[8*1024*1024];
static size_t length,cursor;
static uint64_t readn(unsigned n) {
    if (n>8 || cursor>length || n>length-cursor) die("truncated metadata");
    uint64_t v=0; for (unsigned i=0;i<n;i++) v|=(uint64_t)header[cursor++]<<(8*i); return v;
}
static void skip(uint64_t n) {
    if (cursor>length || n>length-cursor) die("metadata extent exceeded"); cursor+=(size_t)n;
}
static void value(uint32_t kind,int depth) {
    static const unsigned widths[]={1,1,2,2,4,4,4,1,0,0,8,8,8};
    if (depth>4 || kind>12) die("metadata type/nesting exceeded");
    if (kind==8) skip(readn(8));
    else if (kind==9) {
        uint32_t element=(uint32_t)readn(4); uint64_t n=readn(8);
        if (n>8*1024*1024) die("metadata array exceeded");
        for (uint64_t i=0;i<n;i++) value(element,depth+1);
    } else skip(widths[kind]);
}
static void layout_rows(FILE *f) {
    for (size_t i=0;i<count;i++) {
        tensor *t=&tensors[i];
        fprintf(f,"%s\t%u\t%u\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%s\n",
            t->name,t->kind,t->rank,t->dims[0],t->dims[1],t->dims[2],t->offset,t->bytes,t->key);
    }
}
static size_t read_header(const char *path) {
    FILE *input=fopen(path,"rb"); if (!input) die("open header failed");
    length=fread(header,1,sizeof(header),input);
    if (ferror(input) || fclose(input)) die("read header failed");
    if (readn(4)!=0x46554747u || readn(4)!=3) die("requires GGUF v3");
    (void)readn(8); uint64_t metadata=readn(8);
    if (!metadata || metadata>256) die("metadata count exceeded");
    for (uint64_t i=0;i<metadata;i++) { skip(readn(8)); value((uint32_t)readn(4),0); }
    return cursor;
}
/* write_directory writes the GGUF prefix: the sealed header's metadata
 * records verbatim, this plan's tensor directory and zero padding to the
 * 32-byte data start upstream computes when general.alignment is absent. */
static void write_directory(FILE *output,size_t metadata_end) {
    uint64_t metadata=0; for (int i=0;i<8;i++) metadata|=(uint64_t)header[16+i]<<(8*i);
    put(output,0x46554747u,4); put(output,3,4); put(output,count,8); put(output,metadata,8);
    if (fwrite(header+24,1,metadata_end-24,output)!=metadata_end-24) die("metadata write failed");
    for (size_t i=0;i<count;i++) {
        tensor *t=&tensors[i]; string(output,t->name); put(output,t->rank,4);
        for (unsigned j=0;j<t->rank;j++) put(output,t->dims[j],8);
        put(output,t->kind,4); put(output,t->offset,8);
    }
    long pos=ftell(output); if (pos<0) die("tell failed");
    while (pos%32) { put(output,0,1); pos++; }
}
int main(int argc,char **argv) {
    if (argc==2 && !strcmp(argv[1],"--inspect")) {
        plan(); layout_rows(stdout);
        return fflush(stdout) || ferror(stdout) ? 1 : 0;
    }
    if (argc==2 && !strcmp(argv[1],"--samples")) {
        plan();
        for (size_t i=0;i<count;i++) if (tensors[i].first) {
            tensor *t=&tensors[i];
            size_t block=t->kind==8?34:t->kind==16?66:t->kind==10?84:t->kind==1?2:4;
            size_t n=t->bytes<block*7?(size_t)t->bytes:block*7;
            uint64_t offsets[]={0,t->bytes-n};
            unsigned char bytes[84*7];
            for (int j=0;j<2;j++) {
                seed(t,offsets[j],bytes,n);
                fprintf(stdout,"%s\t%" PRIu64 "\t",t->key,offsets[j]);
                for (size_t k=0;k<n;k++) fprintf(stdout,"%02x",bytes[k]);
                fputc('\n',stdout);
            }
        }
        return fflush(stdout) || ferror(stdout) ? 1 : 0;
    }
    if (argc==4 && !strcmp(argv[1],"--directory")) {
        size_t metadata_end=read_header(argv[2]); plan();
        FILE *output=fopen(argv[3],"wx"); if (!output) die("exclusive directory creation failed");
        write_directory(output,metadata_end);
        if (ferror(output) || fclose(output)) die("directory close failed");
        return 0;
    }
    if (argc!=4) { fprintf(stderr,"usage: seed-generator sealed-pro-header fixture.gguf layout.tsv\n"); return 2; }
    size_t metadata_end=read_header(argv[1]); plan();
    FILE *output=fopen(argv[2],"wx"); if (!output) die("exclusive fixture creation failed");
    FILE *layout=fopen(argv[3],"wx"); if (!layout) die("exclusive layout creation failed");
    write_directory(output,metadata_end);
    layout_rows(layout);
    unsigned char stage[84*1024];
    uint64_t written=0;
    for (size_t i=0;i<count;i++) if (tensors[i].first) {
        tensor *t=&tensors[i]; size_t block=t->kind==8?34:t->kind==16?66:t->kind==10?84:t->kind==1?2:4;
        while (written<t->offset) { put(output,0,1); written++; }
        if (written!=t->offset) die("backing offset order differs");
        size_t capacity=sizeof(stage)/block*block;
        for (uint64_t offset=0;offset<t->bytes;) {
            size_t n=t->bytes-offset<capacity?(size_t)(t->bytes-offset):capacity;
            seed(t,offset,stage,n); if (fwrite(stage,1,n,output)!=n) die("seed write failed"); offset+=n; written+=n;
        }
    }
    if (written!=physical) die("written physical extent differs");
    if (ferror(layout) || fclose(layout) || fclose(output)) die("output close failed");
    fprintf(stderr,"SEEDED_LAYOUT tensors=%zu layers=61 physical_bytes=%" PRIu64 " tokens=0,1,2,3,4 checkpoint_qualified=false\n",count,physical);
    return 0;
}
