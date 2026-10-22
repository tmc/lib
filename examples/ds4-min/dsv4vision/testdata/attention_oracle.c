#include "ds4_image.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc != 6) return 2;
    uint32_t vocab=strtoul(argv[2],0,10), pos=strtoul(argv[3],0,10);
    uint32_t count=0, raw=strtoul(argv[4],0,10), window=strtoul(argv[5],0,10);
    if (fread(&count,4,1,stdin)!=1 || count==0 || count>4096) return 2;
    int *tokens=malloc((size_t)count*sizeof(int));
    if (!tokens || fread(tokens,4,count,stdin)!=count) return 2;
    if (!strcmp(argv[1],"bounds")) {
        uint32_t *b=malloc((size_t)count*2*sizeof(uint32_t));
        if (!b) return 2;
        if (!ds4_deepseek4_attention_bounds(tokens,count,vocab,pos,raw,window,b)) return 3;
        fwrite(b,4,(size_t)count*2,stdout);
        free(b);
    } else if (!strcmp(argv[1],"chunk")) {
        uint32_t chunk=0;
        if (!ds4_deepseek4_prefill_chunk(tokens,count,vocab,pos,raw,&chunk)) return 3;
        fwrite(&chunk,4,1,stdout);
    } else return 2;
    free(tokens);
    return ferror(stdout)?2:0;
}
