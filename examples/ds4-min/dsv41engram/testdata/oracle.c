#include "ds4_engram.h"
#include <stdio.h>
#include <stdlib.h>

static void read_exact(void *p, size_t n) { if (fread(p,1,n,stdin)!=n) exit(2); }
static void write_exact(const void *p, size_t n) { if(fwrite(p,1,n,stdout)!=n) exit(2); }

int main(int argc, char **argv) {
    if (argc == 2) {
        ds4_engram_table table;
        uint32_t rows[256];
        float out[256*DS4_ENGRAM_DIM];
        for (int i=0;i<256;i++) rows[i]=(uint32_t)i;
        if (!ds4_engram_table_open(&table,argv[1],0,256)) return 3;
        int ok=ds4_engram_read(&table,rows,256,out);
        ds4_engram_table_close(&table);
        write_exact(&ok,sizeof(ok));
        if(ok) write_exact(out,sizeof(out));
        return 0;
    }
    ds4_engram_layout l={0};
    uint32_t vocab;
    read_exact(&vocab,sizeof(vocab));
    uint32_t *map=calloc(vocab,sizeof(*map));
    if(!map) return 2;
    l.token_map=map;l.vocab_size=vocab;
    read_exact(&l.compressed_vocab_size,sizeof(l.compressed_vocab_size));
    read_exact(&l.pad_id,sizeof(l.pad_id));
    read_exact(l.rows,sizeof(l.rows));
    read_exact(l.multipliers,sizeof(l.multipliers));
    read_exact(l.primes,sizeof(l.primes));
    read_exact(map,vocab*sizeof(*map));
    uint32_t count,split;
    read_exact(&count,sizeof(count));read_exact(&split,sizeof(split));
    int *tokens=calloc(count ? count:1,sizeof(*tokens));
    uint8_t *mask=calloc(count ? count:1,sizeof(*mask));
    uint32_t *rows=calloc(count ? (size_t)count*DS4_ENGRAM_LAYERS*DS4_ENGRAM_COLS:1,sizeof(*rows));
    if(!tokens||!mask||!rows)return 2;
    read_exact(tokens,count*sizeof(*tokens));read_exact(mask,count);
    ds4_engram_history history;
    ds4_engram_history_reset(&history);
    int ok=ds4_engram_layout_valid(&l);
    if(ok) {
        if(split>count)split=count;
        ok=ds4_engram_hash(&l,&history,tokens,mask,split,rows);
        if(ok)ok=ds4_engram_hash(&l,&history,tokens+split,mask+split,count-split,rows+(size_t)split*DS4_ENGRAM_LAYERS*DS4_ENGRAM_COLS);
    }
    write_exact(&ok,sizeof(ok));write_exact(history.tail,sizeof(history.tail));
    if(ok)write_exact(rows,(size_t)count*DS4_ENGRAM_LAYERS*DS4_ENGRAM_COLS*sizeof(*rows));
    free(map);free(tokens);free(mask);free(rows);
    return 0;
}
