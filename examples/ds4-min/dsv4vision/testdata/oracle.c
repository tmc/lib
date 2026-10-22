#include "ds4_image.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    char error[256] = {0};
    if (argc != 5) return 2;
    unsigned h = strtoul(argv[2], 0, 10), w = strtoul(argv[3], 0, 10);
    if (!strcmp(argv[1], "layout")) {
        ds4_deepseek4_image_layout l;
        if (!ds4_deepseek4_image_layout_build(&l, h, w, strtoul(argv[4], 0, 10), error, sizeof error)) return 3;
        fwrite(&l.token_count, 4, 1, stdout);
        fwrite(&l.image_count, 4, 1, stdout);
        fwrite(l.types, 1, l.token_count, stdout);
        fwrite(l.perm, 4, l.image_count, stdout);
        ds4_deepseek4_image_layout_free(&l);
    } else {
        size_t size = (size_t)w*h*3;
        unsigned char *rgb = malloc(size);
        if (!rgb || fread(rgb, 1, size, stdin) != size) return 2;
        ds4_image i = {.width=w, .height=h, .rgb=rgb};
        ds4_deepseek4_image_patches p;
        if (!ds4_image_preprocess_deepseek4(&p, &i, error, sizeof error)) { fprintf(stderr,"%s\n",error); return 3; }
        unsigned meta[] = {p.content_width,p.content_height,p.padded_width,p.padded_height,p.grid_width,p.grid_height,p.llm_grid_width,p.llm_grid_height,p.patch_count};
        fwrite(meta,4,9,stdout);
        fwrite(p.patches,4,(size_t)p.patch_count*3*14*14,stdout);
        ds4_deepseek4_image_patches_free(&p);
        free(rgb);
    }
    return ferror(stdout) ? 2 : 0;
}
