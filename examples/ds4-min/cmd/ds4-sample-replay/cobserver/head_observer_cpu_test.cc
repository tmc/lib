/* Host wiring control only: the CUDA calls below are deliberate stubs. */
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <sys/stat.h>
typedef int cudaStream_t;
enum { cudaSuccess, cudaMemcpyDeviceToHost };
static int cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
static int cudaMemcpy(void *out, const void *in, size_t n, int) {
    memcpy(out, in, n); return cudaSuccess;
}
#include "head_observer.h"
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    setenv("DS4_HEAD_OBSERVER_PREFIX", argv[1], 1);
    float input[4096] = {};
    unsigned char q[128*36] = {};
    input[17] = 1.25f; q[4] = 7;
    if (ds4_head_snapshot(input, q, 129279, 1, 4096, false, 0)) return 3;
    if (ds4_head_snapshot(input, q, 129280, 1, 4096, false, 0)) return 4;
    char path[4096]; struct stat st;
    snprintf(path, sizeof path, "%s.norm.f32", argv[1]);
    if (stat(path, &st) || st.st_size != sizeof input) return 5;
    FILE *f = fopen(path, "rb"); float got[4096];
    if (!f || fread(got, sizeof got, 1, f) != 1 || fclose(f) || memcmp(got,input,sizeof got)) return 6;
    snprintf(path, sizeof path, "%s.q81.bin", argv[1]);
    f = fopen(path, "rb"); unsigned char codes[sizeof q];
    if (!f || fread(codes,sizeof codes,1,f)!=1 || fclose(f) || memcmp(codes,q,sizeof q)) return 7;
    input[17]=2;
    if (ds4_head_snapshot(input,q,129280,1,4096,true,0)) return 8;
    snprintf(path,sizeof path,"%s.route",argv[1]);
    f=fopen(path,"rb");char route[100]={};
    if(!f||!fgets(route,sizeof route,f)||fclose(f)||strcmp(route,"aligned-q8_0-scratch-q8_1\n"))return 9;
    if (!ds4_head_write(argv[1],".q81.bin",q,sizeof q)) return 10;
    return 0;
}
