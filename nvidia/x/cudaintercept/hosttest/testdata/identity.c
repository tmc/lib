/*
 * Argument-identity fixture: the shim must forward every cuLaunchKernel
 * argument and the result unchanged. The kernelParams and extra arrays must
 * arrive at the same addresses with their contents intact. With observation
 * on, the one record drained must hold the same values.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define dlsym fixture_dlsym
#include "interpose-source.c"

#define F ((void *)0x1001)
#define STREAM ((void *)0x2002)
#define GRID_X 0x1111u
#define GRID_Y 0x2222u
#define GRID_Z 0x3333u
#define BLOCK_X 0x4444u
#define BLOCK_Y 0x5555u
#define BLOCK_Z 0x6666u
#define SHMEM 0x7777u

static void *params[3];
static void *extra[3];
static int result;
static int calls;
static char mismatch[128];

static void check(int ok, const char *name) {
    if (ok)
        return;
    strcat(mismatch, " ");
    strcat(mismatch, name);
}

static int real_launch(void *f, unsigned x, unsigned y, unsigned z,
                       unsigned bx, unsigned by, unsigned bz, unsigned sm,
                       void *stream, void **p, void **e) {
    calls++;
    check(f == F, "f");
    check(x == GRID_X, "gridX");
    check(y == GRID_Y, "gridY");
    check(z == GRID_Z, "gridZ");
    check(bx == BLOCK_X, "blockX");
    check(by == BLOCK_Y, "blockY");
    check(bz == BLOCK_Z, "blockZ");
    check(sm == SHMEM, "shmem");
    check(stream == STREAM, "stream");
    check(p == params, "kernelParams");
    check(e == extra, "extra");
    return result;
}

void *fixture_dlsym(void *handle, const char *symbol) {
    if (handle != RTLD_NEXT || strcmp(symbol, "cuLaunchKernel"))
        exit(10);
    return (void *)real_launch;
}

int main(int argc, char **argv) {
    if (argc != 2)
        return 2;
    result = atoi(argv[1]);
    for (int i = 0; i < 3; i++) {
        params[i] = &params[i];
        extra[i] = &extra[i];
    }
    int got = cuLaunchKernel(F, GRID_X, GRID_Y, GRID_Z, BLOCK_X, BLOCK_Y,
                             BLOCK_Z, SHMEM, STREAM, params, extra);
    for (int i = 0; i < 3; i++) {
        check(params[i] == &params[i], "kernelParams contents");
        check(extra[i] == &extra[i], "extra contents");
    }
    struct cudaintercept_record r;
    long n = cudaintercept_drain(&r, 1, NULL);
    if (n == 1) {
        check(r.seq == 0 && r.thread == 1, "record seq");
        check(r.function == (uintptr_t)F && r.stream == (uintptr_t)STREAM, "record handles");
        check(r.grid[0] == GRID_X && r.grid[1] == GRID_Y && r.grid[2] == GRID_Z, "record grid");
        check(r.block[0] == BLOCK_X && r.block[1] == BLOCK_Y && r.block[2] == BLOCK_Z, "record block");
        check(r.shared_mem_bytes == SHMEM, "record shmem");
        check(r.flags == (CUDAINTERCEPT_PARAMS_PRESENT | CUDAINTERCEPT_EXTRA_PRESENT |
                          CUDAINTERCEPT_ARGUMENTS_UNAVAILABLE), "record flags");
        check(r.result == result, "record result");
    }
    printf("result=%d calls=%d drain=%ld mismatch=%s\n", got, calls, n,
           mismatch[0] ? mismatch + 1 : "none");
    return 0;
}
