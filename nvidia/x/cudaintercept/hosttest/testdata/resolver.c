#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define dlsym fixture_dlsym
#include "interpose-source.c"

static int mode;
static int recursive_result = -1;
static atomic_int resolutions;
static atomic_int launches;

static int real_launch(void *f, unsigned x, unsigned y, unsigned z,
                       unsigned bx, unsigned by, unsigned bz, unsigned sm,
                       void *stream, void **params, void **extra) {
    atomic_fetch_add(&launches, 1);
    return 77;
}

static int launch(void) {
    return cuLaunchKernel(NULL, 1, 1, 1, 1, 1, 1, 0, NULL, NULL, NULL);
}

void *fixture_dlsym(void *handle, const char *symbol) {
    if (handle != RTLD_NEXT || strcmp(symbol, "cuLaunchKernel")) exit(10);
    atomic_fetch_add(&resolutions, 1);
    if (mode == 2) recursive_result = launch();
    if (mode == 1) return NULL;
    return (void *)real_launch;
}

static void *worker(void *unused) {
    return (void *)(long)launch();
}

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    mode = atoi(argv[1]);
    int result = -1;
    if (mode == 3) {
        pthread_t threads[32];
        for (int i = 0; i < 32; i++)
            if (pthread_create(&threads[i], NULL, worker, NULL)) return 3;
        for (int i = 0; i < 32; i++) {
            void *value;
            if (pthread_join(threads[i], &value) || (long)value != 77) return 4;
        }
        result = 77;
    } else {
        result = launch();
    }
    printf("result=%d recursive=%d resolutions=%d launches=%d\n", result,
           recursive_result, atomic_load(&resolutions), atomic_load(&launches));
    if (result != (mode == 1 ? 3 : 77) || atomic_load(&resolutions) != 1) return 5;
    if (mode == 2 && recursive_result != 3) return 6;
    if (atomic_load(&launches) != (mode == 3 ? 32 : mode == 1 ? 0 : 1)) return 7;
    return 0;
}
