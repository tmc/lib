/* Isolated diagnostic overlay: include only in the pinned ds4_mmq.cu copy. */
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static int ds4_head_write(const char *prefix, const char *suffix,
                         const void *data, size_t bytes) {
    char path[4096];
    if (snprintf(path, sizeof path, "%s%s", prefix, suffix) >= (int)sizeof path)
        return -1;
    int fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0) return -1;
    const char *p = (const char *)data;
    while (bytes) {
        ssize_t n = write(fd, p, bytes);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { close(fd); return -1; }
        p += n;
        bytes -= (size_t)n;
    }
    return close(fd);
}

/* The first shape-qualified call is the ordinary session_sync output head.
 * Other routes do not emit a packet: absence must fail admission, not fall
 * back to a reconstruction. No performance claim is valid with this enabled. */
static int ds4_head_snapshot(const float *input, const void *q81,
                             int M, int N, int K, bool folded,
                             cudaStream_t stream) {
    const char *prefix = getenv("DS4_HEAD_OBSERVER_PREFIX");
    if (!prefix || !*prefix) return 0;
    if (M != 129280 || N != 1 || K != 4096) return 0;
    static bool emitted = false;
    if (emitted) return 0;
    float normalized[4096];
    unsigned char codes[128 * 36];
    if (cudaStreamSynchronize(stream) != cudaSuccess ||
        cudaMemcpy(normalized, input, sizeof normalized, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(codes, q81, sizeof codes, cudaMemcpyDeviceToHost) != cudaSuccess)
        return -1;
    const char *route = folded ? "aligned-q8_0-folded-q8_1\n" : "aligned-q8_0-scratch-q8_1\n";
    if (ds4_head_write(prefix, ".norm.f32", normalized, sizeof normalized) ||
        ds4_head_write(prefix, ".q81.bin", codes, sizeof codes) ||
        ds4_head_write(prefix, ".route", route, strlen(route))) return -1;
    emitted = true;
    return 0;
}
