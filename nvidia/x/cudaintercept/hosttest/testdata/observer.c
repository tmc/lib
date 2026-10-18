/*
 * Observer fixture. It includes the shim source, so it can read and preset
 * the shim's static ring state directly; the shim has no test hook.
 *
 * Each mode prints one line of counts followed by bad=<first failed check>
 * or bad=none.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define dlsym fixture_dlsym
#include "interpose-source.c"

#define CAP CUDAINTERCEPT_CAPACITY
#define THREADS 32
#define PER_THREAD 2000

static atomic_ulong forwarded;
static _Atomic(const char *) bad;
static struct cudaintercept_record buf[4 * CAP];

/* check records the first failed check; writers call it concurrently. */
static void check(int ok, const char *name) {
    const char *none = NULL;
    if (!ok)
        atomic_compare_exchange_strong(&bad, &none, name);
}

static int want_result(unsigned x, unsigned y) {
    return (int)((x + y) % 997);
}

static int real_launch(void *f, unsigned x, unsigned y, unsigned z,
                       unsigned bx, unsigned by, unsigned bz, unsigned sm,
                       void *stream, void **p, void **e) {
    (void)f, (void)z, (void)bx, (void)by, (void)bz, (void)sm, (void)stream, (void)p, (void)e;
    atomic_fetch_add(&forwarded, 1);
    return want_result(x, y);
}

void *fixture_dlsym(void *handle, const char *symbol) {
    if (handle != RTLD_NEXT || strcmp(symbol, "cuLaunchKernel"))
        exit(10);
    return (void *)real_launch;
}

static void *params[1];

/* launch launches with grid (x, y, 3), block (4, 5, 6), and checks the result. */
static void launch(unsigned x, unsigned y) {
    int got = cuLaunchKernel((void *)0x1001, x, y, 3, 4, 5, 6, 7, (void *)0x2002, params, NULL);
    check(got == want_result(x, y), "forwarded result");
}

/* valid checks the fields of r that do not depend on the launch order. */
static void valid(const struct cudaintercept_record *r) {
    check(r->version == CUDAINTERCEPT_RECORD_VERSION, "version");
    check(r->size == sizeof *r, "size");
    check(r->function == 0x1001 && r->stream == 0x2002, "handles");
    check(r->grid[2] == 3 && r->block[0] == 4 && r->block[1] == 5 && r->block[2] == 6, "dims");
    check(r->shared_mem_bytes == 7, "shmem");
    check(r->flags == (CUDAINTERCEPT_PARAMS_PRESENT | CUDAINTERCEPT_ARGUMENTS_UNAVAILABLE), "flags");
    check(r->result == want_result(r->grid[0], r->grid[1]), "recorded result");
    check(r->reserved == 0, "reserved");
}

static int committed_slots(void) {
    int n = 0;
    for (int i = 0; i < CAP; i++)
        n += atomic_load(&observe_ring[i].commit) != 0;
    return n;
}

/* off: with CUDAINTERCEPT_OBSERVE unset, launches forward and nothing is recorded. */
static void off(void) {
    for (unsigned i = 0; i < 5; i++)
        launch(i, 0);
    uint64_t dropped = 99;
    long n = cudaintercept_drain(buf, CAP, &dropped);
    printf("forwarded=%lu drain=%ld dropped=%llu head=%llu slots=%d", atomic_load(&forwarded), n,
           (unsigned long long)dropped, (unsigned long long)atomic_load(&observe_head), committed_slots());
}

/* bound: CAP+7 launches keep the first CAP and drop 7; draining frees the slots. */
static void bound(void) {
    for (unsigned i = 0; i < CAP + 7; i++)
        launch(i, 0);
    uint64_t dropped;
    long n = cudaintercept_drain(buf, 4 * CAP, &dropped);
    for (long i = 0; i < n; i++) {
        valid(&buf[i]);
        check(buf[i].seq == (uint64_t)i && buf[i].grid[0] == (unsigned)i, "seq order");
        check(buf[i].thread == 1, "thread");
    }
    long again = cudaintercept_drain(buf, 4 * CAP, NULL);
    launch(5000, 0);
    uint64_t dropped2;
    long after = cudaintercept_drain(buf, 4 * CAP, &dropped2);
    check(after == 1 && buf[0].seq == CAP && buf[0].grid[0] == 5000, "record after drain");
    printf("forwarded=%lu records=%ld dropped=%llu again=%ld after=%ld dropped=%llu",
           atomic_load(&forwarded), n, (unsigned long long)dropped, again, after,
           (unsigned long long)dropped2);
}

/* consume: a drain smaller than the backlog returns the oldest records, then the rest. */
static void consume(void) {
    for (unsigned i = 0; i < 20; i++)
        launch(i, 0);
    long n[4];
    uint64_t next = 0;
    for (int d = 0; d < 4; d++) {
        n[d] = cudaintercept_drain(buf, 8, NULL);
        for (long i = 0; i < n[d]; i++, next++) {
            valid(&buf[i]);
            check(buf[i].seq == next && buf[i].grid[0] == next, "seq order");
        }
    }
    printf("drains=%ld,%ld,%ld,%ld", n[0], n[1], n[2], n[3]);
}

static void *writer(void *arg) {
    unsigned x = (unsigned)(uintptr_t)arg;
    for (unsigned y = 0; y < PER_THREAD; y++)
        launch(x, y);
    return NULL;
}

static atomic_int writing;
static uint64_t drained, drains, last_seq;
static int have_seq;
static uint64_t thread_of[THREADS];
static long last_y[THREADS];

static void account(long n) {
    drains++;
    for (long i = 0; i < n; i++) {
        struct cudaintercept_record *r = &buf[i];
        valid(r);
        check(!have_seq || r->seq > last_seq, "seq increasing");
        have_seq = 1;
        last_seq = r->seq;
        unsigned x = r->grid[0];
        if (x >= THREADS) {
            check(0, "grid x");
            continue;
        }
        check(r->thread != 0, "thread ordinal");
        check(thread_of[x] == 0 || thread_of[x] == r->thread, "thread stable");
        thread_of[x] = r->thread;
        check((long)r->grid[1] > last_y[x], "per-thread order");
        last_y[x] = r->grid[1];
    }
    drained += n;
}

static void *drainer(void *unused) {
    (void)unused;
    while (atomic_load(&writing))
        account(cudaintercept_drain(buf, 64, NULL));
    return NULL;
}

/* concurrent: 32 writers race a drainer; every launch is forwarded and either drained or dropped. */
static void concurrent(void) {
    for (int i = 0; i < THREADS; i++)
        last_y[i] = -1;
    pthread_t w[THREADS], d;
    atomic_store(&writing, 1);
    if (pthread_create(&d, NULL, drainer, NULL))
        exit(3);
    for (int i = 0; i < THREADS; i++)
        if (pthread_create(&w[i], NULL, writer, (void *)(uintptr_t)i))
            exit(3);
    for (int i = 0; i < THREADS; i++)
        pthread_join(w[i], NULL);
    atomic_store(&writing, 0);
    pthread_join(d, NULL);
    uint64_t dropped;
    account(cudaintercept_drain(buf, 4 * CAP, &dropped));
    for (int i = 0; i < THREADS; i++)
        for (int j = 0; j < i; j++)
            check(thread_of[i] == 0 || thread_of[i] != thread_of[j], "thread distinct");
    printf("forwarded=%lu total=%llu", atomic_load(&forwarded),
           (unsigned long long)(drained + dropped));
    fprintf(stderr, "drained=%llu dropped=%llu drains=%llu\n", (unsigned long long)drained,
            (unsigned long long)dropped, (unsigned long long)drains);
}

/*
 * exhaust: with the sequence preset 3 short of UINT64_MAX, three launches are
 * recorded and every later one is dropped; no seq wraps into a slot.
 */
static void exhaust(void) {
    atomic_store(&observe_head, UINT64_MAX - 3);
    atomic_store(&observe_tail, UINT64_MAX - 3);
    for (unsigned i = 0; i < 5; i++)
        launch(i, 0);
    uint64_t dropped;
    long n = cudaintercept_drain(buf, 4 * CAP, &dropped);
    for (long i = 0; i < n; i++) {
        valid(&buf[i]);
        check(buf[i].seq == UINT64_MAX - 3 + (uint64_t)i && buf[i].grid[0] == (unsigned)i, "seq order");
    }
    for (unsigned i = 0; i < 3; i++)
        launch(10 + i, 0);
    uint64_t dropped2;
    long after = cudaintercept_drain(buf, 4 * CAP, &dropped2);
    check(atomic_load(&observe_head) == UINT64_MAX && atomic_load(&observe_tail) == UINT64_MAX, "head tail");
    printf("forwarded=%lu records=%ld dropped=%llu after=%ld dropped=%llu slots=%d",
           atomic_load(&forwarded), n, (unsigned long long)dropped, after,
           (unsigned long long)dropped2, committed_slots());
}

int main(int argc, char **argv) {
    if (argc != 2)
        return 2;
    const char *mode = argv[1];
    if (strcmp(mode, "off") == 0)
        off();
    else if (strcmp(mode, "bound") == 0)
        bound();
    else if (strcmp(mode, "consume") == 0)
        consume();
    else if (strcmp(mode, "concurrent") == 0)
        concurrent();
    else if (strcmp(mode, "exhaust") == 0)
        exhaust();
    else
        return 2;
    const char *b = atomic_load(&bad);
    printf(" bad=%s\n", b ? b : "none");
    return 0;
}
