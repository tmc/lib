/*
 * Bounded launch observer for the cuLaunchKernel shim.
 *
 * Observation is off unless the environment variable CUDAINTERCEPT_OBSERVE
 * is exactly "1" when the process first launches or drains. The value is
 * read once. With observation off the shim records nothing and
 * cudaintercept_drain returns -1.
 *
 * With observation on, each launch that the shim forwards is recorded after
 * the real cuLaunchKernel returns. Recording never changes the arguments or
 * the result, never blocks, and calls no function: no allocation, locking,
 * logging, or CUDA call. Records go to a fixed ring of
 * CUDAINTERCEPT_CAPACITY slots. A launch that finds the ring full is not
 * recorded and is counted as dropped; the oldest records are never
 * overwritten. The ring is zero-initialized static storage of 90112 bytes
 * on LP64 (CUDAINTERCEPT_CAPACITY records plus an 8-byte commit word each),
 * present whether or not observation is on.
 *
 * The launch path as a whole is not async-signal-safe. Besides recording,
 * it runs pthread_once (symbol resolution and the one-time environment
 * read) and reads and writes _Thread_local variables, and POSIX makes no
 * async-signal-safety promise for either. A signal handler that calls
 * cuLaunchKernel while the same thread is inside one of those
 * pthread_once calls can deadlock. Recording itself is lock-free; the
 * rest of the launch path is not.
 *
 * Records cover only launches that reach this shim by the cuLaunchKernel
 * symbol; see interpose.c for what bypasses it. They are never a complete
 * census of a process's launches.
 */
#ifndef CUDAINTERCEPT_OBSERVE_H
#define CUDAINTERCEPT_OBSERVE_H

#include <stddef.h>
#include <stdint.h>

#define CUDAINTERCEPT_RECORD_VERSION 1
#define CUDAINTERCEPT_CAPACITY 1024

/* Record flags. */
enum {
    CUDAINTERCEPT_PARAMS_PRESENT = 1 << 0, /* kernelParams != NULL */
    CUDAINTERCEPT_EXTRA_PRESENT = 1 << 1,  /* extra != NULL */
    /*
     * Always set: the shim never dereferences kernelParams or extra, so
     * argument meaning, tensor shape, stride, and dtype are unavailable.
     */
    CUDAINTERCEPT_ARGUMENTS_UNAVAILABLE = 1 << 2,
};

/*
 * A cudaintercept_record holds only values cuLaunchKernel was passed, plus
 * its result. The layout is fixed for a given version.
 */
struct cudaintercept_record {
    uint32_t version;          /* CUDAINTERCEPT_RECORD_VERSION */
    uint32_t size;             /* sizeof(struct cudaintercept_record) */
    uint64_t seq;              /* claim order, starting at 0; strictly increasing */
    uint64_t function;         /* the CUfunction handle value */
    uint64_t stream;           /* the CUstream handle value */
    uint64_t thread;           /* shim-assigned thread ordinal, starting at 1 */
    uint32_t grid[3];
    uint32_t block[3];
    uint32_t shared_mem_bytes;
    uint32_t flags;
    int32_t result;            /* the CUresult the real cuLaunchKernel returned */
    uint32_t reserved;         /* zero */
};

_Static_assert(sizeof(struct cudaintercept_record) == 80, "record layout");

/*
 * cudaintercept_drain moves up to cap of the oldest committed records into
 * out, in seq order, and returns how many it moved. Drained records are
 * consumed: a later drain does not return them again, and their slots become
 * free for new launches. A launch still being recorded ends the drain early;
 * its record and later ones are returned by a later drain.
 *
 * If dropped is not NULL, *dropped is set to the total number of launches
 * not recorded since the process started, either because the ring was full
 * or because the 64-bit sequence space is exhausted. The count is a 64-bit
 * counter that wraps to 0 after 2^64 drops, and it is not reset across
 * fork. Records drained plus *dropped equals the launches forwarded with
 * observation on only once every claimed seq has been drained, and only
 * while the counter has not wrapped and no writer has stalled (see below):
 * a stalled writer's seq is never drained and never counted as dropped.
 *
 * It returns -1, with *dropped set to 0, if observation is off.
 * Drains are serialized with each other but never block launches.
 *
 * Records are drained strictly in seq order, so a launch that has claimed
 * a seq but not yet committed its record stops every drain at that seq.
 * A writer that is preempted there delays the drain; one that never
 * finishes (for example, a thread killed mid-record) stalls it for good.
 * While it is stalled, drains return 0 even though later records may be
 * committed, and once CUDAINTERCEPT_CAPACITY seqs are outstanding every
 * new launch is dropped. No record is overwritten. The drain does not
 * report a stall; drains that return 0 while *dropped keeps rising are its
 * visible sign.
 *
 * out must point to at least cap records; out may be NULL only if cap is 0.
 * Drains take a process-wide mutex. If fork is called while another thread
 * is draining, the child's first drain can deadlock; a child should not
 * drain unless the parent was known not to be draining at the fork.
 *
 * Fork also copies the ring. If any other thread of the parent is between
 * claiming a seq and committing its record when fork is called, the child
 * inherits that claimed slot with no thread left to commit it: the child's
 * drains stall at that seq for good, as described above, and once the
 * child's outstanding seqs reach CUDAINTERCEPT_CAPACITY every later launch
 * in the child is counted as dropped. The child also inherits the parent's
 * undrained records, so a record can be drained once in each process.
 * Nothing reports either condition. A child should rely on its records
 * only if no other parent thread could have been launching at the fork.
 */
long cudaintercept_drain(struct cudaintercept_record *out, size_t cap, uint64_t *dropped);

#endif
