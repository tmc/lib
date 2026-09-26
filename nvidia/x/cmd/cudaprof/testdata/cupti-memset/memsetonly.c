// memsetonly.c is a CUPTI injection library that enables only
// CUPTI_ACTIVITY_KIND_MEMSET and varies when and how it is collected.
//
// Loaded through CUDA_INJECTION64_PATH. It counts cuGraphLaunch calls with a
// driver API callback on that one function.
//
//   MEMSETONLY_ENABLE_AFTER=n   enable MEMSET after the nth graph launch (0: at init)
//   MEMSETONLY_DISABLE_AFTER=n  disable MEMSET after the nth graph launch (0: never)
//   MEMSETONLY_FLUSH_EVERY=n    cuptiActivityFlushAll(0) after every nth launch (0: never)
//   MEMSETONLY_FLUSH_FORCED=1   use CUPTI_ACTIVITY_FLAG_FLUSH_FORCED for those flushes
//   MEMSETONLY_BUFSIZE=bytes    activity buffer size (default 8 MiB)
//   MEMSETONLY_LOG_FROM=n       print the 1-based number of every launch >= n on entry
//   MEMSETONLY_NOCB=1           do not subscribe (ENABLE_AFTER, DISABLE_AFTER and
//                          FLUSH_EVERY are then ignored)
//
// At exit (when the process exits through libc) it prints the record count and
// the number of launches seen.
#include <cupti.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static atomic_ulong nrec, nlaunch;
static long enable_after, disable_after, flush_every, bufsize = 8 << 20;
static long log_from;
static int forced;

static void CUPTIAPI requested(uint8_t **buf, size_t *size, size_t *max) {
	*size = bufsize;
	*buf = aligned_alloc(8, *size);
	*max = 0;
}

static void CUPTIAPI completed(CUcontext ctx, uint32_t stream, uint8_t *buf, size_t size, size_t valid) {
	CUpti_Activity *r = NULL;
	while (cuptiActivityGetNextRecord(buf, valid, &r) == CUPTI_SUCCESS) atomic_fetch_add(&nrec, 1);
	free(buf);
}

static void check(const char *what, CUptiResult r) {
	if (r == CUPTI_SUCCESS) return;
	const char *s = "?";
	cuptiGetResultString(r, &s);
	fprintf(stderr, "memsetonly: %s: %s\n", what, s);
}

static void CUPTIAPI callback(void *ud, CUpti_CallbackDomain d, CUpti_CallbackId id, const void *data) {
	const CUpti_CallbackData *cd = data;
	if (cd->callbackSite == CUPTI_API_ENTER) {
		if (log_from > 0 && atomic_load(&nlaunch) + 1 >= (unsigned long)log_from)
			fprintf(stderr, "memsetonly: launch %lu enter\n", atomic_load(&nlaunch) + 1);
		return;
	}
	unsigned long n = atomic_fetch_add(&nlaunch, 1) + 1;
	if (enable_after > 0 && n == (unsigned long)enable_after) {
		check("enable memset", cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
		fprintf(stderr, "memsetonly: enabled MEMSET after launch %lu\n", n);
	}
	if (disable_after > 0 && n == (unsigned long)disable_after) {
		check("disable memset", cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMSET));
		fprintf(stderr, "memsetonly: disabled MEMSET after launch %lu\n", n);
	}
	if (flush_every > 0 && n % flush_every == 0)
		check("flush", cuptiActivityFlushAll(forced ? CUPTI_ACTIVITY_FLAG_FLUSH_FORCED : 0));
}

static void fini(void) {
	cuptiActivityFlushAll(1);
	fprintf(stderr, "memsetonly: records=%lu launches=%lu\n", atomic_load(&nrec), atomic_load(&nlaunch));
}

static long envl(const char *k) {
	const char *v = getenv(k);
	return v != NULL ? atol(v) : 0;
}

int InitializeInjection(void) {
	enable_after = envl("MEMSETONLY_ENABLE_AFTER");
	disable_after = envl("MEMSETONLY_DISABLE_AFTER");
	flush_every = envl("MEMSETONLY_FLUSH_EVERY");
	forced = envl("MEMSETONLY_FLUSH_FORCED") != 0;
	log_from = envl("MEMSETONLY_LOG_FROM");
	if (envl("MEMSETONLY_BUFSIZE") > 0) bufsize = envl("MEMSETONLY_BUFSIZE");
	check("register", cuptiActivityRegisterCallbacks(requested, completed));
	if (enable_after == 0) check("enable memset", cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
	if (envl("MEMSETONLY_NOCB") == 0) {
		CUpti_SubscriberHandle h;
		check("subscribe", cuptiSubscribe(&h, callback, NULL));
		check("cb launch", cuptiEnableCallback(1, h, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch));
		check("cb launch ptsz", cuptiEnableCallback(1, h, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz));
	}
	atexit(fini);
	fprintf(stderr, "memsetonly: init enable_after=%ld disable_after=%ld flush_every=%ld forced=%d bufsize=%ld\n",
		enable_after, disable_after, flush_every, forced, bufsize);
	return 1;
}
