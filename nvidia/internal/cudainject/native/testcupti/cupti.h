#ifndef CUDAPROF_TESTCUPTI_H
#define CUDAPROF_TESTCUPTI_H

#include "cupti_activity.h"
#include "cupti_callbacks.h"
#include "cupti_version.h"

CUptiResult cuptiGetVersion(uint32_t *version);
CUptiResult cuptiGetResultString(CUptiResult, const char **message);

/* Harness-only entry points. offer() synthesizes one activity buffer of the
   requested shape and delivers it through the registered request/complete
   callbacks exactly as CUPTI would, including the dropped-record count the
   producer must attribute to CUPTI rather than to its own queue. */
void cudaprof_test_set_native_dropped(size_t dropped);
int cudaprof_test_offer(unsigned kernels, unsigned memcpys, unsigned apis, unsigned externals);
int cudaprof_test_offer_markers(unsigned kernels, unsigned memcpys, unsigned apis, unsigned externals, unsigned markers);
int cudaprof_test_offer_mix(unsigned kernels, unsigned memcpys, unsigned apis, unsigned externals, unsigned markers, unsigned memsets);
int cudaprof_test_offer_undecoded(unsigned kernels, unsigned memcpys, unsigned apis, unsigned externals, unsigned markers, unsigned memsets, unsigned undecoded);
int cudaprof_test_callbacks_registered(void);

/* Drives the Callback API surface capture-range gating subscribes to. fire()
   invokes the producer's subscribed callback exactly as CUPTI would, but only
   for a (domain, cbid) the producer actually enabled: a gate that fires for an
   id nobody subscribed to would pass on a producer that subscribed to nothing.
   It reports whether the callback was delivered. */
int cudaprof_test_subscribed(void);
int cudaprof_test_callback_enabled(unsigned domain, unsigned cbid);
int cudaprof_test_fire_callback(unsigned domain, unsigned cbid, unsigned site);

/* Reports how many activity kinds are enabled right now. Gating's whole claim
   is that this is zero before the range opens and nonzero inside it, so the
   control reads it rather than inferring it from record counts. */
unsigned cudaprof_test_enabled_kind_count(void);

/* Invokes only the producer's buffer-request callback and reports whether it
   handed out a buffer. A forked child must be refused one, so the control
   needs to ask without also delivering a completed buffer. */
int cudaprof_test_request_buffer(void);

/* Reports the call ordering CUPTI's timestamp-callback contract turns on: the
   sequence number of the cuptiActivityRegisterTimestampCallback that stuck and
   of the first cuptiActivityEnable, or zero for a call that never happened.
   The registration must come first, or records produced before it keep CUPTI's
   previous timestamp method and the journal straddles two clock bases.
   cudaprof_test_timestamp() invokes the registered callback, which is how a
   control tells a callback on the producer's own clock from any other. */
unsigned cudaprof_test_timestamp_registered_at(void);
unsigned cudaprof_test_first_enable_at(void);
uint64_t cudaprof_test_timestamp(void);

/* Mock struct size functions for layout compatibility test injection. */
CUptiResult cuptiActivityGetStructSize(CUpti_ActivityKind kind, uint32_t flags, size_t *size);
CUptiResult cudaprof_test_mock_short_kernel_struct_size(CUpti_ActivityKind kind, uint32_t flags, size_t *size);
CUptiResult cudaprof_test_mock_short_memcpy_struct_size(CUpti_ActivityKind kind, uint32_t flags, size_t *size);

#endif
