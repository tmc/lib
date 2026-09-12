#ifndef CUDAPROF_TESTCUPTI_H
#define CUDAPROF_TESTCUPTI_H

#include "cupti_activity.h"
#include "cupti_version.h"

CUptiResult cuptiGetVersion(uint32_t *version);
CUptiResult cuptiGetResultString(CUptiResult, const char **message);

/* Harness-only entry points. offer() synthesizes one activity buffer of the
   requested shape and delivers it through the registered request/complete
   callbacks exactly as CUPTI would, including the dropped-record count the
   producer must attribute to CUPTI rather than to its own queue. */
void cudaprof_test_set_native_dropped(size_t dropped);
int cudaprof_test_offer(unsigned kernels, unsigned memcpys, unsigned apis, unsigned externals);
int cudaprof_test_callbacks_registered(void);

/* Invokes only the producer's buffer-request callback and reports whether it
   handed out a buffer. A forked child must be refused one, so the control
   needs to ask without also delivering a completed buffer. */
int cudaprof_test_request_buffer(void);

#endif
