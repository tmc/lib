/* Explicit, off-hook integration for one finite replay. Not transparent CUDA. */
#ifndef CUDAINTERCEPT_REPLAY_H
#define CUDAINTERCEPT_REPLAY_H
#include "registry.h"

#define CUDAINTERCEPT_OWNED_VERSION 2
#define CUDAINTERCEPT_IMAGE_LIMIT (16u << 20)

struct cudaintercept_replay_owner_v2;
/* Off-hook copy of observed target/output facts. Unset facts bits mean unknown.
 * qualifier/adapter/qualification identify the matched independent table row;
 * they are empty for raw qualification, not runtime source/compiler attestation.
 * No qualification boolean or dishonest-loader authentication is supplied. */
enum {
    CUDAINTERCEPT_FACT_PROCESS = 1,
    CUDAINTERCEPT_FACT_IMAGES = 2,
    CUDAINTERCEPT_FACT_CONTEXT = 4,
    CUDAINTERCEPT_FACT_DRIVER = 8,
    CUDAINTERCEPT_FACT_TARGET = 16,
    CUDAINTERCEPT_FACT_MODULES = 32,
    CUDAINTERCEPT_FACT_CANARY = 64,
    CUDAINTERCEPT_FACT_PUBLISHED = 128,
    CUDAINTERCEPT_FACT_DESCRIPTOR_UNKNOWN = 256,
    CUDAINTERCEPT_FACT_LIBRARY_UNKNOWN = 512,
};
struct cudaintercept_replay_evidence_v2 {
    uint32_t version, size, facts, completed_arms;
    uint64_t process, context, functions[2], modules[2];
    uint32_t driver, major, minor;
    unsigned char uuid[16];
    unsigned char outputs[3][3][64]; /* baseline, baseline, candidate; x,y,w */
    char baseline[65], candidate[65], driver_image[65];
    char qualifier[65], adapter[65], qualification[65], capsule[65];
};
struct cudaintercept_replay_request_v2 {
    uint32_t version, size;
    /* function must be zero; the owner resolves it from its own image. Other
     * fields declare the caller's live buffers/context/ordinary stream. */
    struct cudaintercept_replay_v1 operation;
    const void *baseline_image, *candidate_image;
    uint64_t baseline_bytes, candidate_bytes;
    char qualifier[65], adapter[65], qualification[65];
};

/* Inputs are readable and immutable through prepare. Images are copied and
 * hashed before loading. Production independent admission is currently EMPTY:
 * no caller field, boolean, output array, callback or environment can enable it.
 * A private compiled CPU fixture is not production/native authority.
 *
 * out must initially contain NULL; existing custody is never overwritten.
 * A nonnull owner returned with an error retains partial native custody and
 * MUST be passed to cleanup. Unknown completion is not permission to unload.
 * A successful singleton retains modules and its primary context for process
 * lifetime. The caller retains the three declared buffers and exclusive access
 * to their ordinary stream through finish, including on errors. No capture,
 * callbacks, context change or other CUDA work may intervene in owned launch.
 * Prepare and launch are restricted to the same OS thread/process. Any native
 * prepare/qualify/finish/cleanup also requires the caller's externally admitted
 * GPU owner/mediator lease; environment text is not proof of that custody. */
uint32_t cudaintercept_replay_prepare_v2(const struct cudaintercept_replay_request_v2 *,
                                         struct cudaintercept_replay_owner_v2 **);
/* Explicit off-hook RAW producer, for a separately authorized native event.
 * Does not need independent admission and NEVER publishes eligibility. Requires
 * a real-header build (or isolated CPU driver fixture). A zero result means
 * completed finite canary only. It does not authorize replacement. Retain/copy
 * evidence before cleanup; *owner has the same failure-custody rules as prepare.
 * The caller supplies the externally admitted hardware/lease/artifact event,
 * not a success/admission boolean. This API is never invoked by the hook. */
uint32_t cudaintercept_replay_qualify_v2(const struct cudaintercept_replay_request_v2 *,
                                         struct cudaintercept_replay_owner_v2 **);

/* Single-use consumer. Checks current context, capture and actual population
 * OFF hook, then immediately submits one launch using a private C frame.
 * A guard refusal forwards original once; *result is its actual CUresult.
 * A pending public v1 binding is consumed and its refusal vetoes selection.
 * A driver/readback/completion error stops without resubmitting: *result=-1.
 * Reusing this owner cannot select a candidate, even after resetting inputs.
 * The return is a decision, not independent qualification or GPU completion. */
uint32_t cudaintercept_replay_launch_v2(struct cudaintercept_replay_owner_v2 *,
                                        uint64_t caller_generation, int *result);
uint32_t cudaintercept_replay_finish_v2(struct cudaintercept_replay_owner_v2 *);
/* Only failed/unpublished owners can be destroyed; failures retain *owner.
 * An ambiguous descriptor/library close error retains custody until process
 * exit: a possibly reused numeric handle must not be retried. */
uint32_t cudaintercept_replay_cleanup_v2(struct cudaintercept_replay_owner_v2 **);
/* Copies producer-owned observations; cannot supply or modify admission. */
uint32_t cudaintercept_replay_evidence_v2(const struct cudaintercept_replay_owner_v2 *,
                                          struct cudaintercept_replay_evidence_v2 *);
#endif
