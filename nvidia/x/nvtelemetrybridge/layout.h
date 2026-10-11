#ifndef NVTELEMETRY_BRIDGE_LAYOUT_H
#define NVTELEMETRY_BRIDGE_LAYOUT_H

#include <stdint.h>

/* Version 1; all scalars little-endian. No BTF/CO-RE relocation is implied.
 * BPF_MAP_TYPE_HASH, key uint32_t(0), max_entries 1. Publish by whole-entry
 * replacement, never by direct in-place mutation from a BPF program.
 */
struct nvtelemetry_bridge_header {
    uint8_t magic[8]; /* NVMLBPF1 */
    uint32_t version;
    uint32_t reading_count;
    uint64_t generation;
    int64_t published_unix;
    int64_t query_start;
    int64_t query_end;
    uint8_t owner[16];
    char epoch_id[64];
    char clock_domain[64];
    uint32_t device_count;
    uint32_t telemetry_schema;
    uint64_t sample_seq;
};
struct nvtelemetry_bridge_device {
    char uuid[96];
    char mig_parent_uuid[96];
    uint32_t gpu_instance_id;
    uint32_t compute_instance_id;
    uint32_t flags; /* bit 0: GI present; bit 1: CI present */
    uint32_t scope; /* 0 unassessed, 1 unknown, 2 disabled, 3 parent,
                      4 instance, 5 unsupported, 6 permission denied */
    uint8_t reserved[48];
};
struct nvtelemetry_bridge_reading {
    uint32_t device_index;
    uint32_t flags; /* bit 0 integer, 1 sensor timestamp present,
                      2 uncertainty present, 3 query bounds present;
                      bits 4..5: 1 measured, 2 derived, 3 unavailable */
    char name[64];
    char unit[24];
    char unavailable_reason[32];
    uint64_t value_bits; /* IEEE-754 float64 */
    uint64_t int_value;
    int64_t timestamp_monotonic;
    int64_t timestamp_unix;
    int64_t query_start;
    int64_t query_end;
    int64_t source_timestamp;
    int64_t uncertainty;
    char source_domain[64];
};
struct nvtelemetry_bridge_value {
    struct nvtelemetry_bridge_header header;
    struct nvtelemetry_bridge_device devices[8];
    struct nvtelemetry_bridge_reading readings[128];
};
_Static_assert(sizeof(struct nvtelemetry_bridge_header) == 208, "header size");
_Static_assert(sizeof(struct nvtelemetry_bridge_device) == 256, "device size");
_Static_assert(sizeof(struct nvtelemetry_bridge_reading) == 256, "reading size");
_Static_assert(sizeof(struct nvtelemetry_bridge_value) == 35024, "value size");
#endif
