-- Official documented CREATE TABLE bodies; no measured rows.
CREATE TABLE StringIds (
    -- Consolidation of repetitive string values.

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- ID reference value.
    value                       TEXT      NOT NULL                     -- String value.
);
CREATE TABLE META_DATA_EXPORT (
    -- information about nsys export process

    name                        TEXT      NOT NULL,                    -- Name of meta-data record
    value                       TEXT                                   -- Value of meta-data record
);
CREATE TABLE TARGET_INFO_SESSION_START_TIME (
    utcEpochNs                  INTEGER,                               -- UTC Epoch timestamp at start of the capture (ns).
    utcTime                     TEXT,                                  -- Start of the capture in UTC.
    localTime                   TEXT                                   -- Start of the capture in local time of target.
);
CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    deviceId                    INTEGER   NOT NULL,                    -- Device ID.
    contextId                   INTEGER   NOT NULL,                    -- Context ID.
    greenContextId              INTEGER,                               -- Green context ID.
    streamId                    INTEGER   NOT NULL,                    -- Stream ID.
    correlationId               INTEGER,                               -- REFERENCES CUPTI_ACTIVITY_KIND_RUNTIME(correlationId)
    globalPid                   INTEGER,                               -- Serialized GlobalId.
    demangledName               INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Kernel function name w/ templates
    shortName                   INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Base kernel function name
    mangledName                 INTEGER,                               -- REFERENCES StringIds(id) -- Raw C++ mangled kernel function name
    launchType                  INTEGER,                               -- REFERENCES ENUM_CUDA_KERNEL_LAUNCH_TYPE(id)
    cacheConfig                 INTEGER,                               -- REFERENCES ENUM_CUDA_FUNC_CACHE_CONFIG(id)
    registersPerThread          INTEGER   NOT NULL,                    -- Number of registers required for each thread executing the kernel.
    gridX                       INTEGER   NOT NULL,                    -- X-dimension grid size.
    gridY                       INTEGER   NOT NULL,                    -- Y-dimension grid size.
    gridZ                       INTEGER   NOT NULL,                    -- Z-dimension grid size.
    blockX                      INTEGER   NOT NULL,                    -- X-dimension block size.
    blockY                      INTEGER   NOT NULL,                    -- Y-dimension block size.
    blockZ                      INTEGER   NOT NULL,                    -- Z-dimension block size.
    staticSharedMemory          INTEGER   NOT NULL,                    -- Static shared memory allocated for the kernel (B).
    dynamicSharedMemory         INTEGER   NOT NULL,                    -- Dynamic shared memory reserved for the kernel (B).
    localMemoryPerThread        INTEGER   NOT NULL,                    -- Amount of local memory reserved for each thread (B).
    localMemoryTotal            INTEGER   NOT NULL,                    -- Total amount of local memory reserved for the kernel (B).
    gridId                      INTEGER   NOT NULL,                    -- Unique grid ID of the kernel assigned at runtime.
    sharedMemoryExecuted        INTEGER,                               -- Shared memory size set by the driver.
    graphNodeId                 INTEGER,                               -- REFERENCES CUDA_GRAPH_NODE_EVENTS(graphNodeId)
    sharedMemoryLimitConfig     INTEGER                                -- REFERENCES ENUM_CUDA_SHARED_MEM_LIMIT_CONFIG(id)
);
CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    deviceId                    INTEGER   NOT NULL,                    -- Device ID.
    contextId                   INTEGER   NOT NULL,                    -- Context ID.
    greenContextId              INTEGER,                               -- Green context ID.
    streamId                    INTEGER   NOT NULL,                    -- Stream ID.
    correlationId               INTEGER,                               -- REFERENCES CUPTI_ACTIVITY_KIND_RUNTIME(correlationId)
    globalPid                   INTEGER,                               -- Serialized GlobalId.
    bytes                       INTEGER   NOT NULL,                    -- Number of bytes transferred (B).
    copyKind                    INTEGER   NOT NULL,                    -- REFERENCES ENUM_CUDA_MEMCPY_OPER(id)
    deprecatedSrcId             INTEGER,                               -- Deprecated, use srcDeviceId instead.
    srcKind                     INTEGER,                               -- REFERENCES ENUM_CUDA_MEM_KIND(id)
    dstKind                     INTEGER,                               -- REFERENCES ENUM_CUDA_MEM_KIND(id)
    srcDeviceId                 INTEGER,                               -- Source device ID.
    srcContextId                INTEGER,                               -- Source context ID.
    dstDeviceId                 INTEGER,                               -- Destination device ID.
    dstContextId                INTEGER,                               -- Destination context ID.
    migrationCause              INTEGER,                               -- REFERENCES ENUM_CUDA_UNIF_MEM_MIGRATION(id)
    graphNodeId                 INTEGER,                               -- REFERENCES CUDA_GRAPH_NODE_EVENTS(graphNodeId)
    virtualAddress              INTEGER,                               -- Virtual base address of the page/s being transferred.
    copyCount                   INTEGER                                -- The total number of memcopy operations traced in this record. In CUDA MemcpyBatchAsync APIs, multiple memcpy operations may be batched together for optimization purposes based on certain heuristics.
);
CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    deviceId                    INTEGER   NOT NULL,                    -- Device ID.
    contextId                   INTEGER   NOT NULL,                    -- Context ID.
    greenContextId              INTEGER,                               -- Green context ID.
    streamId                    INTEGER   NOT NULL,                    -- Stream ID.
    correlationId               INTEGER,                               -- REFERENCES CUPTI_ACTIVITY_KIND_RUNTIME(correlationId)
    globalPid                   INTEGER,                               -- Serialized GlobalId.
    value                       INTEGER   NOT NULL,                    -- Value assigned to memory.
    bytes                       INTEGER   NOT NULL,                    -- Number of bytes set (B).
    graphNodeId                 INTEGER,                               -- REFERENCES CUDA_GRAPH_NODE_EVENTS(graphNodeId)
    memKind                     INTEGER                                -- REFERENCES ENUM_CUDA_MEM_KIND(id)
);
CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    correlationId               INTEGER,                               -- ID used to identify events that this function call has triggered.
    nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Function name
    returnValue                 INTEGER   NOT NULL,                    -- Return value of the function call.
    callchainId                 INTEGER                                -- REFERENCES CUDA_CALLCHAINS(id)
);
CREATE TABLE NVTX_EVENTS (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER,                               -- Event end timestamp (ns).
    eventType                   INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_TYPE(id)
    rangeId                     INTEGER,                               -- Correlation ID returned from a nvtxRangeStart call.
    category                    INTEGER,                               -- User-controlled ID that can be used to group events.
    color                       INTEGER,                               -- Encoded ARGB color value.
    text                        TEXT,                                  -- Explicit name/text (non-registered string)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    endGlobalTid                INTEGER,                               -- Serialized GlobalId.
    textId                      INTEGER,                               -- REFERENCES StringIds(id) -- Registered NVTX domain/string
    domainId                    INTEGER,                               -- User-controlled ID that can be used to group events.
    uint64Value                 INTEGER,                               -- One of possible payload value union members.
    int64Value                  INTEGER,                               -- One of possible payload value union members.
    doubleValue                 REAL,                                  -- One of possible payload value union members.
    uint32Value                 INTEGER,                               -- One of possible payload value union members.
    int32Value                  INTEGER,                               -- One of possible payload value union members.
    floatValue                  REAL,                                  -- One of possible payload value union members.
    jsonTextId                  INTEGER,                               -- One of possible payload value union members.
    jsonText                    TEXT,                                  -- One of possible payload value union members.
    binaryData                  TEXT                                   -- Binary payload. See docs for format.
);
CREATE TABLE OSRT_API (
    -- OS runtime libraries traced to gather information about low-level userspace APIs.

    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Function name
    returnValue                 INTEGER   NOT NULL,                    -- Return value of the function call.
    nestingLevel                INTEGER,                               -- Zero-base index of the nesting level.
    callchainId                 INTEGER   NOT NULL                     -- REFERENCES OSRT_CALLCHAINS(id)
);
