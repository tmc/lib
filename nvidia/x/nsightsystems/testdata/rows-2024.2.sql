-- Synthetic rows for host import controls, not measured Nsight output.
INSERT INTO "META_DATA_EXPORT" ("name","value") VALUES ('EXPORT_SCHEMA_VERSION','3.11.0');
INSERT INTO "TARGET_INFO_SESSION_START_TIME" ("utcEpochNs","utcTime","localTime") VALUES (1700000000000000000,NULL,NULL);
INSERT INTO "StringIds" ("id","value") VALUES (1,'kernel<float>');
INSERT INTO "StringIds" ("id","value") VALUES (2,'cudaLaunchKernel');
INSERT INTO "StringIds" ("id","value") VALUES (3,'pthread_mutex_lock');
INSERT INTO "CUPTI_ACTIVITY_KIND_RUNTIME" ("start","end","eventClass","globalTid","correlationId","nameId","returnValue","callchainId") VALUES (90,95,0,562950876168193,7,2,0,NULL);
INSERT INTO "CUPTI_ACTIVITY_KIND_KERNEL" ("start","end","deviceId","contextId","streamId","correlationId","globalPid","demangledName","shortName","mangledName","launchType","cacheConfig","registersPerThread","gridX","gridY","gridZ","blockX","blockY","blockZ","staticSharedMemory","dynamicSharedMemory","localMemoryPerThread","localMemoryTotal","gridId","sharedMemoryExecuted","graphNodeId","sharedMemoryLimitConfig") VALUES (100,130,0,0,0,7,562950876168192,1,1,NULL,NULL,NULL,0,0,0,0,0,0,0,0,0,0,0,0,NULL,NULL,NULL);
INSERT INTO "CUPTI_ACTIVITY_KIND_MEMCPY" ("start","end","deviceId","contextId","streamId","correlationId","globalPid","bytes","copyKind","deprecatedSrcId","srcKind","dstKind","srcDeviceId","srcContextId","dstDeviceId","dstContextId","migrationCause","graphNodeId","virtualAddress") VALUES (140,150,0,0,0,8,562950876168192,64,0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
INSERT INTO "CUPTI_ACTIVITY_KIND_MEMSET" ("start","end","deviceId","contextId","streamId","correlationId","globalPid","value","bytes","graphNodeId","memKind") VALUES (160,170,0,0,0,NULL,NULL,0,128,NULL,NULL);
INSERT INTO "NVTX_EVENTS" ("start","end","eventType","rangeId","category","color","text","globalTid","endGlobalTid","textId","domainId","uint64Value","int64Value","doubleValue","uint32Value","int32Value","floatValue","jsonTextId","jsonText") VALUES (80,NULL,0,NULL,NULL,NULL,'nccl:AllReduce',562950876168193,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,'{"synthetic":true}');
INSERT INTO "OSRT_API" ("start","end","eventClass","globalTid","nameId","returnValue","nestingLevel","callchainId") VALUES (85,90,0,562950876168193,3,0,NULL,0);
