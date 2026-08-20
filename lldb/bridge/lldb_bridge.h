#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LLDBSBBreakpointOpaque* LLDBSBBreakpointRef;
typedef struct LLDBSBDebuggerOpaque* LLDBSBDebuggerRef;
typedef struct LLDBSBErrorOpaque* LLDBSBErrorRef;
typedef struct LLDBSBFrameOpaque* LLDBSBFrameRef;
typedef struct LLDBSBProcessOpaque* LLDBSBProcessRef;
typedef struct LLDBSBTargetOpaque* LLDBSBTargetRef;
typedef struct LLDBSBThreadOpaque* LLDBSBThreadRef;
typedef struct LLDBSBValueOpaque* LLDBSBValueRef;

const char* LLDBLastError(void);
LLDBSBErrorRef LLDBSBErrorNew(void);
void LLDBSBErrorDelete(LLDBSBErrorRef self);
void LLDBSBErrorClear(LLDBSBErrorRef self);
bool LLDBSBErrorFail(LLDBSBErrorRef self);
bool LLDBSBErrorSuccess(LLDBSBErrorRef self);
bool LLDBSBErrorIsValid(LLDBSBErrorRef self);
uint32_t LLDBSBErrorGetError(LLDBSBErrorRef self);
const char* LLDBSBErrorGetCString(LLDBSBErrorRef self);
void LLDBSBDebuggerDelete(LLDBSBDebuggerRef self);
void LLDBSBDebuggerInitialize(void);
void LLDBSBDebuggerTerminate(void);
LLDBSBDebuggerRef LLDBSBDebuggerCreateWithInitFiles(bool sourceInitFiles);
void LLDBSBDebuggerDestroy(LLDBSBDebuggerRef debugger);
bool LLDBSBDebuggerIsValid(LLDBSBDebuggerRef self);
void LLDBSBDebuggerClear(LLDBSBDebuggerRef self);
void LLDBSBDebuggerSetAsync(LLDBSBDebuggerRef self, bool enabled);
bool LLDBSBDebuggerGetAsync(LLDBSBDebuggerRef self);
void LLDBSBDebuggerSkipLLDBInitFiles(LLDBSBDebuggerRef self, bool skip);
void LLDBSBDebuggerSkipAppInitFiles(LLDBSBDebuggerRef self, bool skip);
void LLDBSBDebuggerHandleCommand(LLDBSBDebuggerRef self, const char* command);
LLDBSBTargetRef LLDBSBDebuggerCreateTargetSimple(LLDBSBDebuggerRef self, const char* filename);
LLDBSBTargetRef LLDBSBDebuggerGetSelectedTarget(LLDBSBDebuggerRef self);
uint32_t LLDBSBDebuggerGetNumTargets(LLDBSBDebuggerRef self);
const char* LLDBSBDebuggerGetVersionString(void);
void LLDBSBTargetDelete(LLDBSBTargetRef self);
bool LLDBSBTargetIsValid(LLDBSBTargetRef self);
LLDBSBProcessRef LLDBSBTargetGetProcess(LLDBSBTargetRef self);
uint32_t LLDBSBTargetGetNumModules(LLDBSBTargetRef self);
void LLDBSBTargetDeleteAllBreakpoints(LLDBSBTargetRef self);
LLDBSBBreakpointRef LLDBSBTargetBreakpointCreateByNameSimple(LLDBSBTargetRef self, const char* symbol);
LLDBSBBreakpointRef LLDBSBTargetBreakpointCreateByLocationSimple(LLDBSBTargetRef self, const char* file, uint32_t line);
void LLDBSBProcessDelete(LLDBSBProcessRef self);
bool LLDBSBProcessIsValid(LLDBSBProcessRef self);
uint64_t LLDBSBProcessGetProcessID(LLDBSBProcessRef self);
int32_t LLDBSBProcessGetState(LLDBSBProcessRef self);
uint32_t LLDBSBProcessGetNumThreads(LLDBSBProcessRef self);
LLDBSBThreadRef LLDBSBProcessGetThreadAtIndex(LLDBSBProcessRef self, uint32_t index);
LLDBSBThreadRef LLDBSBProcessGetSelectedThread(LLDBSBProcessRef self);
bool LLDBSBProcessSetSelectedThreadByID(LLDBSBProcessRef self, uint64_t threadID);
LLDBSBErrorRef LLDBSBProcessContinue(LLDBSBProcessRef self);
LLDBSBErrorRef LLDBSBProcessStop(LLDBSBProcessRef self);
LLDBSBErrorRef LLDBSBProcessKill(LLDBSBProcessRef self);
LLDBSBErrorRef LLDBSBProcessDetachSimple(LLDBSBProcessRef self);
LLDBSBErrorRef LLDBSBProcessSignal(LLDBSBProcessRef self, int32_t signal);
size_t LLDBSBProcessPutSTDIN(LLDBSBProcessRef self, uint8_t* data, size_t data_len);
size_t LLDBSBProcessGetSTDOUT(LLDBSBProcessRef self, uint8_t* data, size_t data_len);
size_t LLDBSBProcessGetSTDERR(LLDBSBProcessRef self, uint8_t* data, size_t data_len);
size_t LLDBSBProcessReadMemory(LLDBSBProcessRef self, uint64_t address, uint8_t* data, size_t data_len, LLDBSBErrorRef error);
size_t LLDBSBProcessWriteMemory(LLDBSBProcessRef self, uint64_t address, uint8_t* data, size_t data_len, LLDBSBErrorRef error);
void LLDBSBThreadDelete(LLDBSBThreadRef self);
bool LLDBSBThreadIsValid(LLDBSBThreadRef self);
uint64_t LLDBSBThreadGetThreadID(LLDBSBThreadRef self);
uint32_t LLDBSBThreadGetIndexID(LLDBSBThreadRef self);
uint32_t LLDBSBThreadGetNumFrames(LLDBSBThreadRef self);
LLDBSBFrameRef LLDBSBThreadGetFrameAtIndex(LLDBSBThreadRef self, uint32_t index);
int32_t LLDBSBThreadGetStopReason(LLDBSBThreadRef self);
void LLDBSBFrameDelete(LLDBSBFrameRef self);
bool LLDBSBFrameIsValid(LLDBSBFrameRef self);
const char* LLDBSBFrameGetFunctionName(LLDBSBFrameRef self);
const char* LLDBSBFrameGetDisplayFunctionName(LLDBSBFrameRef self);
uint64_t LLDBSBFrameGetPC(LLDBSBFrameRef self);
LLDBSBValueRef LLDBSBFrameEvaluateExpressionSimple(LLDBSBFrameRef self, const char* expression);
void LLDBSBValueDelete(LLDBSBValueRef self);
bool LLDBSBValueIsValid(LLDBSBValueRef self);
const char* LLDBSBValueGetName(LLDBSBValueRef self);
const char* LLDBSBValueGetValue(LLDBSBValueRef self);
const char* LLDBSBValueGetSummary(LLDBSBValueRef self);
uint32_t LLDBSBValueGetNumChildren(LLDBSBValueRef self);
LLDBSBValueRef LLDBSBValueGetChildAtIndex(LLDBSBValueRef self, uint32_t index);
void LLDBSBBreakpointDelete(LLDBSBBreakpointRef self);
bool LLDBSBBreakpointIsValid(LLDBSBBreakpointRef self);
uint32_t LLDBSBBreakpointGetID(LLDBSBBreakpointRef self);
void LLDBSBBreakpointSetEnabled(LLDBSBBreakpointRef self, bool enabled);
bool LLDBSBBreakpointIsEnabled(LLDBSBBreakpointRef self);
#ifdef __cplusplus
}
#endif
