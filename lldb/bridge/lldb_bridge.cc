#include "lldb_bridge.h"
#include "lldb/API/LLDB.h"
#include <algorithm>
#include <exception>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

// bridgeNarrow converts a bridge value to a field's own type, refusing
// values the field cannot hold. See contrib/cpp/narrow.go.
template <typename To, typename From>
static To bridgeNarrow(From v, const char* field) {
	if constexpr (std::is_integral_v<To> && std::is_integral_v<From>) {
		if constexpr (std::is_signed_v<From> && !std::is_signed_v<To>) {
			if (v < 0) {
				throw std::out_of_range(std::string("negative value for unsigned field ") + field);
			}
		}
		To r = static_cast<To>(v);
		if (static_cast<From>(r) != v) {
			throw std::out_of_range(std::string("value out of range for field ") + field);
		}
		return r;
	} else {
		return static_cast<To>(v);
	}
}

static thread_local std::string gLastError;
static thread_local std::string gStringReturn;

extern "C" const char* LLDBLastError(void) {
	return gLastError.empty() ? nullptr : gLastError.c_str();
}

extern "C" LLDBSBErrorRef LLDBSBErrorNew(void) {
	gLastError.clear();
	try {
	return reinterpret_cast<LLDBSBErrorRef>(new lldb::SBError());
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return nullptr;
}

extern "C" void LLDBSBErrorDelete(LLDBSBErrorRef self) {
	gLastError.clear();
	try {
	delete reinterpret_cast<lldb::SBError*>(self);
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
}

extern "C" void LLDBSBErrorClear(LLDBSBErrorRef self) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBError*>(self);
	obj->Clear();
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
}

extern "C" bool LLDBSBErrorFail(LLDBSBErrorRef self) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBError*>(self);
	return obj->Fail();
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return false;
}

extern "C" bool LLDBSBErrorSuccess(LLDBSBErrorRef self) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBError*>(self);
	return obj->Success();
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return false;
}

extern "C" bool LLDBSBErrorIsValid(LLDBSBErrorRef self) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBError*>(self);
	return obj->IsValid();
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return false;
}

extern "C" uint32_t LLDBSBErrorGetError(LLDBSBErrorRef self) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBError*>(self);
	return obj->GetError();
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return 0;
}

extern "C" const char* LLDBSBErrorGetCString(LLDBSBErrorRef self) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBError*>(self);
	gStringReturn = obj->GetCString();
	return gStringReturn.c_str();
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return nullptr;
}

extern "C" void LLDBSBDebuggerDelete(LLDBSBDebuggerRef self) {
	gLastError.clear();
	try {
	delete reinterpret_cast<lldb::SBDebugger*>(self);
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
}

extern "C" void LLDBSBDebuggerInitialize(void) {
	gLastError.clear();
	try {
	lldb::SBDebugger::Initialize();
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
}

extern "C" void LLDBSBDebuggerTerminate(void) {
	gLastError.clear();
	try {
	lldb::SBDebugger::Terminate();
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
}

extern "C" LLDBSBDebuggerRef LLDBSBDebuggerCreateWithInitFiles(bool sourceInitFiles) {
	gLastError.clear();
	try {
	return reinterpret_cast<LLDBSBDebuggerRef>(new lldb::SBDebugger(static_cast<const lldb::SBDebugger&>(lldb::SBDebugger::Create(sourceInitFiles))));
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return nullptr;
}

extern "C" void LLDBSBDebuggerDestroy(LLDBSBDebuggerRef debugger) {
	gLastError.clear();
	try {
	lldb::SBDebugger::Destroy(*reinterpret_cast<lldb::SBDebugger*>(debugger));
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
}

extern "C" bool LLDBSBDebuggerIsValid(LLDBSBDebuggerRef self) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBDebugger*>(self);
	return obj->IsValid();
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return false;
}

extern "C" void LLDBSBDebuggerClear(LLDBSBDebuggerRef self) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBDebugger*>(self);
	obj->Clear();
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
}

extern "C" void LLDBSBDebuggerSetAsync(LLDBSBDebuggerRef self, bool enabled) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBDebugger*>(self);
	obj->SetAsync(enabled);
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
}

extern "C" bool LLDBSBDebuggerGetAsync(LLDBSBDebuggerRef self) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBDebugger*>(self);
	return obj->GetAsync();
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return false;
}

extern "C" void LLDBSBDebuggerSkipLLDBInitFiles(LLDBSBDebuggerRef self, bool skip) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBDebugger*>(self);
	obj->SkipLLDBInitFiles(skip);
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
}

extern "C" void LLDBSBDebuggerSkipAppInitFiles(LLDBSBDebuggerRef self, bool skip) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBDebugger*>(self);
	obj->SkipAppInitFiles(skip);
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
}

extern "C" void LLDBSBDebuggerHandleCommand(LLDBSBDebuggerRef self, const char* command) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBDebugger*>(self);
	obj->HandleCommand(command);
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
}

extern "C" LLDBSBTargetRef LLDBSBDebuggerCreateTargetSimple(LLDBSBDebuggerRef self, const char* filename) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBDebugger*>(self);
	return reinterpret_cast<LLDBSBTargetRef>(new lldb::SBTarget(static_cast<const lldb::SBTarget&>(obj->CreateTarget(filename))));
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return nullptr;
}

extern "C" LLDBSBTargetRef LLDBSBDebuggerGetSelectedTarget(LLDBSBDebuggerRef self) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBDebugger*>(self);
	return reinterpret_cast<LLDBSBTargetRef>(new lldb::SBTarget(static_cast<const lldb::SBTarget&>(obj->GetSelectedTarget())));
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return nullptr;
}

extern "C" uint32_t LLDBSBDebuggerGetNumTargets(LLDBSBDebuggerRef self) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBDebugger*>(self);
	return obj->GetNumTargets();
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return 0;
}

extern "C" const char* LLDBSBDebuggerGetVersionString(void) {
	gLastError.clear();
	try {
	gStringReturn = lldb::SBDebugger::GetVersionString();
	return gStringReturn.c_str();
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return nullptr;
}

extern "C" void LLDBSBTargetDelete(LLDBSBTargetRef self) {
	gLastError.clear();
	try {
	delete reinterpret_cast<lldb::SBTarget*>(self);
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
}

extern "C" bool LLDBSBTargetIsValid(LLDBSBTargetRef self) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBTarget*>(self);
	return obj->IsValid();
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return false;
}

extern "C" LLDBSBProcessRef LLDBSBTargetGetProcess(LLDBSBTargetRef self) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBTarget*>(self);
	return reinterpret_cast<LLDBSBProcessRef>(new lldb::SBProcess(static_cast<const lldb::SBProcess&>(obj->GetProcess())));
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return nullptr;
}

extern "C" uint32_t LLDBSBTargetGetNumModules(LLDBSBTargetRef self) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBTarget*>(self);
	return obj->GetNumModules();
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return 0;
}

extern "C" void LLDBSBTargetDeleteAllBreakpoints(LLDBSBTargetRef self) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBTarget*>(self);
	obj->DeleteAllBreakpoints();
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
}

extern "C" LLDBSBBreakpointRef LLDBSBTargetBreakpointCreateByNameSimple(LLDBSBTargetRef self, const char* symbol) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBTarget*>(self);
	return reinterpret_cast<LLDBSBBreakpointRef>(new lldb::SBBreakpoint(static_cast<const lldb::SBBreakpoint&>(obj->BreakpointCreateByName(symbol))));
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return nullptr;
}

extern "C" LLDBSBBreakpointRef LLDBSBTargetBreakpointCreateByLocationSimple(LLDBSBTargetRef self, const char* file, uint32_t line) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBTarget*>(self);
	return reinterpret_cast<LLDBSBBreakpointRef>(new lldb::SBBreakpoint(static_cast<const lldb::SBBreakpoint&>(obj->BreakpointCreateByLocation(file, line))));
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return nullptr;
}

extern "C" void LLDBSBProcessDelete(LLDBSBProcessRef self) {
	gLastError.clear();
	try {
	delete reinterpret_cast<lldb::SBProcess*>(self);
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
}

extern "C" bool LLDBSBProcessIsValid(LLDBSBProcessRef self) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBProcess*>(self);
	return obj->IsValid();
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return false;
}

extern "C" uint64_t LLDBSBProcessGetProcessID(LLDBSBProcessRef self) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBProcess*>(self);
	return obj->GetProcessID();
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return 0;
}

extern "C" int32_t LLDBSBProcessGetState(LLDBSBProcessRef self) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBProcess*>(self);
	return obj->GetState();
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return 0;
}

extern "C" uint32_t LLDBSBProcessGetNumThreads(LLDBSBProcessRef self) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBProcess*>(self);
	return obj->GetNumThreads();
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return 0;
}

extern "C" LLDBSBThreadRef LLDBSBProcessGetThreadAtIndex(LLDBSBProcessRef self, uint32_t index) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBProcess*>(self);
	return reinterpret_cast<LLDBSBThreadRef>(new lldb::SBThread(static_cast<const lldb::SBThread&>(obj->GetThreadAtIndex(index))));
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return nullptr;
}

extern "C" LLDBSBThreadRef LLDBSBProcessGetSelectedThread(LLDBSBProcessRef self) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBProcess*>(self);
	return reinterpret_cast<LLDBSBThreadRef>(new lldb::SBThread(static_cast<const lldb::SBThread&>(obj->GetSelectedThread())));
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return nullptr;
}

extern "C" bool LLDBSBProcessSetSelectedThreadByID(LLDBSBProcessRef self, uint64_t threadID) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBProcess*>(self);
	return obj->SetSelectedThreadByID(threadID);
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return false;
}

extern "C" LLDBSBErrorRef LLDBSBProcessContinue(LLDBSBProcessRef self) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBProcess*>(self);
	return reinterpret_cast<LLDBSBErrorRef>(new lldb::SBError(static_cast<const lldb::SBError&>(obj->Continue())));
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return nullptr;
}

extern "C" LLDBSBErrorRef LLDBSBProcessStop(LLDBSBProcessRef self) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBProcess*>(self);
	return reinterpret_cast<LLDBSBErrorRef>(new lldb::SBError(static_cast<const lldb::SBError&>(obj->Stop())));
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return nullptr;
}

extern "C" LLDBSBErrorRef LLDBSBProcessKill(LLDBSBProcessRef self) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBProcess*>(self);
	return reinterpret_cast<LLDBSBErrorRef>(new lldb::SBError(static_cast<const lldb::SBError&>(obj->Kill())));
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return nullptr;
}

extern "C" LLDBSBErrorRef LLDBSBProcessDetachSimple(LLDBSBProcessRef self) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBProcess*>(self);
	return reinterpret_cast<LLDBSBErrorRef>(new lldb::SBError(static_cast<const lldb::SBError&>(obj->Detach())));
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return nullptr;
}

extern "C" LLDBSBErrorRef LLDBSBProcessSignal(LLDBSBProcessRef self, int32_t signal) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBProcess*>(self);
	return reinterpret_cast<LLDBSBErrorRef>(new lldb::SBError(static_cast<const lldb::SBError&>(obj->Signal(signal))));
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return nullptr;
}

extern "C" size_t LLDBSBProcessPutSTDIN(LLDBSBProcessRef self, uint8_t* data, size_t data_len) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBProcess*>(self);
	return obj->PutSTDIN(reinterpret_cast<const char*>(data), data_len);
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return 0;
}

extern "C" size_t LLDBSBProcessGetSTDOUT(LLDBSBProcessRef self, uint8_t* data, size_t data_len) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBProcess*>(self);
	return obj->GetSTDOUT(reinterpret_cast<char*>(data), data_len);
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return 0;
}

extern "C" size_t LLDBSBProcessGetSTDERR(LLDBSBProcessRef self, uint8_t* data, size_t data_len) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBProcess*>(self);
	return obj->GetSTDERR(reinterpret_cast<char*>(data), data_len);
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return 0;
}

extern "C" size_t LLDBSBProcessReadMemory(LLDBSBProcessRef self, uint64_t address, uint8_t* data, size_t data_len, LLDBSBErrorRef error) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBProcess*>(self);
	return obj->ReadMemory(address, data, data_len, *reinterpret_cast<lldb::SBError*>(error));
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return 0;
}

extern "C" size_t LLDBSBProcessWriteMemory(LLDBSBProcessRef self, uint64_t address, uint8_t* data, size_t data_len, LLDBSBErrorRef error) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBProcess*>(self);
	return obj->WriteMemory(address, reinterpret_cast<const void*>(data), data_len, *reinterpret_cast<lldb::SBError*>(error));
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return 0;
}

extern "C" void LLDBSBThreadDelete(LLDBSBThreadRef self) {
	gLastError.clear();
	try {
	delete reinterpret_cast<lldb::SBThread*>(self);
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
}

extern "C" bool LLDBSBThreadIsValid(LLDBSBThreadRef self) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBThread*>(self);
	return obj->IsValid();
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return false;
}

extern "C" uint64_t LLDBSBThreadGetThreadID(LLDBSBThreadRef self) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBThread*>(self);
	return obj->GetThreadID();
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return 0;
}

extern "C" uint32_t LLDBSBThreadGetIndexID(LLDBSBThreadRef self) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBThread*>(self);
	return obj->GetIndexID();
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return 0;
}

extern "C" uint32_t LLDBSBThreadGetNumFrames(LLDBSBThreadRef self) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBThread*>(self);
	return obj->GetNumFrames();
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return 0;
}

extern "C" LLDBSBFrameRef LLDBSBThreadGetFrameAtIndex(LLDBSBThreadRef self, uint32_t index) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBThread*>(self);
	return reinterpret_cast<LLDBSBFrameRef>(new lldb::SBFrame(static_cast<const lldb::SBFrame&>(obj->GetFrameAtIndex(index))));
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return nullptr;
}

extern "C" int32_t LLDBSBThreadGetStopReason(LLDBSBThreadRef self) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBThread*>(self);
	return obj->GetStopReason();
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return 0;
}

extern "C" void LLDBSBFrameDelete(LLDBSBFrameRef self) {
	gLastError.clear();
	try {
	delete reinterpret_cast<lldb::SBFrame*>(self);
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
}

extern "C" bool LLDBSBFrameIsValid(LLDBSBFrameRef self) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBFrame*>(self);
	return obj->IsValid();
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return false;
}

extern "C" const char* LLDBSBFrameGetFunctionName(LLDBSBFrameRef self) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBFrame*>(self);
	gStringReturn = obj->GetFunctionName();
	return gStringReturn.c_str();
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return nullptr;
}

extern "C" const char* LLDBSBFrameGetDisplayFunctionName(LLDBSBFrameRef self) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBFrame*>(self);
	gStringReturn = obj->GetDisplayFunctionName();
	return gStringReturn.c_str();
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return nullptr;
}

extern "C" uint64_t LLDBSBFrameGetPC(LLDBSBFrameRef self) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBFrame*>(self);
	return obj->GetPC();
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return 0;
}

extern "C" LLDBSBValueRef LLDBSBFrameEvaluateExpressionSimple(LLDBSBFrameRef self, const char* expression) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBFrame*>(self);
	return reinterpret_cast<LLDBSBValueRef>(new lldb::SBValue(static_cast<const lldb::SBValue&>(obj->EvaluateExpression(expression))));
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return nullptr;
}

extern "C" void LLDBSBValueDelete(LLDBSBValueRef self) {
	gLastError.clear();
	try {
	delete reinterpret_cast<lldb::SBValue*>(self);
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
}

extern "C" bool LLDBSBValueIsValid(LLDBSBValueRef self) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBValue*>(self);
	return obj->IsValid();
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return false;
}

extern "C" const char* LLDBSBValueGetName(LLDBSBValueRef self) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBValue*>(self);
	gStringReturn = obj->GetName();
	return gStringReturn.c_str();
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return nullptr;
}

extern "C" const char* LLDBSBValueGetValue(LLDBSBValueRef self) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBValue*>(self);
	gStringReturn = obj->GetValue();
	return gStringReturn.c_str();
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return nullptr;
}

extern "C" const char* LLDBSBValueGetSummary(LLDBSBValueRef self) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBValue*>(self);
	gStringReturn = obj->GetSummary();
	return gStringReturn.c_str();
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return nullptr;
}

extern "C" uint32_t LLDBSBValueGetNumChildren(LLDBSBValueRef self) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBValue*>(self);
	return obj->GetNumChildren();
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return 0;
}

extern "C" LLDBSBValueRef LLDBSBValueGetChildAtIndex(LLDBSBValueRef self, uint32_t index) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBValue*>(self);
	return reinterpret_cast<LLDBSBValueRef>(new lldb::SBValue(static_cast<const lldb::SBValue&>(obj->GetChildAtIndex(index))));
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return nullptr;
}

extern "C" void LLDBSBBreakpointDelete(LLDBSBBreakpointRef self) {
	gLastError.clear();
	try {
	delete reinterpret_cast<lldb::SBBreakpoint*>(self);
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
}

extern "C" bool LLDBSBBreakpointIsValid(LLDBSBBreakpointRef self) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBBreakpoint*>(self);
	return obj->IsValid();
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return false;
}

extern "C" uint32_t LLDBSBBreakpointGetID(LLDBSBBreakpointRef self) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBBreakpoint*>(self);
	return obj->GetID();
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return 0;
}

extern "C" void LLDBSBBreakpointSetEnabled(LLDBSBBreakpointRef self, bool enabled) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBBreakpoint*>(self);
	obj->SetEnabled(enabled);
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
}

extern "C" bool LLDBSBBreakpointIsEnabled(LLDBSBBreakpointRef self) {
	gLastError.clear();
	try {
	auto* obj = reinterpret_cast<lldb::SBBreakpoint*>(self);
	return obj->IsEnabled();
	} catch (const std::exception& e) {
		gLastError = e.what();
	} catch (...) {
		gLastError = "unknown C++ exception";
	}
	return false;
}

