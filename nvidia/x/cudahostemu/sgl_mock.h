#pragma once
#include "cudahostemu.h"

// Host mock of the parts of SGLang's JIT headers (sgl_kernel/tensor.h,
// sgl_kernel/utils.cuh) and tvm-ffi that SourceTVMFFI's structs use, so a
// generated struct runs on the host shim. Signatures follow SGLang's: sizes
// are SymbolicSize references or int64 literals (-1 matches any), and
// RuntimeCheck failures throw. It includes cudahostemu.h.
#include <cstdint>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

enum DLDeviceType { kDLCPU = 1, kDLCUDA = 2 };
enum DLDataTypeCode { kDLInt = 0, kDLUInt = 1, kDLFloat = 2, kDLBfloat = 4, kDLFloat8_e4m3fn = 10 };
struct DLDataType { uint8_t code, bits; uint16_t lanes; };
struct DLDevice { int device_type, device_id; };

namespace tvm::ffi {
struct TensorView {
    void* ptr;
    std::vector<int64_t> shape, strides;
    DLDataType dtype;
    DLDevice device = {kDLCUDA, 0};
    void* data_ptr() const { return ptr; }
    int64_t ndim() const { return (int64_t)shape.size(); }
    int64_t size(int i) const { return shape[i < 0 ? (int)shape.size() + i : i]; }
    int64_t stride(int i) const { return strides[i < 0 ? (int)strides.size() + i : i]; }
};
}  // namespace tvm::ffi

using fp32_t = float;
using bf16_t = __nv_bfloat16;
struct fp8_e4m3_t { uint8_t b; };

namespace sglang::host {

template <typename... Args>
void RuntimeCheck(bool ok, Args&&... args) {
    if (ok) return;
    std::ostringstream s;
    (s << ... << args);
    throw std::runtime_error(s.str());
}

template <typename T> constexpr DLDataType dl_type();
template <> constexpr DLDataType dl_type<float>() { return {kDLFloat, 32, 1}; }
template <> constexpr DLDataType dl_type<__nv_bfloat16>() { return {kDLBfloat, 16, 1}; }
template <> constexpr DLDataType dl_type<fp8_e4m3_t>() { return {kDLFloat8_e4m3fn, 8, 1}; }
template <> constexpr DLDataType dl_type<uint8_t>() { return {kDLUInt, 8, 1}; }
template <> constexpr DLDataType dl_type<int32_t>() { return {kDLInt, 32, 1}; }
template <> constexpr DLDataType dl_type<int64_t>() { return {kDLInt, 64, 1}; }

struct SymbolicSize {
    SymbolicSize(std::string_view name = {}) : name(name) {}
    SymbolicSize(const SymbolicSize&) = delete;
    void verify(int64_t v) {
        if (!set) { value = v, set = true; return; }
        RuntimeCheck(value == v, name, ": ", v, " != ", value);
    }
    int64_t unwrap() const { RuntimeCheck(set, name, ": unset"); return value; }
    std::string_view name;
    int64_t value = -1;
    bool set = false;
};

struct SymbolicDevice {
    template <DLDeviceType... Codes>
    void set_options() { code = (Codes, ...); }
    void verify(DLDevice d) {
        RuntimeCheck(d.device_type == code, "device type ", d.device_type);
        if (!set) { value = d, set = true; return; }
        RuntimeCheck(d.device_id == value.device_id, "device ", d.device_id, " != ", value.device_id);
    }
    DLDevice unwrap() const { RuntimeCheck(set, "device unset"); return value; }
    int code = 0;
    DLDevice value{};
    bool set = false;
};

struct SizeRef {
    SizeRef(SymbolicSize& s) : ref(&s) {}
    SizeRef(int64_t v) : ref(&own) { if (v != -1) own.verify(v); }
    SizeRef(int v) : SizeRef((int64_t)v) {}
    SizeRef(const SizeRef& o) : ref(o.ref == &o.own ? &own : o.ref) {
        if (o.ref == &o.own && o.own.set) own.verify(o.own.value);
    }
    // Assignment, like copying, must not leave ref pointing at o.own.
    SizeRef& operator=(const SizeRef& o) {
        if (o.ref != &o.own) { ref = o.ref; return *this; }
        own.value = o.own.value, own.set = o.own.set, ref = &own;
        return *this;
    }
    SymbolicSize* ref;
    SymbolicSize own;
};

struct TensorMatcher {
    explicit TensorMatcher(std::initializer_list<SizeRef> shape) : shape(shape) {}
    TensorMatcher&& with_strides(std::initializer_list<SizeRef> s) && { strides.assign(s); return std::move(*this); }
    template <typename T>
    TensorMatcher&& with_dtype() && { dtype = dl_type<T>(); return std::move(*this); }
    TensorMatcher&& with_device(SymbolicDevice& d) && { device = &d; return std::move(*this); }
    TensorMatcher&& verify(const tvm::ffi::TensorView& t) && {
        RuntimeCheck(t.ndim() == (int64_t)shape.size(), "ndim ", t.ndim(), " != ", shape.size());
        for (size_t i = 0; i < shape.size(); ++i) {
            shape[i].ref->verify(t.size((int)i));
            if (!strides.empty()) {
                int64_t got = t.stride((int)i);
                SymbolicSize* sr = strides[i].ref;
                if (sr->set && sr->value != got) {
                    RuntimeCheck(false, "stride ", i, ": ", got, " != ", sr->value);
                }
                sr->verify(got);
            }
        }
        RuntimeCheck(t.dtype.code == dtype.code && t.dtype.bits == dtype.bits, "dtype ", (int)t.dtype.code, "/", (int)t.dtype.bits);
        if (device) device->verify(t.device);
        return std::move(*this);
    }
    std::vector<SizeRef> shape, strides;
    DLDataType dtype{};
    SymbolicDevice* device = nullptr;
};

struct LaunchKernel {
    LaunchKernel(dim3 grid, dim3 block, DLDevice) : grid(grid), block(block) {}
    template <typename K, typename... Args>
    void operator()(K&& kernel, Args&&... args) const {
        const void* p = reinterpret_cast<const void*>(kernel);
        if (shim_cluster_kernels.count(p)) {
            auto it = eg_emu_cluster_sizes.find(p);
            int cluster = (it != eg_emu_cluster_sizes.end()) ? it->second : 8;
            run_cluster_grid(grid, block, cluster, [&] { kernel(args...); });
        } else {
            run_grid(grid, block, [&] { kernel(args...); });
        }
    }
    dim3 grid, block;
};

}  // namespace sglang::host
