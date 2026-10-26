// condgraph.h: CUDA 12.4+/13.x Conditional Graph (cudaGraphConditionalHandle /
// cudaGraphSetConditional) device-side loop controller emulation for cudahostemu.
#ifndef EG_EMU_CONDGRAPH_H
#define EG_EMU_CONDGRAPH_H

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <map>
#include <mutex>
#include <vector>

#ifndef EG_CUDA_GRAPH_CONDITIONAL_HANDLE_DEFINED
#define EG_CUDA_GRAPH_CONDITIONAL_HANDLE_DEFINED 1
typedef unsigned long long cudaGraphConditionalHandle;
#endif

struct eg_condgraph_handle_state {
    unsigned int value = 0;
    int update_count = 0;
    int iter_updates = 0;
    int last_writer_cta = -1;
    int last_writer_tid = -1;
    int race_count = 0;
};

inline std::mutex eg_condgraph_mu;
inline std::map<cudaGraphConditionalHandle, eg_condgraph_handle_state> eg_condgraph_handles;
inline bool eg_condgraph_detect_race = false;

inline void eg_condgraph_reset() {
    std::lock_guard<std::mutex> lock(eg_condgraph_mu);
    eg_condgraph_handles.clear();
    eg_condgraph_detect_race = false;
}

inline void eg_condgraph_set_detect_race(bool enable) {
    std::lock_guard<std::mutex> lock(eg_condgraph_mu);
    eg_condgraph_detect_race = enable;
}

inline void eg_condgraph_init(cudaGraphConditionalHandle handle, unsigned int initial_value) {
    std::lock_guard<std::mutex> lock(eg_condgraph_mu);
    eg_condgraph_handle_state st{};
    st.value = initial_value;
    eg_condgraph_handles[handle] = st;
}

inline void eg_condgraph_begin_iter(cudaGraphConditionalHandle handle) {
    std::lock_guard<std::mutex> lock(eg_condgraph_mu);
    auto& st = eg_condgraph_handles[handle];
    st.iter_updates = 0;
    st.last_writer_cta = -1;
    st.last_writer_tid = -1;
}

inline void eg_condgraph_set(cudaGraphConditionalHandle handle, unsigned int value) {
    std::lock_guard<std::mutex> lock(eg_condgraph_mu);
    auto& st = eg_condgraph_handles[handle];
    const int cta = static_cast<int>((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x);
    const int tid = static_cast<int>((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x);
    if (eg_condgraph_detect_race && st.iter_updates > 0) {
        if (st.last_writer_cta != cta || st.last_writer_tid != tid) {
            ++st.race_count;
        }
    }
    st.value = value;
    ++st.update_count;
    ++st.iter_updates;
    st.last_writer_cta = cta;
    st.last_writer_tid = tid;
}

inline unsigned int eg_condgraph_get(cudaGraphConditionalHandle handle) {
    std::lock_guard<std::mutex> lock(eg_condgraph_mu);
    auto it = eg_condgraph_handles.find(handle);
    if (it == eg_condgraph_handles.end()) return 0u;
    return it->second.value;
}

inline int eg_condgraph_update_count(cudaGraphConditionalHandle handle) {
    std::lock_guard<std::mutex> lock(eg_condgraph_mu);
    auto it = eg_condgraph_handles.find(handle);
    if (it == eg_condgraph_handles.end()) return 0;
    return it->second.update_count;
}

inline int eg_condgraph_iter_updates(cudaGraphConditionalHandle handle) {
    std::lock_guard<std::mutex> lock(eg_condgraph_mu);
    auto it = eg_condgraph_handles.find(handle);
    if (it == eg_condgraph_handles.end()) return 0;
    return it->second.iter_updates;
}

inline int eg_condgraph_race_count(cudaGraphConditionalHandle handle) {
    std::lock_guard<std::mutex> lock(eg_condgraph_mu);
    auto it = eg_condgraph_handles.find(handle);
    if (it == eg_condgraph_handles.end()) return 0;
    return it->second.race_count;
}

__device__ inline void cudaGraphSetConditional(cudaGraphConditionalHandle handle, unsigned int value) {
    eg_condgraph_set(handle, value);
}

inline void eg_condgraph_write_trace(
    const char* path,
    int iterations_executed,
    unsigned int final_condition,
    int handle_update_count,
    int runaway,
    int missing_update,
    int race_count,
    const std::vector<unsigned int>& trace) {
    if (!path || !path[0]) return;
    FILE* f = std::fopen(path, "wb");
    if (!f) return;
    std::fprintf(f, "iterations\t%d\n", iterations_executed);
    std::fprintf(f, "final_condition\t%u\n", final_condition);
    std::fprintf(f, "update_count\t%d\n", handle_update_count);
    std::fprintf(f, "runaway\t%d\n", runaway);
    std::fprintf(f, "missing_update\t%d\n", missing_update);
    std::fprintf(f, "race_count\t%d\n", race_count);
    std::fprintf(f, "trace");
    for (unsigned int v : trace) {
        std::fprintf(f, "\t%u", v);
    }
    std::fprintf(f, "\n");
    std::fclose(f);
}

#define eg_emu_condgraph_set eg_condgraph_set
#define eg_emu_condgraph_get eg_condgraph_get
#define eg_emu_condgraph_update_count eg_condgraph_update_count

#endif // EG_EMU_CONDGRAPH_H
