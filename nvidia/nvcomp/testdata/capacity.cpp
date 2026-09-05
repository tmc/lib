// Standalone nvCOMP LZ4 capacity-contract probe. No Go or FFI wrapper is involved.
// The allocation remains 4096 bytes even when its declared capacity is 1.
#include <cuda_runtime.h>
#include <nvcomp/lz4.h>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

static bool cleanup_failed;
static void cuda_check(cudaError_t result) {
  if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
static void nv_check(nvcompStatus_t result) {
  if (result != nvcompSuccess) throw std::runtime_error("nvCOMP status " + std::to_string(int(result)));
}
struct Buffer {
  void *p = nullptr;
  explicit Buffer(size_t bytes) { if (bytes) cuda_check(cudaMalloc(&p, bytes)); }
  ~Buffer() { if (p) { auto e = cudaFree(p); if (e != cudaSuccess) { std::fprintf(stderr,"cudaFree: %s\n",cudaGetErrorString(e)); cleanup_failed = true; } } }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  template<class T> T *as() { return static_cast<T*>(p); }
  void put(const void *src, size_t n) { cuda_check(cudaMemcpy(p,src,n,cudaMemcpyHostToDevice)); }
  void get(void *dst, size_t n) { cuda_check(cudaMemcpy(dst,p,n,cudaMemcpyDeviceToHost)); }
};
static int run() {
  cuda_check(cudaSetDevice(0));
  constexpr size_t size = 4096;
  nvcompBatchedLZ4CompressOpts_t compress{};
  nvcompBatchedLZ4DecompressOpts_t decompress{};
  decompress.backend = NVCOMP_DECOMPRESS_BACKEND_CUDA;
  size_t max_compressed = 0, compress_scratch = 0, decompress_scratch = 0;
  nv_check(nvcompBatchedLZ4CompressGetMaxOutputChunkSize(size,compress,&max_compressed));
  nv_check(nvcompBatchedLZ4CompressGetTempSizeAsync(1,size,compress,&compress_scratch,size));
  nv_check(nvcompBatchedLZ4DecompressGetTempSizeAsync(1,size,decompress,&decompress_scratch,size));
  Buffer input(size), encoded(max_compressed), output(size), in_ptr(sizeof(void*)), enc_ptr(sizeof(void*)), out_ptr(sizeof(void*));
  Buffer in_size(sizeof(size_t)), enc_size(sizeof(size_t)), capacity(sizeof(size_t)), actual(sizeof(size_t)), status(sizeof(nvcompStatus_t));
  Buffer cs(compress_scratch), ds(decompress_scratch);
  std::vector<unsigned char> original(size), decoded(size);
  for (size_t i=0;i<size;i++) original[i] = i/64;
  input.put(original.data(),size);
  in_ptr.put(&input.p,sizeof(void*)); enc_ptr.put(&encoded.p,sizeof(void*)); out_ptr.put(&output.p,sizeof(void*));
  in_size.put(&size,sizeof(size));
  cuda_check(cudaMemset(status.p,0x7f,sizeof(nvcompStatus_t)));
  nv_check(nvcompBatchedLZ4CompressAsync(in_ptr.as<const void*>(),in_size.as<size_t>(),size,1,cs.p,compress_scratch,enc_ptr.as<void*>(),enc_size.as<size_t>(),compress,status.as<nvcompStatus_t>(),nullptr));
  cuda_check(cudaDeviceSynchronize());
  nvcompStatus_t result; status.get(&result,sizeof(result)); nv_check(result);
  size_t encoded_bytes; enc_size.get(&encoded_bytes,sizeof(encoded_bytes));
  if (encoded_bytes == 0 || encoded_bytes > max_compressed) throw std::runtime_error("invalid compressed size");
  int failure = 0;
  for (size_t declared : {size,size_t(1)}) {
    capacity.put(&declared,sizeof(declared));
    cuda_check(cudaMemset(output.p,0xa5,size));
    cuda_check(cudaMemset(status.p,0x7f,sizeof(nvcompStatus_t)));
    cuda_check(cudaMemset(actual.p,0xff,sizeof(size_t)));
    nv_check(nvcompBatchedLZ4DecompressAsync(enc_ptr.as<const void*>(),enc_size.as<size_t>(),capacity.as<size_t>(),actual.as<size_t>(),1,ds.p,decompress_scratch,out_ptr.as<void*>(),decompress,status.as<nvcompStatus_t>(),nullptr));
    cuda_check(cudaDeviceSynchronize());
    status.get(&result,sizeof(result));
    size_t produced; actual.get(&produced,sizeof(produced)); output.get(decoded.data(),size);
    size_t changed = 0; for (size_t i=declared;i<size;i++) changed += decoded[i] != 0xa5;
    std::printf("backend=%d options=%zu capacity=%zu allocation=%zu status=%d actual=%zu changed_beyond_capacity=%zu\n",int(decompress.backend),sizeof(decompress),declared,size,int(result),produced,changed);
    if (declared == size) {
      if (result != nvcompSuccess || produced != size || decoded != original) throw std::runtime_error("round trip failed");
    } else if (result != nvcompErrorCannotDecompress) {
      std::fprintf(stderr,"capacity contract failed: expected status %d, got %d\n",int(nvcompErrorCannotDecompress),int(result)); failure = 1;
    }
  }
  return failure;
}
int main() {
  int result;
  try { result = run(); } catch (const std::exception& e) { std::fprintf(stderr,"probe: %s\n",e.what()); result = 2; }
  return cleanup_failed ? 3 : result;
}
