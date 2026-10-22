#include <cmath>
#include "ds4_cuda.cu"

static void require(bool good, const char *what) {
    if (!good) { fprintf(stderr, "FAIL %s\n", what); exit(1); }
}
static void checked(cudaError_t e, const char *what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "FAIL %s: %s\n", what, cudaGetErrorString(e)); exit(1);
    }
}
static void save(const char *path, const void *data, size_t bytes) {
    FILE *f = fopen(path, "wb"); require(f != NULL, "artifact open");
    require(fwrite(data, 1, bytes, f) == bytes, "artifact write");
    require(fflush(f) == 0 && !ferror(f), "artifact flush");
    require(fclose(f) == 0, "artifact close");
}
int main() {
    require(getenv("GPU_LOCK_HELD") != NULL, "GPU lease required");
    printf("GPU_LOCK_HELD=%s\n", getenv("GPU_LOCK_HELD"));
    int count, runtime, driver;
    checked(cudaGetDeviceCount(&count), "device count"); require(count == 1, "one device");
    checked(cudaSetDevice(0), "device select");
    cudaDeviceProp prop;
    checked(cudaGetDeviceProperties(&prop, 0), "device properties");
    checked(cudaRuntimeGetVersion(&runtime), "runtime version");
    checked(cudaDriverGetVersion(&driver), "driver API version");
    printf("device=%s cc=%d.%d runtime=%d driver_api=%d\n", prop.name, prop.major, prop.minor, runtime, driver);
    require(prop.major == 12 && prop.minor == 1 && strstr(prop.name, "GB10"), "GB10 sm121");
    require(runtime == 13040, "CUDA 13.4 runtime");
    const uint32_t nt=130, nh=10, hd=512, window=128;
    const size_t n=(size_t)nt*nh*hd;
    std::vector<float> hq(n), hraw((size_t)nt*hd), hsinks(nh), baseline(n), first(n), output(n);
    for (size_t i=0;i<hq.size();i++) hq[i]=(float)((int)(i%37)-18)/32.0f;
    for (size_t i=0;i<hraw.size();i++) hraw[i]=(float)((int)((i*13)%127)-63)/64.0f;
    for (uint32_t i=0;i<nh;i++) hsinks[i]=(float)i/16.0f;
    save("q.bin",hq.data(),hq.size()*4); save("raw.bin",hraw.data(),hraw.size()*4); save("sinks.bin",hsinks.data(),hsinks.size()*4);
    float *q, *raw, *sinks, *heads;
    checked(cudaMalloc(&q,n*4),"q alloc"); checked(cudaMalloc(&raw,hraw.size()*4),"raw alloc");
    checked(cudaMalloc(&sinks,nh*4),"sinks alloc"); checked(cudaMalloc(&heads,n*4),"heads alloc");
    checked(cudaMemcpy(q,hq.data(),n*4,cudaMemcpyHostToDevice),"q upload");
    checked(cudaMemcpy(raw,hraw.data(),hraw.size()*4,cudaMemcpyHostToDevice),"raw upload");
    checked(cudaMemcpy(sinks,hsinks.data(),nh*4,cudaMemcpyHostToDevice),"sinks upload");
    const size_t bytes=8+2*40, allocation=bytes+16;
    unsigned char *buffer;
    checked(cudaMalloc(&buffer,allocation),"observer alloc");
    ds4_gpu_raw_observer o={1,sizeof(o),1,2,40,129,9,2,511,0,(uint64_t)(uintptr_t)buffer,bytes};
    require(cuda_raw_observer_valid(&o),"descriptor validation");
    std::vector<unsigned char> snapshot(allocation), first_record;
    auto reset=[&] {
        checked(cudaMemset(buffer,0,bytes),"observer reset");
        checked(cudaMemset(buffer+bytes,0xa5,16),"guard reset");
    };
    auto launch=[&](bool observe) {
        checked(cudaMemset(heads,0xff,n*4),"output poison");
        if (observe) attention_raw_window_observed_kernel<<<dim3(nt,(nh+7)/8,1),256>>>(heads,sinks,q,raw,raw,nt,0,window,1,nh,hd,o);
        else attention_static_mixed_heads8_online_kernel<<<dim3(nt,(nh+7)/8,1),256>>>(heads,sinks,q,raw,raw,nt,0,window,1,nh,hd);
        checked(cudaGetLastError(),"window launch");
        checked(cudaDeviceSynchronize(),"window completion");
        checked(cudaMemcpy(output.data(),heads,n*4,cudaMemcpyDeviceToHost),"output read");
        for(float value:output) require(std::isfinite(value),"finite output");
        checked(cudaMemcpy(snapshot.data(),buffer,allocation,cudaMemcpyDeviceToHost),"observer read");
        for(size_t i=bytes;i<allocation;i++) require(snapshot[i]==0xa5,"observer trailing guard");
    };
    auto state=[&](uint32_t index,uint32_t overflow) {
        ds4_gpu_raw_observer_state s; memcpy(&s,snapshot.data(),sizeof s);
        printf("state index=%u overflow=%u\n",s.write_index,s.overflow);
        require(s.write_index==index && s.overflow==overflow,"observer state");
    };
    reset(); launch(false); state(0,0); baseline=output;
    save("off-output.bin",output.data(),n*4);
    puts("PASS observer-off original window kernel");
    uint32_t bits; memcpy(&bits,&hraw[2*hd+511],4);
    for(int run=0;run<2;run++) {
        reset(); launch(true); state(1,0);
        ds4_gpu_raw_observer_record r; memcpy(&r,snapshot.data()+8,sizeof r);
        require(r.version==1 && r.size==32 && r.site==2 && r.token==129 && r.head==9 && r.row==2 && r.dim==511 && r.value_bits==bits,"positive selected record");
        for(size_t i=8+32;i<bytes;i++) require(snapshot[i]==0,"padding and unused slot");
        require(memcmp(output.data(),baseline.data(),n*4)==0,"off/on byte identity");
        if(run==0) {first=output;first_record=snapshot;}
        else {
            require(memcmp(output.data(),first.data(),n*4)==0,"A/A output identity");
            require(snapshot==first_record,"A/A entire observer buffer identity");
        }
        save(run==0?"on-a-output.bin":"on-b-output.bin",output.data(),n*4);
        save(run==0?"on-a-record.bin":"on-b-record.bin",snapshot.data()+8,32);
        save(run==0?"on-a-buffer.bin":"on-b-buffer.bin",snapshot.data(),snapshot.size());
        printf("PASS positive_%d site=2 token=129 head=9 row=2 dim=511 bits=%08x off_on_equal=1\n",run+1,bits);
    }
    puts("PASS A/A output and record identity");
    o.row=1; reset(); launch(true); state(0,0);
    for(size_t i=8;i<bytes;i++) require(snapshot[i]==0,"negative zero records");
    require(memcmp(output.data(),baseline.data(),n*4)==0,"negative output identity");
    save("negative-buffer.bin",snapshot.data(),snapshot.size());
    puts("PASS adjacent out-of-window row=1; token=129 window rows=2..129");
    o.row=2; o.capacity=0; o.bytes=8;
    require(cuda_raw_observer_valid(&o),"zero capacity descriptor");
    reset(); launch(true); state(0,1);
    for(size_t i=8;i<bytes;i++) require(snapshot[i]==0,"overflow no append");
    require(memcmp(output.data(),baseline.data(),n*4)==0,"overflow output identity");
    save("overflow-buffer.bin",snapshot.data(),snapshot.size());
    o.row=1; launch(true); state(0,1);
    for(size_t i=8;i<bytes;i++) require(snapshot[i]==0,"sticky overflow no append");
    require(memcmp(output.data(),baseline.data(),n*4)==0,"sticky overflow output identity");
    save("sticky-buffer.bin",snapshot.data(),snapshot.size());
    puts("PASS capacity=0 overflow sticky across subsequent nonmatching launch");
    checked(cudaFree(buffer),"observer free"); checked(cudaFree(heads),"heads free");
    checked(cudaFree(sinks),"sinks free"); checked(cudaFree(raw),"raw free"); checked(cudaFree(q),"q free");
    puts("PASS window site-2 qualification; original=1 observed=5 launches=6; direct device pointers only");
    return 0;
}
