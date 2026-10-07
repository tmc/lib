/* Fixed native fixture. No child processes or detached groups. */
#define _POSIX_C_SOURCE 200809L
#include <cuda.h>
#include <poll.h>
#include <stdio.h>
#include <unistd.h>

static int check(CUresult result) {
	if (result == CUDA_SUCCESS) return 0;
	fprintf(stderr, "driver status %d\n", (int)result);
	return 1;
}

int main(void) {
	CUdevice device;
	CUcontext context;
	CUmodule module;
	CUfunction function;
	CUuuid uuid;
	int version;
	if (check(cuInit(0)) || check(cuDeviceGet(&device,0)) ||
	    check(cuDeviceGetUuid(&uuid,device)) || check(cuDriverGetVersion(&version))) return 1;
	printf("driver=%d uuid=",version);
	for (unsigned i=0;i<sizeof(uuid.bytes);i++) printf("%02x",(unsigned char)uuid.bytes[i]);
	printf(" pid=%ld\n",(long)getpid());
	if (check(cuCtxCreate_v2(&context,0,device))) return 1;
	const char *ptx=".version 6.0\n.target sm_52\n.address_size 64\n.visible .entry lane87_periodic() { ret; }\n";
	if (check(cuModuleLoadData(&module,ptx))) { cuCtxDestroy_v2(context);return 1; }
	if (check(cuModuleGetFunction(&function,module,"lane87_periodic"))) goto failed;
	for (unsigned i=0;i<32;i++) if (check(cuLaunchKernel(function,1,1,1,1,1,1,0,0,0,0))) goto failed;
	if (check(cuCtxSynchronize())) goto failed;
	puts("waiting-before-exit");fflush(stdout);
	struct pollfd input={STDIN_FILENO,POLLIN,0};
	if (poll(&input,1,3000)!=1 || !(input.revents&POLLIN)) goto failed;
	char release;
	if (read(STDIN_FILENO,&release,1)!=1 || release!='x') goto failed;
	puts("released");fflush(stdout);
	if (check(cuModuleUnload(module)) || check(cuCtxDestroy_v2(context))) return 1;
	return 0;
failed:
	cuModuleUnload(module);cuCtxDestroy_v2(context);return 1;
}
