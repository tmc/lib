// stub120.c builds a shared library with the export surface of the CUDA 12.0
// libnvJitLink: every entry point carries the _12_0 suffix its header pinned,
// and there is no unsuffixed alias for the loader to fall back to. Verified
// against the real libnvjitlink-linux-sbsa-12.0.76 redistributable, whose
// only nvJitLinkCreate spelling is __nvJitLinkCreate_12_0.
//
// It exists so the clean-failure contract can be tested without a CUDA 12.0
// installation. It is compiled at test time by TestNoFallbackFailsCleanly and
// is not part of the package build.

int __nvJitLinkCreate_12_0(void *h, unsigned n, const char **opts) { return 0; }
int __nvJitLinkDestroy_12_0(void *h) { return 0; }
int __nvJitLinkComplete_12_0(void *h) { return 0; }
