// stubalias.c builds a shared library that exports its entry points only
// under their unversioned names, as a 13.x libnvJitLink built before the
// header's __nvJitLink*_13_2 symbols does. NVJITLINK_STUB_MAJOR, when
// defined, is the major its nvJitLinkVersion reports; without it the library
// exports no nvJitLinkVersion, so its version cannot be checked.
//
// It is compiled at test time by the fallback tests and is not part of the
// package build.

static int linker;

int nvJitLinkCreate(void **handle, unsigned n, const char **opts) { *handle = &linker; return 0; }
int nvJitLinkDestroy(void **handle) { *handle = 0; return 0; }

#ifdef NVJITLINK_STUB_MAJOR
int nvJitLinkVersion(unsigned *major, unsigned *minor) { *major = NVJITLINK_STUB_MAJOR; *minor = 0; return 0; }
#endif
