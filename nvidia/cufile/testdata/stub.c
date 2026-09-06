// stub.c builds a shared library that loads but exports no cuFile symbols.
// The nil-guard test points CUFILE_LIB_PATH at it to reach the code path where
// the library is present but the symbol is not, which is what an installed
// libcufile older than a given symbol looks like.
//
// It is compiled at test time by TestMissingSymbolWithStubLibrary; it is not
// part of the package build.

int cufile_stub_marker(void) { return 0; }
