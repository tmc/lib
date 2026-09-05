# cuRAND validation

The generated loader uses Unix dynamic-library functions on Unix and the Windows
DLL API on Windows. Windows amd64 and arm64 builds are checked; this does not
establish vendor runtime availability or ABI coverage on either architecture.

`TestWindowsLoaderFixture` uses a synthetic DLL that exports only
`curandGetVersion` and returns 4242. It checks loading after a failed attempt,
function lookup/call, missing symbols, conflicting loads, close, and reload.
It does not generate random numbers or load NVIDIA's runtime.

From the lib repository root, with MinGW:

```sh
x86_64-w64-mingw32-gcc -shared -Wall -Wextra -Werror \
  nvidia/curand/testdata/loader_windows.c -o "$HOME/tmp/curand-loader-fixture.dll"
GOWORK=off GOOS=windows GOARCH=amd64 go -C nvidia test -c \
  -o "$HOME/tmp/curand-windows.test.exe" ./curand
```

Run the test executable on Windows, or under Wine, with `CURAND_TEST_DLL` set to
the Windows-visible fixture path and `-test.run '^TestWindowsLoaderFixture$'`.

The separate `curand_physical` build tag selects actual device generation.
`TestDeviceUniform` checks 4096 samples per run for range, repeatability with the
same seed, and variation with another seed. It is not a statistical quality test.

Regeneration uses codegen's `cmd/curand-survey`, with `-header`, `-headers-dir`,
and `-output` selecting pinned input headers and an isolated output directory.
