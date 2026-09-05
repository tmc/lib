/* Synthetic DLL for loader plumbing only; this is not a cuRAND runtime. */
__declspec(dllexport) int __stdcall curandGetVersion(int *version) {
  if (!version) return 100;
  *version = 4242;
  return 0;
}
