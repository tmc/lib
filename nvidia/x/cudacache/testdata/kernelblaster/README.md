# KernelBlaster catalog fixture

Source: NVlabs/KernelBlaster `data/kernelblaster/optimization_database.json`,
commit `84237f91a391971e566cd9066bfb7e9514e957ee`.
JSON SHA256: `4b93cbd87c533225b8be8a933683303c552d152845353dfaf0117d1ac864b4a9`.
The upstream license is retained unchanged in LICENSE.

This catalog has three known states and sixteen strategy groups. The eight
local exploration families and group mapping are explicitly defined by
KernelBlasterGroups. They are not eight upstream known states. Predictions,
nullable actual-speedup fields and elapsed cycles are not hardware receipts.
The fixture works offline and is pinned by TestKernelBlasterPinnedCatalog.
