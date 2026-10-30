# ELF metadata reader

`Read(io.ReaderAt)` returns typed ELF headers, sections (with indices), program headers, and ordinary/dynamic symbol tables. Each symbol retains its source table, raw binding/type/info/other, section index, value and size; ordinary defined symbols link to their section metadata. Missing tables are explicitly `TablesAbsent`, distinct from a present table containing only its null entry.

The reader supplies metadata needed by an operation-specific PT_LOAD/storage observer. It does not choose symbols, enforce a cache inventory, translate addresses, load a DSO, or prove readable runtime storage. Callers retain source/config identity, actual mappings, segment containment/permissions, cache policy, ELF-address observer and qualification gates.

Existing readers were considered first: `cudaoccupancy` reads CUDA kernel resources; `cudasass.CubinImage.Symbols` filters an ordinary table by symbol name and is scoped to cubin images. Neither exports dual-table source identity plus section and PT_LOAD metadata for host ELF observers. They remain unchanged. Future integrations can consume this small reader after separate review; existing frozen 6/64/72 policies are unchanged.

Tests construct ELF bytes in memory; no compiler, CUDA, loaded library or binary fixture is required. Controls cover ELF64 little-endian and ELF32 big-endian, GNU-unique/TLS/UNDEF/ABS/COMMON, dual-table identity, defined sections, writable PT_LOAD, absent versus null-only tables, malformed/overflowing ranges, invalid names/links/sections, unsupported compressed/extended tables and duplicate table kinds.

```
GOWORK=off go test ./x/elfinspect
GOWORK=off go vet ./x/elfinspect
```

Run from the `nvidia` module. No dependencies were added. This is file metadata/source evidence only; native qualification and performance are not claimed.
