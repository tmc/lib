package cuptirange

// The Range Profiler's host parameter structs come in two measured shapes.
//
// The base shape ends CUpti_Profiler_Host_Initialize_Params at pHostObject and
// CUpti_Profiler_Host_GetMetricProperties_Params at metricType. The scoped
// shape appends pSinglePassMetricSetName to the first and metricCollectionScope
// to the second, so a caller must size each struct to the field its CUPTI
// actually has.
//
// Measured in cupti_profiler_host.h on the GB10 host: CUPTI 130001 (CUDA 13.0)
// is the base shape; 130400 (CUDA 13.4) is the scoped shape, with both structs
// extended exactly as 130200 documented. CUPTI 27 is the legacy CUDA 12
// encoding and is base. Every other version fails closed, because a struct size
// guessed from a version number is a native memory error, not a mismatch.
func qualifiedLayout(version uint32) bool {
	return version == 27 || version == 130001 || scopedLayout(version)
}

// scopedLayout reports whether version's parameter structs carry the appended
// single-pass and collection-scope fields.
func scopedLayout(version uint32) bool {
	return version == 130200 || version == 130400
}
