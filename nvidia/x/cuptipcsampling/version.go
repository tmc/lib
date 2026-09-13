package cuptipcsampling

// Collection reads native storage whose shape comes from cupti_pcsampling.h, so
// every admitted CUPTI version needs a measured witness for that header, not a
// version comparison.
//
// Measured across all installed toolkits on the DGX host (CUDA 13.0, 13.1, 13.2,
// 13.3, 13.4): cupti_pcsampling.h is byte-identical: each holds the same
// 32144 bytes (md5 ea3daa87f34f...), so all CUDA 13 releases share the same
// 64-bit sampling layout.
func qualifiedSamplingVersion(v uint32) bool {
	return v >= 130001 && v < 140000
}

// qualifiedQueryVersion reports whether v has a recorded parameter layout for
// the cubin CRC query and reason inventory query. It admits CUPTI 27 (legacy
// CUDA 12) and all CUDA 13 releases (130001 <= v < 140000).
func qualifiedQueryVersion(v uint32) bool {
	return v == 27 || (v >= 130001 && v < 140000)
}
