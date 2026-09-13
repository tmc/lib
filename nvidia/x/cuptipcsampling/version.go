package cuptipcsampling

// Collection reads native storage whose shape comes from cupti_pcsampling.h, so
// every admitted CUPTI version needs a measured witness for that header, not a
// version comparison. CUDA 13.0 ships 130001 and CUDA 13.4 ships 130400, and
// their headers are byte-identical: /usr/local/cuda-13.0/include and
// /usr/local/cuda-13.4/include hold the same 32144 bytes (md5
// ea3daa87f34f...), measured on the GB10 host. Identical bytes means identical
// structures, which is a stronger witness than a field-by-field comparison, and
// it is why 130400 is admitted here while the layout assertions in
// layout_test.go remain unchanged.
//
// 130200 is not admitted for collection. It appears in the reason-inventory and
// cubin-CRC gates, whose parameter sizes were recorded separately, but no
// toolkit on any host we measure ships it, so its sampling output layout has no
// witness at all.
func qualifiedSamplingVersion(v uint32) bool {
	return v == 130001 || v == 130400
}

// qualifiedQueryVersion reports whether v has a recorded parameter layout for
// the cubin CRC query, which reads no sampling output and so admits the older
// layouts collection rejects. It holds the same versions as the stall-reason
// layouts in [cupti.PCSamplingReasonLayoutFor], which that query consults
// directly.
func qualifiedQueryVersion(v uint32) bool {
	return v == 27 || v == 130001 || v == 130200 || v == 130400
}
