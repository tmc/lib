package cuptipcsampling

// Collection reads native storage whose shape comes from cupti_pcsampling.h, so
// every admitted CUPTI version needs a measured witness for that header, not a
// version comparison.
//
// Measured across the five CUDA 13 toolkits installed on the GB10 host, whose
// CUPTI_API_VERSION values are 130001, 130101, 130202, 130301 and 130400:
// cupti_pcsampling.h is byte-identical in all five, the same 32144 bytes with
// md5 ea3daa87f34f. Identical bytes mean identical structures, which is a
// stronger witness than a field-by-field comparison. 130200 is admitted on a
// different witness: it is [cupti.ApiVersion], the version the Go types in
// nvidia/cupti were generated from, so those types are that header's shape.
//
// The list is an enumeration rather than a CUDA 13 range because the witness is
// per version, not per family, and within this same family the range profiler's
// cupti_profiler_host.h changed shape twice: at 13.1 and again at 13.2, measured
// in x/cuptirange. A range would admit a release whose header no host here has,
// on the strength of headers measured for earlier ones, and CUPTI's backward
// compatibility runs through the size these structs declare: a size declared for
// an unmeasured layout is a native write at the wrong offset, not an error
// return. Admitting a new version means measuring its header.
func qualifiedSamplingVersion(v uint32) bool {
	switch v {
	case 130001, 130101, 130200, 130202, 130301, 130400:
		return true
	}
	return false
}

// qualifiedQueryVersion reports whether v has a recorded parameter layout for
// the cubin CRC query and the stall-reason inventory, which read no sampling
// output and so also admit CUPTI 27, the legacy CUDA 12 encoding. It holds the
// same CUDA 13 versions as [cupti.PCSamplingReasonLayoutFor], which those
// queries consult directly.
func qualifiedQueryVersion(v uint32) bool {
	return v == 27 || qualifiedSamplingVersion(v)
}
