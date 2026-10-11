package cuptipcsampling

import "github.com/tmc/lib/nvidia/internal/cuptiversion"

// Collection reads native storage whose shape comes from cupti_pcsampling.h, so
// every qualified CUPTI version has a measured witness for that header.
//
// Measured across the eight CUDA toolkits installed on the GB10 host, whose
// CUPTI_API_VERSION values are 24, 26, 28, 130001, 130101, 130202, 130301 and
// 130400: cupti_pcsampling.h is byte-identical in all eight, the same 32144
// bytes with md5 ea3daa87f34f. Identical bytes mean identical structures, which
// is a stronger witness than a field-by-field comparison. 130200 retains
// its historical generated-header witness.
//
// That md5 is a witness for cupti_pcsampling.h and for nothing else. The
// sibling header next door is the counterexample: cupti_profiler_host.h, in the
// same toolkits, takes four distinct sizes over 13.0 through 13.4 and grows at
// two different releases for its two range-profiler structs.
//
// Admission is exact. Runtime 130401 retains historical source admission
// with layout 130400; this is not fresh physical qualification.
func qualifiedSamplingVersion(v uint32) bool {
	_, ok := cuptiversion.AdmittedLayout(v, 130001, 130101, 130200, 130202, 130301, 130400)
	return ok
}

// qualifiedQueryVersion reports whether v has a recorded parameter layout for
// the cubin CRC query and the stall-reason inventory. Those read no sampling
// output, so they also admit the CUDA 12 encodings: 24, 26 and 28, reported by
// the 12.6, 12.8 and 12.9 libraries on the GB10 host, whose cupti_pcsampling.h
// is the same 32144 bytes as every CUDA 13 toolkit's.
//
// CUPTI 27 denotes CUDA 12.9 and 28 denotes CUDA 12.9 Update 1.
// Version 27 has no recorded PC-query witness here and is refused.
//
// Collection stays on the CUDA 13 versions even though the header witness
// reaches further back, because no CUDA 12 library here has ever been run
// through the sampling path. Admitting one means measuring a run, not reading
// the header again.
func qualifiedQueryVersion(v uint32) bool {
	_, ok := cuptiversion.AdmittedLayout(v, 24, 26, 28, 130001, 130101, 130200, 130202, 130301, 130400)
	return ok
}
