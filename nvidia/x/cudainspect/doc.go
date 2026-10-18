// Package cudainspect inspects supplied CUDA artifacts and emits bounded,
// schema-versioned reports. Unavailable static facts remain explicit.
//
// [Inspector.Report] is the validating entry point. It refuses input it
// cannot describe exactly, such as a non-CUDA image or a symbol that is not
// one code section, and returns a [Report] whose features each carry a state,
// with an explicit null value when unavailable. [Inspector.Inspect] is
// lenient: it describes any bytes, including non-ELF input, in the older
// [ReportSchemaV1] form, and never refuses a selection.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package cudainspect
