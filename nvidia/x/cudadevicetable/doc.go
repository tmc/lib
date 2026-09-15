// Package cudadevicetable holds GPU device facts that were read from a real
// device, separated from the ones that were not.
//
// Device tables in this module have been hand-written from datasheets: an
// architecture name mapped to a dozen integers, with nothing recording where
// any of them came from. That shape fails silently. A wrong shared-memory
// capacity does not produce an error, it produces an occupancy number that is
// plausible and wrong, and no arithmetic downstream can tell it from a right
// one. The same table is also maintained in more than one package, so two
// copies can disagree indefinitely without anything noticing.
//
// This package keeps one table, generated rather than typed, in which every
// value carries its own provenance:
//
//	Probed      read from a live device through a named driver attribute
//	Derived     computed from probed values in the same record
//	Unverified  hand-entered; no device has confirmed it
//
// The rule that makes the distinction hold is structural rather than
// editorial: a [Value] may claim [Probed] only if it names the
// driver.DevAttr constant it was read through, and [Table.Validate] rejects a
// record that breaks it. A datasheet figure cannot be relabelled as a
// measurement without also naming a real attribute to have measured it with,
// which is a check a test can run and a claim a reader can reproduce.
//
// The table is deliberately sparse. It holds architectures somebody has
// actually run [cmd/cudadevicetable] against, not every architecture the
// module can compile for; [Lookup] reports a miss for the rest rather than
// inventing a plausible entry. Absence here means nobody probed it, which is
// the honest answer and a useful one.
//
// Regenerate on a host with the device attached:
//
//	go run ./cmd/cudadevicetable -update x/cudadevicetable/devices.json
package cudadevicetable
