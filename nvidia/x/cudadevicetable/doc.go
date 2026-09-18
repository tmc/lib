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
// The table holds a second kind of fact, keyed separately. [Architecture]
// describes silicon; [Target] describes what an assembler accepts for it.
// They are not the same question and cannot share a key: setmaxnreg is
// accepted at sm_90a and rejected at sm_90, which is one piece of hardware, so
// a table that normalized the feature suffix away would have to record one of
// those two answers wrongly.
//
// Feature facts are read with ptxas rather than a device, which is why the
// table can describe Hopper and GB200 while this project has only ever
// attached a GB10. The instrument bounds the claim, and the bound is worth
// stating plainly: an accepted fixture means the assembler emitted code for
// that target, not that the instruction is fast, not that it is correct on
// silicon, and not that any device here ran it. It is the answer to "will this
// build", which is the question the packages that consumed hand-written
// capability flags were actually asking.
//
// Two rules make the feature side hold. A [Probed] feature must name the
// ptxas invocation and the fixture that produced it, so a reader who doubts it
// can re-run it. And a feature rejected at every probed target is refused by
// [Table.Validate] rather than recorded: a fixture that assembles nowhere is
// broken, and its negatives are not measurements. That rule has already
// earned its place -- the first wgmma fixture written here used a constant
// matrix descriptor that segfaults ptxas 13.0.88, and a classifier reading any
// non-zero exit as "unsupported" reported wgmma unsupported on sm_90a, the one
// target that supports it.
//
// Regenerate on a host with the device attached:
//
//	go run ./cmd/cudadevicetable -update x/cudadevicetable/devices.json
//
// Regenerate the feature facts on any host with a CUDA toolchain:
//
//	go run ./cmd/cudadevicetable -ptxas /usr/local/cuda/bin/ptxas -update x/cudadevicetable/devices.json
package cudadevicetable
