// Package dcgm provides Go bindings for the NVIDIA Data Center GPU Manager
// client library, libdcgm.
//
// DCGM is a monitoring and management service. A host engine samples GPU
// telemetry and answers queries about it; a client attaches to that host
// engine, asks for entities and fields, and reads values back. This package
// binds the read-only part of that client API: library and connection
// lifetime, device and entity discovery, groups and field groups, field
// watches and value retrieval, and the diagnostic and health state a client
// can read.
//
// Configuration-changing management is deliberately not bound. dcgmConfigSet,
// dcgmPolicySet, dcgmHealthSet, dcgmActionValidate, dcgmRunDiagnostic and the
// job-tracking calls change the state of a shared service that other programs
// on the same node are also using, and several of them make the GPU
// unavailable for the duration. They are listed with a reason each in the
// generator's skipFuncReasons, so their absence is a recorded decision rather
// than an oversight.
//
// # Loading
//
// The library is opened on first use, not at import time, so importing this
// package on a machine without DCGM installed is legal. [Load] opens it
// explicitly; [Path] reports where it came from; [Has] reports whether one C
// symbol resolved. When the library or a symbol is missing, calls return
// *LibraryError or *SymbolError. Nothing in this package panics for an absent
// library, an absent symbol, or a failed call.
//
// DCGM_LIB_PATH overrides the search. The default names are libdcgm.so.4 then
// libdcgm.so.
//
// # Lifetime
//
// [Init] must be called before anything else, and it is not called implicitly:
// [Shutdown] is process-wide, so a package that started the library would have
// to guess when to stop it.
//
// After Init, get a session either by starting a host engine inside this
// process with [StartEmbedded], or by attaching to one that is already running
// with [Connect]. See [Session] for what each owns.
//
//	if err := dcgm.Init(); err != nil { ... }
//	defer dcgm.Shutdown()
//	s, err := dcgm.StartEmbedded(dcgm.OperationModeAuto)
//	if err != nil { ... }
//	defer s.Close()
//
// # Ownership
//
// Groups, field groups and watches live in the host engine and outlive the
// process that created them. The raw API cannot tell an object this process
// created from one it found, so calling dcgmGroupDestroy on another program's
// group succeeds and stops that program's monitoring. [Session] exists to draw
// that line: it records what it created and [Session.Close] removes exactly
// that, in reverse order. It never destroys a group obtained from
// [Handle.Groups], never touches a pseudo-group such as [GroupAllGPUs], never
// stops a host engine reached with [Connect], and never calls [Shutdown].
// [Session.Owns], [Session.OwnsFieldGroup] and [Session.OwnsHostEngine] report
// the line from the outside.
//
// Buffers follow the same rule everywhere: a slice or string passed into a
// call is read during the call and not retained, and a value returned is a Go
// copy, not a view into library memory. No callback in the bound surface is
// registered with the library, so no Go function pointer outlives a call. The
// only handles are [Handle], [GpuGrp], [FieldGrp] and [StatusHandle]; each is
// an opaque integer the host engine owns, and none of them is finalized.
//
// # Versioned structures
//
// This is the hazard that breaks bindings. Most DCGM structures begin with a
// version field, and the caller must stamp it with the exact constant for the
// structure layout it compiled against:
//
//	MAKE_DCGM_VERSION(t, v) = sizeof(t) | (v << 24)
//
// The size is part of the number. A structure whose Go definition drifts from
// the header it was generated from produces a version constant DCGM does not
// recognize, and every call using it fails with [ErrVerMismatch] rather than
// misbehaving quietly. The generated versions.gen.go holds one constant per
// versioned structure, computed by the C compiler from the pinned headers;
// versions_test.go recomputes each one from unsafe.Sizeof of the Go type and
// fails if the two disagree.
//
// Callers must stamp the field. The helpers in this package do it for the
// structures they own, but a structure passed to a generated function directly
// is the caller's to stamp:
//
//	info := dcgm.GroupInfo{Version: dcgm.GroupInfoVersion3}
//	err := dcgm.GroupGetInfo(h, groupID, &info)
//
// # Values
//
// A field read is not a number; it is a status, a timestamp and a payload, and
// all three matter. DCGM reports "no data" by returning the field successfully
// with a blank value rather than by failing, so a reading whose payload is
// used without checking is how an unsupported field becomes a plausible-looking
// zero. Check [FieldValue.Err], then [FieldValue.Blank], then read the payload;
// [FieldValue.Time] is when the host engine sampled it, which for a watch that
// has not ticked yet is the zero time.
//
// Blankness is a sentinel comparison, not an equality: [Int64Blank],
// [Int64NotFound], [Int64NotSupported] and [Int64NotPermissioned] are
// consecutive, so the test is >=. [FieldValueV1.Blank] and [FieldValueV2.Blank]
// implement it for each type.
//
// # Field ids
//
// Every quantity DCGM can sample has a numeric id, defined as a macro in
// dcgm_fields.h. The clang AST carries no macros, so the generator scans the
// header for them and has the C compiler evaluate each one; fields.gen.go holds
// the result as [FieldID] constants with a name table behind [FieldID.String].
// The macros the header computes rather than assigns -- counts and range bounds
// such as DCGM_FI_MAX_FIELDS -- are generated as uint16 rather than FieldID, so
// passing one where a field id is expected does not compile.
//
// # Testing without a GPU
//
// The generated tests check the ABI, not the hardware: struct sizes, alignments
// and field offsets measured by clang for x86_64-unknown-linux-gnu; every
// version constant against the Go type it stamps; every buffer bound against
// the array it governs; the blank sentinel ordering; the field-id table. They
// need no GPU, no driver and no libdcgm, and passing them says the bindings
// describe the pinned headers correctly. It says nothing about whether a call
// works.
//
// A successful NVML query is not a DCGM test. NVML talks to the driver
// directly and answers from the calling process; DCGM answers from a host
// engine, over a connection, out of a sampling history that a watch had to
// establish first. NVML reporting a temperature proves the GPU and the driver
// are there. It does not prove that libdcgm loaded, that a host engine is
// running or reachable, that this process may talk to it, that a group or
// field group was created, that a watch was accepted, or that a sample ever
// arrived -- which is every step this package's paths depend on. The physical
// test in physical_test.go, behind the dcgm_physical build tag, is what checks
// those, and it has never been run.
package dcgm
