// Package kernelagent drives cudasearch beam or MCTS search over kernel
// schedule edits, admits data-only proposals, and records every attempt in a
// durable append-only archive.
//
// Search, numerical comparison, interleaving, executable identity, work
// budgets, confirmation and promotion evidence belong to cudasearch and
// cudatrial. Each epoch uses a concrete [cudasearch.ExecutableEvaluator]
// whose Observer is the archive's only attempt write path: an attempt is
// appended before any extraction or compilation, and its outcome after
// candidate cleanup. A failed append refuses the attempt or withholds its
// candidate. This package adds structured proposal admission, bounded
// diagnostics, the archive, and a next-launch selection of borrowed cudaplan
// plans. Trusted Go adapters are not sandboxed. Only their model response
// data is untrusted here.
//
// [FixedEnumerator] is the proposal step used before any model; no model
// client is provided.
//
// HostStub records synthetic device timings and conditions supplied by a test
// adapter. It provides no hardware, performance, or native qualification.
// Native promotion is refused with [ErrPlanAssociationUnavailable] until a
// compiler adapter can authenticate a search executable against a plan.
package kernelagent
