// Package cudashadow provides an observation-only evaluation and deterministic
// replay framework for GPU driver and runtime policies.
//
// Its illustrative UVM eviction, page prefetch, scheduling and PTX policies
// consume caller-supplied observations. They do not attach kernel hooks or
// execute proposed decisions. Arbitrary caller Policy implementations remain
// responsible for their own side effects and cooperative cancellation.
//
// Proposed decisions are captured in bounded, serializable snapshots tracking
// complete input provenance, policy configuration, proposed decisions with
// rationale, and subsequently observed baseline outcomes. Stale and unsupported
// inputs are explicitly distinguished from valid observations.
// Built-in policy state defaults to 4096 entries; max_entries can configure
// the bound. Exhaustion returns ErrCapacity without silently evicting evidence.
// Replay requires a fresh policy and its complete initial-state history.
// Use one evaluator and policy instance for one ordered device/source epoch;
// source changes require a new instance. Snapshots use Unix nanoseconds.
package cudashadow
