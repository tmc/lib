// Package dsv4chat runs a line-oriented chat using one persistent model state.
// The caller supplies the chat template, token sampler, and token decoder.
// Run accepts /reset, /quit, and /exit commands and releases its state on EOF
// or error. Generated token IDs remain available to subsequent render calls.
// Optional state configuration runs before replay; publication callbacks borrow
// the state only until its next successful replacement or Run returns.
package dsv4chat
