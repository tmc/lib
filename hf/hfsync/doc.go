// Package hfsync implements a peer-to-peer synchronization protocol for
// Hugging Face model caches over go-iroh QUIC connections.
//
// The protocol enables distributed inference and training nodes to synchronize
// snapshot weights, configurations, and tokenizers directly from peer machines
// across local networks and NATs with zero cloud egress cost.
//
// # Protocol Overview
//
// Nodes negotiate the ALPN "tmc/hfsync/v1" over Iroh QUIC streams. Synchronization
// proceeds in two phases:
//
//  1. Manifest Exchange: The client requests a ModelManifest for a given model ID
//     (e.g., "unsloth/Llama-3.2-1B-Instruct"). The server returns a list of relative
//     file paths, sizes, and SHA-256 content hashes for the active snapshot.
//
//  2. Parallel Chunk Transfer: The client compares the manifest against its local
//     Hugging Face cache directory, determines missing or partial files, and fetches
//     chunks in parallel. Completed files are validated and linked into standard
//     snapshot and blob directories.
//
// # Basic Usage
//
// To serve a local Hugging Face cache:
//
//	store := hfsync.NewStore("/path/to/.cache/huggingface/hub")
//	server := hfsync.NewServer(endpoint, store)
//	go server.Serve(ctx)
//
// To sync a model from a peer:
//
//	client := hfsync.NewClient(endpoint, store)
//	stats, err := client.Sync(ctx, peerAddr, "unsloth/Llama-3.2-1B-Instruct")
package hfsync
