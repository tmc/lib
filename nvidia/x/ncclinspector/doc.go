// Package ncclinspector reads bounded NCCL Inspector v4.1 JSON Lines.
//
// It distinguishes collective, point-to-point and dump-statistics records.
// Communicator, rank, process and operation sequence identity are retained;
// collective and point-to-point sequence numbers are separate domains.
// A record is not a health verdict or proof of a complete profiling session.
// Dropped records remain explicit; absent dump statistics remain unavailable.
// Exact raw lines preserve verbose event traces and unknown fields.
//
// Text, Prometheus, OTLP and other format versions are not supported.
// Official source and example provenance is in testdata/README.md.
package ncclinspector
