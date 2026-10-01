package main

import (
	"errors"
	"flag"
)

// traceViewLimits keeps native and external analysis bounds identical across
// summary and named queries. Overrides change limits, never retained evidence.
type traceViewLimits struct {
	input       traceInputLimits
	backend     traceProcessorLimits
	outputBytes int64
}

func registerTraceViewLimits(fs *flag.FlagSet) *traceViewLimits {
	l := &traceViewLimits{input: defaultTraceInputLimits(), backend: defaultTraceProcessorLimits(), outputBytes: defaultExportBytes}
	fs.Int64Var(&l.outputBytes, "output-bytes", l.outputBytes, "maximum diagnostic or HTML output bytes")
	fs.Int64Var(&l.input.InputBytes, "input-bytes", l.input.InputBytes, "maximum input bytes")
	fs.Int64Var(&l.input.JournalBytes, "journal-bytes", l.input.JournalBytes, "maximum decoded journal bytes")
	fs.Int64Var(&l.input.Records, "max-records", l.input.Records, "maximum decoded records per journal")
	fs.Int64Var(&l.input.ArchiveBytes, "archive-bytes", l.input.ArchiveBytes, "maximum expanded archive bytes")
	fs.IntVar(&l.input.ArchiveMembers, "archive-members", l.input.ArchiveMembers, "maximum archive members")
	fs.DurationVar(&l.backend.Timeout, "backend-timeout", l.backend.Timeout, "Trace Processor timeout per invocation")
	fs.IntVar(&l.backend.OutputBytes, "backend-output-bytes", l.backend.OutputBytes, "maximum Trace Processor stdout/stderr bytes each")
	fs.IntVar(&l.backend.Rows, "backend-rows", l.backend.Rows, "maximum Trace Processor result rows")
	return l
}
func (l *traceViewLimits) validate() error {
	if l.outputBytes <= 0 || l.input.InputBytes <= 0 || l.input.InputBytes == int64(^uint64(0)>>1) || l.input.JournalBytes <= 0 || l.input.Records <= 0 || l.input.ArchiveBytes <= 0 || l.input.ArchiveMembers <= 0 || l.backend.Timeout <= 0 || l.backend.OutputBytes <= 0 || l.backend.Rows <= 0 {
		return errors.New("trace: resource limits must be positive and bounded")
	}
	return nil
}
func (l *traceViewLimits) receipt() map[string]int64 {
	return map[string]int64{"output_bytes": l.outputBytes, "input_bytes": l.input.InputBytes, "journal_bytes": l.input.JournalBytes, "journal_records": l.input.Records, "archive_bytes": l.input.ArchiveBytes, "archive_members": int64(l.input.ArchiveMembers), "backend_timeout_ns": int64(l.backend.Timeout), "backend_output_bytes": int64(l.backend.OutputBytes), "backend_rows": int64(l.backend.Rows)}
}
