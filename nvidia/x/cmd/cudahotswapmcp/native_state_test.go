package main

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/tmc/lib/nvidia/x/cudahotswap"
	"github.com/tmc/lib/nvidia/x/cudahotswap/httpapi"
)

func nativeMCPResponse(result *mcp.CallToolResult) (httpapi.Response, error) {
	var reply httpapi.Response
	if result == nil || len(result.Content) != 1 {
		return reply, errors.New("MCP control: one report required")
	}
	text, ok := result.Content[0].(*mcp.TextContent)
	if !ok {
		return reply, errors.New("MCP control: text report required")
	}
	if err := json.Unmarshal([]byte(text.Text), &reply); err != nil {
		return reply, err
	}
	return reply, nil
}

func nativeMCPPushCheck(prior, current cudahotswap.State, result *mcp.CallToolResult, source, label string, wrong bool) error {
	reply, err := nativeMCPResponse(result)
	if err != nil {
		return err
	}
	p := reply.Push
	want := cudahotswap.OutcomeOK
	if wrong {
		want = cudahotswap.OutcomeWrongAnswer
	}
	if p == nil || p.Report.Outcome != want || !p.Report.Recognized || result.IsError != wrong || !p.Published || p.RolledBack != wrong {
		return errors.New("MCP control: publication or classified result differs")
	}
	digest := fmt.Sprintf("%x", sha256.Sum256([]byte(source)))
	if p.Report.Compile == nil || p.Report.Compile.SourceSHA256 != digest || p.State.SourceDigest != digest || p.Selected != current {
		return errors.New("MCP control: source or selected state differs")
	}
	if prior.Name != "saxpy" || prior.Kind != "kernel" || prior.Generation == 0 || prior.Version == 0 || prior.ConfigRevision == 0 ||
		p.State.Name != prior.Name || p.State.Kind != prior.Kind || p.State.Label != label || p.State.Generation != prior.Generation+1 || p.State.Version <= prior.Version || p.State.ConfigRevision != prior.ConfigRevision ||
		p.Report.Generation != p.State.Generation || p.Report.ConfigRevision != p.State.ConfigRevision {
		return errors.New("MCP control: candidate state or report stamp differs")
	}
	if wrong {
		if current.Name != prior.Name || current.Kind != prior.Kind || current.Version != prior.Version || current.SourceDigest != prior.SourceDigest || current.CompilerKey != prior.CompilerKey || current.ConfigRevision != prior.ConfigRevision || current.Generation != p.State.Generation+1 || current.Label != label+": rollback" {
			return errors.New("MCP control: wrong candidate did not restore prior code")
		}
	} else if current != p.State {
		return errors.New("MCP control: correct candidate is not selected")
	}
	// The demo uses a custom CPU-reference Checker. A native replay report is
	// neither required nor fabricated for that checker.
	return nil
}
