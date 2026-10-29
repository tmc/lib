package main

import (
	"bytes"
	"context"
	"fmt"
	"testing"

	"rsc.io/script"
	"rsc.io/script/scripttest"
)

// TestScript runs the scripts in testdata/script. The nsysgaps command
// calls run in this process; no program is built or executed.
func TestScript(t *testing.T) {
	cmds := scripttest.DefaultCmds()
	cmds["nsysgaps"] = script.Command(
		script.CmdUsage{Summary: "run nsysgaps in process", Args: "args..."},
		func(s *script.State, args ...string) (script.WaitFunc, error) {
			var stdout, stderr bytes.Buffer
			code := run(args, &stdout, &stderr)
			return func(*script.State) (string, string, error) {
				if code != 0 {
					return stdout.String(), stderr.String(), fmt.Errorf("exit status %d", code)
				}
				return stdout.String(), stderr.String(), nil
			}, nil
		})
	engine := &script.Engine{Cmds: cmds, Conds: scripttest.DefaultConds()}
	scripttest.Test(t, context.Background(), engine, nil, "testdata/script/*.txt")
}
