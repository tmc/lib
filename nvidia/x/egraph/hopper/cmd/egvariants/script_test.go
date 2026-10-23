package main

import (
	"bytes"
	"context"
	"fmt"
	"path/filepath"
	"strings"
	"testing"

	"rsc.io/script"
	"rsc.io/script/scripttest"
)

// TestScript runs the scripts in testdata/script with egvariants in process.
func TestScript(t *testing.T) {
	cmds := scripttest.DefaultCmds()
	cmds["egvariants"] = script.Command(
		script.CmdUsage{Summary: "run egvariants in process", Args: "args..."},
		func(s *script.State, args ...string) (script.WaitFunc, error) {
			var stderr bytes.Buffer
			// Paths are relative to the script's directory.
			abs := make([]string, len(args))
			for i, a := range args {
				abs[i] = a
				if !strings.HasPrefix(a, "-") {
					abs[i] = s.Path(a)
				}
			}
			code := run(abs, &stderr)
			return func(*script.State) (string, string, error) {
				if code != 0 {
					return "", stderr.String(), fmt.Errorf("exit status %d", code)
				}
				return "", stderr.String(), nil
			}, nil
		})
	moe8, err := filepath.Abs("moe8.json")
	if err != nil {
		t.Fatal(err)
	}
	engine := &script.Engine{Cmds: cmds, Conds: scripttest.DefaultConds()}
	scripttest.Test(t, context.Background(), engine, []string{"MOE8=" + moe8}, "testdata/script/*.txt")
}
