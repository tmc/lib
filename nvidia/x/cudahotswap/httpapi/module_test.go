package httpapi

import (
	"os"
	"os/exec"
	"strings"
	"testing"
)

func TestMCPModuleBoundary(t *testing.T) {
	command := exec.Command("go", "list", "-m", "all")
	command.Dir = "../../.."
	command.Env = append(os.Environ(), "GOWORK=off")
	output, err := command.CombinedOutput()
	if err != nil {
		t.Fatalf("module inventory: %v: %s", err, output)
	}
	for _, line := range strings.Split(string(output), "\n") {
		if strings.HasPrefix(line, "github.com/modelcontextprotocol/") {
			t.Fatalf("MCP dependency leaked into nvidia: %s", line)
		}
	}
}
