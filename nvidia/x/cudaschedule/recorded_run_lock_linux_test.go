//go:build linux

package cudaschedule

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// This exercises the real descriptor/process reader on Linux. Cross-compiling
// it on Darwin is not execution of this control or of the pinned shell script.
func TestRecordedGuardLinuxOwnerReader(t *testing.T) {
	holder, err := recordedIdentify(os.Getpid())
	if err != nil {
		t.Fatal(err)
	}
	host, err := os.Hostname()
	if err != nil {
		t.Fatal(err)
	}
	short, _, _ := strings.Cut(host, ".")
	config := recordedTestConfig()
	config.LockDirectory = t.TempDir()
	config.Label = "actual process token=label-text"
	token := short + "-" + strconv.Itoa(holder.PID) + "-1-12"
	body := recordedScriptFixture(holder.PID, short, config.Label, token, "lane=short expect=1 limit=3 limit_src=explicit")
	path := filepath.Join(config.LockDirectory, "owner")
	if err := os.WriteFile(path, body, 0600); err != nil {
		t.Fatal(err)
	}
	owner, err := recordedReadLockOwner(config, holder, "")
	if err != nil || owner.token != token {
		t.Fatalf("actual descriptor/process reader=%+v error=%v", owner, err)
	}
	if _, err := recordedReadLockOwner(config, holder, token); err != nil {
		t.Fatal(err)
	}
	if _, err := recordedReadLockOwner(config, holder, token+"0"); err == nil {
		t.Fatal("accepted changed retained token")
	}
	changed := holder
	changed.Start++
	if _, err := recordedReadLockOwner(config, changed, token); err == nil {
		t.Fatal("accepted changed actual process generation")
	}
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(filepath.Join(config.LockDirectory, "not-owner"), path); err != nil {
		t.Fatal(err)
	}
	if _, err := recordedReadLockOwner(config, holder, token); err == nil {
		t.Fatal("accepted symlink owner")
	}
}
