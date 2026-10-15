//go:build darwin || linux

package cudaschedule

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/tmc/lib/nvidia/x/cudalock"
)

// This pin selects the reviewed permanent-guard script, not an equivalent
// command name. Historical scripts and receipts keep their original pins.
const recordedLockDigest = "8d5631e77079fbbc94ef71c567257316c0568e6933c9dd4e2c1d81d7474c8276"

type recordedLeaseOwner struct {
	pid         int
	host        string
	label       string
	token       string
	orphanOf    int
	expected    int64
	limit       int64
	limitSource string
}

// recordedOwner accepts the source formats emitted by cudalock and the pinned
// script. The final epoch/at suffix separates a Go label from metadata. Script
// labels end at the final token field before that suffix, not at label text
// that happens to contain a metadata-looking word.
func recordedOwner(body []byte) (recordedLeaseOwner, error) {
	var owner recordedLeaseOwner
	if len(body) == 0 || len(body) > 8192 {
		return owner, fmt.Errorf("recorded helper: owner size out of range")
	}
	line := strings.TrimSuffix(string(body), "\n")
	if strings.ContainsAny(line, "\r\n\x00") {
		return owner, fmt.Errorf("recorded helper: invalid owner line")
	}
	line, ok := strings.CutPrefix(line, "pid=")
	pid, rest, found := strings.Cut(line, " host=")
	if !ok || !found {
		return owner, fmt.Errorf("recorded helper: missing owner pid or host")
	}
	var err error
	owner.pid, err = strconv.Atoi(pid)
	if err != nil || owner.pid <= 1 || strconv.Itoa(owner.pid) != pid {
		return owner, fmt.Errorf("recorded helper: invalid owner pid")
	}
	owner.host, rest, found = strings.Cut(rest, " label=")
	if !found || owner.host == "" || strings.ContainsAny(owner.host, " \t") {
		return owner, fmt.Errorf("recorded helper: invalid owner host")
	}
	epoch := strings.LastIndex(rest, " epoch=")
	if epoch < 0 {
		return owner, fmt.Errorf("recorded helper: missing owner epoch")
	}
	owner.label = rest[:epoch]
	fields := strings.Fields(rest[epoch+1:])
	if len(fields) != 2 {
		token := strings.LastIndex(owner.label, " token=")
		if token < 0 {
			return owner, fmt.Errorf("recorded helper: missing script token")
		}
		owner.token = owner.label[token+7:]
		owner.label = owner.label[:token]
		if owner.token == "" || strings.ContainsAny(owner.token, " \t") {
			return owner, fmt.Errorf("recorded helper: invalid script token")
		}
	}
	seen := make(map[string]bool)
	for _, field := range fields {
		key, value, ok := strings.Cut(field, "=")
		if !ok || value == "" || seen[key] {
			return owner, fmt.Errorf("recorded helper: duplicate or malformed owner field")
		}
		seen[key] = true
		switch key {
		case "epoch":
			n, err := strconv.ParseInt(value, 10, 64)
			if err != nil || n <= 0 {
				return owner, fmt.Errorf("recorded helper: invalid owner epoch")
			}
		case "at":
			if _, err := time.Parse(time.RFC3339, value); err != nil {
				return owner, fmt.Errorf("recorded helper: invalid owner time: %w", err)
			}
		case "lane":
			if value != "short" && value != "batch" {
				return owner, fmt.Errorf("recorded helper: invalid owner lane")
			}
		case "expect", "limit":
			n, err := strconv.ParseInt(value, 10, 64)
			if err != nil || n <= 0 {
				return owner, fmt.Errorf("recorded helper: invalid owner duration")
			}
			if key == "expect" {
				owner.expected = n
			} else {
				owner.limit = n
			}
		case "limit_src":
			if value != "explicit" && value != "default" {
				return owner, fmt.Errorf("recorded helper: invalid owner limit source")
			}
			owner.limitSource = value
		case "orphan-of":
			owner.orphanOf, err = strconv.Atoi(value)
			if err != nil || owner.orphanOf <= 1 {
				return owner, fmt.Errorf("recorded helper: invalid orphan owner")
			}
		default:
			return owner, fmt.Errorf("recorded helper: unsupported owner field %q", key)
		}
	}
	if !seen["epoch"] || !seen["at"] || (owner.token != "" && seen["lane"] == seen["orphan-of"]) {
		return owner, fmt.Errorf("recorded helper: incomplete owner record")
	}
	return owner, nil
}

func recordedScriptOwner(body []byte, pid int, host, label string) (recordedLeaseOwner, error) {
	owner, err := recordedOwner(body)
	if err != nil {
		return owner, err
	}
	short, _, _ := strings.Cut(host, ".")
	prefix := short + "-" + strconv.Itoa(pid) + "-"
	suffix, ok := strings.CutPrefix(owner.token, prefix)
	epoch, random, split := strings.Cut(suffix, "-")
	digits := func(s string) bool {
		return s != "" && strings.IndexFunc(s, func(r rune) bool { return r < '0' || r > '9' }) < 0
	}
	if short == "" || owner.host != short || owner.pid != pid || owner.label != label ||
		owner.orphanOf != 0 || !ok || !split || !digits(epoch) || !digits(random) {
		return owner, fmt.Errorf("recorded helper: actual script owner mismatch")
	}
	return owner, nil
}

func recordedLeaseMatches(owner recordedLeaseOwner, token string, expected, limit int64) error {
	if owner.token == "" || (token != "" && owner.token != token) || expected <= 0 || limit <= 0 ||
		owner.expected != expected || owner.limit != limit || owner.limitSource != "explicit" {
		return fmt.Errorf("recorded helper: actual script lease changed")
	}
	return nil
}

// A present owner record may belong to a newer Go or shell root. Only our
// holder/token/orphan identity prevents retirement; this function never writes
// the namespace. A missing owner is handled by the bounded file reader.
func recordedLeaseRetired(body []byte, holder int, token string) error {
	owner, err := recordedOwner(body)
	if err != nil {
		return err
	}
	if owner.pid == holder || owner.orphanOf == holder || (token != "" && owner.token == token) {
		return fmt.Errorf("recorded helper: owned lease not released")
	}
	return nil
}

func recordedInherited(ctx context.Context, env func(string) string) (cudalock.Owner, error) {
	for {
		if err := ctx.Err(); err != nil {
			return cudalock.Owner{}, err
		}
		owner, err := cudalock.Inherited(env)
		if err == nil {
			if err := ctx.Err(); err != nil {
				return cudalock.Owner{}, err
			}
			return owner, nil
		}
		if !errors.Is(err, syscall.EAGAIN) && !errors.Is(err, syscall.EWOULDBLOCK) {
			return cudalock.Owner{}, err
		}
		timer := time.NewTimer(10 * time.Millisecond)
		select {
		case <-ctx.Done():
			timer.Stop()
			return cudalock.Owner{}, ctx.Err()
		case <-timer.C:
		}
	}
}

// H can create a separate S group before its peer connects. Empty retained
// H-group inventory therefore does not prove retirement of an unknown S.
func recordedClosedCustody(c RecordedCustody, cleanupErr error) bool {
	if cleanupErr != nil {
		return false
	}
	if c.Holder.PID <= 1 {
		return c.Phase == "keeper-ready" && c.SupervisorCreation == "not-started" && c.Supervisor.PID == 0
	}
	return c.SupervisorCreation == "observed" && c.Supervisor.PID > 1 && c.Supervisor.Group > 1
}
