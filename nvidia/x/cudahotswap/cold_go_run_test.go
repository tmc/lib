package cudahotswap

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/internal/nativereceipt"
)

type publicGoRunBinding struct {
	GoBinary string
	GoSHA256 string
	Workdir  string
	Files    map[string]string
}

func publicGoRunHash(path string) (string, error) {
	info, err := os.Lstat(path)
	if err != nil {
		return "", err
	}
	if !info.Mode().IsRegular() || info.Size() > 16<<20 {
		return "", errors.New("go run binding: nonregular or oversized file")
	}
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()
	current, err := file.Stat()
	if err != nil || !os.SameFile(info, current) {
		return "", errors.New("go run binding: file changed before open")
	}
	hash := sha256.New()
	count, err := io.Copy(hash, io.LimitReader(file, (16<<20)+1))
	if err != nil || count != info.Size() {
		return "", errors.New("go run binding: file size changed")
	}
	return hex.EncodeToString(hash.Sum(nil)), nil
}

func publicGoRunFiles(root string) ([]string, error) {
	var files []string
	err := filepath.WalkDir(root, func(path string, entry os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() {
			return nil
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		{
			if entry.Type()&os.ModeSymlink != 0 {
				return errors.New("go run binding: source symlink")
			}
			files = append(files, filepath.ToSlash(rel))
			if len(files) > 10000 {
				return errors.New("go run binding: too many source files")
			}
		}
		return nil
	})
	sort.Strings(files)
	return files, err
}

func (binding publicGoRunBinding) check() error {
	if !filepath.IsAbs(binding.GoBinary) || !filepath.IsAbs(binding.Workdir) || len(binding.GoSHA256) != 64 {
		return errors.New("go run binding: absolute tool, module and tool digest required")
	}
	got, err := publicGoRunHash(binding.GoBinary)
	if err != nil || got != binding.GoSHA256 {
		return fmt.Errorf("go run binding: tool digest differs: %v", err)
	}
	files, err := publicGoRunFiles(binding.Workdir)
	if err != nil {
		return err
	}
	if len(files) != len(binding.Files) {
		return errors.New("go run binding: source set differs")
	}
	for _, name := range files {
		expected, ok := binding.Files[name]
		if !ok || len(expected) != 64 {
			return errors.New("go run binding: missing source identity")
		}
		got, err := publicGoRunHash(filepath.Join(binding.Workdir, filepath.FromSlash(name)))
		if err != nil || got != expected {
			return fmt.Errorf("go run binding: source %s differs: %v", name, err)
		}
	}
	for _, name := range []string{"go.mod", "go.sum", "examples/kernelharness/main.go", "examples/kernelharness/run.go"} {
		if binding.Files[name] == "" {
			return fmt.Errorf("go run binding: required source %s absent", name)
		}
	}
	module, err := os.ReadFile(filepath.Join(binding.Workdir, "go.mod"))
	if err != nil || !strings.HasPrefix(string(module), "module github.com/tmc/lib/nvidia\n") {
		return errors.New("go run binding: wrong module")
	}
	return nil
}

func publicGoRunCommand(ctx context.Context, binding publicGoRunBinding, file, receipt string) (*exec.Cmd, error) {
	if err := binding.check(); err != nil {
		return nil, err
	}
	if !filepath.IsAbs(file) || !filepath.IsAbs(receipt) {
		return nil, errors.New("go run command: absolute candidate and receipt required")
	}
	command := exec.CommandContext(ctx, binding.GoBinary, "run", "./examples/kernelharness", "-task", "axpby", "-kernel", file, "-json", "-native-receipt", receipt)
	command.Dir = binding.Workdir
	command.Env = append(os.Environ(), "GOWORK=off", "GOFLAGS=", "GOTOOLCHAIN=local")
	command.WaitDelay = 2 * time.Second
	// No new process group: the admitted guardian owns Go and its harness child.
	// Cancellation kills the launcher, but cannot prove descendant retirement;
	// the consumer must abort and retain that ownership for the outer guardian.
	return command, nil
}

func publicGoRunProducer(launcher, parent nativereceipt.Process, worker nativereceipt.Process) error {
	if launcher.PID <= 1 || launcher.Parent != parent.PID || launcher.Group != parent.Group || launcher.Start == 0 || launcher.ExecutableSHA256 == "" {
		return errors.New("go run receipt: launcher identity differs")
	}
	if worker.PID <= 1 || worker.PID == launcher.PID || worker.Parent != launcher.PID || worker.Group != parent.Group || worker.Start == 0 || worker.ExecutableSHA256 == "" {
		return errors.New("go run receipt: actual harness identity differs")
	}
	return nil
}

func publicReadGoRunBinding(reader io.Reader) (publicGoRunBinding, error) {
	var binding publicGoRunBinding
	data, err := io.ReadAll(io.LimitReader(reader, (2<<20)+1))
	if err != nil || len(data) > 2<<20 {
		return binding, errors.New("go run binding: input too large or unreadable")
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	token, err := decoder.Token()
	if err != nil || token != json.Delim('{') {
		return binding, errors.New("go run binding: object required")
	}
	seen := make(map[string]bool)
	for decoder.More() {
		token, err := decoder.Token()
		if err != nil {
			return binding, err
		}
		key, ok := token.(string)
		if !ok || seen[key] {
			return binding, errors.New("go run binding: duplicate key")
		}
		seen[key] = true
		switch key {
		case "GoBinary":
			err = decoder.Decode(&binding.GoBinary)
		case "GoSHA256":
			err = decoder.Decode(&binding.GoSHA256)
		case "Workdir":
			err = decoder.Decode(&binding.Workdir)
		case "Files":
			token, err = decoder.Token()
			if err != nil || token != json.Delim('{') {
				return binding, errors.New("go run binding: files object required")
			}
			binding.Files = make(map[string]string)
			for decoder.More() {
				token, err = decoder.Token()
				if err != nil {
					return binding, err
				}
				name, ok := token.(string)
				if !ok {
					return binding, errors.New("go run binding: file key required")
				}
				if _, ok := binding.Files[name]; ok {
					return binding, errors.New("go run binding: duplicate file")
				}
				if len(binding.Files) >= 10000 {
					return binding, errors.New("go run binding: too many files")
				}
				var digest string
				if err = decoder.Decode(&digest); err != nil {
					return binding, err
				}
				binding.Files[name] = digest
			}
			_, err = decoder.Token()
		default:
			return binding, errors.New("go run binding: unknown key")
		}
		if err != nil {
			return binding, err
		}
	}
	if _, err = decoder.Token(); err != nil {
		return binding, err
	}
	var trailing any
	if err = decoder.Decode(&trailing); err != io.EOF {
		return binding, errors.New("go run binding: trailing data")
	}
	if decoder.InputOffset() > 2<<20 {
		return binding, errors.New("go run binding: input too large")
	}
	return binding, nil
}

func publicColdEdits(t *testing.T, edit func(*testing.T, int)) {
	for i := 0; i < 20; i++ {
		if !t.Run(fmt.Sprintf("edit_%02d", i), func(t *testing.T) { edit(t, i) }) {
			return
		}
	}
}

func publicColdPushCheck(prior, current State, result *PushResult, pushErr error, source string, wrong bool) error {
	expected := OutcomeOK
	if wrong {
		expected = OutcomeWrongAnswer
	}
	if result == nil || result.Report.Outcome != expected || !result.Report.Recognized || result.Report.Replay == nil || !result.Report.Replay.Native || result.Report.Replay.Passed == wrong || result.Published == wrong || result.RolledBack || (pushErr != nil) != wrong {
		return errors.New("cold comparison: wrong public push outcome")
	}
	digest := fmt.Sprintf("%x", sha256.Sum256([]byte(source)))
	if result.Report.Compile == nil || result.Report.Compile.SourceSHA256 != digest || result.Selected != current {
		return errors.New("cold comparison: push source or selected state differs")
	}
	if wrong {
		if current != prior {
			return errors.New("cold comparison: refused replay changed selected code")
		}
	} else if result.State != current || current.SourceDigest != digest || current.Generation != prior.Generation+1 || current.Version <= prior.Version || current.ConfigRevision != prior.ConfigRevision {
		return errors.New("cold comparison: correct push publication differs")
	}
	return nil
}
