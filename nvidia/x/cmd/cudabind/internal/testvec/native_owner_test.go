//go:build native

package testvec

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/internal/nativereceipt"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

var witnessOwner struct {
	sync.Mutex
	dir     string
	next    int
	session *nativereceipt.Session
	images  [][2]string
	roles   map[string]int
}

// The admitted native process owns this receipt. No controller identity or
// prospective library path is substituted for its own observed mappings.
func TestMain(m *testing.M) {
	if os.Getenv("CUDABIND_NATIVE") != "1" {
		os.Exit(m.Run())
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	dir := os.Getenv("CUDABIND_RECEIPTS")
	if err := witnessDirectory(dir); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	s, err := nativereceipt.New()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	witnessOwner.dir, witnessOwner.session = dir, s
	witnessOwner.roles = make(map[string]int)
	vecModuleWitnessModuleLoader = loadWitnessOwnedModule
	vecModuleWitnessCompilationObserver = retainWitnessCompilation
	if vecModuleWitnessModuleLoader == nil || vecModuleWitnessCompilationObserver == nil {
		fmt.Fprintln(os.Stderr, "witness receipt: qualification hooks absent")
		os.Exit(2)
	}
	if err := s.Capture("witness-before-tests", "", ""); err != nil {
		_ = s.Close()
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	code := m.Run()
	proofPath := filepath.Join(dir, "native-libraries.json")
	err = errors.Join(s.Capture("witness-after-tests", "", ""), s.Write(proofPath))
	err = errors.Join(err, s.Close(), checkWitnessRoles(witnessOwner.roles))
	if err == nil {
		proof, readErr := nativereceipt.Read(proofPath)
		err = readErr
		if err == nil {
			if len(witnessOwner.images) == 0 {
				err = errors.New("witness receipt: actual compiled image absent")
			}
			for _, image := range witnessOwner.images {
				err = errors.Join(err, proof.Check(proof.Process, image[0], image[1]))
			}
		}
		if err == nil {
			err = witnessWrite(filepath.Join(dir, "completion.json"), struct {
				Process       nativereceipt.Process
				TestCode      int
				ReceiptClosed bool
			}{proof.Process, code, true})
		}
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		code = 1
	}
	os.Exit(code)
}

func witnessDirectory(dir string) error {
	if !filepath.IsAbs(dir) || filepath.Clean(dir) != dir {
		return errors.New("witness receipt: absolute CUDABIND_RECEIPTS is required")
	}
	for p := dir; ; p = filepath.Dir(p) {
		st, err := os.Lstat(p)
		if err != nil {
			return fmt.Errorf("witness receipt directory: %w", err)
		}
		if !st.IsDir() || st.Mode()&os.ModeSymlink != 0 {
			return errors.New("witness receipt: ordinary directory required")
		}
		if p == filepath.Dir(p) {
			return nil
		}
	}
}

func witnessWrite(path string, value any) error {
	b, err := json.Marshal(value)
	if err != nil {
		return err
	}
	if len(b) > 1<<20 {
		return errors.New("witness receipt: JSON exceeds bound")
	}
	return witnessBody(path, append(b, '\n'))
}

func witnessBody(path string, body []byte) error {
	if len(body) == 0 || len(body) > 8<<20 {
		return errors.New("witness receipt: body exceeds bound")
	}
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return err
	}
	n, err := f.Write(body)
	if err == nil && n != len(body) {
		err = io.ErrShortWrite
	}
	return errors.Join(err, f.Close())
}

func retainWitnessCompilation(t *testing.T, dev *cuda.Device, function driver.Function, extra string, defines []string, cm *nvrtcx.CompiledModule, output nvrtcx.OutputKind) {
	t.Helper()
	witnessOwner.Lock()
	defer witnessOwner.Unlock()
	if witnessOwner.session == nil || cm == nil || dev == nil || function == 0 {
		t.Fatal("witness receipt: missing process session or compilation")
	}
	source, err := vecModuleFiles.ReadFile("vec.cu")
	if err != nil {
		t.Fatal(err)
	}
	source = append(source, extra...)
	image, kind, err := witnessImage(cm, output)
	if err != nil {
		t.Fatal(err)
	}
	if len(image) == 0 {
		t.Fatal("witness receipt: actual compiled image absent")
	}
	witnessOwner.next++
	if witnessOwner.next > 32 {
		t.Fatal("witness receipt: compilation count exceeds bound")
	}
	dir := filepath.Join(witnessOwner.dir, fmt.Sprintf("compile-%03d", witnessOwner.next))
	if err := os.Mkdir(dir, 0700); err != nil {
		t.Fatal(err)
	}
	for name, body := range map[string][]byte{"source.cu": source, "module." + kind: image} {
		if err := witnessBody(filepath.Join(dir, name), body); err != nil {
			t.Fatal(err)
		}
	}
	sourceSHA := fmt.Sprintf("%x", sha256.Sum256(source))
	imageSHA := fmt.Sprintf("%x", sha256.Sum256(image))
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	uuid, err := dev.Raw().Uuid()
	if err != nil {
		t.Fatal(err)
	}
	name, err := dev.Name()
	if err != nil {
		t.Fatal(err)
	}
	driverVersion, err := driver.DriverGetVersion()
	if err != nil {
		t.Fatal(err)
	}
	nvMajor, nvMinor, err := nvrtc.Version()
	if err != nil {
		t.Fatal(err)
	}
	var nativeContext driver.Context
	if err := dev.WithContext(func() error { var err error; nativeContext, err = driver.CtxGetCurrent(); return err }); err != nil {
		t.Fatal(err)
	}
	if nativeContext == 0 {
		t.Fatal("witness receipt: missing actual context")
	}
	row := struct {
		Test, Kind, SourceSHA256, ArtifactSHA256, CompilerKey string
		Defines                                               []string
		DeviceName                                            string
		UUID                                                  driver.Uuid
		Target                                                [2]int
		Driver                                                int
		NVRTC                                                 [2]int32
		Function, Context                                     uint64
	}{t.Name(), kind, sourceSHA, imageSHA, cm.Key(), defines, name, uuid, [2]int{major, minor}, driverVersion, [2]int32{nvMajor, nvMinor}, uint64(function), uint64(nativeContext)}
	if row.CompilerKey == "" {
		t.Fatal("witness receipt: actual compiler key absent")
	}
	if err := witnessWrite(filepath.Join(dir, "compilation.json"), row); err != nil {
		t.Fatal(err)
	}
	if err := witnessOwner.session.Capture("witness-after-load", sourceSHA, imageSHA); err != nil {
		t.Fatal(err)
	}
	witnessOwner.images = append(witnessOwner.images, [2]string{sourceSHA, imageSHA})
	witnessOwner.roles[t.Name()]++
}

// newWitnessModule is limited to the kernel-only witnesses. It does not expose
// the generated global-variable path and never falls back to another image.
func newWitnessModule(t *testing.T, dev *cuda.Device, cm *nvrtcx.CompiledModule, kind nvrtcx.OutputKind) (*vecModule, error) {
	t.Helper()
	if _, _, err := witnessImage(cm, kind); err != nil {
		return nil, err
	}
	mod, _, err := cm.LoadOwnedOutput(dev, kind)
	if err != nil {
		var pending io.Closer
		if errors.As(err, &pending) {
			t.Cleanup(func() {
				if err := pending.Close(); err != nil {
					t.Error(err)
				}
			})
		}
		return nil, err
	}
	m := &vecModule{dev: dev, mod: mod}
	// Register before resolving functions: a partial function table still owns
	// the module, and cleanup runs before the device's earlier cleanup.
	t.Cleanup(func() {
		if err := m.Close(); err != nil {
			t.Error(err)
		}
	})
	for i, k := range vecModuleKernels {
		name := k.name
		if k.cxx {
			var ok bool
			name, ok = cm.LoweredName(k.name)
			if !ok {
				return nil, fmt.Errorf("witness module: missing lowered name %q", k.name)
			}
		}
		m.kernels[i], err = mod.Function(name)
		if err != nil {
			return nil, err
		}
	}
	return m, nil
}

func witnessImage(cm *nvrtcx.CompiledModule, kind nvrtcx.OutputKind) ([]byte, string, error) {
	if cm == nil {
		return nil, "", fmt.Errorf("witness receipt: nil compilation")
	}
	var image []byte
	var name string
	switch kind {
	case nvrtcx.OutputCUBIN:
		image, name = cm.CUBIN(), "cubin"
	case nvrtcx.OutputPTX:
		image, name = []byte(cm.PTX()), "ptx"
	default:
		return nil, "", fmt.Errorf("witness receipt: unsupported selected output %d", kind)
	}
	if len(image) == 0 {
		return nil, "", fmt.Errorf("witness receipt: selected %s unavailable", name)
	}
	return image, name, nil
}
