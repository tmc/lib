//go:build native && linux

package cudaheadroom_test

import (
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/internal/nativereceipt"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cudaheadroom"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudashared"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

//go:embed testdata/headroom.cu
var headroomSource string

func TestNativeConfiguredHeadroom(t *testing.T) {
	if os.Getenv("HEADROOM_NATIVE") != "1" {
		t.Skip("PENDING HARDWARE: set HEADROOM_NATIVE=1 for the reviewed native packet")
	}
	dir := os.Getenv("HEADROOM_RECEIPTS")
	if !filepath.IsAbs(dir) || filepath.Clean(dir) != dir {
		t.Fatal("HEADROOM_RECEIPTS must name an existing absolute task directory")
	}
	if info, err := os.Lstat(dir); err != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		t.Fatal("receipt directory is unavailable or is a symlink")
	}
	owner, err := cudalock.Inherited(os.Getenv)
	if err != nil {
		t.Fatalf("native gate requires actual inherited gpu-lock owner: %v", err)
	}
	dev := cudatest.RequireDevice(t)
	defer func() {
		if err := dev.Close(); err != nil {
			t.Error(err)
		}
	}()
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	compiled, err := nvrtcx.Compile(headroomSource, nvrtcx.WithArch(major, minor))
	if err != nil {
		t.Fatal(err)
	}
	image := []byte(compiled.PTX())
	if len(image) == 0 {
		t.Fatal("compiler returned no selected PTX")
	}
	sourceSHA, imageSHA := digest(headroomSource), digest(string(image))
	if err := writeNew(filepath.Join(dir, "source.cu"), []byte(headroomSource)); err != nil {
		t.Fatal(err)
	}
	if err := writeNew(filepath.Join(dir, "selected.ptx"), image); err != nil {
		t.Fatal(err)
	}
	session, err := nativereceipt.New()
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := session.Write(filepath.Join(dir, "runtime.json")); err != nil {
			t.Error(err)
		}
		if err := session.Close(); err != nil {
			t.Error(err)
		}
	}()
	if err := session.Capture("before_load", sourceSHA, imageSHA); err != nil {
		t.Fatal(err)
	}
	module, _, err := compiled.LoadOwnedOutput(dev, nvrtcx.OutputPTX)
	if err != nil {
		var cleanup io.Closer
		if errors.As(err, &cleanup) {
			if closeErr := cleanup.Close(); closeErr != nil {
				t.Error(closeErr)
			}
		}
		t.Fatal(err)
	}
	defer func() {
		if err := module.Close(); err != nil {
			t.Error(err)
		}
	}()
	fn, err := module.Function("headroom")
	if err != nil {
		t.Fatal(err)
	}
	initial, err := cudaheadroom.Inspect(fn)
	if err != nil {
		t.Fatal(err)
	}
	// The only static declaration in this exact source is fixed[256]. Volatile
	// first/last accesses force storage; the measured total must agree literally.
	if initial.CompiledStatic != 256 {
		t.Fatalf("compiled fixed[256] storage = %d, want 256", initial.CompiledStatic)
	}
	boundary := initial.UserLimit - initial.CompiledStatic
	if boundary <= 1 || boundary > 1<<24 {
		t.Fatalf("measured dynamic boundary is unsupported: %d", boundary)
	}
	if err := fn.SetAttribute(driver.FuncAttributeMaxDynamicSharedSizeBytes, int(boundary)); err != nil {
		t.Fatal(err)
	}
	configured, err := cudaheadroom.Inspect(fn)
	if err != nil {
		t.Fatal(err)
	}
	if configured.FunctionDynamic != boundary || configured.Ceiling != boundary {
		t.Fatalf("setter/readback = %+v, requested %d", configured, boundary)
	}
	out, err := cudamem.Alloc[uint32](dev, 4)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := out.Free(); err != nil {
			t.Error(err)
		}
	}()
	plan := sharedPlan(t, configured, boundary)
	launch := cudalaunch.OwnedKernel(fn).Grid(1, 1, 1).Block(1, 1, 1)
	if _, err := launch.PrepareSharedPlan(plan, 0); err != nil {
		t.Fatal(err)
	}
	var count cudalaunch.Counter
	if err := launch.Count(&count).Launch(out, uint32(boundary)); err != nil {
		t.Fatal(err)
	}
	got := make([]uint32, 4)
	if err := out.CopyToHost(got); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, []uint32{17, 19, 23, 29}) || count.Value() != 1 {
		t.Fatalf("boundary readback = %v, submissions %d", got, count.Value())
	}
	boundaryReadback := append([]uint32(nil), got...)
	// A bound-plus-one plan is valid in the intentionally permissive host plan,
	// but must be refused by fresh measured device/function checks before launch.
	poison := []uint32{101, 103, 107, 109}
	if err := out.CopyFromHost(poison); err != nil {
		t.Fatal(err)
	}
	tooLarge := sharedPlan(t, configured, boundary+1)
	if _, err := cudaheadroom.CheckPlan(fn, tooLarge, 0); err == nil || !strings.Contains(err.Error(), "exceeds ceiling") {
		t.Fatalf("overbudget check = %v", err)
	}
	if _, err := cudalaunch.OwnedKernel(fn).PrepareSharedPlan(tooLarge, 0); err == nil {
		t.Fatal("overbudget launcher preparation accepted")
	}
	if count.Value() != 1 {
		t.Fatal("refusal submitted a kernel")
	}
	if err := out.CopyToHost(got); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, poison) {
		t.Fatalf("refusal changed output: %v", got)
	}
	refusedReadback := append([]uint32(nil), got...)
	// A previously prepared launcher must also refuse a later lower function cap.
	lower := int64(1024)
	if lower >= boundary {
		t.Fatal("measured device does not admit the declared lower-cap control")
	}
	if err := fn.SetAttribute(driver.FuncAttributeMaxDynamicSharedSizeBytes, int(lower)); err != nil {
		t.Fatal(err)
	}
	low, err := cudaheadroom.Inspect(fn)
	if err != nil {
		t.Fatal(err)
	}
	if low.FunctionDynamic != lower || low.Ceiling != lower {
		t.Fatalf("lower-cap readback = %+v", low)
	}
	if err := launch.Launch(out, uint32(boundary)); err == nil || !strings.Contains(err.Error(), "exceeds ceiling") {
		t.Fatalf("lower cap launch = %v", err)
	}
	if count.Value() != 1 {
		t.Fatal("lower-cap refusal submitted a kernel")
	}
	if err := out.CopyToHost(got); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, poison) {
		t.Fatal("lower-cap refusal changed output")
	}
	lowerRefusedReadback := append([]uint32(nil), got...)
	if _, err := launch.PrepareSharedPlan(sharedPlan(t, low, lower), 0); err != nil {
		t.Fatal(err)
	}
	if err := launch.Launch(out, uint32(lower)); err != nil {
		t.Fatal(err)
	}
	if err := out.CopyToHost(got); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, []uint32{17, 19, 23, 29}) || count.Value() != 2 {
		t.Fatalf("lower boundary readback = %v, submissions %d", got, count.Value())
	}
	if err := session.Capture("after_arms", sourceSHA, imageSHA); err != nil {
		t.Fatal(err)
	}
	name, err := dev.Name()
	if err != nil {
		t.Fatal(err)
	}
	uuid, err := dev.Raw().Uuid()
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
	process, err := nativereceipt.Identity(os.Getpid())
	if err != nil {
		t.Fatal(err)
	}
	report := struct {
		SourceSHA, ImageSHA, CacheKey, OutputKind, DeviceName              string
		Function                                                           driver.Function
		DeviceUUID                                                         driver.Uuid
		DriverVersion                                                      int
		NVRTC                                                              [2]int32
		Architecture                                                       [2]int
		HolderPID                                                          int
		Process                                                            nativereceipt.Process
		Initial, Configured, Lower                                         cudaheadroom.Budget
		BoundaryReadback, LowerReadback, RefusedOutput, LowerRefusedOutput []uint32
		Submissions                                                        uint64
	}{sourceSHA, imageSHA, compiled.Key(), "ptx", "", fn.Raw(), uuid, driverVersion, [2]int32{nvMajor, nvMinor}, [2]int{major, minor}, owner.PID, process, initial, configured, low, boundaryReadback, got, refusedReadback, lowerRefusedReadback, count.Value()}
	report.DeviceName = name
	body, err := json.Marshal(report)
	if err != nil {
		t.Fatal(err)
	}
	if err := writeNew(filepath.Join(dir, "outcome.json"), body); err != nil {
		t.Fatal(err)
	}
	// This is an actual native test outcome, not a packet qualification flag.
	t.Logf("HEADROOM_NATIVE_OUTCOME %s", body)
}

func sharedPlan(t *testing.T, b cudaheadroom.Budget, dynamic int64) *cudashared.Plan {
	t.Helper()
	// Deliberately above the measured budget so the owned launcher, not this
	// host-only planner, is responsible for the hardware refusal control.
	limit := b.UserLimit + 1024
	p, err := cudashared.NewPlan(cudashared.Limits{StaticMax: limit, DynamicMax: limit, TotalMax: limit, Source: "exact headroom fixture measured input"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := p.Add(cudashared.Request{Name: "fixed", Size: 256, Alignment: 1}); err != nil {
		t.Fatal(err)
	}
	if _, err := p.Add(cudashared.Request{Name: "dynamic", Size: dynamic, Alignment: 1, Dynamic: true}); err != nil {
		t.Fatal(err)
	}
	return p
}

func digest(s string) string { h := sha256.Sum256([]byte(s)); return hex.EncodeToString(h[:]) }

func writeNew(path string, body []byte) (err error) {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, f.Close()) }()
	n, err := f.Write(body)
	if err != nil {
		return err
	}
	if n != len(body) {
		return io.ErrShortWrite
	}
	return f.Sync()
}
