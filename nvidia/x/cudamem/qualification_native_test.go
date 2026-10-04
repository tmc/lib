//go:build native

package cudamem_test

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalayout"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

// This packet exercises driver device-to-device transfers, not CUDA kernel
// launches. A plan's vector width describes transfer grouping, not an observed
// vector instruction. The enclosing runner retains binary/source/hardware IDs.
func TestCopyQualification_AllElementsAndAliasRefusal(t *testing.T) {
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Skipf("native copy qualification requires inherited gpu-lock: %v", err)
	}
	dir := os.Getenv("COPY_QUALIFICATION_OUTPUT")
	if !filepath.IsAbs(dir) {
		t.Fatal("absolute COPY_QUALIFICATION_OUTPUT required")
	}
	if err := os.MkdirAll(dir, 0700); err != nil {
		t.Fatal(err)
	}
	dev := cudatest.RequireDevice(t)
	t.Cleanup(func() {
		if err := dev.Close(); err != nil {
			t.Error(err)
		}
	})
	name, err := dev.Name()
	if err != nil {
		t.Fatal(err)
	}
	uuid, err := dev.Raw().Uuid()
	if err != nil {
		t.Fatal(err)
	}
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	version, err := driver.DriverGetVersion()
	if err != nil {
		t.Fatal(err)
	}
	var facts cudatest.QualificationFacts
	passed := true
	for _, tc := range qualificationCopyCases() {
		passed = t.Run(tc.name, func(t *testing.T) {
			parse := func(text string) cudalayout.Layout {
				l, err := cudalayout.ParseLayout(text)
				if err != nil {
					t.Fatal(err)
				}
				return l
			}
			sl, dl := parse(tc.source), parse(tc.destination)
			ss, err := sl.Cosize()
			if err != nil {
				t.Fatal(err)
			}
			ds, err := dl.Cosize()
			if err != nil {
				t.Fatal(err)
			}
			src, err := cudamem.Alloc[uint32](dev, ss+2)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := src.Free(); err != nil {
					t.Error(err)
				}
			})
			dst, err := cudamem.Alloc[uint32](dev, ds+2)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := dst.Free(); err != nil {
					t.Error(err)
				}
			})
			input := make([]uint32, ss+2)
			before := make([]uint32, ds+2)
			for i := range input {
				input[i] = 0x01020300 + uint32(i*17)
			}
			for i := range before {
				before[i] = 0xa5a5a5a5
			}
			before[0], before[len(before)-1] = 0x13579bdf, 0x2468ace0
			want := slices.Clone(before)
			if !tc.refuse {
				for i := 0; i < sl.Size(); i++ {
					so, err := sl.Index(i)
					if err != nil {
						t.Fatal(err)
					}
					do, err := dl.Index(i)
					if err != nil {
						t.Fatal(err)
					}
					want[do+1] = input[so+1]
				}
			}
			if err := src.CopyFromHost(input); err != nil {
				t.Fatal(err)
			}
			if err := dst.CopyFromHost(before); err != nil {
				t.Fatal(err)
			}
			plan, copyErr := cudamem.CopyWithLayouts(dst.Slice(1, ds+1), src.Slice(1, ss+1), sl, dl)
			if tc.refuse {
				if !errors.Is(copyErr, cudamem.ErrWriteAfterWrite) {
					t.Fatalf("copy error=%v, want incompatible alias refusal", copyErr)
				}
			} else if copyErr != nil {
				t.Fatal(copyErr)
			} else if plan.VectorWidth != tc.width || plan.CollapseFactor != tc.collapse {
				t.Fatalf("plan width/collapse=%d/%d, want%d/%d", plan.VectorWidth, plan.CollapseFactor, tc.width, tc.collapse)
			}
			if err := dev.WithContext(driver.CtxSynchronize); err != nil {
				t.Fatal(err)
			}
			got := make([]uint32, ds+2)
			if err := dst.CopyToHost(got); err != nil {
				t.Fatal(err)
			}
			sourceAfter := make([]uint32, ss+2)
			if err := src.CopyToHost(sourceAfter); err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(got, want) || !slices.Equal(sourceAfter, input) {
				t.Fatalf("all-element/guard/source comparison failed: destination=%v want=%v source=%v", got, want, sourceAfter)
			}
			// Corrupt an actual device destination through the same native memory
			// API, then prove that the unchanged all-element checker rejects it.
			if err := dst.Slice(1, 2).CopyFromHost([]uint32{want[1] ^ 1}); err != nil {
				t.Fatal(err)
			}
			corrupt := make([]uint32, ds+2)
			if err := dst.CopyToHost(corrupt); err != nil {
				t.Fatal(err)
			}
			if slices.Equal(corrupt, want) || corrupt[0] != want[0] || corrupt[len(corrupt)-1] != want[len(want)-1] {
				t.Fatal("corrupted device output control or its guards failed")
			}
			receipt := struct {
				SourceLayout, DestinationLayout string
				Plan                            cudamem.CopyPlan
				Input, Before, Actual, Expected []uint32
				SourceAfter, Corrupted          []uint32
				RefusedBeforeWrites             bool
				CopyExecuted, CopyCompleted     bool
				SynchronizationChecked          bool
			}{tc.source, tc.destination, plan, input, before, got, want, sourceAfter, corrupt, tc.refuse, !tc.refuse, !tc.refuse, true}
			data, err := json.MarshalIndent(receipt, "", "  ")
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(dir, tc.name+".json"), append(data, '\n'), 0600); err != nil {
				t.Fatal(err)
			}
			// Count completed owner invocations, not individual driver transfers.
			facts.Synchronizations++
			facts.NegativeRuns++ // actual device corruption rejected above
			facts.Expected += uint64(ss + ds)
			facts.Checked += uint64(ss + ds)
			facts.GuardsExpected += 4
			facts.GuardsChecked += 4
			if tc.refuse {
				facts.AliasRefusals++
				facts.NegativeRuns++
			} else {
				facts.PositiveRuns++
				facts.CopyExecutions++
				facts.CompletedCopyExecutions++
				if plan.VectorWidth == 1 {
					facts.ScalarRuns++
				} else {
					facts.VectorRuns++
				}
				if plan.CollapseFactor > 1 {
					facts.CollapsedAliasRuns++
				}
			}
			t.Logf("actual D2D copy %s: checked_destination=%d checked_source=%d guards=4 width=%d collapse=%d alias_refusal=%t corrupted_control=detected", tc.name, ds, ss, plan.VectorWidth, plan.CollapseFactor, tc.refuse)
		}) && passed
	}
	if !passed {
		return
	}
	// Retain the actual temporary go-test executable. A prior -c build is
	// preparation only and cannot identify the binary this direct go test ran.
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(executable)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		t.Fatal(err)
	}
	if !info.Mode().IsRegular() || info.Size() > 16<<20 {
		t.Fatal("executing binary exceeds qualification artifact limit")
	}
	binary, err := io.ReadAll(io.LimitReader(f, (16<<20)+1))
	if err != nil {
		t.Fatal(err)
	}
	if len(binary) > 16<<20 {
		t.Fatal("executing binary grew beyond qualification artifact limit")
	}
	if err := os.WriteFile(filepath.Join(dir, "executed.test"), binary, 0600); err != nil {
		t.Fatal(err)
	}
	producer := struct {
		GoVersion, Platform, DriverLibrary, DeviceName, DeviceUUID, ComputeCapability string
		DriverVersion                                                                 int
		ExecutableSHA256                                                              string
		Facts                                                                         cudatest.QualificationFacts
	}{runtime.Version(), runtime.GOOS + "/" + runtime.GOARCH, driver.Path(), name, fmt.Sprintf("%x", uuid.Bytes), fmt.Sprintf("%d.%d", major, minor), version, fmt.Sprintf("%x", sha256.Sum256(binary)), facts}
	data, err := json.MarshalIndent(producer, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "producer.json"), append(data, '\n'), 0600); err != nil {
		t.Fatal(err)
	}
}
