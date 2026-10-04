//go:build native

package cudablock

import (
	"fmt"
	"path/filepath"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudakernel"
)

type topkTimingReceipt struct {
	// Wall synchronization includes waiting for device work. These fields are
	// separate observations, not additive pieces of a total duration.
	Fixture               string
	Config                TopKConfig
	Image                 topkPacketEntry
	Threads               uint32
	ScratchBytes          uint64
	InputBytes            uint64
	ValueBytes            uint64
	SetupWallNS           int64
	TransferEventNS       int64
	SelectionEventNS      []int64
	PublicationEventNS    []int64
	StatusReadbackWallNS  []int64
	SynchronizationWallNS []int64
	CUPTI                 string
	TimingSemantics       string
	Result                topkNativeReceipt
}

type topkTimingSampleResult struct {
	selection, publication            time.Duration
	readbackWall, synchronizationWall time.Duration
}

func topkTimingSample(t *testing.T, owner *TopK, stream *cuda.Stream, b TopKBuffers, threads uint32) (topkTimingSampleResult, error) {
	t.Helper()
	var result topkTimingSampleResult
	owner.mu.Lock()
	defer owner.mu.Unlock()
	if owner.pendingStream != nil {
		t.Fatal("timing owner awaiting drain")
	}
	releases, err := owner.borrow(b)
	if err != nil {
		t.Fatal(err)
	}
	owner.pendingStream, owner.pendingRelease = stream, releases
	var events [4]*cuda.Event
	for i := range events {
		events[i], err = owner.device.NewEvent()
		if err != nil {
			t.Fatal(err)
		}
		event := events[i]
		t.Cleanup(func() {
			if err := event.Close(); err != nil {
				t.Error(err)
			}
		})
	}
	complete, err := topkInvoke(owner.config, func() error {
		if err := events[0].Record(stream); err != nil {
			return err
		}
		if err := owner.launchInput(stream, b, b.Input.Ptr(), threads); err != nil {
			return err
		}
		return events[1].Record(stream)
	}, func() error {
		start := time.Now()
		err := stream.Synchronize(nil)
		result.synchronizationWall += time.Since(start)
		return err
	}, func() ([]int32, []int32, error) {
		start := time.Now()
		defer func() { result.readbackWall = time.Since(start) }()
		status, counts := make([]int32, owner.config.Rows), make([]int32, owner.config.Rows)
		if err := b.Status.CopyToHost(status); err != nil {
			return nil, nil, err
		}
		if err := b.ScratchCounts.CopyToHost(counts); err != nil {
			return nil, nil, err
		}
		return status, counts, nil
	}, func() error {
		if err := events[2].Record(stream); err != nil {
			return err
		}
		if err := b.Indices.CopyDeviceToDeviceAsync(b.ScratchIndices, stream); err != nil {
			return err
		}
		if err := b.Counts.CopyDeviceToDeviceAsync(b.ScratchCounts, stream); err != nil {
			return err
		}
		if owner.config.Values {
			if err := b.Values.CopyDeviceToDeviceAsync(b.ScratchValues, stream); err != nil {
				return err
			}
		}
		return events[3].Record(stream)
	})
	if complete {
		owner.release()
	}
	if err != nil {
		return result, err
	}
	result.selection, err = events[1].ElapsedSince(events[0])
	if err != nil {
		return result, err
	}
	result.publication, err = events[3].ElapsedSince(events[2])
	return result, err
}

func topkTimingCase(t *testing.T, dev *cuda.Device, a *cudakernel.Artifact, image topkPacketEntry, f topkCase, threads uint32) topkTimingReceipt {
	t.Helper()
	setupStart := time.Now()
	owner, err := topkLoadArtifact(dev, f.c, a)
	if owner != nil {
		defer func() {
			if err := owner.Close(); err != nil {
				t.Error(err)
			}
		}()
	}
	if err != nil {
		t.Fatal(err)
	}
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := stream.Close(); err != nil {
			t.Error(err)
		}
	})
	b, checks := topkExtensionBuffers(t, dev, f.c)
	e, _ := f.c.Extents()
	input, inputCheck := topkNativeFloat(t, dev, e.Input)
	b.Input = input
	checks = append(checks, inputCheck)
	setupWall := time.Since(setupStart)
	start, err := dev.NewEvent()
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := start.Close(); err != nil {
			t.Error(err)
		}
	}()
	end, err := dev.NewEvent()
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := end.Close(); err != nil {
			t.Error(err)
		}
	}()
	if err := start.Record(stream); err != nil {
		t.Fatal(err)
	}
	if err := b.Input.CopyFromHostAsync(f.input, stream); err != nil {
		t.Fatal(err)
	}
	if err := b.Lengths.CopyFromHostAsync(f.lengths, stream); err != nil {
		t.Fatal(err)
	}
	if err := b.Offsets.CopyFromHostAsync(f.offsets, stream); err != nil {
		t.Fatal(err)
	}
	if err := end.Record(stream); err != nil {
		t.Fatal(err)
	}
	if err := stream.Synchronize(nil); err != nil {
		t.Fatal(err)
	}
	transfer, err := end.ElapsedSince(start)
	if err != nil {
		t.Fatal(err)
	}
	r := topkTimingReceipt{Fixture: f.name, Config: f.c, Image: image, Threads: threads, ScratchBytes: e.ScratchBytes, InputBytes: uint64(e.Input) * 4, SetupWallNS: int64(setupWall), TransferEventNS: int64(transfer), CUPTI: "unmeasured", TimingSemantics: "event durations compare device work; wall durations describe setup/readback/wait costs and overlap event durations; do not add channels"}
	if f.c.Values {
		r.ValueBytes = uint64(e.Results) * 4
	}
	want := topkSortReference(f)
	// One numerical gate/warm-up followed by exactly five retained samples.
	for sample := 0; sample < 6; sample++ {
		sampleTimes, err := topkTimingSample(t, owner, stream, b, threads)
		if err != nil {
			t.Fatal(err)
		}
		got := topkExtensionRead(t, f.c, b, nil)
		if err := topkCompare(f.c, got, want); err != nil {
			t.Fatalf("timing candidate numerical refusal: %v", err)
		}
		for _, check := range checks {
			check()
		}
		if sample > 0 {
			r.SelectionEventNS = append(r.SelectionEventNS, int64(sampleTimes.selection))
			r.PublicationEventNS = append(r.PublicationEventNS, int64(sampleTimes.publication))
			r.StatusReadbackWallNS = append(r.StatusReadbackWallNS, int64(sampleTimes.readbackWall))
			r.SynchronizationWallNS = append(r.SynchronizationWallNS, int64(sampleTimes.synchronizationWall))
		}
		r.Result = topkNativeReceipt{Fixture: f.name, Config: f.c, Image: image, InputBits: topkFloatBits(f.input), Lengths: f.lengths, Offsets: f.offsets, Indices: got.indices, Counts: got.counts, Status: got.status, ValueBits: topkFloatBits(got.values), ExpectedIndices: want.indices, ExpectedCounts: want.counts, ExpectedStatus: want.status, ExpectedValueBits: topkFloatBits(want.values), Published: true, RedzonesIntact: true}
		r.Result.LoadedResources = topkExtensionResources(t, owner)
	}
	return r
}

// TestNativeTopKTiming compares two fixed images with CUDA events. Four matched
// shapes, two images and six launches each give exactly 48 bounded launches.
// Setup wall time, transfer events, selection events and publication events are
// separate fields. No host wall duration substitutes for kernel event time.
func TestNativeTopKTiming(t *testing.T) {
	dev, target, out := topkExtensionDevice(t)
	ordinary, ordinaryImage := topkReadExtension(t, "ordinary", TopKSource, "topk_fp32", target)
	parallel, parallelImage := topkReadExtension(t, "parallel", topkParallelSource, "topk_fp32", target)
	for _, k := range []int{1, 32} {
		for _, values := range []bool{false, true} {
			f := topkCase{name: fmt.Sprintf("rows4_width4096_k%d_values%t", k, values), c: TopKConfig{4, 4096, 4099, k, values}, lengths: []int32{4096, 129, 33, 0}, offsets: []int32{0, 100, 200, 300}, input: make([]float32, 3*4099+4096)}
			for i := range f.input {
				f.input[i] = float32((i*37)%257) - 128
			}
			base := topkTimingCase(t, dev, ordinary, ordinaryImage, f, 1)
			candidate := topkTimingCase(t, dev, parallel, parallelImage, f, 128)
			topkWriteJSON(t, filepath.Join(out, "timing-"+f.name+".json"), struct{ Ordinary, Parallel topkTimingReceipt }{base, candidate})
		}
	}
}

// TestNativeTopKParallelReference checks the fixed parallel candidate before
// comparison. These 27 launches include nonfinite values and invalid metadata.
func TestNativeTopKParallelReference(t *testing.T) {
	dev, target, out := topkExtensionDevice(t)
	a, image := topkReadExtension(t, "parallel", topkParallelSource, "topk_fp32", target)
	fixtures := topkFixtures()
	if len(fixtures) > 48 {
		t.Fatal("parallel fixture budget exceeded")
	}
	for _, f := range fixtures {
		t.Run(f.name, func(t *testing.T) {
			owner, err := topkLoadArtifact(dev, f.c, a)
			if owner != nil {
				defer func() {
					if err := owner.Close(); err != nil {
						t.Error(err)
					}
				}()
			}
			if err != nil {
				t.Fatal(err)
			}
			stream, err := dev.NewStream()
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := stream.Close(); err != nil {
					t.Error(err)
				}
			})
			b, checks := topkExtensionBuffers(t, dev, f.c)
			e, _ := f.c.Extents()
			input, errCheck := topkNativeFloat(t, dev, e.Input)
			b.Input = input
			checks = append(checks, errCheck)
			if err := b.Input.CopyFromHost(f.input); err != nil {
				t.Fatal(err)
			}
			if err := b.Lengths.CopyFromHost(f.lengths); err != nil {
				t.Fatal(err)
			}
			if err := b.Offsets.CopyFromHost(f.offsets); err != nil {
				t.Fatal(err)
			}
			_, err = topkTimingSample(t, owner, stream, b, 128)
			got, want := topkExtensionRead(t, f.c, b, err), topkSortReference(f)
			for _, check := range checks {
				check()
			}
			r := topkNativeReceipt{Fixture: f.name, Config: f.c, Image: image, InputBits: topkFloatBits(f.input), Lengths: f.lengths, Offsets: f.offsets, Indices: got.indices, Counts: got.counts, Status: got.status, ValueBits: topkFloatBits(got.values), ExpectedIndices: want.indices, ExpectedCounts: want.counts, ExpectedStatus: want.status, ExpectedValueBits: topkFloatBits(want.values), Published: err == nil, RedzonesIntact: true}
			r.LoadedResources = topkExtensionResources(t, owner)
			if err != nil {
				r.ExecuteError = err.Error()
			}
			topkWriteJSON(t, filepath.Join(out, "parallel-"+f.name+".json"), r)
			if e := topkCompare(f.c, got, want); e != nil {
				t.Fatal(e)
			}
			if (err == nil) != (CheckTopKStatus(want.status) == nil) {
				t.Fatal("parallel publication admission mismatch")
			}
		})
	}
}
