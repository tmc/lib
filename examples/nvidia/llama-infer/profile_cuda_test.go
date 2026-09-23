package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

// delayedWait introduces host delay when the timer begins its wait, after the
// end marker should already be queued. It does not change device work.
type delayedWait struct {
	context.Context
	once sync.Once
}

func (c *delayedWait) Done() <-chan struct{} {
	c.once.Do(func() { time.Sleep(100 * time.Millisecond) })
	return nil
}

func TestStepTimerExcludesPostEnqueueHostDelay(t *testing.T) {
	dev := cudatest.Require(t)
	defer dev.Close()
	src, err := cudamem.Alloc[byte](dev, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	defer src.Free()
	dst, err := cudamem.Alloc[byte](dev, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	defer dst.Free()
	timer, err := newStepTimer(dev, nil, nil, "probe")
	if err != nil {
		t.Fatal(err)
	}
	defer timer.close()
	ctx := &delayedWait{Context: context.Background()}
	if err := timer.measure(ctx, "copy", nil, func(s *cuda.Stream) error {
		return dst.CopyDeviceToDeviceAsync(src, s)
	}); err != nil {
		t.Fatal(err)
	}
	measured, err := timer.end.ElapsedSince(timer.start)
	if err != nil {
		t.Fatal(err)
	}
	// Positive contamination control: delaying before the end marker must
	// appear in the device timeline interval.
	if err := timer.start.Record(timer.stream); err != nil {
		t.Fatal(err)
	}
	time.Sleep(100 * time.Millisecond)
	if err := timer.end.Record(timer.stream); err != nil {
		t.Fatal(err)
	}
	if err := timer.stream.Synchronize(context.Background()); err != nil {
		t.Fatal(err)
	}
	contaminated, err := timer.end.ElapsedSince(timer.start)
	if err != nil {
		t.Fatal(err)
	}
	if measured >= 50*time.Millisecond || contaminated < 90*time.Millisecond {
		t.Fatalf("post-end delay interval=%v, pre-end delay control=%v", measured, contaminated)
	}
	t.Logf("post-end delay interval=%v, pre-end delay control=%v", measured, contaminated)
}

type recordingTokenizer struct {
	Tokenizer
	tokens []int32
}

func (r *recordingTokenizer) Decode(tokens []int32) (string, error) {
	r.tokens = append(r.tokens, tokens...)
	return r.Tokenizer.Decode(tokens)
}

func TestPinnedProfileTokenParity(t *testing.T) {
	dir := os.Getenv("CUDA_PROFILE_MODEL")
	if dir == "" {
		t.Skip("CUDA_PROFILE_MODEL must name the pinned TinyLlama checkpoint")
	}
	for name, want := range map[string]string{
		"config.json":       "486bedda3a6988332e60d9638a09ca4b260d34ebcf1b19e22cf3b140b63d8fe9",
		"tokenizer.json":    "bcd04f0eadf90287bd26e1a183ac487d8a141b09b06aecb7725bbdd343640f2e",
		"model.safetensors": "6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933",
	} {
		f, err := os.Open(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		h := sha256.New()
		_, err = io.Copy(h, f)
		f.Close()
		if err != nil {
			t.Fatal(err)
		}
		if got := fmt.Sprintf("%x", h.Sum(nil)); got != want {
			t.Fatalf("%s hash = %s, want %s", name, got, want)
		}
	}
	dev := cudatest.Require(t)
	defer dev.Close()
	w, cfg, err := LoadModel(dir)
	if err != nil {
		t.Fatal(err)
	}
	tok, err := LoadHFTokenizer(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	oldSteps := *stepsFlag
	*stepsFlag = 6
	defer func() { *stepsFlag = oldSteps }()
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	var outputs [][]int32
	for _, enabled := range []bool{false, true} {
		r := &recordingTokenizer{Tokenizer: tok}
		var profile bytes.Buffer
		var session *cudaprof.Session
		if enabled {
			session = cudaprof.NewSession(&profile)
		}
		err := dev.WithContext(func() error {
			return runGPUInference(dev, w, cfg, r, []int32{1, 450, 7483, 310, 3444, 338}, major, minor, nil, session)
		})
		if session != nil {
			if closeErr := session.Close(); closeErr != nil {
				t.Fatal(closeErr)
			}
		}
		if err != nil {
			t.Fatal(err)
		}
		if len(r.tokens) != 6 {
			t.Fatalf("generated %d tokens, want 6", len(r.tokens))
		}
		outputs = append(outputs, r.tokens)
		t.Logf("profile=%v token IDs=%v profile bytes=%d", enabled, r.tokens, profile.Len())
	}
	if !reflect.DeepEqual(outputs[0], outputs[1]) {
		t.Fatalf("profile changed token IDs: %v versus %v", outputs[0], outputs[1])
	}
}
