//go:build linux

package cudalaunch_test

import (
	"context"
	"math"
	"sync"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

const abiPTX = `.version 8.0
.target sm_80
.address_size 64

.visible .entry arguments(
	.param .u8 p_i8,
	.param .u16 p_i16,
	.param .u32 p_i32,
	.param .u64 p_i64,
	.param .u8 p_u8,
	.param .u16 p_u16,
	.param .u32 p_u32,
	.param .u64 p_u64,
	.param .f32 p_f32,
	.param .f64 p_f64,
	.param .u8 p_bool,
	.param .align 16 .b8 p_vec[16],
	.param .u64 p_out
) {
	.reg .b32 %r;
	.reg .b64 %rd<14>;
	.reg .f32 %f;
	.reg .f64 %d;

	ld.param.u64 %rd0, [p_out];

	ld.param.s8 %r, [p_i8]; cvt.s64.s32 %rd1, %r; st.global.u64 [%rd0+0], %rd1;
	ld.param.s16 %r, [p_i16]; cvt.s64.s32 %rd2, %r; st.global.u64 [%rd0+8], %rd2;
	ld.param.s32 %r, [p_i32]; cvt.s64.s32 %rd3, %r; st.global.u64 [%rd0+16], %rd3;
	ld.param.u64 %rd4, [p_i64]; st.global.u64 [%rd0+24], %rd4;
	ld.param.u8 %r, [p_u8]; cvt.u64.u32 %rd5, %r; st.global.u64 [%rd0+32], %rd5;
	ld.param.u16 %r, [p_u16]; cvt.u64.u32 %rd6, %r; st.global.u64 [%rd0+40], %rd6;
	ld.param.u32 %r, [p_u32]; cvt.u64.u32 %rd7, %r; st.global.u64 [%rd0+48], %rd7;
	ld.param.u64 %rd8, [p_u64]; st.global.u64 [%rd0+56], %rd8;
	ld.param.f32 %f, [p_f32]; mov.b32 %r, %f; cvt.u64.u32 %rd9, %r; st.global.u64 [%rd0+64], %rd9;
	ld.param.f64 %d, [p_f64]; mov.b64 %rd10, %d; st.global.u64 [%rd0+72], %rd10;
	ld.param.u8 %r, [p_bool]; cvt.u64.u32 %rd11, %r; st.global.u64 [%rd0+80], %rd11;
	ld.param.u32 %r, [p_vec+0]; cvt.u64.u32 %rd12, %r; st.global.u64 [%rd0+88], %rd12;
	ld.param.u32 %r, [p_vec+12]; cvt.u64.u32 %rd13, %r; st.global.u64 [%rd0+96], %rd13;
	ret;
}
`

type abiVec struct{ X, Y, Z, W uint32 }

func TestArgumentABIConcurrentStreams(t *testing.T) {
	dev := cudatest.RequireDevice(t)
	defer dev.Close()
	image := append([]byte(abiPTX), 0)
	var mod driver.Module
	var fn driver.Function
	if err := dev.WithContext(func() error {
		var err error
		mod, err = driver.NewModuleLoadData(unsafe.Pointer(&image[0]))
		if err == nil {
			fn, err = mod.GetFunction("arguments")
		}
		return err
	}); err != nil {
		t.Fatalf("load PTX: %v", err)
	}
	defer dev.WithContext(mod.Unload)

	const streams = 4
	var wg sync.WaitGroup
	errs := make(chan error, streams)
	for i := range streams {
		stream, err := dev.NewStreamWithFlags(uint32(driver.StreamNonBlocking))
		if err != nil {
			t.Fatal(err)
		}
		out, err := cudamem.Alloc[uint64](dev, 13)
		if err != nil {
			t.Fatal(err)
		}
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			defer stream.Destroy()
			defer out.Free()
			err := cudalaunch.Kernel(fn).On(stream).Launch(
				int8(-7-i), int16(-300-i), int32(-70000-i), int64(-9000000000-i),
				uint8(7+i), uint16(60000-i), uint32(3000000000+uint32(i)), uint64(1<<60)+uint64(i),
				float32(1.25+float32(i)), float64(-2.5-float64(i)), i%2 == 0,
				abiVec{1 + uint32(i), 2, 3, 4 + uint32(i)}, out,
			)
			if err == nil {
				err = stream.Synchronize(context.Background())
			}
			got := make([]uint64, 13)
			if err == nil {
				err = out.CopyToHost(got)
			}
			want := []uint64{
				uint64(int64(-7 - i)), uint64(int64(-300 - i)), uint64(int64(-70000 - i)), uint64(int64(-9000000000 - i)),
				uint64(7 + i), uint64(60000 - i), uint64(3000000000 + uint32(i)), uint64(1<<60) + uint64(i),
				uint64(math.Float32bits(1.25 + float32(i))), math.Float64bits(-2.5 - float64(i)), uint64(0),
				uint64(1 + uint32(i)), uint64(4 + uint32(i)),
			}
			if i%2 == 0 {
				want[10] = 1
			}
			if err == nil {
				for j := range want {
					if got[j] != want[j] {
						t.Errorf("stream %d result[%d] = %#x, want %#x", i, j, got[j], want[j])
						break
					}
				}
			}
			errs <- err
		}(i)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Errorf("launch: %v", err)
		}
	}
}
