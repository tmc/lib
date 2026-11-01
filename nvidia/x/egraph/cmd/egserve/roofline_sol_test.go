package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestRooflineResolveHardwareSpec(t *testing.T) {
	tests := []struct {
		arch       string
		wantArch   string
		wantDevice string
		wantHBM    float64
		wantL2     float64
		wantFP32   float64
	}{
		{
			arch:       "sm_121a",
			wantArch:   "sm_121a",
			wantDevice: "NVIDIA GB10 Grace-Blackwell",
			wantHBM:    546.0,
			wantL2:     2400.0,
			wantFP32:   31000.0,
		},
		{
			arch:       "sm_100a",
			wantArch:   "sm_100a",
			wantDevice: "NVIDIA GB200 B200",
			wantHBM:    8000.0,
			wantL2:     16000.0,
			wantFP32:   75000.0,
		},
		{
			arch:       "sm_90a",
			wantArch:   "sm_90a",
			wantDevice: "NVIDIA H200 SXM",
			wantHBM:    4800.0,
			wantL2:     12000.0,
			wantFP32:   67000.0,
		},
		{
			arch:       "sm_130a",
			wantArch:   "sm_130a",
			wantDevice: "NVIDIA Vera Rubin VR200",
			wantHBM:    22000.0,
			wantL2:     45000.0,
			wantFP32:   120000.0,
		},
	}

	for _, tc := range tests {
		t.Run(tc.arch, func(t *testing.T) {
			got := ResolveHardwareRooflineSpec(tc.arch)
			if got.Arch != tc.wantArch {
				t.Fatalf("Arch = %q, want %q", got.Arch, tc.wantArch)
			}
			if got.DeviceName != tc.wantDevice {
				t.Fatalf("DeviceName = %q, want %q", got.DeviceName, tc.wantDevice)
			}
			if got.PeakHBMBandwidthGBps != tc.wantHBM {
				t.Fatalf("PeakHBMBandwidthGBps = %.1f, want %.1f", got.PeakHBMBandwidthGBps, tc.wantHBM)
			}
			if got.PeakL2BandwidthGBps != tc.wantL2 {
				t.Fatalf("PeakL2BandwidthGBps = %.1f, want %.1f", got.PeakL2BandwidthGBps, tc.wantL2)
			}
			if got.PeakFP32GFLOPS != tc.wantFP32 {
				t.Fatalf("PeakFP32GFLOPS = %.1f, want %.1f", got.PeakFP32GFLOPS, tc.wantFP32)
			}
			if got.RidgePointFLOPPerByte <= 0 {
				t.Fatalf("RidgePointFLOPPerByte = %.2f, want > 0", got.RidgePointFLOPPerByte)
			}
		})
	}
}

func TestRooflineTrajectory24Candidates(t *testing.T) {
	srv, err := NewEGServer(ServerConfig{
		Arch:      "sm_121a",
		HiddenDim: 256,
		IntermDim: 512,
		Layers:    2,
	})
	if err != nil {
		t.Fatalf("NewEGServer: %v", err)
	}

	iters, err := srv.BuildKernelStreamIterations("simple", 1)
	if err != nil {
		t.Fatalf("BuildKernelStreamIterations: %v", err)
	}
	if len(iters) != 24 {
		t.Fatalf("len(iters) = %d, want 24", len(iters))
	}

	rep, err := BuildRooflineTrajectory("sm_121a", iters)
	if err != nil {
		t.Fatalf("BuildRooflineTrajectory: %v", err)
	}
	if len(rep.Points) != 24 {
		t.Fatalf("len(rep.Points) = %d, want 24", len(rep.Points))
	}
	if rep.InitialSOLPct >= 25.0 {
		t.Fatalf("InitialSOLPct = %.2f, want < 25.0 (expected ~14.6%%)", rep.InitialSOLPct)
	}
	if rep.ChampionSOLPct <= 80.0 {
		t.Fatalf("ChampionSOLPct = %.2f, want > 80.0 (expected ~88.4%%)", rep.ChampionSOLPct)
	}
	if rep.ChampionIter != 23 {
		t.Fatalf("ChampionIter = %d, want 23", rep.ChampionIter)
	}

	for i := 1; i < len(rep.Points); i++ {
		prev := rep.Points[i-1].BestSOLPctSoFar
		cur := rep.Points[i].BestSOLPctSoFar
		if cur < prev {
			t.Fatalf("BestSOLPctSoFar decreased at iter %d: %.2f < %.2f", i, cur, prev)
		}
	}

	p0 := rep.Points[0]
	p3 := rep.Points[3]
	p6 := rep.Points[6]
	p16 := rep.Points[16]
	p22 := rep.Points[22]
	p23 := rep.Points[23]

	// #03 has local-memory spill traffic -> higher HBMBytesTransferred than #00.
	if p3.HBMBytesTransferred <= p0.HBMBytesTransferred {
		t.Fatalf("#03 HBMBytesTransferred = %d, want > #00 (%d) due to local spills", p3.HBMBytesTransferred, p0.HBMBytesTransferred)
	}

	// #06 has uncoalesced sector split -> 4x HBMBytesTransferred of #00.
	if p6.HBMBytesTransferred != p0.HBMBytesTransferred*4 {
		t.Fatalf("#06 HBMBytesTransferred = %d, want 4x #00 (%d)", p6.HBMBytesTransferred, p0.HBMBytesTransferred*4)
	}

	// #16 register tiling shifts ArithmeticIntensityFLOPPerByte rightward vs #00.
	if p16.ArithmeticIntensityFLOPPerByte <= p0.ArithmeticIntensityFLOPPerByte {
		t.Fatalf("#16 AI = %.2f, want > #00 AI (%.2f)", p16.ArithmeticIntensityFLOPPerByte, p0.ArithmeticIntensityFLOPPerByte)
	}

	// #22 and #23 L2::evict_last have higher ArithmeticIntensityFLOPPerByte than #00 and #16.
	if p22.ArithmeticIntensityFLOPPerByte <= p0.ArithmeticIntensityFLOPPerByte {
		t.Fatalf("#22 AI = %.2f, want > #00 AI (%.2f)", p22.ArithmeticIntensityFLOPPerByte, p0.ArithmeticIntensityFLOPPerByte)
	}
	if p23.ArithmeticIntensityFLOPPerByte <= p16.ArithmeticIntensityFLOPPerByte {
		t.Fatalf("#23 AI = %.2f, want > #16 AI (%.2f)", p23.ArithmeticIntensityFLOPPerByte, p16.ArithmeticIntensityFLOPPerByte)
	}
}

func TestRooflineHTTPHandler(t *testing.T) {
	srv, err := NewEGServer(ServerConfig{
		Arch:      "sm_121a",
		HiddenDim: 256,
		IntermDim: 512,
		Layers:    2,
	})
	if err != nil {
		t.Fatalf("NewEGServer: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/v1/kernels/roofline?arch=sm_121a&mode=simple&bucket=1", nil)
	rec := httptest.NewRecorder()
	srv.handleKernelsRoofline(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200: %s", rec.Code, rec.Body.String())
	}

	var rep RooflineTrajectoryReport
	if err := json.Unmarshal(rec.Body.Bytes(), &rep); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}
	if rep.Hardware.Arch != "sm_121a" {
		t.Fatalf("Hardware.Arch = %q, want sm_121a", rep.Hardware.Arch)
	}
	if len(rep.Points) != 24 {
		t.Fatalf("len(rep.Points) = %d, want 24", len(rep.Points))
	}
	if rep.InitialSOLPct >= 25.0 || rep.ChampionSOLPct <= 80.0 {
		t.Fatalf("InitialSOLPct=%.1f%% ChampionSOLPct=%.1f%% out of expected range", rep.InitialSOLPct, rep.ChampionSOLPct)
	}
}
