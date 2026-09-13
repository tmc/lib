package cudaroofline

import (
	"fmt"
	"io"
	"math"
)

// RenderSVG writes an interactive, publication-ready SVG roofline chart for the model
// and the evaluated kernel points.
func (m *Model) RenderSVG(w io.Writer, points []KernelPoint) error {
	const (
		svgWidth  = 800
		svgHeight = 520
		padLeft   = 90.0
		padRight  = 50.0
		padTop    = 50.0
		padBottom = 60.0
	)

	plotWidth := float64(svgWidth) - padLeft - padRight
	plotHeight := float64(svgHeight) - padTop - padBottom

	// Logarithmic bounds for axes:
	// X: Arithmetic Intensity (0.01 to 10,000 FLOP/B) -> -2 to 4
	minLogX := -2.0
	maxLogX := 4.0
	spanLogX := maxLogX - minLogX

	// Y: Performance (0.1 to 1,000 TFLOPS) -> -1 to 3
	minLogY := -1.0
	maxLogY := 3.0
	spanLogY := maxLogY - minLogY

	xToPixel := func(intensity float64) float64 {
		if intensity <= 0 {
			intensity = 0.01
		}
		logVal := math.Log10(intensity)
		if logVal < minLogX {
			logVal = minLogX
		}
		if logVal > maxLogX {
			logVal = maxLogX
		}
		return padLeft + ((logVal - minLogX) / spanLogX) * plotWidth
	}

	yToPixel := func(tflops float64) float64 {
		if tflops <= 0 {
			tflops = 0.1
		}
		logVal := math.Log10(tflops)
		if logVal < minLogY {
			logVal = minLogY
		}
		if logVal > maxLogY {
			logVal = maxLogY
		}
		return padTop + ((maxLogY - logVal) / spanLogY) * plotHeight
	}

	fmt.Fprintf(w, `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d" style="background:#ffffff; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,sans-serif;">
  <style>
    .grid { stroke: #e5e7eb; stroke-width: 1px; }
    .axis { stroke: #374151; stroke-width: 1.5px; }
    .axis-label { font-size: 13px; fill: #111827; font-weight: 600; text-anchor: middle; }
    .tick-label { font-size: 11px; fill: #4b5563; text-anchor: middle; }
    .tick-label-y { font-size: 11px; fill: #4b5563; text-anchor: end; dominant-baseline: middle; }
    .roofline { stroke: #2563eb; stroke-width: 2.5px; fill: none; }
    .l2-roofline { stroke: #93c5fd; stroke-width: 1.5px; stroke-dasharray: 4 4; fill: none; }
    .pt-memory { fill: #0284c7; stroke: #0369a1; stroke-width: 1.5px; }
    .pt-compute { fill: #dc2626; stroke: #b91c1c; stroke-width: 1.5px; }
    .pt-unconstrained { fill: #d97706; stroke: #b45309; stroke-width: 1.5px; }
    .pt-label { font-size: 11px; fill: #1f2937; font-weight: 500; }
    .legend-text { font-size: 11px; fill: #374151; }
  </style>

  <!-- Title -->
  <text x="%f" y="30" text-anchor="middle" font-size="16" font-weight="700" fill="#111827">Roofline Model (%s)</text>

  <!-- Background Plot Area -->
  <rect x="%f" y="%f" width="%f" height="%f" fill="#fafafa" stroke="#d1d5db" stroke-width="1"/>
`, svgWidth, svgHeight, svgWidth, svgHeight, float64(svgWidth)/2.0, m.Ceiling.Architecture, padLeft, padTop, plotWidth, plotHeight)

	// Horizontal grid lines (Y: 0.1, 1, 10, 100, 1000)
	for yExp := minLogY; yExp <= maxLogY; yExp += 1.0 {
		val := math.Pow(10, yExp)
		py := yToPixel(val)
		fmt.Fprintf(w, `  <line x1="%f" y1="%f" x2="%f" y2="%f" class="grid"/>
  <text x="%f" y="%f" class="tick-label-y">%.1g</text>
`, padLeft, py, padLeft+plotWidth, py, padLeft-10.0, py, val)
	}

	// Vertical grid lines (X: 0.01, 0.1, 1, 10, 100, 1000, 10000)
	for xExp := minLogX; xExp <= maxLogX; xExp += 1.0 {
		val := math.Pow(10, xExp)
		px := xToPixel(val)
		fmt.Fprintf(w, `  <line x1="%f" y1="%f" x2="%f" y2="%f" class="grid"/>
  <text x="%f" y="%f" class="tick-label">%.1g</text>
`, px, padTop, px, padTop+plotHeight, px, padTop+plotHeight+18.0, val)
	}

	// Axis Titles
	fmt.Fprintf(w, `  <text x="%f" y="%f" class="axis-label">Arithmetic Intensity (FLOPs / Byte)</text>
  <text x="25" y="%f" class="axis-label" transform="rotate(-90 25 %f)">Performance (TFLOPS)</text>
`, padLeft+(plotWidth/2.0), padTop+plotHeight+45.0, padTop+(plotHeight/2.0), padTop+(plotHeight/2.0))

	// Roofline curve: DRAM Bandwidth Slope and Peak Compute Ceiling
	knee := m.KneeIntensity()
	xMin := math.Pow(10, minLogX)
	xMax := math.Pow(10, maxLogX)

	// Points on roofline:
	pStart := [2]float64{xToPixel(xMin), yToPixel(m.CeilingAt(xMin))}
	pKnee := [2]float64{xToPixel(knee), yToPixel(m.Ceiling.PeakTFLOPS)}
	pEnd := [2]float64{xToPixel(xMax), yToPixel(m.Ceiling.PeakTFLOPS)}

	// Optional L2 Roofline
	if m.Ceiling.L2BandwidthGBps > m.Ceiling.PeakMemoryGBps {
		l2Knee := (m.Ceiling.PeakTFLOPS * 1000.0) / m.Ceiling.L2BandwidthGBps
		l2AtMin := xMin * (m.Ceiling.L2BandwidthGBps / 1000.0)
		fmt.Fprintf(w, `  <!-- L2 Cache Ceiling -->
  <polyline points="%f,%f %f,%f" class="l2-roofline"/>
  <text x="%f" y="%f" fill="#60a5fa" font-size="10" font-weight="600">L2: %.0f GB/s</text>
`, xToPixel(xMin), yToPixel(l2AtMin), xToPixel(l2Knee), yToPixel(m.Ceiling.PeakTFLOPS), xToPixel(xMin)+5, yToPixel(l2AtMin)-5, m.Ceiling.L2BandwidthGBps)
	}

	// DRAM Roofline
	fmt.Fprintf(w, `  <!-- DRAM Ceiling & Peak Compute -->
  <polyline points="%f,%f %f,%f %f,%f" class="roofline"/>
  <text x="%f" y="%f" fill="#1d4ed8" font-size="11" font-weight="600">DRAM: %.0f GB/s</text>
  <text x="%f" y="%f" fill="#1d4ed8" font-size="11" font-weight="600">Peak: %.1f TFLOPS</text>
`, pStart[0], pStart[1], pKnee[0], pKnee[1], pEnd[0], pEnd[1],
		pStart[0]+5, pStart[1]-6, m.Ceiling.PeakMemoryGBps,
		pKnee[0]+15, pKnee[1]-8, m.Ceiling.PeakTFLOPS)

	// Plot Kernel Points
	for _, pt := range points {
		if pt.Intensity <= 0 || pt.AttainedTFLOPS <= 0 {
			continue
		}
		cx := xToPixel(pt.Intensity)
		cy := yToPixel(pt.AttainedTFLOPS)

		classStyle := "pt-unconstrained"
		switch pt.Classification {
		case ClassMemoryBound:
			classStyle = "pt-memory"
		case ClassComputeBound:
			classStyle = "pt-compute"
		}

		fmt.Fprintf(w, `  <g>
    <circle cx="%f" cy="%f" r="5" class="%s">
      <title>%s&#10;Intensity: %.2f FLOP/B&#10;Attained: %.2f TFLOPS (%.1f GB/s)&#10;Efficiency: %.1f%%%%&#10;Class: %s</title>
    </circle>
    <text x="%f" y="%f" class="pt-label">%s</text>
  </g>
`, cx, cy, classStyle, pt.Name, pt.Intensity, pt.AttainedTFLOPS, pt.AttainedGBps, pt.Efficiency*100.0, pt.Classification, cx+8.0, cy+4.0, pt.Name)
	}

	// Legend
	legX := padLeft + plotWidth - 210.0
	legY := padTop + plotHeight - 80.0
	fmt.Fprintf(w, `  <!-- Legend -->
  <rect x="%f" y="%f" width="200" height="70" fill="#ffffff" stroke="#d1d5db" rx="4" opacity="0.95"/>
  <circle cx="%f" cy="%f" r="4.5" class="pt-memory"/>
  <text x="%f" y="%f" class="legend-text">Memory Bound</text>
  <circle cx="%f" cy="%f" r="4.5" class="pt-compute"/>
  <text x="%f" y="%f" class="legend-text">Compute Bound</text>
  <circle cx="%f" cy="%f" r="4.5" class="pt-unconstrained"/>
  <text x="%f" y="%f" class="legend-text">Unconstrained / Latency</text>
</svg>
`, legX, legY,
		legX+15, legY+18, legX+30, legY+22,
		legX+15, legY+38, legX+30, legY+42,
		legX+15, legY+58, legX+30, legY+62)

	return nil
}
