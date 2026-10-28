package cudadevicetable

// The datasheet is the one hand-entered table in this package. It exists
// because a generator must be able to plan for hardware nobody here has
// attached, and it is kept honest the same way the probed table is: every
// entry is [Unverified] and cites its source, and [LookupProfile] replaces
// an entry with the probed value wherever a probe of that product exists.
//
// Sources, abbreviated in the notes:
//
//	[cc]      CUDA Programming Guide 13.x, Compute Capabilities, Tables 30
//	          and 31: docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html
//	[cluster] CUDA Programming Guide, Thread Block Clusters (portable size 8):
//	          docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html
//	[hopper]  Hopper Tuning Guide: docs.nvidia.com/cuda/hopper-tuning-guide/
//	[bw]      Blackwell Tuning Guide: docs.nvidia.com/cuda/blackwell-tuning-guide/
//	[ultra]   "Inside NVIDIA Blackwell Ultra", NVIDIA Technical Blog, 2025:
//	          developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/
//	[ptx]     PTX ISA, Tensor Memory (128 lanes × 512 columns × 32 bits per SM):
//	          docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory
//	[product] the NVIDIA product page or datasheet for the named product
//
// A note that says "uncertain" marks a figure the sources above do not
// state directly.

func sheet(v int64, note string) Value { return Value{Value: v, Source: Unverified, Note: note} }

// Per-compute-capability limits. Compute capabilities 9.0, 10.0 and 10.3
// share them; [cc] lists 228 KB per SM and 227 KB per block for all three.
var (
	hopperBlackwellDC = Profile{
		SharedPerBlock:      sheet(48<<10, "[cc] Table 31 note 3: above 48 KB a block needs dynamic shared memory and an opt-in"),
		SharedPerBlockOptin: sheet(227<<10, "[cc] Table 31: 227 KB per thread block at 9.0, 10.0, 10.3"),
		SharedPerSM:         sheet(228<<10, "[cc] Table 31: 228 KB per SM at 9.0, 10.0, 10.3"),
		ThreadsPerSM:        sheet(2048, "[cc] Table 30: 2048 resident threads per SM at 9.0 through 10.x"),
		BlocksPerSM:         sheet(32, "[cc] Table 30: 32 resident blocks per SM at 9.0 through 10.x"),
		ClusterSize:         sheet(8, "[cluster] the portable maximum"),
	}
	blackwellEdge = Profile{
		SharedPerBlock:      sheet(48<<10, "[cc] Table 31 note 3: above 48 KB a block needs dynamic shared memory and an opt-in"),
		SharedPerBlockOptin: sheet(99<<10, "[cc] Table 31: 99 KB per thread block at 12.x"),
		SharedPerSM:         sheet(100<<10, "[cc] Table 31: 100 KB per SM at 12.x; [bw] says 128 KB, which is the unified L1/shared capacity"),
		ThreadsPerSM:        sheet(1536, "[cc] Table 30: 1536 resident threads per SM at 12.x"),
		BlocksPerSM:         sheet(24, "[cc] Table 30: 24 resident blocks per SM at 12.x"),
		TMEMPerSM:           sheet(0, "no Tensor Memory: 12.x has no tcgen05 (see the tcgen05 feature probe)"),
		ClusterSize:         sheet(8, "[cluster] the portable maximum"),
		ClusterSizeOptin:    sheet(8, "uncertain: no source documents a non-portable cluster size at 12.x, so this is the portable maximum; cudaOccupancyMaxPotentialClusterSize on a device would settle it"),
	}
)

// datasheet lists one Profile per product. The first product of a target
// is the one a bare target name selects.
var datasheet = []Profile{
	product("sm_90a", "H200", hopperBlackwellDC, func(p *Profile) {
		p.TMEMPerSM = sheet(0, "no Tensor Memory: 9.0 has wgmma, not tcgen05")
		p.ClusterSizeOptin = sheet(16, "[hopper] H100 allows a non-portable cluster size of 16; H200 is the same GH100 SM array")
		p.SMs = sheet(132, "[product] H200 SXM: 132 SMs (GH100 as in H100 SXM5)")
		p.MemoryGBps = sheet(4800, "[product] H200 SXM: 4.8 TB/s HBM3e")
		p.NVLinkGBps = sheet(900, "[product] H200 SXM: NVLink 4, 900 GB/s")
	}),
	product("sm_90a", "H100", hopperBlackwellDC, func(p *Profile) {
		p.TMEMPerSM = sheet(0, "no Tensor Memory: 9.0 has wgmma, not tcgen05")
		p.ClusterSizeOptin = sheet(16, "[hopper] H100 allows a non-portable cluster size of 16")
		p.SMs = sheet(132, "[product] H100 SXM5: 132 SMs")
		p.MemoryGBps = sheet(3350, "[product] H100 SXM: 3.35 TB/s HBM3")
		p.NVLinkGBps = sheet(900, "[product] H100 SXM: NVLink 4, 900 GB/s")
	}),
	product("sm_100a", "B200", hopperBlackwellDC, func(p *Profile) {
		p.TMEMPerSM = sheet(256<<10, "[ptx] 128 lanes × 512 columns × 4 bytes")
		p.ClusterSizeOptin = sheet(16, "[bw] B200 allows a non-portable cluster size of 16")
		p.SMs = sheet(148, "uncertain: 148 SMs (74 per die) as reported by chipsandcheese.com's B200 measurements; NVIDIA's pages do not state it")
		p.MemoryGBps = sheet(8000, "[product] HGX B200: 8 TB/s HBM3e per GPU")
		p.NVLinkGBps = sheet(1800, "[product] HGX B200: NVLink 5, 1.8 TB/s per GPU")
	}),
	product("sm_100a", "GB200", hopperBlackwellDC, func(p *Profile) {
		p.TMEMPerSM = sheet(256<<10, "[ptx] 128 lanes × 512 columns × 4 bytes")
		p.ClusterSizeOptin = sheet(16, "[bw] B200 allows a non-portable cluster size of 16; GB200 uses the same GPU")
		p.SMs = sheet(148, "uncertain: assumed equal to B200's 148")
		p.MemoryGBps = sheet(8000, "[product] GB200 NVL72: 8 TB/s HBM3e per GPU")
		p.NVLinkGBps = sheet(1800, "[product] GB200 NVL72: NVLink 5, 1.8 TB/s per GPU")
	}),
	product("sm_103a", "GB300", hopperBlackwellDC, func(p *Profile) {
		p.TMEMPerSM = sheet(256<<10, "[ultra] 256 KB of TMEM per SM")
		p.ClusterSizeOptin = sheet(16, "uncertain: [bw] documents 16 for B200; assumed unchanged at 10.3")
		p.SMs = sheet(160, "uncertain: [ultra] gives 160 SMs for the full Blackwell Ultra chip, not the count enabled on GB300")
		p.MemoryGBps = sheet(8000, "[ultra] 8 TB/s HBM3e per GPU")
		p.NVLinkGBps = sheet(1800, "[ultra] NVLink 5, 1.8 TB/s per GPU (18 links × 100 GB/s)")
	}),
	product("sm_103a", "B300", hopperBlackwellDC, func(p *Profile) {
		p.TMEMPerSM = sheet(256<<10, "[ultra] 256 KB of TMEM per SM")
		p.ClusterSizeOptin = sheet(16, "uncertain: [bw] documents 16 for B200; assumed unchanged at 10.3")
		p.SMs = sheet(160, "uncertain: [ultra] gives 160 SMs for the full Blackwell Ultra chip, not the count enabled on B300")
		p.MemoryGBps = sheet(8000, "[ultra] 8 TB/s HBM3e per GPU")
		p.NVLinkGBps = sheet(1800, "[ultra] NVLink 5, 1.8 TB/s per GPU (18 links × 100 GB/s)")
	}),
	product("sm_120a", "RTX PRO 6000", blackwellEdge, func(p *Profile) {
		p.SMs = sheet(188, "[product] RTX PRO 6000 Blackwell: 24064 CUDA cores, 128 per SM")
		p.MemoryGBps = sheet(1792, "[product] RTX PRO 6000 Blackwell: 1792 GB/s GDDR7")
		p.NVLinkGBps = sheet(0, "[product] no NVLink")
	}),
	product("sm_120a", "RTX 5090", blackwellEdge, func(p *Profile) {
		p.SMs = sheet(170, "[product] RTX 5090: 21760 CUDA cores, 128 per SM")
		p.MemoryGBps = sheet(1792, "[product] RTX 5090: 1792 GB/s GDDR7")
		p.NVLinkGBps = sheet(0, "[product] no NVLink")
	}),
	product("sm_121a", "GB10", blackwellEdge, func(p *Profile) {
		p.SMs = sheet(48, "[product] DGX Spark GB10; replaced by the probe")
		p.MemoryGBps = sheet(273, "[product] DGX Spark: 273 GB/s LPDDR5x; a streaming kernel on dgx sustains about 238")
		p.NVLinkGBps = sheet(0, "[product] no GPU-to-GPU NVLink; NVLink-C2C joins the GPU to the Grace CPU")
	}),
}

func product(target, name string, base Profile, set func(*Profile)) Profile {
	p := base
	p.Target, p.Product = target, name
	set(&p)
	return p
}
