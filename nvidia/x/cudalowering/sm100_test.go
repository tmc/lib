package cudalowering

import (
	"strings"
	"testing"
)

func TestSM100ComparePTXAndSASS(t *testing.T) {
	t.Run("cta_group_1_f16", func(t *testing.T) {
		ptx := `
.version 8.8
.target sm_100a
.address_size 64

.visible .entry gemm_cta1()
{
	tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%r1], 128;
	elect.sync %r0|%p0, 0xffffffff;
	@%p0 tcgen05.mma.cta_group::1.kind::f16 [%tmem0], %descA, %descB, %idesc, %p1;
	@%p0 tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%mbar0];
	tcgen05.wait::ld.sync.aligned;
	tcgen05.ld.sync.aligned.32x32b.x4.b32 {%r4, %r5, %r6, %r7}, [%tmem0];
	tcgen05.dealloc.cta_group::1.sync.aligned.b32 %r1, 128;
	ret;
}
`
		sass := `
	.headerflags @"EF_CUDA_SM100"
	.section .text.gemm_cta1,"ax",@progbits
.text.gemm_cta1:
	/*0000*/                   ELECT P0, RZ, 0xffffffff ;
	/*0010*/               @P0 UTMMA.F16.CTA_GROUP_1 [TMEM:0x0], [SMEM:0x100], [SMEM:0x200] ;
	/*0020*/               @P0 UTMACMDFLUSH.CTA_GROUP_1 [SMEM:0x300] ;
	/*0030*/                   LDTM.128 R4, [TMEM:0x0] ;
	/*0040*/                   EXIT ;
`
		rep, err := CompareSM100PTXAndSASS(ptx, sass)
		if err != nil {
			t.Fatalf("CompareSM100PTXAndSASS error: %v", err)
		}
		if !rep.OK() {
			t.Fatalf("expected OK report, got divergences: %v", rep.Divergences)
		}
		if rep.Err() != nil {
			t.Fatalf("expected nil Err(), got %v", rep.Err())
		}
		if rep.PTX.Alloc != 1 || rep.PTX.Dealloc != 1 || rep.PTX.MMA != 1 || rep.PTX.MMACTAGroup1 != 1 || rep.PTX.Commit != 1 || rep.PTX.Wait != 1 || rep.PTX.Ld != 1 || rep.PTX.Elect != 1 {
			t.Errorf("unexpected PTX summary: %+v", rep.PTX)
		}
		if rep.SASS.UTMMA != 1 || rep.SASS.CTAGroup1 != 1 || rep.SASS.UTMACMDFLUSH != 1 || rep.SASS.LDTM != 1 || rep.SASS.ELECT != 1 {
			t.Errorf("unexpected SASS summary: %+v", rep.SASS)
		}
	})

	t.Run("cta_group_2_mxf4nvf4_block_scale_with_tmem_a_and_clc", func(t *testing.T) {
		ptx := `
.version 8.8
.target sm_100a
.address_size 64

.visible .entry gemm_cta2_mxf4()
{
	tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%r1], 256;
	elect.sync %r0|%p0, 0xffffffff;
	@%p0 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%smem0], [%tensorMap0, {%r2, %r3}], [%mbar0];
	@%p0 tcgen05.cp.cta_group::2.128x256b [%tmem_scale], [%smem_scale];
	tcgen05.st.sync.aligned.32x32b.x4.b32 [%tmem_a], {%r8, %r9, %r10, %r11};
	tcgen05.wait::st.sync.aligned;
	@%p0 tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale [%tmem_d], [%tmem_a], %descB, %idesc, [%tmem_scaleA], [%tmem_scaleB], %p1;
	@%p0 tcgen05.commit.cta_group::2.mbarrier::arrive::one.b64 [%mbar1];
	clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [%smem_clc], [%mbar_clc];
	tcgen05.wait::ld.sync.aligned;
	tcgen05.ld.sync.aligned.32x32b.x8.b32 {%r16, %r17, %r18, %r19, %r20, %r21, %r22, %r23}, [%tmem_d];
	tcgen05.dealloc.cta_group::2.sync.aligned.b32 %r1, 256;
	ret;
}
`
		sass := `
	.headerflags @"EF_CUDA_SM100"
	.section .text.gemm_cta2_mxf4,"ax",@progbits
.text.gemm_cta2_mxf4:
	/*0000*/                   ELECT UP0, URZ, 0xffffffff ;
	/*0010*/              @UP0 UTMALDG.2D.CTA_GROUP_2 [SMEM:0x0], [UR4] ;
	/*0020*/              @UP0 UTMACP.CTA_GROUP_2 [TMEM:0x100], [SMEM:0x400] ;
	/*0030*/                   STTM.64 [TMEM:0x80], R8 ;
	/*0040*/              @UP0 UTMMA.MXF4NVF4.CTA_GROUP_2.BLOCK_SCALE [TMEM:0x0], [TMEM:0x80], [SMEM:0x0], [TMEM:0x100] ;
	/*0050*/              @UP0 UTMACMDFLUSH.CTA_GROUP_2 [SMEM:0x800] ;
	/*0060*/                   LDTM.128 R16, [TMEM:0x0] ;
	/*0070*/                   EXIT ;
`
		rep, err := CompareSM100PTXAndSASS(ptx, sass)
		if err != nil {
			t.Fatalf("CompareSM100PTXAndSASS error: %v", err)
		}
		if !rep.OK() {
			t.Fatalf("expected OK report, got divergences: %v", rep.Divergences)
		}
		if rep.PTX.MMACTAGroup2 != 1 || rep.PTX.MMABlockScaled != 1 || rep.PTX.MMATMEMSourceA != 1 || rep.PTX.Cp != 1 || rep.PTX.St != 1 || rep.PTX.Ld != 1 || rep.PTX.ClusterTryCancel != 1 {
			t.Errorf("unexpected PTX summary: %+v", rep.PTX)
		}
		if rep.SASS.CTAGroup2 != 1 || rep.SASS.BlockScaled != 1 || rep.SASS.TMEMSourceA != 1 || rep.SASS.UTMACP != 1 || rep.SASS.STTM != 1 || rep.SASS.LDTM != 1 || rep.SASS.UTMALDG != 1 {
			t.Errorf("unexpected SASS summary: %+v", rep.SASS)
		}
		for _, c := range rep.Correlations {
			if !c.Matched {
				t.Errorf("unmatched correlation: %+v", c)
			}
		}
	})

	t.Run("negative_missing_utmacmdflush", func(t *testing.T) {
		ptx := `
.version 8.8
.target sm_100a
.address_size 64

.visible .entry bad_commit()
{
	elect.sync %r0|%p0, 0xffffffff;
	@%p0 tcgen05.mma.cta_group::1.kind::f16 [%tmem0], %descA, %descB, %idesc, %p1;
	@%p0 tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%mbar0];
	tcgen05.ld.sync.aligned.32x32b.x4.b32 {%r4, %r5, %r6, %r7}, [%tmem0];
	ret;
}
`
		sass := `
.text.bad_commit:
	/*0000*/                   ELECT P0, RZ, 0xffffffff ;
	/*0010*/               @P0 UTMMA.F16.CTA_GROUP_1 [TMEM:0x0], [SMEM:0x100], [SMEM:0x200] ;
	/*0020*/                   LDTM.64 R4, [TMEM:0x0] ;
	/*0030*/                   EXIT ;
`
		rep, err := CompareSM100PTXAndSASS(ptx, sass)
		if err != nil {
			t.Fatalf("unexpected parse error: %v", err)
		}
		if rep.OK() {
			t.Fatal("expected report to fail for missing UTMACMDFLUSH")
		}
		if !strings.Contains(rep.String(), "UTMACMDFLUSH") {
			t.Errorf("expected divergence mentioning UTMACMDFLUSH, got:\n%s", rep.String())
		}
	})

	t.Run("negative_missing_utmacp_for_block_scaled_utmmma", func(t *testing.T) {
		ptx := `
.version 8.8
.target sm_100a
.address_size 64

.visible .entry bad_block_scale()
{
	elect.sync %r0|%p0, 0xffffffff;
	@%p0 tcgen05.cp.cta_group::1.128x256b [%tmem_scale], [%smem_scale];
	@%p0 tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale [%tmem0], %descA, %descB, %idesc, [%tmem_scaleA], [%tmem_scaleB], %p1;
	@%p0 tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%mbar0];
	tcgen05.ld.sync.aligned.32x32b.x4.b32 {%r4, %r5, %r6, %r7}, [%tmem0];
	ret;
}
`
		sass := `
.text.bad_block_scale:
	/*0000*/                   ELECT UP0, URZ, 0xffffffff ;
	/*0010*/              @UP0 UTMMA.MXF4NVF4.CTA_GROUP_1.BLOCK_SCALE [TMEM:0x0], [SMEM:0x100], [SMEM:0x200] ;
	/*0020*/              @UP0 UTMACMDFLUSH.CTA_GROUP_1 [SMEM:0x300] ;
	/*0030*/                   LDTM.64 R4, [TMEM:0x0] ;
	/*0040*/                   EXIT ;
`
		rep, err := CompareSM100PTXAndSASS(ptx, sass)
		if err != nil {
			t.Fatalf("unexpected parse error: %v", err)
		}
		if rep.OK() {
			t.Fatal("expected report to fail for missing UTMACP")
		}
		if !strings.Contains(rep.String(), "UTMACP") {
			t.Errorf("expected divergence mentioning UTMACP, got:\n%s", rep.String())
		}
	})

	t.Run("negative_unexpected_register_spills", func(t *testing.T) {
		ptx := `
.version 8.8
.target sm_100a
.address_size 64

.visible .entry spill_kernel()
{
	elect.sync %r0|%p0, 0xffffffff;
	@%p0 tcgen05.mma.cta_group::1.kind::bf16 [%tmem0], %descA, %descB, %idesc, %p1;
	@%p0 tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%mbar0];
	tcgen05.ld.sync.aligned.32x32b.x4.b32 {%r4, %r5, %r6, %r7}, [%tmem0];
	ret;
}
`
		sass := `
.text.spill_kernel:
	/*0000*/                   ELECT P0, RZ, 0xffffffff ;
	/*0010*/               @P0 UTMMA.BF16.CTA_GROUP_1 [TMEM:0x0], [SMEM:0x100], [SMEM:0x200] ;
	/*0020*/               @P0 UTMACMDFLUSH.CTA_GROUP_1 [SMEM:0x300] ;
	/*0030*/                   LDTM.64 R4, [TMEM:0x0] ;
	/*0040*/                   STL [R1+0x10], R4 ;
	/*0050*/                   LDL R5, [R1+0x10] ;
	/*0060*/                   EXIT ;
`
		rep, err := CompareSM100PTXAndSASS(ptx, sass)
		if err != nil {
			t.Fatalf("unexpected parse error: %v", err)
		}
		if rep.OK() {
			t.Fatal("expected report to fail when SASS contains unsourced STL/LDL spills")
		}
		if rep.SpillLoads != 1 || rep.SpillStores != 1 {
			t.Errorf("SpillLoads=%d SpillStores=%d, want 1, 1", rep.SpillLoads, rep.SpillStores)
		}
	})
}
