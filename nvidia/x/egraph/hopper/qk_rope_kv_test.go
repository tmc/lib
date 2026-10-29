package hopper

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
)

func buildCosSinTable(maxPos, rotaryDim int, interleaved bool, theta float64) []float32 {
	if theta == 0 {
		theta = 10000.0
	}
	half := rotaryDim / 2
	tab := make([]float32, maxPos*rotaryDim)
	for pos := range maxPos {
		row := tab[pos*rotaryDim : (pos+1)*rotaryDim]
		for i := range half {
			freq := 1.0 / math.Pow(theta, float64(2*i)/float64(rotaryDim))
			ang := float64(pos) * freq
			c, s := float32(math.Cos(ang)), float32(math.Sin(ang))
			if interleaved {
				row[2*i] = c
				row[2*i+1] = s
			} else {
				row[i] = c
				row[i+half] = s
			}
		}
	}
	return tab
}

func runQKNormRoPEKVAppendEmu(t *testing.T, k QKNormRoPEKVAppend, m int, in *QKNormRoPEKVInputs) (qOut, kCache, vCache []float64) {
	t.Helper()
	kNorm, err := k.norm()
	if err != nil {
		t.Fatal(err)
	}
	src, err := kNorm.Source()
	if err != nil {
		t.Fatal(err)
	}

	kvDType := kNorm.DType
	if kNorm.KVFP8 {
		kvDType = E4M3
	}

	var cbufs []cudahostemu.Buffer
	var args []string

	cbufs = append(cbufs, cudahostemu.Buffer{
		Name:  "qkv",
		CType: kNorm.DType.cType(),
		Data:  typed{kNorm.DType, in.QKV}.bytes(),
	})
	args = append(args, "qkv")

	if kNorm.QKNorm {
		cbufs = append(cbufs,
			cudahostemu.Buffer{
				Name:  "q_norm_weight",
				CType: kNorm.DType.cType(),
				Data:  typed{kNorm.DType, in.QNormWeight}.bytes(),
			},
			cudahostemu.Buffer{
				Name:  "k_norm_weight",
				CType: kNorm.DType.cType(),
				Data:  typed{kNorm.DType, in.KNormWeight}.bytes(),
			},
		)
		args = append(args, "q_norm_weight", "k_norm_weight")
	}

	cbufs = append(cbufs,
		cudahostemu.Buffer{Name: "cos_sin_table", CType: "float", Data: f32Bytes(in.CosSinTable)},
		cudahostemu.Buffer{Name: "positions", CType: "int32_t", Data: i32Bytes(in.Positions)},
	)
	args = append(args, "cos_sin_table", "positions")

	if in.SeqIDs != nil {
		cbufs = append(cbufs, cudahostemu.Buffer{Name: "seq_ids", CType: "int32_t", Data: i32Bytes(in.SeqIDs)})
		args = append(args, "seq_ids")
	} else {
		args = append(args, "nullptr")
	}

	if in.SeqLens != nil {
		cbufs = append(cbufs, cudahostemu.Buffer{Name: "seqlens", CType: "int32_t", Data: i32Bytes(in.SeqLens)})
		args = append(args, "seqlens")
	} else {
		args = append(args, "nullptr")
	}

	if in.KVPositions != nil {
		cbufs = append(cbufs, cudahostemu.Buffer{Name: "kv_positions", CType: "int32_t", Data: i32Bytes(in.KVPositions)})
		args = append(args, "kv_positions")
	} else {
		args = append(args, "nullptr")
	}

	cbufs = append(cbufs, cudahostemu.Buffer{Name: "page_table", CType: "int32_t", Data: i32Bytes(in.PageTable)})
	args = append(args, "page_table")

	if kNorm.KVFP8 {
		ks, vs := in.KScale, in.VScale
		if ks == 0 {
			ks = 1
		}
		if vs == 0 {
			vs = 1
		}
		cbufs = append(cbufs,
			cudahostemu.Buffer{Name: "k_scale", CType: "float", Data: f32Bytes([]float32{ks})},
			cudahostemu.Buffer{Name: "v_scale", CType: "float", Data: f32Bytes([]float32{vs})},
		)
		args = append(args, "k_scale", "v_scale")
	}

	qOutElems := m * kNorm.NumHeads * kNorm.HeadDim
	cbufs = append(cbufs,
		cudahostemu.Buffer{
			Name:  "q_out",
			CType: kNorm.DType.cType(),
			Data:  poison(qOutElems * kNorm.DType.Bits() / 8),
		},
		cudahostemu.Buffer{
			Name:  "k_cache",
			CType: kvDType.cType(),
			Data:  typed{kvDType, in.KCache}.bytes(),
		},
		cudahostemu.Buffer{
			Name:  "v_cache",
			CType: kvDType.cType(),
			Data:  typed{kvDType, in.VCache}.bytes(),
		},
	)
	args = append(args, "q_out", "k_cache", "v_cache", "nullptr")

	launch := fmt.Sprintf("%s_launch(%d, %s)", kNorm.Name, m, strings.Join(args, ", "))
	res, err := cudahostemu.Run(t.Context(), t.TempDir(), cudahostemu.Program{
		Source:  src,
		Launch:  launch,
		Buffers: cbufs,
		Flags:   strings.Fields(os.Getenv("EG_EMU_FLAGS")),
	})
	if errors.Is(err, cudahostemu.ErrNoCompiler) {
		t.Skip("no host C++ compiler; emulated kernel not run")
	}
	if err != nil {
		t.Fatalf("emulated kernel: %v\n--- source ---\n%s", err, numbered(src))
	}

	return decode(kNorm.DType, res["q_out"]), decode(kvDType, res["k_cache"]), decode(kvDType, res["v_cache"])
}

func TestQKNormRoPEKVAppend(t *testing.T) {
	// 20/10 heads spread over three CTAs per token, the last with fewer
	// than four heads.
	for _, heads := range [][2]int{{4, 2}, {20, 10}} {
		NumHeads, KVHeads := heads[0], heads[1]
		t.Run(fmt.Sprintf("bf16_qknorm_neox_bf16kv_%dh", NumHeads), func(t *testing.T) {
			const (
				M        = 4
				HeadDim  = 64
				PageSize = 16
				MaxPages = 4
			)
			k := QKNormRoPEKVAppend{
				Name:            "qk_rope_kv_bf16_neox",
				NumHeads:        NumHeads,
				KVHeads:         KVHeads,
				HeadDim:         HeadDim,
				RotaryDim:       HeadDim,
				PageSize:        PageSize,
				MaxPages:        MaxPages,
				Eps:             1e-5,
				QKNorm:          true,
				InterleavedRoPE: false,
				KVFP8:           false,
				DType:           BF16,
			}

			r := rand.New(rand.NewPCG(11, 22))
			qkvStride := (NumHeads + 2*KVHeads) * HeadDim
			totalPages := M * MaxPages
			cacheElems := totalPages * PageSize * KVHeads * HeadDim

			pageTable := make([]int32, M*MaxPages)
			perm := r.Perm(totalPages)
			for i := range pageTable {
				pageTable[i] = int32(perm[i])
			}
			seqLens := []int32{1, 17, 32, 45}
			positions := []int32{0, 16, 31, 44}

			initK := randVals(r, cacheElems, 0.25)
			initV := randVals(r, cacheElems, 0.25)
			for i := range cacheElems {
				initK[i] = float64(BF16.Round(float32(initK[i])))
				initV[i] = float64(BF16.Round(float32(initV[i])))
			}

			inRef := &QKNormRoPEKVInputs{
				QKV:         randVals(r, M*qkvStride, 1.2),
				QNormWeight: randVals(r, HeadDim, 0.8),
				KNormWeight: randVals(r, HeadDim, 0.8),
				CosSinTable: buildCosSinTable(64, HeadDim, false, 10000.0),
				Positions:   positions,
				SeqIDs:      []int32{0, 1, 2, 3},
				SeqLens:     seqLens,
				PageTable:   pageTable,
				KCache:      append([]float64(nil), initK...),
				VCache:      append([]float64(nil), initV...),
			}
			inEmu := *inRef
			inEmu.KCache = append([]float64(nil), initK...)
			inEmu.VCache = append([]float64(nil), initV...)

			wantQ, err := k.Reference(M, inRef)
			if err != nil {
				t.Fatal(err)
			}
			gotQ, gotK, gotV := runQKNormRoPEKVAppendEmu(t, k, M, &inEmu)

			agree(t, "q_out", BF16, gotQ, wantQ)
			agree(t, "k_cache", BF16, gotK, inRef.KCache)
			agree(t, "v_cache", BF16, gotV, inRef.VCache)
		})
	}

	t.Run("bf16_qknorm_interleaved_partial_rotary_fp8kv", func(t *testing.T) {
		const (
			M         = 4
			NumHeads  = 4
			KVHeads   = 2
			HeadDim   = 64
			RotaryDim = 32
			PageSize  = 16
			MaxPages  = 4
		)
		k := QKNormRoPEKVAppend{
			Name:            "qk_rope_kv_fp8_interleaved",
			NumHeads:        NumHeads,
			KVHeads:         KVHeads,
			HeadDim:         HeadDim,
			RotaryDim:       RotaryDim,
			PageSize:        PageSize,
			MaxPages:        MaxPages,
			Eps:             1e-5,
			QKNorm:          true,
			InterleavedRoPE: true,
			KVFP8:           true,
			DType:           BF16,
		}

		r := rand.New(rand.NewPCG(33, 44))
		qkvStride := (NumHeads + 2*KVHeads) * HeadDim
		batch := 2
		totalPages := batch * MaxPages
		cacheElems := totalPages * PageSize * KVHeads * HeadDim

		pageTable := make([]int32, batch*MaxPages)
		perm := r.Perm(totalPages)
		for i := range pageTable {
			pageTable[i] = int32(perm[i])
		}

		initK := make([]float64, cacheElems)
		initV := make([]float64, cacheElems)
		for i := range cacheElems {
			initK[i] = float64(FromE4M3Bits(E4M3Bits(float32(r.NormFloat64() * 0.25))))
			initV[i] = float64(FromE4M3Bits(E4M3Bits(float32(r.NormFloat64() * 0.25))))
		}

		// Two tokens per sequence using explicit SeqIDs and KVPositions.
		inRef := &QKNormRoPEKVInputs{
			QKV:         randVals(r, M*qkvStride, 1.0),
			QNormWeight: randVals(r, HeadDim, 0.9),
			KNormWeight: randVals(r, HeadDim, 0.9),
			CosSinTable: buildCosSinTable(64, RotaryDim, true, 500000.0),
			Positions:   []int32{10, 11, 25, 26},
			SeqIDs:      []int32{0, 0, 1, 1},
			KVPositions: []int32{10, 11, 25, 26},
			PageTable:   pageTable,
			KScale:      0.5,
			VScale:      0.25,
			KCache:      append([]float64(nil), initK...),
			VCache:      append([]float64(nil), initV...),
		}
		inEmu := *inRef
		inEmu.KCache = append([]float64(nil), initK...)
		inEmu.VCache = append([]float64(nil), initV...)

		wantQ, err := k.Reference(M, inRef)
		if err != nil {
			t.Fatal(err)
		}
		gotQ, gotK, gotV := runQKNormRoPEKVAppendEmu(t, k, M, &inEmu)

		agree(t, "q_out", BF16, gotQ, wantQ)
		agree(t, "k_cache", E4M3, gotK, inRef.KCache)
		agree(t, "v_cache", E4M3, gotV, inRef.VCache)
	})

	t.Run("chain_qk_rope_kv_to_paged_attention_fp8", func(t *testing.T) {
		// End-to-end chain test:
		// 1. QKNormRoPEKVAppend populates q_out (BF16) and appends the newest
		//    token's K and V (quantized to FP8 E4M3) into k_cache and v_cache;
		// 2. PagedAttention (and MTPPagedAttention{KVFP8: true}) immediately
		//    reads that exact q_out, k_cache, and v_cache in a single emulated
		//    program launch, matching the chained Go reference!
		const (
			batch    = 2
			NumHeads = 4
			KVHeads  = 2
			HeadDim  = 64
			PageSize = 16
			MaxPages = 3
			kScale   = 0.5
			vScale   = 0.25
		)
		qk := QKNormRoPEKVAppend{
			Name:            "chain_qk_rope_kv",
			NumHeads:        NumHeads,
			KVHeads:         KVHeads,
			HeadDim:         HeadDim,
			RotaryDim:       HeadDim,
			PageSize:        PageSize,
			MaxPages:        MaxPages,
			Eps:             1e-5,
			QKNorm:          true,
			InterleavedRoPE: false,
			KVFP8:           true,
			DType:           BF16,
		}
		pa := PagedAttention{
			Name:  "chain_paged_attn",
			Hq:    NumHeads,
			Hkv:   KVHeads,
			QLen:  1,
			Page:  PageSize,
			Q:     BF16,
			Out:   F32,
			Chunk: 64,
		}
		mtp := MTPPagedAttention{
			Name:          "chain_mtp_paged_attn",
			NumHeads:      NumHeads,
			KVHeads:       KVHeads,
			HeadDim:       HeadDim,
			PageSize:      PageSize,
			MaxPages:      MaxPages,
			QTokensPerSeq: 1,
			KVFP8:         true,
		}

		r := rand.New(rand.NewPCG(77, 88))
		seqLens := []int32{19, 35}
		maxKeys := 35
		positions := []int32{seqLens[0] - 1, seqLens[1] - 1}

		// Build initial scattered FP8 KV cache for prefix tokens 0..seqlen-2;
		// poison the last token slot (seqlen-1) with NaN (0x7f) so PagedAttention
		// would fail if QKNormRoPEKVAppend did not overwrite that exact slot.
		_, cache := pagedCase(r, pa, seqLens, kScale)
		cache.VScale = vScale
		for b, s := range seqLens {
			lastTok := int(s) - 1
			page := int(cache.PageTable[b*MaxPages+lastTok/PageSize])
			slot := page*PageSize + lastTok%PageSize
			for i := range KVHeads * HeadDim {
				cache.K[slot*KVHeads*HeadDim+i] = 0x7f
				cache.V[slot*KVHeads*HeadDim+i] = 0x7f
			}
		}

		qkvStride := (NumHeads + 2*KVHeads) * HeadDim
		qkv := randVals(r, batch*qkvStride, 0.8)
		qNormW := randVals(r, HeadDim, 0.9)
		kNormW := randVals(r, HeadDim, 0.9)
		cosSin := buildCosSinTable(64, HeadDim, false, 10000.0)

		kCacheF64 := decode(E4M3, cache.K)
		vCacheF64 := decode(E4M3, cache.V)
		inRef := &QKNormRoPEKVInputs{
			QKV:         qkv,
			QNormWeight: qNormW,
			KNormWeight: kNormW,
			CosSinTable: cosSin,
			Positions:   positions,
			SeqLens:     seqLens,
			PageTable:   cache.PageTable,
			KScale:      kScale,
			VScale:      vScale,
			KCache:      kCacheF64,
			VCache:      vCacheF64,
		}
		wantQ64, err := qk.Reference(batch, inRef)
		if err != nil {
			t.Fatal(err)
		}

		// Encode updated KCache and VCache back to E4M3 bytes for PagedAttention.Reference.
		refCache := cache
		refCache.K = typed{E4M3, inRef.KCache}.bytes()
		refCache.V = typed{E4M3, inRef.VCache}.bytes()
		wantQ32 := make([]float32, len(wantQ64))
		wantQBF16 := make([]uint16, len(wantQ64))
		for i, v := range wantQ64 {
			wantQ32[i] = float32(v)
			wantQBF16[i] = BF16Bits(float32(v))
		}
		wantAttn, err := pa.Reference(wantQ32, refCache)
		if err != nil {
			t.Fatal(err)
		}
		wantMTP, err := mtp.Reference(wantQBF16, MTPPagedCache{
			KFP8:      refCache.K,
			VFP8:      refCache.V,
			PageTable: refCache.PageTable,
			SeqLens:   seqLens,
			KScale:    kScale,
			VScale:    vScale,
		})
		if err != nil {
			t.Fatal(err)
		}

		// Compile and launch QKNormRoPEKVAppend -> PagedAttention -> MTPPagedAttention{KVFP8: true} in one CU!
		qkCUDA, err := qk.CUDA()
		if err != nil {
			t.Fatal(err)
		}
		qkLaunch, err := qk.Launcher()
		if err != nil {
			t.Fatal(err)
		}
		paCUDA, err := pa.CUDA()
		if err != nil {
			t.Fatal(err)
		}
		paLaunch, err := pa.Launcher()
		if err != nil {
			t.Fatal(err)
		}
		mtpCUDA, err := mtp.CUDA()
		if err != nil {
			t.Fatal(err)
		}
		mtpLaunch, err := mtp.Launcher()
		if err != nil {
			t.Fatal(err)
		}

		combinedSrc := strings.Join([]string{qkCUDA, qkLaunch, paCUDA, paLaunch, mtpCUDA, mtpLaunch}, "\n")
		launchStmt := fmt.Sprintf(
			"chain_qk_rope_kv_launch(%d, qkv, q_norm_w, k_norm_w, cos_sin, positions, nullptr, seqlens, nullptr, page_table, k_scale, v_scale, q_out, k_cache, v_cache, nullptr);\n"+
				"    chain_paged_attn_launch(%d, %d, q_out, k_cache, v_cache, page_table, %d, seqlens, k_scale, v_scale, attn_out, ws, nullptr);\n"+
				"    chain_mtp_paged_attn_launch(%d, q_out, k_cache, v_cache, page_table, seqlens, nullptr, k_scale, v_scale, mtp_out, nullptr)",
			batch, batch, maxKeys, MaxPages, batch,
		)

		qOutElems := batch * NumHeads * HeadDim
		res := runEmu(t, combinedSrc, launchStmt,
			emuBuf{"qkv", "uint16_t", typed{BF16, qkv}.bytes()},
			emuBuf{"q_norm_w", "uint16_t", typed{BF16, qNormW}.bytes()},
			emuBuf{"k_norm_w", "uint16_t", typed{BF16, kNormW}.bytes()},
			emuBuf{"cos_sin", "float", f32Bytes(cosSin)},
			emuBuf{"positions", "int32_t", i32Bytes(positions)},
			emuBuf{"seqlens", "int32_t", i32Bytes(seqLens)},
			emuBuf{"page_table", "int32_t", i32Bytes(cache.PageTable)},
			emuBuf{"k_scale", "float", f32Bytes([]float32{kScale})},
			emuBuf{"v_scale", "float", f32Bytes([]float32{vScale})},
			emuBuf{"q_out", "uint16_t", poison(2 * qOutElems)},
			emuBuf{"k_cache", "uint8_t", append([]byte(nil), cache.K...)},
			emuBuf{"v_cache", "uint8_t", append([]byte(nil), cache.V...)},
			emuBuf{"attn_out", "float", poison(4 * qOutElems)},
			emuBuf{"ws", "unsigned char", attnWorkspace(pa, batch, maxKeys)},
			emuBuf{"mtp_out", "uint16_t", poison(2 * qOutElems)},
		)

		gotQ := decode(BF16, res["q_out"])
		agree(t, "chain.q_out", BF16, gotQ, wantQ64)

		gotAttn := decode(F32, res["attn_out"])
		if errRel := relErr(gotAttn, wantAttn); errRel > 1e-3 || math.IsNaN(errRel) {
			t.Fatalf("PagedAttention chain relative L2 error %.4e > 1e-3", errRel)
		} else {
			t.Logf("PagedAttention chain relative L2 error %.4e", errRel)
		}

		gotMTP := decode(BF16, res["mtp_out"])
		if errRel := relErr(gotMTP, wantMTP); errRel > 2e-3 || math.IsNaN(errRel) {
			t.Fatalf("MTPPagedAttention chain relative L2 error %.4e > 2e-3", errRel)
		} else {
			t.Logf("MTPPagedAttention chain relative L2 error %.4e", errRel)
		}
	})

	t.Run("validation_errors", func(t *testing.T) {
		bad := []QKNormRoPEKVAppend{
			{Name: "1bad", NumHeads: 4, KVHeads: 2, HeadDim: 64, PageSize: 16, MaxPages: 4},
			{Name: "bad_heads", NumHeads: 5, KVHeads: 2, HeadDim: 64, PageSize: 16, MaxPages: 4},
			{Name: "bad_dim", NumHeads: 4, KVHeads: 2, HeadDim: 63, PageSize: 16, MaxPages: 4},
			{Name: "bad_rotary", NumHeads: 4, KVHeads: 2, HeadDim: 64, RotaryDim: 128, PageSize: 16, MaxPages: 4},
			{Name: "bad_page", NumHeads: 4, KVHeads: 2, HeadDim: 64, PageSize: 0, MaxPages: 4},
			{Name: "bad_dtype", NumHeads: 4, KVHeads: 2, HeadDim: 64, PageSize: 16, MaxPages: 4, DType: E4M3},
		}
		for _, b := range bad {
			if _, err := b.Source(); err == nil {
				t.Errorf("expected validation error for %+v, got nil", b)
			}
		}
	})
}
