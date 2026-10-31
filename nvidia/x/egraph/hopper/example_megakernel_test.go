package hopper_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

func exampleDecoderLayerPlans(layerIdx int) hopper.DecoderLayerPlans {
	const (
		m   = 1
		h   = 256
		eps = 1e-5
	)
	bAttn := hopper.NewBuilder()
	xn := bAttn.Input("xn", hopper.BF16, m, h)
	wqkv := bAttn.Input("w_qkv", hopper.BF16, h, h)
	attnExpr, err := hopper.Optimize(bAttn.G, bAttn.MatMul(xn, wqkv))
	if err != nil {
		panic(err)
	}
	attnIn, err := hopper.Compile(attnExpr, fmt.Sprintf("l%d_attn_in", layerIdx), hopper.F32)
	if err != nil {
		panic(err)
	}

	bPost := hopper.NewBuilder()
	attn := bPost.Input("attn", hopper.BF16, m, h)
	x := bPost.Input("x", hopper.F32, m, h)
	wo := bPost.Input("w_o", hopper.BF16, h, h)
	hid := bPost.Map(hopper.MustLambda("(add $0 $1)"), bPost.MatMul(attn, wo), x)
	nw := bPost.Input("next_norm", hopper.F32, h)
	xnOut := bPost.Cast(hopper.BF16, bPost.RMSNorm(hid, nw, h, eps))
	postExpr, err := hopper.Optimize(bPost.G, xnOut)
	if err != nil {
		panic(err)
	}
	post, err := hopper.Compile(postExpr, fmt.Sprintf("l%d_post", layerIdx), hopper.BF16)
	if err != nil {
		panic(err)
	}

	return hopper.DecoderLayerPlans{
		LayerIndex: layerIdx,
		AttnIn:     attnIn,
		Post:       post,
	}
}

func ExampleFuseDecoderLayerStack() {
	bHead := hopper.NewBuilder()
	xn := bHead.Input("xn", hopper.BF16, 1, 256)
	wHead := bHead.Input("w_head", hopper.BF16, 256, 256)
	headExpr, err := hopper.Optimize(bHead.G, bHead.MatMul(xn, wHead))
	if err != nil {
		fmt.Println(err)
		return
	}
	head, err := hopper.Compile(headExpr, "lm_head", hopper.F32)
	if err != nil {
		fmt.Println(err)
		return
	}

	layers := []hopper.DecoderLayerPlans{
		exampleDecoderLayerPlans(0),
		exampleDecoderLayerPlans(1),
	}
	stack, err := hopper.FuseDecoderLayerStack(layers, hopper.CrossLayerStackConfig{
		NamePrefix:         "cross",
		BridgeInput:        "xn",
		RetainBridgeOutput: true,
		Head:               head,
		FuseFinalHead:      true,
		PackArena:          true,
		Mega:               true,
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("unfused=%d fused=%d cross_megas=%d final_mega=%v arena_bytes=%d\n",
		stack.UnfusedLaunchCount(),
		stack.FusedLaunchCount(),
		len(stack.CrossMegakernels),
		stack.FinalMegakernel != nil,
		stack.ArenaBytes)
	// Output: unfused=5 fused=3 cross_megas=1 final_mega=true arena_bytes=1536
}

func ExampleBuildFullLayerMegakernel() {
	specs := []hopper.FullDecoderLayerSpec{
		{
			LayerIndex: 0,
			BatchM:     1,
			HiddenDim:  256,
			IntermDim:  512,
			NumQHeads:  4,
			NumKVHeads: 2,
			HeadDim:    64,
			PageSize:   16,
			MaxPages:   4,
			Arch:       hopper.TargetSM121A,
		},
		{
			LayerIndex: 1,
			BatchM:     1,
			HiddenDim:  256,
			IntermDim:  512,
			NumQHeads:  4,
			NumKVHeads: 2,
			HeadDim:    64,
			PageSize:   16,
			MaxPages:   4,
			Arch:       hopper.TargetSM121A,
		},
	}
	_, mega, err := hopper.BuildFullLayerMegakernel("llama_2l_mega", specs)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("name=%s stages=%d s2=%s s3=%s ready_words=%d\n",
		mega.Name, len(mega.Stages), mega.Stages[2].Name, mega.Stages[3].Name, mega.ReadyWords())
	// Output: name=llama_2l_mega stages=18 s2=l0_rope_kv s3=l0_paged_attn ready_words=18
}

func ExampleBuildWarpSpecializedMegakernel() {
	_, mega, err := hopper.BuildFullLayerMegakernel("full_ws_mega", []hopper.FullDecoderLayerSpec{
		{
			LayerIndex: 0,
			BatchM:     1,
			HiddenDim:  256,
			IntermDim:  512,
			NumQHeads:  4,
			NumKVHeads: 2,
			HeadDim:    64,
			PageSize:   16,
			MaxPages:   4,
			Arch:       hopper.TargetSM121A,
		},
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	wm, err := hopper.BuildWarpSpecializedMegakernel(mega, hopper.DefaultWarpSpecializedConfig())
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("prod=%dw*%dregs cons=%dw*%dregs ring=%d pdl=%v hints=%d\n",
		wm.Config.ProducerWarps, wm.Config.ProducerMaxRegs,
		wm.Config.ConsumerWarps, wm.Config.ConsumerMaxRegs,
		wm.Config.RingBufferStages, wm.Config.EnableGridDepControl,
		len(wm.StageHints))
	// Output: prod=2w*24regs cons=6w*232regs ring=3 pdl=true hints=9
}

func ExamplePlanL2Persistence() {
	l0 := exampleDecoderLayerPlans(0)
	l0.AttnIn.Inputs = append(l0.AttnIn.Inputs,
		hopper.Tensor{Name: "kv_prefix_pages", DType: hopper.BF16, Shape: []int{16, 64}},
	)
	l1 := exampleDecoderLayerPlans(1)
	stack, err := hopper.FuseDecoderLayerStack([]hopper.DecoderLayerPlans{l0, l1}, hopper.CrossLayerStackConfig{
		PackArena: true,
		Mega:      true,
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	lp, err := hopper.PlanL2Persistence(stack, 50*1024*1024)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("arena=%s ready=%s kv_prefix=%s w_qkv=%s\n",
		lp.ByName["eg_arena_even"].Policy,
		lp.ByName["ready"].Policy,
		lp.ByName["kv_prefix_pages"].Policy,
		lp.ByName["w_qkv"].Policy)
	// Output: arena=evict_last ready=evict_last kv_prefix=evict_last w_qkv=evict_first
}
