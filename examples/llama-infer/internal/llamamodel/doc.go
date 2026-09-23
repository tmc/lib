// Package llamamodel resolves a model's architecture through modelir.
//
// [Load] reads a model directory or a .sexpr module set and returns an [Arch]:
// the concrete, per-layer facts that lowering and the reference forward pass
// consume. The module set is the architecture record; Arch is what one
// configuration makes of it. Nothing is guessed: a module set modelir's
// validator refuses, a field reference the configuration cannot satisfy, or
// a dimension left at zero is an error, and modelir's diagnostics stay
// recoverable with errors.As.
//
//	arch, ms, err := llamamodel.Load("models/Llama-3.2-1B-Instruct")
//	if err != nil {
//		return err
//	}
//	fmt.Println(arch.Family, len(arch.Layers), arch.Layers[0].Heads)
//
// Three facts modelir's config frontend does not carry are read from
// config.json when a directory is loaded and recorded in the module set, so
// the .sexpr written from it resolves to the same Arch: tie_word_embeddings
// and max_position_embeddings as configuration fields, and bos_token_id and
// eos_token_id as generation facts when generation_config.json has none.
//
// A directory declares no KV cache: config.json says nothing about caches,
// and the partial declaration modelir v0.2.0's frontend invents (kind and
// family without retention, rollback, update or representation) fails its
// own validator, so Load drops it and Arch.KV is zero. A .sexpr with a
// complete kv-cache form resolves into Arch.KV.
//
// Not resolved, because modelir v0.2.0 does not resolve it: RoPE scaling. A
// configuration's rope_scaling is neither read nor refused here, which
// matches the JSON loader this package replaces.
package llamamodel
