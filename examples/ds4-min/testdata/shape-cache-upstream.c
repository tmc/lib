static uint32_t ds4_expected_layer_compress_ratio(uint32_t il) {
    if (il >= DS4_N_LAYER) ds4_die("DeepSeek4 layer index is outside the loaded model layout");

    switch (DS4_MODEL_VARIANT) {
    case DS4_VARIANT_FLASH:
        if (il < 2) return 0;
        return (il & 1u) == 0 ? 4u : 128u;
    case DS4_VARIANT_PRO:
        if (il < 2) return 128u;
        return (il & 1u) == 0 ? 4u : 128u;
    case DS4_VARIANT_FLASH41:
        return il < 2 ? 0u : il < 20 ? 2u : 1u;
    default:
        ds4_die("unsupported DeepSeek4 model variant");
    }
    return 0;
}
static void kv_cache_init(ds4_kv_cache *cache, uint32_t ctx_size, uint32_t raw_cap) {
    memset(cache, 0, sizeof(*cache));
    if (raw_cap == 0) raw_cap = ds4_default_raw_cap(ctx_size);
    if (raw_cap > ctx_size) raw_cap = ctx_size;
    if (raw_cap == 0) raw_cap = 1;

    cache->head_dim = DS4_N_HEAD_DIM;

    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        cache->layer[il].cap_raw = raw_cap;
        cache->layer[il].raw_kv = xmalloc_zeroed((size_t)raw_cap * DS4_N_HEAD_DIM, sizeof(float));
        cache->layer[il].compress_ratio = ratio;

        if (ratio != 0) {
            const uint32_t coff = ratio == 4 ? 2u : 1u;
            const uint32_t comp_cap = ctx_size / ratio + 2;
            const uint32_t attn_width = coff * DS4_N_HEAD_DIM;
            const uint32_t attn_rows = coff * ratio;

            cache->layer[il].comp_cap = comp_cap;
            cache->layer[il].attn_comp_kv = xmalloc_zeroed((size_t)comp_cap * DS4_N_HEAD_DIM, sizeof(float));
            cache->layer[il].attn_state_kv = xmalloc_zeroed((size_t)attn_width * attn_rows, sizeof(float));
            cache->layer[il].attn_state_score = xmalloc((size_t)attn_width * attn_rows * sizeof(float));
            for (uint64_t i = 0; i < (uint64_t)attn_width * attn_rows; i++) {
                cache->layer[il].attn_state_score[i] = DS4_NEG_INF;
            }

            if (ratio == 4) {
                const uint32_t index_width = coff * DS4_N_INDEXER_HEAD_DIM;
                const uint32_t index_rows = coff * ratio;
                cache->layer[il].index_comp_kv = xmalloc_zeroed((size_t)comp_cap * DS4_N_INDEXER_HEAD_DIM, sizeof(float));
                cache->layer[il].index_state_kv = xmalloc_zeroed((size_t)index_width * index_rows, sizeof(float));
                cache->layer[il].index_state_score = xmalloc((size_t)index_width * index_rows * sizeof(float));
                for (uint64_t i = 0; i < (uint64_t)index_width * index_rows; i++) {
                    cache->layer[il].index_state_score[i] = DS4_NEG_INF;
                }
            }
        }
    }
}
