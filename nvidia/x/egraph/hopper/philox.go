package hopper

// Philox4x32 is the Philox-4x32-10 counter-based generator of Salmon et al.
// (SC'11), the generator behind curand's Philox engine. It maps a 128-bit
// counter and a 64-bit key to four 32-bit words, so a kernel and a host
// reference that agree on the counter draw the same numbers with no state.
func Philox4x32(ctr [4]uint32, key [2]uint32) [4]uint32 {
	const (
		m0, m1 = 0xD2511F53, 0xCD9E8D57
		w0, w1 = 0x9E3779B9, 0xBB67AE85
	)
	c, k := ctr, key
	for i := range 10 {
		if i > 0 {
			k[0] += w0
			k[1] += w1
		}
		p0 := uint64(m0) * uint64(c[0])
		p1 := uint64(m1) * uint64(c[2])
		c = [4]uint32{
			uint32(p1>>32) ^ c[1] ^ k[0],
			uint32(p1),
			uint32(p0>>32) ^ c[3] ^ k[1],
			uint32(p0),
		}
	}
	return c
}

// philoxUniform returns a float32 in [0, 1) with 24 random bits: the top 24
// bits of word, scaled by 2⁻²⁴. Every value is exact in float32.
func philoxUniform(word uint32) float32 {
	return float32(word>>8) * (1.0 / (1 << 24))
}

// philoxOpenUniform returns a float32 in (0, 1), the midpoint of the
// 2⁻²⁴-wide bucket named by the top 24 bits of word, so its logarithm and
// the logarithm of 1 minus it are finite.
func philoxOpenUniform(word uint32) float32 {
	return (float32(word>>8) + 0.5) * (1.0 / (1 << 24))
}

// philoxDevice is the device form of [Philox4x32], [philoxUniform] and
// [philoxOpenUniform]. Multiplying with 64-bit integers instead of
// __umulhi keeps the one source valid for nvcc, NVRTC and the host
// emulator.
const philoxDevice = `#ifndef EG_PHILOX_DEFINED
#define EG_PHILOX_DEFINED 1
// eg_philox4x32 is Philox-4x32-10; it matches hopper.Philox4x32.
EG_DEV void eg_philox4x32(uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3,
                          uint32_t k0, uint32_t k1, uint32_t out[4]) {
#pragma unroll
    for (int i = 0; i < 10; ++i) {
        if (i > 0) {
            k0 += 0x9E3779B9u;
            k1 += 0xBB67AE85u;
        }
        const uint64_t p0 = (uint64_t)0xD2511F53u * c0;
        const uint64_t p1 = (uint64_t)0xCD9E8D57u * c2;
        const uint32_t n0 = (uint32_t)(p1 >> 32) ^ c1 ^ k0;
        const uint32_t n2 = (uint32_t)(p0 >> 32) ^ c3 ^ k1;
        c0 = n0;
        c1 = (uint32_t)p1;
        c2 = n2;
        c3 = (uint32_t)p0;
    }
    out[0] = c0;
    out[1] = c1;
    out[2] = c2;
    out[3] = c3;
}
EG_DEV float eg_philox_uniform(uint32_t w) { return (float)(w >> 8) * (1.0f / 16777216.0f); }
EG_DEV float eg_philox_open_uniform(uint32_t w) { return ((float)(w >> 8) + 0.5f) * (1.0f / 16777216.0f); }
#endif
`
