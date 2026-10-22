# Retained temperature regressions

These are complete, unmodified 129280-entry little-endian float32 target-logit
rows retained from the real G2 capture, steps10,8,5. They are gzip streams with
no timestamp, base64-encoded as text so no binary artifact is staged. Tests
verify the decoded original length and SHA256 before sampling; encoding changes
cannot substitute a synthetic row or truncate the vocabulary.

The owning C release is ds4 revision11e8cad4473716a41baf4d62e92ed9764a00fce6,
GCC13.3 on Linux ARM64 with its release -ffast-math schedule. The corpus was
captured at the same G2 decisions used in the 192-tuple,257-seed,32-row replay.
The temperature1.3/topP1/topK0/minP0 shard144 exposed exactly these three token
mismatches, while C and Go RNG states remained identical. The regression names
identify replay seeds41,69,149 and steps10,8,5. Each test pins its actual before
state, C after state and C token, not a seed inferred from the row filename.

Decoded row hashes:

- step0010: 98663dbbbcda7e5f98db1bc92c656ebabcc15bfc32a54f994581b7f0268cfc05
- step0008: 12e582946fa4b1594a574f6d29aaf893a23386a0a36e99394e77d393e27906f2
- step0005: d91b9134f257825c9ff0ea44e3a7b9f2a98ff775c76b8635ed272884b4645cdd

Actual release C computes one float32 reciprocal and multiplies each subtracted
logit; the old Go implementation divided each subtracted logit. The private
baseline and division-site mutant both reproduce the exact old tokens74797,
5970,121023; the reciprocal candidate produces C tokens74807,5967,121064.
This fixture qualifies that Linux release schedule only, not Darwin, other
libcs/compilers, complete release replay, or physical model inference.
