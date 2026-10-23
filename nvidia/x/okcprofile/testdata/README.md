prefill.json is an Open Kernel Challenge judge profile for problem
moe8_prefill_layer, job f3fd094dca984e34 (H200), trimmed with jq to the
fields this package reads: problem, job_id, summary.{speedup,speedup_ci,passed},
and for each benchmark and profile the case's load, weight, layer, batch, ctx
and q_len, the sample arrays, replay_overhead.overhead_ns, event, index and
report. Source sha256 2231a76493ff16722eecc7b65089bb081f5d77b862c038d1a52bcc9243c9be3c.

Its reports have hidden kernel groups, overlapping kernels with a GPU busy
line, and two norm_residual instantiations that truncate to one name.
