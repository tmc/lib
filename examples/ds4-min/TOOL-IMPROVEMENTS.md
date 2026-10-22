# Tool improvements the campaign paid for

Every entry here was bought with a wrong number, a lost run, or an hour. The
raw log is `cudaprof-errata-bugs-and-gaps.txt` (git-excluded, 6307 lines:
prose sections 1-20 on lines 1-1302, the numbered log 21.1-21.209 after). This
file is the actionable extract, keyed to those entry numbers so a fix traces
back to the observation that motivated it.

Coverage: an exhaustive read of all 6307 lines yields **74 qualifying
tool-defect entries**, 39 of them cudaprof. A mechanical keyword scan returns
65 candidate entries and is the cheap way to re-derive the set:

    python3 - <<'PY'
    import re, io
    s = io.open('cudaprof-errata-bugs-and-gaps.txt', errors='replace').read().split('\n')
    idx = [i for i, l in enumerate(s) if re.match(r'^\d+\.\d+(\s|$)', l)] + [len(s)]
    kw = ('cudaprof', 'cupti', 'journal', 'nsys', 'ncu', 'gpu-lock',
          'capture_journal', 'nvrtc', 'pcsampl', 'ptxas')
    for a, b in zip(idx, idx[1:]):
        if any(k in '\n'.join(s[a:b]).lower() for k in kw):
            print(s[a])
    PY

The scan misses the lettered entries in the prose sections (Bug 1-3, Gap 1-4,
B1-B7) and picks up entries that merely *run* under a tool. The triage below
is a reading; the counts above are not.

Status vocabulary: **open** -- nothing landed. **worked around** -- we changed
how we use the tool, the tool is unchanged. **fixed** -- landed, and this file
names the change.

## 1. cudaprof reports a number the capture cannot support

Nineteen entries share this shape, and it is the theme that cost the campaign
the most time. In every one of them the failing instrument produced a
well-formed artifact that read as a clean negative result.

| # | failure | fix request | status |
|---|---|---|---|
| 21.119 | `run_completeness: COMPLETE` on a capture with **zero** activity records -- `capture_error None`, all three lifecycle booleans true, a 3440-byte pprof on disk. Complete means nothing was dropped; nothing was dropped because nothing was produced. Only `outcome` told the truth. | Make `run_completeness` refuse to read COMPLETE at zero records, or rename it `nothing_dropped`. A gate that cannot fail on an empty population is not a gate. | **fixed**, `877b31b0` -- and **not by the remedy this row leads with**, correctly. F76EF7EB declined to overturn the classification: the field answers *was anything dropped*, and at zero records it answers correctly. The scope is written down beside the code (`unqualifiedGenerations`: *"a finalized generation that retained no records is qualified, because a zero count is not evidence of lost work"*), so the count, not the classification, was the thing missing. The capture headline now says under the complete branch that the file holds **NO activity records** and that no ranking, census or duty cycle from it covers any part of the run -- beside the existing qualifier for a journal over budget, the identical shape of healthy counts over an unusable artifact. `run_completeness` left alone because other tools parse the token. This is my own general statement applied back to me: the honest discriminator is a denominator printed beside the answer. |
| 7.2 | `outcome transport-failed`, `bootstrap_seen false`, 0 records, and a syntactically valid 8512-byte pprof. **My description of the symptom was stale and wrong.** I recorded a clean empty table from `-top`; by the time the row was worked, `preflightCoverage` already refused a zero-sample profile. The live defect was one step subtler -- it **misdiagnosed** the refusal: the coverage line read *"the profile carries no device time, only sample counts"*, false when there are no samples, and prescribed *"capture activity tracing alongside PC sampling"*. That sends someone with an empty file to pay a run under GPU lock and get the same empty file back. | An empty ranking must say the capture was empty. | **fixed**, `08e149b6`. The empty case is ordered ahead of the device-time case and named: the refusal says the file has nothing to rank, not that it lacks a denominator, and denies the workload reading in words. Deliberate behaviour change: `-allow_unknown_coverage` no longer reaches the empty case -- the flag exists so a reader who knows what a sample share is can read one, and there is no share of zero samples; allowing it printed the very table the refusal exists to prevent. A sample-only profile stays bypassable, with a control asserting it. **Third instance of my stale-row shape**, and the first where the *failure column*, not the status, was the stale part: the standing rule now covers both -- re-read the tree before citing a row, including what it claims the tool does. |
| 8.1(b) | Instrument **false negative**: the controller declined an unfinalized *target* journal and reported everything in it as absent, while `capture.journal` held a bootstrap record and `handshake.json` said `transport_ready: true`. The worker branch handled `errJournalNotFinalized`; the target branch discarded. | Report what was read. | **fixed** in `capture_controller.go`, with `TestUnfinalizedTargetStillReportsWhatItRead` and a mutation control |
| 21.111 | After `e8d1a6c8`, the status record reads `transport_ready TRUE / collector_finalized FALSE / run_completeness FAILED` -- which says what happened -- while `outcome` still reads `transport-failed`. It is now the only field contradicting the other three. | Rename the token. F76E declines to rename a published one unilaterally, which is the right instinct and also why it is still wrong. | **open** (naming) |
| 21.151 | `advise` and `-top` announce the absence of data that `gaps` decodes **from the same file** by default: item 15, `evidence_level "unmeasured"`, recommending a re-capture. Their total (464.685 ms) is also less than `gaps`' union (753.775 ms), impossible for one record set. | Verbatim: *"advise should read the journal when one is attached, or -- if that is a bigger change than it wants -- at minimum stop recommending a re-capture when the journal is present, and say instead that duty cycle is available from `cudaprof gaps` on this same file."* | **half fixed, and I had it wrong.** The journal half is **fixed and was already in the tree while this row said open** -- `advise.go:4074 recaptureOrRead` returns *"Read the journal already in this capture rather than re-running the workload: cudaprof gaps <capture> reports the duty cycle, the idle attribution and the gap distribution from it"* when `captureHasJournal`, and `advise_test.go:3859/3871` asserts both directions. Verified before this rewrite. The **464.685 against 753.775 disagreement is still live** and unverified by either side: two commands reporting incompatible totals from one file. |
| 21.155 | `gaps --diff_base` spans the whole capture while the prefill t/s metric divides by a window, so a change that merely relocated 24 repack launches out of the prefill window read as a speedup -- total repack time actually **+89.934 ms**. Both outputs correct; neither flags the boundary. | Verbatim: *"when a whole-capture diff's two arms disagree about WHERE work happened rather than whether it happened, say so above the table."* | **fixed**, `8670d6c3`. The census fires a caveat **above** the table when kernels present in both arms moved through the run, naming the kernel that moved furthest. Mutation-tested in a detached worktree under `GOWORK=off`, four probes, all killed -- including moving the caveat *below* the sections, which is the erratum's actual claim. The three-class split already carried the sentence in a section heading, and a heading is reached after the totals have been read; an unconditional preamble was rejected for the same reason an unconditional gate is useless -- it prints on every report, so it says nothing about the pair in front of you. |

`advise` already implements the honest behaviour this section is asking for,
twice: it carries an `evidence_level` field (21.151), and at 21.67 it
**refused** to rank a PC-sampling profile -- *"a sample count has no duration:
it says where execution was, never for how long"* -- offering
`-allow_unknown_coverage` to read it as a sample share "and nothing more".
That refusal caught an error I was about to make. The request for the whole
section is to make that the default rather than the exception.

## 2. cudaprof attribution

Fourteen entries. This cluster contains both the deepest defect in the file
and its fix.

| # | failure | fix request | status |
|---|---|---|---|
| 21.145 + 21.149 | **The largest single misreport in the campaign.** `gaps` excluded `graph_trace` from the device timeline on the premise that the envelope spans its nodes' kernel records -- but **zero of 9711** kernel records carried a nonzero `graph_id`. CUPTI emitted the envelope and nothing inside it, so a graph's entire execution was invisible and landed in idle: reported duty **14.54%** against a true **71.32%**, with 27 kernel names and 2640.544 ms (78.5% of prefill device work) absent from every ranking built on that capture. | Promote the envelope to busy. The capture carries the discriminator: a timeline with zero graph-tagged kernels is "contents never recorded", not "idle inside a graph". | **fixed** by F76E in `c40b1884`, and validated from **outside** by a graphs-off arm -- envelope arithmetic 72.24% against resolved-records 70.83%, right to 1.4 points. That is the positive control the fix never had. |
| 21.135 / 21.141 | `cupti_agent.c:903` enables CONCURRENT_KERNEL/KERNEL, MEMCPY, EXTERNAL_CORRELATION, MARKER, RUNTIME, DRIVER -- and **no** GRAPH_TRACE or graph-node kind. 21.141 argued from source that node records must exist; 21.145 refuted it by counting the field. *A capability present in the producer is not a value present in the data.* | Enable a graph-node activity kind, so the fix above stops being arithmetic over an opaque envelope. | **open** (capability gap) |
| 21.143 / 21.148 | `gaps.go:688-692` added Unmatched gaps wholesale to `Attribution.HardwareNanos` -- "we could not find the launch" printed as a named mechanism, 98.9% of the bucket. Root cause at `:463`: the filter excluded marker and capture_range but not OVERHEAD. | Give Unmatched its own line; never merge a residual class into a named one. | **fixed** (`13d746d2`) -- but 21.148 **withdrew** my report of it: it was produced by a stale dirty build, against a fix that had landed in `86bad446` a day earlier, with line numbers cited from my worktree rather than main. At HEAD these captures show zero Unmatched. |
| B4 | `advise` computes def-use distance in a **different address space** than the stall it explains, and reports it as constant across unrelated kernels -- an identical "R0 loaded at PC 0x0170 ... consumed 8 instructions later" for two kernels 0.6 MB apart. Def/use PCs are kernel-relative; stall PCs are image-absolute. | The requested control: two kernels with deliberately different def-use distances must report different distances. | **open** |
| 21.124 | An `advise` finding names a cause **that does not exist in the source**: SM count never enters the code path it blames. | | **open** |
| 21.113 | `advise` ranks by device time only, so its top Go item was a 2-launch single-thread 0.01 ms kernel flagged Critical, and 62 of 66 findings were "grid leaves SMs idle" boilerplate -- while the real finding sat in the same capture: 8192 consecutive `ds4_get_row_f16` launches, and stream 13 carrying 32814 micro-kernels, 47% of all launches for 3.5% of the time. | Rank by launch count as well as duration; collapse the boilerplate finding to one row with a count. | **open**, worked around -- *treat advise as a launch-geometry census, not a ranking* |
| 21.42 | Three defects in one. (i) The NVTX label **cannot fire** on the capture path: `BuildActivityProfile` pairs NVTX from marker records (`activity.go:259`) but `capture_controller.go:566` drops marker/capture_range first, so `labels["nvtx"]` is never set -- `-tagfocus=nvtx=token` says "matched no samples" while `export` on the same capture reports 19 ranges. The unit tests pass because they feed marker records directly, over a path the pipeline never takes. (ii) `nvidia/x/nvtx` is a no-op unless `NVTX_INJECTION64_PATH` names a library exporting `InitializeInjectionNvtx2`; cudaprof sets `LD_PRELOAD` but not that, so 0 markers were collected while the capability record said `nvtx_markers: true`. Pointing it at libcupti turned 0 into 38. (iii) Without `-cuda-profiler-stop` the buffered tail is discarded at `runtime.exit` -- for this workload, decode entirely. | Stop dropping marker subtypes before attribution; set the injection path; stop reporting `nvtx_markers: true` when none were collected. | **open**, worked around |
| B7 | `-top` **rejects a valid profile** -- "PTX and SASS are not from one compilation" -- then prints ~400 KB of line numbers. `go tool pprof -top` reads the same file fine. Two defects: a bad cross-tier identity check, and a diagnostic that dumps every offending line instead of a count and a sample. | | **open**; workaround is `go tool pprof` |
| 21.67, 21.115(a) | The pprof embeds a CUBIN build ID but **no PTX build ID**, so PTX rows cannot be tied to the compilation that produced them. | Fix at the producer: emit a PTX build ID. | **open**, and owned here -- F76E's handoff failed (9DD05C62 is a bare shell; the codex session is out of quota) |

## 3. Denominator and scope not stated

Thirteen entries, the theme with the widest spread across tools. Each is
presentation, and each produced a wrong number in someone's hands.

| # | failure | fix request | status |
|---|---|---|---|
| B2 | `advise` summary counts do not match the findings it prints: 117 Critical / 454 Warning with 10 printed; later 536 against 12 at `-limit 12`, leaving **524 unreachable at any limit**. | Verbatim: print *"showing top N of M"* and name the ordering key. | **open**, persists |
| B3 | Severity ordering **is** sample-share ordering, so `advise` can never yield a structural conclusion -- only a louder version of the same ranking. | | **open** |
| B5 | Empty instruction text in finding titles: 7 of 12 read "...on  at PC 0x4874d0"; 4 read "sampled at " with nothing after. | | **open** |
| 21.117 / 21.118 | `gaps` prints four idle numbers per device -- total idle, three attribution arms, overlay rows -- three of them ms-valued and similar in magnitude, and only total idle is the coverage denominator. Two readers re-derived figures wrongly in one evening. Gap counts print as "gap(s)" and were mislabeled as replay counts. | Convention adopted: any per-gap or per-pct figure quoted from this tool must name its denominator row. | **open** (presentation) |
| 21.140 | "graph replay" is an **overlay row, not an attribution arm**, and nothing in the output says so. | Name whether a row is an arm or an overlay, not just which row it is. | **open** (presentation) |
| 21.153 | Feature request, verbatim: *"a diff that separates 'more launches' from 'same launches, slower', because those have completely different causes and only the second is pressure."* | as quoted | **open** |

## 4. Capture lifecycle

Six entries, now essentially closed.

| # | failure | fix request | status |
|---|---|---|---|
| 8.1(a), 21.85, 21.110 | Go's `runtime.exit` issues `exit_group`, running neither atexit handlers nor ELF destructors, and ~3 MB never filled one 8 MiB CUPTI buffer, so nothing drained mid-run either: `transport_ready false`, 0 records, "capture journal missing finalized record" -- at exit 0. 21.85 isolated it with a **double dissociation**: a C++ control on the same agent, tree and lock collected 65 records complete. | | **fixed** -- `nvidia/x/cudaprofagent`, whose `Finalize()` dlsyms `cudaprof_finalize`; targets call it before every exit path, never via defer. Control with the pre-fix target reproduces the failure. |
| 8.1(c) | Record loss (2.0M, then 513,943 `native_dropped`). The agent retained 100% of what it was offered, so the buffer pool was the constraint; the tmpfs hypothesis was falsified. | Raise BUFFER_POOL. | **fixed** (4 -> 64) |
| 10.6 | (i) `-sample_index=activity_ns` is rejected by `go tool pprof`; the valid names are `kernel_time`, `copy_time`, `copy_bytes`, `activity`. (ii) A dropped `--max-records 8388608` presented as a transport failure. (iii) cudaprof **refuses to overwrite** an existing output, so the zero-record file from a failed run blocks the retry under its own name. | Validate `-sample_index` against the emitted sample types and list them on rejection. Allow `--force`, or name failed captures differently. | **open** |
| 21.112 | `--deep`, `--exhaustive` and `--follow-children` are all rejected or fatal on this host. | | **open** |
| 8.2 | With `CUDAPROF_API_ACTIVITY=0`, correlation is 0.0% and all 537,080 gaps land in Unmatched, so "idle attribution: 100% hardware queue" is vacuous. | | flag semantics; worked around |
| 21.96 | Killing a run under `gpu-lock` leaves an **orphan holding the GPU** while the lock reports free. Twice; the second orphan caused the OOM that made the next arm vacuous. | `kill -TERM -$PGID`, then confirm with `nvidia-smi --query-compute-apps` before the next run. "released" is a statement about the lock, never about the device. | **open**, worked around |

## 5. Our own instrument, fixed

| # | failure | fix request | status |
|---|---|---|---|
| 21.205 | The journal decoder every ledger number passed through -- `e0.py`, registered as E12's instrument -- lived only in `~/tmp/claude-tmp/` and wrapped `json.loads` in `except Exception: pass`. A record that failed to parse vanished with no count, so its record total could not distinguish a complete census from a lossy one. | Ported to `examples/ds4-min/capture_journal.py`: counts malformed and truncated records, verifies the journal hash, refuses site attribution unless the 15-launch grid pattern holds in all 43 layers. All four checks fired against mutated input before being cited (bad sha rc=1; wrong grid entry rc=1; one corrupted byte -> malformed 1; half file -> truncated 1). Both loss counters read zero on the published capture, so no ledger number moved -- now measured rather than assumed. | **fixed** |
| 21.59, 21.115(b) | `DS4_CUDA_PC_PROFILE_PHASE=decode` sets `r.phaseMark`, which **silently gates CUDA graphs off** (`cuda_session_linux.go:1787/:1799/:2022`), so any phase-tagged profile describes an un-graphed program the shipping binary never runs -- and invites fusing kernels to remove launches only the observer pays for. | Refuse, or stamp the profile. An instrument must not change the route under test in silence. | **open**, owned here; blocks the PC-sampling measurement proposed for `ds4_f16_pair_ordered_chunks` |
| 21.28 | `pcSamplingBudget` computed a capacity of 2978 from `MaxOutputBytes` and then clamped it to a flat 512 "not derived from `MaxOutputBytes`, `MaxNames`, or anything else". | Raise the clamp. | **fixed** (`maxPCsPerBuffer` -> 2048); `Close` correctly refuses on loss |
| 21.210 | The scratch C-vs-Go comparison dumper indexed the Go cache buffers at the **state width 1024** where the emitted cache head dimension is **512**. Every layer-2 cache row therefore compared misaligned memory, and the tool reported the result as ordinary tolerance violations with no indication that the two buffers were differently shaped. It read as a severe numerical fault (`KVcompress` 512/512 at max 4.5); the true figure is 993/16384 at max 0.125. I reasoned from the bad table and forwarded it as advice before the stride was found. | **Assert the shapes agree before differencing, and refuse otherwise.** A comparison tool that accepts two buffers of different strides has no way to report the one error it is most likely to make. Cheap version: require both sides to declare `(elements, stride)` and fail on mismatch; better: make the dumper emit the stride it used, so the artifact is self-describing. | **open**, owned here |
| -- | Method rule the above earns, and it is general: **a 100%-violation row is a diagnostic before it is a severity.** Every element differing is the signature of misaligned memory, a wrong dtype, or a wrong base pointer -- real numerical faults are almost never uniform across a buffer whose neighbours are fine. `KVcompress` at 512/512 sat beside `KVcur` at 496/512 and `kqv_back` at 15519/32768, and I read the outlier as the worst finding rather than as the odd one out. Stride-check any row at or near 100% before quoting it. | Add the check to the comparison tool's output: flag rows at >=99% violations as `SUSPECT-SHAPE` rather than printing them like any other row. | **open**, owned here |

## 6. The reference observer

Not a defect -- the observer was never built for this -- but it is the thing
currently blocking an experiment, which makes it the most actionable item here.

| # | failure | fix request | status |
|---|---|---|---|
| observer, 2026-09-19 | The pinned C observer's **CUDA decode path emits no cache history**: only the current `KVcompress` row at the decode position, never the prior raw-cache rows or the compressed-cache history that `encodeMixedAttention` consumes. A position-0 prefill capture gives `attn_comp_*_raw` and compressor scratch but no committed compressed row. A matched-input replay at any decode position is therefore not constructible from current artifacts. | Extend cache extraction in a **scratch copy** -- the reference tree is read-only -- and verify the extended observer takes the same route rather than checking only that endpoints still agree. Retaining history is itself a change to the cache path, and this campaign has already been caught once by a dump that disabled fused norm/RoPE while the endpoints matched. | **open**, blocking |

## 7. Crashes

| # | failure | fix request | status |
|---|---|---|---|
| 21.33 | `cuptiPCSamplingGetData` intermittent null deref in the drain: SIGSEGV at addr=0x0 inside libcupti during cgo execution, 1 crash in 12 drained runs. Reported as a symptom, not claimed as a library defect; it fails loudly, so no measurement is contaminated. | Not ours. Request for cudaprof: survive it -- report a partial capture with a loss count rather than taking the process down. | **open** |
| 21.165(a) | **Never logged before this errata.** `nvidia/x/cudagraph` end-capture thread affinity recurs at **~4% of runs**: `CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD` from "capture batch ffn layer N: end capture", layer varying (6, 7, 17, 19, 33), aborting at 8-55s before any logits exist. The goroutine that ends the capture is not always the one that began it. Invisible to every digest-reading gate in the campaign, because it aborts before there is a digest to read. | | **open** |
| 21.39 | The same class, earlier: `Capture()` lacked `runtime.LockOSThread` -- 4 of 30 in the shipped-defaults arm. A census then found `LockOSThread` in ten physical tests and **no library code**. | Put it in the library. | **fixed** in the library |
| 7.1 | Template kernels crashed the profile reader ("missing CUDA definition"): `code.go`, and a verbatim copy in `profilebundle/bundle.go:162`, matched `"void "+fn.Name+"("` against the **mangled** symbol, and an unresolved kernel was fatal. The device profiler had been dead since `fead1114`; three targets landed with no device profile. | Fix the match key. | **fixed** via `cudalowering.MatchKey` in `04924422` + regression test. Deployment gap worth naming: sampler and reader are separate artifacts, so the installed `/home/tmc/bin/cudaprof` kept failing until rebuilt on DGX. |

## 8. gpu-lock

Twenty-six errata entries mention gpu-lock; six are defects. Correctness was
resolved at 21.90. Fairness was not.

| # | failure | fix request | status |
|---|---|---|---|
| 21.65 | Releases by **path, not by token**: two unconditional `rm -rf "$LOCKDIR"` (the EXIT/INT/TERM trap at :52, the stale reclaim at :60), neither re-reading the owner file. Acquire is non-atomic -- `mkdir` (:50), owner write (:51), trap armed (:52) -- so a death between them leaves an ownerless lockdir, and the waiter branch (:63-67) then prints and waits forever. Reproduced on a scratch lockdir: holder B acquired while A was still alive. Also `env GPU_LOCK_HELD=1` overwrites line 78's export, destroying the value a child could use to verify ownership. | Ownership check before every delete; atomic acquire. | **fixed** by a peer at 21.90 |
| 21.68 | **Starvation.** The acquire loop is an unqueued spin on `mkdir` -- every release is a lottery, no fairness, no aging. parity10 waited 2301s (~760 polls) without acquiring while a peer re-acquired repeatedly. Wait time is unbounded in the presence of any steady producer. | Verbatim: *"Any fix that adds ownership verification should add a FIFO ticket at the same time."* | **open** -- 21.90 added the first and not the second |
| 21.90 | The binary was **replaced under a running session** at ~17:50 on 2026-09-18. Everything measured before that has weaker exclusivity than its "acquired" line asserts. `install(1)` unlinks rather than truncates, so the running holder kept its inode. | The fix and its three-arm positive control are right (old code reproduces two overlapping holders; new gives max-concurrent 1; a genuinely dead holder is still reclaimed -- the third arm is what separates a fix from disabling reclaim). Procedural request: stamp a version in the "acquired" line so a capture records which lock it ran under. | **open** (the stamp) |
| 21.98 | A peer's benchmark harness sampled GPU co-residency without ever calling `nvidia-smi`, and its preflight default string manufactured the condition its own gate checked. | Sample at **draw** time, not only at preflight. | reported to the owner |

## 9. Session tooling

- `~/bin/git-auto-commit-message --auto` fails with "missing the Anthropic API
  key". Every commit in this campaign's later phases is hand-written. **open**
- `it2 session send-text` returns exit 2 or 3 with "line terminator not sent"
  while the text has in fact landed in the peer's composer as pasted content.
  A blind retry delivers a duplicate. Worked around with `send-key ... enter`
  and a `get-screen` check. **open, worked around**
- A second, worse `it2` failure, observed 2026-09-19: the API times out
  ("context deadline exceeded" fetching screen contents, or "failed to send
  terminator") and the CLI reports it by printing **a usage line and exiting
  0**. A transport failure is indistinguishable from a syntax error, and the
  exit status says success. Three retries read as three malformed commands.
  Worse, a one-word probe sent to test it *did* land in the peer's composer,
  so the failing path still mutates shared state. Cleared with `send-key
  ctrl-u`; the real send then succeeded with `--timeout 60s`. Fix requests:
  distinguish transport failure from usage error, exit nonzero on transport
  failure, and say when text landed but the terminator did not. Note also
  that `--timeout` requires a unit (`30s`, not `30`), which the help text
  does not show. **open**
  - **Second occurrence the same day**, on a `send-text` to the same peer:
    exit status printed `Error: failed to get post-send screen contents:
    ... context deadline exceeded` immediately followed by `Run 'it2 session
    send-text --help' for usage.` The text had in fact been delivered to the
    composer in full and was merely unsubmitted -- `send-key enter` completed
    it. So the failure is not in the send at all: it is in the *post-send
    verification read*, and the CLI attributes a read timeout to the write.
    That is the sharper statement of the defect and it raises the priority:
    the two occurrences differ in which stage timed out, and both were
    reported identically as a usage error. A caller who trusts the message
    resends and duplicates. **Fix request, restated narrowly: separate the
    exit status of the send from the exit status of the confirmation read,
    and never print a usage line for either timeout.**
  - **Third occurrence, and it settles the shape.** This one printed
    `Error: failed to send text: context deadline exceeded` -- naming the
    *send* stage, where the second named the post-send *read*. The text had
    again been delivered to the composer **in full**, unsubmitted, and
    `send-key enter` completed it. So across three occurrences the error
    message named three different things and the delivery succeeded every
    time. **The fix request no longer needs to guess at stages: in 3 of 3
    observed timeouts the body was fully delivered, so a timeout must not be
    reported as a send failure.** Report it as `delivered, unconfirmed` and
    exit with a distinct nonzero status; the caller can then submit rather
    than resend. The current behaviour trains callers to retry, which is the
    one action guaranteed to duplicate. Standing workaround, now 3 for 3:
    `get-screen` the target, then `send-key enter` -- never resend blind. - **`it2 send-text` silently command-substitutes the message body**, observed
  2026-09-19 on an inbound peer message. C8332C52's report arrived with
  `ds4_half_round(v.x)+1e-3` and `__float2half_rn` replaced by *empty strings*
  -- the surrounding prose intact, the identifiers gone. The blanks are the
  signature of sender-side substitution, which means the shell attempted to
  **execute** those strings rather than merely dropping them. Harmless with
  these payloads; the same mechanism with a different one is not, and a reader
  who does not know the failure shape reads the gaps as the sender being
  imprecise rather than as text that ran. Our own procedure already bans
  backticks in `send-text` for this reason, which makes this the second
  independent confirmation of the hazard and the first from the *receiving*
  side. **Fix requests: do not evaluate the message body at all; failing that,
  refuse a body containing unescaped backticks or `$(` rather than sending a
  silently redacted message.** Receiver-side mitigation worth adopting: treat
  a message with missing identifiers as *substituted, not truncated*, and ask
  for a resend before reasoning from it. **open**
- **RESOLVED, and the defect was in our procedure.** `F76E` is not an address:
  it is a four-character prefix matching at least two live sessions
  (`F76E579E`, `F76EF7EB`), so `it2` refuses it, correctly. The standing
  procedure named an ambiguous prefix as the cudaprof address, which means it
  has been unroutable since the second `F76E` session existed -- and neither
  side could see it, because the sender reads "no such agent" and the
  recipient is never told anything was addressed to it. **The address is
  `F76EF7EB`, in full.** This is the same rule as run labels and note
  identity lines, one layer down: an identifier must *discriminate*, and
  uniqueness at the time of writing is a property of the population, never of
  the prefix. Fix request for the procedure itself: never record a truncated
  id as an address.
- Superseded, retained for the record: `F76E` was named in our standing
  procedure as the address for cudaprof fix requests, and it does not resolve to an agent. `it2 session list` shows
  `F76E579E` (irssi) and `F76EF7EB` (clangd); no `F76E` appears among this
  session's peers or teammates. The requests above have therefore not been
  delivered. **open, blocked on an address**

## Three method rules

Both were adopted after 21.143 had to be withdrawn:

1. **Rebuild the instrument before reporting a defect in it, and read its
   source from the branch it ships on.** Two entries in this file exist only
   because a stale local build was believed. A standing request follows from
   it: cudaprof should print its own build SHA and dirty state into every
   report it emits.
2. **A correction is a measurement and takes the same controls as the thing
   it corrects.**
3. **A passing gate at a discontinuity is not evidence.** From the parity
   campaign, carried back as a design rule for any numeric gate cudaprof ever
   ships. `kqv_back` differed from C by one ULP -- 0/32768 violations, a clean
   pass -- and that pass was causally sufficient for 5484/8192 downstream,
   because a Q8 quantizer moves a value across a block boundary where an
   arbitrarily small input difference produces a whole-code output difference.
   Tolerance gates assume smooth propagation; a quantizer is not smooth. The
   instrument is sound and the reading it invites is wrong, and it reports
   clean in the direction that stops the search -- the same false-clear as an
   unconditional preamble or a run-order check that cannot fail. **What such a
   gate must state is how far its input sits from the nearest boundary, not
   how far its output sits from the reference.** Adopted by F76EF7EB for
   cudaprof; recorded here so it survives this campaign.

## What the list says as a whole

The largest theme is not a missing feature. It is a tool printing a confident
value where the honest output is "not measured" -- nineteen entries, and in
every one the artifact looked fine. The second largest is attribution, where
the single worst case (21.149) moved a duty cycle from 14.54% to 71.32% and
had been sitting in the data for a full day before anyone read it as a
statement about coverage rather than about the profiler.

Both have a fixed exemplar. 21.67 shows the refusal working. `c40b1884` shows
the attribution fix working, validated from outside by an arm that did not
need it. The requests in this file are mostly asking for those two behaviours
to be applied where they are not yet.
