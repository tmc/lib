# Bounded policy comparison

Run the public BeamSearch consumer fixture with:

    go test ./x/cudasearch -run 'PolicyComparison|ExampleBeamSearch_policyComparison'

The fixture compares optional profile-guided action ordering, timing-only
ordering, and an ablation removing only the configured priors. All arms use
the same graph, three legal matching rewrite alternatives, objective table,
and three declared comparison units. A candidate evaluation costs one unit;
offline profile admission costs one unit, including refused evidence. Thus
timing-only evaluates three candidates while guided and ablated arms evaluate
two and pay one admission unit. Profile-free ordering retains the declared
rewrite order. Profile evidence changes exploration, never objective values.

These are deterministic host accounting units and estimated objectives, not
nanoseconds, measured compiler costs, correctness-qualified executable plans,
or hardware timing receipts. The deliberately chosen fixture demonstrates a
policy mechanism; it does not establish general policy superiority. WorkAccounted
remains false because the external evaluator does not report compile/extraction
work. Profile acquisition, matching cost, device execution, held-out numerical
validation and real equal-total-cost performance require separate native trials.

Controls retain invalid coverage, mismatched workload, incompatible units and
an unmatched rule, repeated paths, cancellation, empty evaluation allowance,
and deterministic equal-objective tie selection. Native trials must account for
actual evidence acquisition and all search/compile/evaluation overhead, interleave
arms on identical bindings, preserve timing kinds and independently confirm the
winner. This fixture neither launches a process nor mutates a live workload.
