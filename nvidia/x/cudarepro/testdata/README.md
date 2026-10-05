# Historical lane 59 refusal fixtures

The two seed records are unchanged JSON objects selected from `report.seeds`
in the 2026-10-01 `planted.json` receipt, then serialized with indentation.
They were not enriched with missing fields. Original receipt SHA-256:
`d916c328d091d4148af354bcb361199a990a5ab8d36d6295e7cbfdfcd6b7b6e8`.

Seed 17 recorded two findings in two attempts. Seed 39 recorded one in two.
Both lack the runner-produced mismatch identity and confirmation run records;
both must be refused. Historical paths in the records are labels, not test
dependencies.

The pristine PTX is the exact file from commit `fc2eb5d9c` at
`nvidia/x/cudaschedule/testdata/schedule_race.ptx`, matching both records'
pristine-source SHA-256:
`e7270d009661c19b75d197dffea64c824080c1b21fa637dfa5543d36b29be49d`.

These are negative import controls, not native qualification or a fresh
reproduction-rate measurement.
