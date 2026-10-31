# Trinity-Mini source and toy fixtures

Model revision: arcee-ai/Trinity-Mini at cdb81e8130815fec299cccc8f8700004b8366e1e.
config.json is the pinned official config. index-contract.json retains the
original index hash, sorted-name digest and total payload, instead of copying
the megabyte index. Independent range reads of all eleven shard headers
confirm all 12031 names/shapes/dtypes, including BF16 expert_bias.

reference.json is the independent synthetic reference, SHA256
460caf3ea6192d17c19b5eac56ef16616bf58bf96983f16a9d64a8e2db41b145.
The official Python model source SHA256 is
67950e54244545ad8b6016dbfafdf0addb05c398f6afc0e77b621a5693be189c.
Generation and independent NumPy checks are retained in the task's
20261029-afmoe-compare/reference/ directory. These are toy arithmetic fixtures,
not model weights or native CUDA qualification.

Weighted expert contributions have a fixed BF16 boundary. Scatter-add
accumulation can depend on backend; the fixture's named policies are not a
universal bitwise promise. muP scaling follows declared Torch 2.13 opmath
scalar behavior, which differs from older Torch 2.0.
