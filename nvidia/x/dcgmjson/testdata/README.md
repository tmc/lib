# Constructed 4.6.1 diagnostic output

constructed-4.6.1.json is constructed, not measured or an upstream full output.
Field spellings and structure follow official NVIDIA/DCGM commit
64df9f894541e426e416131a9820cae97aa4dd81 (4.6.1):
- dcgmi/Diag.cpp HelperJsonAddResult/Entities/Metadata/TestSummary/BuildOutput;
- dcgmi/tests/DiagTests.cpp (NOT_RUN omission, Pass/Fail/Skip rendering);
- nvvs/include/NvvsJsonStrings.h;
- testing/python3/dcgm_diag_schema.json.

Source: https://github.com/NVIDIA/DCGM/tree/64df9f894541e426e416131a9820cae97aa4dd81

The GPU/model/driver/message/code values here are test data, not observed hardware.
DCGM WARN is rendered Fail by default. NOT_RUN rows and summaries are omitted.
Therefore the parser cannot recover warn-vs-fail or classify omitted entities as
not-run. Error IDs are numeric DCGM_FR codes, not guessed from message text.
Runtime-error-only and multi-iteration profiles are outside this first reader.
