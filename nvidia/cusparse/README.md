# cuSPARSE

`GetErrorName` and `GetErrorString` return `(string, error)`. Callers must
check the error before using the returned text. An unavailable library or
symbol returns an error; the supplied cuSPARSE status is the value being
described, not the helper call's error.

Host tests cover a missing library in an isolated subprocess. Set
`CUSPARSE_PHYSICAL_TEST=1` to require native error-string and numerical SpMV
tests. Native tests fail if the required library or device is unavailable.
