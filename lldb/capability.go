package lldb

// Require reports whether symbol can be called.
//
// Require loads the bridge if necessary. It returns the bridge load error or
// a [SymbolError] when the loaded bridge does not provide symbol. Use Require
// when a caller needs to distinguish an unavailable bridge from an older
// bridge; [Has] intentionally reports both cases as false.
func Require(symbol string) error {
	return checkSym(symbol)
}
