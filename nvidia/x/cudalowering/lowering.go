package cudalowering

// A Lowering records the SASS opcodes ptxas emits for a PTX opcode.
//
// PTX and SASS prefixes are matched by cudaptx.OpcodeSummary.Count and
// cudasass.OpcodeSummary.Count respectively, so a prefix names whole opcode
// fields: "ld.global" covers "ld.global.nc.f32" and "LDG" covers
// "LDG.E.CONSTANT.F32".
type Lowering struct {
	// PTX is the PTX opcode prefix the kernel asked for.
	PTX string

	// SASS lists the opcode prefixes ptxas may emit for PTX. The lowering is
	// satisfied if any of them occurs.
	SASS []string

	// Exclusive reports that ptxas emits these SASS opcodes only for this PTX
	// opcode. Finding them in a kernel whose PTX does not use PTX is then a
	// divergence. It is false for opcodes ptxas also emits on its own account:
	// IMAD for address arithmetic, DFMA for a contracted mul.f64 and add.f64.
	Exclusive bool

	// Spill reports that the SASS opcodes are local memory traffic, so that an
	// unsourced occurrence is a register spill ptxas introduced rather than an
	// access the kernel wrote. Only meaningful with Exclusive.
	Spill bool
}

// DefaultLowerings is the table [Compare] uses when [Options.Lowerings] is nil.
//
// It is deliberately short. An entry earns its place by naming a lowering whose
// failure is worth a report: a specialized unit that went unused, an
// instruction ptxas has no other reason to emit, or memory traffic that says
// where a value ended up. Opcodes that ptxas rearranges freely are absent,
// because an entry that fires on every kernel reports nothing.
//
// Notably absent are the scalar mul and add opcodes. ptxas contracts them into
// FFMA and DFMA, so an entry for them would flag every kernel compiled with the
// default -fmad=true. A caller auditing for contraction adds them.
var DefaultLowerings = []Lowering{
	{PTX: "ld.local", SASS: []string{"LDL"}, Exclusive: true, Spill: true},
	{PTX: "st.local", SASS: []string{"STL"}, Exclusive: true, Spill: true},

	{PTX: "shfl.sync", SASS: []string{"SHFL"}, Exclusive: true},
	{PTX: "vote.sync", SASS: []string{"VOTE"}, Exclusive: true},
	{PTX: "dp4a", SASS: []string{"IDP4A"}, Exclusive: true},
	{PTX: "dp2a", SASS: []string{"IDP2A"}, Exclusive: true},
	{PTX: "tex", SASS: []string{"TEX", "TLD"}, Exclusive: true},
	{PTX: "mma", SASS: []string{"HMMA", "IMMA", "OMMA", "QMMA"}, Exclusive: true},

	{PTX: "fma.rn.f32", SASS: []string{"FFMA"}},
	{PTX: "fma.rn.f64", SASS: []string{"DFMA"}},
	{PTX: "atom.global", SASS: []string{"ATOMG"}},
	{PTX: "atom.shared", SASS: []string{"ATOMS"}},
	{PTX: "red.global", SASS: []string{"REDG"}},
	{PTX: "ld.global", SASS: []string{"LDG"}},
	{PTX: "st.global", SASS: []string{"STG"}},
	{PTX: "ld.shared", SASS: []string{"LDS"}},
	{PTX: "st.shared", SASS: []string{"STS"}},
}
