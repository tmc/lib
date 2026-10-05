package libclang

// A CursorKind is the kind of a [Cursor], a value of CXCursorKind.
type CursorKind int32

const (
	StructDecl       CursorKind = 2
	UnionDecl        CursorKind = 3
	ClassDecl        CursorKind = 4
	EnumDecl         CursorKind = 5
	FieldDecl        CursorKind = 6
	EnumConstantDecl CursorKind = 7
	FunctionDecl     CursorKind = 8
	VarDecl          CursorKind = 9
	ParmDecl         CursorKind = 10
	TypedefDecl      CursorKind = 20
	FunctionTemplate CursorKind = 30
	Namespace        CursorKind = 22
	LinkageSpec      CursorKind = 23
	TypeAliasDecl    CursorKind = 36
	MacroDefinition  CursorKind = 501
	CUDAConstantAttr CursorKind = 412
	CUDADeviceAttr   CursorKind = 413
	CUDAGlobalAttr   CursorKind = 414
	CUDASharedAttr   CursorKind = 416

	cursorInvalidFile CursorKind = 70 // the kind of the null cursor
)

// A TypeKind is the kind of a [Type], a value of CXTypeKind.
type TypeKind int32

const (
	Invalid         TypeKind = 0
	Void            TypeKind = 2
	Bool            TypeKind = 3
	CharU           TypeKind = 4
	UChar           TypeKind = 5
	UShort          TypeKind = 8
	UInt            TypeKind = 9
	ULong           TypeKind = 10
	ULongLong       TypeKind = 11
	CharS           TypeKind = 13
	SChar           TypeKind = 14
	Short           TypeKind = 16
	Int             TypeKind = 17
	Long            TypeKind = 18
	LongLong        TypeKind = 19
	Float           TypeKind = 21
	Double          TypeKind = 22
	Half            TypeKind = 31
	Float16         TypeKind = 32
	BFloat16        TypeKind = 39
	Pointer         TypeKind = 101
	LValueReference TypeKind = 103
	Record          TypeKind = 105
	Enum            TypeKind = 106
	Typedef         TypeKind = 107
	FunctionProto   TypeKind = 111
	ConstantArray   TypeKind = 112
	IncompleteArray TypeKind = 114
	Elaborated      TypeKind = 119
)
