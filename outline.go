package main

import (
	"go/ast"
	"go/token"
	"log"
	"slices"
	"strconv"

	"github.com/ncruces/wasm2go/internal/passes"
	"golang.org/x/tools/go/ast/astutil"
)

// Function splitting (outlining).
//
// The Go compiler's per-function optimization cost (register allocation, stack
// slot interference graphs) grows super-linearly with function size, so a
// handful of enormous functions can make a module slow or impossible to
// compile. Splitting them into smaller helpers keeps that cost bounded.
//
// The transform exploits the fact that Wasm control flow is structured: every
// br/br_if/br_table is a goto to an *enclosing* block's boundary label, and
// branch() has already copied any escaping operand-stack values into that
// block's result/param variables before the goto. So a whole structured block
// can be moved into a helper that returns a small integer "selector" telling
// the caller where to continue:
//
//	selNormal  the block fell off its end; the caller continues after it.
//	selReturn  a Wasm `return`; results are in the frame, caller returns them.
//	1+labelID  a br to a label outside the block; caller does `goto` (if the
//	           label lives in the caller) or returns the same selector to
//	           propagate it further out.
//
// Locals shared between a function and its helpers live in a heap frame struct
// passed by pointer; references are rewritten from `vN` to `fr.vN`. Only
// :=-defined temporaries (materialized stack values) stay function-local, and
// a block is only ever extracted when it is closed over them (it defines every
// such temporary it uses), so nothing live ever crosses a helper boundary
// outside the frame. This invariant is asserted, not assumed.
const (
	selNormal = 0  // block completed normally (fell through)
	selReturn = -1 // function should return; results are in the frame
)

type outliner struct {
	*funcCompiler
	max  int
	base string // base name for helpers and the frame type

	// frame holds the names/types of locals promoted to the frame struct, in
	// declaration order, plus the result fields r0.. appended at the end.
	frame      []frameField
	frameNames map[string]bool // membership test for frame
	saDefined  map[string]bool // names defined via := anywhere (kept local)

	frameType *ast.Ident
	nresults  int

	helpers []ast.Decl
	count   int // helper counter, for naming

	// labelSel maps label names to non-zero selector ids (and back), stable for
	// the whole function so a `return sel` and the matching `goto` agree.
	labelSel map[string]int
	selLabel map[int]string
}

type frameField struct {
	name string
	typ  ast.Expr
}

// outline splits fn.decl into smaller helper functions if its body exceeds max
// AST nodes. Generated helpers and the frame struct are appended to t.out.
func (fn *funcCompiler) outline(max int) {
	body := fn.decl.Body
	if body == nil || fn.decl.Recv == nil || nodeCount(body) <= max {
		return
	}
	if *outlineLimit >= 0 {
		if fn.outlineDone >= *outlineLimit {
			return
		}
		fn.outlineDone++
	}

	// Functions are not all named yet when cleanup runs (only exports are), so
	// fall back to a unique synthetic base name for helpers and the frame type.
	base := fn.decl.Name.Name
	if base == "" {
		fn.outlineSeq++
		base = "outline" + strconv.Itoa(fn.outlineSeq)
	}

	o := &outliner{
		funcCompiler: fn,
		max:          max,
		base:         base,
		frameNames:   map[string]bool{},
		saDefined:    map[string]bool{},
		frameType:    newID(base + "_frame"),
		labelSel:     map[string]int{},
		selLabel:     map[int]string{},
	}
	o.collectLocals()

	size := nodeCount(body)
	log.Printf("outlining %s (%d nodes)", base, size)

	// Recursively extract oversized structured blocks into helpers.
	escapes := o.split(body, true)
	if len(escapes) != 0 {
		// Every escape from the top-level function targets one of its own
		// labels (handled by goto) or is a return (handled directly), so
		// nothing should propagate out. If it does, our assumptions are wrong.
		panic("outline: unhandled escapes from " + fn.decl.Name.Name)
	}

	if len(o.helpers) == 0 {
		return // nothing was extractable
	}

	// Rewrite frame locals to fr.<name> across the function and every helper,
	// then drop their now-redundant declarations.
	o.rewriteFrame(body)
	for _, h := range o.helpers {
		o.rewriteFrame(h.(*ast.FuncDecl).Body)
	}

	// Prepend frame allocation and parameter copies to the function body.
	var prelude []ast.Stmt
	prelude = append(prelude, &ast.AssignStmt{
		Tok: token.DEFINE,
		Lhs: []ast.Expr{newID("fr")},
		Rhs: []ast.Expr{&ast.UnaryExpr{Op: token.AND, X: &ast.CompositeLit{Type: o.frameType}}}})
	if fn.decl.Type.Params != nil {
		for _, f := range fn.decl.Type.Params.List {
			for _, name := range f.Names {
				prelude = append(prelude, &ast.AssignStmt{
					Tok: token.ASSIGN,
					Lhs: []ast.Expr{&ast.SelectorExpr{X: newID("fr"), Sel: newID(name.Name)}},
					Rhs: []ast.Expr{newID(name.Name)}})
			}
		}
	}
	body.List = append(prelude, body.List...)

	// Emit the frame struct type and the helpers; tidy each helper.
	o.emitFrameType()
	for _, h := range o.helpers {
		hd := h.(*ast.FuncDecl)
		passes.RemoveUnusedLocals(hd)
		if !*noopt {
			passes.UnnestBlocks(hd)
			passes.RemoveSelfAssigns(hd)
			passes.RemoveBlankAssigns(hd)
			passes.RemoveEmptyStmts(hd)
		}
		ast.Inspect(hd, fn.resolveImports)
	}
	o.out.Decls = append(o.out.Decls, o.helpers...)
	log.Printf("outlined  %s: %d -> %d helpers", base, size, len(o.helpers))
}

// collectLocals classifies the function's locals: parameters and var-declared
// locals become frame fields; :=-defined names stay function-local.
func (o *outliner) collectLocals() {
	add := func(name string, typ ast.Expr) {
		if o.frameNames[name] {
			return
		}
		o.frameNames[name] = true
		o.frame = append(o.frame, frameField{name, typ})
	}

	if o.decl.Type.Params != nil {
		for _, f := range o.decl.Type.Params.List {
			for _, name := range f.Names {
				add(name.Name, f.Type)
			}
		}
	}

	ast.Inspect(o.decl.Body, func(n ast.Node) bool {
		switch s := n.(type) {
		case *ast.DeclStmt:
			if gd, ok := s.Decl.(*ast.GenDecl); ok && gd.Tok == token.VAR {
				for _, spec := range gd.Specs {
					if vs, ok := spec.(*ast.ValueSpec); ok && vs.Type != nil {
						for _, name := range vs.Names {
							if name.Name != "_" {
								add(name.Name, vs.Type)
							}
						}
					}
				}
			}
		case *ast.AssignStmt:
			if s.Tok == token.DEFINE {
				for _, lhs := range s.Lhs {
					if id, ok := lhs.(*ast.Ident); ok && id.Name != "_" {
						o.saDefined[id.Name] = true
					}
				}
			}
		}
		return true
	})

	// Result fields r0..r{n-1}.
	if o.decl.Type.Results != nil {
		for _, f := range o.decl.Type.Results.List {
			n := len(f.Names)
			if n == 0 {
				n = 1
			}
			for range n {
				name := "r" + strconv.Itoa(o.nresults)
				o.nresults++
				o.frame = append(o.frame, frameField{name, f.Type})
				o.frameNames[name] = true
			}
		}
	}
}

// minUnit is the smallest block worth extracting; below it the dispatch
// overhead outweighs the gain.
const minUnit = 24

// unit is an extractable structured block found within a function body.
type unit struct {
	node    ast.Node        // statement to replace (a BlockStmt or its LabeledStmt)
	block   *ast.BlockStmt  // the block's body
	size    int             // node count of block
	replace func(ast.Node)  // O(1) in-place replacement of node
}

// split extracts oversized structured blocks from body into helpers until body
// is no larger than o.max (or no further progress is possible). isMain marks
// the top-level function body. It returns the selectors that escape body and
// must be handled by an outer caller.
//
// One scan finds every outermost extractable block; the largest are extracted
// (each recursively split) until the residual fits. Label collection is hoisted
// out of the per-unit work, keeping the whole pass roughly O(nodes * depth).
func (o *outliner) split(body *ast.BlockStmt, isMain bool) map[int]bool {
	escapes := map[int]bool{}
	total := nodeCount(body)
	if total <= o.max {
		return escapes
	}

	var units []unit
	o.findUnits(body, &units)
	slices.SortFunc(units, func(a, b unit) int { return b.size - a.size })

	// Labels available as goto targets at this level. Computed once; each
	// unit's own (internal) labels are subtracted in extract. A unit never
	// targets a sibling unit's labels (Wasm only branches to enclosing blocks),
	// so leaving extracted siblings' labels in the set is harmless.
	callerLabels := collectLabels(body)

	for i := range units {
		if total <= o.max {
			break
		}
		o.extract(units[i], isMain, escapes, callerLabels)
		total -= units[i].size - 4 // minus the small dispatch that replaces it
	}
	return escapes
}

// findUnits appends the outermost extractable blocks within b to out, recursing
// through non-extractable structure (ifs, switches, labels) but not descending
// into a block once it is recorded as a unit.
func (o *outliner) findUnits(b *ast.BlockStmt, out *[]unit) {
	for i := range b.List {
		i := i
		o.findInStmt(b.List[i], func(n ast.Node) { b.List[i] = n.(ast.Stmt) }, out)
	}
}

func (o *outliner) findInStmt(s ast.Stmt, replace func(ast.Node), out *[]unit) {
	record := func(node ast.Node, block *ast.BlockStmt) bool {
		if isCaseBody(block) || !o.closed(block) {
			return false
		}
		if n := nodeCount(block); n > minUnit {
			*out = append(*out, unit{node: node, block: block, size: n, replace: replace})
		}
		return true // extractable (or a closed block we won't descend into)
	}

	switch x := s.(type) {
	case *ast.BlockStmt:
		if record(x, x) {
			return
		}
		o.findUnits(x, out)
	case *ast.LabeledStmt:
		if blk, ok := x.Stmt.(*ast.BlockStmt); ok && record(x, blk) {
			return
		}
		o.findInStmt(x.Stmt, func(n ast.Node) { x.Stmt = n.(ast.Stmt) }, out)
	case *ast.IfStmt:
		o.findInStmt(x.Body, func(n ast.Node) { x.Body = n.(*ast.BlockStmt) }, out)
		if x.Else != nil {
			o.findInStmt(x.Else, func(n ast.Node) { x.Else = n.(ast.Stmt) }, out)
		}
	case *ast.SwitchStmt:
		for _, cc := range x.Body.List {
			if c, ok := cc.(*ast.CaseClause); ok {
				for j := range c.Body {
					j := j
					o.findInStmt(c.Body[j], func(n ast.Node) { c.Body[j] = n.(ast.Stmt) }, out)
				}
			}
		}
	}
}

// closed reports whether b defines every :=-defined name it references, i.e. no
// function-local temporary is live across b's boundary.
func (o *outliner) closed(b *ast.BlockStmt) bool {
	defined := map[string]bool{}
	ast.Inspect(b, func(n ast.Node) bool {
		if a, ok := n.(*ast.AssignStmt); ok && a.Tok == token.DEFINE {
			for _, lhs := range a.Lhs {
				if id, ok := lhs.(*ast.Ident); ok {
					defined[id.Name] = true
				}
			}
		}
		return true
	})
	ok := true
	ast.Inspect(b, func(n ast.Node) bool {
		if id, ok2 := n.(*ast.Ident); ok2 && o.saDefined[id.Name] && !defined[id.Name] {
			ok = false
		}
		return ok
	})
	return ok
}

// extract moves the unit's block into a new helper and replaces it in place
// with a dispatch on the helper's selector. callerLabels is the set of label
// names usable as goto targets at the call site (the containing function).
func (o *outliner) extract(u unit, isMain bool, escapes map[int]bool, callerLabels map[string]bool) {
	if *extractLimit >= 0 {
		if o.extractCount >= *extractLimit {
			return // debug: leave this block inline
		}
		o.extractCount++
		log.Printf("extract #%d: %s block (%d nodes)", o.extractCount, o.base, u.size)
	}
	node, block := u.node, u.block
	o.count++
	name := o.base + "_" + strconv.Itoa(o.count)

	// Labels defined anywhere in this unit (including a wrapping loop label).
	internal := collectLabels(node)

	// Rewrite escaping gotos to selector returns. Genuine Wasm `return`s are
	// rewritten to a frame store + selReturn, but only the first time code
	// leaves the top-level function (isMain): deeper extractions only ever see
	// selector returns we already produced, which must not be touched again.
	o.convertEscapes(block, internal, isMain)

	// Build the helper body, followed by a normal-completion return. For a loop
	// (a labeled block) the loop label is emitted on an empty statement before
	// the block's statements rather than wrapping them, so recursive splitting
	// descends into the block's children instead of re-extracting the whole
	// block; `goto <loop label>` still restarts the loop.
	hbody := &ast.BlockStmt{}
	if ls, ok := node.(*ast.LabeledStmt); ok {
		hbody.List = append(hbody.List, &ast.LabeledStmt{Label: ls.Label, Stmt: &ast.EmptyStmt{}})
		hbody.List = append(hbody.List, block.List...)
		hbody.List = append(hbody.List, retLit(selNormal))
	} else {
		block.List = append(block.List, retLit(selNormal))
		hbody = block
	}

	helper := &ast.FuncDecl{
		Recv: modRecvList,
		Name: newID(name),
		Type: &ast.FuncType{
			Params: &ast.FieldList{List: []*ast.Field{{
				Names: []*ast.Ident{newID("fr")},
				Type:  &ast.StarExpr{X: o.frameType}}}},
			Results: &ast.FieldList{List: []*ast.Field{{Type: newID("int32")}}}},
		Body: hbody}
	o.helpers = append(o.helpers, helper)

	// Recurse: split the helper itself.
	subEsc := o.split(hbody, false)

	// The selectors this helper can return: every literal selector return in
	// its body (except normal completion), plus whatever its own callees
	// propagate.
	esc := scanSelectorReturns(hbody)
	delete(esc, selNormal)
	for sel := range subEsc {
		esc[sel] = true
	}

	// The labels reachable as goto targets at this call site are those visible
	// in the containing function that don't belong to the extracted unit.
	avail := make(map[string]bool, len(callerLabels))
	for l := range callerLabels {
		if !internal[l] {
			avail[l] = true
		}
	}

	dispatch := o.makeDispatch(name, esc, isMain, avail, escapes)
	// Wrap in a block so the replacement is valid both as a list element and in
	// a *ast.BlockStmt slot (an if branch); UnnestBlocks tidies it away later.
	u.replace(&ast.BlockStmt{List: []ast.Stmt{dispatch}})
}

// convertEscapes rewrites, within block, every goto to a label outside internal
// into `return <selector>`. If convertReturns is set, genuine Wasm `return`s are
// also rewritten to store results in the frame and `return selReturn`.
func (o *outliner) convertEscapes(block *ast.BlockStmt, internal map[string]bool, convertReturns bool) {
	astutil.Apply(block, nil, func(c *astutil.Cursor) bool {
		switch s := c.Node().(type) {
		case *ast.BranchStmt:
			if s.Tok == token.GOTO && !internal[s.Label.Name] {
				c.Replace(retLit(o.selFor(s.Label.Name)))
			}
		case *ast.ReturnStmt:
			if !convertReturns {
				break
			}
			stmts := make([]ast.Stmt, 0, len(s.Results)+1)
			for i, r := range s.Results {
				stmts = append(stmts, &ast.AssignStmt{
					Tok: token.ASSIGN,
					Lhs: []ast.Expr{&ast.SelectorExpr{X: newID("fr"), Sel: newID("r" + strconv.Itoa(i))}},
					Rhs: []ast.Expr{r}})
			}
			stmts = append(stmts, retLit(selReturn))
			c.Replace(&ast.BlockStmt{List: stmts})
		}
		return true
	})
}

// scanSelectorReturns returns the set of literal selector values appearing in
// `return <int>` statements within n (selector returns produced by this pass).
func scanSelectorReturns(n ast.Node) map[int]bool {
	sels := map[int]bool{}
	ast.Inspect(n, func(n ast.Node) bool {
		if ret, ok := n.(*ast.ReturnStmt); ok && len(ret.Results) == 1 {
			if v, ok := intLitValue(ret.Results[0]); ok {
				sels[v] = true
			}
		}
		return true
	})
	return sels
}

// makeDispatch builds the statement that calls helper `name` and acts on its
// selector. Selectors naming a label in callerLabels become a goto; selReturn
// in the top-level function returns the frame results; anything else is
// propagated to the caller (and recorded in the enclosing escapes set).
func (o *outliner) makeDispatch(name string, esc map[int]bool, isMain bool, callerLabels map[string]bool, escapes map[int]bool) ast.Stmt {
	sel := newID("__sel")
	var cases []ast.Stmt

	// Normal completion: fall through (continue after the block).
	cases = append(cases, &ast.CaseClause{List: []ast.Expr{intLit(selNormal)}})

	for s := range esc {
		switch {
		case s == selReturn:
			if isMain {
				cases = append(cases, &ast.CaseClause{
					List: []ast.Expr{intLit(selReturn)},
					Body: []ast.Stmt{o.returnResults()}})
			} else {
				escapes[selReturn] = true
			}
		case callerLabels[o.selLabel[s]]:
			cases = append(cases, &ast.CaseClause{
				List: []ast.Expr{intLit(s)},
				Body: []ast.Stmt{&ast.BranchStmt{Tok: token.GOTO, Label: newID(o.selLabel[s])}}})
		default:
			escapes[s] = true // propagate further out
		}
	}

	// Default: in a helper, propagate the selector to the caller; in the
	// top-level function every selector is handled above, so this is a guard.
	var def []ast.Stmt
	if isMain {
		def = []ast.Stmt{&ast.ExprStmt{X: &ast.CallExpr{
			Fun:  newID("panic"),
			Args: []ast.Expr{&ast.BasicLit{Kind: token.STRING, Value: `"unreachable"`}}}}}
	} else {
		def = []ast.Stmt{&ast.ReturnStmt{Results: []ast.Expr{sel}}}
	}
	cases = append(cases, &ast.CaseClause{Body: def})

	call := &ast.CallExpr{
		Fun:  &ast.SelectorExpr{X: newID("m"), Sel: newID(name)},
		Args: []ast.Expr{newID("fr")}}

	return &ast.SwitchStmt{
		Init: &ast.AssignStmt{Tok: token.DEFINE, Lhs: []ast.Expr{sel}, Rhs: []ast.Expr{call}},
		Tag:  sel,
		Body: &ast.BlockStmt{List: cases}}
}

// returnResults returns the function's results from the frame.
func (o *outliner) returnResults() ast.Stmt {
	if o.nresults == 0 {
		return &ast.ReturnStmt{}
	}
	var res []ast.Expr
	for i := range o.nresults {
		res = append(res, &ast.SelectorExpr{X: newID("fr"), Sel: newID("r" + strconv.Itoa(i))})
	}
	return &ast.ReturnStmt{Results: res}
}

// rewriteFrame rewrites references to frame locals as fr.<name>.
func (o *outliner) rewriteFrame(n ast.Node) {
	// First replace declarations of frame locals with zeroing assignments. The
	// frame struct already declares the fields, but a `var x T` declaration
	// re-zeroes x every time control reaches it; as a persistent frame field x
	// would instead keep its value across loop iterations or repeated helper
	// calls, so we must reproduce the zeroing. (This must also precede the
	// reference rewrite, since a var spec's Names are *ast.Ident slots that
	// cannot hold the fr.<name> selector we substitute below.)
	astutil.Apply(n, nil, func(c *astutil.Cursor) bool {
		ds, ok := c.Node().(*ast.DeclStmt)
		if !ok {
			return true
		}
		gd, ok := ds.Decl.(*ast.GenDecl)
		if !ok || gd.Tok != token.VAR {
			return true
		}
		var lhs, rhs []ast.Expr
		for _, spec := range gd.Specs {
			if vs, ok := spec.(*ast.ValueSpec); ok {
				for _, name := range vs.Names {
					if name.Name == "_" {
						continue
					}
					lhs = append(lhs, newID(name.Name)) // rewritten to fr.<name> below
					rhs = append(rhs, &ast.BasicLit{Kind: token.INT, Value: "0"})
				}
			}
		}
		var repl ast.Stmt = &ast.EmptyStmt{}
		if len(lhs) > 0 {
			repl = &ast.AssignStmt{Tok: token.ASSIGN, Lhs: lhs, Rhs: rhs}
		}
		if c.Index() >= 0 {
			c.Replace(repl)
		} else if len(lhs) > 0 {
			c.Replace(repl)
		} else {
			c.Replace(&ast.EmptyStmt{})
		}
		return true
	})

	astutil.Apply(n, func(c *astutil.Cursor) bool {
		// Don't rewrite identifiers in declaration/label/selector positions,
		// which require a bare *ast.Ident.
		switch c.Name() {
		case "Sel", "Label", "Names":
			return false
		}
		if id, ok := c.Node().(*ast.Ident); ok && o.frameNames[id.Name] {
			c.Replace(&ast.SelectorExpr{X: newID("fr"), Sel: newID(id.Name)})
			return false
		}
		return true
	}, nil)
}

// emitFrameType appends the frame struct type declaration to the output.
func (o *outliner) emitFrameType() {
	fields := make([]*ast.Field, len(o.frame))
	for i, f := range o.frame {
		fields[i] = &ast.Field{Names: []*ast.Ident{newID(f.name)}, Type: f.typ}
	}
	o.out.Decls = append(o.out.Decls, &ast.GenDecl{
		Tok: token.TYPE,
		Specs: []ast.Spec{&ast.TypeSpec{
			Name: o.frameType,
			Type: &ast.StructType{Fields: &ast.FieldList{List: fields}}}}})
}

// selFor returns the stable, non-zero selector id for a label.
func (o *outliner) selFor(label string) int {
	if s, ok := o.labelSel[label]; ok {
		return s
	}
	s := len(o.labelSel) + 1
	o.labelSel[label] = s
	o.selLabel[s] = label
	return s
}

// isCaseBody reports whether b is the body of a switch/select (a list of case
// clauses), which cannot be moved into a helper as an ordinary block.
func isCaseBody(b *ast.BlockStmt) bool {
	for _, s := range b.List {
		switch s.(type) {
		case *ast.CaseClause, *ast.CommClause:
			return true
		}
	}
	return false
}

// collectLabels returns the set of label names defined anywhere in n.
func collectLabels(n ast.Node) map[string]bool {
	labels := map[string]bool{}
	ast.Inspect(n, func(n ast.Node) bool {
		if ls, ok := n.(*ast.LabeledStmt); ok {
			labels[ls.Label.Name] = true
		}
		return true
	})
	return labels
}

// nodeCount returns the number of AST nodes in n, a proxy for compile cost.
func nodeCount(n ast.Node) int {
	count := 0
	ast.Inspect(n, func(n ast.Node) bool {
		if n != nil {
			count++
		}
		return true
	})
	return count
}

func retLit(sel int) ast.Stmt {
	return &ast.ReturnStmt{Results: []ast.Expr{intLit(sel)}}
}

func intLit(v int) ast.Expr {
	if v < 0 {
		return &ast.UnaryExpr{Op: token.SUB, X: &ast.BasicLit{Kind: token.INT, Value: strconv.Itoa(-v)}}
	}
	return &ast.BasicLit{Kind: token.INT, Value: strconv.Itoa(v)}
}

// intLitValue reads back a literal produced by intLit.
func intLitValue(e ast.Expr) (int, bool) {
	neg := false
	if u, ok := e.(*ast.UnaryExpr); ok && u.Op == token.SUB {
		neg, e = true, u.X
	}
	lit, ok := e.(*ast.BasicLit)
	if !ok || lit.Kind != token.INT {
		return 0, false
	}
	v, err := strconv.Atoi(lit.Value)
	if err != nil {
		return 0, false
	}
	if neg {
		v = -v
	}
	return v, true
}
