"""Real SysML v2 textual-notation parser (CoSMA subset).

Replaces the fragile regex extraction engine of :mod:`cuas_sim.sysml_loader`
with a tokenizer + recursive-descent parser that builds an explicit AST:

    SysMLPackage -> SysMLPartDef (:> Super) -> SysMLAttribute (name, type, value, unit)

This is **pure** (no RNG, no I/O, no global state):
``parse_package(text)`` is a deterministic function of its input. The loader keeps
the legacy regex extractor as a fallback so the invariant "``sysml_loader`` never
raises on .sysml content" holds.

Scope: the parser covers only the subset of the SysML v2 textual notation actually
emitted by the auto-generated CoSMA model under ``sysml/``, not the
full KerML/SysML v2 grammar. Constructs it does not model (``import``, ``port``,
nested ``part``, ``doc /* … */``, line comments, ``calc/action/state/requirement
def``) are skipped leniently; only structurally broken input (unbalanced braces,
unterminated string/doc) raises :class:`SysMLParseError`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import NamedTuple

__all__ = [
    "ConstraintBinary",
    "ConstraintExpr",
    "ConstraintLiteral",
    "ConstraintRef",
    "ConstraintUnary",
    "SysMLAttribute",
    "SysMLCalcDef",
    "SysMLCalcInput",
    "SysMLConstraint",
    "SysMLPackage",
    "SysMLParseError",
    "SysMLPartDef",
    "SysMLRequirementDef",
    "SysMLStateAction",
    "SysMLStateDef",
    "parse_package",
]


class SysMLParseError(Exception):
    """Raised on structurally malformed SysML text (unbalanced braces, etc.)."""


@dataclass(frozen=True, slots=True)
class SysMLAttribute:
    """One ``attribute`` declaration inside a ``part def`` body.

    ``value`` is the parsed literal (``float`` for physical/real quantities and
    ``ISQ::*``, ``int`` for ``ScalarValues::Integer``, ``str`` for strings with
    escapes resolved, ``bool`` for booleans). ``unit`` is the raw unit string with
    surrounding ``['…']`` quotes stripped when fully quoted (e.g. ``"kn"``,
    ``"m/s"``, ``"arcmin"``), or ``None`` when the RHS carried no bracketed unit.
    SI conversion is NOT done here — it stays in the loader (``_to_si``).
    """

    name: str
    type_ref: str | None
    value: float | int | str | bool | None
    unit: str | None


@dataclass(frozen=True, slots=True)
class SysMLPartDef:
    """A ``part def Name [:> Super, …] { … }`` declaration."""

    name: str
    specializes: tuple[str, ...]
    attributes: tuple[SysMLAttribute, ...]


# ─── Constraint expression AST ─────────────────────────────


@dataclass(frozen=True, slots=True)
class ConstraintRef:
    """A reference to an ``attribute`` by name inside a ``require constraint``."""

    name: str


@dataclass(frozen=True, slots=True)
class ConstraintLiteral:
    """A numeric / string / boolean literal inside a ``require constraint``."""

    value: float | int | str | bool


@dataclass(frozen=True, slots=True)
class ConstraintUnary:
    """A unary operation (``not`` / ``-``) over a sub-expression."""

    op: str
    operand: ConstraintExpr


@dataclass(frozen=True, slots=True)
class ConstraintBinary:
    """A binary operation: comparison (``<= >= == < >``) or boolean (``and`` / ``or``)."""

    op: str
    left: ConstraintExpr
    right: ConstraintExpr


#: A parsed ``require constraint`` expression tree.
ConstraintExpr = ConstraintBinary | ConstraintUnary | ConstraintRef | ConstraintLiteral


@dataclass(frozen=True, slots=True)
class SysMLConstraint:
    """A ``require constraint { <expr> }`` block."""

    expr: ConstraintExpr


@dataclass(frozen=True, slots=True)
class SysMLRequirementDef:
    """A ``requirement def <'SHORT'> Name [:> Super, …] { … }`` declaration."""

    name: str
    short_name: str | None
    specializes: tuple[str, ...]
    attributes: tuple[SysMLAttribute, ...]
    constraints: tuple[SysMLConstraint, ...]
    refinements: tuple[str, ...] = ()


# ─── State-machine AST ─────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SysMLStateAction:
    """One ``then action stepName : StepType;`` step inside a ``do action`` body."""

    name: str
    type_ref: str | None


@dataclass(frozen=True, slots=True)
class SysMLStateDef:
    """A ``state def Name [:> Super, …] [{ do action { first start; … then done; } }]``.

    ``actions`` are the ordered ``then action … : Type`` steps of the phase's
    ``do action`` body (``first start`` / ``then done`` markers are not actions).
    Special states (``Initial`` / ``MissionComplete``) have empty ``actions``.
    """

    name: str
    specializes: tuple[str, ...]
    actions: tuple[SysMLStateAction, ...]


# ─── Calculation AST ──────────────────────────────────────


class SysMLCalcInput(NamedTuple):
    """One ``in name [: Type]`` parameter of a ``calc def``."""

    name: str
    type_ref: str | None


@dataclass(frozen=True, slots=True)
class SysMLCalcDef:
    """A ``calc def Name [:> Super] { doc; in x : T; … ; return r : T = <expr>; }``.

    ``inputs`` are the ordered ``in`` parameters; ``expr`` is the parsed return
    expression tree (``None`` when the RHS uses a construct outside the
    ``+ - * / ( )`` / reference / literal subset). Metric binding
    is NOT done here — it stays in ``sysml_calculations``.
    """

    name: str
    specializes: tuple[str, ...]
    inputs: tuple[SysMLCalcInput, ...]
    return_name: str | None
    return_type: str | None
    expr: ConstraintExpr | None


@dataclass(frozen=True, slots=True)
class SysMLPackage:
    """A parsed ``package Name { … }`` with its children (file order).

    ``requirement_defs``, ``state_defs`` and ``calc_defs``
    are additive fields; the legacy loader only consumes ``part_defs``,
    so its golden output is unchanged.
    """

    name: str
    part_defs: tuple[SysMLPartDef, ...]
    requirement_defs: tuple[SysMLRequirementDef, ...] = ()
    state_defs: tuple[SysMLStateDef, ...] = ()
    calc_defs: tuple[SysMLCalcDef, ...] = ()


# ─── Tokenizer ─────────────────────────────────────────────────────────────


class _Token(NamedTuple):
    kind: str
    value: str


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
_IDENT_RE = re.compile(r"[A-Za-z_]\w*")

#: Single-character structural symbols (checked after ':>' and '::').
_SINGLE_SYMBOLS: dict[str, str] = {
    "{": "LBRACE",
    "}": "RBRACE",
    ";": "SEMI",
    ",": "COMMA",
    "=": "EQ",
    ":": "COLON",
}

#: Constraint comparison token-kind → operator string.
_CMP_OPS: dict[str, str] = {"LE": "<=", "GE": ">=", "EQEQ": "==", "LT": "<", "GT": ">"}


def _tokenize(text: str) -> list[_Token]:
    """Lex SysML text into tokens, discarding whitespace and comments.

    Raises :class:`SysMLParseError` on an unterminated string or ``/* … */`` doc.
    """
    tokens: list[_Token] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        # Whitespace
        if c in " \t\r\n":
            i += 1
            continue
        # Line comment
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            j = text.find("\n", i + 2)
            i = n if j == -1 else j + 1
            continue
        # Block comment / doc body
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            if j == -1:
                raise SysMLParseError("unterminated block comment (doc)")
            i = j + 2
            continue
        # String literal (handles \" and \\ escapes)
        if c == '"':
            buf: list[str] = []
            i += 1
            closed = False
            while i < n:
                ch = text[i]
                if ch == "\\" and i + 1 < n:
                    nxt = text[i + 1]
                    buf.append('"' if nxt == '"' else ("\\" if nxt == "\\" else nxt))
                    i += 2
                    continue
                if ch == '"':
                    closed = True
                    i += 1
                    break
                buf.append(ch)
                i += 1
            if not closed:
                raise SysMLParseError("unterminated string literal")
            tokens.append(_Token("STRING", "".join(buf)))
            continue
        # Unit bracket [...]
        if c == "[":
            j = text.find("]", i + 1)
            if j == -1:
                raise SysMLParseError("unterminated unit bracket '['")
            tokens.append(_Token("UNIT", text[i + 1 : j]))
            i = j + 1
            continue
        # Two-char symbols
        if c == ":" and i + 1 < n and text[i + 1] == ">":
            tokens.append(_Token("SPECIALIZES", ":>"))
            i += 2
            continue
        if c == ":" and i + 1 < n and text[i + 1] == ":":
            tokens.append(_Token("DCOLON", "::"))
            i += 2
            continue
        # Comparison operators — two-char first, then single.
        if c == "<":
            if i + 1 < n and text[i + 1] == "=":
                tokens.append(_Token("LE", "<="))
                i += 2
            else:
                tokens.append(_Token("LT", "<"))
                i += 1
            continue
        if c == ">":
            if i + 1 < n and text[i + 1] == "=":
                tokens.append(_Token("GE", ">="))
                i += 2
            else:
                tokens.append(_Token("GT", ">"))
                i += 1
            continue
        if c == "=" and i + 1 < n and text[i + 1] == "=":
            tokens.append(_Token("EQEQ", "=="))
            i += 2
            continue
        # Single-quoted short-name / id string (e.g. 'TR-001'). Bare single quotes
        # appear only in the Requirements packages; in the other packages every
        # single quote lives inside a [...] unit or "..." string (golden-safe).
        if c == "'":
            j = text.find("'", i + 1)
            if j == -1:
                raise SysMLParseError("unterminated single-quoted string")
            tokens.append(_Token("SQSTRING", text[i + 1 : j]))
            i = j + 1
            continue
        # Single-char structural symbols
        single = _SINGLE_SYMBOLS.get(c)
        if single is not None:
            tokens.append(_Token(single, c))
            i += 1
            continue
        # Number (only when '-' precedes a digit)
        if c.isdigit() or (c == "-" and i + 1 < n and text[i + 1].isdigit()):
            m = _NUMBER_RE.match(text, i)
            if m is not None:
                tokens.append(_Token("NUMBER", m.group()))
                i = m.end()
                continue
        # Identifier / keyword
        if c.isalpha() or c == "_":
            m = _IDENT_RE.match(text, i)
            assert m is not None
            tokens.append(_Token("IDENT", m.group()))
            i = m.end()
            continue
        # Anything else (e.g. '*' in imports) — a single skippable symbol.
        tokens.append(_Token("SYMBOL", c))
        i += 1
    return tokens


# ─── Recursive-descent parser ──────────────────────────────────────────────


def _strip_unit(raw: str) -> str:
    """Normalise a raw unit string: strip surrounding ``'…'`` only if fully wrapped."""
    u = raw.strip()
    if len(u) >= 2 and u.startswith("'") and u.endswith("'"):
        return u[1:-1]
    return u


def _number_literal(raw: str) -> float | int:
    """Parse a numeric token to ``int`` (plain integer) or ``float`` (decimal/sci)."""
    if any(ch in raw for ch in ".eE"):
        return float(raw)
    return int(raw)


class _Parser:
    def __init__(self, tokens: list[_Token]) -> None:
        self._t = tokens
        self._i = 0

    def _peek(self, ahead: int = 0) -> _Token | None:
        j = self._i + ahead
        return self._t[j] if j < len(self._t) else None

    def _advance(self) -> _Token | None:
        t = self._peek()
        if t is not None:
            self._i += 1
        return t

    def _is(self, kind: str, value: str | None = None, ahead: int = 0) -> bool:
        t = self._peek(ahead)
        if t is None or t.kind != kind:
            return False
        return value is None or t.value == value

    # ── package ──
    def parse_package(self) -> SysMLPackage:
        if not self._is("IDENT", "package"):
            raise SysMLParseError("expected 'package' keyword at start")
        self._advance()  # 'package'
        name = self._read_qualified_name()
        if not self._is("LBRACE"):
            raise SysMLParseError("expected '{' after package name")
        self._advance()  # '{'
        part_defs: list[SysMLPartDef] = []
        requirement_defs: list[SysMLRequirementDef] = []
        state_defs: list[SysMLStateDef] = []
        calc_defs: list[SysMLCalcDef] = []
        while True:
            t = self._peek()
            if t is None:
                raise SysMLParseError("unbalanced braces: package not closed")
            if t.kind == "RBRACE":
                self._advance()
                break
            if t.kind == "IDENT" and t.value == "part" and self._is("IDENT", "def", 1):
                part_defs.append(self._parse_part_def())
            elif t.kind == "IDENT" and t.value == "requirement" and self._is("IDENT", "def", 1):
                requirement_defs.append(self._parse_requirement_def())
            elif t.kind == "IDENT" and t.value == "state" and self._is("IDENT", "def", 1):
                state_defs.append(self._parse_state_def())
            elif t.kind == "IDENT" and t.value == "calc" and self._is("IDENT", "def", 1):
                calc_defs.append(self._parse_calc_def())
            else:
                self._skip_member(in_body=False)
        return SysMLPackage(
            name=name,
            part_defs=tuple(part_defs),
            requirement_defs=tuple(requirement_defs),
            state_defs=tuple(state_defs),
            calc_defs=tuple(calc_defs),
        )

    # ── requirement def ──
    def _parse_requirement_def(self) -> SysMLRequirementDef:
        self._advance()  # 'requirement'
        self._advance()  # 'def'
        short_name: str | None = None
        if self._is("LT"):  # optional <'SHORT-NAME'>
            self._advance()
            sn = self._peek()
            if sn is not None and sn.kind == "SQSTRING":
                short_name = sn.value
                self._advance()
            if self._is("GT"):
                self._advance()
        name_tok = self._advance()
        if name_tok is None or name_tok.kind != "IDENT":
            raise SysMLParseError("expected requirement def name")
        name = name_tok.value
        specializes: list[str] = []
        if self._is("SPECIALIZES"):
            self._advance()
            specializes.append(self._read_qualified_name())
            while self._is("COMMA"):
                self._advance()
                specializes.append(self._read_qualified_name())
        if not self._is("LBRACE"):
            raise SysMLParseError(f"expected '{{' in requirement def {name!r}")
        self._advance()  # '{'
        attrs: list[SysMLAttribute] = []
        constraints: list[SysMLConstraint] = []
        refinements: list[str] = []
        while True:
            t = self._peek()
            if t is None:
                raise SysMLParseError(f"unbalanced braces in requirement def {name!r}")
            if t.kind == "RBRACE":
                self._advance()
                break
            if t.kind == "IDENT" and t.value == "attribute":
                attrs.append(self._parse_attribute())
            elif t.kind == "IDENT" and t.value == "require" and self._is("IDENT", "constraint", 1):
                constraints.append(self._parse_constraint())
            elif t.kind == "SYMBOL" and t.value == "#":
                ref = self._parse_refinement()
                if ref is not None:
                    refinements.append(ref)
            else:
                self._skip_member(in_body=True)
        return SysMLRequirementDef(
            name=name,
            short_name=short_name,
            specializes=tuple(specializes),
            attributes=tuple(attrs),
            constraints=tuple(constraints),
            refinements=tuple(refinements),
        )

    # ── require constraint ──
    def _parse_constraint(self) -> SysMLConstraint:
        self._advance()  # 'require'
        self._advance()  # 'constraint'
        if not self._is("LBRACE"):
            raise SysMLParseError("expected '{' after 'require constraint'")
        self._advance()  # '{'
        expr = self._parse_expr()
        # Consume through the matching '}' (lenient on any residual tokens).
        depth = 1
        while depth > 0:
            t = self._advance()
            if t is None:
                raise SysMLParseError("unbalanced braces in require constraint")
            if t.kind == "LBRACE":
                depth += 1
            elif t.kind == "RBRACE":
                depth -= 1
        return SysMLConstraint(expr=expr)

    def _parse_refinement(self) -> str | None:
        """Capture a ``#refinement dependency 'X' to <target>;`` target (lenient)."""
        self._advance()  # '#'
        target: list[str] = []
        seen_to = False
        while True:
            t = self._peek()
            if t is None or t.kind == "RBRACE":
                break
            self._advance()
            if t.kind == "SEMI":
                break
            if t.kind == "IDENT" and t.value == "to":
                seen_to = True
            elif seen_to and t.kind in ("IDENT", "SQSTRING"):
                target.append(t.value)
            elif seen_to and t.kind == "DCOLON":
                target.append("::")
        return "".join(target) or None

    # ── constraint expression (precedence: or < and < comparison < unary) ──
    def _parse_expr(self) -> ConstraintExpr:
        return self._parse_or()

    def _parse_or(self) -> ConstraintExpr:
        left = self._parse_and()
        while self._is("IDENT", "or"):
            self._advance()
            left = ConstraintBinary("or", left, self._parse_and())
        return left

    def _parse_and(self) -> ConstraintExpr:
        left = self._parse_comparison()
        while self._is("IDENT", "and"):
            self._advance()
            left = ConstraintBinary("and", left, self._parse_comparison())
        return left

    def _parse_comparison(self) -> ConstraintExpr:
        left = self._parse_additive()
        t = self._peek()
        if t is not None and t.kind in _CMP_OPS:
            self._advance()
            return ConstraintBinary(_CMP_OPS[t.kind], left, self._parse_additive())
        return left

    # Arithmetic levels — `*`/`/` bind tighter than `+`/`-`,
    # both below comparison and above unary. Reuses ``ConstraintBinary`` so the
    # constraint evaluator structure is shared; tokens are the existing ``SYMBOL``s.
    def _parse_additive(self) -> ConstraintExpr:
        left = self._parse_multiplicative()
        while self._is("SYMBOL", "+") or self._is("SYMBOL", "-"):
            op_tok = self._advance()
            assert op_tok is not None
            left = ConstraintBinary(op_tok.value, left, self._parse_multiplicative())
        return left

    def _parse_multiplicative(self) -> ConstraintExpr:
        left = self._parse_unary()
        while self._is("SYMBOL", "*") or self._is("SYMBOL", "/"):
            op_tok = self._advance()
            assert op_tok is not None
            left = ConstraintBinary(op_tok.value, left, self._parse_unary())
        return left

    def _parse_unary(self) -> ConstraintExpr:
        if self._is("IDENT", "not"):
            self._advance()
            return ConstraintUnary("not", self._parse_unary())
        if self._is("SYMBOL", "-"):
            self._advance()
            return ConstraintUnary("-", self._parse_unary())
        return self._parse_primary()

    def _parse_primary(self) -> ConstraintExpr:
        t = self._peek()
        if t is None:
            raise SysMLParseError("unexpected end of constraint expression")
        if t.kind == "SYMBOL" and t.value == "(":
            self._advance()
            inner = self._parse_expr()
            if not self._is("SYMBOL", ")"):
                raise SysMLParseError("expected ')' in constraint expression")
            self._advance()
            return inner
        if t.kind == "NUMBER":
            self._advance()
            return ConstraintLiteral(_number_literal(t.value))
        if t.kind == "STRING":
            self._advance()
            return ConstraintLiteral(t.value)
        if t.kind == "IDENT":
            self._advance()
            if t.value in ("true", "false"):
                return ConstraintLiteral(t.value == "true")
            return ConstraintRef(t.value)
        raise SysMLParseError(f"unexpected token in constraint: {t.kind} {t.value!r}")

    # ── state def ──
    def _parse_state_def(self) -> SysMLStateDef:
        self._advance()  # 'state'
        self._advance()  # 'def'
        name_tok = self._advance()
        if name_tok is None or name_tok.kind != "IDENT":
            raise SysMLParseError("expected state def name")
        name = name_tok.value
        specializes: list[str] = []
        if self._is("SPECIALIZES"):
            self._advance()
            specializes.append(self._read_qualified_name())
            while self._is("COMMA"):
                self._advance()
                specializes.append(self._read_qualified_name())
        # Empty form: `state def Name;` (e.g. Initial / MissionComplete).
        if self._is("SEMI"):
            self._advance()
            return SysMLStateDef(name=name, specializes=tuple(specializes), actions=())
        if not self._is("LBRACE"):
            raise SysMLParseError(f"expected '{{' or ';' in state def {name!r}")
        self._advance()  # '{'
        actions: list[SysMLStateAction] = []
        while True:
            t = self._peek()
            if t is None:
                raise SysMLParseError(f"unbalanced braces in state def {name!r}")
            if t.kind == "RBRACE":
                self._advance()
                break
            if t.kind == "IDENT" and t.value == "do" and self._is("IDENT", "action", 1):
                actions.extend(self._parse_do_action())
            else:
                # Unmodeled constructs (entry/exit/transition/accept) — skip leniently.
                self._skip_member(in_body=True)
        return SysMLStateDef(name=name, specializes=tuple(specializes), actions=tuple(actions))

    def _parse_do_action(self) -> list[SysMLStateAction]:
        """Parse a ``do action [wrapper] { first start; then action x : T; … then done; }``."""
        self._advance()  # 'do'
        self._advance()  # 'action'
        if self._is("IDENT"):  # optional wrapper action name
            self._advance()
            if self._is("COLON"):  # optional wrapper type
                self._advance()
                self._read_qualified_name()
        if not self._is("LBRACE"):
            self._skip_to_semi()  # `do action wrapper;` — no body
            return []
        self._advance()  # '{'
        steps: list[SysMLStateAction] = []
        while True:
            t = self._peek()
            if t is None:
                raise SysMLParseError("unbalanced braces in do action")
            if t.kind == "RBRACE":
                self._advance()
                break
            if t.kind == "IDENT" and t.value == "then" and self._is("IDENT", "action", 1):
                self._advance()  # 'then'
                step = self._parse_action_step()
                if step is not None:
                    steps.append(step)
            else:
                # `first start;`, `then done;`, guards, etc. — skip leniently.
                self._skip_to_semi()
        return steps

    def _parse_action_step(self) -> SysMLStateAction | None:
        """Parse one ``action stepName [: StepType];`` ('then' already consumed)."""
        self._advance()  # 'action'
        name_tok = self._peek()
        if name_tok is None or name_tok.kind != "IDENT":
            self._skip_to_semi()
            return None
        self._advance()
        type_ref: str | None = None
        if self._is("COLON"):
            self._advance()
            type_ref = self._read_qualified_name()
        self._skip_to_semi()
        return SysMLStateAction(name=name_tok.value, type_ref=type_ref)

    def _skip_to_semi(self) -> None:
        """Consume tokens up to and including the next ';' (skipping nested blocks)."""
        while True:
            t = self._peek()
            if t is None or t.kind == "RBRACE":
                return
            self._advance()
            if t.kind == "SEMI":
                return
            if t.kind == "LBRACE":
                self._skip_balanced_braces()

    # ── calc def ──
    def _parse_calc_def(self) -> SysMLCalcDef:
        self._advance()  # 'calc'
        self._advance()  # 'def'
        name_tok = self._advance()
        if name_tok is None or name_tok.kind != "IDENT":
            raise SysMLParseError("expected calc def name")
        name = name_tok.value
        specializes: list[str] = []
        if self._is("SPECIALIZES"):
            self._advance()
            specializes.append(self._read_qualified_name())
            while self._is("COMMA"):
                self._advance()
                specializes.append(self._read_qualified_name())
        # Empty form `calc def Name;` (unlikely in CoSMA) — no body.
        if self._is("SEMI"):
            self._advance()
            return SysMLCalcDef(name, tuple(specializes), (), None, None, None)
        if not self._is("LBRACE"):
            raise SysMLParseError(f"expected '{{' or ';' in calc def {name!r}")
        self._advance()  # '{'
        inputs: list[SysMLCalcInput] = []
        return_name: str | None = None
        return_type: str | None = None
        expr: ConstraintExpr | None = None
        while True:
            t = self._peek()
            if t is None:
                raise SysMLParseError(f"unbalanced braces in calc def {name!r}")
            if t.kind == "RBRACE":
                self._advance()
                break
            if t.kind == "IDENT" and t.value == "in":
                inputs.append(self._parse_calc_input())
            elif t.kind == "IDENT" and t.value == "return":
                return_name, return_type, expr = self._parse_calc_return()
            else:
                # `doc` (comment body already stripped), `out`, annotations, etc.
                self._skip_calc_member()
        return SysMLCalcDef(
            name, tuple(specializes), tuple(inputs), return_name, return_type, expr
        )

    def _parse_calc_input(self) -> SysMLCalcInput:
        """Parse one ``in name [: Type | :> Type];`` parameter ('in' not consumed)."""
        self._advance()  # 'in'
        name_tok = self._advance()
        if name_tok is None or name_tok.kind != "IDENT":
            raise SysMLParseError("expected calc input name")
        type_ref: str | None = None
        if self._is("COLON") or self._is("SPECIALIZES"):
            self._advance()
            type_ref = self._read_qualified_name()
        self._skip_to_semi()
        return SysMLCalcInput(name_tok.value, type_ref)

    def _parse_calc_return(self) -> tuple[str | None, str | None, ConstraintExpr | None]:
        """Parse ``return [name] [: Type | :> Type] [= <expr>];`` ('return' not consumed)."""
        self._advance()  # 'return'
        return_name: str | None = None
        if self._is("IDENT"):
            rn = self._advance()
            assert rn is not None
            return_name = rn.value
        return_type: str | None = None
        if self._is("COLON") or self._is("SPECIALIZES"):
            self._advance()
            return_type = self._read_qualified_name()
        expr: ConstraintExpr | None = None
        if self._is("EQ"):
            self._advance()
            try:
                expr = self._parse_expr()
            except SysMLParseError:
                expr = None  # non-arithmetic construct — not an error
        self._skip_to_semi()
        return return_name, return_type, expr

    def _skip_calc_member(self) -> None:
        """Skip one unmodeled calc-def member (``doc``/``out``/annotation) leniently.

        Stops (without consuming) at the next ``in``/``return``/``}`` so a member
        without its own ``;`` (e.g. ``doc /* … */``, whose body the tokenizer
        already dropped) never swallows the following parameter.
        """
        while True:
            t = self._peek()
            if t is None or t.kind == "RBRACE":
                return
            if t.kind == "IDENT" and t.value in ("in", "return"):
                return
            self._advance()
            if t.kind == "SEMI":
                return
            if t.kind == "LBRACE":
                self._skip_balanced_braces()
                return

    # ── part def ──
    def _parse_part_def(self) -> SysMLPartDef:
        self._advance()  # 'part'
        self._advance()  # 'def'
        name_tok = self._advance()
        if name_tok is None or name_tok.kind != "IDENT":
            raise SysMLParseError("expected part def name")
        name = name_tok.value
        specializes: list[str] = []
        if self._is("SPECIALIZES"):
            self._advance()
            specializes.append(self._read_qualified_name())
            while self._is("COMMA"):
                self._advance()
                specializes.append(self._read_qualified_name())
        if not self._is("LBRACE"):
            raise SysMLParseError(f"expected '{{' in part def {name!r}")
        self._advance()  # '{'
        attrs: list[SysMLAttribute] = []
        while True:
            t = self._peek()
            if t is None:
                raise SysMLParseError(f"unbalanced braces in part def {name!r}")
            if t.kind == "RBRACE":
                self._advance()
                break
            if t.kind == "IDENT" and t.value == "attribute":
                attrs.append(self._parse_attribute())
            else:
                self._skip_member(in_body=True)
        return SysMLPartDef(name=name, specializes=tuple(specializes), attributes=tuple(attrs))

    # ── attribute ──
    def _parse_attribute(self) -> SysMLAttribute:
        self._advance()  # 'attribute'
        name_tok = self._advance()
        if name_tok is None or name_tok.kind != "IDENT":
            raise SysMLParseError("expected attribute name")
        name = name_tok.value
        type_ref: str | None = None
        if self._is("SPECIALIZES") or self._is("COLON"):
            self._advance()
            type_ref = self._read_qualified_name()
        value: float | int | str | bool | None = None
        unit: str | None = None
        if self._is("EQ"):
            self._advance()
            value, unit = self._read_value(type_ref)
        # Consume up to and including the terminating ';' (tolerate stray tokens).
        while True:
            t = self._peek()
            if t is None:
                break
            self._advance()
            if t.kind == "SEMI":
                break
            if t.kind == "RBRACE":  # malformed: no ';' — push back is not needed
                self._i -= 1
                break
        return SysMLAttribute(name=name, type_ref=type_ref, value=value, unit=unit)

    def _read_value(
        self, type_ref: str | None
    ) -> tuple[float | int | str | bool | None, str | None]:
        t = self._peek()
        if t is None:
            return None, None
        if t.kind == "STRING":
            self._advance()
            return t.value, None
        if t.kind == "NUMBER":
            self._advance()
            unit: str | None = None
            if self._is("UNIT"):
                ut = self._advance()
                assert ut is not None
                unit = _strip_unit(ut.value)
            if type_ref == "ScalarValues::Integer":
                return int(t.value), unit
            return float(t.value), unit
        if t.kind == "IDENT" and t.value in ("true", "false"):
            self._advance()
            return (t.value == "true"), None
        # Unmodeled value form (enum/ref) — leave value None; caller skips to ';'.
        return None, None

    # ── helpers ──
    def _read_qualified_name(self) -> str:
        parts: list[str] = []
        t = self._advance()
        if t is None or t.kind != "IDENT":
            raise SysMLParseError("expected identifier")
        parts.append(t.value)
        while self._is("DCOLON") and self._is("IDENT", ahead=1):
            self._advance()  # '::'
            nxt = self._advance()
            assert nxt is not None
            parts.append(nxt.value)
        return "::".join(parts)

    def _skip_member(self, in_body: bool) -> None:
        """Skip one unmodeled member (import/port/nested part/doc/…) leniently."""
        while True:
            t = self._peek()
            if t is None:
                return
            if t.kind == "RBRACE":
                return  # let the enclosing loop handle the closing brace
            if in_body and t.kind == "IDENT" and t.value == "attribute":
                return  # next attribute member
            if (
                not in_body
                and t.kind == "IDENT"
                and t.value == "part"
                and self._is("IDENT", "def", 1)
            ):
                return  # next part def
            if (
                not in_body
                and t.kind == "IDENT"
                and t.value == "requirement"
                and self._is("IDENT", "def", 1)
            ):
                return  # next requirement def
            if (
                not in_body
                and t.kind == "IDENT"
                and t.value == "state"
                and self._is("IDENT", "def", 1)
            ):
                return  # next state def
            if (
                not in_body
                and t.kind == "IDENT"
                and t.value == "calc"
                and self._is("IDENT", "def", 1)
            ):
                return  # next calc def
            self._advance()
            if t.kind == "SEMI":
                return
            if t.kind == "LBRACE":
                self._skip_balanced_braces()
                if self._is("SEMI"):
                    self._advance()
                return

    def _skip_balanced_braces(self) -> None:
        """Consume tokens through the matching '}' ('{' already consumed)."""
        depth = 1
        while depth > 0:
            t = self._advance()
            if t is None:
                raise SysMLParseError("unbalanced braces in skipped block")
            if t.kind == "LBRACE":
                depth += 1
            elif t.kind == "RBRACE":
                depth -= 1


def parse_package(text: str) -> SysMLPackage:
    """Parse a single ``.sysml`` package into an AST.

    Raises :class:`SysMLParseError` on structurally malformed input (unbalanced
    braces, unterminated string/doc, missing ``package``/``{``).
    """
    return _Parser(_tokenize(text)).parse_package()
