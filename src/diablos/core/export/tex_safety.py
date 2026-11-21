r"""Decide whether a user-typed math label may be passed through to LaTeX.

A ``.diablos`` file is a document people exchange, and a line label ends up
verbatim inside a ``.tex`` file that *someone else* compiles. TeX has no
sandbox: ``\input``, ``\openout``, ``\write18`` and ``\begin{filecontents}``
read and write files, and ``\newcommand`` or ``\catcode`` change the host paper
for every line that follows. So "starts and ends with ``$``" is not enough to
justify handing a body through.

The gate is an **allowlist**. A blocklist of dangerous commands cannot be
completed -- ``\csname input\endcsname`` and the ``^^5c`` notation for a
backslash both spell ``\input`` without the regex ever seeing it -- whereas the
set of commands a signal label legitimately needs is small and known: Greek
letters, operators, relations, accents, fonts, delimiters, spacing and the
amsmath matrix environments. Anything outside it is not passed through; the
callers fall back to escaping the label as text, so an unusual label still
compiles, it just shows the source instead of the typeset math.

Both exporters (``TikZExporter._format_explicit_label`` and
``BloxExporter._latex_label``) are reachable from the same export dialog and
must apply this same gate.
"""

import re

_GREEK = {
    "alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta", "eta",
    "theta", "vartheta", "iota", "kappa", "varkappa", "lambda", "mu", "nu", "xi",
    "pi", "varpi", "rho", "varrho", "sigma", "varsigma", "tau", "upsilon", "phi",
    "varphi", "chi", "psi", "omega",
    "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Upsilon", "Phi",
    "Psi", "Omega", "varGamma", "varDelta", "varTheta", "varLambda", "varXi",
    "varPi", "varSigma", "varUpsilon", "varPhi", "varPsi", "varOmega",
}  # fmt: skip

_OPERATORS = {
    "frac", "dfrac", "tfrac", "cfrac", "binom", "sqrt", "sum", "prod", "coprod",
    "int", "iint", "iiint", "oint", "lim", "limsup", "liminf", "sup", "inf",
    "max", "min", "argmax", "argmin", "det", "dim", "ker", "deg", "gcd", "hom",
    "exp", "log", "ln", "lg", "sin", "cos", "tan", "cot", "sec", "csc", "arcsin",
    "arccos", "arctan", "sinh", "cosh", "tanh", "coth", "Pr", "operatorname",
    "mathop", "partial", "nabla", "infty", "cdot", "cdots", "ldots", "vdots",
    "ddots", "dots", "times", "div", "pm", "mp", "ast", "star", "circ", "bullet",
    "oplus", "ominus", "otimes", "oslash", "odot", "wedge", "vee", "land", "lor",
    "lnot", "neg", "cap", "cup", "setminus", "emptyset", "varnothing", "forall",
    "exists", "nexists", "in", "notin", "ni", "subset", "supset", "subseteq",
    "supseteq", "mid", "nmid", "parallel", "perp", "angle", "triangle", "top",
    "bot", "prime", "ell", "hbar", "imath", "jmath", "Re", "Im", "aleph", "wp",
    "degree", "colon", "backslash", "vert", "Vert",
}  # fmt: skip

_RELATIONS_ARROWS = {
    "le", "leq", "leqslant", "ge", "geq", "geqslant", "ne", "neq", "ll", "gg",
    "approx", "equiv", "sim", "simeq", "cong", "propto", "asymp", "doteq",
    "prec", "succ", "preceq", "succeq", "models", "vdash", "dashv", "triangleq",
    "to", "gets", "rightarrow", "leftarrow", "leftrightarrow", "Rightarrow",
    "Leftarrow", "Leftrightarrow", "longrightarrow", "longleftarrow",
    "longleftrightarrow", "Longrightarrow", "Longleftarrow", "Longleftrightarrow",
    "mapsto", "longmapsto", "hookrightarrow", "hookleftarrow", "uparrow",
    "downarrow", "updownarrow", "Uparrow", "Downarrow", "nearrow", "searrow",
    "swarrow", "nwarrow", "implies", "impliedby", "iff", "xrightarrow",
    "xleftarrow", "rightleftharpoons", "leadsto",
}  # fmt: skip

_ACCENTS_FONTS = {
    "hat", "widehat", "bar", "overline", "underline", "tilde", "widetilde",
    "vec", "dot", "ddot", "dddot", "acute", "grave", "breve", "check", "mathring",
    "overrightarrow", "overleftarrow", "overbrace", "underbrace", "overset",
    "underset", "stackrel", "substack", "boldsymbol", "bm", "pmb", "mathbf",
    "mathrm", "mathit", "mathsf", "mathtt", "mathcal", "mathbb", "mathscr",
    "mathfrak", "mathnormal", "text", "textrm", "textit", "textbf", "textsf",
    "texttt", "textnormal", "emph", "mbox", "hbox", "textcolor", "color",
    "displaystyle", "textstyle", "scriptstyle", "scriptscriptstyle", "limits",
    "nolimits", "phantom", "hphantom", "vphantom", "mathstrut", "not", "left",
    "right", "big", "Big", "bigg", "Bigg", "bigl", "bigr", "Bigl", "Bigr",
    "biggl", "biggr", "Biggl", "Biggr", "langle", "rangle", "lfloor", "rfloor",
    "lceil", "rceil", "lvert", "rvert", "lVert", "rVert", "lbrace", "rbrace",
    "lbrack", "rbrack", "quad", "qquad", "thinspace", "medspace", "thickspace",
    "negthinspace", "enspace", "hspace", "mspace", "smash", "begin", "end",
    "ensuremath", "mathclap", "mathllap", "mathrlap",
}  # fmt: skip

#: Every control word (``\letters``) a passed-through label may contain.
SAFE_MATH_COMMANDS = frozenset(_GREEK | _OPERATORS | _RELATIONS_ARROWS | _ACCENTS_FONTS)

#: Environments a label may open. ``filecontents`` writes files, so this is an
#: allowlist too, not "anything after \begin".
SAFE_MATH_ENVIRONMENTS = frozenset(
    {
        "matrix", "pmatrix", "bmatrix", "Bmatrix", "vmatrix", "Vmatrix",
        "smallmatrix", "cases", "aligned", "gathered", "split", "array",
        "subarray",
    }
)  # fmt: skip

_CONTROL_WORD_RE = re.compile(r"\\([a-zA-Z]+)")
_ENVIRONMENT_RE = re.compile(r"\\(?:begin|end)\s*\{\s*([^}]*?)\s*\}")


def math_body_is_safe(body: str) -> bool:
    r"""True when *body* is self-contained math built only from allowlisted commands.

    Structure first: the body may not leave math mode (``$``), may not
    unbalance the braces (``x$} \renewcommand.. \node{$z`` closes the group
    and keeps going), and may not use TeX's ``^^`` character notation, which
    spells any byte -- including a backslash -- without writing it. Then every
    control word must be in :data:`SAFE_MATH_COMMANDS` and every environment in
    :data:`SAFE_MATH_ENVIRONMENTS`.
    """
    if "$" in body or "^^" in body:
        return False
    depth = 0
    for ch in body:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return False
    if depth != 0:
        return False
    if any(cmd not in SAFE_MATH_COMMANDS for cmd in _CONTROL_WORD_RE.findall(body)):
        return False
    return all(env in SAFE_MATH_ENVIRONMENTS for env in _ENVIRONMENT_RE.findall(body))
