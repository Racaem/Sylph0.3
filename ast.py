from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any, List, Optional


class Stmt(ABC):
    """Base class for all statements."""


class Expr(ABC):
    """Base class for all expressions."""


@dataclass
class Program:
    statements: List[Stmt]


@dataclass
class OutputStmt(Stmt):
    expr: Expr
    type_tag: str = "OutputStmt"


@dataclass
class DefStmt(Stmt):
    name: str
    param: str
    body: List[Stmt]
    is_public: bool = False
    type_tag: str = "DefStmt"


@dataclass
class LetStmt(Stmt):
    name: str
    value: Expr
    type_tag: str = "LetStmt"


@dataclass
class IfStmt(Stmt):
    condition: Expr
    body: List[Stmt]
    type_tag: str = "IfStmt"


@dataclass
class ReturnStmt(Stmt):
    expr: Expr
    type_tag: str = "ReturnStmt"


@dataclass
class UseStmt(Stmt):
    module_name: str
    type_tag: str = "UseStmt"


@dataclass
class Literal(Expr):
    value: Any  # int | str | bool
    type_tag: str = "Literal"


@dataclass
class Identifier(Expr):
    name: str
    type_tag: str = "Identifier"


@dataclass
class BinaryExpr(Expr):
    left: Expr
    operator: str
    right: Expr
    type_tag: str = "BinaryExpr"


@dataclass
class CallExpr(Expr):
    callee: str
    argument: Expr
    type_tag: str = "CallExpr"


@dataclass
class UnaryExpr(Expr):
    operator: str
    right: Expr
    type_tag: str = "UnaryExpr"

