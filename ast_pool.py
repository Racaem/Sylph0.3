from __future__ import annotations

from typing import Dict, Any, List, Optional

from .ast import (
    Stmt, Expr, OutputStmt, DefStmt, LetStmt, IfStmt, ReturnStmt,
    Literal, Identifier, BinaryExpr, CallExpr, UnaryExpr
)


class ASTPool:
    """AST节点对象池，用于缓存和复用AST节点对象，减少内存分配和回收的开销。"""
    
    def __init__(self) -> None:
        # 为每种AST节点类型创建一个对象池
        self.pools: Dict[str, List[Any]] = {
            "OutputStmt": [],
            "DefStmt": [],
            "LetStmt": [],
            "IfStmt": [],
            "ReturnStmt": [],
            "Literal": [],
            "Identifier": [],
            "BinaryExpr": [],
            "CallExpr": [],
            "UnaryExpr": []
        }
    
    def get_output_stmt(self, expr: Expr) -> OutputStmt:
        """获取或创建OutputStmt对象"""
        if self.pools["OutputStmt"]:
            stmt = self.pools["OutputStmt"].pop()
            stmt.expr = expr
            return stmt
        return OutputStmt(expr)
    
    def get_def_stmt(self, name: str, param: str, body: List[Stmt]) -> DefStmt:
        """获取或创建DefStmt对象"""
        if self.pools["DefStmt"]:
            stmt = self.pools["DefStmt"].pop()
            stmt.name = name
            stmt.param = param
            stmt.body = body
            return stmt
        return DefStmt(name, param, body)
    
    def get_let_stmt(self, name: str, value: Expr) -> LetStmt:
        """获取或创建LetStmt对象"""
        if self.pools["LetStmt"]:
            stmt = self.pools["LetStmt"].pop()
            stmt.name = name
            stmt.value = value
            return stmt
        return LetStmt(name, value)
    
    def get_if_stmt(self, condition: Expr, body: List[Stmt]) -> IfStmt:
        """获取或创建IfStmt对象"""
        if self.pools["IfStmt"]:
            stmt = self.pools["IfStmt"].pop()
            stmt.condition = condition
            stmt.body = body
            return stmt
        return IfStmt(condition, body)
    
    def get_return_stmt(self, expr: Expr) -> ReturnStmt:
        """获取或创建ReturnStmt对象"""
        if self.pools["ReturnStmt"]:
            stmt = self.pools["ReturnStmt"].pop()
            stmt.expr = expr
            return stmt
        return ReturnStmt(expr)
    
    def get_literal(self, value: Any) -> Literal:
        """获取或创建Literal对象"""
        if self.pools["Literal"]:
            literal = self.pools["Literal"].pop()
            literal.value = value
            return literal
        return Literal(value)
    
    def get_identifier(self, name: str) -> Identifier:
        """获取或创建Identifier对象"""
        if self.pools["Identifier"]:
            ident = self.pools["Identifier"].pop()
            ident.name = name
            return ident
        return Identifier(name)
    
    def get_binary_expr(self, left: Expr, operator: str, right: Expr) -> BinaryExpr:
        """获取或创建BinaryExpr对象"""
        if self.pools["BinaryExpr"]:
            expr = self.pools["BinaryExpr"].pop()
            expr.left = left
            expr.operator = operator
            expr.right = right
            return expr
        return BinaryExpr(left, operator, right)
    
    def get_call_expr(self, callee: str, argument: Expr) -> CallExpr:
        """获取或创建CallExpr对象"""
        if self.pools["CallExpr"]:
            expr = self.pools["CallExpr"].pop()
            expr.callee = callee
            expr.argument = argument
            return expr
        return CallExpr(callee, argument)
    
    def get_unary_expr(self, operator: str, right: Expr) -> UnaryExpr:
        """获取或创建UnaryExpr对象"""
        if self.pools["UnaryExpr"]:
            expr = self.pools["UnaryExpr"].pop()
            expr.operator = operator
            expr.right = right
            return expr
        return UnaryExpr(operator, right)
    
    def release(self, node: Any) -> None:
        """释放AST节点对象到对象池中"""
        if hasattr(node, "type_tag"):
            node_type = node.type_tag
            if node_type in self.pools:
                self.pools[node_type].append(node)
    
    def clear(self) -> None:
        """清空所有对象池"""
        for pool in self.pools.values():
            pool.clear()
