from __future__ import annotations

from typing import List

import colorama
from colorama import Fore, Style

from .ast import (
    Program, Stmt, Expr, OutputStmt, Literal, Identifier, BinaryExpr, CallExpr,
    DefStmt, LetStmt, IfStmt, ReturnStmt, UseStmt
)
from .ast_pool import ASTPool
from .errors import ParseError
from .lexer import Token, TokenType

# 初始化colorama，使Windows终端也能显示颜色
colorama.init(autoreset=True)


class Parser:
    def __init__(self, tokens: List[Token], filename: str = "unknown") -> None:
        self.tokens = tokens
        self.current = 0
        self.filename = filename
        # AST节点对象池，用于缓存和复用AST节点对象
        self.ast_pool = ASTPool()

    def parse(self) -> Program:
        statements: List[Stmt] = []
        while not self._is_at_end():
            statements.append(self._statement())
        return Program(statements)

    # --- helpers ---

    def _is_at_end(self) -> bool:
        return self._peek().type == TokenType.EOF

    def _peek(self) -> Token:
        return self.tokens[self.current]

    def _previous(self) -> Token:
        return self.tokens[self.current - 1]

    def _advance(self) -> Token:
        if not self._is_at_end():
            self.current += 1
        return self._previous()

    def _check(self, type_: TokenType) -> bool:
        if self._is_at_end():
            return False
        return self._peek().type == type_

    def _match(self, *types: TokenType) -> bool:
        for t in types:
            if self._check(t):
                self._advance()
                return True
        return False

    def _consume(self, type_: TokenType, message: str) -> Token:
        if self._check(type_):
            return self._advance()
        token = self._peek()
        # 检查是否是缺少分号的错误
        if type_ == TokenType.SEMICOLON:
            # 对于缺少分号的错误，我们应该使用前一个token的位置，而不是当前token的位置
            # 这样可以更准确地指示缺少分号的位置
            if self.current > 0:
                # 使用前一个token的位置
                prev_token = self._previous()
                line = prev_token.line
                # 列号应该是前一个token的列号加上前一个token的长度
                # 这样可以指示缺少分号的准确位置
                column = getattr(prev_token, 'column', 1) + len(prev_token.lexeme)
            else:
                # 如果没有前一个token，使用当前token的位置
                line = token.line
                column = getattr(token, 'column', 1)
            
            # 尝试读取文件内容，获取导致错误的具体代码行
            code_line = ""
            try:
                with open(self.filename, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if line <= len(lines):
                        code_line = lines[line - 1].rstrip()
            except Exception:
                # 如果无法读取文件，使用空字符串
                code_line = ""
            
            # 移除文件名中的 .syl 扩展名，因为错误提示模板中已经包含了
            filename_without_ext = self.filename.replace('.syl', '')
            
            # 定义错误提示模板，使用ANSI转义序列实现颜色渲染
            # 颜色代码定义
            # 使用RGB值构建ANSI转义序列
            COLOR_SCENE = '\x1b[38;2;107;99;90m'  # #6b635a
            COLOR_BORDER = '\x1b[38;2;184;169;154m'  # #b8a99a
            COLOR_POEM = '\x1b[38;2;62;58;53m'  # #3e3a35
            COLOR_PROBLEM = '\x1b[38;2;90;122;140m'  # #5a7a8c
            COLOR_STAGE = '\x1b[38;2;90;90;90m'  # #5a5a5a
            COLOR_TEXT = '\x1b[38;2;74;69;64m'  # #4a4540
            COLOR_SEMICOLON = '\x1b[38;2;166;124;0m'  # #a67c00
            COLOR_LINK = '\x1b[38;2;126;182;255m'  # 海盐蓝
            COLOR_THIN_LINE = '\x1b[38;2;224;220;213m'  # #e0dcd5
            COLOR_NOTE = '\x1b[38;2;122;111;100m'  # #7a6f64
            COLOR_QUOTE = '\x1b[38;2;156;140;128m'  # #9c8c80
            COLOR_SILVER_MOON = '\x1b[38;2;184;169;154m'  # #b8a99a
            COLOR_RESET = '\x1b[0m'  # 重置颜色
            
            # 构建错误提示信息
            error_message = f"""
{COLOR_SCENE}[ 场景：{filename_without_ext}.syl · 第{line}行 · 黎明 ]{COLOR_RESET}
{COLOR_BORDER}─────────────── ⋅☾⋅ ───────────────{COLOR_RESET}
{COLOR_POEM}语句至此，气息未收，
句读无凭，意犹悬空。
当结未结，其言未终；
句钤未下，章不成工。{COLOR_RESET}
{COLOR_PROBLEM}> Problem：Problem::UnexpectedEndOfStatement
> Place: {filename_without_ext}:{line}:{column}{COLOR_RESET}
{COLOR_STAGE}[ 舞台提示 ]{COLOR_RESET}
{COLOR_TEXT}Add a {COLOR_SEMICOLON}semicolon(;){COLOR_TEXT} at the end of the sentence to make it complete and in place;{COLOR_RESET}
{COLOR_LINK}Reference link: `https://github.com/Racaem/Problem/UnexpectedEndOfStatement`{COLOR_RESET}
{COLOR_THIN_LINE}─────────────────────────────────{COLOR_RESET}
{COLOR_NOTE}（幕间低语）
{COLOR_QUOTE}“{COLOR_NOTE}言有尽，意方立。{COLOR_QUOTE}”
——{COLOR_QUOTE}《{COLOR_NOTE}Sylph{COLOR_QUOTE}》{COLOR_RESET}
{COLOR_SILVER_MOON}─────────────── ⋅☾⋅ ───────────────{COLOR_RESET}
"""
            # 输出错误提示
            print(error_message)
            # 退出程序
            import sys
            sys.exit(1)
        else:
            # 使用token.column作为列号，如果token没有column属性，使用1
            column = getattr(token, 'column', 1)
            raise ParseError(f"ParseError: {message} (at line {token.line}, column {column})")

    # --- grammar ---

    def _statement(self) -> Stmt:
        if self._match(TokenType.USE):
            return self._use_stmt()
        elif self._match(TokenType.OUT):
            return self._output_stmt()
        elif self._match(TokenType.PUB):
            if self._match(TokenType.DEF):
                return self._def_stmt(is_public=True)
            token = self._peek()
            column = getattr(token, 'column', 1)
            raise ParseError(f"ParseError: Expect 'def' after 'pub'. (at line {token.line}, column {column})")
        elif self._match(TokenType.DEF):
            return self._def_stmt(is_public=False)
        elif self._match(TokenType.LET):
            return self._let_stmt()
        elif self._match(TokenType.IF):
            return self._if_stmt()
        elif self._match(TokenType.RETURN):
            return self._return_stmt()
        else:
            # Try to parse as an expression statement
            expr = self._expression()
            self._consume(TokenType.SEMICOLON, "Expect ';' after expression statement.")
            # For expression statements, we'll just evaluate them and discard the result
            # We'll use a dummy OutputStmt with a literal None for now
            return self.ast_pool.get_output_stmt(self.ast_pool.get_literal(None))
        token = self._peek()
        column = getattr(token, 'column', 1)
        raise ParseError(f"ParseError: Expect statement. (at line {token.line}, column {column})")

    def _use_stmt(self) -> Stmt:
        module_name = self._consume(TokenType.IDENTIFIER, "Expect module name after 'use'.").literal
        self._consume(TokenType.SEMICOLON, "Expect ';' after use statement.")
        return UseStmt(module_name)

    def _output_stmt(self) -> Stmt:
        expr = self._expression()
        self._consume(TokenType.SEMICOLON, "Expect ';' after output statement.")
        return self.ast_pool.get_output_stmt(expr)

    def _def_stmt(self, is_public: bool = False) -> DefStmt:
        name = self._consume(TokenType.IDENTIFIER, "Expect function name after 'def'").literal
        # 检查是否有参数
        param = None
        if not self._check(TokenType.COLON):
            param = self._consume(TokenType.IDENTIFIER, "Expect parameter name after function name").literal
        self._consume(TokenType.COLON, "Expect ':' after function name or parameter")
        body = []
        while not self._check(TokenType.END):
            body.append(self._statement())
        self._consume(TokenType.END, "Expect 'end' after function body")
        return DefStmt(name, param, body, is_public)

    def _let_stmt(self) -> LetStmt:
        name = self._consume(TokenType.IDENTIFIER, "Expect variable name after 'let'.").literal
        self._consume(TokenType.EQUAL, "Expect '=' after variable name.")
        value = self._expression()
        self._consume(TokenType.SEMICOLON, "Expect ';' after variable declaration.")
        return self.ast_pool.get_let_stmt(name, value)

    def _if_stmt(self) -> IfStmt:
        condition = self._expression()
        self._consume(TokenType.COLON, "Expect ':' after if condition.")
        body = []
        while not self._check(TokenType.END):
            body.append(self._statement())
        self._consume(TokenType.END, "Expect 'end' after if body.")
        return self.ast_pool.get_if_stmt(condition, body)

    def _return_stmt(self) -> ReturnStmt:
        expr = self._expression()
        self._consume(TokenType.SEMICOLON, "Expect ';' after return statement.")
        return self.ast_pool.get_return_stmt(expr)

    def _expression(self) -> Expr:
        return self._comparison()

    def _comparison(self) -> Expr:
        expr = self._term()
        while self._match(TokenType.LESS):
            operator = self._previous().lexeme
            right = self._term()
            expr = self.ast_pool.get_binary_expr(expr, operator, right)
        return expr

    def _term(self) -> Expr:
        expr = self._factor()
        while self._match(TokenType.PLUS, TokenType.MINUS):
            operator = self._previous().lexeme
            right = self._factor()
            expr = self.ast_pool.get_binary_expr(expr, operator, right)
        return expr

    def _factor(self) -> Expr:
        expr = self._primary()
        while self._match(TokenType.MULTIPLY):
            operator = self._previous().lexeme
            right = self._primary()
            expr = self.ast_pool.get_binary_expr(expr, operator, right)
        return expr

    def _primary(self) -> Expr:
        if self._match(TokenType.INT, TokenType.STRING, TokenType.BOOL):
            return self.ast_pool.get_literal(self._previous().literal)
        elif self._match(TokenType.LEFT_PAREN):
            # 处理括号表达式
            expr = self._expression()
            self._consume(TokenType.RIGHT_PAREN, "Expect ')' after expression.")
            return expr
        elif self._match(TokenType.LEFT_BRACKET):
            # 处理列表字面量
            elements = []
            if not self._check(TokenType.RIGHT_BRACKET):
                elements.append(self._expression())
                while self._match(TokenType.COMMA):
                    elements.append(self._expression())
            self._consume(TokenType.RIGHT_BRACKET, "Expect ']' after list elements.")
            return self.ast_pool.get_literal(elements)
        elif self._match(TokenType.IDENTIFIER):
            expr = self.ast_pool.get_identifier(self._previous().literal)
            
            # 处理成员访问和方法调用 (dot notation)
            while self._match(TokenType.DOT):
                if not self._match(TokenType.IDENTIFIER):
                    token = self._peek()
                    column = getattr(token, 'column', 1)
                    raise ParseError(f"ParseError: Expect identifier after '.'. (at line {token.line}, column {column})")
                member_name = self._previous().literal
                
                # Parse arguments for method call (parenthesis-free)
                args = []
                if not self._is_at_end() and self._peek().type not in (TokenType.PLUS, TokenType.MINUS, TokenType.MULTIPLY, TokenType.LESS, TokenType.SEMICOLON, TokenType.COLON, TokenType.RIGHT_PAREN, TokenType.RIGHT_BRACKET):
                    # Parse first argument
                    args.append(self._expression())
                    # Parse additional arguments separated by commas
                    while self._match(TokenType.COMMA):
                        args.append(self._expression())
                
                # Create CallExpr for method call
                expr = self.ast_pool.get_call_expr(f"{expr.name}.{member_name}", self.ast_pool.get_literal(args))
            
            # Check if it's a function call with arguments
            # 函数调用的参数应该是一个表达式，不应该是运算符
            if not self._is_at_end() and self._peek().type not in (TokenType.PLUS, TokenType.MINUS, TokenType.MULTIPLY, TokenType.LESS, TokenType.SEMICOLON, TokenType.COLON, TokenType.RIGHT_PAREN, TokenType.RIGHT_BRACKET):
                # This is a function call with arguments
                argument = self._expression()
                return self.ast_pool.get_call_expr(expr.name, argument)
            
            return expr
        token = self._peek()
        column = getattr(token, 'column', 1)
        raise ParseError(f"ParseError: Expect expression. (at line {token.line}, column {column})")
