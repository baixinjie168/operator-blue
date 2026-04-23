# -*- coding: UTF-8 -*-
"""
功能：基于z3求解器求解SMT问题，约束表达式必须为标准的python合法表达式
使用示例：
    builder = Z3ConstraintBuilder()

    # 1. 声明变量 x，限制其 dtype 只能是 int8 或 uint8
    builder.declare_var("x", allowed_dtypes=["int8", "uint8"])

    # 2. 声明变量 y，限制其 dtype 只能是 float32
    builder.declare_var("y", allowed_dtypes=["float32"])

    # 3. 变量 z 未显式声明，将在使用时自动创建，默认支持所有类型

    # 添加约束
    builder.add_constraint('x.dtype == "int8"')  # 有效约束
    # builder.add_constraint('x.dtype == "float32"') # 这将导致 UNSAT，因为 float32 不在 x 的允许列表中

    builder.add_constraint('len(x.shape) == 2')
    builder.add_constraint('x[0] > 100')  # 触发 x 的数据约束，由于 x 是 int8，范围是 [-128, 127]，所以 x[0] 最大为 127

    builder.add_constraint('y[0] < 0.5')  # 触发 y 的数据约束

    # 求解
    builder.solve()
"""
import ast
from collections import defaultdict
from typing import List

import z3

from common_utils.logger_util import LazyLogger
from data_definition.constants import DataMatchMap
from param_constraint_solve.expression_preprocess_utils import TensorVar, ASTtoZ3Converter

logger = LazyLogger()


class ExpressionPreprocessor:
    """Preprocesses expressions before solving."""

    @staticmethod
    def apply_keyword_replace(expr: str) -> str:
        """
        基于人工经验替换表达式中的一些关键词，如nullptr不是python中合法字段，替换为None.需要替换的关键词定义于EXPR_KEYWORD_REPLACE中
        :param expr: 待处理的表达式
        :return: 替换之后的表达式
        """
        for keyword, replacement in DataMatchMap.EXPR_KEYWORD_REPLACE.items():
            if replacement is None:
                expr = expr.replace(keyword, 'None')
            elif isinstance(replacement, str):
                expr = expr.replace(keyword, f"'{replacement}'")
            else:
                expr = expr.replace(keyword, str(replacement))
        return expr

    @staticmethod
    def preprocess_expressions(expressions: List[str]) -> List[str]:
        """
        批量校验表达式的合法性
        :param expressions:
        :return:
        """
        processed = []
        for expr in expressions:
            expr = ExpressionPreprocessor.apply_keyword_replace(expr)
            processed.append(expr)
        return processed

    @staticmethod
    def validate_expression(expr: str) -> bool:
        """
        校验表达式的合法性
        :param expr: 待校验表达式
        :return: True/False
        """
        try:
            ast.parse(expr, mode='eval')
            return True
        except SyntaxError as e:
            logger.error(f"Expression {expr} is invalid by ast validation, err msg : {str(e)}")
            return False


# ==========================================
# 4. 求解器构建
# ==========================================
class Z3ConstraintBuilder:
    def __init__(self):
        self.solver = z3.Solver()
        self.var_map = {}
        self._slice_counter = 0  # 切片计数器

    def get_next_slice_id(self):
        self._slice_counter += 1
        return self._slice_counter

    def declare_var(self, var_name, allowed_dtypes=None):
        """
        显式声明变量，并指定其允许的 dtype 列表。
        """
        if var_name in self.var_map:
            logger.warning(f"[Warn] var {var_name} already declared, duplicate declaration ignored")
            return
        self.var_map[var_name] = TensorVar(var_name, self.solver, allowed_dtypes)
        dtypes_str = allowed_dtypes if allowed_dtypes else "All"
        logger.info(f"[Declare] {var_name} (allowed_dtypes: {dtypes_str})")

    def get_or_create_var(self, var_name):
        # 如果未显式声明，则自动创建（默认支持所有类型）
        if var_name not in self.var_map:
            self.var_map[var_name] = TensorVar(var_name, self.solver, allowed_dtypes=None)
        return self.var_map[var_name]

    def add_constraints(self, expr_str_list: List[str]):
        """
        批量添加表达式
        :param expr_str_list: 表达式列表，其中的每个元素为字符串，必须是python的合法表达式
        """
        for expr_str in expr_str_list:
            replace_expr = ExpressionPreprocessor.apply_keyword_replace(expr_str)
            if ExpressionPreprocessor.validate_expression(replace_expr):
                self.add_constraint(replace_expr)

    def add_constraint(self, expr_str):
        try:
            tree = ast.parse(expr_str, mode='eval')
            converter = ASTtoZ3Converter(self)
            z3_constraint = converter.visit(tree.body)
            self.solver.add(z3_constraint)
            logger.info(f"[OK] {expr_str}")
        except Exception as e:
            logger.error(f"[FAIL] {expr_str}: {e}")

    def solve(self):
        logger.info(f"Solve result: {self.solver.check()}")
        solve_result = defaultdict(dict)
        if self.solver.check() == z3.sat:
            logger.info("Model details:")
            m = self.solver.model()
            for name, t_var in self.var_map.items():
                dtype_val = str(m.eval(t_var.dtype))
                shape_len = m.eval(z3.Length(t_var.shape)).as_long()
                solve_result[name]['dtype'] = dtype_val
                try:
                    shape_vals = [m.eval(t_var.shape[i]).as_long() for i in range(shape_len)]
                    solve_result[name]['shape'] = shape_vals
                except Exception as e:
                    logger.info(f"Shape value solve result is UnKnown, err msg : {str(e)}")
                    shape_vals = "Unknown"

                logger.info(f"  {name}:")
                logger.info(f"    dtype: {dtype_val}")
                logger.info(f"    shape len: {shape_len}")
                logger.info(f"    shape: {shape_vals}")

                if t_var._data is not None:
                    logger.info("    data (sample):")
                    logger.info(f"      [0]: {m.eval(t_var.data[0])}")
                else:
                    logger.info("    data: <No Constraint>")
        return solve_result
