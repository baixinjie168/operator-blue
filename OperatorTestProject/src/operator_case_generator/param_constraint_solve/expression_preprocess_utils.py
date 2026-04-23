# -*- coding: UTF-8 -*-
"""
功能：基于AST转换器，对输入的表达式进行处理，转换为Z3可接受的数据格式
"""
import ast
from typing import Dict, List

import z3

from common_utils.logger_util import LazyLogger, init_logger

logger = LazyLogger()
# ==========================================
# 1. 数据类型定义
# ==========================================

DTYPE_SPECS = {
    "int": (-128, 127, True),
    "int8": (-128, 127, True),
    "uint8": (0, 255, True),
    "int16": (-32768, 32767, True),
    "int32": (-2 ** 31, 2 ** 31 - 1, True),
    "uint32": (0, 2 ** 32 - 1, True),
    "uint64": (0, 2 ** 64 - 1, True),
    "int64": (-2 ** 63, 2 ** 63 - 1, True),
    "bfp16": (-3.389e38, 3.389e38, False),
    "fp16": (-65504, 65504, False),
    "float": (-3.4e38, 3.4e38, False),
    "fp32": (-3.4e38, 3.4e38, False),
    "fp64": (-1.797e308, 1.797e308, False),
}
# 创建全局枚举
DType, DT_ENUMS = z3.EnumSort('DType', list(DTYPE_SPECS.keys()))
DTYPE_MAP = {name: const for name, const in zip(DTYPE_SPECS.keys(), DT_ENUMS)}


# ==========================================
# 2. 变量模型定义
# ==========================================
class TensorVar:
    def __init__(self, name, solver, allowed_dtypes=None):
        self.name = name
        self.solver = solver

        # 1. 确定允许的 dtype 列表
        if allowed_dtypes:
            # 检查传入的类型是否在全局定义中
            self.allowed_dtypes = [dt for dt in allowed_dtypes if dt in DTYPE_SPECS]
        else:
            # 默认支持所有类型
            self.allowed_dtypes = list(DTYPE_SPECS.keys())

        # 2. 定义 Z3 变量
        self.dtype = z3.Const(f"{name}.dtype", DType)
        self.shape = z3.Const(f"{name}.shape", z3.SeqSort(z3.IntSort()))
        self._data = None  # 惰性初始化

        # 3. 添加基础约束
        self.solver.add(z3.Length(self.shape) >= 0)

        # 4. 【关键】注入 dtype 取值域约束
        # 逻辑：x.dtype 必须等于 allowed_dtypes 中的某一个
        if self.allowed_dtypes:
            domain_constraint = z3.Or([self.dtype == DTYPE_MAP[dt] for dt in self.allowed_dtypes])
            self.solver.add(domain_constraint)

    @property
    def data(self):
        if self._data is None:
            self._data = z3.Array(f"{self.name}.data", z3.IntSort(), z3.RealSort())
            self._add_data_constraints()
        return self._data

    def _add_data_constraints(self):
        # 【优化】仅遍历 allowed_dtypes，减少约束数量
        idx = z3.Int('idx')
        for dt_name in self.allowed_dtypes:
            min_val, max_val, is_int = DTYPE_SPECS[dt_name]
            dtype_const = DTYPE_MAP[dt_name]

            val = z3.Select(self._data, idx)
            range_constraint = z3.And(val >= min_val, val <= max_val)
            if is_int:
                range_constraint = z3.And(range_constraint, z3.IsInt(val))

            self.solver.add(
                z3.Implies(
                    self.dtype == dtype_const,
                    z3.ForAll([idx], z3.Implies(idx >= 0, range_constraint))
                )
            )


# ==========================================
# 3. AST 转换器
# ==========================================
class ASTtoZ3Converter(ast.NodeVisitor):
    def __init__(self, builder):
        self.builder = builder

    def visit(self, node):
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise NotImplementedError(f"不支持的语法节点: {type(node).__name__}")

    # --- 逻辑与比较 ---
    def visit_BoolOp(self, node):
        vals = [self.visit(v) for v in node.values]
        return z3.And(*vals) if isinstance(node.op, ast.And) else z3.Or(*vals)

    def visit_UnaryOp(self, node):
        op = self.visit(node.operand)
        if isinstance(node.op, ast.Not): return z3.Not(op)
        if isinstance(node.op, ast.USub): return -op
        return op

    def visit_Compare(self, node):
        left = self.visit(node.left)
        ops = node.ops
        comps = [self.visit(c) for c in node.comparators]

        res = []
        cur_left = left
        for op, right in zip(ops, comps):
            # --- 处理相等 == ---
            if isinstance(op, ast.Eq):
                # 特殊处理：x.shape == [10, 127]
                if isinstance(right, list) and z3.is_seq(cur_left):
                    len_constraint = z3.Length(cur_left) == len(right)
                    elem_constraints = [cur_left[i] == val for i, val in enumerate(right)]
                    res.append(z3.And(len_constraint, *elem_constraints))
                else:
                    res.append(cur_left == right)

            # --- 处理不等 != ---
            elif isinstance(op, ast.NotEq):
                if isinstance(right, list) and z3.is_seq(cur_left):
                    # 形状不等：长度不同 OR 任意元素不同
                    len_diff = z3.Length(cur_left) != len(right)
                    # 注意：这里需要小心处理，如果长度不同，访问元素可能越界
                    # 简化处理：如果长度不同，则不等；如果长度相同，找元素不同
                    # Z3 中处理序列不等比较复杂，这里简化为：长度不同 OR 存在索引 i < len 使得元素不同
                    # 但为了简单起见，通常只约束长度不同或已知长度下的元素不同
                    # 这里暂只处理长度不同的情况，或已知长度匹配时的元素差异
                    # 对于一般使用，直接 != 可能更安全，但 Z3 Seq 不支持直接 !=
                    # 暂时简化为长度不等（如果列表长度固定）
                    res.append(len_diff)
                else:
                    res.append(cur_left != right)

            # --- 处理其他比较符 (修正了这里的 typo) ---
            elif isinstance(op, ast.Lt):
                res.append(cur_left < right)
            elif isinstance(op, ast.LtE):
                res.append(cur_left <= right)
            elif isinstance(op, ast.Gt):
                res.append(cur_left > right)
            elif isinstance(op, ast.GtE):
                res.append(cur_left >= right)

            # --- 处理 in / not in ---
            elif isinstance(op, ast.In):
                if isinstance(right, list):
                    res.append(z3.Or([cur_left == v for v in right]))
            elif isinstance(op, ast.NotIn):
                if isinstance(right, list):
                    res.append(z3.And([cur_left != v for v in right]))

            # --- 处理 is / is not ---
            elif isinstance(op, ast.IsNot):
                res.append(z3.BoolVal(True))  # 简化处理
            elif isinstance(op, ast.Is):
                res.append(z3.BoolVal(False))

            # 更新左侧，支持链式比较 (如 a < b < c)
            cur_left = right

        return z3.And(*res) if len(res) > 1 else res[0]

    def visit_Attribute(self, node):
        var_name = node.value.id
        t_var = self.builder.get_or_create_var(var_name)
        if node.attr == 'dtype':
            return t_var.dtype
        elif node.attr == 'shape':
            return t_var.shape
        raise NotImplementedError(f"不支持属性: {node.attr}")

    def visit_IfExp(self, node):
        """
        处理三元表达式: body if test else orelse
        例如: x.shape[0] if x.shape[0] > 0 else 0
        """
        # 1. 解析条件部分
        test_expr = self.visit(node.test)

        # 2. 解析真值部分
        body_expr = self.visit(node.body)

        # 3. 解析假值部分
        orelse_expr = self.visit(node.orelse)

        # 4. 转换为 Z3 的 If 函数
        # Z3 语法: If(condition, then_value, else_value)
        return z3.If(test_expr, body_expr, orelse_expr)

    # --- 下标与切片访问 (核心修改) ---
    def visit_Subscript(self, node):
        value = self.visit(node.value)

        # 情况1: 切片操作 x.shape[start:stop]
        if isinstance(node.slice, ast.Slice):
            if not z3.is_seq(value):
                raise NotImplementedError("只能对序列进行切片操作")
            return self._handle_slice(value, node.slice)

        # 情况2: 索引操作 x.shape[0] 或 x.shape[-1]
        else:
            # 处理索引
            idx_node = node.slice

            idx = self.visit(idx_node)

            # 如果是 TensorVar，访问 data
            if isinstance(value, TensorVar):
                # 处理负索引: If(idx < 0, Length + idx, idx)
                # 注意：TensorVar 的 data 是 Array，没有 Length，通常用于具体下标访问
                # 这里假设对 TensorVar 使用负索引较少，暂直接返回 Select
                return z3.Select(value.data, idx)

            # 如果是 Z3 Seq (shape)，访问元素
            elif z3.is_seq(value):
                # 处理负索引: Z3 Seq 支持负数吗？不明确，手动转换最稳
                # actual_idx = If(idx < 0, Length(value) + idx, idx)
                actual_idx = z3.If(idx < 0, z3.Length(value) + idx, idx)
                return value[actual_idx]

            else:
                raise NotImplementedError(f"无法对 {value} 进行下标访问")

    def _handle_slice(self, seq, slice_node):
        """
        将切片操作转换为新的 Z3 序列变量 + 约束
        seq[a:b] -> new_seq, where len(new_seq) = b-a and new_seq[k] == seq[a+k]
        """
        # 1. 解析 start 和 stop
        len_seq = z3.Length(seq)

        # 解析 start (默认为 0)
        if slice_node.lower is None:
            start = z3.IntVal(0)
        else:
            start = self.visit(slice_node.lower)
        # 处理负数 start
        start = z3.If(start < 0, len_seq + start, start)

        # 解析 stop (默认为 len)
        if slice_node.upper is None:
            stop = len_seq
        else:
            stop = self.visit(slice_node.upper)
        # 处理负数 stop
        stop = z3.If(stop < 0, len_seq + stop, stop)

        # 2. 创建一个新的临时序列变量
        slice_id = self.builder.get_next_slice_id()
        slice_var = z3.Const(f"__slice_{slice_id}", z3.SeqSort(z3.IntSort()))

        # 3. 添加约束
        # 约束A: 长度关系 (防止 stop < start 的情况，取 Max(0, stop-start))
        slice_len = z3.If(stop > start, stop - start, 0)
        self.builder.solver.add(z3.Length(slice_var) == slice_len)

        # 约束B: 元素对应关系
        # ForAll k, (0 <= k < slice_len) => slice_var[k] == seq[start + k]
        k = z3.Int(f"__k_{slice_id}")
        body = z3.Implies(
            z3.And(k >= 0, k < slice_len),
            z3.And(
                start + k < len_seq,  # 防止越界
                slice_var[k] == seq[start + k]
            )
        )
        self.builder.solver.add(z3.ForAll([k], body))

        return slice_var

    def visit_Name(self, node):
        return self.builder.get_or_create_var(node.id)

    def visit_Call(self, node):
        func_name = node.func.id
        if func_name == 'len':
            arg = self.visit(node.args[0])
            if isinstance(arg, TensorVar): return z3.Length(arg.shape)
            return z3.Length(arg)
        elif func_name == 'all':
            if len(node.args) == 1 and isinstance(node.args[0], ast.GeneratorExp):
                return self._handle_all_generator(node.args[0])
        raise NotImplementedError(f"Function {func_name} is not achieved")

    def _handle_all_generator(self, gen_node):
        comprehension = gen_node.generators[0]

        # 1. 解析迭代目标
        iter_target = self.visit(comprehension.iter)

        # 2. 确定遍历的 Z3 对象及元素类型
        target_obj = None
        is_sequence = False
        element_sort = None  # 元素的 Z3 类型

        if isinstance(iter_target, TensorVar):
            # 遍历张量数据 -> Array(Int, Real)
            _ = iter_target.data
            target_obj = iter_target.data
            is_sequence = False
            element_sort = z3.RealSort()  # data 定义为 Real 数组
        elif z3.is_seq(iter_target):
            # 遍历形状 -> Seq
            target_obj = iter_target
            is_sequence = True
            element_sort = z3.IntSort()  # shape 定义为 Int 序列
        else:
            raise NotImplementedError(f"all() 不支持遍历类型: {iter_target}")

        # 3. 准备量词变量和占位符
        loop_var_name = comprehension.target.id
        condition_ast = gen_node.elt
        slice_id = self.builder.get_next_slice_id()
        k = z3.Int(f"__k_all_{slice_id}")

        # 【关键修正】根据元素类型创建对应类型的占位符
        placeholder = z3.Const(f"__placeholder_{loop_var_name}", element_sort)

        # 4. 解析条件表达式
        class TempVisitor(ASTtoZ3Converter):
            def __init__(self, builder, var_name, ph):
                super().__init__(builder)
                self.var_name = var_name
                self.ph = ph

            def visit_Name(self, node):
                if node.id == self.var_name:
                    return self.ph
                return super().visit_Name(node)

        temp_visitor = TempVisitor(self.builder, loop_var_name, placeholder)
        cond_expr = temp_visitor.visit(condition_ast)

        # 5. 获取实际元素值
        if is_sequence:
            actual_val = target_obj[k]
        else:
            actual_val = z3.Select(target_obj, k)

        # 6. 替换并构建约束
        # substitute 要求 from 和 to 的类型严格一致，现在已修正
        final_cond = z3.substitute(cond_expr, (placeholder, actual_val))

        if is_sequence:
            bound_constraint = z3.Implies(
                z3.And(k >= 0, k < z3.Length(target_obj)),
                final_cond
            )
        else:
            bound_constraint = z3.Implies(k >= 0, final_cond)

        return z3.ForAll([k], bound_constraint)

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add): return left + right
        if isinstance(node.op, ast.Sub): return left - right
        if isinstance(node.op, ast.Mult): return left * right
        if isinstance(node.op, ast.Div): return left / right
        return NotImplementedError(f"Operator {node.op} is not archieved")

    def visit_Constant(self, node):
        val = node.value
        if isinstance(val, str) and val in DTYPE_MAP: return DTYPE_MAP[val]
        return val

    def visit_Num(self, node):
        return node.n

    def visit_List(self, node):
        return [self.visit(e) for e in node.elts]


class ShapeDimValueExtractor:
    """
    从字符串表达式中提取 shape 维度的上下限。
    严格限制：仅处理 len(xxx.shape) 形式的表达式。
    """
    # 内部标记，用于区分“解析失败”和“解析结果为None”
    _PARSE_ERROR = object()
    _NO_CONSTRAINT = object()  # 标记 is not None (无约束)


    def extract(self, expr_str) -> List[Dict] | None:
        """
        入口方法：解析字符串并返回结果。

        Args:
            expr_str (str): 输入的约束字符串，如 'len(x.shape) <= 5'

        Returns:
            dict or None: 包含 min/max 的字典，如果格式不符则返回 None。
        """
        if not isinstance(expr_str, str):
            logger.error("Shape dim expr must be string")
            return None

        try:
            # 解析字符串为 AST 树
            tree = ast.parse(expr_str.strip(), mode='eval')
        except SyntaxError as e:
            logger.error(f"Parse expr by ast failed, err msg : {str(e)}")
            return None

        node = tree.body
        # 1. 解析节点
        constraint_list = self._dispatch_node(node)

        if constraint_list is self._PARSE_ERROR:
            return None

            # 2. 应用默认最小值规则
        return self._apply_default_min(constraint_list)

    @staticmethod
    def _apply_default_min(constraint_list):
        """
        后处理：补全默认 min=0
        :param constraint_list: 如果有复合表达式，拆解为多个，放入列表中
        """
        final_list = []
        for item in constraint_list:
            if item is None:
                continue

            if isinstance(item, dict):
                if item.get('min') is None:
                    item['min'] = 0
                elif item['min'] < 0:
                    item['min'] = 0
                final_list.append(item)

        return final_list if final_list else None

    def _dispatch_node(self, node) -> List[Dict] | None | object:
        """分发逻辑，统一返回列表"""
        # 处理 or 连接
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
            return self._process_and(node)

            # 处理 or 连接 (并集)
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
            return self._process_or(node)

            # 处理单个比较
        if isinstance(node, ast.Compare):
            return self._process_compare(node)

        return self._PARSE_ERROR

    def _process_or(self,node) -> List[Dict] | object:
        """处理 Or：收集所有结果"""
        collected_list = []
        for value_node in node.values:
            sub_res = self._dispatch_node(value_node)
            if sub_res is self._PARSE_ERROR:
                return self._PARSE_ERROR

            # 如果分支是 is not None (无约束)，则整个 or 表达式实际上无约束（或忽略）
            # 但在 shape 上下文中，通常 is not None 不产生数值约束
            if sub_res is self._NO_CONSTRAINT:
                continue

            if isinstance(sub_res, list):
                collected_list.extend(sub_res)
            else:
                collected_list.append(sub_res)

        return collected_list if collected_list else self._PARSE_ERROR

    def _process_and(self, node) -> List[Dict] | object:
        """处理 And：计算交集"""
        # 假设初始区间为全集
        current_ranges = [{'min': None, 'max': None}]

        for value_node in node.values:
            sub_res = self._dispatch_node(value_node)

            if sub_res is self._PARSE_ERROR:
                return self._PARSE_ERROR

            if sub_res is self._NO_CONSTRAINT:
                continue

            # 处理 is None 的情况
            if None in sub_res:
                return [None]

            # sub_res 现在必然是 List[Dict]
            current_ranges = self._intersect_lists(current_ranges, sub_res)
            if not current_ranges:
                return self._PARSE_ERROR

        return current_ranges

    def _intersect_lists(self, list_a, list_b):
        """计算两个约束列表的交集"""
        result = []
        for a in list_a:
            for b in list_b:
                inter = self._intersect_intervals(a, b)
                if inter:
                    result.append(inter)
        return result

    @staticmethod
    def _intersect_intervals(a, b):
        """计算两个区间的交集"""
        # 计算下界：取最大值
        # None 表示负无穷
        if a['min'] is None:
            new_min = b['min']
        elif b['min'] is None:
            new_min = a['min']
        else:
            new_min = max(a['min'], b['min'])

        # 计算上界：取最小值
        # None 表示正无穷
        if a['max'] is None:
            new_max = b['max']
        elif b['max'] is None:
            new_max = a['max']
        else:
            new_max = min(a['max'], b['max'])

        # 检查区间有效性
        # 如果 new_min 或 new_max 为 None，说明有一边无界，有效
        if new_min is None or new_max is None:
            return {'min': new_min, 'max': new_max}

        if new_min <= new_max:
            return {'min': new_min, 'max': new_max}

        return None  # 交集为空

    @staticmethod
    def _is_valid_shape_len_expr(node):
        """
        校验节点是否为 len(xxx.shape)
        """
        # 1. 必须是函数调用
        if not isinstance(node, ast.Call):
            logger.error("Shape dim expr must be function call")
            return False

        # 2. 函数名必须是 'len'
        if not (isinstance(node.func, ast.Name) and node.func.id == 'len'):
            logger.error("Shape dmi expr must be function call of 'len'")
            return False

        # 3. 参数必须只有一个
        if len(node.args) != 1:
            logger.error("Shape dim expr must contain only one parameter")
            return False

        arg = node.args[0]

        # 4. 参数必须是属性访问
        if isinstance(arg, ast.Attribute):
            # 5. 属性名必须是 'shape'
            if arg.attr == 'shape':
                return True
            logger.error("Shape dim expr's parameter attribute must be 'shape'")

        return False

    @staticmethod
    def _get_num(node):
        """
        从 AST 节点获取数字常量
        """
        if isinstance(node, ast.Constant):  # Python 3.8+
            return node.value
        logger.error("python version is too low")
        return None
    
    @staticmethod
    def _check_is_none(node):
        """检查 node 是否为 'x is None'"""
        if len(node.ops) == 1 and isinstance(node.ops[0], ast.Is):
            right = node.comparators[0]
            if isinstance(right, ast.Constant) and right.value is None:
                return True
        return False

    @staticmethod
    def _check_is_not_none(node):
        """检查 node 是否为 'x is not None'"""
        if len(node.ops) == 1 and isinstance(node.ops[0], ast.IsNot):
            right = node.comparators[0]
            if isinstance(right, ast.Constant) and right.value is None:
                return True
        return False

    def _process_compare(self, node) -> Dict | None | object:
        """
        处理比较操作节点，提取上下限
        :param node: ast解析树中的节点
        """
        # 1. 检查 is None -> 返回包含 None 的列表
        if self._check_is_none(node):
            return [None]

            # 2. 检查 is not None -> 返回哨兵
        if self._check_is_not_none(node):
            return self._NO_CONSTRAINT

        # 3. 检查 len(shape) 比较
        results = {'min': None, 'max': None}

        # 链式比较
        if len(node.ops) == 2:
            left_val = self._get_num(node.left)
            mid_var = node.comparators[0]
            right_val = self._get_num(node.comparators[1])

            if self._is_valid_shape_len_expr(mid_var) and left_val is not None and right_val is not None:
                op1 = type(node.ops[0])
                op2 = type(node.ops[1])

                if op1 == ast.LtE and op2 == ast.LtE:
                    results['min'] = left_val;
                    results['max'] = right_val
                    return [results]  # 包装成列表返回
                elif op1 == ast.GtE and op2 == ast.GtE:
                    results['max'] = left_val;
                    results['min'] = right_val
                    return [results]  # 包装成列表返回

        # 单个比较
        if len(node.ops) == 1:
            op_type = type(node.ops[0])
            left = node.left
            right = node.comparators[0]

            if self._is_valid_shape_len_expr(left):
                val = self._get_num(right)
                if val is None: return self._PARSE_ERROR
                if op_type == ast.LtE:
                    results['max'] = val
                elif op_type == ast.Lt:
                    results['max'] = val - 1
                elif op_type == ast.GtE:
                    results['min'] = val
                elif op_type == ast.Gt:
                    results['min'] = val + 1
                elif op_type == ast.Eq:
                    results['min'] = results['max'] = val
                return [results]  # 包装成列表返回

            elif self._is_valid_shape_len_expr(right):
                val = self._get_num(left)
                if val is None: return self._PARSE_ERROR
                if op_type == ast.GtE:
                    results['max'] = val
                elif op_type == ast.Gt:
                    results['max'] = val - 1
                elif op_type == ast.LtE:
                    results['min'] = val
                elif op_type == ast.Lt:
                    results['min'] = val + 1
                elif op_type == ast.Eq:
                    results['min'] = results['max'] = val
                return [results]  # 包装成列表返回

        return self._PARSE_ERROR

if __name__ == "__main__":
    init_logger("test_shape_dim_extractor")
    extractor = ShapeDimValueExtractor()

    test_cases = [
        'len(x.shape) == 2',  # 期望输出
        'len(ShapeDimValueExtractor.shape) <= 8',  # 期望输出
        '0 <= len(ShapeDimValueExtractor.shape) <= 8',  # 期望输出
        'len(x.shape) == 2',  # 期望输出
        'x.shape[1] == 2',  # 非len函数，期望 None
        'x.shape[1] <= 65535',  # 非len函数，期望 None
        'len(x.data) <= 5',  # 非shape属性，期望 None
        '8 >= len(ShapeDimValueExtractor.shape) >= 0',  # 期望输出
        'len(x.shape) < 5',  # 期望 max=4
        'len(x.shape) >= 2 and len(x.shape) <= 8',
        'weightOptional is None or len(weightOptional.shape) == 1'
    ]

    print(f"{'Input String':<35} | {'Result'}")
    print("-" * 60)
    for case in test_cases:
        res = extractor.extract(case)
        print(f"{case:<35} | {res}")
