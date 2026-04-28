# -*- coding: UTF-8 -*-
"""
功能：参数约束关系实现
"""
import copy
import random
from collections import defaultdict
from typing import List, Dict, Optional

import torch
import z3
from pydantic import BaseModel

from atk_common_utils.case_config import CaseConfig
from common_utils.common_dispatcher import CommonDispatcher
from common_utils.logger_util import LazyLogger
from data_definition.common_models import DispatcherTargetType
from data_definition.constants import ParamModelConfig, DataMatchMap
from data_definition.param_models_def import CaseInputRangeModel, ParameterPropertyData
from operator_param_models.case_generate import CaseGenerate
from param_constraint_solve.customize_expression_solver_utils import CustomizeConstraintPatch
from param_constraint_solve.z3_expression_solver_utils import Z3ConstraintBuilder
# from param_constraint_solve.expression_solver import ExpressionSolver
from src.common_model_definition import InterParamConstraint, InterConstraintsRuleType, OperatorRule

logger = LazyLogger()


class ParamSetValueFlag(BaseModel):
    dtype: bool = False
    shape: bool = False
    range_value: bool = False


class ParamConstraintUtils(CommonDispatcher):
    def __init__(self, case: CaseConfig, case_generate_instance: CaseGenerate,
                 inter_parameter_constraints: List[InterParamConstraint], operator_rule_data: OperatorRule,
                 param_combinations: Dict[str, ParameterPropertyData] = None,
                 is_generate_real_data: bool = False):
        self.inter_parameter_constraints = inter_parameter_constraints
        # 形如{'x1': {'DType': 'BFLOAT16', 'DataProfile': 'NaN', 'DimCount': '2', 'DimProperty': 'Has_Large_Size',
        # 'DataProfile': 'SubNormal', 'Memory': 'Contiguous'}, 'epsilon': {'Value': '1e-5'},
        # 'additionalOutput': {'Mode': 'True'}}
        self.param_combinations = param_combinations
        self.operator_rule_data = operator_rule_data
        # 是否生成真实数据，若为FALSE只保留生成数据模型的名称，用于后续实际执行的时候调用生成真实数据
        self.is_generate_real_data = is_generate_real_data
        # 数据生成实例，用于在不满足约束条件时，重新生成参数的数据
        self.case_generate_instance = case_generate_instance
        self.operator_name = case.name
        self.case = case
        self.case_input_map = {case_input.name: case_input for case_input in case.inputs}
        self.other_input_map = {each.name: each for each in operator_rule_data.other_parameters}
        self.broadcast_master_params = defaultdict(List[InterParamConstraint])
        self.has_set_value_param = defaultdict(ParamSetValueFlag)
        self.customize_constraint_patch = CustomizeConstraintPatch(case=case,
                                                                   case_generate_instance=case_generate_instance,
                                                                   inter_parameter_constraints=inter_parameter_constraints,
                                                                   operator_rule_data=operator_rule_data,
                                                                   param_combinations=param_combinations,
                                                                   is_generate_real_data=is_generate_real_data)

    def is_param_all_input(self, relation_params: List[str]):
        """
        判断参数的角色是否都为输入
        :param relation_params: 约束相关的参数名称
        """
        for param_name in relation_params:
            if param_name not in self.case_input_map and param_name not in self.other_input_map:
                logger.error(f"Can't match this parameter in input params, param name : {param_name}")
                return False
        return True

    def set_has_value_param_status(self, param_name, constraint_type: str):
        """
        将已确定的参数的属性的状态在self.has_set_value_param中设置为True，
        :param param_name: 参数名称
        :param constraint_type: 约束的类型
        :return: None
        """
        param_set_value_flag = self.has_set_value_param.get(param_name, ParamSetValueFlag())
        for type_key, type_list in DataMatchMap.CONSTRAINT_TYPE_MAP.items():
            if constraint_type in type_list:
                param_set_value_flag.__setattr__(type_key, True)
        self.has_set_value_param[param_name] = param_set_value_flag

    def correct_operator_param(self, correct_run_time: int = 2):
        """
        修复case的参数，在源数据上修改
        :param correct_run_time: 修正操作执行次数，避免部分约束条件之间存在冲突，多次执行确保所有的约束条件都可以满足
        :return: None
        """
        logger.info(f"Start correct case param, operator name : {self.operator_name}")
        all_constraint_relations = [relation.value for relation in InterConstraintsRuleType]
        match_relations = []
        for relation in all_constraint_relations:
            match_relation = [each for each in self.inter_parameter_constraints if each.type == relation]
            if match_relation:
                match_relations.extend(match_relation)
        customize_constraints = []
        for _ in range(correct_run_time):
            for constraint_relation in match_relations:
                if not self.is_param_all_input(constraint_relation.params):
                    continue
                relation_type = constraint_relation.type.value
                if relation_type in ParamModelConfig.STRICT_CONSTRAINT_TYPE:
                    customize_constraints.append(constraint_relation)
                else:
                    logger.debug(f"Relation type : {relation_type}, use z3 solver")
                    z3_check_result = self.solve_constraint_by_z3(constraint_relation)
                    if not z3_check_result:
                        return False
            for customize_constraint in customize_constraints:
                relation_type = customize_constraint.type.value
                logger.debug(f"Relation type : {relation_type}, use strict constraint logical")
                strict_check_result = self.customize_constraint_patch.dispatch(relation_type, customize_constraint)
                if not strict_check_result:
                    logger.debug(
                        f"Relation type : {relation_type}, use strict constraint logical failed, "
                        f"check result : {strict_check_result}")
                    return False
                for param_name in customize_constraint.params:
                    self.set_has_value_param_status(param_name, relation_type)
                logger.debug(f"Relation type : {relation_type}, use strict constraint logical success")
        logger.info(f"End correct case param, operator name : {self.operator_name}")
        return True

    def generate_dtype_string_domain(self, param_name: str) -> List[str]:
        """
        获取数据类型dtype可取值，用列表表示
        :param param_name: 参数名称
        :return: List[dtype]
        """
        parameter_constraints = self.operator_rule_data.parameter_constraints
        for param_constraint in parameter_constraints:
            if param_constraint.name != param_name:
                continue
            param_dtype_list = param_constraint.constraints.data_types
            choose_param_dtype = random.choice(param_dtype_list).types
            choose_param_dtype = [DataMatchMap.ACL_DTYPE_TRANSFER_TENSOR_MAP.get(type_str) for type_str in
                                  choose_param_dtype]
            dtype_domain = choose_param_dtype
            return dtype_domain

        return []

    def generate_param_range_value_domain(self, param_name: str) -> List:
        """
        获取数据取值的取值范围可取值，用列表表示，只获取只有离散枚举值的值域
        :param param_name:参数名称
        :return: List
        """
        parameter_constraints = self.operator_rule_data.parameter_constraints
        range_value_domain = []
        for param_constraint in parameter_constraints:
            if param_constraint.name != param_name:
                continue
            param_range_values = param_constraint.constraints.allowed_values
            for param_range_value in param_range_values:
                for value in param_range_value.value:
                    if isinstance(value, List):
                        return []
                    range_value_domain.append(value)
        return range_value_domain

    @staticmethod
    def split_object_attribute(expression):
        """
        Split an object attribute expression into object name and attribute name.
        Args:
            expression: String like "x.shape", "x.dtype", or "weight"
        Returns:
            dict: {'has_attribute': bool, 'object': str, 'attribute': str or None}
        """
        if not isinstance(expression, str):
            logger.error("Expression is not a string, can't be split to object and attribute")
            return False, None, None

        if '.' in expression:
            obj_name, attr_name = expression.split('.', 1)
            return True, obj_name, attr_name
        else:
            return False, expression, None

    @staticmethod
    def is_shape_value_match_rule(param_name: str, shape_value: List[int], shape_rule: str):
        """
        判断当前shape的取值是否满足shape的约束条件
        :param param_name: 参数名称
        :param shape_value: shape取值
        :param shape_rule: shape的约束表达式
        """
        shape_value_expr = f"{param_name}.shape == {shape_value}"
        try:
            builder = Z3ConstraintBuilder()
            builder.add_constraint(shape_rule)
            builder.add_constraint(shape_value_expr)
            check_result = builder.solver.check()
            if check_result == z3.sat:
                return True
            return False
        except Exception as e:
            logger.error(f"Valid shape value expr failed, err msg : {str(e)}")
            return False

    def solve_param_has_determine_value(self, relation_params):
        """
        查找是否有参数的属性取值已确定，如果有，则添加为求解条件
        添加规则：(1)对于条件表达式涉及到的所有参数，每个参数的dtype，shape，range_value属性，如果某个属性的取值已在
        self.has_set_value_param中，则将其添加为求解条件表达式，即param.shape=[x,x,x]
        (2) 对于range_value，如果range_value是浮点数或整数，则直接加入求解条件表达式，否则不加入求解表达式
        (3) 对于shape,如果该参数的parameter_constraints中shape约束乜有axis_value的约束表达式，则不添加确定值的约束，避免约束冲突，
        此时将axis_value中的约束表达式添加至列表中，如果没有axis_value的约束表达式，则将shape当前取值添加至表达式列表
        (4) 如果没有一个参数的取值已经确定，则将relation_params中的第一个参数取值作为确定取值，构建求解条件表达式，以此保留建模信息
        :param relation_params: 这一组表达式涉及到的所有参数名称
        :return: [str]所有构建的条件表达式
        """
        determine_value_expr_list = []
        dtype_exp_list = []
        shape_expr_list = []
        range_value_expr_list = []
        if not relation_params:
            logger.error("Relation params is empty, can't solve param has determine value or not")
            return determine_value_expr_list
        parameter_constraints_dict = {each.name: each.constraints for each in self.operator_rule_data.parameter_constraints}
        has_add_shape_value_rule = False
        for param_name in relation_params:
            # 部分参数可能不在case_input_map中，需要在other_parameter中查找，将other中此参数的rule字段内容添加至约束集合
            if param_name not in self.case_input_map:
                if param_name not in self.other_input_map:
                    logger.error(f"Can't match this parameter, param name : {param_name}")
                    continue
                param_rule = self.other_input_map.get(param_name)
                for constraint in param_rule.constraints:
                    param_rule = constraint.rule
                    if param_rule:
                        range_value_expr_list.append(param_rule)
                continue
            # dtype属性约束值添加
            if self.has_set_value_param.get(param_name) and self.has_set_value_param.get(param_name).dtype:
                param_dtype = self.case_input_map.get(param_name).dtype
                dtype_exp_list.append(f"{param_name}.dtype == '{param_dtype}'")
            # shape属性约束值添加
            # 需要把shape的当前取值添加到约束里，需要先判断是否满足rule规则，借助ast判断
            shape_current_value = self.case_input_map.get(param_name).shape
            is_shape_value_valid = True
            param_shape_constraints = parameter_constraints_dict.get(param_name).shape
            for shape_constraint in param_shape_constraints:
                for shape_rule in shape_constraint.constraint:
                    shape_expr_list.append(shape_rule.rule)
                    is_shape_value_valid = is_shape_value_valid and self.is_shape_value_match_rule(param_name,
                                                                                                   shape_current_value,
                                                                                                   shape_rule.rule)
            if is_shape_value_valid and self.has_set_value_param.get(
                    param_name) is not None and self.has_set_value_param.get(param_name).shape:
                shape_expr_list.append(f"{param_name}.shape == {shape_current_value}")
                has_add_shape_value_rule = True
            # range_value属性约束添加
            param_range_value_constraint = parameter_constraints_dict.get(param_name).allowed_values
            for range_value_constraint in param_range_value_constraint:
                for value_rule in range_value_constraint.value:
                    if isinstance(value_rule, list) and len(value_rule) >= 2:
                        range_value_expr_list.append(f"{param_name} > {value_rule[0]}")
                        range_value_expr_list.append(f"{param_name} < {value_rule[1]}")
        first_param = relation_params[0]
        if not dtype_exp_list:
            dtype_exp_list.append(f"{first_param}.dtype == '{self.case_input_map.get(first_param).dtype}'")
        if not has_add_shape_value_rule:
            shape_expr_list.append(f"{first_param}.shape == {self.case_input_map.get(first_param).shape}")
        determine_value_expr_list.extend(dtype_exp_list)
        determine_value_expr_list.extend(shape_expr_list)
        determine_value_expr_list.extend(range_value_expr_list)
        return determine_value_expr_list

    def solve_constraint_by_z3(self, constraint_expr: InterParamConstraint):
        """
        基于Z3求解器处理所有的约束条件，输出满足约束条件的参数属性的取值，如果相关参数中包含output角色的参数，则放弃该条约束，不处理
        :param constraint_expr: 约束条件数据
        :return: 满足条件的参数属性
        """
        logger.info(f"Start solving solution of constraints by Z3, operator name : {self.operator_name}")
        logger.debug(f"Constraint expression : {constraint_expr.__dict__}")
        solver = Z3ConstraintBuilder()
        relation_param = constraint_expr.params
        expr_list = self.solve_param_has_determine_value(relation_param)
        # 替换表达式中字符串名称，规范化为小写的统一格式
        for key, value in DataMatchMap.ACL_DTYPE_TRANSFER_TENSOR_MAP.items():
            if isinstance(constraint_expr.expr, str):
                constraint_expr.expr = constraint_expr.expr.replace(key, value)
        expr_list.append(constraint_expr.expr)
        for param_name in relation_param:
            dtype_domain = self.generate_dtype_string_domain(param_name)
            if dtype_domain:
                solver.declare_var(param_name, allowed_dtypes=dtype_domain)
        solver.add_constraints(expr_str_list=expr_list)
        solver_result = solver.solve()
        if not solver_result:
            logger.error(
                f"Z3 solver error, no solution can satisfy constraints, operator name : {self.operator_name}")
            return False
        logger.info(f"End solving solution of constraints by Z3, operator name : {self.operator_name}")
        for param_name, property_dict in solver_result.items():
            for attr_name, attr_value in property_dict.items():
                # 如果self.has_set_value_param中该参数的属性已被设置为True，表示该参数已被设置过，不再修改case_input中的值
                param_set_status = self.has_set_value_param.get(param_name)
                if param_set_status is None or (not getattr(self.has_set_value_param.get(param_name), attr_name)):
                    self.case_input_map[param_name].__setattr__(attr_name, attr_value)
            self.set_has_value_param_status(param_name, constraint_expr.type)
        return True
