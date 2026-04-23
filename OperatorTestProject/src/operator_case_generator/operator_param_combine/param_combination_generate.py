# -*- coding: UTF-8 -*-
"""
功能：基于结构化数据以及参数语义角色确定参数具体取值或使用模型名称
"""
import ast
import random
import re
from typing import List

from common_utils.logger_util import get_logger
from data_definition.constants import DataMatchMap, ParamModelConfig
from data_definition.param_models_def import OperatorParameterCombination, ParameterPropertyData, \
    ParameterShapeProperty, ShapePropertyConstraintStructure
from param_constraint_solve.expression_preprocess_utils import ShapeDimValueExtractor
from src.common_model_definition import OperatorRule, ParamShape


class ParamCombinationGenerator:
    def __init__(self, operator_rule_data: OperatorRule, case_num: int = 1):
        self.operator_rule_data = operator_rule_data
        self.case_num = case_num
        self.logger = get_logger()
        self.parameters_constraints = {param.name: param for param in
                                       self.operator_rule_data.parameter_constraints}
        self.choose_dtype_map_combination = None

    def get_param_combination_input(self) -> List[OperatorParameterCombination] | None:
        """
        根据提取的算子参数约束数据(JSON)，提取参数具体信息，设置参数属性的具体设置值或属性对应模型的名称
        :param operator_constrain_data: 算子约束数据,已筛选过有效参数({opName}GetWorkspace方法下role为input的参数)
        :param case_num: 用例数量
        class ParameterPropertyData(BaseModel):
        param_name: str
        param_type: str
        shape_property: ParameterShapeProperty = None
        dtype: str
        format : str = None
        range_value_profile: str | int | float | bool
        memory_continuity: bool = False
        :return: 算子参数属性组合
        """
        if self.operator_rule_data is None:
            self.logger.error("Get param combination, input operator constraint data is None")
            return None
        self.logger.info("Start generate parameter combinations...")
        param_combination_list = []
        for _ in range(self.case_num):
            operator_parameter_combination = OperatorParameterCombination(
                operator_name=self.operator_rule_data.operation_name)
            for function in self.operator_rule_data.functions:
                for param_base_info in function.parameters:
                    param_type = DataMatchMap.ACL_TYPE_TRANSFER_ATK_MAP.get(param_base_info.type,
                                                                            ParamModelConfig.DEFAULT_ATK_TYPE)
                    param_format = None if param_base_info.format is None else random.choice(param_base_info.format)
                    param_dtype = self.generate_dtype_property(param_base_info.name)
                    param_range_value_profile = self.generate_range_value_property(param_base_info.name, param_dtype)
                    parameter_property_data = ParameterPropertyData(param_name=param_base_info.name, dtype=param_dtype,
                                                                    param_type=param_type, format=param_format,                                                                    range_value_profile=param_range_value_profile,
                                                                    is_optional=param_base_info.is_optional)
                    if param_type in ParamModelConfig.TENSOR_ATK_TYPE:
                        param_shape_property = self.generate_shape_property(param_base_info.name)
                        parameter_property_data.shape_property = param_shape_property
                        range_value_profile = DataMatchMap.PARAM_VALUE_TO_ROLE_MODEL.get(param_range_value_profile,
                                                                                         param_range_value_profile)
                        parameter_property_data.range_value_profile = range_value_profile
                    operator_parameter_combination.parameter_property.append(parameter_property_data)
            param_combination_list.append(operator_parameter_combination)
        self.logger.info("End generate parameter combinations...")
        return param_combination_list

    def generate_shape_property(self, param_name) -> ParameterShapeProperty | None:
        """
        生成参数的shape描述属性，包含shape的维度以及生成其中取值的模型名称：Has_Large_Size，Has_Size_1，Has_Odd_Size，Typical
        :return: shape属性取值，dim_count, dim_value_profile
        """
        self.logger.debug("Start generate parameter shape property...")
        if not self.parameters_constraints.get(param_name).constraints.shape:
            self.logger.error(
                f"End generate parameter shape property, param name : {param_name}, "
                f"param constraint shape is None, use default value")
            shape_property = ParameterShapeProperty(dim_count=ParamModelConfig.DEFAULT_TENSOR_SHAPE_DIM,
                                                    dim_value_profile=random.choice(
                                                        ParamModelConfig.DIM_VALUE_PROFILE_LIST))
            return shape_property
        shape_constraint_parsers = self.parse_shape_constraint(
            random.choice(self.parameters_constraints.get(param_name).constraints.shape))
        dim_count = None
        for shape_rule_parser in shape_constraint_parsers.constraint:
            if shape_rule_parser.dim_num:
                dim_count = random.choice(shape_rule_parser.dim_num)
                break

        if isinstance(dim_count, list):
            dim_count_value = random.randint(dim_count[0], dim_count[1])
        else:
            dim_count_value = dim_count
        dim_value_profile = random.choice(ParamModelConfig.DIM_VALUE_PROFILE_LIST)
        shape_property = ParameterShapeProperty(dim_count=dim_count_value, dim_value_profile=dim_value_profile)
        self.logger.debug(
            f"End generate parameter shape property, param name : {param_name}, shape property : {shape_property}")
        return shape_property

    def parse_shape_constraint(self, shape_constraint: ParamShape) -> ParamShape:
        """
        解析tensor.shape的约束字符串
        :param shape_constraint: 参数shape的约束
        :return: 解析之后的shape约束，包含[dim_min_count, dim_max_count]或fix_dim_count
        """
        self.logger.debug("Start generate parameter shape dim parser...")
        shape_dim_value_extractor = ShapeDimValueExtractor()
        for shape_rule in shape_constraint.constraint:
            self.logger.info(f"shape_rule.dim_num value is {shape_rule.dim_num}")
            if shape_rule.structure == ShapePropertyConstraintStructure.DIMS.value:
                if shape_rule.dim_num:
                    shape_dim_num = []
                    for each_dim_num in shape_rule.dim_num:
                        if isinstance(each_dim_num, int):
                            shape_dim_num.append([each_dim_num, each_dim_num])
                        elif isinstance(each_dim_num, List) and len(each_dim_num) == 1:
                            shape_dim_num.append(each_dim_num * 2)
                        elif isinstance(each_dim_num, List) and len(each_dim_num) == 2:
                            shape_dim_num.append(each_dim_num)
                    if shape_dim_num:
                        self.logger.info(f"Get shape dim num from shape_rule.dim_num, result : {shape_dim_num}")
                        shape_rule.dim_num = shape_dim_num
                        return shape_constraint
                shape_rule_rule = shape_rule.rule
                self.logger.debug(f"Shape dim rule : {shape_rule_rule}")
                # ShapeDimValueExtractor.extract的返回结果格式为[{'min':v1, 'max':v2},['min':v3, 'max':v3]]
                shape_dim_extract_result = shape_dim_value_extractor.extract(shape_rule_rule)
                if shape_dim_extract_result is None:
                    return shape_constraint
                shape_rule.dim_num = []
                for shape_dim_range in shape_dim_extract_result:
                    if shape_dim_range is None:
                        continue
                    shape_dim_min = shape_dim_range.get("min")
                    shape_dim_max = shape_dim_range.get("max")
                    dim_min_count = shape_dim_min if shape_dim_min is not None \
                        else ParamModelConfig.DEFAULT_TENSOR_SHAPE_DIM_MIN
                    dim_max_count = shape_dim_max if shape_dim_max is not None \
                        else ParamModelConfig.DEFAULT_TENSOR_SHAPE_DIM_MAX
                    self.logger.debug(f"Shape dim min : {dim_min_count}, dim max : {dim_max_count}")
                    shape_rule.dim_num.append([dim_min_count, dim_max_count])
        self.logger.debug(f"End generate parameter shape dim parser, param shape len : {shape_constraint.__dict__}")
        return shape_constraint

    def generate_dtype_property(self, param_name: str) -> str:
        """
        选择参数的数据类型,如果dtype_map不为空，则在dtype_map中选择一组数据类型作为参数的数据类型，
        否则从parameter_constraints的合法值随机选择
        :param param_name: 参数名称
        :return: 数据类型
        """
        self.logger.debug("Start generate dtype property...")
        if self.operator_rule_data.dtype_map:
            dtype_map = random.choice(self.operator_rule_data.dtype_map)
            dtype_combination_num = len(dtype_map.rows)
            if self.choose_dtype_map_combination is None:
                dtype_maps_choose_index = random.randint(0, dtype_combination_num - 1)
                self.choose_dtype_map_combination = dtype_map.rows[dtype_maps_choose_index]
            if param_name not in dtype_map.columns:
                param_dtype = ParamModelConfig.DEFAULT_PARAM_DTYPE_DTYPE_IN_ORIGINAL_DOC
            else:
                param_name_index = dtype_map.columns.index(param_name)
                param_dtype = self.choose_dtype_map_combination[param_name_index]
        else:
            param_constraint = self.parameters_constraints.get(param_name)
            if not param_constraint.constraints.data_types:
                self.logger.error("Generate dtype property, param name dtype set is empty, use default data dtype")
                return ParamModelConfig.DEFAULT_PARAM_DTYPE_DTYPE_IN_ORIGINAL_DOC
            choose_dtype_set = random.choice(param_constraint.constraints.data_types)
            if not choose_dtype_set.types:
                self.logger.error("Generate dtype property, param name dtype set is empty, use default data dtype")
                return ParamModelConfig.DEFAULT_PARAM_DTYPE_DTYPE_IN_ORIGINAL_DOC
            param_dtype = random.choice(choose_dtype_set.types)
        self.logger.debug(f"End generate dtype property, param name : {param_name}, dtype : {param_dtype}")
        return param_dtype

    def generate_range_value_property(self, param_name: str, dtype: str) -> str | int | float | bool:
        """
        生成参数的取值范围属性,检查parameter_constraint.allowed_values和parameter_constraint.not_allowed_values，
        1. 如果合法取值指定的固定取值，则设置为该值，如allowed_values = [0.01]
        2. 如果合法取值指定的是取值范围，则离散化为：[ min_val ], [ max_val ], [ mid_val ], [ near_min_val ],
        [ near_max_val ], Normal. (Also include NaN if the type is float)
        3. 如果未指定任何信息：则离散化为：(Float): PosNormal, NegNormal, Zero, NaN, PosInf, NegInf, SubNormal
        (Integer): Pos, Neg, Zero, Max, Min
        :param param_name: 参数名称
        :param dtype: 数据类型
        :return: 数据取值模型名称或具体值
        """
        self.logger.debug("Start generate range_value_property...")
        parameter_constraint = self.parameters_constraints.get(param_name)
        allowed_values = parameter_constraint.constraints.allowed_values
        if allowed_values:
            select_allowed_value = random.choice(random.choice(allowed_values).value)
            if isinstance(select_allowed_value, list):
                range_value_profile_list = [select_allowed_value[0], select_allowed_value[1],
                                            (select_allowed_value[0] + select_allowed_value[1]) / 2,
                                            select_allowed_value[0] + 0.01, select_allowed_value[1] - 0.01]
            else:
                range_value_profile_list = [select_allowed_value]
        else:
            if DataMatchMap.ACL_DTYPE_TRANSFER_TENSOR_MAP.get(dtype) in ParamModelConfig.FLOAT_DTYPE:
                range_value_profile_list = ParamModelConfig.FLOAT_TENSOR_DATA_PROFILE
            else:
                range_value_profile_list = ParamModelConfig.INT_TENSOR_DATA_PROFILE
        range_value_profile = random.choice(range_value_profile_list)
        self.logger.debug(
            f"End generate range_value_property, param name : {param_name}, value profile : {range_value_profile}")
        return range_value_profile
