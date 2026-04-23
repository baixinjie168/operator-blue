# -*- coding: UTF-8 -*-
"""
功能：数据处理相关功能
"""
import json
import os
from typing import List

from pydantic import ValidationError

from atk_common_utils.case_config import CaseConfig
from common_utils.logger_util import LazyLogger
from data_definition.constants import GlobalConfig
from data_definition.param_models_def import RunPlatform
from src.common_model_definition import OperatorRule, ParameterRole

logger = LazyLogger()


class DataHandleUtil:

    @staticmethod
    def save_cases_to_json(api_name, generate_case_list: List[CaseConfig], json_save_path):
        """
        保存生成的case数据为JSON文件
        :param api_name: api名称，用来确认json文件名称
        :param generate_case_list: 将生成的case数据保存为json
        :param json_save_path: json保存路径
        :return: None
        """
        logger.info(f"Start save case json, api name : {api_name}")
        case_config_json_list = []
        for case_config in generate_case_list:
            case_config_json = case_config.model_dump()
            case_config_json_list.append(case_config_json)
        if not os.path.exists(json_save_path):
            os.makedirs(json_save_path)
        json_save_file = os.path.join(json_save_path, api_name + ".json")
        with open(json_save_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(case_config_json_list, ensure_ascii=False, indent=4))
        logger.info(f"End save case json, api name : {api_name}")

    @staticmethod
    def handle_operator_rule_data(operator_rule_file_path: str) -> OperatorRule | None:
        """
        从约束数据中获取所有约束信息，即inter_parameter_constraints中的数据
        :param operator_rule_file_path: 约束数据rule.json路径
        :return: 所有的参数约束关系数据对象: OperatorRule
        """
        if not os.path.exists(operator_rule_file_path):
            logger.error(f"Operator constraint data file is not find, file path : {operator_rule_file_path}")
            return None
        with open(operator_rule_file_path, "r", encoding="utf-8") as f:
            operator_rule_data = json.load(f)
        try:
            operator_rule_instance = OperatorRule(**operator_rule_data)
            for function in operator_rule_instance.functions:
                if GlobalConfig.PROCESS_FUNCTION_MARKER not in function.api_name:
                    continue
                for param_info in function.parameters:
                    if param_info.format == "None":
                        param_info.format = None
        except ValidationError as e:
            logger.error(f"Operator constraint data type validation, err msg : {str(e)}")
            operator_rule_instance = None
        return operator_rule_instance

    @staticmethod
    def select_effective_parameters(operator_constraint_data: OperatorRule,
                                    target_platform: str = RunPlatform.DEFAULT_PLATFORM.value) -> OperatorRule | None:
        """
        筛选有效的参数进行处理，1. 只处理{OpName}GetWorkspaceSize方法中，role为input的参数; 2. 根据执行平台选择对应的约束信息;
        :param operator_constraint_data: 原始结构化数据
        :param target_platform: 算子运行平台设备类型
        :return: 筛选有效数据之后约束数据
        """
        logger.debug("Start select effective parameters")
        if not operator_constraint_data:
            logger.error("Operator constraint data is None")
            return None
        effective_param_name = []
        effective_functions = []
        for function in operator_constraint_data.functions:
            if GlobalConfig.PROCESS_FUNCTION_MARKER not in function.api_name:
                continue
            effective_parameters = []
            for param_info in function.parameters:
                if param_info.role == ParameterRole.INPUT.value:
                    effective_parameters.append(param_info)
                    effective_param_name.append(param_info.name)
            function.parameters = effective_parameters
            effective_functions.append(function)
        operator_constraint_data.functions = effective_functions
        effective_parameter_constraints = []
        for parameters_constraint in operator_constraint_data.parameter_constraints:
            if parameters_constraint.name not in effective_param_name:
                continue
            effective_shape_constraint = DataHandleUtil.select_constraint_by_target(
                parameters_constraint.constraints.shape, target_platform)
            parameters_constraint.constraints.shape = effective_shape_constraint
            logger.debug(f"Parameter : {parameters_constraint.name}, shape effective data select success")
            effective_dtype_constraint = DataHandleUtil.select_constraint_by_target(
                parameters_constraint.constraints.data_types, target_platform)
            parameters_constraint.constraints.data_types = effective_dtype_constraint
            logger.debug(f"Parameter : {parameters_constraint.name}, dtype effective data select success")
            effective_memory_constraint = DataHandleUtil.select_constraint_by_target(
                parameters_constraint.constraints.memory, target_platform)
            parameters_constraint.constraints.memory = effective_memory_constraint
            logger.debug(f"Parameter : {parameters_constraint.name}, memory effective data select success")
            allowed_values_constraint = DataHandleUtil.select_constraint_by_target(
                parameters_constraint.constraints.allowed_values, target_platform)
            parameters_constraint.constraints.allowed_values = allowed_values_constraint
            logger.debug(f"Parameter : {parameters_constraint.name}, allowed value effective data select success")
            not_allowed_values = DataHandleUtil.select_constraint_by_target(
                parameters_constraint.constraints.not_allowed_values, target_platform)
            parameters_constraint.constraints.not_allowed_values = not_allowed_values
            logger.debug(f"Parameter : {parameters_constraint.name}, not allowed effective data select success")
            effective_parameter_constraints.append(parameters_constraint)
        operator_constraint_data.parameter_constraints = effective_parameter_constraints
        effective_dtype_map_constraint = DataHandleUtil.select_constraint_by_target(operator_constraint_data.dtype_map,
                                                                                    target_platform)
        operator_constraint_data.dtype_map = effective_dtype_map_constraint
        logger.debug(f"Dtype map effective data select success")
        for hyper_param in operator_constraint_data.other_parameters:
            effective_hyper_param_constraint = DataHandleUtil.select_constraint_by_target(hyper_param.constraints,
                                                                                          target_platform)
            hyper_param.constraints = effective_hyper_param_constraint
            logger.debug(f"Hyper parameter : {hyper_param.name}, effective data select success")
        logger.debug("End select effective parameters")
        return operator_constraint_data

    @staticmethod
    def select_constraint_by_target(constraint_list, target_platform: str) -> List:
        """
        根据platform的类型筛选
        :param constraint_list: 约束数据列表
        :param target_platform: 算子运行平台设备类型
        :return: 符合运行平台的数据
        """
        effective_constraint_list = []
        if constraint_list is None:
            return effective_constraint_list
        for constraint in constraint_list:
            if constraint.platform != target_platform and constraint.platform != RunPlatform.ALL_SERIES.value:
                continue
            effective_constraint_list.append(constraint)
        return effective_constraint_list
