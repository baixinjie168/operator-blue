# -*- coding: UTF-8 -*-
"""
功能：算子参数主函数入口
"""
import argparse
import os.path
import time
from typing import List, Optional, Union

from common_utils.data_handle_utils import DataHandleUtil
from common_utils.logger_util import LazyLogger, init_logger
from data_definition.constants import GlobalConfig
from data_definition.param_models_def import RunPlatform
from operator_param_combine.param_combination_generate import ParamCombinationGenerator
from operator_param_models.batch_case_generate import OperatorCaseGenerator

logger = LazyLogger()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator_constraint_directory", type=str, required=False, default=None,
                        help="Operator constraint json file directory")
    parser.add_argument("--operator_constraint_path", type=str, required=False, default=None,
                        help="Operator constraint json path")
    parser.add_argument("--platform", type=str, default=RunPlatform.ATLAS_A3_TRAIN_AND_INFER_SERIES.value,
                        required=False,
                        help="Test case executor environment platform")
    parser.add_argument("--case_save_path", type=str, default="output", required=False,
                        help="Case result json save path")
    parser.add_argument("--operators", type=str, default=None, required=False,
                        help="Name of the operator to be generated")
    parser.add_argument("--case_num", type=int, default=1, required=False,
                        help="Name of the operator to be generated")
    return parser.parse_args()


def _normalize_optional_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _normalize_operators(operators: Optional[Union[str, List[str]]]) -> Optional[List[str]]:
    if operators is None:
        return None
    if isinstance(operators, str):
        operators = [item.strip() for item in operators.split(",")]
    normalized_operators = [item.strip() for item in operators if item and item.strip()]
    return normalized_operators or None


def single_operator_handle(operator_constraint_path, platform=RunPlatform.ATLAS_A3_TRAIN_AND_INFER_SERIES.value,
                           case_num=1) -> List:
    """
    算子说明文档路径，若需要从网页抓取，需实现网页抓取模块。并将抓取内容保存至参数所在路径
    :param operator_constraint_path: 算子结构化规则json文件路径
    :param platform: 执行机对应的平台，”Atlas 推理系列产品“ -> Platform_G2, "Atlas A3 训练系列产品" -> Platform_G1
    :param case_num: 生成用例个数
    :return: case数据json文件
    """
    operator_case_generate = OperatorCaseGenerator()
    operator_name = os.path.splitext(os.path.basename(operator_constraint_path))[0]
    logger.info(f"Start handle operator, operator name : {operator_name}")
    constraint_data_path = operator_constraint_path
    operator_constraint_data = DataHandleUtil.handle_operator_rule_data(constraint_data_path)
    effective_operator_constraint_data = DataHandleUtil.select_effective_parameters(operator_constraint_data)
    if effective_operator_constraint_data is None:
        logger.error(f"Effective operator rule data is None, operator name : {operator_name}")
        return []
    param_combination_generator = ParamCombinationGenerator(operator_rule_data=effective_operator_constraint_data,
                                                            case_num=case_num)
    param_combination_list = param_combination_generator.get_param_combination_input()
    case_list = operator_case_generate.handle_single_operator(
        operator_constraint_data=effective_operator_constraint_data,
        param_combination_list=param_combination_list, target_platform=platform,
        case_num=case_num)
    return case_list


def batch_operator_handel(operator_constraint_directory, operators: List = None,
                          platform=RunPlatform.ATLAS_A3_TRAIN_AND_INFER_SERIES.value,
                          case_save_path=None, case_num=1):
    """
    批量处理算子
    :param operator_constraint_directory: 算子约束文件目录
    :param operators: 需要生成用例的算子的名称
    :param platform: 执行机对应平台
    :param case_save_path: 用例保存路径
    :param case_num: 生成用例的个数
    :return: None
    """
    init_logger(log_name="main")
    data_handle_utils = DataHandleUtil()
    if not os.path.exists(operator_constraint_directory):
        raise FileNotFoundError("Operator constraint directory not existed")
    file_list = os.listdir(operator_constraint_directory)
    if operators is not None and len(operators) == 0:
        logger.error("Operators is empty, no operator need to be solve")
        return
    # 默认算子识别结果的文件为json文件
    operator_constraint_file_list = [file_name for file_name in file_list if file_name.endswith(".json")]
    operator_constraint_num = len(operator_constraint_file_list)
    logger.info(f"Start handle operator constraint, operator file num : {operator_constraint_num}")
    if case_save_path is None:
        case_save_path = GlobalConfig.CASE_RESULT_SAVE_PATH
    if not os.path.exists(case_save_path):
        os.makedirs(case_save_path)
    for index, file in enumerate(operator_constraint_file_list):
        operator_constraint_path = os.path.join(operator_constraint_directory, file)
        logger.info(
            f"Start handle operator, file index : {index + 1}/{operator_constraint_num}, "
            f"operator constraint path : {operator_constraint_path}")
        operator_name, _ = os.path.splitext(file)
        if operators is not None and operator_name not in operators:
            logger.error(f"Operators is not None and {operator_name} not in operators")
            continue
        time_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
        init_logger(log_name=operator_name + "_" + time_str)

        case_list = single_operator_handle(operator_constraint_path, platform=platform,
                                           case_num=case_num)
        data_handle_utils.save_cases_to_json(api_name=operator_name, generate_case_list=case_list,
                                             json_save_path=case_save_path)
        init_logger(log_name="main")
        logger.info(
            f"End handle operator, file index : {index + 1}/{operator_constraint_num}, "
            f"operator constraint path : {operator_constraint_path}")
    logger.info(f"End handle operator constraint, operator file num : {operator_constraint_num}")


def main(operator_constraint_directory: Optional[str] = None,
         operator_constraint_path: Optional[str] = None,
         platform: Optional[str] = None,
         case_save_path: str = r"C:\Users\ttelab\code\OperatorTestProject-codex\workspace_2\aclnnAdaLayerNorm\output",
         operators: Optional[Union[str, List[str]]] = None,
         case_num: int = 1):
    operator_constraint_directory = _normalize_optional_text(operator_constraint_directory)
    operator_constraint_path = _normalize_optional_text(operator_constraint_path)
    platform = _normalize_optional_text(platform) or RunPlatform.ATLAS_A3_TRAIN_AND_INFER_SERIES.value
    case_save_path = _normalize_optional_text(case_save_path) or "output"
    operators = _normalize_operators(operators)

    if operator_constraint_directory is None and operator_constraint_path is None:
        raise ValueError("operator_constraint_directory and operator_constraint_path cannot be empty at the same time")
    if operator_constraint_directory is not None:
        batch_operator_handel(operator_constraint_directory=operator_constraint_directory, operators=operators,
                              platform=platform, case_save_path=case_save_path, case_num=case_num)
    else:
        operator_name, _ = os.path.splitext(os.path.basename(operator_constraint_path))
        time_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
        init_logger(log_name=operator_name + "_" + time_str)
        case_list = single_operator_handle(operator_constraint_path=operator_constraint_path,
                                           platform=platform, case_num=case_num)
        DataHandleUtil.save_cases_to_json(api_name=operator_name, generate_case_list=case_list,
                                          json_save_path=case_save_path)


def cli_main():
    args = parse_args()
    main(operator_constraint_directory=args.operator_constraint_directory,
         operator_constraint_path=args.operator_constraint_path,
         platform=args.platform,
         case_save_path=args.case_save_path,
         operators=args.operators,
         case_num=args.case_num)


if __name__ == '__main__':
    cli_main()
