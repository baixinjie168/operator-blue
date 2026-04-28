"""
测试用例生成器 - 按照参考格式生成,只保存range_values而不保存实际tensor值
"""
import json
import random
from typing import List, Dict, Any, Optional, Tuple


class TestCaseGeneratorV2:
    def __init__(self, config_path: str):
        """初始化生成器,加载算子配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                raise ValueError(f"配置文件为空: {config_path}")
            self.config = json.loads(content)

        self.op_name = self.config.get('operation_name', 'unknown')
        self.aclnn_name = self._extract_aclnn_name()
        self.dtype_map = self.config.get('dtype_map', [])

        # 提取参数信息 - 只从GetWorkspaceSize函数中提取输入参数
        self.input_tensors, self.input_params = self._extract_parameters()

        # 提取参数间约束
        self.inter_constraints = self.config.get('inter_parameter_constraints', [])
        self.shape_equal_groups = self._build_shape_equal_groups()
        self.fixed_values = self._build_fixed_values()

    def _build_shape_equal_groups(self) -> List[List[str]]:
        """构建形状相等的参数组"""
        groups = []

        for constraint in self.inter_constraints:
            # 支持两种类型:shape_unification 和 shape_equality
            constraint_type = constraint.get('type')
            if constraint_type not in ['shape_unification', 'shape_equality']:
                continue

            params = constraint.get('params', [])

            if constraint_type == 'shape_equality':
                # shape_equality: params中的参数形状完全相同
                group = [p for p in params if p in self.input_tensors or p in self.input_params]
                if len(group) > 1:
                    groups.append(group)
            else:
                # shape_unification: 需要解析expr表达式
                expr = constraint.get('expr', '')

                if '&&' in expr:
                    # 表达式包含多个相等组,简化处理:将所有params作为一组
                    group = [p for p in params if p in self.input_tensors or p in self.input_params]
                    if len(group) > 1:
                        groups.append(group)
                else:
                    # 单个相等组
                    group = [p for p in params if p in self.input_tensors or p in self.input_params]
                    if len(group) > 1:
                        groups.append(group)

        return groups

    def _build_fixed_values(self) -> Dict[str, Any]:
        """构建固定值映射"""
        fixed_map = {}

        for constraint in self.inter_constraints:
            if constraint.get('type') == 'fixed_value':
                expr = constraint.get('expr', '')
                # 解析固定值表达式
                # 例如: epsilon == 1e-5
                if '==' in expr:
                    parts = expr.split('==')
                    if len(parts) == 2:
                        param_name = parts[0].strip()
                        value_str = parts[1].strip()
                        try:
                            # 尝试解析为浮点数(支持科学计数法)
                            value = float(value_str)
                            fixed_map[param_name] = value
                        except ValueError:
                            pass

        return fixed_map

    def _extract_aclnn_name(self) -> str:
        """从operation_name提取aclnn名称"""
        if not self.op_name:
            return "unknown"
        # 去掉aclnn前缀
        if self.op_name.startswith('aclnn'):
            return self.op_name[5:]
        return self.op_name

    def _extract_parameters(self) -> Tuple[Dict, Dict]:
        """
        从GetWorkspaceSize函数中提取输入参数信息
        只提取输入参数,不提取输出参数
        """
        input_tensors = {}
        input_params = {}

        # 查找GetWorkspaceSize函数
        get_workspace_func = None
        for func in self.config.get('functions', []):
            if 'GetWorkspaceSize' in func.get('api_name', ''):
                get_workspace_func = func
                break

        if not get_workspace_func:
            raise ValueError(f"未找到包含GetWorkspaceSize的函数: {self.op_name}")

        # 从parameter_constraints中约束信息
        parameter_constraints = {}
        for param in self.config.get('parameter_constraints', []):
            parameter_constraints[param['name']] = param

        # 从GetWorkspaceSize函数中提取输入参数
        for param in get_workspace_func.get('parameters', []):
            name = param['name']
            role = param.get('role')
            type = param.get("type")

            # 只处理输入参数
            if role == "input":
                # 获取参数约束
                constraints = parameter_constraints.get(name, {}).get('constraints', {})

                # 获取数据类型
                data_types = []
                for dtype_entry in constraints.get('data_types', []):
                    data_types.extend(dtype_entry.get('types', []))

                shape_rules = []
                for shape_entry in constraints.get('shape', []):
                    constraints = shape_entry.get('constraint', [])
                    for constraint in constraints:
                        shape_rules.append({
                            'structure': constraint.get('structure', ''),
                            'rule': constraint.get('rule', '')
                        })

                param_info = {
                    'name': name,
                    'role': role,
                    'data_types': list(set(data_types)),
                    'shape_rules': shape_rules,
                    'is_optional': param.get('is_optional', False),
                    'param_type': param.get('type', '')
                }

                if type in ["aclTensor", "aclTensorList"]:
                    input_tensors[name] = param_info
                else:
                    # 检查允许的值
                    allowed_values = {}
                    for val_entry in constraints.get('allowed_values', []):
                        allowed_values[val_entry.get('platform', "default")] = val_entry.get('value', [])
                    param_info['allowed_values'] = allowed_values
                    input_params[name] = param_info

        return input_tensors, input_params

    def _get_platforms(self) -> List[str]:
        """获取所有支持平台"""
        platforms = set()
        for dtype_entry in self.dtype_map:
            platforms.add(dtype_entry.get('platform', ''))
        return list(platforms)

    def _get_dtype_combinations_for_platform(self, platform: str) -> List[Dict[str, str]]:
        """获取指定平台的数据类型组合"""
        try:
            for dtype_entry in self.dtype_map:
                if platform in dtype_entry.get('platform', ''):
                    columns = dtype_entry.get('columns', [])
                    rows = dtype_entry.get('rows', [])
                    combinations = []
                    for row in rows:
                        combos = {col: dtype for col, dtype in zip(columns, row)}
                        combinations.append(combos)
                    return combinations
        except Exception as e:
            print(f"Get dtype combination failed, err msg : {str(e)}")
        return []

    def _map_dtype_to_pytorch(self, dtype: str) -> str:
        """将aclnn dtype映射到pytorch dtype"""
        mapping = {
            'FLOAT32': 'float32',
            'FLOAT16': 'float16',
            'BFLOAT16': 'bfloat16',
            'INT32': 'int32',
            'INT64': 'int64',
            'BOOL': 'bool',
            'DOUBLE': 'float64'
        }
        return mapping.get(dtype, dtype.lower())

    def _generate_random_shape(self, ndim_min: int = 1, ndim_max: int = 8, max_elements: int = 100) -> List[int]:
        """生成随机形状"""
        try:
            ndim = random.randint(ndim_min, ndim_max)

            shape = []
            remaining = max_elements

            for i in range(ndim):
                if i < ndim - 1:
                    dim_val = random.randint(1, min(remaining, 10))
                    shape.append(dim_val)
                    remaining = max(1, remaining // dim_val)
                else:
                    shape.append(max(1, remaining))
        except Exception as e:
            print(f"Generate random shape failed, err msg: {str(e)}")
            return []

        return shape

    def _generate_range_values(self, dtype: str) -> Any:
        """生成range_values"""
        # 对于tensor,生成范围 [min, max]
        # 对于标量,可能是固定值或范围

        dtype_lower = dtype.lower()

        if dtype_lower in ['float32', 'float16', 'bfloat16', 'float', 'double']:
            # 浮点数范围
            return [random.uniform(-10.0, -1.0), random.uniform(1.0, 10.0)]
        elif dtype_lower in ['int32', 'int64', 'int', 'bool']:
            # 整数范围
            return [random.randint(0, 100), random.randint(100, 200)]
        else:
            # 默认浮点数范围
            return [-5.0, 5.0]

    def _infer_main_input(self) -> Optional[str]:
        """推断主要输入参数(只处理GetWorkspaceSize函数中的输入参数)"""
        # 查找GetWorkspaceSize函数
        get_workspace_func = None
        for func in self.config.get('functions', []):
            if 'GetWorkspaceSize' in func.get('api_name', ''):
                get_workspace_func = func
                break

        if get_workspace_func:
            for param in get_workspace_func.get('parameters', []):
                if param.get('role') == 'computation_input' and not param.get('is_optional', False):
                    return param.get('name')

        if self.input_tensors:
            return list(self.input_tensors.keys())[0]

        return None

    def _find_shape_equal_reference(self, param_name: str, param_info: Dict) -> Optional[str]:
        """查找shape等于哪个参数"""
        try:
            name = param_info['name']
            constraints = self.config.get('parameter_constraints', {})

            for param in constraints:
                if param['name'] != name:
                    continue

                for shape_entry in param.get('constraints', {}).get('shape', []):
                    constraints = shape_entry.get('constraint', {})
                    for constraint in constraints:
                        rule = constraint.get('rule', '')
                        if '==' in rule and '.shape' in rule:
                            # 提取等号右边的参数名
                            parts = rule.split('==')
                            if len(parts) > 1:
                                ref_name = parts[1].strip().replace('.shape', '')
                                if ref_name in self.input_tensors or ref_name in self.input_params:
                                    return ref_name
        except Exception as e:
            print(f"_find_shape_equal_reference failed, err msg : {str(e)}")

        return None

    def generate_single_case(self, test_case_id: int = 0) -> Dict[str, Any]:
        """生成单个测试用例"""
        platforms = self._get_platforms()
        platform = platforms[0] if platforms else 'default'

        case = {
            'id': test_case_id,
            'name': self.op_name,
            'aclnn_name': self.aclnn_name,
            'triton_name': None,
            'version': 'v1.0',
            'expected_error_msg': None,
            'api': 'pytorch',
            'api_type': 'function_' + self.aclnn_name.lower().replace('_', '_'),
            'aclnn_api_type': 'pyaclnn_aclnn_' + self.aclnn_name.lower().replace('_', '_'),
            'triton_api_type': 'triton_function',
            'fusion_api_type': 'fusion_function',
            'fusion_mode': None,
            'dist_api_type': 'dist_function',
            'backward': False,
            'standard': {
                'acc': {},
                'perf': [0.95, 0.95]
            },
            'outputs': None,
            'inputs': []
        }
        # 获取数据类型组合
        dtype_combinations = self._get_dtype_combinations_for_platform(platform)
        dtype_dict = dtype_combinations[0] if dtype_combinations else {}

        # 维护生成的形状
        generated_shapes = {}
        processed_params = set()

        try:
            # 处理形状相等的参数组
            for group in self.shape_equal_groups:
                # 过滤出当前算子参数组中的参数
                valid_params = [p for p in group if p in self.input_tensors or p in self.input_params]

                if not valid_params:
                    continue

                # 为该组生成一个共享的形状
                # 优先组中的第一个输入tensor参数作为参考
                reference_param = None
                for param in valid_params:
                    if param in self.input_tensors:
                        reference_param = param
                        break

                if not reference_param:
                    reference_param = valid_params[0]

                # 根据参考参数生成形状
                if reference_param in self.input_tensors:
                    shape = self._generate_random_shape(1, 8, 50)
                else:
                    shape = None

                # 为组中所有参数应用相同的形状
                for param in valid_params:
                    if param in self.input_tensors:
                        generated_shapes[param] = list(shape)
                    processed_params.add(param)
        except Exception as e:
            print(f"Solve shape equality constraint failed, err msg : {str(e)}")

        # 处理独立的输入参数(不在任何相等组中)
        for param_name, param_info in self.input_tensors.items():
            if param_name in processed_params:
                continue

            if param_info.get('is_optional') and random.random() < 0.5:
                continue

            # 确定形状
            shape = None

            # 查找shape等于其他参数的约束
            ref_name = self._find_shape_equal_reference(param_name, param_info)
            if ref_name and ref_name in generated_shapes:
                shape = list(generated_shapes[ref_name])

            if not shape:
                if param_name == self._infer_main_input():
                    shape = self._generate_random_shape(1, 8, 50)
                elif generated_shapes:
                    # 使用第一个已生成形状的子集
                    ref_shape = list(generated_shapes.values())[0]
                    if ref_shape and len(ref_shape) > 0:
                        shape = ref_shape[-random.randint(1, min(3, len(ref_shape))):]

                if not shape:
                    shape = self._generate_random_shape(1, 8, 30)

            generated_shapes[param_name] = shape

            dtype = dtype_dict.get(param_name, 'FLOAT32')

            input_entry = {
                'name': param_name,
                'type': 'tensor',
                'required': not param_info.get('is_optional', False),
                'dtype': self._map_dtype_to_pytorch(dtype),
                'shape': shape,
                'range_values': self._generate_range_values(dtype),
                'backward': False,
                'align_32B': None,
                'outlier_values': None
            }

            case['inputs'].append(input_entry)

        try:
            # 生成输入标量参数(input_parameter)
            for param_name, param_info in self.input_params.items():
                dtype = param_info.get('param_type', '')
                if not dtype:
                    dtype = param_info.get('data_types', ['int'])[0]

                # 首先检查是否有固定值约束
                value = None
                if param_name in self.fixed_values:
                    value = self.fixed_values[param_name]
                else:
                    # 查找允许的值
                    allowed = param_info.get('allowed_values', {})

                    for platform_key, vals in allowed.items():
                        if platform in platform_key or not platform_key:
                            if vals:
                                value = random.choice(vals)
                                break

                    if value is None:
                        # 根据类型生成范围或值
                        if dtype.lower() in ['int64_t', 'int64', 'int32', 'int32_t']:
                            value = random.randint(0, 100)
                        elif dtype.lower() in ['double', 'float']:
                            value = random.uniform(-1.0, 1.0)
                        elif dtype.lower() in ['bool', 'boolean']:
                            value = random.choice([True, False])
                        else:
                            value = 0
        except Exception as e:
            print(f"generate scalar param failed, err msg : {str(e)}")

            input_entry = {
                'name': param_name,
                'type': 'attrs',
                'required': True,
                'dtype': 'int' if 'int' in dtype.lower() else 'float' if 'float' in dtype.lower() or 'double' in dtype.lower() else 'bool',
                'shape': None,
                'range_values': value if not isinstance(value, bool) else value,
                'backward': False,
                'align_32B': None,
                'outlier_values': None
            }

            case['inputs'].append(input_entry)

        return case

    def generate_cases(self, count: int = 10) -> List[Dict[str, Any]]:
        """生成指定数量的测试用例"""
        cases = []

        for i in range(count):
            case = self.generate_single_case(i)
            cases.append(case)

        return cases


def process_single_file(config_path: str, output_path: str, count: int) -> bool:
    """
    处理单个文件
    
    Args:
        config_path: 输入规则文件路径
        output_path: 输出JSON文件路径
        count: 生成的测试用例数量
        
    Returns:
        bool: 是否成功
    """
    import os
    
    try:
        print(f"处理文件: {config_path}")
        generator = TestCaseGeneratorV2(config_path)
        cases = generator.generate_cases(count=count)

        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # 保存结果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cases, f, ensure_ascii=False, indent=2)

        print(f"成功生成 {len(cases)} 个测试用例 -> {output_path}")
        return True

    except Exception as e:
        print(f"失败: {e}")
        return False


def process_batch_files(input_dir: str, output_dir: str, count: int) -> None:
    """
    批量处理文件
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        count: 生成的测试用例数量
    """
    import os
    import glob
    
    if not os.path.exists(input_dir):
        print(f"输入目录不存在: {input_dir}")
        return

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有JSON文件
    json_files = glob.glob(os.path.join(input_dir, '*.json'))

    print(f"找到 {len(json_files)} 个JSON文件")

    success_count = 0
    skipped_empty = 0
    failed_count = 0

    for json_file in json_files:
        filename = os.path.basename(json_file)
        output_file = os.path.join(output_dir, filename)

        try:
            # 跳过空文件
            if os.path.getsize(json_file) == 0:
                print(f"跳过空文件: {filename}")
                skipped_empty += 1
                continue

            # 检查文件内容
            with open(json_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if not content.strip():
                    print(f"跳过空内容文件: {filename}")
                    skipped_empty += 1
                    continue

            print(f"处理: {filename}")

            # 生成测试用例
            generator = TestCaseGeneratorV2(json_file)
            cases = generator.generate_cases(count=count)

            # 保存结果
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(cases, f, ensure_ascii=False, indent=2)

            print(f"  生成 {len(cases)} 个测试用例 -> {output_file}")
            success_count += 1

        except Exception as e:
            print(f"  失败: {e}")
            failed_count += 1

    print(f"\n处理完成:")
    print(f"  成功: {success_count}")
    print(f"  跳过(空文件): {skipped_empty}")
    print(f"  失败: {failed_count}")


def generate_test_cases_from_file(
    config_path: str,
    output_path: str,
    count: int = 10
) -> List[Dict[str, Any]]:
    """
    从规则文件生成测试用例(直接调用接口)
    
    Args:
        config_path: 输入规则文件路径
        output_path: 输出测试用例文件路径
        count: 生成的测试用例数量
        
    Returns:
        List[Dict[str, Any]]: 生成的测试用例列表
        
    Raises:
        FileNotFoundError: 规则文件不存在
        ValueError: 规则文件内容无效
    """
    import os
    
    # 检查输入文件是否存在
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"规则文件不存在: {config_path}")
    
    # 生成测试用例
    generator = TestCaseGeneratorV2(config_path)
    cases = generator.generate_cases(count=count)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    
    return cases


def main():
    """
    主函数 - 支持两种模式:
    1. 单文件模式: python test_case_generator.py --config <input.json> --output <output.json> [--count 10]
    2. 批量模式: python test_case_generator.py --input-dir <dir> --output-dir <dir> [--count 10]
    """
    import sys
    import argparse

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='测试用例生成器')
    parser.add_argument('--config', type=str, help='输入规则文件路径(单文件模式)')
    parser.add_argument('--output', type=str, help='输出JSON文件路径(单文件模式)')
    parser.add_argument('--input-dir', type=str, help='输入目录路径(批量模式)')
    parser.add_argument('--output-dir', type=str, help='输出目录路径(批量模式)')
    parser.add_argument('--count', type=int, default=10, help='生成的测试用例数量(默认10)')

    args = parser.parse_args()

    # 单文件模式
    if args.config and args.output:
        success = process_single_file(args.config, args.output, args.count)
        sys.exit(0 if success else 1)

    # 批量模式
    elif args.input_dir and args.output_dir:
        process_batch_files(args.input_dir, args.output_dir, args.count)

    # 默认批量模式(向后兼容)
    else:
        process_batch_files('../data/input/rule', '../data/output', 10)


if __name__ == '__main__':
    main()
