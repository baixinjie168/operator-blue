# other_parameters 检验规则

本文件用于同时校验 `other_parameters` 的结构和业务语义。

`other_parameters` 来源于 `src/common_model_definition.py` 中 `OperatorRule` 的一个属性，但本提示词不能依赖外部 Python 文件内容，因此相关模型定义直接内联如下。

## 模型定义

### 顶层字段来源

```python
class OperatorRule(BaseModel):
    other_parameters: List["OtherParameters"] = Field(
        default_factory=list,
        description="其他未在'函数原型'中定义的参数"
    )
```

### 参考类型枚举

`OtherParameters.type` 在模型中定义为 `str`，不是严格枚举；但 `src/common_model_definition.py` 中提供了常见参数类型枚举，可作为结构和业务校验时的参考取值范围：

```python
class ParameterType(str, Enum):
    ACL_TENSOR = "aclTensor"
    ACL_SCALAR = "aclScalar"
    ACL_INT_ARRAY = "aclIntArray"
    ACL_FLOAT_ARRAY = "aclFloatArray"
    ACL_BOOL_ARRAY = "aclBoolArray"
    ACL_TENSOR_LIST = "aclTensorList"
    ACL_SCALAR_LIST = "aclScalarList"
    ACL_OP_EXECUTOR = "aclOpExecutor"
    ACL_RT_STREAM = "aclrtStream"
    DOUBLE = "double"
    BOOL = "bool"
    UINT64_T = "uint64_t"
    INT64_T = "int64_t"
    INT32_T = "int32_t"
    FLOAT = "float"
    FLOAT16 = "float16"
    VOID = "void"
    SIZE_T = "size_t"
```

### other_parameters 相关模型

```python
class OtherParameterConstraint(BaseModel):
    platform: str
    value: List[List[float]] | List[float] | List[List[int]] | List[int] | List[str] | List[bool] = []
    rule: str = ""

    model_config = {"extra": "forbid"}


class OtherParameters(BaseModel):
    name: str
    type: str
    description: str = None
    constraints: List[OtherParameterConstraint]

    model_config = {"extra": "forbid"}
```

## 校验目标

本文件同时做两类校验：

### 1. 结构校验

检查以下内容：
- JSON 是否合法
- 字段名是否正确
- 层级结构是否正确
- 必填字段是否存在
- 字段值的数据类型是否正确
- `value` 的数组形态是否符合模型定义
- 是否存在模型未定义的额外字段

### 2. 业务校验

检查以下内容：
- 是否正确识别了文档中“未在函数原型中定义，但被用作参数约束或说明符号”的参数
- 是否错误把真实函数参数、一次性中间表达式或纯说明性文本当成 `other_parameters`
- `name`、`type`、`description` 是否与文档中的真实含义一致
- `constraints` 是否正确表达了平台差异、取值限制和关系规则
- `rule` 与 `value` 是否各自承载了正确的信息，没有混用或遗漏

## 校验流程

按以下顺序执行：
1. 先做结构校验
2. 结构校验通过后，再做业务校验
3. 任意一类校验失败，都应判定为不通过

## 顶层结构校验

- `other_parameters` 必须是数组
- 数组中的每个元素都必须是对象
- 如果文档中没有任何符合定义的额外参数，则该字段应为 `[]`

业务要求：
- `other_parameters` 只包含“未在函数原型中定义”的额外参数
- 已经出现在 `functions.parameters` 或函数原型中的真实入参、出参，不应出现在这里
- 校验时先确认该名字是否已经出现在函数原型或 `functions.parameters`；真实函数参数及其别名不应重复进入 `other_parameters`。
- 不要凭空生成文档中不存在的额外参数

## OtherParameters 结构校验

- 每个元素只允许包含以下字段：
- `name`
- `type`
- `description`
- `constraints`

字段要求：
- `name`：字符串
- `type`：字符串
- `description`：字符串或 `null`
- `constraints`：数组

补充约束：
- 所有对象都不允许出现未定义字段。
- `constraints` 可以为空数组，但不能是对象、字符串或 `null`。
- `constraints` 中如果出现 `rule` 字段，其值必须是字符串；没有关系规则时也应为 `""`，不允许为 `null`。

业务要求：
- `name` 应是文档中被显式命名、可独立引用的额外参数名，例如维度符号、计数符号、辅助标识符。
- 不要把完整公式、表达式片段、平台名或自然语言句子写入 `name`。
- 同一个额外参数应只出现一次；如存在多平台差异，应在同一参数下拆到多个 `constraints` 项中。
- `description` 只负责解释“这个参数是什么”，不应承担 shape / dtype / rank / 取值关系。
- 如果文档中的关系被写进 `description`，而 `rule` 为空，应判定为职责分工错误。

## type 校验

### 结构校验

- `type` 必须是字符串

### 业务校验

### 1. 识别 other_parameters 的来源

- 优先从“功能说明”“约束说明”“参数说明中的使用说明”“公式说明”“shape 说明”等位置识别额外参数。
- 重点关注文档中被单独命名、但没有出现在函数原型中的符号，例如维度变量、分组数、隐藏维、分片因子等。
- 如果某个符号只是公式里的临时中间量，且没有独立参数意义，不应提取为 `other_parameters`。
- 只有文档明确把该符号当作稳定约束符号使用时，才应保留。
- 示例 shape 里的 `A/B/C/D`、公式临时变量、说明性占位符，如果没有被文档定义为后续约束可引用的中间符号，应判定为误提取。
- 文档明确写出“r 为输入数据的维度”等函数原型之外的中间符号时，应检查是否漏提；这类符号要能被其他约束稳定引用。

### 2. 与 functions 的边界

- `other_parameters` 与 `functions.parameters` 不能重复承载同一个真实参数。
- 真正出现在 API 原型里的输入、输出、workspace、stream、executor 等参数，应保留在 `functions`。
- `other_parameters` 只负责承载函数签名之外、但文档又明确约束或引用的额外参数。

### 3. 参数去重与合并

- 同一个 `name` 不要重复生成多个顶层元素。
- 平台差异应放在同一参数的 `constraints` 列表中，不要为不同平台重复创建同名参数。
- 同一平台下语义完全重复的约束不要重复出现。
- 如果规则需要额外中间符号才能表达，应先确认该符号被正确补齐，而不是把关系粗暴省略。

### 4. 描述与约束分工

- `description` 负责解释“这个参数是什么”。
- `rule` 负责表达“这个参数与其他量的关系”。
- `value` 负责表达“这个参数允许取什么值或范围”。
- 不要把三者混写到同一个字段中。
- `rule` 不允许为 `null`；没有关系规则时应写成空字符串 `""`。
- 对保留下来的额外参数，要校验 `type`、`description`、`constraints.rule/value` 是否各司其职；不要把关系规则写进 `value`，也不要把纯取值范围写成自然语言规则。
- 如果文档中同时存在多种 rank / 输入输出形态，应检查 `rule` 是否完整覆盖，而不是只保留某一种场景。

## 通用规则

- 所有对象都不允许出现未定义字段
- 不要生成空名字参数、重复名字参数或明显不是参数名的条目
- 不要把平台、值域、关系规则混到错误字段里
- 如果文档没有任何符合条件的额外参数，则 `other_parameters` 必须为 `[]`

## 校验结论原则

- 结构校验和业务校验都应严格执行
- 结构不合法时，直接判定不通过
- 结构合法但业务语义错误时，也应判定不通过
- 只有结构和业务都通过，才能判定 `other_parameters` 校验通过
