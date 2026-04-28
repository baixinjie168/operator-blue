# parameter_constraints 检验规则

本文件用于同时校验 `parameter_constraints` 的数据格式和业务语义。

`parameter_constraints` 来源于 `src/common_model_definition.py` 中 `OperatorRule` 的一个属性，但本提示词不能依赖外部 Python 文件内容，因此相关模型定义直接内联如下。

## 模型定义

### 顶层字段来源

```python
class OperatorRule(BaseModel):
    parameter_constraints: List["ParamConstraints"] = Field(..., description="参数约束列表")
```

### 相关枚举

```python
class ShapeStructure(str, Enum):
    DIMS = "dims"
    AXIS_VALUE = "axis_value"
```

### parameter_constraints 相关模型

```python
class ShapeRule(BaseModel):
    structure: ShapeStructure
    rule: str
    dim_num: List[List[int]] | List[int] = None
    dim_valid_value: List[List[int]] | List[int] = None
    dim_invalid_value: List[List[int]] | List[int] = None

    model_config = {"extra": "forbid"}


class ParamShape(BaseModel):
    platform: str
    constraint: List[ShapeRule]

    model_config = {"extra": "forbid"}


class ParamDataType(BaseModel):
    platform: str
    types: List[str]

    model_config = {"extra": "forbid"}


class ParamMemory(BaseModel):
    platform: str
    discontinuous: bool

    model_config = {"extra": "forbid"}


class ParamValue(BaseModel):
    platform: str
    value: List[List[float]] | List[float] | List[List[int]] | List[int] | List[str] | List[bool] = []

    model_config = {"extra": "forbid"}


class SingleParamConstraints(BaseModel):
    shape: List[ParamShape]
    data_types: List[ParamDataType]
    memory: Optional[List[ParamMemory]]
    allowed_values: Optional[List[ParamValue]]
    not_allowed_values: Optional[List[ParamValue]]

    model_config = {"extra": "forbid"}


class ParamConstraints(BaseModel):
    name: str
    constraints: SingleParamConstraints

    model_config = {"extra": "forbid"}
```

## 校验目标

本文件同时做两类校验：

### 1. 格式校验

检查以下内容：
- JSON 是否合法
- 字段名是否正确
- 层级结构是否正确
- 必填字段是否存在
- 字段值的数据类型是否正确
- 枚举字段取值是否在允许范围内
- 是否存在模型未定义的额外字段

### 2. 业务校验

检查以下内容：
- 是否与算子说明文档中的 `parameter_constraints` 约束一致
- `shape` 是否正确表达了文档中的 shape 约束
- `data_types` 是否正确表达了文档中的数据类型约束
- `memory` 是否正确表达了文档中的非连续 Tensor 约束
- `allowed_values` / `not_allowed_values` 是否正确表达了文档中的取值限制
- 不同平台的约束是否被正确拆分和表达

## 校验流程

按以下顺序执行：
1. 先做格式校验
2. 格式校验通过后，再做业务校验
3. 任意一类校验失败，都应判定为不通过

## 顶层结构

- `parameter_constraints` 必须是数组
- 数组中的每个元素都必须是对象
- 每个元素只允许包含以下字段：
- `name`
- `constraints`

字段要求：
- `name`：字符串
- `constraints`：对象

业务要求：
- `name` 应对应算子文档中的参数名
- 不应凭空生成文档中不存在的参数约束项

## constraints 对象结构

`constraints` 只允许包含以下字段：
- `shape`
- `data_types`
- `memory`
- `allowed_values`
- `not_allowed_values`

字段要求：
- `shape`：数组
- `data_types`：数组
- `memory`：数组或 `null`
- `allowed_values`：数组或 `null`
- `not_allowed_values`：数组或 `null`

业务要求：
- 各字段应分别承载对应类型的约束，不要混用。
- 文档明确给出的约束类型，应出现在对应字段中。
- 不要把 dtype、format、memory、平台说明写进 `shape.rule`；也不要把 shape / 关系规则塞进 `data_types` 或 `value` 字段。
- 如果文档在某个参数下明确写了“与另一参数一致”“由另一参数推导”等关系，校验时应确认结果没有只保留 rank 或类型枚举而丢失关键约束语义。

## shape 校验

### 格式校验

- `shape` 中的每个元素都必须是对象
- 每个元素只允许包含以下字段：
- `platform`
- `constraint`

字段要求：
- `platform`：字符串
- `constraint`：数组
- `platform` 必须校验为字符串；如果出现字符串数组，应直接报结构错误。多个平台约束不同应拆分，平台名也要尽量保持文档原文。

`constraint` 数组中的每个元素都必须是对象，且只允许包含以下字段：
- `structure`
- `rule`
- `dim_num`
- `dim_valid_value`
- `dim_invalid_value`

字段要求：
- `structure`：只能是 `"dims"` 或 `"axis_value"`
- `rule`：字符串
- `dim_num`：`null`、整数数组，或二维整数数组
- `dim_valid_value`：`null`、整数数组，或二维整数数组
- `dim_invalid_value`：`null`、整数数组，或二维整数数组
- `rule` 不允许为 `null`；如果某个约束元素没有表达式内容，也只能使用空字符串 `""`

### 业务校验

- 从算子说明文档的“参数说明”表格中提取每个参数的“维度(shape)”列信息。
- 检查 JSON 中的 `shape` 约束是否与文档一致。
- 同一平台下，`constraint` 数组中不能出现多个相同 `structure` 的元素。
- 如果同一平台下存在多个相同 `structure` 的条件，应该合并为一个元素，并在 `rule` 中组合表达。
- `dims` 仅用于表达 shape 的维度数量约束。
- `axis_value` 仅用于表达 shape 各维取值约束以及由其他参数推导出的轴值关系。
- 如果文档明确写了 `out.shape == x.shape`、`shape 为 [B,H] 或 [B,1,H]`、某一维与另一参数某一维相等等关系，校验时必须确认这些关系被完整表达，而不是仅保留 rank 或部分轴值。
- 对 `dim`、`pad`、`keepDim`、`minOut/maxOut`、`out` 等常见参数，要检查是否完整表达与输入 shape 的联动关系，而不是只校验 rank 或一个局部轴。
- 不同平台的 shape 约束不同，需要拆成多个 `platform` 元素；所有平台一致时才允许合并。
- `shape.rule` 中不应混入 dtype、format、memory 或自然语言长段说明。
- `shape.rule` 只能承载 rank、轴值、broadcast、shape 推导等 shape 语义；混入 dtype、format、memory、平台说明或自然语言长句都应报错。
- `shape.rule` 不能为 `null`；没有内容时应写成空字符串 `""`。
- 对 `workspaceSize`、`executor`、`workspace`、`stream` 等流程参数，若文档没有 tensor shape 约束，`shape` 应为空结构，不要套用 Tensor 约束。
- 当参数名与 Python 内置函数冲突，如 `max`、`min`，应检查表达式是否产生歧义；含糊使用 `max(...)` / `min(...)` 的规则应要求改写。

常见业务错误：
- 文档中有 shape 约束，但 JSON 中缺失。
- 同一平台下重复出现多个 `dims` 或多个 `axis_value`。
- `dims` 和 `axis_value` 的职责混用。
- 文档中的关键跨参数 shape 对应关系没有体现在 `rule` 中。
- 文档中按平台区分约束，但 JSON 没有拆分平台。

## data_types 校验

### 格式校验

- `data_types` 中的每个元素都必须是对象
- 每个元素只允许包含以下字段：
- `platform`
- `types`

字段要求：
- `platform`：字符串
- `types`：字符串数组

### 业务校验

- 从算子说明文档的“参数说明”表格中提取每个参数的“数据类型”列信息。
- 检查 JSON 中的 `data_types` 是否与文档一致。
- 如果文档中不同平台支持的数据类型不同，需要拆成多个 `platform` 元素。
- 如果文档中所有平台支持的数据类型相同，可以合并为统一的平台描述。
- `types` 中应只保留文档明确支持的类型，不应凭空添加。
- `data_types.types` 只能是 dtype 名称；出现 `COND:`、`can_cast(...)`、`self.dtype == out.dtype`、中文关系说明或其他逻辑表达式时，应判定为字段污染。
- 数据类型名称应规范、统一，避免同义写法混杂。
- 如果文档除枚举外还明确要求“与另一参数同 dtype”，校验时应确认结果没有把该显式关系完全丢掉；不能只留下类型枚举而忽略关键对齐语义。
- 如果文档明确写了 dtype 推导、同 dtype、可转换、复数输入输出必须为复数等关系，不能只接受类型枚举；应要求结果在合适位置补充跨参数 dtype 关系。

常见业务错误：
- 文档中支持的类型没有写全。
- 文档中不支持的类型被错误加入。
- 不同平台的数据类型约束没有拆分。
- 只保留了类型列表，但遗漏了文档明确给出的 dtype 对齐要求。

## memory 校验

### 格式校验

- `memory` 可以是 `null` 或数组
- 如果是数组，数组中的每个元素都必须是对象
- 每个元素只允许包含以下字段：
- `platform`
- `discontinuous`

字段要求：
- `platform`：字符串
- `discontinuous`：布尔值

### 业务校验

- 从算子说明文档的“参数说明”表格中提取“非连续 Tensor”相关信息。
- 如果文档明确支持非连续 Tensor，应设置为 `discontinuous: true`。
- 如果文档明确不支持非连续 Tensor，应设置为 `discontinuous: false`。
- 如果不同平台的非连续 Tensor 支持情况不同，需要拆成多个 `platform` 元素。
- 如果文档没有提供该类约束，可使用 `null` 或空数组表达“无明确约束”。
- `discontinuous` 必须是布尔值，不接受字符串形式的 `"True"` / `"False"`。

常见业务错误：
- 文档明确支持非连续 Tensor，但 JSON 写成 `false`。
- 文档明确不支持非连续 Tensor，但 JSON 写成 `true`。
- 不同平台差异未拆分。
- 把布尔值错误写成字符串。

## allowed_values / not_allowed_values 校验

### 格式校验

- `allowed_values` 和 `not_allowed_values` 都可以是 `null` 或数组
- 如果是数组，数组中的每个元素都必须是对象
- 每个元素只允许包含以下字段：
- `platform`
- `value`

字段要求：
- `platform`：字符串
- `value`：以下格式之一
- 基础类型数组：整数数组、浮点数数组、字符串数组、布尔值数组
- 二维数值数组：二维整数数组或二维浮点数数组

### 业务校验

- 从算子说明文档中提取参数的取值限制信息
- 允许的取值应写入 `allowed_values`
- 禁止的取值应写入 `not_allowed_values`
- 如果文档中未给出取值限制，可使用 `null` 或空数组
- 如果某个平台项已经生成了 `allowed_values` 记录，但文档没有读到明确的取值约束，则其中的 `value` 应为 `[]`。
- 数值型限制应尽量规范表达：
- 单值：如 `[1e-5]`
- 区间：如 `[[0, 100]]`
- 多段区间：如 `[[0, 7168], [8192, 10240]]`
- 如果文档使用中文枚举值或文本值，应在 `value` 中保留对应字符串
- 如果不同平台有不同取值限制，需要拆成多个 `platform` 元素
- 不要把 `format`、layout、数据格式名称等信息误写到 `value` 中，这些内容不属于取值约束
- `allowed_values.value` / `not_allowed_values.value` 只能是合法离散值或区间数组；`range(...)`、条件表达式、函数调用、中文说明、数据格式名都不应出现在 value 中。

常见业务错误：
- 文档中有取值限制，但 JSON 缺失
- 文档中没有取值限制，却错误生成了限制
- 允许值和禁止值放反
- 区间、多段区间表达错误
- 不同平台的取值限制没有拆分
- 文档中的关键文本值没有被保留
- 把 `format` 或 layout 信息错误写进 `allowed_values.value`

## 通用规则

- 所有对象都不允许出现未定义字段。
- 数组字段如果没有内容，应使用 `[]`。
- 只有 `memory`、`allowed_values`、`not_allowed_values` 可以使用 `null`。
- 格式校验和业务校验都应严格执行。
- 不要因为字段“看起来合理”就忽略格式错误。
- 不要因为格式合法就忽略业务错误。
- 任何来源于文档原句的约束，都应尽量保持机器可判定；如果结果只剩模糊自然语言而无法稳定判定，应视为提取质量不足。
- 对 optional 参数，校验时不仅要看是否被标记为可选，还要看其存在时的 shape / dtype / value 约束是否完整保留。

## 校验结论原则

- 同时报告格式错误和业务错误
- 如果格式不合法，直接判定为不通过
- 如果格式合法但业务内容与文档不一致，也应判定为不通过
- 只有格式和业务都通过，才能判定 `parameter_constraints` 校验通过
