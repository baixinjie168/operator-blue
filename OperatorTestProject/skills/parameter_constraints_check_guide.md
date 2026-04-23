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

**基本提取校验**：
- 从算子说明文档的”参数说明”表格中提取每个参数的”维度(shape)”列信息。
- 检查 JSON 中的 `shape` 约束是否与文档一致。
- 同一平台下，`constraint` 数组中不能出现多个相同 `structure` 的元素。
- 如果同一平台下存在多个相同 `structure` 的条件，应该合并为一个元素，并在 `rule` 中组合表达。
- `dims` 仅用于表达 shape 的维度数量约束。
- `axis_value` 仅用于表达 shape 各维取值约束以及由其他参数推导出的轴值关系。
- 如果文档明确写了 `out.shape == x.shape`、`shape 为 [B,H] 或 [B,1,H]`、某一维与另一参数某一维相等等关系，校验时必须确认这些关系被完整表达，而不是仅保留 rank 或部分轴值。
- 不同平台的 shape 约束不同，需要拆成多个 `platform` 元素；所有平台一致时才允许合并。
- `shape.rule` 中不应混入 dtype、format、memory 或自然语言长段说明。
- `shape.rule` 不能为 `null`；没有内容时应写成空字符串 `””`。

**shape.rule / axis_value / dims 职责分离校验**：
- **shape.rule 仅承载纯 shape/轴值约束**：dtype 条件、dtype 绑定的 shape 限制须拆到专门的约束表达。
- **维度数量判断放 dims，轴值关系放 axis_value**：`len(x.shape) == 3` 等归入 `dims.rule`；`self.shape[1] == mat2.shape[0]` 等归入 `axis_value.rule`。
- **broadcast 关系拆分表达**：rank 关系放 `dims`，逐轴 broadcast 关系放 `axis_value`。输出 shape 推导应放在 `out` 的 shape 约束中。
- **reduce/归约轴的 shape 推导须完整表达**：`keepDim=true/false`、`dim=[]` 时对所有维度 reduce 等约束须在 `out` 的 shape 中显式表达。
- **条件化 shape 约束须标注生效条件**：如”非连续 Tensor 时 len(input.shape) <= 8”须在 rule 中明确条件。

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

**基本提取校验**：
- 从算子说明文档的”参数说明”表格中提取每个参数的”数据类型”列信息。
- 检查 JSON 中的 `data_types` 是否与文档一致。
- 如果文档中不同平台支持的数据类型不同，需要拆成多个 `platform` 元素。
- 如果文档中所有平台支持的数据类型相同，可以合并为统一的平台描述。
- `types` 中应只保留文档明确支持的类型，不应凭空添加。
- 数据类型名称应规范、统一，避免同义写法混杂。
- 如果文档除枚举外还明确要求”与另一参数同 dtype”，校验时应确认结果没有把该显式关系完全丢掉；不能只留下类型枚举而忽略关键对齐语义。

**规范化校验**：
- **types 仅保留合法 dtype 枚举**：删除 `COND:` 条件、`output.dtype == input.dtype`、`same_format_as_x` 等关系表达式。跨参数 dtype 关系必须用独立约束字段或扩展 schema 表达。
- **统一 FLOAT/FLOAT32 命名**：同一结构化结果中 `FLOAT` 与 `FLOAT32` 命名须统一，避免混用。
- **按平台合并为一条 data_types 记录**：同一平台不应出现多条 `data_types` 条目；条件化限制应改为条件约束表达。
- **函数签名类型不写入 data_types**：C/C++ 形参类型（如 `uint64_t`、`aclrtStream`）属于 `functions` 模块，不应写入 `parameter_constraints.data_types`。

**跨参数 dtype 关系约束校验**：
- **补充可机器判定的跨参数 dtype 对齐/推导关系**：如 `out.dtype == self.dtype` 等一致性要求，不能只保留类型枚举，必须补充可机判的显式关系约束。
- **条件化 dtype 关系须显式表达**：如”input 为 INT8 时 out 必须为 FLOAT”等条件关系，须用条件化约束显式表达。
- **禁止的混合 dtype 组合须显式表达**：如 A2/A3 平台禁止 BFLOAT16+FLOAT16 混合输入等规则，须在 dtype 约束中显式列出禁止组合。
- **dstType 与 y.dtype 的映射关系须独立表达**：不应写入 `types`，须用独立可判定约束表达。

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

**基本提取校验**：
- 从算子说明文档的”参数说明”表格中提取”非连续 Tensor”相关信息。
- 如果文档明确支持非连续 Tensor，应设置为 `discontinuous: true`。
- 如果文档明确不支持非连续 Tensor，应设置为 `discontinuous: false`。
- 如果不同平台的非连续 Tensor 支持情况不同，需要拆成多个 `platform` 元素。
- 如果文档没有提供该类约束，可使用 `null` 或空数组表达”无明确约束”。
- `discontinuous` 必须是布尔值，不接受字符串形式的 `”True”` / `”False”`。

**memory 字段规范化校验**：
- **非 Tensor 参数不生成 memory/discontinuous 约束**：`workspaceSize`、`keepDim` 等非 Tensor 参数，`memory` 应设为 `[]` 或 `null`。
- **仅文档明确时才填写 discontinuous**：只有文档明确写出支持或不支持非连续 Tensor 时，才填写具体的布尔值。
- 非 Tensor 参数通常不应落到 `memory`，若文档只给出标量/索引等参数，`memory` 置为 `[]` 或 `null`。

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

**基本提取校验**：
- 从算子说明文档中提取参数的取值限制信息。
- 允许的取值应写入 `allowed_values`。
- 禁止的取值应写入 `not_allowed_values`。
- 如果文档中未给出取值限制，可使用 `null` 或空数组。
- 如果某个平台项已经生成了 `allowed_values` 记录，但文档没有读到明确的取值约束，则其中的 `value` 应为 `[]`。
- 数值型限制应尽量规范表达：
  - 单值：如 `[1e-5]`
  - 区间：如 `[[0, 100]]`
  - 多段区间：如 `[[0, 7168], [8192, 10240]]`
- 如果文档使用中文枚举值或文本值，应在 `value` 中保留对应字符串。
- 如果不同平台有不同取值限制，需要拆成多个 `platform` 元素。
- 不要把 `format`、layout、数据格式名称等信息误写到 `value` 中，这些内容不属于取值约束。

**allowed_values / not_allowed_values 规范化校验**：
- **allowed_values.value 仅放具体允许值**：不放关系表达式（如 `out.dtype == self.dtype`）。
- **not_allowed_values.value 仅放具体禁止值**：不放条件关系字符串（如 `value < 0`）。条件性禁值须用可判定的条件约束表达。
- **条件化禁值/允许值须显式表达**：如"divMode=True 且 scales2Optional 存在时禁止 0"等，须拆分为条件+禁值。
- **空指针/None 统一为 `空指针` 或 `nullptr`**。
- **value 字段改为 [] 表达"值为空/无约束"**：当 `allowed_values.value` 仅承载推导关系而非具体枚举时，应改为 `[]`。

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
- 不要因为字段”看起来合理”就忽略格式错误。
- 不要因为格式合法就忽略业务错误。
- 任何来源于文档原句的约束，都应尽量保持机器可判定；如果结果只剩模糊自然语言而无法稳定判定，应视为提取质量不足。
- 对 optional 参数，校验时不仅要看是否被标记为可选，还要看其存在时的 shape / dtype / value 约束是否完整保留。

## 两段式接口参数处理校验

**executor 处理校验**：
- **executor 按接口阶段拆分或补充来源约束**：第一段接口的 `aclOpExecutor**`（出参）与第二段接口的 `aclOpExecutor*`（入参）不应合并为同一参数条目。

**workspaceSize 处理校验**：
- **workspaceSize 按接口阶段区分**：补充”第二段入参必须使用第一段返回的 workspaceSize”的来源约束。

**同名参数处理校验**：
- **同名参数按接口场景拆分或标注**：Inplace 系列算子中 `self` 与 `selfRef` 的约束不应合并。

## 平台与条件约束校验

- **按平台拆分/合并约束**：仅保留文档中实际支持平台约束，删除无依据的平台项。同平台类型列表须合并为一条记录。
- 每个参数的 `shape`、`data_types`、`memory`、`allowed_values`、`not_allowed_values` 都必须按字段边界分别承载约束，不要跨字段混写。
- 如果文档在某个参数下明确写了”与另一参数一致””由另一参数推导””输出与输入同 shape / 同 dtype”等关系，不能只保留 rank、范围或类型枚举；应将这些关系落成可解析规则，必要时再同步到 `inter_parameter_constraints`。
- 每个参数的 shape 数组仅允许单个平台对应一个数组元素，禁止在同一个 shape 元素中通过 rule 内嵌 platform 判断逻辑（如 `if platform == xxx`）。
- 不同平台的约束差异必须拆分为多个元素；如果所有平台完全一致，可以合并为统一的平台描述。

## Additional Notes 校验

- “platform”字段的取值应与文档中的平台描述一致；不同平台约束不同就显式拆分，不要混在一个元素中。
- 不要在 `shape.rule` 中混入 `dtype`、`format`、`memory`、平台说明或长段自然语言；每个字段只承载自己的语义边界。
- 所有表达式字段都必须输出字符串；`shape.rule` 没有内容时使用空字符串 `””`，不要使用 `null`。
- 任何来源于文档原句的约束都要尽量保持机器可判定，避免只写模糊自然语言描述。
- 如果文档明确给出空 tensor、禁止值、允许值、平台差异、可选参数存在条件等约束，都要落到对应字段，而不是依赖人工理解。
- 涉及跨参数的整体关系时，可以在当前参数字段中保留必要的可解析约束，同时在 `inter_parameter_constraints` 中补充更完整的多参数关系表达。
- 如果某字段没有内容，使用与 schema 一致的空结构，不要凭空补写文档中不存在的规则。
- 不要从单参数类型列表反推跨参数 dtype 关系；跨参数关系必须有明确来源并按 `inter_parameter_constraints` 或专用可解析字段表达。
- 对于有条件生效的 shape/dtype/value 约束，在表达式中显式体现条件，避免仅靠注释文本。
- 两段式接口下，`executor`/`workspaceSize` 来源关系要与接口阶段对应，不要混写同一参数。
- **可选参数间的依赖关系须显式表达**：如”仅当 smoothScale1Optional 存在时才允许 smoothScale2Optional 存在”等，须用可判定约束显式表达。

## 校验结论原则

- 同时报告格式错误和业务错误
- 如果格式不合法，直接判定为不通过
- 如果格式合法但业务内容与文档不一致，也应判定为不通过
- 只有格式和业务都通过，才能判定 `parameter_constraints` 校验通过
