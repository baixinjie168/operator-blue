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

**description 字段校验**：
- `description` 只负责解释”这个参数是什么”，不应承担 shape / dtype / rank / 取值关系。
- `description` 职责边界：只解释”这个符号代表什么”。
- 如果文档中的关系被写进 `description`，而 `rule` 为空或过于简化，应把这些关系迁回 `rule`。
- 如果文档中的关系被写进 `description`，而 `rule` 为空，应判定为职责分工错误。

**constraints.platform 校验**：
- 每个 `constraints` 元素只对应一个平台范围。
- 不要在单个元素中内嵌 `if platform == xxx` 这类平台分支逻辑。
- 如果多个平台约束相同，可将 `platform` 设为 `All`；如果平台差异不同，则必须拆成多个元素。

**constraints.rule 字段校验**：
- `rule` 用于表达该符号与其他参数、维度或符号之间的关系；凡是和张量 shape、dtype、维度位置、rank 场景有关的关系，都应放在 `rule` 中。
- 对 B、S、H、N、C、Hin、Hout 等中间符号，要覆盖其与相关张量所有关键维度的对应关系，不要只写某一种 rank 的下标关系。
- 如果文档隐含了缺失的中间符号但确实需要它来表达约束，应先补齐该符号，再用 `rule` 建立对应关系。
- `rule` 不允许为 `null`；如果没有关系规则，使用空字符串 `””`。
- 不要把长段自然语言说明、平台分支或一次性临时表达式写进 `rule`。
- **职责分离**：仅保留关系约束，删除定义性描述；参数含义保留在 `description`，取值限制放入 `value`。
- **删除评注性内容**：不要在 `rule` 中写入”应视为文档前后不一致”等评注性内容，仅保留可直接校验的关系表达。
- **条件化表述**：将无条件约束改为条件化描述，明确适用前提。
- **按场景拆分**：如 `ceilMode=false` 和 `ceilMode=true`、`transposed=true` 和 `transposed=false` 应分别给出对应规则。

**constraints.value 字段校验**：
- `value` 描述参数的取值限制，遵循与 `allowed_values.value` 相同的规则。
- **职责分离**：取值范围归 value，关系约束归 rule；不伪造上界，不补充未文档化的下界。
- **内容限制**：不要把关系规则、条件判断、函数调用或自然语言说明写进 `value`。
- **条件化处理**：无法枚举的条件（如 `channels % 8 = 0`）保留在 rule，可枚举的上界拆到 value 中表达。

## type 校验

### 结构校验

- `type` 必须是字符串

### 业务校验

### 1. 识别 other_parameters 的来源

**基本识别校验**：
- 优先从”功能说明””约束说明””参数说明中的使用说明””公式说明””shape 说明”等位置识别额外参数。
- 重点关注文档中被单独命名、但没有出现在函数原型中的符号，例如维度变量、分组数、隐藏维、分片因子等。

**识别条件校验**：
- 只有文档明确把某个符号量当作约束符号、维度符号、计数符号或中间定义量时，才将其提取到 `other_parameters`。
- 如果某个符号只是公式里的临时中间量，且没有独立参数意义，不应提取为 `other_parameters`。
- 只有文档明确把该符号当作稳定约束符号使用时，才应保留。

**候选符号示例**：
- 在 `shape 为 (max(tpWorldSize, 1) * A, H)` 这类描述中，`A` 和 `H` 可以作为候选额外参数。
- 但如果某个量只是一次性公式中的临时中间表达式，则不要提取。

### 2. 与 functions 的边界

- `other_parameters` 与 `functions.parameters` 不能重复承载同一个真实参数。
- 真正出现在 API 原型里的输入、输出、workspace、stream、executor 等参数，应保留在 `functions`。
- `other_parameters` 只负责承载函数签名之外、但文档又明确约束或引用的额外参数。

**通用规则校验**：
- 不得重复同一个参数名。
- 不要把真实函数参数、平台名、公式片段或纯说明文字误抽为 `other_parameters`。
- 如果没有任何符合条件的额外参数，则该字段应为 `[]`。
- 规则只针对文档明确且稳定出现的符号；一次性临时标记（如临时索引、局部中间量）不应单独保留。
- 规则优先使用 `rule` 承载关系、`value` 承载取值范围，避免将关系文字叠在 `description`。
- 不要凭经验补充约束上下界；文档未给出上界/下界时对应字段置空并保留可判定关系。

### 3. 参数去重与合并

- 同一个 `name` 不要重复生成多个顶层元素。
- 平台差异应放在同一参数的 `constraints` 列表中，不要为不同平台重复创建同名参数。
- 同一平台下语义完全重复的约束不要重复出现。
- 如果规则需要额外中间符号才能表达，应先确认该符号被正确补齐，而不是把关系粗暴省略。

### 4. 描述与约束分工

- `description` 负责解释”这个参数是什么”。
- `rule` 负责表达”这个参数与其他量的关系”。
- `value` 负责表达”这个参数允许取什么值或范围”。
- 不要把三者混写到同一个字段中。
- `rule` 不允许为 `null`；没有关系规则时应写成空字符串 `””`。
- 如果文档中同时存在多种 rank / 输入输出形态，应检查 `rule` 是否完整覆盖，而不是只保留某一种场景。

**条目删除与空值规则校验**：
- **无合法参数时置空**：若文档没有需要抽取的合法 `other_parameters`，将该字段设为 `[]`，删除所有示例维度条目。
- **删除重复语义条目**：不要在 `other_parameters` 中重复承载函数参数或输出参数的语义（如 `rstd` 对应 `rstdOut`、`E(x)`/`Var(x)` 对应均值/方差输出）。
- **删除公式局部符号**：不要将公式中仅作为临时求和下标或局部符号（如 `i`、`E(x)`、`Var(x)`）提取为独立条目。
- **删除未文档化参数**：仅在文档显式定义了函数原型之外、可独立引用的稳定符号时才提取；否则删除该条目并将 `other_parameters` 设为 `[]`。

**维度符号与参数补充校验**：
- **补充文档中显式出现的维度符号**：如 `C`、`Hin`、`Win`、`N`、`M`、`k` 等，`type` 设为 `int64_t`，`description` 仅说明符号含义，`constraints.rule` 表达与相关张量的维度关系。
- **拆分卷积参数为方向分量**：将 `stride`/`padding`/`dilation` 拆分为方向分量（如 `strideH`/`strideW`），`type` 设为 `int32_t`。
- **合并同义维度符号**：合并 `D` 与 `Din`、`H` 与 `Hin`、`H_o` 与 `Hout` 等同义符号，只保留一个顶层参数。
- **不自行扩展文档约束**：严格按原文提取，若确认原文有笔误，应先修正文档再重新生成。

**数据格式与维度场景校验**：
- **区分数据格式**：若规则仅适用于 NCHW 格式，应在 `rule` 中显式限定；不要把 NCHW 专属下标关系写成通用规则。
- **区分维度场景（4维 vs 5维）**：补充 4 维与 5 维两种场景的 `constraints`；不要把 5 维下标关系写成唯一约束。
- **区分卷积维度（1D/2D/3D）**：按卷积 rank 区分表述，或明确限定为特定维度场景。

**平台与场景拆分校验**：
- **按平台和场景拆分 constraints**：不同平台和不同场景的约束应分别建模，不要用单个 `All` 范围覆盖所有场景。
- **软性约束不写入 value**：如”超过该阈值可能导致超时”这类软性约束不要用 `value` 表达成硬性合法范围，应在 `rule` 中补充说明。

**跨参数联动与完整约束校验**：
- **补充跨参数联动约束**：在 `rule` 中补充跨参数的 shape 联动关系（如 `groups` 与 `C_in`、`weight` 第二维的关系）。
- **补充输出张量的条件化 shape 约束**：如 `deformOutOptional` 存在时 shape 为 `[N, inC, outH*K_H, outW*K_W]`。
- **补充乘积范围约束**：如 `Din*Hin*Win <= 2147483647` 等乘积限制应显式写入 `rule`。

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
