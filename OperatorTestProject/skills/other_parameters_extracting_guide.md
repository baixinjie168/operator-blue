# other_parameters 提取规则

本文件定义了从 CANN 算子说明文档 中提取 **other_parameters** 的严格规则。

## 目标输出数据结构
```python 
class ParameterType(str, Enum):
    """参数类型枚举"""
    ACL_TENSOR = "aclTensor"
    ACL_SCALAR = "aclScalar"
    ACL_INT_ARRAY = "aclIntArray"
    ACL_FLOAT_ARRAY = "aclFloatArray"
    ACL_BOOL_ARRAY = "aclBoolArray"
    ACL_TENSOR_LIST = "aclTensorList"
    ACL_SCALAR_LIST = "aclScalarList"
    ACL_OP_EXECUTOR = "aclOpExecutor"
    ACL_RT_STREAM = "aclrtStream"
    # 标量类型
    DOUBLE = "double"
    BOOL = "bool"
    UINT64_T = "uint64_t"
    INT64_T = "int64_t"
    INT32_T = "int32_t"
    FLOAT = "float"
    FLOAT16 = "float16"
    VOID = "void"
    SIZE_T = "size_t"
    
class OtherParameterConstraint(BaseModel):
    platform: str = Field(..., description="平台名称")
    value: List[List[float]] | List[float] | List[List[int]] | List[int] | List[str] | List[bool] = Field(
        default_factory=list, description="参数的允许或禁止的值")
    rule: str = Field(default="", description="未在'函数原型'中定义的参数需要满足的约束条件")

    model_config = {"extra": "forbid"}

class OtherParameters(BaseModel):
    name: str = Field(..., description="参数名称")
    type: str = Field(..., description="参数类型")
    description: str = Field(default=None, description="参数描述")
    constraints: List[OtherParameterConstraint] = Field(..., description="参数相关约束")

    model_config = {"extra": "forbid"}

class OtherParametersOutput(BaseModel):
    """顶层模型"""
    other_parameters: List["OtherParameters"] = Field(default_factory=list, description="其他未在'函数原型'中定义的参数")

    model_config = {"extra": "forbid"}
```
【输出规则（必须严格遵守，缺一不可）】
1.  输出格式：仅返回纯JSON字符串，无任何多余内容（无解释、无代码块、无换行备注）；
2.  输出范围：只返回【顶层类】的完整结构，自动嵌套填充所有内层类，不单独输出任何内层类（如内层类1、内层类2）；
3.  字段约束：字段名、字段类型、层级结构，必须与上面定义的类完全一致，禁止新增、缺失、修改字段；
4.  类型匹配：严格遵循类型注解（str/int/bool/List/Optional等），空值统一用null（JSON规范），不随意填充无效值；
5.  嵌套要求：所有嵌套结构必须完整，若待处理内容中无相关信息，可选字段填默认值，必填字段（Field(...)）填合理空值（如""、0、[]）。

## other_parameters 提取规则

`other_parameters` 属性是一个列表，其中包含未在主函数参数中定义的额外参数。这些参数通常是在算子文档中引入的，用于描述关系或约束，但并不属于实际函数签名的一部分。

### 识别规则
- **识别条件**：只有文档明确把某个符号量当作约束符号、维度符号、计数符号或中间定义量时，才将其提取到 `other_parameters`。
- **查找区域**：在"功能说明""约束说明""参数说明中的使用说明""公式说明""shape 说明"等区域查找未在"参数说明"或函数原型中定义、但被稳定引用的符号。
- **候选符号**：例如，在 `shape 为 (max(tpWorldSize, 1) * A, H)` 这类描述中，`A` 和 `H` 可以作为候选额外参数；但如果某个量只是一次性公式中的临时中间表达式，则不要提取。

### 结构规则
列表中的每个元素应为一个对象，包含以下字段：

#### name
- 参数名称，为字符串类型。

#### description
- **参数定义性描述**：只解释"这个符号代表什么"。
- **职责边界**：`description` 只负责定义符号含义，不要把 shape/dtype/range 关系堆进 `description`。
- **关系迁移**：如果文档中的关系被写进了 `description`，而 `rule` 为空或过于简化，应把这些关系迁回 `rule`。

#### type
- 参数类型，参考 `ParameterType`。

#### constraints
约束对象的列表，每个约束对象包含以下字段：

##### platform
- 执行平台（例如 `'Atlas A3 训练系列产品/Atlas A3 推理系列产品'`、`'Atlas A2 训练系列产品/Atlas A2 推理系列产品'`、`'Atlas 200I/500 A2 推理产品'`、`'Atlas 推理系列产品'`、`'Atlas 训练系列产品'` 或 `'All'`）。
- 每个 `constraints` 元素只对应一个平台范围。
- 不要在单个元素中内嵌 `if platform == xxx` 这类平台分支逻辑。
- 如果多个平台约束相同，可将 `platform` 设为 `All`；如果平台差异不同，则必须拆成多个元素。

##### rule
- **用途**：`rule` 用于表达该符号与其他参数、维度或符号之间的关系；凡是和张量 shape、dtype、维度位置、rank 场景有关的关系，都应放在 `rule` 中。
- **完整性**：对 B、S、H、N、C、Hin、Hout 等中间符号，要覆盖其与相关张量所有关键维度的对应关系，不要只写某一种 rank 的下标关系。
- **符号补充**：如果文档隐含了缺失的中间符号但确实需要它来表达约束，应先补齐该符号，再用 `rule` 建立对应关系。
- **格式要求**：`rule` 不允许为 `null`；如果没有关系规则，使用空字符串 `""`。
- **内容限制**：不要把长段自然语言说明、平台分支或一次性临时表达式写进 `rule`。
- **职责分离**：仅保留关系约束，删除定义性描述；参数含义保留在 `description`，取值限制放入 `value`。
- **删除评注性内容**：不要在 `rule` 中写入"应视为文档前后不一致"等评注性内容，仅保留可直接校验的关系表达。
- **条件化表述**：将无条件约束改为条件化描述，明确适用前提。
- **按场景拆分**：如 `ceilMode=false` 和 `ceilMode=true`、`transposed=true` 和 `transposed=false` 应分别给出对应规则。

##### value
- **用途**：`value` 描述参数的取值限制，遵循与 `allowed_values.value` 相同的规则。
- **数值参数**：直接列出允许值，例如"取值为0和1" -> `[0, 1]`。
- **范围参数**：使用区间表达，例如"0到epWorldSize-1之间的整数" -> `[[0, epWorldSize-1]]`。
- **分段参数**：使用多个区间元素，例如"0到7168之间的整数或8192到10240之间的整数" -> `[[0, 7168], [8192, 10240]]`。
- **职责分离**：取值范围归 value，关系约束归 rule；不伪造上界，不补充未文档化的下界。
- **内容限制**：不要把关系规则、条件判断、函数调用或自然语言说明写进 `value`。
- **条件化处理**：无法枚举的条件（如 `channels % 8 = 0`）保留在 rule，可枚举的上界拆到 value 中表达。

### 通用规则
- **不得重复同一个参数名**。
- **不要把真实函数参数、平台名、公式片段或纯说明文字误抽为 `other_parameters`**。
- **如果没有符合条件的额外参数，则该字段应为 `[]`**。
- 规则只针对文档明确且稳定出现的符号；一次性临时标记（如临时索引、局部中间量）不应单独保留。
- 规则优先使用 `rule` 承载关系、`value` 承载取值范围，避免将关系文字叠在 `description`。
- 不要凭经验补充约束上下界；文档未给出上界/下界时对应字段置空并保留可判定关系。

### 条目删除与空值规则
- **无合法参数时置空**：若文档没有需要抽取的合法 `other_parameters`，将该字段设为 `[]`，删除所有示例维度条目。
- **删除重复语义条目**：不要在 `other_parameters` 中重复承载函数参数或输出参数的语义（如 `rstd` 对应 `rstdOut`、`E(x)`/`Var(x)` 对应均值/方差输出）。
- **删除公式局部符号**：不要将公式中仅作为临时求和下标或局部符号（如 `i`、`E(x)`、`Var(x)`）提取为独立条目。
- **删除未文档化参数**：仅在文档显式定义了函数原型之外、可独立引用的稳定符号时才提取；否则删除该条目并将 `other_parameters` 设为 `[]`。

### 维度符号与参数补充
- **补充文档中显式出现的维度符号**：如 `C`、`Hin`、`Win`、`N`、`M`、`k` 等，`type` 设为 `int64_t`，`description` 仅说明符号含义，`constraints.rule` 表达与相关张量的维度关系。
- **拆分卷积参数为方向分量**：将 `stride`/`padding`/`dilation` 拆分为方向分量（如 `strideH`/`strideW`），`type` 设为 `int32_t`。
- **合并同义维度符号**：合并 `D` 与 `Din`、`H` 与 `Hin`、`H_o` 与 `Hout` 等同义符号，只保留一个顶层参数。
- **不自行扩展文档约束**：严格按原文提取，若确认原文有笔误，应先修正文档再重新生成。

### 数据格式与维度场景
- **区分数据格式**：若规则仅适用于 NCHW 格式，应在 `rule` 中显式限定；不要把 NCHW 专属下标关系写成通用规则。
- **区分维度场景（4维 vs 5维）**：补充 4 维与 5 维两种场景的 `constraints`；不要把 5 维下标关系写成唯一约束。
- **区分卷积维度（1D/2D/3D）**：按卷积 rank 区分表述，或明确限定为特定维度场景。

### 平台与场景拆分
- **按平台和场景拆分 constraints**：不同平台和不同场景的约束应分别建模，不要用单个 `All` 范围覆盖所有场景。
- **软性约束不写入 value**：如"超过该阈值可能导致超时"这类软性约束不要用 `value` 表达成硬性合法范围，应在 `rule` 中补充说明。

### 跨参数联动与完整约束
- **补充跨参数联动约束**：在 `rule` 中补充跨参数的 shape 联动关系（如 `groups` 与 `C_in`、`weight` 第二维的关系）。
- **补充输出张量的条件化 shape 约束**：如 `deformOutOptional` 存在时 shape 为 `[N, inC, outH*K_H, outW*K_W]`。
- **补充乘积范围约束**：如 `Din*Hin*Win <= 2147483647` 等乘积限制应显式写入 `rule`。
