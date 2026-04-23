# inter_parameter_constraints 提取规则

本文件定义了从 CANN 算子说明文档 中提取 **inter_parameter_constraints（参数之间的约束关系）** 的严格规则。

## 目标输出数据结构
```python 
class InterConstraintsRuleType(str, Enum):
    """参数间约束类型枚举"""
    SHAPE_BROADCAST = "shape_broadcast"
    SHAPE_CHOICE = "shape_choice"
    SHAPE_EQUALITY = "shape_equality"
    SHAPE_DEPENDENCY = "shape_dependency"
    SHAPE_VALUE_DEPENDENCY = "shape_value_dependency"
    PRESENCE_DEPENDENCY = "presence_dependency"
    TYPE_DEPENDENCY = "type_dependency"
    TYPE_EQUALITY = "type_equality"
    VALUE_DEPENDENCY = "value_dependency"
    FORMAT_EQUALITY = "format_equality"

class InterParamConstraint(BaseModel):
    """参数间约束模型"""
    type: InterConstraintsRuleType = Field(..., description="约束类型")
    params: List[str] = Field(..., description="涉及的参数列表")
    expr: str = Field(..., description="约束表达式")
    description: str = Field(default="", description="约束描述")
    source: Optional[str] = Field(default=None, description="源参数")
    target: Optional[List[str]] = Field(default=None, description="目标参数列表")

    model_config = {"extra": "forbid"}

class InterParameterConstraintsOutput(BaseModel):
    """顶层模型"""
    inter_parameter_constraints: List["InterParamConstraint"] = Field(default_factory=list, description="参数间约束")

    model_config = {"extra": "forbid"}
```
【输出规则（必须严格遵守，缺一不可）】
1.  输出格式：仅返回纯JSON字符串，无任何多余内容（无解释、无代码块、无换行备注）；
2.  输出范围：只返回【顶层类】的完整结构，自动嵌套填充所有内层类，不单独输出任何内层类（如内层类1、内层类2）；
3.  字段约束：字段名、字段类型、层级结构，必须与上面定义的类完全一致，禁止新增、缺失、修改字段；
4.  类型匹配：严格遵循类型注解（str/int/bool/List/Optional等），空值统一用null（JSON规范），不随意填充无效值；
5.  嵌套要求：所有嵌套结构必须完整，若待处理内容中无相关信息，可选字段填默认值，必填字段（Field(...)）填合理空值（如""、0、[]）。

## 函数属性

### type
`type`属性用于指定参数之间的约束类型。该值必须是以下字符串字面量之一：
- `shape_broadcast`：表示形状需要满足广播（broadcast）关系。
- `shape_choice`：表示一个形状可以是多个候选参数之一。
- `shape_equality`：表示形状必须完全相同。
- `shape_dependency`：表示 shape 元素值之间的依赖关系。
- `shape_value_dependency`：表示特定轴值之间的依赖关系。
- `value_dependency`：表示张量元素值或参数值之间的约束关系。
- `presence_dependency`：表示共存规则（例如 `(A is None) == (B is None)`）。
- `type_dependency`：表示数据类型依赖关系。
- `type_equality`：表示数据类型必须一致。
- `format_equality`：表示数据格式必须一致。

**要求**：
- 该值必须严格遵循上述选项之一，不允许使用其他值。
- 只抽取真正依赖多个参数的约束；单参数约束、公式解释或功能性描述不要放进 `inter_parameter_constraints`。
- 先判断语义再选 `type`，不要把 dependency、choice、equality、value_dependency 等关系混用。
- 像"shape 为 [B,H] 或 [B,1,H]，且由 x 推导"这类关系，本质上是 `shape_dependency`，不是 `shape_choice`。
- "与另一参数同 dtype / 同 format / 同 shape"这类显式跨参数关系，应优先在本模块中选择语义最准确的 `type` 表达。
- 形状相关约束按语义进一步细分：轴值关系优先 `shape_value_dependency`，广播关系使用 `shape_broadcast`。
- 同时为空/不为空类规则优先使用 `presence_dependency`。
- **shape 轴值与 dim 之间的约束**：`type` 应为 `shape_value_dependency`，而非 `shape_dependency` 或 `value_dependency`。
- **"同时为空或同时不为空"的逻辑**：使用 `presence_dependency`。

### params
`params`属性是一个字符串列表，用于指定约束所涉及的参数。约束表达式中提到的所有参数都必须包含在此列表中。
- `expr`字段中引用的所有参数都必须包含在`params`列表中。
- 不要包含与约束无关的参数。
- 列表中应仅包含参数名称作为字符串。
- **`params` 应包含该约束直接涉及的所有参数**，不要遗漏关键参与参数（如 `dim`、`keepDim`、`ignoreIndex`、`bidirection` 等）。
- **`params` 中不要包含与约束逻辑无关的参数**。

### expr
`expr`：必填字段（CRITICAL）。你必须为每个参数间约束提供一个正式、合法的 Python 表达式，以表示逻辑关系，禁止仅使用标点符号表示逻辑。

**基本规则**：
- `expr` 不允许为 `null`；如果当前约束暂时无法写出表达式，也必须使用空字符串 `""`，不要输出 `null`。
- 不要使用 `implies` 这个词。
- 不要使用伪代码。
- 不要在 `expr` 中使用平台值作为判断条件。
- 不要使用自然语言式 `if/else` 说明语句。
- 蕴含逻辑：对于"如果 A 则 B"的逻辑，优先使用合法 Python 布尔表达式，例如 `(B) if (A) else True`，或其他语义等价且可执行的布尔表达式。
- 等价逻辑：对于"A当且仅当B"的逻辑，使用相等性 `(A) == (B)`；例如 `(scales2 is None) == (zeroPoints2 is None)`。
- 文档资料中的 tensor "元素个数"是指 tensor 中所有元素数量；若 shape 为 `[x, y, z]`，则元素个数为 `x * y * z`。
- Python 表达式必须合法有效，且返回值为 bool；若涉及生成器表达式，必须包裹在 `all()` 或 `any()` 等函数中，且不允许使用 `lambda`。
- `expr` 必须只引用本条约束真正涉及的参数；`params` 要覆盖 `expr` 中出现的关键参数。
- 除非是在表达合法的蕴含逻辑，否则不要滥用三元表达式。
- `expr` 不能混入单参数限制条件、平台判断或恒真式占位逻辑。
- 当约束是条件生效时，`expr` 中应显式写明条件，不要靠 `else True` 兜底。

**修正规则**：
- **`expr` 必须是合法的 Python 布尔表达式**；无法精确表达时，不要保留空 `expr` 的约束。
- **`expr` 应精确校验各轴值**，而不是仅比较 `len(out.shape)`。
- **broadcast 判定应使用标准语义**：逐维满足"相等或其中一个为 1"，不要直接用 `max(...)` 推导。
- **`expr` 中不要混入单参数限制条件**（如 rank 限制、`len(kernelSize) in (1, 3)` 等），仅保留跨参数关系。
- **`expr` 中不要使用恒为 True 的表达式**。
- **`expr` 应覆盖所有分支场景**，不要用 `else True` 放过未覆盖的分支。
- **不要在 `expr` 中引用跨调用历史返回值**（如前次返回的 fd），仅保留单次调用内可判定的条件。

### description
`description`属性提供约束的人类可读描述。应以自然语言清晰解释该约束。
- 应以清晰、简洁的语言描述约束。
- 应引用所涉及的具体参数。
- 应解释约束的性质（例如，"输入x1、x2和输出yOut、xOut必须具有相同的形状"）。
- 如果没有描述，该字段可以为空字符串。
- `description` 只说明约束语义，不额外承载平台矩阵或单参数限定。
- **`description` 应仅描述文档明确给出的约束语义**，不要包含平台说明、产品条件或单参数限制。
- **`description` 应与 `expr` 保持一致**，同步修改。

### source
`source`属性用于指定方向性约束关系中的源参数。
- 当参数之间存在明显方向性的关系时使用。
- 例如，在 broadcast、choice、dependency 等关系中，源参数通常是"作为基准"的那个参数。
- 该字段是可选的；如果关系不是方向性的，则可以省略。
- 当存在时，应包含一个参数名称作为字符串。
- 所有 dependency 类约束都应优先补齐 `source`，不要把明显有方向的关系留成 `null`。
- 当存在明显方向时，`source` 应为触发/基准参数。
- **`source` 应设为基准/触发参数**（如 `source="self"`, `target=["dim"]`）。

### target
`target`属性用于指定方向性约束关系中的目标参数。
- 当参数之间存在明显方向性关系时使用。
- 例如，在 broadcast、choice、dependency 等关系中，目标参数是被约束或被推导的参数。
- 该字段是可选的；如果关系不是方向性的，则可以省略。
- 当存在时，应包含参数名称的字符串列表。
- 当多个参数受同一源参数影响时，可以包含多个目标参数。
- 所有 dependency 类约束都应优先补齐 `target`，体现"谁约束谁、谁依赖谁"的方向。
- `target` 建议显式列出当前约束被制约的参数（如 `["out"]` / `["dim"]`）。
- **`target` 至少补充为 `["out"]`**（或实际被约束的参数），不要留空或置为 null。
- **多来源依赖时**：可拆分约束，或将 `source`/`target` 同时置为 `null`，通过 `params` + `expr` 表达多参数联合依赖。
- **不要把单个参数标成唯一 `source` 来表达多参数共同约束关系**：应置 `source` 为 null 或拆分约束。

## 文档来源与约束提取规则

### 文档来源严格性原则
- **仅保留文档明确给出的跨参数约束**：不要从公式语义、示例代码、返回值错误码表、常识推断补造约束。
- **不要从单参数 dtype 列表反推跨参数转换矩阵**：仅当文档明确给出 dtype 转换关系/映射表时，才补充 `type_dependency`。
- **不要将功能说明/计算公式/示例代码中的 out shape 推导关系写入**：计算公式语义应放到专门描述算子计算逻辑的字段中。
- **不要将返回值错误码表中的校验场景写入**。
- **不要引入文档未出现的伪属性**（如 `workspace.size`）或未明确说明的内存容量判断。
- **空Tensor传播规则只保留文档明确支持空Tensor的输入场景**，不要为文档未明确支持空Tensor的参数补造传播约束。

### 特定约束模式补充规则
- **dim 合法范围约束**：补充 `value_dependency`，`params` 至少包含 `["self", "dim"]`，`expr` 为 `(-len(self.shape) <= dim <= len(self.shape) - 1)`。
- **BFLOAT16 下 dim 轴长度约束**：当 `self.dtype` 为 BFLOAT16 时，`self.shape[dim] != 1`；使用 `shape_value_dependency`。
- **keepdim 对 out.shape 的影响**：`keepdim=False` 时 out.shape 等于删除 dim 对应轴后的 self.shape；`keepdim=True` 时保留原轴数且被 reduce 轴为 1。
- **broadcast 关系约束**：`self` 与 `other` 的 shape 必须满足 broadcast 关系，`out.shape` 等于广播结果 shape。
- **target 元素合法索引范围约束**：`0 <= value < self.shape[-1]`，允许等于 `ignoreIndex`；`params` 至少包含 `["self", "target", "ignoreIndex"]`。
- **type_dependency 约束**：表达 `self.dtype` 必须可转换为 `out.dtype` 的关系。
- **stride 默认等于 kernelSize**：当 `len(stride) == 0` 时，stride 的有效步长等于 kernelSize。
- **padding 不大于 kernelSize 一半**：`all(2 * p <= k for p, k in zip(padding, kernelSize))`。
- **卷积 output shape 推导**：新增 `shape_value_dependency`，按 1D/2D/3D 场景表达 output 各空间维的 infershape 公式。
- **两段式接口 workspaceSize/executor 一致性**：补充 `value_dependency`，表达第二段的 workspaceSize/executor 必须等于第一段返回值。

### 条件/分支约束处理规则
- **带条件的约束应显式表达生效前提**：如仅在 `isbias=true`、`bidirection=true`、`packed=true`、`reduction==1`、特定 dtype 等条件下才生效的约束，应在 `expr` 中加入条件判断。

### 约束删除/迁移规则
- **删除文档未明确给出的约束**：如 minOut/maxOut shape 约束，只有文档明确说明其关系时才保留。
- **删除仅表达计算公式的约束**：计算公式应放入描述功能/公式的对应结构化字段。
- **删除仅表达单参数限制的跨参数约束**：如"当前仅支持 axis=-1"应放到单参数规则中。
- **删除仅表达默认值/默认行为的约束**：应放到参数说明或默认值逻辑位置。
- **删除凭空写死具体 dtype 组合的约束**：若文档不足以支撑完整推导规则，应删除。

## 重要提醒
- expr 表达式应尽可能简洁，不包含冗余条件。
- 所有表达式字段都要输出字符串；没有内容时使用空字符串 `""`，不要使用 `null`。
- 如果文档关系需要额外中间符号，先确认这些符号是否应由 `other_parameters` 定义，再在本模块的 `expr` 中引用。
- 不要把冗余关系、重复关系或本应删除的单参数限制硬塞进 `inter_parameter_constraints`。
- 仅抽取文档明确给出的约束，不要从示例代码、错误码表或公式语义推导跨参数关系。
- 与公式推导相关的 shape 推导语义，若非约束性关系，避免写入 `inter_parameter_constraints`。
