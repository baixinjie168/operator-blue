# parameter_constraints 提取规则

本文件定义了从 CANN 算子说明文档 中提取 **parameter_constraints（参数约束）** 的严格规则。

## 目标输出数据结构
```python 
class ShapeStructure(str, Enum):
    """Shape约束结构类型枚举"""
    DIMS = "dims"
    AXIS_VALUE = "axis_value"
    
class ShapeRule(BaseModel):
    """Shape约束规则模型"""
    structure: ShapeStructure = Field(..., description="结构类型: dims 或 axis_value")
    rule: str = Field(..., description="Python表达式规则")
    dim_num: List[List[int]] | List[int] = Field(default_factory=list, description="dim的取值范围，如[[2,4],6]表示shape的dim取值为6或者2<=dim<=4")
    dim_valid_value: List[List[int]] | List[int] = Field(default_factory=list, description="shape中每一维合法取值的范围")
    dim_invalid_value: List[List[int]] | List[int] = Field(default_factory=list, description="shape中每一维非法取值范围")

    model_config = {"extra": "forbid"}

class ParamShape(BaseModel):
    """参数Shape约束模型"""
    platform: str = Field(..., description="平台名称或列表")
    constraint: List[ShapeRule] = Field(..., description="约束规则列表")

    model_config = {"extra": "forbid"}

class ParamDataType(BaseModel):
    """参数数据类型约束模型"""
    platform: str = Field(..., description="平台名称或列表")
    types: List[str] = Field(..., description="支持的类型列表")

    model_config = {"extra": "forbid"}

class ParamMemory(BaseModel):
    """参数内存约束模型"""
    platform: str = Field(..., description="平台名称或列表")
    discontinuous: bool = Field(..., description="是否支持非连续Tensor")

    model_config = {"extra": "forbid"}

class ParamValue(BaseModel):
    """参数值约束模型"""
    platform: str = Field(..., description="平台名称或列表")
    value: List[List[float]] | List[float] | List[List[int]] | List[int] | List[str] | List[bool] = Field(
        default_factory=list, description="允许或禁止的值")

    model_config = {"extra": "forbid"}

class SingleParamConstraints(BaseModel):
    """单个参数的约束模型"""
    shape: List[ParamShape] = Field(..., description="Shape约束列表")
    data_types: List[ParamDataType] = Field(..., description="数据类型约束列表")
    memory: Optional[List[ParamMemory]] = Field(..., description="内存约束列表")
    allowed_values: Optional[List[ParamValue]] = Field(..., description="允许的值列表")
    not_allowed_values: Optional[List[ParamValue]] = Field(..., description="禁止的值列表")

    model_config = {"extra": "forbid"}

class ParamConstraints(BaseModel):
    """参数约束模型"""
    name: str = Field(..., description="参数名称")
    constraints: SingleParamConstraints = Field(..., description="参数约束详情")

    model_config = {"extra": "forbid"}

class ParameterConstraintsOutput(BaseModel):
    """顶层模型"""
    parameter_constraints: List["ParamConstraints"] = Field(..., description="参数约束列表")

    model_config = {"extra": "forbid"}
```
【输出规则（必须严格遵守，缺一不可）】
1.  输出格式：仅返回纯JSON字符串，无任何多余内容（无解释、无代码块、无换行备注）；
2.  输出范围：只返回【顶层类】的完整结构，自动嵌套填充所有内层类，不单独输出任何内层类（如内层类1、内层类2）；
3.  字段约束：字段名、字段类型、层级结构，必须与上面定义的类完全一致，禁止新增、缺失、修改字段；
4.  类型匹配：严格遵循类型注解（str/int/bool/List/Optional等），空值统一用null（JSON规范），不随意填充无效值；
5.  嵌套要求：所有嵌套结构必须完整，若待处理内容中无相关信息，可选字段填默认值，必填字段（Field(...)）填合理空值（如""、0、[]）。

## 参数约束的属性

### 平台提取规则
- 从算子说明文档的"产品支持情况"表格中提取 platform 有效列表，platform 从第一列"产品"中提取平台名字，根据第二列的"是否支持"来判断有效性。
  - 有 `√` 标志 → 有效，支持该产品，列入 platform 有效列表。
  - 有 `×` 标志或无标志 → 无效，不支持该产品，不统计到 platform 有效列表。

### 参数遍历规则
- 遍历所有参数，提取以下数据（shape 为 tensor 类数据专有）。每一个参数都需要进行描述，不要缺失任何一个参数；如果某个参数属性没有约束条件，则使用一致的数据格式表示为空值。

### 平台字段通用规则
对于以下参数属性的 `platform` 字段，需满足如下要求：
- 所有 `platform` 字段都应该从 platform 有效列表获取。
- 每个参数的 `shape`、`data_types`、`memory`、`allowed_values`、`not_allowed_values` 都必须按字段边界分别承载约束，不要跨字段混写。
- 如果文档在某个参数下明确写了"与另一参数一致""由另一参数推导""输出与输入同 shape / 同 dtype"等关系，不能只保留 rank、范围或类型枚举；应将这些关系落成可解析规则，必要时再同步到 `inter_parameter_constraints`。
- 每个参数的 shape 数组仅允许单个平台对应一个数组元素，禁止在同一个 shape 元素中通过 rule 内嵌 platform 判断逻辑（如 `if platform == xxx`）。
- 不同平台的约束差异必须拆分为多个元素；如果所有平台完全一致，可以合并为统一的平台描述。
- 对 optional 参数，不仅要保留"可选"语义，还要把其存在时的 shape / dtype / value 关系补齐。
- **按平台拆分/合并约束**：仅保留文档中实际支持平台约束，删除无依据的平台项。同平台类型列表须合并为一条记录。

## shape 提取规则

### 基本规则
- 提取与"shape"、"dimensions"或类似关键词相关的信息。
- 结果应为一个列表，其中每个元素包含特定于平台的形状约束。
- 使用"platform"来定义执行平台。
- 每个形状约束元素必须对应一个平台，并且不能在规则中包含特定于平台的逻辑。

### constraint 字段规则
使用"constraint"字段来指定平台特定规则，其中包含：

#### structure
- "dims"：用于约束维度的数量（例如，"1 < len(x.shape) <= 8"）
- "axis_value"：用于约束特定维度值或维度之间的对应关系（例如，"x.shape[0] * x.shape[1] <= 2147483647"）

#### rule
- 必须是一个有效的 Python 布尔表达式。
- `dims` 只表达 shape 的维度数量约束；`axis_value` 只表达各轴取值关系以及由其他参数推导出的 shape 对应关系，不要把 dtype、format、memory 等语义混进 shape。
- `shape.constraint[*].rule` 不允许为 `null`；如果某个约束元素暂时没有可写表达式，只能使用空字符串 `""`。
- 如果文档明确写了"shape 为 [B,H] 或 [B,1,H]""out.shape == x.shape""最后一维等于另一参数某一维"这类关系，必须在 `axis_value.rule` 中完整表达，不要只保留 rank 或部分轴值。
- 同一平台下如果存在多个相同 `structure` 的条件，应优先合并为一个元素，在 `rule` 中组合表达完整业务语义，避免重复或拆分失衡。
- 所有表达式必须是有效的 Python 布尔表达式，并且返回布尔值；涉及循环的生成器表达式时，应使用 `all()` 或 `any()` 等函数调用将其包裹起来。
- 不要使用 `lambda` 表达式；多个表达式应使用 `and` 或 `or` 连接；除非确实用于表达合法蕴含逻辑，否则不要滥用三元表达式。
- 如果没有约束条件，则将相关列表置为空。
- **shape.rule 仅承载纯 shape/轴值约束**：dtype 条件、dtype 绑定的 shape 限制须拆到专门的约束表达。
- **维度数量判断放 dims，轴值关系放 axis_value**：`len(x.shape) == 3` 等归入 `dims.rule`；`self.shape[1] == mat2.shape[0]` 等归入 `axis_value.rule`。
- **broadcast 关系拆分表达**：rank 关系放 `dims`，逐轴 broadcast 关系放 `axis_value`。输出 shape 推导应放在 `out` 的 shape 约束中。
- **reduce/归约轴的 shape 推导须完整表达**：`keepDim=true/false`、`dim=[]` 时对所有维度 reduce 等约束须在 `out` 的 shape 中显式表达。
- **条件化 shape 约束须标注生效条件**：如"非连续 Tensor 时 len(input.shape) <= 8"须在 rule 中明确条件。

#### 职责分离
- `dims` 与 `axis_value` 要分开承载职责：`dims` 只做维度数量约束，`axis_value` 只做轴值与对应关系。
- 同一个参数的同一 platform 不应出现重复结构，必要时在 `rule` 内并列表达多个条件。

## data_types 提取规则

### 基本规则
- 提取不同平台对数据类型的支持信息。
- 结果应为一个列表，其中每个元素包含特定于平台的数据类型约束。
- 使用"platform"来定义平台。
- 使用"types"来定义该平台允许的数据类型。
- `types` 中只保留文档明确支持的类型，不要凭经验补充、泛化或改写名称。
- 如果文档同时给出了"支持类型枚举"和"与其他参数同 dtype"的关系，这两类信息都不能遗漏；不要只保留枚举而丢掉显式的跨参数 dtype 对齐语义。
- 当当前结构无法仅在 `types` 中表达"与另一参数同 dtype"时，应在可解析规则或 `inter_parameter_constraints` 中补齐，而不是静默丢弃该关系。
- 如果不同平台支持类型不同，需要拆成多个平台元素；如果所有平台一致，可以合并为统一的平台描述。
- 如果没有提到特定约束，则使用空列表。
- **types 仅保留合法 dtype 枚举**：删除 `COND:` 条件、`output.dtype == input.dtype`、`same_format_as_x` 等关系表达式。跨参数 dtype 关系必须用独立约束字段或扩展 schema 表达。
- **统一 FLOAT/FLOAT32 命名**：同一结构化结果中 `FLOAT` 与 `FLOAT32` 命名须统一，避免混用。
- **按平台合并为一条 data_types 记录**：同一平台不应出现多条 `data_types` 条目；条件化限制应改为条件约束表达。
- **函数签名类型不写入 data_types**：C/C++ 形参类型（如 `uint64_t`、`aclrtStream`）属于 `functions` 模块，不应写入 `parameter_constraints.data_types`。

### 跨参数 dtype 关系约束
- **补充可机器判定的跨参数 dtype 对齐/推导关系**：如 `out.dtype == self.dtype` 等一致性要求，不能只保留类型枚举，必须补充可机判的显式关系约束。
- **条件化 dtype 关系须显式表达**：如"input 为 INT8 时 out 必须为 FLOAT"等条件关系，须用条件化约束显式表达。
- **禁止的混合 dtype 组合须显式表达**：如 A2/A3 平台禁止 BFLOAT16+FLOAT16 混合输入等规则，须在 dtype 约束中显式列出禁止组合。
- **dstType 与 y.dtype 的映射关系须独立表达**：不应写入 `types`，须用独立可判定约束表达。

## memory 提取规则

### 基本规则
- 提取有关内存布局约束的信息。
- 结果应为一个列表，其中每个元素包含特定平台的内存约束。
- 使用"platform"字段定义平台。
- 使用"discontinuous"字段表示是否支持非连续内存布局。
- `discontinuous` 的取值必须是布尔值：
  - `true` 表示支持非连续布局
  - `false` 表示不支持非连续布局
- 如果未提及内存布局约束，则使用空列表。
- 从算子说明文档的"参数说明"表格中提取"非连续Tensor"列信息：
  - 有 `√` 标志 → `discontinuous: true`
  - 有 `×` 标志或无标志 → `discontinuous: false`
- **非 Tensor 参数不生成 memory/discontinuous 约束**：`workspaceSize`、`keepDim` 等非 Tensor 参数，`memory` 应设为 `[]` 或 `null`。
- **仅文档明确时才填写 discontinuous**：只有文档明确写出支持或不支持非连续 Tensor 时，才填写具体的布尔值。
- 非 Tensor 参数通常不应落到 `memory`，若文档只给出标量/索引等参数，`memory` 置为 `[]` 或 `null`。

## allowed_values 提取规则

### 基本规则
- 提取有关不同平台上参数允许值的信息。
- 结果应为一个列表，其中每个元素包含特定平台的允许值。
- 使用"platform"字段定义平台。
- 使用"value"字段定义该平台的允许值。
- 值数组应满足以下条件：
  - 对于数值参数：直接列出允许的数值，如文档描述为："取值为0和1"，则数组表示为：[0, 1]，
  - 对于范围参数：只允许使用表示范围的表达式，如文档描述为："0到epWorldSize-1之间的整数"，则数组表示为[[0,epWorldSize-1]]
  - 对于分段参数：如果某个参数的取值为分段表示的，则数组中用多个元素表示，如文档描述为："取值为0到7168之间的整数或8192到10240之间的整数"，则数组中表示为[[0,7168], [8192, 10240]]，需关注区间边界的取值。
  - 表达式必须是有效的Python表达式，并表示张量中元素的允许值集合。
  - 不得包含编程语言的条件表达式（如if-else、三元运算符、变量引用如range(2,769)）或函数调用。
  - 不得包含任何Python无法识别的内容，例如中文字符或无效的Python表达式。
  - 如果没有读取到 `value` 的取值约束，则 `value` 必须是空数组 `[]`。
  - 不要把 `format`、数据格式名称或 layout 信息填进 `value`。

### 规范化要求
- **allowed_values.value 仅放具体允许值**：不放关系表达式（如 `out.dtype == self.dtype`）。
- **条件化禁值/允许值须显式表达**：如"divMode=True 且 scales2Optional 存在时禁止 0"等，须拆分为条件+禁值。
- **空指针/None 统一为 `空指针` 或 `nullptr`**。
- **value 字段改为 [] 表达"值为空/无约束"**：当 `allowed_values.value` 仅承载推导关系而非具体枚举时，应改为 `[]`。

## not_allowed_values 提取规则

### 基本规则
- 提取有关不同平台上参数禁止值的信息。
- 结果应为一个列表，其中每个元素包含特定平台的禁止值。
- 使用"platform"字段定义平台。
- 使用"value"字段定义该平台的禁止值。
- 值数组应满足与"allowed_values"相同的条件。
- 表达式必须是有效的Python表达式，并表示张量中元素的禁止值集合。
- 不得包含编程语言的条件表达式（如if-else、三元运算符、变量引用如range(2,769)）或函数调用。
- 不得包含任何Python无法识别的内容，例如中文字符或无效的Python表达式。
- 如果未指定禁止值，则使用空列表。

### 规范化要求
- **not_allowed_values.value 仅放具体禁止值**：不放条件关系字符串（如 `value < 0`）。条件性禁值须用可判定的条件约束表达。

## 两段式接口参数处理

### executor 处理
- **executor 按接口阶段拆分或补充来源约束**：第一段接口的 `aclOpExecutor**`（出参）与第二段接口的 `aclOpExecutor*`（入参）不应合并为同一参数条目。

### workspaceSize 处理
- **workspaceSize 按接口阶段区分**：补充"第二段入参必须使用第一段返回的 workspaceSize"的来源约束。

### 同名参数处理
- **同名参数按接口场景拆分或标注**：Inplace 系列算子中 `self` 与 `selfRef` 的约束不应合并。

## Additional Notes

- "platform"字段的取值应与文档中的平台描述一致；不同平台约束不同就显式拆分，不要混在一个元素中。
- 不要在 `shape.rule` 中混入 `dtype`、`format`、`memory`、平台说明或长段自然语言；每个字段只承载自己的语义边界。
- 所有表达式字段都必须输出字符串；`shape.rule` 没有内容时使用空字符串 `""`，不要使用 `null`。
- 任何来源于文档原句的约束都要尽量保持机器可判定，避免只写模糊自然语言描述。
- 如果文档明确给出空 tensor、禁止值、允许值、平台差异、可选参数存在条件等约束，都要落到对应字段，而不是依赖人工理解。
- 涉及跨参数的整体关系时，可以在当前参数字段中保留必要的可解析约束，同时在 `inter_parameter_constraints` 中补充更完整的多参数关系表达。
- 如果某字段没有内容，使用与 schema 一致的空结构，不要凭空补写文档中不存在的规则。
- 不要从单参数类型列表反推跨参数 dtype 关系；跨参数关系必须有明确来源并按 `inter_parameter_constraints` 或专用可解析字段表达。
- 对于有条件生效的 shape/dtype/value 约束，在表达式中显式体现条件，避免仅靠注释文本。
- 两段式接口下，`executor`/`workspaceSize` 来源关系要与接口阶段对应，不要混写同一参数。
- **可选参数间的依赖关系须显式表达**：如"仅当 smoothScale1Optional 存在时才允许 smoothScale2Optional 存在"等，须用可判定约束显式表达。
