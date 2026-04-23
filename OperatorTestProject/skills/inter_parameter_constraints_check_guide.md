# inter_parameter_constraints 检验规则

本文件用于同时校验 `inter_parameter_constraints` 的结构和业务语义。

`inter_parameter_constraints` 来源于 `src/common_model_definition.py` 中 `OperatorRule` 的一个属性，但本提示词不能依赖外部 Python 文件内容，因此相关模型定义直接内联如下。

## 模型定义

### 顶层字段来源

```python
class OperatorRule(BaseModel):
    inter_parameter_constraints: List["InterParamConstraint"] = Field(default_factory=list, description="参数间约束")
```

### 相关枚举

```python
class InterConstraintsRuleType(str, Enum):
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
```

### inter_parameter_constraints 相关模型

```python
class InterParamConstraint(BaseModel):
    type: InterConstraintsRuleType
    params: List[str]
    expr: str
    description: str = ""
    source: Optional[str] = None
    target: Optional[List[str]] = None

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
- 枚举字段取值是否在允许范围内
- 是否存在模型未定义的额外字段

### 2. 业务校验

检查以下内容：
- 是否正确识别文档中的参数间约束
- `type` 是否选对
- `params` 是否覆盖了参与该约束的关键参数
- `expr` 是否正确表达了文档中的约束关系
- `description` 是否合理描述该约束
- `source` / `target` 是否在适合的约束类型下被正确使用

## 校验流程

按以下顺序执行：
1. 先做结构校验
2. 结构校验通过后，再做业务校验
3. 任意一类校验失败，都应判定为不通过

## 顶层结构校验

- `inter_parameter_constraints` 必须是数组
- 数组中的每个元素都必须是对象
- 每个元素只允许包含以下字段：
- `type`
- `params`
- `expr`
- `description`
- `source`
- `target`

字段要求：
- `type`：枚举字符串
- `params`：字符串数组
- `expr`：字符串
- `description`：字符串
- `source`：字符串或 `null`
- `target`：字符串数组或 `null`

业务要求：
- 不要凭空生成文档中不存在的参数间约束
- 如果文档没有任何参数间约束，则该字段应为 `[]`

## InterParamConstraint 结构校验

- 每个约束对象只允许包含：
- `type`
- `params`
- `expr`
- `description`
- `source`
- `target`

字段要求：
- `type` 必须是以下之一：
- `shape_broadcast`
- `shape_choice`
- `shape_equality`
- `shape_dependency`
- `shape_value_dependency`
- `presence_dependency`
- `type_dependency`
- `type_equality`
- `value_dependency`
- `format_equality`
- `params` 必须是字符串数组
- `expr` 必须是字符串
- `description` 必须是字符串，允许为空字符串
- `source` 必须是字符串或 `null`
- `target` 必须是字符串数组或 `null`

补充约束：
- 所有对象都不允许出现额外字段
- `params` 中的每个元素都必须是参数名字符串
- `target` 中的每个元素都必须是参数名字符串
- `expr` 不允许为 `null`；如果没有内容，只能是空字符串 `""`

## 业务校验

### 1. 约束来源识别

**基本提取来源校验**：
- 从算子说明文档的”参数说明”表格中的”使用说明”列提取参数间关系。
- 从算子说明文档的”约束说明”章节提取跨参数约束。
- 如果同一约束在多个位置重复出现，应按同一条约束理解，不要重复生成多条完全等价的约束。

**文档来源严格性校验**：
- 仅保留文档明确给出的跨参数约束；不接受从公式语义、示例代码、返回值错误码表、常识推断补造的约束。
- 不接受从单参数 dtype 列表反推跨参数转换矩阵；仅当文档明确给出 dtype 转换关系/映射表时，才补充 `type_dependency`。
- 不接受将功能说明/计算公式/示例代码中的 out shape 推导关系写入；计算公式语义应放到专门描述算子计算逻辑的字段中。
- 不接受将返回值错误码表中的校验场景写入。
- 不接受引入文档未出现的伪属性（如 `workspace.size`）或未明确说明的内存容量判断。
- 空Tensor传播规则只保留文档明确支持空Tensor的输入场景，不接受为文档未明确支持空Tensor的参数补造传播约束。

### 2. type 校验

支持的约束类型及其语义如下：

- `shape_broadcast`
- 表示 target 中的参数与 source 参数满足 broadcast 关系

- `shape_choice`
- 表示某参数的 shape 可以在多个候选参数之间选择其一作为参考

- `shape_equality`
- 表示多个参数 shape 完全相同

- `shape_dependency`
- 表示 shape 的维度值之间存在依赖关系

- `shape_value_dependency`
- 表示 shape 中具体轴值之间存在依赖关系

- `presence_dependency`
- 表示参数是否出现之间存在依赖或共存关系

- `type_dependency`
- 表示参数数据类型之间存在依赖关系

- `type_equality`
- 表示多个参数数据类型一致

- `value_dependency`
- 表示参数值或 tensor 元素值之间存在依赖关系

- `format_equality`
- 表示参数的数据格式必须相同或一致

业务要求：
- `type` 必须与文档中的真实约束语义匹配。
- 不允许用一个宽泛类型替代更准确的类型。
- 如果文档表达的是”相等”，不要误写成 dependency。
- 如果文档表达的是”broadcast”，不要误写成 equality。
- 如果文档表达的是”由某个参数推导出两种合法 shape 形式”，通常应判定为 `shape_dependency`，而不是 `shape_choice`。
- 单参数约束、公式解释或纯说明性文字不应抽入本模块。
- 只抽取真正依赖多个参数的约束；单参数约束、公式解释或功能性描述不要放进 `inter_parameter_constraints`。
- 先判断语义再选 `type`，不要把 dependency、choice、equality、value_dependency 等关系混用。
- 像”shape 为 [B,H] 或 [B,1,H]，且由 x 推导”这类关系，本质上是 `shape_dependency`，不是 `shape_choice`。
- “与另一参数同 dtype / 同 format / 同 shape”这类显式跨参数关系，应优先在本模块中选择语义最准确的 `type` 表达。
- 形状相关约束按语义进一步细分：轴值关系优先 `shape_value_dependency`，广播关系使用 `shape_broadcast`。
- 同时为空/不为空类规则优先使用 `presence_dependency`。

**约束类型修正校验**：
- shape 轴值与 dim 之间的约束：`type` 应为 `shape_value_dependency`，而非 `shape_dependency` 或 `value_dependency`。
- broadcast 关系：`type` 可设为 `shape_broadcast`。
- “同时为空或同时不为空”的逻辑：使用 `presence_dependency`。

### 3. params 校验

- `params` 应包含参与该约束的关键参数。
- 顺序可以不作为强制要求，但集合语义必须正确。
- 不要遗漏关键参数。
- 不要加入与该约束无关的参数。

业务要求：
- 对于二元约束，`params` 通常至少应包含相关的两个参数。
- 对于多参数相等或依赖关系，`params` 应覆盖所有参与方。
- `params` 应覆盖 `expr` 中出现的关键参数引用，不接受 `expr` 和 `params` 脱节。
- 如果约束需要额外中间符号，应先确认该符号是否应由 `other_parameters` 定义，再校验本模块的 `params`。

### 4. expr 校验

#### 结构要求

- `expr` 必须是合法的 Python 表达式。
- `expr` 必须是字符串，不能是 `null`；如果结果没有写出表达式，也只能使用空字符串 `""`。
- `expr` 的语义结果必须是布尔值。
- 不允许使用伪代码。
- 不允许使用 `implies`。
- 不允许把平台值写成 `expr` 内的判断条件。

#### 语义要求

- `expr` 必须准确表达文档中的参数间逻辑关系。
- 约束语义判断应按语义理解，不应只做字符串级别比较。
- 语义等价的表达式应视为有效。
- 像”同 dtype / 同 format / 同 shape”这类关系，应校验表达式是否真正覆盖了所有参与参数，而不是只校验部分对象。

**修正规则校验**：
- `expr` 必须是合法的 Python 布尔表达式；无法精确表达时，不要保留空 `expr` 的约束。
- `expr` 应精确校验各轴值，而不是仅比较 `len(out.shape)`。
- broadcast 判定应使用标准语义：逐维满足”相等或其中一个为 1”，不要直接用 `max(...)` 推导。
- `expr` 中不要混入单参数限制条件（如 rank 限制、`len(kernelSize) in (1, 3)` 等），仅保留跨参数关系。
- `expr` 中不要使用恒为 True 的表达式。
- `expr` 应覆盖所有分支场景，不要用 `else True` 放过未覆盖的分支。
- 不要在 `expr` 中引用跨调用历史返回值（如前次返回的 fd），仅保留单次调用内可判定的条件。

例如，下列表达式语义等价：
- `A == B == C`
- `A == B and B == C`

#### 特殊逻辑要求

- 对于“如果 A 则 B”语义，应使用合法 Python 布尔表达。
- 推荐表达方式：
- `(B) if (A) else True`
- 或其他明确等价、合法、可执行的布尔表达式
- 对于“当且仅当”语义，可使用：
- `(A) == (B)`

#### 其他要求

- 如果涉及生成器表达式，应包裹在 `all()` 或 `any()` 等函数调用中。
- 不允许使用 `lambda`。
- `expr` 不能只保留宽泛的自然语言概括，而应落成可执行、可判定的逻辑。
- 不要把本应删除的冗余关系、单参数限制或平台差异错误塞进 `expr`。

### 5. description 校验

- `description` 应简洁描述该约束含义。
- 可为空字符串。
- 如果填写了内容，应与 `type` 和 `expr` 语义一致。
- 不应写成与约束无关的解释性文本。
- 不应把单参数限制、平台说明或其他模块内容混进当前约束描述。

### 6. source / target 校验

- `source` / `target` 主要用于表达方向性明显的约束。
- 例如 broadcast、choice、dependency 一类约束，通常更适合使用 `source` / `target`。
- 对于纯对称约束，如 `type_equality`、`shape_equality`，可以为空。

业务要求：
- 如果文档表达中有明显”以谁为基准””谁约束谁”的方向，应正确填写 `source` 和 `target`。
- `source` 应为单个参数。
- `target` 应为一个或多个被约束参数。
- 对 dependency 类约束，如果方向明确却仍把 `source` / `target` 置空，应判为业务错误。
- 如果不存在明显方向性，可为 `null`。

**方向性修正校验**：
- `target` 至少补充为 `[“out”]`（或实际被约束的参数），不要留空或置为 null。
- 多来源依赖时：可拆分约束，或将 `source`/`target` 同时置为 `null`，通过 `params` + `expr` 表达多参数联合依赖。
- 不要把单个参数标成唯一 `source` 来表达多参数共同约束关系：应置 `source` 为 null 或拆分约束。
- `source` 应设为基准/触发参数，`target` 应设为被约束参数（如 `source=”self”`, `target=[“dim”]`）。
- 所有 dependency 类约束都应优先补齐 `source`，不要把明显有方向的关系留成 `null`。
- 所有 dependency 类约束都应优先补齐 `target`，体现”谁约束谁、谁依赖谁”的方向。

## 边界情况处理

- 如果文档中存在多种表述方式但语义相同，只生成一条语义正确的约束即可。
- 如果同一类约束可以写成不同但等价的 Python 表达式，应视为允许。
- 即使 `expr` 中包含冗余比较，只要语义正确，也应视为满足约束。
- 如果文档没有足够信息支撑某条参数间约束，不要凭空补造。

## 特定约束模式校验

**补充规则校验**：
- **dim 合法范围约束**：应补充 `value_dependency`，`params` 至少包含 `["self", "dim"]`，`expr` 为 `(-len(self.shape) <= dim <= len(self.shape) - 1)`。
- **BFLOAT16 下 dim 轴长度约束**：当 `self.dtype` 为 BFLOAT16 时，`self.shape[dim] != 1`；使用 `shape_value_dependency`。
- **keepdim 对 out.shape 的影响**：`keepdim=False` 时 out.shape 等于删除 dim 对应轴后的 self.shape；`keepdim=True` 时保留原轴数且被 reduce 轴为 1。
- **broadcast 关系约束**：`self` 与 `other` 的 shape 必须满足 broadcast 关系，`out.shape` 等于广播结果 shape。
- **target 元素合法索引范围约束**：`0 <= value < self.shape[-1]`，允许等于 `ignoreIndex`；`params` 至少包含 `["self", "target", "ignoreIndex"]`。
- **type_dependency 约束**：表达 `self.dtype` 必须可转换为 `out.dtype` 的关系。
- **stride 默认等于 kernelSize**：当 `len(stride) == 0` 时，stride 的有效步长等于 kernelSize。
- **padding 不大于 kernelSize 一半**：`all(2 * p <= k for p, k in zip(padding, kernelSize))`。
- **卷积 output shape 推导**：应新增 `shape_value_dependency`，按 1D/2D/3D 场景表达 output 各空间维的 infershape 公式。
- **两段式接口 workspaceSize/executor 一致性**：应补充 `value_dependency`，表达第二段的 workspaceSize/executor 必须等于第一段返回值。

**条件/分支约束处理校验**：
- 带条件的约束应显式表达生效前提：如仅在 `isbias=true`、`bidirection=true`、`packed=true`、`reduction==1`、特定 dtype 等条件下才生效的约束，应在 `expr` 中加入条件判断。

**约束删除/迁移校验**：
- 删除文档未明确给出的约束：如 minOut/maxOut shape 约束，只有文档明确说明其关系时才保留。
- 删除仅表达计算公式的约束：计算公式应放入描述功能/公式的对应结构化字段。
- 删除仅表达单参数限制的跨参数约束：如"当前仅支持 axis=-1"应放到单参数规则中。
- 删除仅表达默认值/默认行为的约束：应放到参数说明或默认值逻辑位置。
- 删除凭空写死具体 dtype 组合的约束：若文档不足以支撑完整推导规则，应删除。

## 通用规则

- 结构校验和业务校验都应严格执行
- 不要因为字段“看起来合理”就忽略结构错误
- 不要因为结构合法就忽略业务错误
- 语义判断要以文档真实含义为准，而不是只看字段表面形式

## 校验结论原则

- 同时报告结构错误和业务错误
- 如果结构不合法，直接判定为不通过
- 如果结构合法但业务内容与文档不一致，也应判定为不通过
- 只有结构和业务都通过，才能判定 `inter_parameter_constraints` 校验通过
