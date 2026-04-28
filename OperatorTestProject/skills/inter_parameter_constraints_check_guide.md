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
- `params` 必须覆盖 `expr`、`source`、`target` 涉及的全部真实参数；漏掉共同参与推导的 `pad`、`dim`、`tensors` 等应判定为不完整。

## 业务校验

### 1. 约束来源识别

- 从算子说明文档的“参数说明”表格中的“使用说明”列提取参数间关系
- 从算子说明文档的“约束说明”章节提取跨参数约束
- 如果同一约束在多个位置重复出现，应按同一条约束理解，不要重复生成多条完全等价的约束

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
- 如果文档表达的是“相等”，不要误写成 dependency。
- 如果文档表达的是“broadcast”，不要误写成 equality。
- 如果文档表达的是“由某个参数推导出两种合法 shape 形式”，通常应判定为 `shape_dependency`，而不是 `shape_choice`。
- 具体轴值关系要检查 `type` 是否准确：拼接轴求和、非拼接轴相等、输出 rank 与输入 rank 一致等应优先视为 `shape_value_dependency`。
- 单参数约束、公式解释或纯说明性文字不应抽入本模块。

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
- 像“同 dtype / 同 format / 同 shape”这类关系，应校验表达式是否真正覆盖了所有参与参数，而不是只校验部分对象。
- `expr` 不能为空泛化占位；`type_dependency`、`shape_dependency`、`value_dependency` 等只要保留，就必须有可判定的 Python 布尔表达式。

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
- 如果文档表达中有明显“以谁为基准”“谁约束谁”的方向，应正确填写 `source` 和 `target`。
- `source` 应为单个参数。
- `target` 应为一个或多个被约束参数。
- 对 dependency 类约束，如果方向明确却仍把 `source` / `target` 置空，应判为业务错误。
- dependency 类约束必须校验方向性；如果文档表达 `out.shape` 由 `self/pad/dim/tensors` 推导，而 `target` 为空或缺少 `out`，应报错。
- 如果不存在明显方向性，可为 `null`。

## 边界情况处理

- 如果文档中存在多种表述方式但语义相同，只生成一条语义正确的约束即可
- 如果同一类约束可以写成不同但等价的 Python 表达式，应视为允许
- 即使 `expr` 中包含冗余比较，只要语义正确，也应视为满足约束
- 如果文档没有足够信息支撑某条参数间约束，不要凭空补造

## 通用规则

- 结构校验和业务校验都应严格执行
- 不要因为字段“看起来合理”就忽略结构错误
- 不要因为结构合法就忽略业务错误
- 语义判断要以文档真实含义为准，而不是只看字段表面形式
- 如果 dtype 推导、可转换、同 dtype 等跨参数关系被写在 `parameter_constraints.data_types.types`、`allowed_values` 或自然语言描述里，应提示迁回本模块或使用合适的类型关系表达。

## 校验结论原则

- 同时报告结构错误和业务错误
- 如果结构不合法，直接判定为不通过
- 如果结构合法但业务内容与文档不一致，也应判定为不通过
- 只有结构和业务都通过，才能判定 `inter_parameter_constraints` 校验通过
