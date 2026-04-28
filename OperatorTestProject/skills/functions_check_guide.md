# functions 检验规则

本文件用于同时校验 `functions` 的结构和业务语义。

`functions` 来源于 `src/common_model_definition.py` 中 `OperatorRule` 的一个属性，但本提示词不能依赖外部 Python 文件内容，因此相关模型定义直接内联如下。

## 模型定义

### 顶层字段来源

```python
class OperatorRule(BaseModel):
    functions: List["OperatorFunction"] = Field(..., description="函数列表")
```

### 相关枚举

```python
class ParameterRole(str, Enum):
    INPUT = "input"
    OUTPUT = "output"
```

### 参考类型枚举

`Parameter.type` 在模型中定义为 `str`，不是枚举类型；但 `src/common_model_definition.py` 中提供了常见参数类型枚举，可作为结构和业务校验时的参考取值范围：

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

### functions 相关模型

```python
class Parameter(BaseModel):
    name: str
    role: ParameterRole
    type: str
    is_optional: bool
    description: str
    format: Optional[Union[List[str], str]] = None

    model_config = {"extra": "forbid"}


class OperatorFunction(BaseModel):
    api_name: str
    description: str
    parameters: List[Parameter]

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
- `format` 字段是否满足 `null` / `string` / `string[]`
- 是否存在模型未定义的额外字段

### 2. 业务校验

检查以下内容：
- 是否正确识别文档中的函数原型区域
- 是否提取了应保留的核心函数
- `api_name` 是否与文档中的函数名一致
- `description` 是否正确概括对应函数
- `parameters` 是否与函数原型和参数说明一致
- `name`、`role`、`type`、`is_optional`、`description`、`format` 是否正确反映文档内容

## 校验流程

按以下顺序执行：
1. 先做结构校验
2. 结构校验通过后，再做业务校验
3. 任意一类校验失败，都应判定为不通过

## 顶层结构校验

- `functions` 必须是数组
- 数组中的每个元素都必须是对象
- 每个元素只允许包含以下字段：
- `api_name`
- `description`
- `parameters`

字段要求：
- `api_name`：字符串
- `description`：字符串
- `parameters`：数组

业务要求：
- `functions` 应只包含文档中应保留的核心函数
- 不要凭空生成文档中不存在的函数
- 如果文档完全没有任何函数原型信息，则 `functions` 应为 `[]`

## OperatorFunction 结构校验

- 每个 `OperatorFunction` 对象只允许包含：
- `api_name`
- `description`
- `parameters`

字段要求：
- `api_name`：非空字符串
- `description`：字符串
- `parameters`：数组

业务要求：
- `api_name` 应对应文档中的真实函数名
- `description` 应描述当前函数，而不是整个算子或其他函数
- `parameters` 应与该函数原型对应，不应混入其他函数的参数

## Parameter 结构校验

- `parameters` 中的每个元素都必须是对象
- 每个元素只允许包含以下字段：
- `name`
- `role`
- `type`
- `is_optional`
- `description`
- `format`

字段要求：
- `name`：字符串
- `role`：只能是 `"input"` 或 `"output"`
- `type`：字符串
- `is_optional`：布尔值
- `description`：字符串
- `format`：`null`、字符串或字符串数组

补充约束：
- 所有参数对象都不允许出现额外字段
- `format` 只能用于表达数据格式，不要塞入其他业务信息
- 如果没有格式信息，`format` 应为 `null`

## 业务校验

### 1. 识别函数原型区域

- 优先级顺序查找以下章节或关键词，任一命中即可：
- 函数原型
- `## 接口说明`
- `## API 说明`
- `## 函数签名`
- 执行函数
- `## aclnnXXX`
- `## aclopXXX`
- 包含 `aclnn`、`aclop`、`ti_` 等前缀的代码块或标题
- 表格标题为“函数原型”或“参数说明”的表格上方内容

常见模式至少包括以下之一：
- `函数名(参数列表)`
- `返回类型 + 函数名 + (参数列表)`
- 代码块中以 `aclnn` 开头的声明

### 2. api_name 校验

提取优先级从高到低：
1. 小节标题，如 `### aclnnAdd`
2. 函数原型代码块第一行中的函数名
3. 参数表格上方的函数声明行

业务要求：
- 只提取最主要的执行函数和必须的前置函数，例如 `GetWorkspaceSize`
- 不要包含重载版本、内部实现函数、废弃函数
- 保留完整函数名，包括 `aclnn` / `aclop` 前缀

可接受示例：
- `aclnnMatMul`
- `aclnnAddLayerNormGetWorkspaceSize`
- `aclnnFlashAttention`

### 3. description 校验

- 优先使用函数原型下方紧邻的一段话或官方一句话描述。
- 如果没有独立描述，则从当前函数参数表格上方说明文字中提炼。
- 控制在 1 到 3 句，保持原文语义。
- 如果文档完全没有函数描述，填写：
  - `文档中未提供函数描述`
- `description` 必须严格对应当前函数，不要混入算子整体功能、计算公式、其他函数说明或当前函数原型中不存在的参数。
- 如果文档没有提供独立函数级描述，允许保守使用默认占位，也不接受凭空扩写函数语义。
- 二段式执行函数的 `description` 应只描述当前执行函数；如果写成“第一段接口返回 workspace”，应报错，因为第一段通常返回 `workspaceSize` 和 `executor`，`workspace` 由调用方申请。

### 4. parameters 校验

- 从文档中的“函数原型”以及 `{OperatorName}GetWorkspaceSize` 相关参数表中提取参数
- 如果文档同时给出函数签名和参数说明，优先以参数说明表为准，函数签名作补充

#### 4.1 name

- 直接从“参数名”列或函数参数列表中的变量名提取
- 保持原文大小写
- 参数名、参数数量、参数顺序必须逐项对齐当前函数原型和对应参数说明表；`keepdim` 被写成 `keepDim`、多出一个参数、少一个参数都应报错。

示例：
- `x`
- `weight`
- `epsilon`
- `executor`

#### 4.2 role

优先根据文档中的明确标注判断：
- 列标题包含“输入/输出”或 “I/O” 时：
  - `输入` / `Input` / `I` -> `"input"`
  - `输出` / `Output` / `O` -> `"output"`

若文档没有明确标注，必须结合函数原型、参数说明和调用语义综合判断：
- 不允许只凭参数名进行机械猜测。
- `role` 只能是 `input` 或 `output`；如果某个参数兼具输入和输出语义，校验时应要求结果统一记为 `input`。
- `workspace`、`workspaceSize`、`executor`、`stream` 等流程参数，应严格按文档中的责任边界判定。
- 对“计算输入/计算输出”或原地更新参数，如果 schema 只能使用单值 `role`，应检查结果是否记为 `input`，并在 `description` 中保留双向或原地更新语义。
- 原地接口的 `selfRef`、`inputRef` 等只能作为一个形参出现；如果结果拆成 input/output 两个参数或重复新增同名 output 参数，应判定为业务错误。
- 当 schema 只有 `input` / `output` 两种 role 时，原地参数可用合法单值 role，但 `description` 必须保留“计算输入|计算输出”或原地写回语义；若原文没有该语义，也不能凭空添加。
- 不接受把明显的返回项误写成 `input`，也不接受把调用方传入的流程参数误写成 `output`。
- `workspace`、`workspaceSize`、`executor`、`stream` 等流程参数的 role 要按当前函数职责校验，不要套用算子整体的输入输出语义。

#### 4.3 type

- 优先从“参数说明”中参数名称后的括号内容提取。
- 或从函数原型中解析，例如：
  - `const aclTensor* x` -> `aclTensor`
  - `uint64_t *workspaceSize` -> `uint64_t`
  - `aclOpExecutor **executor` -> `aclOpExecutor`
- `type` 必须与当前函数原型和参数说明中的基础类型一致，不允许凭经验泛化成其他类型。
- 允许按项目 schema 去掉 `*`、`const`、`struct` 等修饰符，但不允许改变核心类型语义。
- 如果文档明确是 `float*`、`aclTensor*` 这类指针型声明，校验时应确认抽取结果对应的基础类型没有被错误改写。

#### 4.4 is_optional

文档中出现以下任一关键词可判为 `true`：
- 可选
- Optional
- default
- 可为空
- 可以为 nullptr
- 可不传
- 缺省值

业务要求：
- 只有文档明确表达可选、可为空、可不传、可为 `nullptr` 或存在默认值时，`is_optional` 才能为 `true`。
- “支持空Tensor”不等于参数可选；遇到这类描述时，不应据此把 `is_optional` 判为 `true`。
- 参数名中包含 `Optional` 也不必然表示可选，必须结合正文说明校验。
- 如果文档未明确说明可选性，则应为 `false`。

#### 4.5 description

- 直接复用“描述”列或“说明”列的内容
- 可以做轻微去重和压缩，但不能改变核心含义
- 如果参数描述缺失，填写：
  - `文档中未提供该参数说明`

业务要求：
- 描述必须针对当前参数
- 不要填入其他参数的说明

#### 4.6 format

- 从“数据格式”列提取
- 如果文档中数据格式显示为 `-`、空、未提供，则 `format` 应为 `null`
- 如果文档给出单个格式，则使用字符串
- 如果文档给出多个格式，则使用字符串数组
- 单个 `ND` 应为字符串 `"ND"`，文档为 `-` 时应为 JSON `null`，不是字符串 `"null"`。

业务要求：
- `format` 只在文档明确提供数据格式时填写
- 不要把数据类型、shape、约束规则误填到 `format`

## 边界情况处理

- 文档中每个函数通常会有对应参数表，优先提取主执行函数的参数
- 其他函数如果信息明显不完整，可以省略，或只保留确有必要的前置函数
- 存在重载函数时，只提取文档最推荐、最完整描述的那个版本
- 参数表缺失时，可以从函数签名中解析参数，但这属于降级提取
- 如果文档完全没有任何函数原型信息，则 `functions: []`

## 通用规则

- 结构校验和业务校验都应严格执行
- 不要因为字段“看起来合理”就忽略结构错误
- 不要因为结构合法就忽略业务错误
- 不需要校验函数返回值，只校验函数本身及其参数

## 校验结论原则

- 同时报告结构错误和业务错误
- 如果结构不合法，直接判定为不通过
- 如果结构合法但业务内容与文档不一致，也应判定为不通过
- 只有结构和业务都通过，才能判定 `functions` 校验通过
