# dtype_map 检验规则

本文件用于同时校验 `dtype_map` 的结构和业务语义。

`dtype_map` 来源于 `src/common_model_definition.py` 中 `OperatorRule` 的一个属性，但本提示词不能依赖外部 Python 文件内容，因此相关模型定义直接内联如下。

## 模型定义

### 顶层字段来源

```python
class OperatorRule(BaseModel):
    dtype_map: List["DtypeMapItem"] = Field(
        default_factory=list,
        description="数据类型映射表"
    )
```

### dtype_map 相关模型

```python
class DtypeMapItem(BaseModel):
    platform: str
    columns: List[str]
    rows: List[List[str]]

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
- `rows` 是否为二维字符串数组
- 是否存在模型未定义的额外字段

### 2. 业务校验

检查以下内容：
- 是否正确识别了文档中的数据类型组合支持表
- 是否遗漏了文档中明确给出的平台级 dtype 组合表
- 是否把不属于组合映射表的信息误写进了 `dtype_map`
- `platform`、`columns`、`rows` 是否准确表达了原表格语义
- 多平台、多张表是否被正确拆分和归并

## 校验流程

按以下顺序执行：
1. 先做结构校验
2. 结构校验通过后，再做业务校验
3. 任意一类校验失败，都应判定为不通过

## 顶层结构校验

- `dtype_map` 必须是数组
- 数组中的每个元素都必须是对象
- 如果文档中不存在数据类型组合表，则该字段应为 `[]`

业务要求：
- `dtype_map` 只承载“多参数 dtype 组合映射表”
- 如果文档只有单个参数支持哪些类型、或平台支持哪些 dtype 的文字说明，但没有组合表，不要生成 `dtype_map`
- 如果原文没有显式的多参数 dtype 组合表或映射表，而结果根据“单参数支持类型”或“输入输出类型文字关系”拼出了 `dtype_map`，必须判定为业务错误。
- 不要凭空生成文档中不存在的 dtype 组合表

## DtypeMapItem 结构校验

- 每个元素只允许包含以下字段：
- `platform`
- `columns`
- `rows`

字段要求：
- `platform`：字符串
- `columns`：字符串数组
- `rows`：二维字符串数组

补充约束：
- 所有对象都不允许出现未定义字段
- `columns` 可以为空数组，但只有在 `rows` 也为空且确实没有表格时才合理
- `rows` 可以为空数组，但不能是对象、字符串或一维数组

业务要求：
- 每个 `DtypeMapItem` 应对应一张明确的 dtype 组合表，或一张表在某个平台范围下的一个映射结果
- 同一平台下如果存在两张语义不同的组合表，可以拆成多个元素
- 不要把完全无关的两张表强行合并到同一个元素里

## platform 校验

### 结构校验

- `platform` 必须是字符串

### 业务校验

### 1. 识别 dtype_map 的来源

- 优先从“约束说明”“各产品支持数据类型说明”“数据类型组合表”“支持矩阵”等章节识别。
- 重点关注明确列出多参数 dtype 组合、输入输出类型对应或量化映射的表格。
- 如果文档只是文字说明“某参数支持哪些 dtype”，但没有显式组合表或映射表，不应提取为 `dtype_map`。
- 如果文档完全没有显式 dtype 映射/组合表，则 `dtype_map` 必须为 `[]`。

### 2. 与其他模块的边界

- `dtype_map` 负责表达“多参数联合 dtype 组合”或显式类型映射表。
- 单参数的 dtype 支持范围，更适合体现在 `parameter_constraints.data_types`。
- “某参数与另一参数同 dtype”属于跨参数关系，应优先体现在 `inter_parameter_constraints` 或可解析规则中，而不是 `dtype_map`。
- 如果文字描述的是“输入为 INT/BOOL 时输出为 FLOAT”这类条件 dtype 关系，应检查它是否落在类型依赖或参数约束中；出现在 `dtype_map.rows` 中通常是不合格的。
- 平台专属但不是组合表的说明，更适合体现在 `platform_specifics`。
- 不要把本应写进 `parameter_constraints`、`inter_parameter_constraints` 或 `platform_specifics` 的内容误放进 `dtype_map`。

### 3. 表头映射

- 表头应映射成真实参数名，而不是保留自然语言表头。
- 只有真正参与 dtype 组合校验的参数才应进入 `columns`。
- 输出参数如果出现在组合表中，也应进入 `columns`。
- 可选参数如果在表中出现，也应进入 `columns`。
- 不要把“数据类型”“备注”“平台”“序号”等非参数列写进 `columns`。
- 校验 `columns` 时要逐一确认是否为真实函数参数名；备注列、平台列、序号列、自然语言表头都不能进入 `columns`。

### 4. 行内容保真

- 每一行都应忠实反映原表中的一个合法组合。
- 允许保留表中出现的重复语义平台拆分，但不允许凭空合成新组合。
- 如果原表明确给出不同平台不同组合，应分别写入各自平台项。
- 不要从单参数支持列表、支持矩阵说明或经验规则反推出新的 `rows` 组合。
- 校验 `rows` 时要确认每一行都能在原始表格中找到对应组合；推导补全、经验补全或合并多平台差异都应报错。
- dtype 名称必须与原文一致，不要擅自改写大小写、缩写或同义写法。

## 通用规则

- 所有对象都不允许出现未定义字段
- `columns` 与 `rows` 的语义必须一一对应
- 不要生成不存在于文档中的 dtype 组合
- 如果文档没有 dtype 组合表，则 `dtype_map` 必须为 `[]`

## 校验结论原则

- 结构校验和业务校验都应严格执行
- 结构不合法时，直接判定不通过
- 结构合法但业务语义错误时，也应判定不通过
- 只有结构和业务都通过，才能判定 `dtype_map` 校验通过
