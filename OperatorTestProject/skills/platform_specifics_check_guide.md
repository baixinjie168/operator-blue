# platform_specifics 检验规则

本文件用于同时校验 `platform_specifics` 的结构和业务语义。

`platform_specifics` 来源于 `src/common_model_definition.py` 中 `OperatorRule` 的一个属性，但本提示词不能依赖外部 Python 文件内容，因此相关模型定义直接内联如下。

## 模型定义

### 顶层字段来源

```python
class OperatorRule(BaseModel):
    platform_specifics: List["PlatformSpecific"] = Field(
        default_factory=list,
        description="平台特定规则"
    )
```

### platform_specifics 相关模型

```python
class PlatformSpecific(BaseModel):
    platform: str
    description: str
    constraint_detail: str

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
- 是否存在模型未定义的额外字段

### 2. 业务校验

检查以下内容：
- 是否正确识别了文档中的平台特定约束
- 是否遗漏了文档中明确写出的平台差异
- 是否把全局约束误放进了 `platform_specifics`
- `platform`、`description`、`constraint_detail` 是否准确表达了对应平台下的限制或要求
- 多平台、多条规则是否被正确拆分和归并

## 校验流程

按以下顺序执行：
1. 先做结构校验
2. 结构校验通过后，再做业务校验
3. 任意一类校验失败，都应判定为不通过

## 顶层结构校验

- `platform_specifics` 必须是数组
- 数组中的每个元素都必须是对象
- 如果文档中没有任何平台特定约束，则该字段应为 `[]`

业务要求：
- `platform_specifics` 只包含“仅对特定平台成立”的约束、限制、能力差异或额外要求。
- 如果文档显式给出了支持矩阵或 `是否支持：√/×` 这类平台差异，即使约束很简单，也不应遗漏。
- 对所有平台都成立的全局规则不应放入本模块。

## PlatformSpecific 结构校验

- 每个元素只允许包含以下字段：
- `platform`
- `description`
- `constraint_detail`

字段要求：
- `platform`：字符串
- `description`：字符串
- `constraint_detail`：字符串

补充约束：
- 所有对象都不允许出现未定义字段
- `description` 可以是多行字符串
- `constraint_detail` 不能为空值类型；如果没有可解析表达式，应使用空字符串 `""`

业务要求：
- 每个元素应只表达一条相对独立的平台规则
- 同一平台可以出现多条规则，但每条规则应语义清晰、边界明确
- 如果同一条规则同时适用于多个平台，允许把多个平台名称放在一个 `platform` 字符串中
- 不要把多个完全无关的规则硬塞到同一个元素里

## platform 校验

### 结构校验

- `platform` 必须是字符串

### 业务校验

### 1. 识别 platform_specifics 的来源

- 优先从“约束说明”“限制”“注意事项”“平台差异”“硬件要求”“支持矩阵”等章节提取。
- 也要检查参数说明、功能说明、补充说明中的平台特定句子。
- 凡是出现“某平台支持/不支持”“某平台仅支持”“某平台要求”“某平台下无效”等表述，都应考虑是否属于 `platform_specifics`。
- 单参数数据类型支持表只有在不同平台确实存在差异时才属于本模块；如果所有支持平台一致，应要求移到 `parameter_constraints`。

### 2. 与全局规则的边界

- 只在特定平台成立的限制，放入 `platform_specifics`。
- 对所有平台都成立的输入输出约束、shape 关系、dtype 关系，不应放入 `platform_specifics`。
- 校验时要区分平台差异和全局规则；对所有已支持平台都成立的 dtype、shape、format、memory、broadcast 约束，不应出现在 `platform_specifics`。
- 不要把本应写进 `parameter_constraints`、`inter_parameter_constraints` 或 `dtype_map` 的全局规则误归到这里。

### 3. 规则拆分与合并

- 如果同一平台有多条独立限制，应拆成多条记录。
- 如果多个平台共享同一条规则，可以合并为一条记录。
- 如果一段描述里同时包含多条独立要求，应按语义拆分，避免一条记录过于混杂。
- 同一平台下语义完全重复的记录不要重复出现。
- 支持矩阵、dtype 限制、模式开关等不同语义的规则，不应强行挤进同一条记录。
- 如果同一平台下既有“是否支持”又有 dtype 不支持、模式开关、字节对齐等限制，应检查是否拆分清楚；混成一条导致语义边界不清时应报错。

### 4. 平台差异表达

- 数据类型支持差异、尾轴字节要求、某些输出无效、组合支持矩阵等，都属于典型的平台特异性规则。
- 文档中如果明确按平台给出支持矩阵，应在 `description` 中保留完整信息。
- 产品支持矩阵要检查是否补全所有明确列出的平台，尤其是不支持平台；遗漏 `Atlas 200I/500 A2 推理产品` 等明确不支持项应报错。
- `constraint_detail` 能表达时再表达，不能稳定表达时宁可留空字符串，也不要编造错误规则。
- `description` 应保留平台规则的人类可读语义，`constraint_detail` 只写简洁可判定表达式；长段自然语言、JSON、数组或多个混杂规则都不合格。
- 如果只是平台支持状态，允许 `constraint_detail` 使用统一的支持状态表达或空字符串；不要把 `supported == true/false` 当成 dtype、shape 等参数约束。
- 如果文档已经明确列出多个平台的支持/不支持状态，校验时应检查结果是否遗漏任何已明确给出的平台项。

## 通用规则

- 所有对象都不允许出现未定义字段
- 不要在 `platform_specifics` 中写入全局规则
- 不要遗漏文档中明确声明的平台差异
- 如果文档没有平台特异性信息，则 `platform_specifics` 必须为 `[]`

## 校验结论原则

- 结构校验和业务校验都应严格执行
- 结构不合法时，直接判定不通过
- 结构合法但业务语义错误时，也应判定不通过
- 只有结构和业务都通过，才能判定 `platform_specifics` 校验通过
