# platform_specifics 提取规则（平台特异性信息）

本文件定义了从 CANN 算子说明文档中提取 **platform_specifics（平台特异性信息）** 的严格规则，用于描述特定硬件平台（如 Atlas A2、Atlas A3）下的特殊约束或要求。

## 目标输出数据结构
```python 
class PlatformSpecific(BaseModel):
    """平台特定规则模型"""
    platform: str = Field(..., description="平台名称")
    description: str = Field(..., description="平台描述")
    constraint_detail: str = Field(..., description="约束详情表达式")

    model_config = {"extra": "forbid"}

class PlatformSpecificsOutput(BaseModel):
    """顶层模型"""
    platform_specifics: List["PlatformSpecific"] = Field(default_factory=list, description="平台特定规则")

    model_config = {"extra": "forbid"}
```
【输出规则（必须严格遵守，缺一不可）】
1.  输出格式：仅返回纯JSON字符串，无任何多余内容（无解释、无代码块、无换行备注）；
2.  输出范围：只返回【顶层类】的完整结构，自动嵌套填充所有内层类，不单独输出任何内层类（如内层类1、内层类2）；
3.  字段约束：字段名、字段类型、层级结构，必须与上面定义的类完全一致，禁止新增、缺失、修改字段；
4.  类型匹配：严格遵循类型注解（str/int/bool/List/Optional等），空值统一用null（JSON规范），不随意填充无效值；
5.  嵌套要求：所有嵌套结构必须完整，若待处理内容中无相关信息，可选字段填默认值，必填字段（Field(...)）填合理空值（如""、0、[]）。

## 数据结构说明

- `platform_specifics` 是一个数组，每个元素代表一个特定平台下的特殊规则。
- `platform` 字段记录适用的平台名称，必须与文档中的原文完全一致。
- `description` 字段是约束的完整自然语言描述，必须直接复用文档中的原文。
- `constraint_detail` 字段是可选的，用于提供一个机器可解析的 Python 表达式，如果文档中没有明确的表达式，该字段可为 `null` 或空字符串。

## 提取规则（逐字段）

### 1. 识别 platform_specifics 区域

**优先级顺序**查找以下关键词或结构（任一命中即可）：
- 小节标题包含 `限制`、`约束`、`要求`、`特异性`、`特定平台`、`硬件要求`、`支持矩阵` 等。
- 以 `注意`、`说明`、`限制` 开头，且明确提到某个平台的段落或列表项。
- 文本中包含 `是否支持`、`仅支持`、`不支持`、`某平台下无效`、`某平台要求` 等表述。

**判断原则**：
- 只要文档明确存在平台差异、支持矩阵或平台专属限制，就应考虑抽取到 `platform_specifics`。

### 2. platform（平台信息）

**提取规则**：
- 从约束描述的上下文中提取平台名，必须完整保留文档原文中的平台写法。
- 如果同一条规则明确同时适用于多个平台，可以保留为一个组合平台字符串；如果不同平台规则不同，必须拆成多条记录。
- 不要把与平台无关的说明、dtype 组合、自然语言解释写进 `platform`。

**平台合并与拆分规范**：
- **按文档原文合并平台**：若同一条规则适用于多个平台，应将 `platform` 改为文档原文的组合平台字符串，而非拆分为多条单平台记录。
- **若拆分则 description 仅描述当前平台**：`description` 必须只描述当前 `platform` 字段所指定平台的约束。
- **删除重复记录**：同一平台下语义完全重复的条目应合并为一条。
- **不要按平台重复记录全局规则**：所有平台完全一致的约束不应按平台分别写入。

### 3. description（约束描述）

**提取规则**：
- `description` 用于保留人类可读的约束说明，应准确表达该平台下的限制、差异或支持状态。
- 可以复用原文句子或段落，但不要把多个互不相干的平台规则混成同一条描述。
- 当一段文字里同时包含多条独立的平台要求时，应按语义拆分为多条记录，而不是堆成一段长描述。
- **description 参数上下文明确化**：当规则描述某参数的 dtype 支持时，`description` 应包含参数名（如 self、out、other 等）。
- **按参数拆分为多条记录**：若一条记录同时涉及多个参数的 dtype 约束，应按参数拆分为多条独立记录。

### 4. constraint_detail（机器可解析规则）

**基本规则**：
- `constraint_detail` 必须是字符串，用于承载 machine-readable 的约束表达式。
- 只要文档明确给出"是否支持：√/×"或等价支持矩阵，就必须同步生成可判定表达式，例如 `supported == true` 或 `supported == false`。
- 对于 dtype 限制、模式开关、`cubeMathType`、`divMode`、shape/字节对齐等平台条件，只要能稳定转写，就应提供简洁、可判定的表达式。
- 只有当文档内容确实无法稳定抽象为表达式时，`constraint_detail` 才可以为空字符串 `""`；不要因为偷懒而留空。
- `constraint_detail` 必须与 `description` 语义一致，不要把整段自然语言、JSON、字典或数组对象直接塞进去。
- `constraint_detail` 不应引入文档未给出的强制性判断；能力性描述保留为 `description`，避免误判。
- 条件性规则需显式体现条件前提（如可选参数存在、确定性开关开启等）。
- 对平台特有逻辑可用可判定布尔或蕴含表达式（例如 `cubeMathType != 1 or compute_dtype == 'HFLOAT16'`）。

**可解析表达式补充**：
- **支持状态为"支持"时，补充可判定表达式**：`constraint_detail` 应补充为 `supported == true` 等可解析布尔表达式。
- **支持状态为"不支持"时，补充可判定表达式**：`constraint_detail` 应补充为 `supported == false` 等可解析布尔表达式。
- **constraint_detail 与 description 语义必须一致**：补充的表达式必须与 `description` 的语义保持一致。
- **无法稳定表达时置空**：若规则过于复杂或无法稳定转写为可解析表达式，`constraint_detail` 应置为空字符串 `""`。

**精确性与语义规范**：
- **可选参数须加空值保护**：对可选参数，`constraint_detail` 中应使用 `(param is None or param.dtype != 'BFLOAT16')` 形式的条件表达式。
- **不要把"允许"写成"必须"**：如"允许降精度到 HFLOAT16"不应写成强制等式判断，应改为非强制的能力描述或置空 `constraint_detail`。
- **使用条件蕴含式表达 cubeMathType 规则**：如 `cubeMathType != 1 or input.dtype != 'FLOAT' or compute_dtype == 'HFLOAT16'`。
- **不要覆盖文档未明确的分支**：`constraint_detail` 只表达文档已明确说明的正向条件。
- **FLOAT 应统一为 FLOAT32**：`constraint_detail` 中出现的 `'FLOAT'` 应改为 `'FLOAT32'`。

**确定性计算与特殊约束处理**：
- **确定性计算条件须显式表达**：当规则仅在"开启确定性计算时"生效，`description` 应补充该前提，`constraint_detail` 增加确定性开关条件。
- **超时/性能风险用 description 保留**：`constraint_detail` 可写为参数范围表达式或置空，`description` 保留原始风险说明。

## 边界情况处理

### 全局约束移出 platform_specifics
- **删除全局 shape/连续性/格式约束**：全局性的约束应移到全局参数约束字段，`platform_specifics` 仅保留平台间存在差异的内容。
- **删除全局 dtype 一致性/推导规则**：如 `out.dtype == self.dtype` 等全局一致的规则应移到全局约束位置。
- **删除全局参数说明/输出说明**：全局通用的参数语义说明不应出现在 `platform_specifics` 中。
- **仅保留平台特有的 dtype 支持差异**：如 BFLOAT16 仅 A2/A3 支持，全局通用的类型集合移出。

### 无平台特异性信息
- 如果文档中完全未提及任何平台差异，则 `platform_specifics` 应为 `[]`。

### 支持矩阵与差异规则
- 如果文档显式列出平台支持矩阵，需尽量补全所有明确列出的平台，不要只抽取其中一部分。

### 产品支持情况矩阵补全
- **补充缺失平台的支持/不支持记录**：当文档"产品支持情况"表格中列出的平台在 `platform_specifics` 中缺失时，应补充对应记录。
- **支持状态记录与 dtype 规则分开**：产品支持/不支持的判断记录应与 dtype 支持范围规则分开存放。
- **保留文档原始平台写法与 √/× 语义**：`description` 中应明确写出"是否支持：√/×"，`constraint_detail` 用可解析表达式表达。

### 规则拆分与结构规范
- 同一平台下的多条独立限制应拆成多条记录；多个平台共享同一条规则时才允许合并。
- 隐式约束：如约束只是弱提示或无法稳定判定，`description` 中可以注明"（推断）"，但 `constraint_detail` 需谨慎填写，避免过度推断。
- 平台支持记录与 dtype 记录分离：全局 dtype 约束尽量不放入 `platform_specifics`，仅在该字段下保留明确的差异化平台规则。
- 一条记录承载单一语义：不同参数或不同接口阶段的约束应拆分为独立记录。
- **一条记录只表达一种独立规则**：若同一元素包含多条独立规则，应拆分为多条记录。
- **不同接口的规则分开建模**：应在 `description` 或 `constraint_detail` 中明确接口范围，或拆分到不同记录。
