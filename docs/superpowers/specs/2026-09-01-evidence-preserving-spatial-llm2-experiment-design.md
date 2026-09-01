# 证据保留与空间分组 LLM2 诊断实验设计

## 状态

- 日期：2026-09-01
- 状态：已实施（诊断代码待服务器实测）
- 范围：仅诊断脚本、诊断配置、离线回放工具和对应单元测试
- 非范围：生产识别服务、API、数据模型、数据库、部署配置

## 背景与证据

固定六页、三轮双方案基准共有 28 道人工真值。`old_solution` 平均召回为
23.8%，`main_single_pass` 平均召回为 85.7%。main 的 CV/LLM1 风险保留候选
三轮均覆盖 27/28 道真值，但 LLM2 每轮只保留 24/28，且漏题身份漂移。

唯一三轮都未进入 LLM2 的真值是 `page20-T7`。三轮独立红叉扫描均返回与该
真值相交的高置信红叉，但当前配置禁止独立扫描新增锚点，因此这些证据只被记录、
未进入定位。LLM2 同时占 main 核心耗时约 54%，六页平均执行约 30 个定位批次。

当前六页已经参与分析，只能作为开发/回归集，不能证明泛化。所有实验规则必须与
页面名称、truth_id、人工框坐标及图片固定尺寸无关；truth 只能在输出完成后审计。

## 目标

1. 验证独立扫描能否作为通用、可审计的新锚点来源，而不是只辅助已有 CV 锚点。
2. 验证 LLM2 `matched=false` 或定位异常时，保留强证据锚点是否能提高单轮真值召回。
3. 验证空间分组能否减少 LLM2 请求和跨批次重复，同时不降低证据保留后的召回。
4. 每项实验可独立启用、独立产出报告，能够做逐项消融。

## 非目标

- 不修改生产识别流程。
- 不识别题目内容，不执行 OCR。
- 不增加固定第二轮 LLM2。
- 不以三轮结果并集代替单轮完整召回。
- 不依据 truth 自动选择阈值或改写候选、题框和事件。

## 共同约束

1. 所有运行时决策只读取图片、CV 几何、模型响应和外部实验配置。
2. truth 仅生成 `*-truth-comparison.json`，不得进入候选、分组、保底或去重函数。
3. 每个输入锚点必须在 LLM2 响应中恰好归属一个 question group 或一个 unmatched
   结果；缺失、重复和未知 cross_id 均为格式失败。
4. 红叉是决定性锚点；红圈和教师批注只帮助确定题目区域边界，不能创建错题。
5. `confirmed` 与 `needs_review` 分开统计，不能把本地保底伪装成模型确认结果。
6. 所有新增参数写入外部 JSON 配置并附中文含义说明。
7. 当前基线保持可运行，实验通过显式 profile 选择，禁止静默改变既有基准语义。

## 实验 1：独立扫描补锚离线回放

### 输入

读取现有归档中的：

- CV candidates；
- LLM1 candidate verdicts；
- independent cross scan；
- 当前 confirmed crosses；
- truth regions，仅用于最终审计。

### 处理

在不调用 LLM 的情况下重放候选构建：

1. 先按当前规则构建并去重 CV 锚点。
2. 将独立扫描框与已有锚点按配置的 IoU/中心距离进行匹配。
3. 已匹配扫描框只增加 `independent_scan_supported=true`。
4. 未匹配扫描框只有通过通用证据门槛后才新增 `independent_scan_rescue` 锚点：
   模型置信度、本地红像素支持、最小/最大面积和页边界完整性。
5. 新锚点与 CV 锚点再次做本地确定性去重，但不得使用 truth。

### 产物

- `replay-anchor-union.json`
- `replay-anchor-rescue-audit.json`
- `replay-anchor-truth-comparison.json`
- 六页/三轮汇总：新增锚点数、新增假锚点数、候选召回、重复候选变化

### 判定

本实验只验证可行性，不直接选定生产阈值。若恢复已漏真值的同时新增锚点无上界，
则补锚规则不进入实验 2。离线候选召回不等于最终识别召回；通过离线门槛后，才允许
用同一冻结配置将补充锚点送入当前 LLM2，测量实际收益和成本。

## 实验 2：锚点不可静默丢失

### 输入与 LLM2

保持当前 LLM2 分批、图片叠加和 Prompt 主任务不变。继续要求每个 cross_id 恰好返回
一次；本实验只改变 LLM2 之后的证据保留语义。

### 状态模型

- `confirmed`：LLM2 返回 matched=true，题框通过现有几何审计。
- `needs_review`：强证据锚点 matched=false、返回框无效或所在请求批次失败。
- `rejected`：低证据锚点 matched=false；仅保留审计记录，不生成题目事件。

强证据由来源风险、独立扫描支持、本地红像素支持和配置阈值共同决定，不读取 truth。
请求批次失败时，批次内每个输入锚点分别落入 `needs_review` 或 `rejected`，同时保留
批次错误和原始响应；禁止因一个批次异常而静默丢弃全部锚点。

### 本地保底区域

`needs_review` 使用锚点和附近红色证据生成保守上下文框：

1. 以红叉框为核心按配置水平/垂直扩张；
2. 若邻近红色连通区域与扩张框相交，可并入边界；
3. 限制最小尺寸、最大页面积和越界裁剪；
4. 保留 `bbox_source=local_anchor_context` 和生成参数，禁止标记为 LLM 定位。

### 产物

- `anchor-preservation-events.json`
- `anchor-preservation-audit.json`
- `anchor-preservation-truth-comparison.json`
- 分别报告 confirmed recall、confirmed+needs_review recall、两类误报和额外输出

## 实验 3：空间分组 LLM2 与确定性去重

实验 3 继承实验 1 的候选联合和实验 2 的证据保留，只替换 LLM2 组织方式与事件归并。

### 空间分组

1. 按锚点纵向中心形成题目行带，再按水平位置排序。
2. 同一局部邻域、可能属于同一作答单元的锚点必须优先进入同一组。
3. 每组受最大锚点数和最大裁图面积限制；分组只影响请求组织，不合并锚点身份。
4. 为每组生成带上下文的局部裁图，并将锚点坐标映射到裁图归一化坐标；LLM2 的
   `question_bbox` 也定义为裁图归一化坐标，结果通过已记录的裁图原点和尺寸映射回
   整页归一化坐标。

### LLM2 返回契约

LLM2 返回 question groups，而非独立事件列表：

```json
{
  "groups": [
    {
      "group_id": 0,
      "cross_ids": [1, 2],
      "question_bbox": [0.1, 0.2, 0.4, 0.5],
      "confidence": 0.95
    }
  ],
  "unmatched": [
    {"cross_id": 3, "reason": "不是明确红叉", "confidence": 0.8}
  ]
}
```

每个输入 cross_id 必须且只能出现一次。一个 question group 可包含多个红叉，从源头
表达“一题多叉”，避免每个 cross_id 固定生成一个事件。

### 跨组去重

1. 同一批内以模型 group 为事件单位，不再拆成逐锚点事件。
2. 跨批只按配置的高 IoU/高包含率和空间一致性产生合并候选。
3. 合并必须保留全部 cross_ids、来源风险和原始题框；不得仅因低阈值 IoU 合并相邻题。
4. 唯一强证据事件不得被去重删除；冲突时保留为 `needs_review` 并记录原因。

### 产物

- `spatial-anchor-groups.json`
- 每组输入裁图、坐标映射和原始响应
- `group-membership-audit.json`
- `deduplicated-question-events.json`
- `deduplication-audit.json`
- 与实验 2 并列的召回、额外输出、逻辑/实际请求、分阶段耗时报告

## Profile 与消融顺序

诊断入口显式选择以下 profile：

- `baseline`：当前 main_single_pass，不改变已有结果。
- `independent-rescue`：先通过离线回放门槛，再以冻结规则联合候选并送入当前 LLM2；
  不启用锚点保底或空间分组。
- `anchor-preserving`：实验 1 + 实验 2，保持当前 LLM2 分批。
- `spatial-grouped`：实验 1 + 实验 2 + 实验 3。

服务器测试必须按该顺序运行和比较，不能只跑最终 profile 后反推收益。

## 配置边界

新增配置至少覆盖：

- 独立扫描补锚的置信度、红像素、面积和页边界门槛；
- 强证据锚点来源和红色支持门槛；
- 本地保底框水平/垂直扩张、最小尺寸和最大面积；
- 空间行带距离、邻域距离、每组最大锚点数、裁图 padding 和最大面积；
- 跨组去重 IoU、包含率和最大锚点距离。

配置默认值在查看新的 holdout 结果前冻结。不得增加页面专用配置。

## 测试策略

### 单元测试

1. 独立扫描未匹配且证据达标时新增锚点；不达标时只审计。
2. 补锚与现有 CV 锚点正确去重，cross_id 稳定且连续。
3. matched=false 的强锚点生成 needs_review；低证据锚点不生成事件。
4. 本地保底框归一化、面积受限、边界裁剪并保留来源。
5. 空间分组不丢失、不重复 cross_id，且相邻锚点优先同组。
6. LLM2 group membership 对缺失、重复和未知 ID 明确失败。
7. 跨组去重合并明显同题框，不合并相邻兄弟题。
8. baseline profile 的现有测试和产物语义保持不变。

### 离线回归

对现有六页、三轮归档运行实验 1；truth 只参与最后比较。实验 2 和实验 3 的新增
LLM 行为先在诊断脚本中运行一次，达到方向性门槛后再各连续运行两次。

### Holdout

当前六页全部视为开发/回归集。配置和 Prompt 冻结后，再用未参与设计的新页面及逐题
位置真值连续运行三次；holdout 结果不得反向触发页面专用规则。

## 阶段验收建议

- 候选联合：现有回归集每轮 28/28，且新增假锚点有明确上界和来源审计。
- 最终输出：现有回归集单轮 confirmed+needs_review 召回 28/28，不以多轮并集计算。
- 额外输出：较当前三轮平均 34.3 个至少下降 30%。
- 请求：平均不超过 6 个逻辑请求/页，实际 HTTP 重试单独统计。
- 耗时：平均 60～65 秒/页，最差单页不超过 90 秒。
- 固定第二轮 LLM2：0 次。

这些是诊断阶段的建议门槛，不构成生产上线承诺；最终门槛需由冻结配置后的 holdout
结果确认。

## 预计修改范围

- `scripts/diagnose_vision_pipeline.py`
- `scripts/cv_cross_experiment_config.json`
- 新增离线回放脚本（名称在实施计划中确定）
- `tests/unit/test_diagnose_vision_pipeline.py`
- 新增离线回放单元测试

不修改 `app/` 下生产代码。`scripts/benchmark_vision_solution.py` 只在需要暴露显式
profile 参数且能保持 baseline 默认语义时才修改；否则保持不变。
