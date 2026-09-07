# 8题上下文 A/B 实验：实施记录与服务器步骤

后续更新：本实验48次结果已审计，下一轮改用[物理分区与文本解题交接](partition-text-oracle-server-runbook.md)。原8题包保留用于追溯，不再原样重复；用户已确认新一轮16文件提交，git push由用户执行。

2026-09-07。目的：判断补齐题型、原始作答位置和订正角色后，MiniMax 能否稳定转写，以及同时解题是否损害转写。当前只改诊断脚本，生产 `app/`、CV 和数据库均未改。本地不调用真实 MiniMax。

## 已完成与待完成

| 工作 | 状态与证据 |
|---|---|
| 原84条响应离线回放 | 62条严格有效、18条仅补引号转义的待审候选、4条不修复；原始结果未改 |
| 8题输入与角色修正 | 已逐图核对；7题沿用原区域，运题扩为两行句子；格外订正和邻题单独排除 |
| 参考值 | 6题助手视觉审核，满足的第二声调、神奇的第二字形仍待用户核实；不能称8题金标准全部完成 |
| 配对输入冻结 | 8图、16条请求定义、3轮共48次；同题A/B图片与角色说明完全一致，参考答案不进入请求 |
| 本地验证 | 47项相关测试通过，包括HTTP模拟、B组禁止解题字段、空白语义、题/格ID校验、输入配对和格式回放；没有真实网络调用 |
| 服务器 | 待同步本次代码并执行；不能仅pull原9ed7286就运行新包 |
| 最终95%/10%/30秒 | 尚未验收；该实验含人工题型/角色位置辅助，是能力诊断，不代表自动生产效果 |

本地审核入口：`D:/cc_project/review_error_notes/output/content-context-ab-v1-20260907/review.html`。左侧是实际请求图，右侧是位置审核图；审核图不发给模型。完整参考和待核实项在包内 `reference-review.json`。

旧84条离线记录：`output/content-response-replay-20260907/report.json`（相对项目根）。18条候选已逐条检查插入位置，删除插入的反斜杠后与解包原JSON逐字符一致；其中仍存在明显识字和解析错误。4条保留失败：round2-page33-4、round2-page7-2、round3-page7-2、round3-page20-10。不补字段、不重排结构、不替换答案、不接入生产容错。

## 实验与验收规则

| 项目 | A：完整解题 | B：只转写 |
|---|---|---|
| 共同输入 | 同图、同题型、同作答格/排除位置 | 与A相同 |
| 共同输出 | 题面、逐格原始作答与不确定项 | 与A相同 |
| 额外输出 | 正确答案、基于原答的错因解析 | 禁止这两个字段 |
| 请求数 | 8题×3轮=24 | 8题×3轮=24 |
| 顺序 | 第一/三轮每题A→B，第二轮每题B→A | 与A逐题配对 |
| 协议通过目标 | 每轮8/8，不补格式后计成功 | 每轮8/8，不补格式后计成功 |
| 内容通过目标 | 每轮8/8题面、原答、答案、解析均正确 | 每轮8/8题面、逐格原答均正确 |

`answer_slots` 为逐格输出，`state` 只能是 written、blank、uncertain。blank 必须为 `text=""`；不能把看不清当作未作答，不能把格外订正填回。written 保留错误字/声调；非规范字形不能忠实编码时用 uncertain 并说明。

8题太少，一题失败即只有87.5%，不能证明总体95%。未核实的两题保留在每轮8题分母内，标为pending，不静默删除或计通过；可以单列已审核6题的局部结果，但不能用6题通过宣称全组成功。B没有解题任务，其正确答案/解析栏记“不适用”，不能套用旧联合验收脚本计算95%。比较应逐题逐轮配对，空格排版差异可规范化，字形、拼音声调、留白和原答角色不能规范化掉。

收到结果后逐题填写 `judgments-template.json`：prompt_correct、student_answer_correct，A另填correct_answer_correct、explanation_correct；parsed仅表示格式/ID有效。缺字段、异常、超时、unknown/重复格全部计失败并保留。两处真值补充后另存审核记录，不覆盖已冻结请求。

| 结果 | 下一步决策 |
|---|---|
| A/B都稳定，A答案与解析正确 | 扩到28题回归；再测少量并发/组批与完整30秒链路，不直接生产上线 |
| B稳定、A转写或解析明显更差 | 保留视觉转写输出，准备文本解题实验；新增阶段仍需整页时间预算 |
| A/B都在原答、声调、订正角色上失败 | 此输入辅助条件下转写仍是瓶颈；针对失败项验证本地OCR证据融合，不继续盲加重试 |
| 主要是协议失败 | 对新原始响应单独审计，再决定是否值得增加有界格式处理；本次旧格式修复器不会自动用于新Schema |

“稳定”要求三轮各自报告；不能挑最好一轮。延迟按A/B分别报告中位、最大值及每次失败耗时。本次单题串行请求不含CV/OCR/保存，不能替代整页30秒验收。

## 服务器执行

### 1. 同步代码和数据

用户已于2026-09-07确认将下述11文件提交并推送main，不创建tag。推送完成后，在服务器后端仓库执行：

```bash
git pull --ff-only origin main
```

新增运行依赖 `scripts/content_context_experiment.py` 必须与修改后的 `scripts/benchmark_content_oracle.py` 同步；其余现有 `app/services/vision_recognition.py`、`scripts/prepare_convergence_dataset.py` 随仓库保留。运行时直接读取冻结prepared中的prompt，不在服务器重新prepare或调参。

将本地 `D:/cc_project/review_error_notes/output/content-context-ab-v1-20260907.zip` 上传至服务器后端的 `convergence-data/`：

```bash
mkdir -p convergence-data
unzip -n convergence-data/content-context-ab-v1-20260907.zip -d convergence-data
```

应得到 `convergence-data/content-context-ab-v1-20260907/prepared.json`。包内 `source-files.sha256` 用于核对脚本版本，`input-files.sha256` 用于核对冻结输入；输入有差异时停止，使用新目录重新解包。

```bash
sha256sum -c convergence-data/content-context-ab-v1-20260907/source-files.sha256
(cd convergence-data/content-context-ab-v1-20260907 && sha256sum -c input-files.sha256)
```

### 2. 一条命令运行48次

沿用上次已成功的worker环境，无需重启业务。仓库compose将scripts挂载到/app/scripts，临时容器以Python替换Celery入口；不消费业务队列、不写业务数据库。

```bash
docker compose run --rm --no-deps \
  -v "$PWD/convergence-data:/data/convergence" \
  --entrypoint python worker scripts/benchmark_content_oracle.py \
  --mode run \
  --prepared /data/convergence/content-context-ab-v1-20260907 \
  --output /data/convergence/content-context-ab-results-20260907
```

不加 `--allow-draft`。`context-reviewed` 表示本次输入角色已由助手视觉核对，**不表示所有参考答案已经用户认证**。每次最多一次HTTP尝试，单次配置超时20秒，无自动重试。输出目录必须不存在；中断重跑使用新目录并保留中断结果，不拼接最佳结果。

### 3. 完整性检查和回传

```bash
python3 - <<'PY'
import json
from collections import Counter
from pathlib import Path
p = Path('convergence-data/content-context-ab-results-20260907')
r = json.loads((p / 'results.json').read_text())
assert r['complete'] is True
assert len(r['results']) == r['http_attempts'] == 48
assert Counter((x['round'], x['arm']) for x in r['results']) == {
    (n, arm): 8 for n in (1, 2, 3) for arm in ('A', 'B')}
assert len(list(p.glob('*-raw.json'))) == 48
print(Counter((x['arm'], x['status']) for x in r['results']))
PY
zip -r convergence-data/content-context-ab-results-20260907.zip \
  convergence-data/content-context-ab-results-20260907 \
  convergence-data/content-context-ab-v1-20260907
```

回传这个zip，保留所有失败与raw响应。检查不通过也请回传已有目录和终端错误，不需要重新批量调用来补齐成功数。无需传.env或密钥。

## 本地复现与提交范围

离线准备：`python scripts/content_context_experiment.py --dataset ../output/convergence-20260905 --output ../output/新的输入目录`。参数含义在 `scripts/content_context_config.json`，题型/区域/参考在 `scripts/content_context_cases.json`，通用Prompt在 `scripts/content_context_prompt.md`。

离线回放：`python scripts/replay_content_responses.py --results ../output/content-server-review-20260907/content-server-results-20260907 --output ../output/新的回放目录`。

已确认提交范围：benchmark_content_oracle.py，新增content_context_experiment.py、content_context_config.json、content_context_cases.json、content_context_prompt.md、replay_content_responses.py；两个新增测试；本交接文档、原交接文档入口更新和原计划执行记录。共11个文件，明确逐文件git add。提交消息：`test: add paired content context experiment and offline response audit`，不创建tag。

根目录docs下SPEC、CHANGELOG、TodoList和分析记录是本地协作记录，不属于后端git仓库。output数据包和原有未跟踪实验/models均不纳入提交。
