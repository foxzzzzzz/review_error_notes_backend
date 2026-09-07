# 文本方案阶段性锁定、扩围与完整裁格修正

后续进展：本地OCR模型扩围未通过，未切换生产模型；已准备MiniMax只看原答格的30次隔离实验，见[新一轮结果及交接](ocr-model-comparison-and-blind-original.md)。本页36次文本扩围输入保持不变。

2026-09-07。用户已确认阶段性锁定“独立求标准答案＋本地逐格比较”，并授权本轮9文件提交及阶段标签创建。带说明的tag已在本地创建并验证指向1054885；本轮修改随本记录提交，push由用户执行。

## 已锁定基线

- 对应已提交代码：`1054885d73598d49dffd4e417ba3207892a3439e`。
- 服务器6题×3轮标准答案和全部错格事实说明18/18通过，33/33标准格正确，单次平均1.023秒、最大1.451秒。
- MiniMax仍负责原有图像链路；本次通过的是LLM_*独立文本链路。两者不能混读。
- 求解请求仅含题面和目标格；原答单独保留并逐格比较。冻结求解Prompt、调用配置、标准答案Schema、ID校验和本地比较规则。修改这些内容必须明确记录偏离基线并重新回归。
- 本轮核对基线清单9个源文件均未变化。可移植的结果摘要、源码与回传包哈希见[基线记录](benchmarks/standard-answer-v1-baseline.json)。原始回传仍完整保留在本地分析目录。
- 锁定仅覆盖闭合答案题型及所有错格的事实说明。开放题等价答案、教学推理解析、自动视觉与整页性能未锁定。

## 已创建的阶段Tag

用户确认后创建带说明的阶段标签 `v0.2.0-text-baseline.1`，**指向已验证的1054885**。使用预发布名称标明只是文本阶段，不是生产整体达标版本。已验证tag对象类型和目标commit，未覆盖既有标签；新扩围和裁格修改通过本次新commit记录。

已写入的annotation：

```text
Text standard-answer baseline, 2026-09-07
Commit: 1054885d73598d49dffd4e417ba3207892a3439e
Six development cases, three rounds: 18/18 standard answers and complete factual slot comparisons.
Mean request 1.023s; max 1.451s.
Input ZIP SHA256: f49b07da2e638179eb0a0d7482d345c0c1495546552f908e57ba3f468835c722
Return ZIP SHA256: 087f633b1a839dbc83b5fd32e337894ab89beb5351ef0b6de84f9eec2e77fa17
Scope excludes automatic vision, independent real-image accuracy, false discovery, teaching explanations and page latency.
```

用户已明确授权本轮commit与tag两个范围。助手未push；用户推送main后，可单独执行`git push origin refs/tags/v0.2.0-text-baseline.1`同步本标签。

## 本地已完成：裁格问题定位与修正

冬瓜整题裁图完整，但旧已知格子OCR将第一格范围设为相对B图 `[0.05,0.33,0.45,0.88]`，右侧切掉“东”的笔画；第二格从0.45开始也侵入前格。此前“东→车”至少包含明确的裁格输入问题，不能全部归因于OCR模型能力。

新增可选answer_regions配置，仅诊断准备器使用；历史配置和参考不变。新配置`known_slot_ocr_whole_grid_regions.json`按完整田字格将冬瓜第一格设为`[0.22,0.31,0.57,0.88]`、第二格为`[0.57,0.31,0.918,0.88]`，其他20块保持原像素。坐标来自原图格线审核，未按答案自动选择框。仍是人工已知格子上限，未完成自动格线定位。

同本地RapidOCR 3.9.1、同3个ONNX模型哈希、同直接识别参数，22块×3种输入×3轮，共198次观测；原裁格66条文本输出与历史直接OCR逐条一致，基线复现成功。

| 输入 | 有字原答正确/7格 | 含调题面正确/11块 | 空白参考返回空串/4格 | 平均每块OCR |
|---|---:|---:|---:|---:|
| 原裁格 | 5/7 | 8/11 | 4/4 | 21.21ms |
| 完整田字格 | **6/7** | 8/11 | 4/4 | 20.08ms |
| 完整格＋R通道灰度 | 5/7 | 9/11 | 3/4 | 20.04ms |

三轮各格输出稳定。完整格的东3/3正确；做仍误读体，qiū丢调、dōng误读dóng、zuò误读Zlè仍存在。R通道虽然改善dōng，却将合读成合$、一个空白读成s，因此**不采用统一R通道预处理**。不得按参考逐题挑选不同arm再汇报为一个自动方案。

上述毫秒仅为预加载图块的OCR推理，初始化和图片准备另计；本次未声明提速，也不是整页耗时。空串仍不等于已证明空白。完整格修正保留为诊断基线，不直接接生产或恢复已否定的双路MiniMax组合。

项目根本地产物：`output/slot-crop-audit-20260907/`包含冻结config、可重跑的run_probe.py、198条结果、原/新图块及review.html。仓库准备器产生的22块与已实测whole_grid图块逐字节一致。

复现完整格准备（本地项目根）：

```powershell
& './backend/.venv312/Scripts/python.exe' -B backend-main-benchmark/scripts/benchmark_known_slot_ocr.py --mode prepare --source output/content-context-ab-v1-20260907 --output output/known-slot-whole-grid-new-run --regions backend-main-benchmark/scripts/known_slot_ocr_whole_grid_regions.json
```

## 已准备：12题文本扩围，不修改求解器

- 4道历史图片的新增文本题：非常、经常、性别、蝌蚪；助手逐图审核题面、原答和标准答案，附原裁图与SHA256。它们此前参与过视觉实验，不能称为独立真实新图集。
- 8道人工构造文本对照：天气、学校、明天、朋友、认真、水果、句子中的北/唱。含3道全对、全空、声调错误、单格和多格错误；它们检验解题及比较行为，不代表现实负样本比例。
- 每题3轮，共36次文本请求。4道历史题和8道构造题分别计分；重复轮只用于稳定性。控制目标：每轮12/12标准答案正确、错格全覆盖；3道全对题均correct、无错因，0误报。该小集严格门槛用于排除已知回归，不替代独立实图总体95%统计。
- 输入文件`standard_answer_expansion_cases.json`记录来源、参考及参数说明；prepare_standard_answer_expansion.py复用原prepare，生成只含题面的请求、独立原答、审核专用标准答案和来源清单。不会修改模型配置或求解Prompt。
- 本地相关94项测试通过，新增检查包括完整格覆盖、历史图哈希不符时不生成输出，以及扩围请求不含原答/答案、全对题本地不误判。

冻结包：项目根`output/standard-answer-expansion-20260907.zip`。完整输入审核页在同名目录的review.html。ZIP SHA256为`809a2640a03e8b59be034b3e6bc079debcc1b75cf94b994037329429e762dc1a`；16个文件、CRC和冻结源码检查通过。包内RUN-SERVER.md为提交前记录快照，本仓库文档为最新状态；包及输入保持冻结，无须重新prepare。

## 服务器下一次只运行文本扩围36次

**现有1054885的runner即可执行**；新修改主要用于本地准备、诊断裁格和记录。本轮服务器不需要为此修改业务配置或重启，不运行MiniMax图片请求。先将新ZIP上传至服务器后端仓库的convergence-data/，并比对本地提供的ZIP哈希。

```bash
cd ~/review_error_notes/review_error_notes_backend
mkdir -p convergence-data
unzip -n convergence-data/standard-answer-expansion-20260907.zip -d convergence-data
sha256sum -c convergence-data/standard-answer-expansion-20260907/source-files.sha256
(cd convergence-data/standard-answer-expansion-20260907 && sha256sum -c input-files.sha256)
```

必须全部OK；不一致停止回传，不手改配置。零网络预检：

```bash
docker compose run --rm --no-deps \
  -v "$PWD/convergence-data:/data/convergence" \
  --entrypoint python worker scripts/standard_answer_experiment.py \
  --mode preflight \
  --source /data/convergence/standard-answer-expansion-20260907/prepared
```

预检通过后执行一次；结果目录必须不存在，不补成功覆盖失败：

```bash
docker compose run --rm --no-deps \
  -v "$PWD/convergence-data:/data/convergence" \
  --entrypoint python worker scripts/standard_answer_experiment.py \
  --mode run \
  --source /data/convergence/standard-answer-expansion-20260907/prepared \
  --output /data/convergence/standard-answer-expansion-results-20260907
```

完整性检查与打包（出现failed也保留全部回传，字段complete不代表语义正确）：

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path('convergence-data/standard-answer-expansion-results-20260907')
r = json.loads((p / 'results.json').read_text())
assert r['complete'] and len(r['results']) == r['http_attempts'] == 36
assert len(list(p.glob('*-raw.json'))) == 36
assert len({(x['request_id'], x['round']) for x in r['results']}) == 36
print('字段交付:', r['delivery_counts'])
print('最大单次请求秒:', max(x['elapsed_ms'] for x in r['results']) / 1000)
PY
zip -r convergence-data/standard-answer-expansion-return-20260907.zip \
  convergence-data/standard-answer-expansion-20260907 \
  convergence-data/standard-answer-expansion-results-20260907
```

回收后逐组核验标准答案与完整差异；通过则继续保持文本基线，失败则按具体类型决定是否处理，不能把它转成无限Prompt迭代。MiniMax下一轮须另有完整字形/声调输入证据与冻结候选；当前不会因东单格修复就宣称整个视觉路线已收敛。

## 剩余验收

真正未参与调参的真实新图（含全对题、干扰红标和实际错题）内容真值仍待建立；现有28题仅用户确认了整题裁框完整性。最终联合召回≥95%、误报及重复输出占全部输出<10%、上传至保存扣除排队≤30秒保持不变；人工裁格、待核实和合成对照不能计为自动完整成功。

## 本轮已确认提交范围

已确认commit message：`test: freeze text baseline and audit complete OCR slots`。共9文件：scripts/prepare_standard_answer_expansion.py、scripts/standard_answer_expansion_cases.json、scripts/known_slot_ocr_whole_grid_regions.json、scripts/benchmark_known_slot_ocr.py、tests/unit/test_standard_answer_expansion.py、tests/unit/test_known_slot_ocr.py、docs/text-baseline-lock-and-expansion.md、docs/benchmarks/standard-answer-v1-baseline.json、docs/standard-answer-and-image-gate-round1.md。逐文件添加；既有未跟踪模型与其他实验不纳入，不push。提交前94项测试通过，冻结9个文本源文件与工作区、标签目标1054885逐项哈希一致。
