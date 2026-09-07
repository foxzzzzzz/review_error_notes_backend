# 本地OCR扩围结论与MiniMax原答隔离实验

2026-09-07。延续已确认优化方向。用户已确认本轮10文件提交，commit message为`test: benchmark OCR models and isolate MiniMax original answers`，随本记录提交；不push、不创建新tag。锁定的文本求解器、生产OCR配置、MiniMax生产流程和既有阶段tag不变。完成294次本地OCR观测，准备10题×3轮的MiniMax原答隔离实验；后者尚未真实调用。

## 先固定像素，再比较本地识别器

保持上一轮完整田字格输入，关闭检测和方向分类，使用同一RapidOCR 3.9.1。比较PP-OCRv5 mobile、PP-OCRv5 server和PP-OCRv6 small。候选来自当前已安装库的模型配置，并核对了[RapidOCR官方模型列表](https://rapidai.github.io/RapidOCRDocs/main/model_list/)。v5 server模型从RapidAI公开模型仓库下载并验证SHA256；v6 small已在本地。下载模型未上传用户图片，所有OCR推理在本机执行。

模型路径、版本、重复次数、阈值及含义均在`ocr_model_comparison_config.json`，执行前检查全部模型和图块哈希。检测及方向模型也指定本地文件，不在推理时自动下载。原答和题面参考只用于输出后的评分，不进入识别器，不按真值选模型或修字。原模型66条结果与上一轮完整格对照逐条一致。

### 开发集：历史6题，22块×3模型×3轮

| 模型 | 有字原答/7格 | 含调题面/11块 | 空白参考返回空串/4格 | 题面＋原答整题/6题 | 每块平均OCR |
|---|---:|---:|---:|---:|---:|
| v5 mobile | 6/7 | 8/11 | 4/4 | 3/6 | 14.77ms |
| v5 server | 7/7 | 9/11 | 1/4 | 3/6 | 473.57ms |
| v6 small | 7/7 | 9/11 | 0/4 | 3/6 | 13.18ms |

三个模型各66条观测，无执行失败，各格三轮输出稳定。v6 small确实读对了做，且没有v5 server的较高推理耗时，但其空白会读成C、—X、—、1；qiū读成qiu（0.92373）、yǐn读成yin（0.95323）。高置信度仍不能证明声调正确。三个模型在本组的整题正确数均为3/6，不能把有字原答7/7当作内容方案收敛。

初始化和预热单独记录；模型按固定顺序运行，磁盘/运行时缓存可能影响初始化，不能据此比较真正冷启动。表中耗时为预加载图块的OCR推理，不含自动裁格和整页保存，未宣称整页提速或30秒通过。

### 验证集：另外4题，16块×2模型×3轮

在看到这4题的OCR输出前，冻结完整格、参考和模型配置；按开发结果选v6 small作为候选，保留v5 mobile对照，未扩跑已在开发集显示较慢且空白失败的server模型。

题目为非常、经常、性别、蝌蚪，来自上一轮文本扩围的4张已审核历史裁图。它们未用于本轮模型选择，但曾用于历史视觉实验，不能称为独立全链路新图集。共5个有字原答、3个空白格、8个题面提示块。

| 模型 | 有字原答/5格 | 含调题面/8块 | 空白参考返回空串/3格 | 题面＋原答整题/4题 | 每块平均OCR |
|---|---:|---:|---:|---:|---:|
| v5 mobile | 4/5 | 6/8 | 1/3 | 2/4 | 13.20ms |
| v6 small | **2/5** | 7/8 | 0/3 | **0/4** | 12.58ms |

v6 small的关键失败为飞→76、今→全、姓→女生；空白格可读成G且置信度0.94962。原模型也会把姓读成生，并在部分空白输出C/1。已重新查看实际输入图块，三种有字失败均保留完整格与原笔画，不能再归因于上一轮东字的裁断问题。

**决策：不切换生产OCR模型，不采用“v6读取有字原答＋MiniMax补题面/空白”的混合方案。** 开发7/7不能泛化到这4题，继续混合会把错误原答交给已锁定的文本比较器。该结论否定当前候选与组合，不能证明所有专用OCR或后续训练路线都无效。

## 下一次有限实验：MiniMax只看原始作答

原先视觉任务同时看到题面拼音、词语语境、订正，容易混合“看见的原字”与“应填的字”；上下文是否造成部分误读仍是待验证假设。本次不声称已证明其因果作用，而是先测撤去这些线索后的忠实转写上限。

- 10题、19格：开发6题11格＋验证4题8格。其中12格有字、7格空白；每题3轮，30次MiniMax视觉请求。
- 每个请求只含该题原答格的无损PNG拼图。19块逐一与本地冻结原图块核对像素一致；仅添加白边和编号，不缩放、滤色或覆盖笔迹。
- 不含题面拼音、标准答案、错因或OCR预测。原始格内红色批改痕迹保留；不是把全部红色抹除。题面/原答角色已人工标定，仍是能力上限实验，不计自动定位成功。
- 模型只返回question_id、逐格state/text及不确定位置；复用原AnswerSlot的状态约束和严格ID校验。保留未知，不把无法辨认强判空白，不强行将两个部件拆成两格或把错字改成正确词。
- 使用现有MINIMAX_*主机配置，预检必须为api.minimaxi.com。无DeepSeek参数、不配置thinking、不更换视觉服务。每题每轮一次请求，关闭重试，所有失败/raw都保留。
- 准备器不将参考字、是否空白放入请求；runner不读取reference-review.json。该文件仅供回传后评分。

本轮新增三项原答实验检查覆盖输入像素与参考隔离、缺失/重复slot拒绝、完整runner保留失败/raw且不读取参考；加上模型对照测试，共98项相关测试通过。服务器实际模型能力与运行耗时仍待用户测试。

### 验收及停止条件

| 指标 | 条件 |
|---|---|
| 数据完整 | 30次结果、30份raw、每题3轮、无失败覆盖；实际HTTP次数与计划一致 |
| 原答能力门槛 | 每轮10/10题的所有目标格状态及原始文字均正确；有字/空白分别核对，未确定不计通过 |
| 已知回归 | 行不变席，做不变作/体，飞不变76，今不变全，姓不拆成女生，空白不补字符 |
| 通过后 | 仅锁定“人工完整格下的原答读取候选”；再解决题面/声调与自动题框、页内调度，不能直接宣布完整95%或30秒 |
| 不通过 | 记录具体失败，停止当前通用OCR替换和本原答Prompt的原样扩跑；没有新输入/训练证据前不继续同类提示词循环。评估专用识别数据或人工确认范围时需重新讨论，保持原95%目标不变 |

这是新的隔离任务，不是与旧B/C严格配对的因果实验；即使准确率提高，也不能全部归因于撤掉题面。19个人工格子及10道历史题也不能估计真实误报分母。若最终95%目标与可读字形范围冲突，需据失败实例讨论，不能通过排除难题后改分母达标。

## 服务器交接：30次MiniMax，与36次文本扩围分开

上一轮`standard-answer-expansion-20260907.zip`的36次独立文本任务继续保持原计划；如已完成，直接回传原结果，不重跑。本轮新包为项目根`output/blind-original-v1-20260907.zip`；SHA256为`206dc99ce1345034967808272ffcccc3a40de66132a301843ac49708144222f3`，共16文件、CRC及输入/源码哈希检查通过。原答实验依赖本轮新脚本，**用户push本次提交后才能在服务器同步运行**；旧aff0b0f尚不含该runner。包内RUN-SERVER.md为提交前快照，当前仓库文档为最新交接记录；冻结ZIP保持不变，不重新prepare。

将ZIP上传服务器后端仓库convergence-data/，核对ZIP哈希后执行：

```bash
cd ~/review_error_notes/review_error_notes_backend
git pull --ff-only origin main
git log -1 --oneline
mkdir -p convergence-data
printf '%s  %s\n' '206dc99ce1345034967808272ffcccc3a40de66132a301843ac49708144222f3' 'convergence-data/blind-original-v1-20260907.zip' | sha256sum -c -
unzip -n convergence-data/blind-original-v1-20260907.zip -d convergence-data
sha256sum -c convergence-data/blind-original-v1-20260907/source-files.sha256
(cd convergence-data/blind-original-v1-20260907 && sha256sum -c input-files.sha256)
```

必须全部OK，不重新prepare或手改冻结输入。零网络身份与输入检查：

```bash
docker compose run --rm --no-deps \
  -v "$PWD/convergence-data:/data/convergence" \
  --entrypoint python worker scripts/blind_original_experiment.py \
  --mode preflight \
  --source /data/convergence/blind-original-v1-20260907
```

预检显示MINIMAX_*及api.minimaxi.com、计划30次后，执行一次；不需要重启业务或更改模型配置，结果目录必须不存在：

```bash
docker compose run --rm --no-deps \
  -v "$PWD/convergence-data:/data/convergence" \
  --entrypoint python worker scripts/blind_original_experiment.py \
  --mode run \
  --source /data/convergence/blind-original-v1-20260907 \
  --output /data/convergence/blind-original-v1-results-20260907
```

完整性检查后连同失败全部回传，字段complete不是内容正确：

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path('convergence-data/blind-original-v1-results-20260907')
r = json.loads((p / 'results.json').read_text())
assert r['complete'] and len(r['results']) == r['http_attempts'] == 30
assert len(list(p.glob('*-raw.json'))) == 30
assert len({(x['request_id'], x['round']) for x in r['results']}) == 30
print('字段交付:', r['delivery_counts'])
print('最大单次秒:', max(x['elapsed_ms'] for x in r['results']) / 1000)
PY
zip -r convergence-data/blind-original-v1-return-20260907.zip \
  convergence-data/blind-original-v1-20260907 \
  convergence-data/blind-original-v1-results-20260907
```

若完整性断言失败，也保留现有目录和控制台日志回传，不删失败或覆盖目录。30个raw全部成功保存且其中有模型失败时，仍应直接打包供审核。

## 本地产物与实现范围

- 项目根`output/ocr-model-comparison-20260907/review.html`：294条本地观测对应的逐格对照；同目录保留两组结果、验证坐标/配置及与运行哈希一致的源码快照。
- 项目根`output/blind-original-v1-20260907/review.html`：10题实际请求图与只供审核的原答参考。
- [可同步的本地结果摘要](benchmarks/ocr-model-comparison-20260907.json)。完整raw保留在上述本地目录。
- 新增本地OCR对照runner/config/test，以及MiniMax隔离runner/config/prompt/test；生产app/与文本基线源文件均未修改，不创建新tag。

本次提交明确包含10个文件：scripts/benchmark_ocr_models.py、scripts/ocr_model_comparison_config.json、scripts/blind_original_experiment.py、scripts/blind_original_config.json、scripts/blind_original_prompt.md、tests/unit/test_ocr_model_comparison.py、tests/unit/test_blind_original_experiment.py、docs/ocr-model-comparison-and-blind-original.md、docs/benchmarks/ocr-model-comparison-20260907.json、docs/text-baseline-lock-and-expansion.md。逐文件添加，不纳入本地模型、输出包或既有未跟踪实验。提交前98项相关测试通过，已锁定的9个文本源文件哈希均未变化。
