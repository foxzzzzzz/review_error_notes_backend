# 收敛能力实验：服务器交接

当前只新增诊断脚本，未接入生产，也未提交代码。先做单项能力检查，再决定是否值得集成；不要再次直接跑一整套串行识别来猜瓶颈。

## 统一验收表

| 项目 | 分母与判定 | 通过条件 | 当前证据 |
|---|---|---|---|
| 联合正确召回 | 所有人工真实错题；同一题的题框、题干/拼音、原始作答、正确答案、解析全部正确才计一次 | ≥95%；69题至少66题；每轮单独通过 | 只有区域标注，内容尚待审核 |
| 误报 | 非真实错题输出＋重复输出，除以全部输出 | 严格小于10% | 现有CV“未匹配候选”不能直接当最终误报 |
| 内容错误 | 定位正确但内容、答案或解析错误 | 不计联合成功，同时报告无效输出数 | 新增字段审核表 |
| 单页耗时 | 保存完成−上传完成−排队时长 | 每张≤30秒；记录全部页及最大值，超时/失败仍保留 | 本地CV/OCR或内容实验都不能冒充此项 |
| CV等价提速 | 同机同图同配置，交替运行，逐页预热1次、正式3次 | 候选所有字段、3种掩码及v3过滤完全一致；中位速度比≥3；任何页不回退超过10% | 全量基准输出单独报告 |
| 真假叉能力 | 冻结人工实际叉标签，区分cross/not_cross/uncertain | 真叉召回≥98%，输出叉中误报<10%；三轮分别报告，uncertain计漏检 | 85候选＋3处漏检叉视觉草稿，标签待审核 |
| 题框候选上限 | 固定人工真叉输入；全候选与Top-3分别找完整且不侵入邻题的框 | 目标≥98%；低于95%停止模型选框 | 先做全候选几何覆盖审计；不等于干净框验收 |
| 内容能力上限 | 人工确认完整题框、必要共享说明；单题原尺寸，逐题人工判内容/答案/解析 | 目标≥98%；低于95%不能支持总体95%；三轮分别报告 | 单题28题×3轮=84次请求，尚未运行 |

98%是阶段余量目标，不是统计置信保证。69题已经用于分析和迭代，只能作回归集；生产宣布通过还需独立新页，并加入零错题页面。现阶段按有红色批改标记的错题整理，范围待最终确认。

## 本地资料

- `output/convergence-20260905/manifest.json`：26张、69题，原图和裁图哈希。
- `output/convergence-20260905/review.html`：原图、裁图审核入口。
- `output/convergence-20260905/reference-review.json`：完整边界、共享说明、原始作答、正确答案、解析、真实叉位置审核表。
- `output/content-oracle-single-prepared-20260905`：单题上限草稿输入，28题、84次请求。
- `output/content-oracle-prepared-20260905`：同页拼图吞吐草稿，28题、21次请求。它使用更小的单题有效分辨率，不能替代单题上限。
- `output/cross-capability-prepared-20260905`：新增集85个原始候选＋3处漏检叉，共88样本的真假叉审核资料。3处额外真叉来自已知失败诊断，须单列，不能直接报告整页召回。

以上路径相对 `D:/cc_project/review_error_notes`。服务器可以放任意目录，准备包内只用相对路径。

## 同步范围

同步后端脚本 `benchmark_content_oracle.py`、`content_oracle_prompt.md`、`prepare_convergence_dataset.py`、`convergence_config.json`；真假叉准备另需 `prepare_cross_capability.py` 和 `cross_capability_prompt.md`。运行复用已存在的 `app/services/vision_recognition.py` 及服务器 MiniMax 环境，不改接口和密钥。

若采用git同步，先确认提交范围、commit message、是否建tag；当前建议只提交本次诊断、测试、计划文档，不提交模型、数据包及原有未跟踪文件。数据包单独传输。

## 推荐执行顺序

1. 先审核历史6页28题的裁图完整性及共享题干；缺说明或裁错范围先修正数据，不能把输入缺失归罪模型。修改裁图后同时更新清单哈希，并重新prepare。
2. 冻结 `reference-review.json` 内容真值。不可辨认的原始作答不要猜；记录待复核。先做单题原尺寸内容实验，观察能力瓶颈。
3. 单题内容通过后才测同页拼图吞吐。如果单题都失败，先分析转写、答案、解析各自错误，不进入选框集成。
4. 真假叉先对85候选盲审标签，另核对已补入的三处实际漏检叉草稿。不要先看模型答案再标注，不将候选“落在错题框内”当真叉。该实验独立于内容实验。

## 命令

在后端目录、原有可访问MiniMax的Python环境执行。将 `/data/convergence/...` 换为实际目录。无需启动API或Worker，脚本不写业务数据库。

先准备正式单题输入（此命令不联网）：

```bash
python scripts/benchmark_content_oracle.py --mode prepare \
  --dataset /data/convergence/convergence-20260905 \
  --output /data/convergence/content-verified-prepared
```

审核完整框后，记录 `region_complete_verified=true` 再prepare，正式运行：

```bash
python scripts/benchmark_content_oracle.py --mode run \
  --prepared /data/convergence/content-verified-prepared \
  --output /data/convergence/content-server-results
```

如果希望先看草稿输入的失败模式，可显式加 `--allow-draft`；结果只作探索，不得据此验收通过。默认3轮，每次最多1次HTTP尝试，单次超时20秒；这不是完整生产流程30秒验收。

调试时若要先执行1轮，在单独复制的配置JSON中将 `content_rounds` 改为1，使用 `--config` 重新prepare。完整对比须使用冻结配置重新跑3轮，不能拼接不同输入的最佳轮。

每页请求耗时可以求和，但该值缺少上传后调度、CV/OCR、业务保存阶段。HTTP客户端超时约束也不能代替生产单页墙钟截止时间。

## 回传与评分

请回传整个服务器结果目录，以及本次使用的prepared目录与审核后的reference-review文件；必须保留失败请求、raw响应、全部轮次。不要传 `.env` 或密钥。

`results.json` 保留每轮每批预期题目ID、状态、解析结果、实际请求数与耗时。`*-raw.json` 保留响应供错误归因，不保存请求认证头或base64图片。实际请求图保存在prepared目录，使用哈希关联。

`parsed` 只表示协议解析成功，`review_status` 始终待人工评分。每轮按同一审核规则评分，不用模型自报置信度代替人工真值。

## 接入生产之前

先看两条硬结论：人工完整裁图下是否能正确读题并给答案/解析；本地候选集合里是否存在完整题框。任一项明显低于95%，增加后续重试不能弥补能力上限。只有局部能力通过后，才决定最少请求链路，并在SPEC中设计答案/解析的保存与展示契约。

## 真假叉专用命令

审核 `cross-reference-review.json` 中每个样本的 `verdict`（cross/not_cross）与 `reviewed=true`，保留图片哈希。重新准备会核对审核图像哈希，且不会把人工标签发给模型。

```bash
python scripts/prepare_cross_capability.py --dataset /data/convergence/convergence-20260905 \
  --cv-results /data/convergence/cv-v3-blind-944c9f4-20260905-103708 \
  --review /data/convergence/cross-reference-review.json \
  --output /data/convergence/cross-verified-prepared
python scripts/benchmark_content_oracle.py --mode run \
  --prepared /data/convergence/cross-verified-prepared \
  --output /data/convergence/cross-server-results
```

真假叉额外同步 `scripts/convergence_cross_drafts.json`。全量88样本三轮共264次请求；也可先用1轮配置探索。未审核样本默认拒绝正式运行，显式 `--allow-draft` 才允许探索。输出标签与内容结果使用不同Schema，原始响应和每轮分母均保留。

## 已确认的提交范围（2026-09-05）

用户已确认在main提交并推送以下21个文件，commit message：`perf: add equivalent CV scoring and convergence diagnostics`，不创建tag。旧的未跟踪v2/题框实验、models及output不在范围内。根目录docs/SPEC.md、CHANGELOG.md、TodoList.md已在本地更新，但不属于此后端git仓库，不能通过本次提交同步。

- `scripts/diagnose_vision_pipeline.py`
- `scripts/red_cross_scoring.py`
- `scripts/benchmark_cv_equivalence.py`
- `scripts/convergence_config.json`
- `scripts/prepare_convergence_dataset.py`
- `scripts/refresh_convergence_ocr.py`
- `scripts/evaluate_convergence.py`
- `scripts/diagnose_cross_misses.py`
- `scripts/convergence_cross_drafts.json`
- `scripts/audit_convergence_boundaries.py`
- `scripts/prepare_cross_capability.py`
- `scripts/cross_capability_prompt.md`
- `scripts/benchmark_content_oracle.py`
- `scripts/content_oracle_prompt.md`
- `tests/unit/test_red_cross_scoring.py`
- `tests/unit/test_convergence_dataset.py`
- `tests/unit/test_convergence_evaluation.py`
- `tests/unit/test_content_oracle.py`
- `docs/superpowers/plans/2026-09-05-local-convergence-and-server-validation.md`
- `docs/content-oracle-server-runbook.md`
- `docs/local-convergence-results-20260905.md`

## 最短执行步骤：历史6页28题

### 1. 在电脑上核对裁图

打开 `D:/cc_project/review_error_notes/output/convergence-20260905/pilot-review.html`，这里只展示page33、34、35、5、7、20，共28题。每题可以点开整页原图和原尺寸裁图。

逐题检查：题目必需的说明、拼音、选项、学生原始作答有没有裁漏；是否带入容易混淆的邻题；字迹是否可辨。老师的红叉或留白无需机械地全部保留，必需内容必须完整。

直接在会话中反馈即可，例如：`page33__T2 缺上方说明；page20__T5 带入右边邻题；其余完整`。不必手改JSON。若全部完整，可以回复“28题裁图均完整”。这一步只确认裁图，不表示答案和解析真值已经审核。

收到反馈后，由助手修改有问题的裁图、记录共享题干、更新哈希及审核状态，再补齐或核对内容参考。仅明确完整的题设置 `region_complete_verified=true`；不会为了绕过运行检查批量标真。

### 2. 同步代码和数据

服务器在当前后端仓库执行 `git pull --ff-only origin main`。本轮未修改生产app/，无需重启业务Worker。

当前可上传的包：`D:/cc_project/review_error_notes/output/convergence-20260905-draft.zip`，约76.16MiB，SHA256为 `d34335c1697d654de86f1a4f777c0bb2e62359f79d79b6c8c35af4af6878832d`。它包含完整26页回归资料及28题审核入口；默认内容实验只选择历史6页28题。

这是未审核数据快照。审核完成后，使用助手提供的更新包或更新文件，不能继续使用旧草稿生成正式输入。数据不在git里，pull不会下载这些图像。

在服务器后端目录创建 `convergence-data`，将包上传至该目录并解压；最终应能看到 `convergence-data/convergence-20260905/manifest.json` 和 `reference-review.json`。

```bash
mkdir -p convergence-data
unzip convergence-data/convergence-20260905-draft.zip -d convergence-data
```

同名数据已存在时不要重复解压覆盖审核记录。审核有裁图变化时，以更新包的新路径和说明为准。

### 3. 在既有Worker环境中运行

已核对本仓库docker-compose：worker包含MiniMax环境变量，scripts挂载在 `/app/scripts`。以下临时容器继承worker环境，以Python脚本替换Celery入口，不消费队列、不改生产数据库。请在服务器后端仓库目录执行；不会在本地电脑调用MiniMax。

审核状态更新后，先离线prepare：

```bash
docker compose run --rm --no-deps \
  -v "$PWD/convergence-data:/data/convergence" \
  --entrypoint python worker scripts/benchmark_content_oracle.py \
  --mode prepare \
  --dataset /data/convergence/convergence-20260905 \
  --output /data/convergence/content-verified-prepared
```

检查摘要应为 `expected_questions_per_round: 28`、`planned_http_attempts: 84`。如果已在电脑上生成并传输了正式prepared目录，则跳过prepare。

然后执行真实MiniMax实验：

```bash
docker compose run --rm --no-deps \
  -v "$PWD/convergence-data:/data/convergence" \
  --entrypoint python worker scripts/benchmark_content_oracle.py \
  --mode run \
  --prepared /data/convergence/content-verified-prepared \
  --output /data/convergence/content-server-results
```

本次正式实验不加 `--allow-draft`。若提示 `unverified crop`，说明仍有未审核题或用了旧prepare目录；应修正输入并在新目录重新prepare。输出目录必须是新目录；重复实验请使用不同目录名以保留所有轮次。

默认3轮共84次请求，单次20秒HTTP超时，无自动重试。每轮28题，98%门槛在此小样本上意味着28/28正确；27/28只有96.43%。不能把局部条件成功率当最终联合指标。

### 4. 回传

将 `convergence-data/content-server-results` 整个目录打包，并附上实际使用的 `content-verified-prepared` 与审核后的 `reference-review.json`。返回会话时提供本地存放目录，助手会按每题、每轮核对原始转写、正确答案、解析及失败类型，再决定后续拼图吞吐或分阶段方案。不能只回传成功结果或最好的一轮。
