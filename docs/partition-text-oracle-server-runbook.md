# 物理分区与文本解题实验交接

2026-09-07。本轮在main工作区准备，用户已确认本次16文件提交，不创建tag；git push由用户执行。修改仅涉及诊断脚本、配置、测试和文档，`app/`、生产CV、数据库均未修改。本地没有调用MiniMax或文本服务。

## 本地结果

- 4题B/C图已逐图审核：谈吐、主席、蚯蚓、运。B的JPEG和Prompt逐字节保留旧冻结版本；C把原题面、对应每格的题面提示、原答格放成独立图块，排除格外订正。运保留两行背景及独立目标格。
- C保留原色、不去红、不二值化。原答显示放大2倍，原像素裁图另存PNG及坐标。图块标签不覆盖笔迹；提示与格子对应关系是人工诊断辅助，不能算作自动CV成果。
- 现有RapidOCR检测＋识别接口检查19块，7个原答格均未输出文字。进一步关闭文字检测、直接识别已知小格后，主/行/秋均正确，4个空白格返回空串；7格输出与助手参考一致。单格约11–14ms，不含模型初始化、版面定位、CV、服务调用或保存。
- 控制识别分数阈值为0后，完整检测路径对主只框到一个局部笔划并读成低置信度0；秋只框到碎片并读成1/7；行未检出。这支持继续验证“已定位作答格直接送识别器”，而非把低置信度文字都放行。检测框证据保存在 `ocr/detection-score-zero-control.json`。
- 直接识别的qiū仍丢声调，空串也不能普遍证明未作答；以上只是人工已知格子的局部信号。OCR旁证不进入B/C的任何模型请求，也不修改参考答案。
- 61项相关测试通过：旧实验兼容、B/C顺序和格ID校验、文本单次请求、失败保留、UTC/底层超时与请求标识记录、秘密信息不进入异常日志。

## 三个独立实验

| 实验 | 输入 | 调用服务 | 次数 | 作用 |
|---|---|---|---:|---|
| 服务烟测 | 旧冬瓜A/B冻结输入，3轮 | 现有MiniMax | 6 | 检查本轮服务延迟/传输错误，不作质量验收 |
| 原B/物理分区C | 4题，每组3轮 | 现有MiniMax | 24 | 同一转写Schema，验证图像角色分开能否减少补字/订正混入 |
| 文本解题上限 | 6题人工核对的题面与原始错答，各3轮 | 服务器既有LLM配置 | 18 | 排除视觉误读，验证正确答案与解析 |

不是把48次混成一个成功率。先做烟测；视觉服务异常时先回传。文本上限独立运行，不依赖C已通过。

文本沿用服务器 `LLM_API_BASE`、`LLM_MODEL`、`LLM_API_KEY`。未更换MiniMax视觉模型，文本模型名与服务主机名会记录在结果中。诊断单次HTTP不复用生产文本函数的三次重试；输出上限2048 tokens、温度0.3、超时20秒均在外部配置中说明。只有严格JSON最终答案计解析成功，不用reasoning_content冒充答案。

文本六题：谈吐、冬瓜、合作、主席、蚯蚓、运；排除满足/神奇两个仍有参考歧义的样本。输入只包含原题面、原答及各格对应题面，正确答案和参考解析不会发送。运题明确目标格对应yùn，避免纯文本输入失去目标绑定。参考为助手视觉审核，不是用户认证的生产金标准。

## 文件与同步

本地数据包：`D:/cc_project/review_error_notes/output/content-next-stage-20260907.zip`。
审核页：`output/content-next-stage-20260907/partition-bc/review.html`（相对项目根）。

不能只用旧045a5e0运行。用户推送本次 `test: add partition and text oracle diagnostics` 提交后，在服务器后端仓库执行 `git pull --ff-only origin main`，用 `git log -1 --oneline` 核对提交。数据包独立上传至后端的 `convergence-data/`。

```bash
mkdir -p convergence-data
unzip -n convergence-data/content-next-stage-20260907.zip -d convergence-data
sha256sum -c convergence-data/content-next-stage-20260907/source-files.sha256
(cd convergence-data/content-next-stage-20260907 && sha256sum -c input-files.sha256)
```

源码和输入必须全部OK。输入目录含partition-bc、partition-bc/smoke、text-oracle和本地ocr旁证。无需重新prepare；不要手改冻结参数或Prompt。

## 1. MiniMax烟测：6次

沿用上次worker环境，以临时Python入口运行；不消费业务队列，无需重启业务。

```bash
docker compose run --rm --no-deps \
  -v "$PWD/convergence-data:/data/convergence" \
  --entrypoint python worker scripts/benchmark_content_oracle.py \
  --mode run \
  --prepared /data/convergence/content-next-stage-20260907/partition-bc/smoke \
  --output /data/convergence/partition-smoke-results-20260907
```

检查6次均收到正常服务响应；下列检查不把内部答案JSON格式正确作为网络烟测条件：

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path('convergence-data/partition-smoke-results-20260907')
r = json.loads((p / 'results.json').read_text())
assert r['complete'] and len(r['results']) == r['http_attempts'] == 6
for row in r['results']:
    events = json.loads((p / f"round{row['round']}-{row['request_id']}-raw.json").read_text())
    responses = [e for e in events if e['kind'] == 'http_response']
    assert len(responses) == 1 and responses[0]['status_code'] == 200
    body = json.loads(responses[0]['response_body'])
    assert (body.get('base_resp') or {}).get('status_code', 0) == 0
print('6/6服务正常响应，最大请求耗时秒：', max(x['elapsed_ms'] for x in r['results']) / 1000)
PY
```

若出现超时、429/5xx或上游拒绝，先回传烟测与当时其他MiniMax并发/代理网络情况，不继续跑24次。烟测无异常只说明可开始小实验，不保证生产稳定或30秒达标。

## 2. B/C视觉实验：24次

```bash
docker compose run --rm --no-deps \
  -v "$PWD/convergence-data:/data/convergence" \
  --entrypoint python worker scripts/benchmark_content_oracle.py \
  --mode run \
  --prepared /data/convergence/content-next-stage-20260907/partition-bc \
  --output /data/convergence/partition-bc-results-20260907
```

第一/三轮每题B→C，第二轮C→B。每组每轮4题，输出均为transcription_v2。C只改变图块呈现、位置及对应关系说明，不加入OCR文字或正确答案；因此评估的是“物理呈现＋显式角色辅助”这一干预，不能拆称单一坐标阈值收益。

## 3. 文本解题：18次

```bash
docker compose run --rm --no-deps \
  -v "$PWD/convergence-data:/data/convergence" \
  --entrypoint python worker scripts/benchmark_text_oracle.py \
  --mode run \
  --source /data/convergence/content-next-stage-20260907/text-oracle \
  --output /data/convergence/text-oracle-results-20260907
```

只需已有worker文本LLM配置；若配置缺失会在调用前明确失败，不会选择新模型或调用其他账号。单次失败不重试，保存实际模型、usage、finish_reason、原始响应和人工评审表。

## 验收与回传

正常完成时，三个结果目录的 `complete=true`，实际HTTP尝试分别为6/24/18；raw文件数量对应一致。新记录含每次UTC起止、HTTP耗时、返回请求标识（如有）、底层Transport/Timeout异常类型，结果runtime含实际代码LF哈希。连接超时与读取超时仍不等价于上游内部排队时长。

输出目录必须不存在；重复运行用新目录，不覆盖失败样本。所有失败保留；不要为了补成功而重跑。

```bash
zip -r convergence-data/content-next-stage-results-20260907.zip \
  convergence-data/content-next-stage-20260907 \
  convergence-data/partition-smoke-results-20260907 \
  convergence-data/partition-bc-results-20260907 \
  convergence-data/text-oracle-results-20260907
```

若提前停止，只打包已生成的目录及终端错误。回传zip后由助手逐题逐轮核对。

- C推进门槛：每轮4/4题面与原答正确，原始空白不补字、错字不补正、订正不混入。通过后才扩到28题；失败先判断局部OCR识别器旁证是否可帮助，不无限改Prompt。
- 文本推进门槛：每轮6/6正确答案与错因正确，不能编造声调、笔画或偏旁。即使通过，也不能修复上游错误转写。
- 所有小实验均非最终95%/误报<10%/每页30秒验收；未测自动格子定位、完整题框和业务保存。不以OCR高置信度或空串替代真实内容审核。

## 本地复现与已确认提交范围

在backend-main-benchmark目录离线prepare：

```powershell
& '../backend/.venv312/Scripts/python.exe' -B scripts/content_partition_experiment.py --source ../output/content-context-ab-v1-20260907 --output ../output/新的分区目录
& '../backend/.venv312/Scripts/python.exe' -B scripts/benchmark_text_oracle.py --source ../output/content-context-ab-v1-20260907 --output ../output/新的文本目录
& '../tmp/backend-py312/Scripts/python.exe' -B scripts/content_partition_experiment.py --mode ocr --source ../output/content-next-stage-20260907/partition-bc --output ../output/新的OCR目录
& '../tmp/backend-py312/Scripts/python.exe' -B ../output/content-next-stage-20260907/ocr/replay_recognizer_probe.py --backend . --output ../output/新的识别器旁证.json
```

已确认提交16文件：

- 修改scripts/benchmark_content_oracle.py、scripts/content_context_experiment.py。
- 新增scripts/experiment_telemetry.py、scripts/content_partition_experiment.py、scripts/content_partition_config.json、scripts/content_partition_regions.json。
- 新增scripts/benchmark_text_oracle.py、scripts/text_oracle_config.json、scripts/text_oracle_inputs.json、scripts/text_oracle_prompt.md。
- 新增tests/unit/test_experiment_telemetry.py、tests/unit/test_content_partition_experiment.py、tests/unit/test_text_oracle.py。
- 新增本交接文档、docs/superpowers/plans/2026-09-07-partition-and-text-oracle.md，更新docs/content-context-ab-server-runbook.md入口。

已确认commit message：`test: add partition and text oracle diagnostics`，不创建tag，git push由用户执行。逐文件git add；models、原有未跟踪实验、output包、根目录协作docs均不纳入后端提交。
