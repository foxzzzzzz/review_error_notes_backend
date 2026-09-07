# 服务身份核验、文本契约修复与本地OCR对照

2026-09-07。用户已确认本次11文件提交到main，不创建tag，git push由用户执行。本轮修改诊断脚本；未改app/、MiniMax请求或服务器配置，未在本地调用真实模型。

## 先澄清两条实际链路

回传 `content-next-stage-results-20260907.zip` 的记录如下：

| 实验 | 次数 | runtime.endpoint_host | 请求/返回模型 |
|---|---:|---|---|
| 图片识别烟测 | 6 | api.minimaxi.com | 视觉接口未记录模型版本 |
| 图片识别B/C | 24 | api.minimaxi.com | 视觉接口未记录模型版本 |
| 独立文本解题 | 18 | api.deepseek.com | deepseek-v4-flash |

图片识别始终使用MiniMax。文本实验复用了既有 `LLM_API_BASE/LLM_MODEL/LLM_API_KEY`，不是图片识别客户端，也不是本轮切换了识别模型。上轮引用DeepSeek文档仅适用于这18次文本实验；不能据此调整MiniMax。用户未另行指定文本服务前，按回传记录保留现有双链路，不切换任何服务。

## 已实施与验收

1. 文本增加零网络preflight：分别显示MINIMAX_*视觉主机、LLM_*文本主机/模型以及密钥是否配置，不打印密钥、完整URL、认证头。
2. 正式文本配置冻结预期主机/模型；不匹配时，在创建客户端或输出目录之前失败。新非推理配置只匹配api.deepseek.com / deepseek-v4-flash，不修改settings，不将参数传入MiniMax。
3. 文本Prompt与Schema要求一题一个item、题ID只出现一次、多格合成完整答案、可确定错因时必须非空。null/不确定仍保留供审核，不能算完整交付。
4. 记录output_truncated、empty_final_content、question_id_mismatch、incomplete_solution等独立类别；`status=parsed`仅表示解析有效，新增`delivery_status=complete/incomplete/failed`表示字段交付情况。complete仍需人工验证语义。
5. 新增本地已知格子OCR对照：同一PNG、原像素、原色、相同text_score=0和use_cls=false，仅比较use_det=true/false；参考文字只进入离线评分。空串明确标记待验证，不能充当空白判定器。

相关72项单元测试通过（含原有Pydantic弃用提示）；测试覆盖真实HTTP请求体中的显式thinking选项、主机/模型错误的零请求退出、旧runner兼容、严格ID及秘密信息不输出。离线回放上轮18次raw：8截断、3重复ID、1不完整、6字段完整，分母和失败不变。本地未产生新的真实文本结果。

## 本地OCR结果

四题原有7个原答格，新增冬瓜/合作4个原答格，共6题、11原答格＋11题面提示块，各路径3轮，132条观测；三轮同格输出一致。裁框和内容参考由助手视觉核对，不冒充用户认证或自动格子定位。

| 唯一图块指标 | 检测＋识别 | 已知格子直接识别 |
|---|---:|---:|
| 7个有字原答格 | 0/7 | 5/7 |
| 4个人工参考空白格输出空串 | 4/4 | 4/4 |
| 11个题面提示（含声调） | 0/11 | 8/11 |
| 预热后单图块平均耗时 | 370.41ms | 14.30ms |
| 本轮初始化耗时 | 1237.26ms | 333.78ms |

直接识别：主/行/秋/瓜/合正确；东→车、做→体。题面qiū→qiu、dōng→dóng、zuò→Zlè。dōng误读仍有0.93356置信度，说明不能以高分替代声调审核。

初始化测量顺序为检测路径先、直接路径后，可能受缓存影响，不是独立冷启动对照。正式计时包含图块加载和OCR，未含自动定位、CV、网络或业务保存；不能从14ms推导整页30秒达标。4个空串只是与参考文本相符，不证明通用空白能力。

结论：直接识别支持作为旁证，尚未达到接入生产的原答上限门槛；不继续扩大自动格框实现，也不根据已知正确答案纠正OCR。下一步需围绕残留字形/声调错误建立独立验证，避免继续盲调阈值。本轮不重跑MiniMax B/C。

本地审核与原始观测在项目根 `output/known-slot-ocr-v2-20260907/`，文本离线回放与新输入在 `output/text-contract-v2-20260907/`。

## 服务器下一次只需18次文本实验

用户push本次 `test: guard text service identity and compare known-slot OCR` 提交后，在服务器后端仓库目录同步；仅旧d4aac67不包含preflight和修复：

```bash
git pull --ff-only origin main
git log -1 --oneline
mkdir -p convergence-data
```

数据包：项目根 `output/text-contract-v2-20260907.zip`，独立于旧包，SHA256为 `82ae56b511ce685e34bd94b3f09ee6da6b53e95e175524137287cfb05617d8eb`。本包保持冻结，其中RUN-SERVER.md是提交确认前的说明快照；本仓库文档为当前交接说明。将ZIP上传服务器后端仓库 `convergence-data/` 后执行：

```bash
unzip -n convergence-data/text-contract-v2-20260907.zip -d convergence-data
sha256sum -c convergence-data/text-contract-v2-20260907/source-files.sha256
(cd convergence-data/text-contract-v2-20260907 && sha256sum -c input-files.sha256)
```

两项哈希检查必须全部OK；无需重新prepare或重启业务。现有worker通过只读挂载加载scripts，以下命令启动临时Python进程，不消费业务队列。

先核验身份，不会请求模型：

```bash
docker compose run --rm --no-deps \
  -v "$PWD/convergence-data:/data/convergence" \
  --entrypoint python worker scripts/benchmark_text_oracle.py \
  --mode preflight \
  --source /data/convergence/text-contract-v2-20260907/prepared
```

应看到vision使用MINIMAX_*，text使用LLM_*；本包文本预期为api.deepseek.com / deepseek-v4-flash，thinking=disabled。若服务器配置确实不同，脚本会停止：把核验输出/错误回传，不改LLM_MODEL、不把thinking参数塞给MiniMax，也不原地修改冻结包。若用户要求文本也使用MiniMax，需另按实际文本接口准备实验，不能沿用本包证明其能力。

身份正确后：

```bash
docker compose run --rm --no-deps \
  -v "$PWD/convergence-data:/data/convergence" \
  --entrypoint python worker scripts/benchmark_text_oracle.py \
  --mode run \
  --source /data/convergence/text-contract-v2-20260907/prepared \
  --output /data/convergence/text-contract-v2-results-20260907
```

输出目录必须不存在。配置沿用单次20秒、2048 token、3轮，禁止补跑成功覆盖失败。新文本输入同时包含契约修正，不能把总改善全归因于thinking单变量；分别观察截断、最终内容长度、usage和耗时。

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path('convergence-data/text-contract-v2-results-20260907')
r = json.loads((p / 'results.json').read_text())
assert r['complete'] and len(r['results']) == r['http_attempts'] == 18
assert len(list(p.glob('*-raw.json'))) == 18
print('字段交付统计（语义仍需人工审核）:', r['delivery_counts'])
print('最大单次请求秒:', max(x['elapsed_ms'] for x in r['results']) / 1000)
PY
zip -r convergence-data/text-contract-v2-return-20260907.zip \
  convergence-data/text-contract-v2-20260907 \
  convergence-data/text-contract-v2-results-20260907
```

回传完整ZIP，门槛仍是每轮6/6答案与错因正确，0截断、0重复ID、0缺解析。通过后再评估每页批量与排队外完整30秒，不能把单次请求<30秒当整页验收。

## 本次后端修改范围

- scripts/benchmark_text_oracle.py、text_oracle_prompt.md、text_oracle_config.json、text_oracle_non_thinking_config.json。
- scripts/benchmark_known_slot_ocr.py、known_slot_ocr_config.json、known_slot_ocr_regions.json。
- tests/unit/test_text_oracle.py、test_known_slot_ocr.py。
- 本文档与partition-text-oracle-server-runbook.md的后续入口。

共11文件，用户已确认提交。提交消息：`test: guard text service identity and compare known-slot OCR`，不创建tag；逐文件添加，git push由用户执行。原有models与其他未跟踪实验不纳入。
