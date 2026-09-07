# 第一轮能力验证：独立求标准答案与本地逐格差异

2026-09-07。用户已确认本次6文件提交到main，不创建tag，git push由用户执行。本轮不修改生产app/、MiniMax图片识别参数或既有文本服务配置，未调用真实模型。

## 实施内容与边界

| 工作 | 验收 | 本地状态 |
|---|---|---|
| 求解输入只包含题面与目标格提示 | 请求不含学生原答、空白状态、参考标准答案或参考解析 | 已核对6请求、11目标格 |
| 模型按slot_id返回标准答案 | 题ID/slot_id严格匹配；缺失/重复/未知ID失败 | 模拟验证通过，真实能力待服务器18次 |
| 本地逐格生成事实错因 | 所有错格均覆盖；声调保留；不确定不强判 | 历史13条有效标准答案全部比较完整，另外5条仍不可用 |
| 图像侧保守组合 | 内容≥95%，不把人工框或冲突当自动成功 | 历史回放9/12，未通过；停止该组合 |

先得到标准答案，再与原答比较，解决上轮“学生空白导致拒绝求解”和漏讲多个错格的职责混合。模型只返回question_id、standard_slots和uncertain_segments，不要求生成错因；本地按冻结原答状态生成说明。例如：第1格应为“蚯”，原答为“秋”。第2格未作答，应填“蚓”。

本轮只支持已审核的看拼音写词语、看词语写拼音、指定目标格的拼音句子填空。Unicode NFC和首尾空白规范化不删除声调、不改字形；不能将简单字符串比较用于开放题、多种等价表达或未明确规则的答案。原答不确定、标准答案null/空串或模型明确不确定时，整体needs_review，保留已知差异，不算完整交付。原答全部与标准答案一致则返回correct，不强行生成错因。

新脚本与旧benchmark_text_oracle.py并存，旧结果与旧Schema不改。新脚本复用既有服务身份预检和请求配置构造，文本仍绑定回传中核实的api.deepseek.com / deepseek-v4-flash非推理模式；返回模型不一致也保留失败。MiniMax仅是图像链路，本轮没有新增MiniMax请求。

## 输入隔离与可追溯性

- prepared.json只保存求解Prompt、题ID/目标格ID、配置及文件哈希；Prompt白名单为question_id、task_type、answer_scope、instruction、prompt_text、slot_prompt_texts。
- student-inputs.json单独保存真实原答，仅用于本地逐格比较；不含参考标准答案，文件哈希冻结并在发起请求前校验。
- reference-review.json只用于人工审核，runner不读取、不发送。
- 题面来自已审核参考记录，但标准答案及解释未进入求解请求；这是文本输入正确时的上限实验，不能计为自动视觉成果。
- 一次请求、无重试，保留HTTP状态、UTC、usage、模型名、原始响应、标准答案与本地比较结果。原始响应永不覆写。
- 仅允许规范化完整包住一个JSON的单层```json围栏，并记录format_normalization=outer_json_fence；不去其他前后缀、不修补内容、字段或ID。围栏规范化后的结果仍需严格Schema及逐题审核。

## 本地验证结果

91项相关单元测试通过，含原有Pydantic弃用提示。新增19项检查错字＋漏答全覆盖、声调差异、Unicode等价、不确定处理、全对原答、目标格限定、重复/缺失/未知ID、请求不泄漏原答与参考、服务模型变化、超时保留、围栏范围以及冻结原答被修改后零请求停止。

历史文本V2实际完整成功仍为5/18。本轮把其中13条严格有效且标准答案已审核正确的历史响应，按此6题的明确格子结构转为离线输入，用本地比较器重算错因；13条均覆盖全部差异，另外4条没有标准答案、1条原格式失败仍保留。该13条结果仅证明本地比较逻辑，不是新模型13/18成功率。新服务器协议直接返回slot_id，不使用离线历史字符串拆格适配。

对应本地文件：项目根 `output/standard-answer-v1-20260907/local-comparison-replay.json` 和 `offline_checks.py`。

## 图像组合的停止结论

冻结的开发假设：B原上下文提供题面；C物理分区提供学生原答；同像素格子的直接OCR只负责发现冲突，绝不替换模型原答，也不把OCR空串视为已证明空白。

历史4题×3轮回放，所有OCR原答图块哈希与C的原像素裁图一致。路由只使用模型/OCR输出，路由完成后才读人工审核标签计分，不按每题真值挑最好的结果：

- 完整题面＋原答9/12，内容召回75%；主席三轮视觉结果与OCR的行冲突，需人工核实3/12。
- 两次视觉调用的历史平均耗时合计约8.86秒/题，不含OCR、文本或保存。按4题串行粗估，仅视觉约35.5秒，不能支持整页30秒；这不是实测生产链路耗时。
- 全部为人工框、反复使用的开发样本，不能据此认证独立集准确率或误报率。先前C的4/12与该组合的9/12也是不同职责配置，不能直接宣布新生产效果。

**停止该组合，不发起同组合的服务器重测、不接入生产。** 文本实验若通过，只能锁定这一窄范围内的标准答案/差异说明设计；图像读取仍是独立阻塞项，不能宣称整套95%/10%/30秒方案已锁定。后续图像路线必须有新的字形/声调读取证据和独立验收，不继续针对这4题盲调Prompt或按正确答案补笔迹。

本地冻结策略与回放：项目根 `output/standard-answer-v1-20260907/image-combination-policy.json`、`image-combination-replay.json`。

## 服务器交接：仅18次文本调用

新数据包为项目根 `output/standard-answer-v1-20260907.zip`。用户push本次 `test: separate standard answer solving from local slot comparison` 提交后，在服务器同步新提交并执行；仅旧4d60b72不包含新runner。

ZIP的SHA256为 `f49b07da2e638179eb0a0d7482d345c0c1495546552f908e57ba3f468835c722`。保持原包冻结；包内RUN-SERVER.md为提交确认前的快照，当前仓库文档为最新交接说明。

将新ZIP上传服务器后端仓库的convergence-data/，从后端仓库目录运行：

```bash
git pull --ff-only origin main
git log -1 --oneline
mkdir -p convergence-data
unzip -n convergence-data/standard-answer-v1-20260907.zip -d convergence-data
sha256sum -c convergence-data/standard-answer-v1-20260907/source-files.sha256
(cd convergence-data/standard-answer-v1-20260907 && sha256sum -c input-files.sha256)
```

必须全部OK，不重新prepare、不手改冻结输入。无需重启业务；worker现有scripts挂载会加载同步后的脚本，临时Python入口不消费业务队列。服务身份与本地原答文件先做零网络预检：

```bash
docker compose run --rm --no-deps \
  -v "$PWD/convergence-data:/data/convergence" \
  --entrypoint python worker scripts/standard_answer_experiment.py \
  --mode preflight \
  --source /data/convergence/standard-answer-v1-20260907/prepared
```

预检必须匹配当前冻结的文本主机与模型；不一致则停止、回传错误，不切换模型来凑匹配。通过后：

```bash
docker compose run --rm --no-deps \
  -v "$PWD/convergence-data:/data/convergence" \
  --entrypoint python worker scripts/standard_answer_experiment.py \
  --mode run \
  --source /data/convergence/standard-answer-v1-20260907/prepared \
  --output /data/convergence/standard-answer-v1-results-20260907
```

输出目录必须不存在；即使出现failed/incomplete也继续保留本轮18次结果，不补成功覆盖失败。检查并打包：

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path('convergence-data/standard-answer-v1-results-20260907')
r = json.loads((p / 'results.json').read_text())
assert r['complete'] and len(r['results']) == r['http_attempts'] == 18
assert len(list(p.glob('*-raw.json'))) == 18
print('字段交付（标准答案正确性仍需审核）:', r['delivery_counts'])
print('围栏处理:', r['normalization_counts'])
print('最大单次请求秒:', max(x['elapsed_ms'] for x in r['results']) / 1000)
PY
zip -r convergence-data/standard-answer-v1-return-20260907.zip \
  convergence-data/standard-answer-v1-20260907 \
  convergence-data/standard-answer-v1-results-20260907
```

回传ZIP后逐题检查标准答案和原答差异；门槛：每轮6/6逐格标准答案正确、所有错格解释完整，0无依据拒答、0遗漏/重复ID。只返回字段不空不算答案正确。通过后才考虑扩围、批量和完整链路；未通过则执行一次有依据的方案决策，不原样无限循环。

## 本次已确认提交范围

新增scripts/standard_answer_experiment.py、standard_answer_config.json、standard_answer_prompt.md，新增tests/unit/test_standard_answer_experiment.py、新增本文档，更新docs/text-contract-and-known-slot-ocr-v2.md入口，共6文件。

已确认commit message：`test: separate standard answer solving from local slot comparison`，不创建tag；逐文件提交，git push由用户执行。根目录协作记录、output包、模型缓存和既有未跟踪实验不纳入后端提交。
