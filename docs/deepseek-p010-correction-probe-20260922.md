# P010 红笔证据逐项复核探针

## 范围

本探针仅对 P010 的第一次整页识别结果发起一次独立 DeepSeek 修正请求。它不修改主请求 Prompt，不写数据库，不改变 worker 正式路径，也不自动接受模型的二次判断。第一次的 18 个候选逐项编号后完整附在请求中；第二次只判断这些候选是否能在原图中指出明确的红圈或红叉，并返回红笔标记的 `mark_bbox`。缺少任一候选决策、重复 ID、或保留项缺少有效证据坐标时，探针报错并保留原始响应供检查。

## 云服务器运行

以下命令在后端仓库目录执行；先确认 `.env` 中 DeepSeek 凭据仍可供 `docker compose run worker` 使用。不需要 `ACCESS_TOKEN`、影子数据库或 legacy 容器。每次运行会产生一个新目录，不覆盖旧结果。

```bash
export RETURN_ARCHIVE="$PWD/chinese-marked-evidence-server-a394222-20260921T150817Z.tar.gz"
export RETURN_ROOT='chinese-marked-evidence-server-a394222-20260921T150817Z'
mkdir -p "$PWD/p010-correction-input" "$PWD/p010-correction-output"
PROBE_RUN_LABEL="$(date -u +%Y%m%dT%H%M%SZ)-$$"

tar -xOf "$RETURN_ARCHIVE" "$RETURN_ROOT/stage-audit/P010/00-source.jpg" \
  > "$PWD/p010-correction-input/P010.jpg"
tar -xOf "$RETURN_ARCHIVE" "$RETURN_ROOT/stage-audit/P010/primary-raw-response.md" \
  > "$PWD/p010-correction-input/first-response.md"

for run in 1 2 3; do
  sudo docker compose run --rm --no-deps -T \
    -v "$PWD/p010-correction-input:/probe-input:ro" \
    -v "$PWD/p010-correction-output:/probe-output" \
    --entrypoint python worker -X utf8 -B \
    -m scripts.deepseek_page_correction_probe \
    --image /probe-input/P010.jpg \
    --first-response /probe-input/first-response.md \
    --prompt /app/config/deepseek-p010-correction-probe-prompt.md \
    --output-dir "/probe-output/$PROBE_RUN_LABEL/run-$run"
done
```

每轮查看 `p010-correction-output/$PROBE_RUN_LABEL/run-*/decision.json`：`first_candidate_count` 应为 18，`kept_candidate_ids` 和 `kept_candidates` 是二次模型保留结果，`decisions[*].mark_bbox` 是它声称看到的红笔位置。`input.jpg` 是实际发送的标准化图；`request-prompt.md`、`answer.md`、`response.json` 和 `metadata.json` 用于重放与排错；文本输出不包含 API 密钥或图片 base64。

## 进入正式流程之前的判定

先人工在 P010 原图上核对每轮保留项及 `mark_bbox`，确认“冰块”附近的真实红圈/红叉被保留、无标记项被剔除。三轮之间应基本一致；仅数量缩减或模型自报证据不足以证明有效。此探针不参与九页评分，也不能据此推断 P003/P041 不回退。只有 P010 单页结果稳定后，才讨论二阶段接入主线与九页回归。
