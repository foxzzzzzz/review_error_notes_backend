# 整页错题答案建议策略对比

此脚本是离线只读基准，不调用业务 API、不访问数据库，也不修改正式识别流程。A 使用现有整页识别要求的独立实验 Prompt，并在 JSON 示例与规则中增加正确答案建议字段；B 的第一次请求使用现有整页 Prompt 原文，再对第一阶段候选发起一次批量答案请求。两种方法接收同一份预处理 JPEG 字节。重复运行时按奇数轮 A→B、偶数轮 B→A 交替请求顺序。

## 云端运行（P003、P015、P041、P010）

按 [2026-09-23 服务器测试说明](deepseek-cross-page-correction-server-test-20260923.md) 确认归档位置。前三页从归档导出 `stage-audit/Pxxx/00-source.jpg`。P010 使用服务器已有的 `p010-correction-input/P010.jpg`，运行前检查文件存在。示例：

```bash
(
set -euo pipefail
cd /home/ubuntu/review_error_notes/review_error_notes_backend
test -f .env
archive="$PWD/chinese-marked-evidence-server-a394222-20260921T150817Z.tar.gz"
test -s "$archive"
root='chinese-marked-evidence-server-a394222-20260921T150817Z'
run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "answer-strategy-input/$run_id" "answer-strategy-output"
input="$PWD/answer-strategy-input/$run_id"
for page in P003 P015 P041; do
  tar -xOf "$archive" "$root/stage-audit/$page/00-source.jpg" > "$input/$page.jpg"
done
test -s p010-correction-input/P010.jpg
cp p010-correction-input/P010.jpg "$input/P010.jpg"
for page in P003 P015 P041 P010; do test -s "$input/$page.jpg"; done
cat > "$input/manifest.json" <<EOF
{"pages":[
 {"label":"P003","image_path":"/benchmark-input/P003.jpg"},
 {"label":"P015","image_path":"/benchmark-input/P015.jpg"},
 {"label":"P041","image_path":"/benchmark-input/P041.jpg"},
 {"label":"P010","image_path":"/benchmark-input/P010.jpg"}
]}
EOF
if sudo docker compose run --rm --no-deps -T \
  -v "$input:/benchmark-input:ro" \
  -v "$PWD/answer-strategy-output:/benchmark-output" \
  --entrypoint python worker -X utf8 -B -m scripts.deepseek_answer_strategy_benchmark \
  --manifest /benchmark-input/manifest.json \
  --output-dir "/benchmark-output/$run_id" --repeats 2; then
  printf 'PROBE_EXIT=0\n'
else
  status=$?
  printf 'PROBE_EXIT=%s; preserve partial artifacts for analysis\n' "$status"
fi
test -d "answer-strategy-output/$run_id" || mkdir -p "answer-strategy-output/$run_id"
tar -czf "$PWD/answer-strategy-$run_id.tar.gz" -C "$PWD" \
  "answer-strategy-input/$run_id" "answer-strategy-output/$run_id"
printf 'RESULT_DIR=%s\nRESULT_BUNDLE=%s\n' "$PWD/answer-strategy-output/$run_id" "$PWD/answer-strategy-$run_id.tar.gz"
)
```

如果项目 Compose 的 worker 没有把仓库 `config/` 挂载进容器，按现有 worker 配置挂载只读 `./config:/app/config:ro`；脚本、prompt 与代码版本应来自同一 checkout。先在服务器确认 `git status --short`，按项目流程同步本次脚本后再运行。该基准需要 DeepSeek 视觉凭据，沿用应用 `.env` 设置读取，不传入或记录密钥。

## 产物与评分

- `summary.json`：逐页、逐轮完整结构化结果；每个 API 请求另保存原始 HTTP body、原始模型文本和响应 JSON（如可解析），包括输出截断时的原始文本。整页 JSON 采用正式识别的解析与逐项校验规则：规范 JSON 代码块可解析，坏候选逐项剔除并记录数量，其他有效候选继续参与 B 的批量请求。批量答案按候选 ID 校验；缺项、坏项或重复 ID 会标记该批为不完整，唯一且有效的答案仍供人工评分，重复 ID 的两行都不采纳；不补造答案或重排 ID。
- 每轮 `input.jpg` 为发给 A、B 所有请求的完全相同图像字节；调用元数据记录图片 SHA-256、prompt SHA-256、单次耗时、模型、finish reason 和 usage。
- `automatic-metrics.json` 只报告请求数、耗时、解析状态、无效候选数、A 答案字段缺漏数、B 答案 ID 缺漏数、token usage 和候选数，不将模型输出称为准确率。A 某题未给答案字段时，仍保留该题供错题发现评分，答案视为缺失。
- `human-review.csv` 有 candidate 行和 truth 行。先为每道真值建立唯一 `truth_item_id`；candidate 行填写是否真错题及匹配的真值 ID，答案非空时填写 `answer_correct`。未被任何 candidate 行匹配的 truth 行计为 FN。TP/FP 由 candidate 行判断；答案覆盖率按匹配到真值的 TP 中建议非空比例计算，建议正确率按 TP 中非空建议计算，端到端正确率按正确建议 TP / 全部人工真值计算。
- 每页每轮每策略输出 `A-review-overlay.jpg` / `B-review-overlay.jpg`，蓝框编号对应 CSV candidate_id，与教师红笔区分。人工核验边界时看叠框原图并填写 truth ID；不要把历史候选 ID 当作新输出的真值。

P003/P015/P041 的历史记录可协助人工核对漏题，但不能自动映射本轮候选 ID；P010 已知有“冰快”错题，也不代表答案真值列表完整。尚无人工确认的完整答案 gold list，因此要先审核表格，再汇总识别率。建议同一输入至少跑两轮；较少轮数不能代表稳定模型准确率或延迟分布。
