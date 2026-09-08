# MiniMax 原答波动最后一组验证与关闭条件

## 范围与结论边界（2026-09-07）

用户批准：针对识别波动再做**一组固定实验**，三个时段各10题×2次，共最多新增60次MiniMax请求。三个时段是同一组实验，不是三轮改方案。此前关于迅速整体收敛的判断过于乐观；目前不能承诺再一两轮即可达到95%/10%/30秒。

上一组同10题连续3遍，严格原答正确16/30，逐遍5/10、6/10、5/10；包含3次超时、1次非法JSON、1次不确定输出和9次字段完整但原答错误。已观察到字形错误，不能把全部失败解释成网络波动。新实验只回答跨时段稳定性及可见失败重试的收益，不调整模型、Prompt、图片或参考。

原“停止同协议扩跑”记录保留。此次是用户明确批准的、有60次硬预算的稳定性补证；完成后不再自动追加同协议实验。

## 冻结条件与实施

1. 输入复用blind-original-v1的全部10题、19格及其顺序、原像素、逐题Prompt、20秒请求超时和零自动重试。每时段遍历两次，单请求并发为1。
2. 三个时段w1/w2/w3按顺序执行，相邻开始时间至少隔3600秒。每个时段预留20次预算后才发请求；中断也保留配额。同一campaign禁止覆盖或重跑，禁止换目录绕过预算。
3. 只供离线评分的参考文件也冻结哈希。参考仍为助手审核，其中少数字形尚未得到用户逐格认证；用户确认28题裁图完整，不等于认证所有格子文字。争议另列，不通过改真值使本轮过关。
4. 每个时段后台采样主机load average与Celery active任务数，只保留数量与时间，不保存任务参数。采样不可用标null；Celery任务数不是实际MiniMax全局并发，其他客户端和平台侧负载均未知。三个时段不必然构成不同负载条件，不强制制造业务流量。
5. 本地仅运行模拟客户的测试、输入校验和离线统计。真实调用由用户在服务器执行。生产流程和已锁定文本基线不改。

参数与中文说明见[vision_stability_config.json](../scripts/vision_stability_config.json)。执行与汇总见[vision_stability.py](../scripts/vision_stability.py)。新输出目录为项目根的`output/blind-original-stability-20260907`，上传同名ZIP即可；源图片无需重新裁剪。

## 统一验收和停止条件

| 对象 | 本次验收条件 | 不满足时的关闭动作 |
|---|---|---|
| 输入与预算 | 图片/Prompt/参考哈希一致；3时段×20条；60份raw，每份恰好1次request；保留失败 | 记为数据不完整或约束不一致，回传已有数据，不删失败、不补跑 |
| 原答稳定性必要门槛 | 字段完整且所有原答格状态/文字正确≥57/60，各时段≥19/20；每题6次中明确字形/状态错误≤1次 | 关闭当前原答协议的生产候选资格，不再做同协议Prompt微调/重复抽样 |
| 时段差异 | 分时段列严格正确、字段状态、传输错误、耗时、负载采样；逐题列正确次数及输出变化 | 只能报告相关现象；负载不可观测时明确“原因不能归因”，不因此追加调用 |
| 固定重试离线回放 | 仅首条failed/incomplete时选第二条；选择不读参考。统计30个题×时段，首条完整但错误不得用第二条替换 | 无收益或错误仍多则关闭该重试路线；即使有收益，也不直接接生产 |
| 单题预算否决 | 回放计入首条和重试请求的时间；单题已超过30秒则不满足整页预算 | 否决该样本上的重试时延可行性；单题≤30秒仍不是整页达标 |

重试回放的第二条发生在第二遍，**不是真实即时重试**，其耗时合计不是整页实测。总计60条不会为了重试模拟再请求。先前30条仅作历史对照，不并入新60条的通过率或删选“状态好”的时段。10题的重复观测不能当成60道独立新题证明总体准确率。

回收后一次给出关闭记录：

- 门槛未过：关闭“当前MiniMax原答协议可直接支撑目标”的假设；标明主要障碍是稳定字形错、输出波动或交付失败。文本基线继续保留，但完整产品目标仍未达成。若约束下必须继续追求原目标，需要另行确定有新证据的识别方案，不能用不断重跑替代方案变化。
- 门槛通过：仅标为“本10题跨时段原答必要门槛通过”；转向已有文本扩围结果与整页链路集成验收，不再追加原答能力实验。
- 中断/缺文件：标“实验不完整，不能验收”，保留证据和预算，不伪称通过或继续默认扩跑。

## 最终产品关闭清单（本实验不替代）

| 最终验收项 | 固定口径 | 当前状态 |
|---|---|---|
| 联合召回 | 真实错题中，区域完整、题干/拼音/学生原答、正确答案、全部错因解析均正确≥95%；待确认也不能算自动识别成功 | 尚未达到；人工原答格实验只覆盖一个环节 |
| 误报 | 误报与重复输出占全部输出<10%；不能只看原答错误数代替误报率 | 本实验不测 |
| 整页耗时 | 上传完成至结果保存，扣除排队，包含内容识别/重试/保存；每张≤30秒 | 本实验不测；需真实整页计时 |
| 标准答案文本扩围 | 已准备12题×3次，保留开发/历史新增/合成来源区别 | 等待现有36次回传；若已执行不重跑 |

阶段锁定tag `v0.2.0-text-baseline.1`保持原指向。本实验尚未证明产品里程碑，不创建新tag。用户于2026-09-08确认仅提交，push由用户执行。本次范围为本专项文档、scripts/vision_stability.py、scripts/vision_stability_config.json、tests/unit/test_vision_stability.py共4个文件；提交信息为`test: bound MiniMax stability validation to three windows`。本地106项相关测试通过，无真实模型调用。

## 服务器命令

先同步本次提交的main代码，把`blind-original-stability-20260907.zip`放进服务器仓库`convergence-data/`。保持当前MINIMAX_*配置，不切换模型；此实验不调用文本模型。无须重启业务容器。冻结ZIP沿用2026-09-07版本，SHA256为`13606891142692d5d84c4bddc1430c35221190f6572db396455f70b893aab05b`；包内说明保留准备时的提交状态，以本仓库文档为准，图片、请求及执行命令未变。

在仓库目录执行输入检查（ZIP哈希以包旁的`.zip.sha256`文件为准，一并上传）：

```bash
cd ~/review_error_notes/review_error_notes_backend
(cd convergence-data && sha256sum -c blind-original-stability-20260907.zip.sha256)
unzip -n convergence-data/blind-original-stability-20260907.zip -d convergence-data
sha256sum -c convergence-data/blind-original-stability-20260907/source-files.sha256
(cd convergence-data/blind-original-stability-20260907 && sha256sum -c input-files.sha256)

docker compose run --rm --no-deps \
  -v "$PWD/convergence-data:/data/convergence" \
  --entrypoint python worker scripts/vision_stability.py \
  --mode preflight --source /data/convergence/blind-original-stability-20260907
```

检查必须全部通过。预检不发模型请求，输出最大60次及3600秒间隔。如果已存在campaign结果目录，不再次从w1执行；保留并回传现有文件。

以下**整段只执行一次**，约需两小时以上。使用能够持续保持的服务器终端。每个时段结束后等一小时再运行下个时段；不要同时启动其他人工MiniMax实验，业务负载正常运行即可。`--load-note`可填写已知业务负载情况；未知就保留unknown，不填写未经核实的“零并发”。

```bash
(
  set -e
  for window in w1 w2 w3; do
    sha256sum -c convergence-data/blind-original-stability-20260907/source-files.sha256
    docker compose run --rm --no-deps \
      -v "$PWD/convergence-data:/data/convergence" \
      --entrypoint python worker scripts/vision_stability.py \
      --mode run-window \
      --source /data/convergence/blind-original-stability-20260907 \
      --output /data/convergence/blind-original-stability-results-20260907 \
      --window "$window" --load-note "Other concurrent MiniMax clients: unknown"
    if [ "$window" != w3 ]; then sleep 3600; fi
  done

  docker compose run --rm --no-deps \
    -v "$PWD/convergence-data:/data/convergence" \
    --entrypoint python worker scripts/vision_stability.py \
    --mode summarize \
    --source /data/convergence/blind-original-stability-20260907 \
    --output /data/convergence/blind-original-stability-results-20260907
)
```

汇总是离线评分，不增加请求。字段failed/incomplete会保留并计失败，但只要20次全部执行完，会继续下个时段。若脚本中断、哈希失败、raw缺失或目录锁残留，不删除/解锁重跑，直接回传已有产物与控制台报错；不打印密钥或整个环境配置。

完成或中断后都可执行打包（包含所有失败和采样记录）：

```bash
zip -r convergence-data/blind-original-stability-return-20260907.zip \
  convergence-data/blind-original-stability-20260907 \
  convergence-data/blind-original-stability-results-20260907
```

正常完成预期有campaign.json、每时段environment.jsonl与results/results.json及20份raw，另有summary.json、judgments.json、retry-replay.json。采样失败允许原答统计完成，但相关归因必须标未知。不要依据字段complete数量自行判定识别通过。
