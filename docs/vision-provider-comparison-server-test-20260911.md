# MiniMax与DeepSeek V4.1 Flash服务器对照

流程：本地代码提交后由用户push；服务器同步代码、上传并解压独立测试包；执行check/run/verify/pack；回传结果ZIP分析。生产默认仍为MiniMax。

## 1. 同步代码和上传测试包

沿用现有服务器后端目录，在用户完成push后执行：

```bash
cd ~/review_error_notes/review_error_notes_backend
git pull --ff-only
```

将 `vision-provider-ab-v1-20260911.zip` 上传到这个目录。测试图片独立交付，不在Git中，也不需要上传本地评分标签。ZIP应为72,157,775字节，SHA256如下：

```text
09a69954f448fe2e44e6ed2c4d054fc0413cd6160ff1e7cadb3e1cc78037563e
```

```bash
sha256sum vision-provider-ab-v1-20260911.zip
sudo unzip -n vision-provider-ab-v1-20260911.zip -d convergence-data
```

本轮实验使用已有worker镜像和其原有环境，实验包携带已封存的适配器及运行器；不需要为实验重建镜像、重启生产worker或修改VISION_PROVIDER。代码同步用于保存后端接入能力，正式切换留待结果验收。

## 2. 凭据及模型

- MiniMax沿用worker的MINIMAX_API_KEY、MINIMAX_API_HOST。
- DeepSeek实验显式复用worker已有LLM_API_KEY，并校验LLM_API_BASE主机是api.deepseek.com。视觉模型由封存配置指定为 `deepseek-flash`（V4.1 Flash的官方API名称），不受文本LLM_MODEL影响。
- 不需要将真实Key发给助手。预检只输出是否就绪和无密钥的模型/客户端信息；若主机或凭据不匹配，保留报错后反馈，不改用另一供应商重试。

## 3. 执行对照

保持在上述后端Compose目录，先预检：

```bash
sudo bash convergence-data/vision-provider-ab-v1-20260911/vision_provider_comparison_server.sh check
```

应看到155例、providers包含minimax/deepseek、planned_attempts为620、clients_ready为true、network_calls为0。预检不调用视觉服务；通过后执行：

```bash
sudo bash convergence-data/vision-provider-ab-v1-20260911/vision_provider_comparison_server.sh run
sudo bash convergence-data/vision-provider-ab-v1-20260911/vision_provider_comparison_server.sh verify
sudo bash convergence-data/vision-provider-ab-v1-20260911/vision_provider_comparison_server.sh pack
```

155例×2轮×2家共620次请求，两家交错执行，每次20秒超时，不自动重试、不投票。实际总时长取决于服务响应，使用已有持久终端避免SSH断开中断任务。鉴权拒绝或任一家连续3次传输/解析失败会停止两家后续调度，保存已完成记录；已有结果目录拒绝覆盖。

正常完成时，两家分别有310条结果、run_complete=true、integrity_complete=true；parsed和failed均如实保留。中止或失败时，也先verify/pack回传已有结果和终端报错，不原样反复重跑。

## 4. 返回结果

回传以下文件：

```text
convergence-data/vision-provider-ab-v1-20260911-return.zip
```

其中有两家分开的原始响应、输入哈希、耗时及客户端/请求模型记录，不含环境文件或认证头。助手随后与本地标签关联，分别分析新旧正例、负例、5保护目标及两轮稳定性。

本次比较同一批已知局部图、同一prompt和评分规则，模型内部视觉编码及推理参数仍有差异。这不是整页自动发现或未见图泛化成绩，不能据此直接切换生产。
