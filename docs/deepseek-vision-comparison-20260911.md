# DeepSeek V4.1 Flash接入与MiniMax配对对照

日期：2026-09-11。实现和本地验证完成，实际API对照未运行，未部署或改变服务器默认供应商。

## 官方接口核实

DeepSeek在2026-09-10发布V4.1 Flash，官方当前API模型名为 `deepseek-flash`，支持原生图像输入；旧V4 Flash/Vision Exp名称临时路由到该模型，不使用旧名称作为版本固定证据。[官方发布说明](https://www.deepseek.com/en/news/deepseek-v4-1-flash/)

采用 `/chat/completions` 的 user content 图文数组，直接传相同base64图像；`detail=original`、`thinking=disabled`、`max_tokens=4096`在本次配置中明确记录，不额外启用JSON模式或给DeepSeek修改prompt。参数按[官方接口文档](https://api-docs.deepseek.com/api/create-chat-completion/)核对。MiniMax没有暴露同等参数，不能把这次称为内部采样、推理预算或图像编码完全一致。

## 后端改动

- `app/services/deepseek_vision.py`只适配请求/响应，继承原识别客户端的整页/局部阶段、prompt、图像预处理和校验逻辑，避免两套业务流程分叉。
- `app/services/vision_provider.py`由 `VISION_PROVIDER` 显式选择，worker改为从工厂创建客户端。默认仍为minimax，无跨供应商自动回退，避免把失败掩盖为另一家成功。
- `app/config.py`、`.env.example`及Compose新增DeepSeek专用配置和说明。默认专用凭据；只有显式设 `DEEPSEEK_VISION_KEY_SOURCE=text_llm` 才复用现有LLM_API_KEY，且API主机必须相同。视觉模型独立于文本 `LLM_MODEL`，不将文本服务的旧模型名带入图像请求。
- 原生DeepSeek响应保存后解析最终message.content；截断、异常结构、空答案和仅有reasoning_content均不当作成功结果。仍使用原有结构化校验与候选收录策略。
- 共享图像/阶段/几何参数目前沿用既有 `MINIMAX_*` 配置名，避免本轮重命名引入行为差异；这些共享参数同样作用于DeepSeek路径。DeepSeek网络超时和重试次数单独可配，对照实验统一覆盖为20秒与0重试。

## 对照范围和公平性

原155例及所有图片、prompt、调度和评分标签保持原哈希，新包直接携带旧输入包，不重新裁图，不追加新旧题名规则。包括新55正例、旧69历史上下文正例、新3/旧28负例；5保护目标包含在旧69中，另列评分。旧字词是助手复核，不把旧69上下文当作用户穷尽认证的精确字格。

两轮×155例×两家=620次计划请求，640次硬预算；每个案例两家紧邻执行，奇偶序号交换先后。两家各自保存结果、原始HTTP响应、输入哈希、失败和完成状态，不重试、不多数投票。任何一方鉴权拒绝或连续3次失败会停止配对调度，未完成结果仍保留回传。

使用同一评分函数分别输出新旧正例、负例和保护目标每轮表现，再按同case_id/round配对。缺失、待定、重复以及传输/解析失败均不能变成正确数。语义或等价表述另行看图复核，不改既有标签迎合模型。原始响应记录实际返回model及usage（若服务提供），运行记录请求模型、适配器哈希和服务器基础客户端哈希；供应商别名不保证永久固定模型版本。

这是“已知局部范围”的分类及字词条件能力比较，不能宣称整页自动发现召回、全覆盖或生产耗时达标。若DeepSeek局部表现更好，后续仍须在同41原图全流程对照和未见照片上验泛化；不能因这155例领先就直接替换生产。

## 验证与服务器交付

179项相关测试通过：DeepSeek原生请求/响应、图片与prompt逐字段一致、默认路由、凭据隔离、失败/截断、配对执行及新旧评分，以及既有整页/局部识别、worker收录与部署配置回归。仅原Pydantic弃用警告。

本机MINIMAX_API_KEY、LLM_API_KEY、DEEPSEEK_VISION_API_KEY均未设置，没有进行实际服务调用，当前无法给出两家识别率或优胜结论。

实验包自带与后端相同哈希的DeepSeek适配器，可在原服务器worker环境运行，无需先部署后端改动或切换VISION_PROVIDER。此包显式使用 `text_llm` 复用服务器已有DeepSeek文本Key；预检验证主机及两家凭据，仅输出是否就绪，不输出密钥。

见[双通道服务器步骤](vision-provider-comparison-server-test-20260911.md)。代码与实验图片包分别交付；输入、prompt不含评分答案。原91项盲测输出及55项离线推理源码继续保持，推送由用户操作。

单独上传 `vision-provider-ab-v1-20260911.zip`，72,157,775字节（约68.8MiB），SHA256：`09a69954f448fe2e44e6ed2c4d054fc0413cd6160ff1e7cadb3e1cc78037563e`。本地校验记录 `output/vision-provider-ab-20260911/verification.json` 确认155图片/prompt配对一致，完整dry-run通过、实际API调用0。
