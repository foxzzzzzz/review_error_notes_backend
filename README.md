# 错题本 — Wrong Homework Collection App

面向小学生家长的错题管理应用。拍照录入错题 → MiniMax 多模态识别与分类 → 重建干净题干并生成 A4 错题集，可打印练习。

---

## 项目架构

```
┌─────────────────────┐       HTTP/JWT       ┌────────────────────────────────┐
│   微信小程序（前端）   │ ←──────────────────→ │   Python FastAPI 后端           │
│                     │                      │                                │
│  · 拍照录入          │                      │  · 账号体系（微信登录 + JWT）     │
│  · 错题库管理        │                      │  · MiniMax 多模态图片识别        │
│  · 错题集生成        │                      │  · LLM 分析 + 衍生题生成         │
│  · PDF 预览/分享     │                      │  · A4 PDF 渲染（WeasyPrint）    │
│                     │                      │  · 异步任务（Celery + Redis）    │
└─────────────────────┘                      └────────────────────────────────┘
                                                     │
                                            ┌────────┴────────┐
                                            │  PostgreSQL 16  │
                                            │  Redis 7        │
                                            └─────────────────┘
```

| 层 | 技术 | 说明 |
|---|------|------|
| 前端 | 微信原生小程序 + WeUI | 4 个 Tab 页面，拍照/管理/出卷/设置 |
| 后端框架 | Python FastAPI (async) | REST API，自动生成 Swagger 文档 |
| 数据库 | PostgreSQL 16 | JSONB 字段，GIN 索引，5 张表 |
| 异步队列 | Celery + Redis | 图片识别和 LLM 调用走异步任务 |
| 图片识别 | MiniMax Token Plan | 手写内容识别、版面分组和结构化输出 |
| LLM | OpenAI 兼容 API | 题目分析 + 衍生题难度递进 |
| PDF | WeasyPrint + Jinja2 | A4 HTML 模板渲染 |
| 部署 | Docker Compose | 5 容器：API / Worker / Beat / PostgreSQL / Redis |

---

## 仓库结构

```
review_error_notes/
├── backend/                # Python FastAPI 后端（独立 Git 仓库）
│   ├── app/
│   │   ├── api/            # 路由：auth, upload, questions, sheets
│   │   ├── models/         # ORM：Student, WrongImage, WrongQuestion, PracticeSheet, SheetItem
│   │   ├── schemas/        # Pydantic 请求/响应模型
│   │   ├── services/       # MiniMax Vision, LLM, Derivative, PDF
│   │   ├── tasks/          # Celery 异步任务定义
│   │   └── utils/          # JWT, AES 加密
│   ├── templates/          # Jinja2 PDF 模板
│   ├── tests/              # pytest 测试用例
│   ├── docker-compose.yml
│   ├── Dockerfile
│   └── requirements.txt
│
└── miniprogram/            # 微信小程序前端（独立 Git 仓库）
    ├── pages/
    │   ├── capture/        # 拍照录入
    │   ├── questions/      # 错题库
    │   ├── question-detail/ # 错题详情
    │   ├── sheet/          # 出卷
    │   └── profile/        # 我的
    ├── utils/api.js        # 网络请求封装
    ├── app.js / app.json
    └── project.config.json
```

---

## 后端部署

### 环境要求

- Git
- Docker + Docker Compose v2
- curl
- 可访问 MiniMax API 的服务器

### Ubuntu 开发环境一键部署（新服务器联调）

在后端仓库根目录同步代码后执行：

```bash
git pull
bash scripts/dev_deploy.sh
```

首次运行发现 `.env` 不存在时，脚本会复制 `.env.example` 为 `.env`，
打印待配置项并停止。开发部署必须配置：

- `APP_ENV=development`、`DEV_MODE=true`
- 稳定且非空的 `DEV_LOGIN_IDENTITY`
- `MINIMAX_API_KEY`、`MINIMAX_API_HOST`
- 随机且互不共用的 `JWT_SECRET`、`AES_KEY`、`PHONE_HMAC_SECRET`

`LLM_API_KEY` 可暂时留空，但衍生题功能不可用。`WECHAT_APP_ID` 或
`WECHAT_APP_SECRET` 缺失时，真实微信登录和手机号能力不可用，可先使用
开发登录验证基础流程。补齐配置后再次运行同一命令即可构建镜像、迁移数据库、
启动全部服务并执行健康检查。

### Ubuntu 生产环境一键部署

代码同步完成后执行：

```bash
git pull
cd backend
bash scripts/deploy.sh
```

首次运行发现 `.env` 不存在时，脚本会自动复制 `.env.example` 为
`.env`，打印待配置项并停止，不会启动 Docker。编辑 `.env` 后再次运行
同一命令。`scripts/deploy.sh` 仅用于生产部署，必须配置：

- `APP_ENV=production`
- `DEV_MODE=false`
- `LLM_API_KEY`、`MINIMAX_API_KEY`、`MINIMAX_API_HOST`
- `WECHAT_APP_ID`、`WECHAT_APP_SECRET`
- 随机且互不共用的 `JWT_SECRET`、`AES_KEY`、`PHONE_HMAC_SECRET`

脚本会拒绝空值、示例 API Key、示例微信配置和默认安全密钥，然后依次
校验 Compose、构建镜像、启动 PostgreSQL/Redis、等待 PostgreSQL
就绪、执行 Alembic 迁移、启动 API/Worker/Beat 并检查 `/health`。
它不会安装 Docker、执行 `git pull`、删除卷或清空数据。

数据库就绪与 API 健康检查可在 `.env` 中调整：

```dotenv
DEPLOY_DATABASE_ATTEMPTS=30
DEPLOY_DATABASE_INTERVAL_SECONDS=2
DEPLOY_HEALTH_URL=http://127.0.0.1:8000/health
DEPLOY_HEALTH_ATTEMPTS=30
DEPLOY_HEALTH_INTERVAL_SECONDS=2
```

失败时脚本会停止并输出相关服务日志。可通过以下命令继续检查：

```bash
docker compose ps
docker compose logs --tail=100 api worker beat
```

### 手动部署与排错

如果需要逐步执行或定位脚本失败，可使用等价的手动流程：

```bash

# 1. 配置环境变量
cp .env.example .env
# 编辑 .env，填入：
#   LLM_API_KEY=sk-xxx          （OpenAI 兼容 API Key）
#   LLM_API_BASE=https://api.openai.com/v1
#   MINIMAX_API_KEY=...         （MiniMax Token Plan Key）
#   MINIMAX_API_HOST=https://api.minimaxi.com  （国内 Key）
#   WECHAT_APP_ID=wxXXX         （小程序 AppID，生产环境必填）
#   WECHAT_APP_SECRET=xxx       （小程序 Secret，生产环境必填）

# 2. 备份 PostgreSQL 后构建镜像
docker compose build api worker beat

# 3. 只启动基础服务，并在 API/Worker/Beat 启动前运行数据库迁移
docker compose up -d db redis
docker compose run --rm api alembic upgrade head

# 4. 启动应用服务并验证
docker compose up -d api worker beat
curl http://localhost:8000/health
# → {"status": "ok"}
```

Dockerfile 安装 Debian 系统包时默认使用清华镜像 `https://mirrors.tuna.tsinghua.edu.cn`。如需临时切换到其他兼容镜像，可在构建时覆盖 `DEBIAN_MIRROR`：

```bash
docker compose build --build-arg DEBIAN_MIRROR=https://mirrors.aliyun.com worker
```

镜像地址应提供标准的 `/debian` 和 `/debian-security` 仓库路径；覆盖参数只影响本次镜像构建。

本版本的 PDF 镜像会安装 Noto CJK 中文字体。仅执行 `docker compose restart` 不会更新镜像；从 Git 拉取本次改动后必须重新执行上面的 `build` 和 `up -d`。历史错题如果没有 `instruction`、`prompt_text` 结构化字段，出卷接口会返回 422，需要重新上传识别后再生成错题集。

### 软删除迁移、Beat 与 Worker

软删除功能依赖 Alembic 迁移。部署时先备份 PostgreSQL，再执行上面的 `alembic upgrade head`，最后确认 API、Worker 和 Beat 均已启动：

```bash
docker compose ps
docker compose logs --tail=100 api worker beat
```

Beat 仅按 `QUESTION_CLEANUP_INTERVAL_SECONDS` 投递任务；Worker 执行实际清理，因此 Worker 必须与 API 挂载同一个 `uploads` volume。默认每天一次清理，删除超过保留期的软删除错题，并清理无题目引用的历史图片。历史 `sheet_items` 外键会自动置空，题目快照、错题集和 PDF 不会删除。

管理员恢复不提供公开 API，只能在物理清理前通过数据库执行。以下 SQL 同时限定题目、学生和当前配置的保留窗口；将三个占位符替换为实际 UUID 和 `QUESTION_SOFT_DELETE_RETENTION_DAYS` 的整数值后执行：

```sql
UPDATE wrong_questions
SET deleted_at = NULL
WHERE id = '<question_id>'
  AND student_id = '<student_id>'
  AND deleted_at IS NOT NULL
  AND deleted_at > (NOW() AT TIME ZONE 'UTC')
      - INTERVAL '1 day' * <retention_days>;
```

影响行数为 `0` 表示题目不属于该学生、未删除、已超过当前保留窗口，或已经被物理清理。已物理清理的题目不能恢复；原图文件已经丢失时，数据库记录也无法重建图片，只能重新录入。

部署后至少验证：已出卷题目可软删除且历史错题集/PDF 仍可查看；删除题目不能通过列表、详情、图片或新出卷访问；`/api/questions/{question_id}/image` 已出现在 OpenAPI；有文件的图片能读取，缺失文件由接口返回 404，且小程序显示“原图文件不存在，请重新录入”。

### 测试环境全量初始化

当前数据库仅用于联调，不保留旧结构或测试数据。执行前必须确认当前目录、
Compose 项目名和卷列表；生产环境禁止运行。

```bash
# 1. 拉取代码后先检查目标
pwd
git rev-parse --short HEAD
docker compose ps
docker volume ls | grep review_error_notes

# 2. 经人工确认后，停止服务并删除当前 Compose 项目的数据库/文件卷
docker compose down -v

# 3. 重建镜像并初始化全新数据库
docker compose build api worker beat
docker compose up -d db redis
docker compose run --rm api alembic upgrade head

# 4. 启动应用并验证
docker compose up -d api worker beat
curl http://localhost:8000/health
docker compose logs --tail=100 api worker beat
```

若只需清空当前测试环境的数据而不删除卷，可在确认
`APP_ENV=development` 或 `APP_ENV=test` 后执行：

```bash
docker compose run --rm api \
  python -m app.maintenance.reset_debug_data \
  --confirm RESET_ALL_TEST_DATA
```

该命令会删除账户、微信身份、学生、错题、图片记录和错题集记录，并清理
`uploads`、`pdfs`、`avatars` 中的文件。确认口令由
`DEBUG_DATA_RESET_CONFIRMATION_PHRASE` 配置；其他环境或错误口令会直接拒绝。

### 服务端口

| 服务 | 端口 | 用途 |
|------|------|------|
| API | 8000 | FastAPI REST 接口 |
| PostgreSQL | 5432 | 数据库 |
| Redis | 6379 | Celery 消息队列 |
| Swagger UI | 8000/docs | 接口文档 + 在线调试 |

---

## 后端独立测试（无需小程序）

### 开发模式登录

后端支持 `DEV_MODE` 开发模式，可绕过微信直接获取 JWT Token：

```bash
# 获取 Token
curl -X POST http://localhost:8000/api/auth/dev-login \
  -H "Content-Type: application/json" \
  -d '{"code": "test_user_001"}'

# 返回
# {"token": "eyJ...", "student_id": "uuid", "need_phone": true}
```

### Swagger 在线调试

浏览器访问 `http://<server>:8000/docs`，所有接口可视化调用：

1. 先调 `POST /api/auth/dev-login` 获取 token
2. 点右上角 **Authorize** 填入 `Bearer <token>`
3. 依次调试：上传图片 → 查看错题 → 生成错题集 → 下载 PDF

### pytest 自动化测试

后端代码语法最低要求 Python 3.10；标准开发、服务器和 CI 测试环境统一
使用 Docker 镜像中的 Python 3.11。不要使用低版本本机 Python 直接执行
`pytest`。

```bash
# 推荐：无需启动数据库的单元测试
bash scripts/test_unit.sh
```

```bash
# 在容器内运行
docker-compose exec api pytest tests/ -v

# 指定测试文件
docker-compose exec api pytest tests/test_auth.py -v
docker-compose exec api pytest tests/test_questions.py -v

# 运行全部 + 输出详细结果
docker-compose exec api pytest tests/ -v --tb=short
```

```
tests/test_auth.py ........      # 8 tests: 登录/鉴权/手机绑定
tests/test_questions.py ........ # 10 tests: CRUD/过滤/404/分页/数据隔离
tests/test_sheets.py ....        # 3 tests: 出卷/边界/鉴权
tests/test_upload.py ...         # 3 tests: 图片上传
─────────────────────────────────
24 passed
```

---

## 小程序对接

### 前期准备

1. 在[微信公众平台](https://mp.weixin.qq.com)注册小程序，获取 AppID
2. 后端 `.env` 配置 `WECHAT_APP_ID` 和 `WECHAT_APP_SECRET`

### 配置步骤

```bash
cd miniprogram

# 1. 修改 project.config.json 中的 appid
#    "appid": "wxYOUR_REAL_APPID"

# 2. 修改 utils/api.js 中的后端地址
#    const BASE_URL = 'https://your-server.com/api';
#    const SERVER_BASE = 'https://your-server.com';  (sheet.js)
```

### 微信开发者工具

1. 下载[微信开发者工具](https://developers.weixin.qq.com/miniprogram/dev/devtools/download.html)
2. 导入项目 → 选择 `miniprogram/` 目录 → 填入 AppID
3. 工具内即可预览四个 Tab 页面的完整交互

### 真机测试

1. 开发者工具点击「预览」→ 手机微信扫码
2. 在手机上完整体验：拍照 → 错题库 → 出卷 → PDF 预览 → 分享到电脑打印
3. **注意**：手机端 `wx.login` 和 `getPhoneNumber` 需要真实小程序 AppID + 后端配置微信 Secret

### 页面一览

| 页面 | 功能 | 主要操作 |
|------|------|---------|
| 📷 拍照录入 | 拍照上传错题 | `wx.chooseMedia` 拍照/选图 → 上传后端 → 科目回显 |
| 📚 错题库 | 浏览管理错题 | 按科目/标签筛选 → 勾选 → 跳转出卷 |
| 🔍 错题详情 | 修正识别结果 | 编辑文字/调整标签/修改难度/删除 |
| 📝 出卷 | 生成错题集 | 配置衍生题数+难度 → 生成 A4 PDF → 预览/分享 |
| 👤 我的 | 个人设置 | 年级/册别设定、手机绑定、统计信息 |

---

## API 接口一览

### 错题区域定位坐标格式

第一阶段只为独立红色错误标记返回 `error_marks[].bbox`，题目内容本身不返回坐标。第二阶段根据整图、题目内容和已验证红标独立返回完整作答单元的 `bbox`，统一使用归一化角点坐标：

```text
[left, top, right, bottom]
```

坐标必须满足 `0 <= left < right <= 1` 和 `0 <= top < bottom <= 1`。后端还会校验红色像素、标记归属、标记中心、最大面积和区域内文字证据；RapidOCR 仅在高置信文字明确匹配另一道题时否决坐标。通过验证后写入 `crop_region.bbox` 和 `bbox_format: "normalized_ltrb"`，错题详情使用该坐标裁图。

### 红色批改标记与错题粒度

图片中存在红圈、红叉、红色删除线或纠错批注时，只识别与标记关联的最小可独立作答单元。对于看词语写拼音、看拼音写词语等词语类练习，即使红色标记只覆盖一个汉字或拼音音节，也按完整词语格组保存 `raw_text`、`prompt_text`、`answer`、`question_type` 和 `bbox`；例如“课文”不能只保存 `kè`，“hé zuò”写成“合做”不能只保存“做”。完整词语范围不得扩展到相邻未标记词语。同一道编号大题中有多个标记时分别生成多条，未标记的兄弟小题不写入。落在同一作答单元上的红圈、红叉和纠正笔迹视为同一标记组。图片中没有明确红色错误标记时，识别全部独立作答单元，但仍不把整道编号大题合并成一条记录。

### 结构化题干与错题集

每个新识别项同时保存：学生实际作答 `raw_text`、原练习要求 `instruction`、干净提示材料 `prompt_text`、正确答案 `answer` 和稳定题型 `question_type`。错题详情可以展示学生错答和模型参考，但 PDF 只根据 `instruction`、`prompt_text`、`question_type` 重建练习题，不打印学生错答和答案。

`POST /api/sheets` 的 `derived_per_original` 支持 0 至 3，默认 0。值为 0 时仅生成原题且不依赖 LLM；值为 1 至 3 时需要配置 `LLM_API_KEY`，并对结构化衍生题执行非空、非原题复制和同组去重校验。PDF 使用 A4 双栏分组布局，不生成答案页。

`POST /api/sheets` 校验成功后以 HTTP 202 创建等待记录；衍生题和 PDF 由 Celery Worker 后台生成。Worker 逐原题更新进度，失败时写入安全提示且不保留部分题目或半成品 PDF。`LLM_API_KEY`、`LLM_API_BASE`、`LLM_MODEL` 必须注入 Worker，单任务软超时由 `SHEET_GENERATION_SOFT_TIME_LIMIT_SECONDS` 配置。

衍生题默认使用 `SHEET_DERIVATIVE_GENERATION_MODE=serial` 保持逐题串行行为。云服务器完成小样本验证后可改为 `batch`，并通过 `SHEET_DERIVATIVE_BATCH_SIZE` 和 `SHEET_DERIVATIVE_MAX_CONCURRENCY` 控制每批题数及同一错题集内的最大并发批次数；修改这些变量后必须重启 Worker。单次 LLM HTTP 请求超时由 `LLM_REQUEST_TIMEOUT_SECONDS` 配置。批量响应的 JSON 解析或结构校验失败时，可由 `SHEET_DERIVATIVE_RESPONSE_VALIDATION_RETRY_COUNT` 控制同一批次的额外重试次数；重复题、题目数量不匹配等业务校验失败不会重试。

批量模式首次上线按以下顺序验证：先保留 `serial` 部署并用 8 道原题×每题 1 道衍生题确认原流程正常；再设置 `SHEET_DERIVATIVE_GENERATION_MODE=batch`、`SHEET_DERIVATIVE_BATCH_SIZE=8`、`SHEET_DERIVATIVE_MAX_CONCURRENCY=3` 并仅重启 Worker，依次使用 8、24、57 道原题且每题 1 道衍生题验证数量、顺序、内容和耗时，57 道目标为 180 秒以内。出现 HTTP 429 或排队时先把并发数降至 `2`；出现响应截断、缺题或结构校验失败时先把批大小降至 `6`。批量失败不会静默回退串行，避免任务在用户不知情时再次运行约 30 分钟。

Worker 在领取任务时记录 `generation_started_at`，完成题目、PDF 和状态写入时记录向上取整的 `generation_duration_seconds`；该时长不包含队列等待，仅在已完成历史记录中展示。终态错题集可通过 `DELETE /api/sheets/{id}` 删除；接口会同步删除其练习与题目快照、按剩余历史回算原错题统计，并将 PDF 纳入持久化文件清理任务。正在等待或处理的任务返回 409，原始错题和上传图片不会被删除。

部署本变更后必须执行 `alembic upgrade head`，重新构建并启动 API 与 Worker。验收时选择 57 道原题、每题 2 道衍生题，确认创建请求快速返回、进度持续更新、离开页面后任务继续且完成后 PDF 可查看。

| Method | Path | 说明 | 鉴权 |
|--------|------|------|------|
| GET | `/health` | 健康检查 | ❌ |
| POST | `/api/auth/login` | 微信登录 | ❌ |
| POST | `/api/auth/dev-login` | 开发模式登录 (DEV_MODE) | ❌ |
| POST | `/api/auth/bind-phone` | 绑定手机号 | ✅ |
| POST | `/api/upload/image` | 上传错题图片 | ✅ |
| GET | `/api/questions` | 错题列表（支持筛选分页） | ✅ |
| GET | `/api/questions/{id}` | 错题详情 | ✅ |
| PATCH | `/api/questions/{id}` | 修改错题 | ✅ |
| DELETE | `/api/questions/{id}` | 删除错题 | ✅ |
| POST | `/api/sheets` | 生成错题集 | ✅ |
| GET | `/api/sheets` | 历史错题集列表 | ✅ |
| DELETE | `/api/sheets/{id}` | 删除终态错题集并回算练习统计 | ✅ |
| GET | `/api/sheets/{id}/generation` | 查询后台生成状态与进度 | ✅ |
| POST | `/api/sheets/{id}/retry` | 重新投递失败的错题集 | ✅ |
| GET | `/api/sheets/{id}/review` | 读取错题集题目与最新练习结果 | ✅ |
| GET | `/api/sheets/{id}/attempts?limit=20&offset=0` | 分页查询错题集练习历史 | ✅ |
| POST | `/api/sheets/{id}/attempts` | 幂等记录一次练习结果 | ✅ |
| PATCH | `/api/sheets/{id}/attempts/{attempt_id}` | 修改最新一次练习结果 | ✅ |

---

## 数据模型

```
student (学生)
  │
  ├── 1:N ── wrong_image (错题图片，一页含多题)
  │              │
  │              └── 1:N ── wrong_question (单道错题)
  │                              │
  │                              └── N:M ── practice_sheet (错题集)
  │                                              │
  │                                              └── 1:N ── sheet_item (卷中题目)
```

5 张核心表：`students` → `wrong_images` → `wrong_questions` → `practice_sheets` → `sheet_items`

---

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DATABASE_URL` | PostgreSQL 连接串 | `postgresql+asyncpg://...` |
| `REDIS_URL` | Redis 连接串 | `redis://localhost:6379/0` |
| `JWT_SECRET` | JWT 签名密钥 | `change-me-in-production` |
| `AES_KEY` | 手机号加密密钥 (32字节) | - |
| `LLM_API_KEY` | OpenAI 兼容 API Key | 空=不启用 LLM |
| `LLM_API_BASE` | LLM API 地址 | `https://api.openai.com/v1` |
| `LLM_MODEL` | 模型名称 | `gpt-4o-mini` |
| `LLM_REQUEST_TIMEOUT_SECONDS` | 单次 LLM HTTP 请求超时秒数 | `60` |
| `MINIMAX_API_KEY` | MiniMax Token Plan Key | - |
| `MINIMAX_API_HOST` | 与 Key 地区匹配的 API Host | - |
| `MINIMAX_VISION_TIMEOUT_SECONDS` | 图片理解请求超时秒数 | `60` |
| `MINIMAX_VISION_MAX_RETRIES` | 瞬时错误最大重试次数 | `2` |
| `MINIMAX_VISION_RETRY_DELAY_SECONDS` | 重试等待秒数 | `1` |
| `MINIMAX_CONFIDENCE_THRESHOLD` | 自动确认最低置信度 | `0.85` |
| `MINIMAX_MARK_CONFIDENCE_THRESHOLD` | 红色错误标记最低模型置信度 | `0.85` |
| `MINIMAX_LOCALIZATION_CONFIDENCE_THRESHOLD` | 独立二次定位最低置信度 | `0.85` |
| `MINIMAX_LOCALIZATION_MAX_AREA_RATIO` | 单题 bbox 占整图最大面积比例 | `0.35` |
| `QUESTION_CROP_CONTEXT_PADDING_RATIO` | 详情展示框每侧上下文扩展比例，并在包含完整题目的前提下向红标中心移动 | `0.15` |
| `MARK_RED_PIXEL_MIN_RATIO` | 标记框内最低红色像素比例 | `0.005` |
| `MARK_RED_PIXEL_EXPANSION_RATIO` | 红色像素检查框向外扩展比例 | `0.08` |
| `MARK_CORRECTION_GROUP_ENABLED` | 将邻近红×与红圈合并为一次判错事件 | `true` |
| `MARK_PAIR_MAX_DISTANCE_RATIO` | 红×与红圈组合的最大归一化距离 | `0.12` |
| `MARK_ANCHOR_MAX_GAP_RATIO` | 判错事件锚点与题目候选的最大间距 | `0.08` |
| `MARK_CROSS_ONLY_MAX_GAP_RATIO` | 仅红×与题目候选的最大间距 | `0.08` |
| `MARK_DEDUP_IOU_THRESHOLD` | 重复红标合并的 IoU 阈值 | `0.8` |
| `MINIMAX_IMAGE_MAX_EDGE` | 预处理图片最长边像素数 | `2048` |
| `MINIMAX_IMAGE_JPEG_QUALITY` | 预处理 JPEG 质量 | `90` |
| `MINIMAX_MARK_MISMATCH_RETRY_COUNT` | 本地检测到红标但模型未返回有效标记时的额外重试次数 | `1` |
| `MINIMAX_LOCALIZATION_SEMANTIC_RETRY_COUNT` | 定位关联异常时的额外语义纠偏次数 | `1` |
| `LOCAL_OCR_ENABLED` | 启用 RapidOCR 反证复核 | `true` |
| `LOCAL_OCR_ENGINE` | RapidOCR 推理引擎 | `onnxruntime` |
| `LOCAL_OCR_VERSION` | RapidOCR 固定版本 | `3.9.1` |
| `LOCAL_OCR_MODEL_VERSION` | OCR 模型版本 | `PP-OCRv5` |
| `LOCAL_OCR_MODEL_TYPE` | OCR 模型规格 | `mobile` |
| `LOCAL_OCR_MODEL_PATH` | Docker 构建时预下载的模型目录 | `./models/rapidocr` |
| `LOCAL_OCR_LINE_CONFIDENCE_THRESHOLD` | 参与反证的文本行最低置信度 | `0.85` |
| `LOCAL_OCR_MIN_EFFECTIVE_CHARACTERS` | 参与反证的最少有效字符数 | `2` |
| `LOCAL_OCR_SUPPORT_SIMILARITY_THRESHOLD` | OCR 支持当前题目的相似度阈值 | `0.8` |
| `LOCAL_OCR_CONTRADICTION_SIMILARITY_THRESHOLD` | OCR 明确匹配另一题的否决阈值 | `0.9` |
| `LOCAL_OCR_FULL_PAGE_MAX_EDGE` | 无红标模式整页 OCR 的最长边像素数 | `1600` |
| `LOCAL_OCR_CROP_RECHECK_LIMIT` | 单图最多补充裁图 OCR 次数 | `3` |
| `LOCAL_OCR_MARKED_RECHECK_LIMIT` | 有红标模式歧义候选最多补充裁图 OCR 次数 | `3` |
| `LOCAL_RED_SCAN_MAX_EDGE` | 独立本地红标扫描的最长边像素数 | `1600` |
| `LOCAL_RED_COMPONENT_MIN_PIXELS` | 红色连通区域最少像素数 | `12` |
| `LOCAL_RED_COMPONENT_MAX_AREA_RATIO` | 红色连通区域最大整图面积比例 | `0.08` |
| `LOCAL_RED_COMPONENT_MAX_THINNESS_RATIO` | 红色连通区域最大长宽比 | `18` |
| `CELERY_WORKER_CONCURRENCY` | CPU OCR Worker 并发数 | `2` |
| `QUESTION_SOFT_DELETE_RETENTION_DAYS` | 软删除错题和无引用图片在物理清理前的保留天数 | `30` |
| `QUESTION_CLEANUP_INTERVAL_SECONDS` | Beat 投递周期清理任务的间隔秒数 | `86400` |
| `QUESTION_CLEANUP_BATCH_SIZE` | Worker 单次清理最多锁定和处理的记录数 | `100` |
| `SHEET_GENERATION_SOFT_TIME_LIMIT_SECONDS` | 单个异步错题集任务软超时秒数 | `7200` |
| `SHEET_DERIVATIVE_GENERATION_MODE` | 衍生题生成模式：`serial` 或 `batch` | `serial` |
| `SHEET_DERIVATIVE_BATCH_SIZE` | 批量模式每个 Prompt 的原题数 | `8` |
| `SHEET_DERIVATIVE_MAX_CONCURRENCY` | 同一错题集最大并发批次数 | `3` |
| `SHEET_DERIVATIVE_RESPONSE_VALIDATION_RETRY_COUNT` | 批量响应 JSON/结构校验失败时的额外重试次数（0-3） | `2` |
| `DEV_MODE` | 开发模式（启用 dev-login） | `false` |
| `WECHAT_APP_ID` | 小程序 AppID | - |
| `WECHAT_APP_SECRET` | 小程序 Secret | - |

### 自适应 OCR 服务器验收

代码部署到后端服务器后，传入一张或多张真实错题图片运行：

```bash
bash scripts/verify_adaptive_ocr_server.sh /path/to/image-1.jpg /path/to/image-2.jpg
```

脚本会构建 API/Worker 镜像、运行方案 B 聚焦测试、核对 `rapidocr==3.9.1` 与 `onnxruntime==1.27.0`，最后输出红标扫描和 1600 像素整页 OCR 的 p50/p95、调用次数及峰值内存。真实图片仅以只读文件挂载到临时容器，不会复制进镜像或仓库。

默认只运行不依赖数据库的方案 B 聚焦测试和真实性能测试，不会操作服务器数据库。仅在 `DEV_MODE=true` 的非生产验收环境需要同时运行实时 API 测试时使用：

```bash
RUN_LIVE_API_TESTS=true bash scripts/verify_adaptive_ocr_server.sh /path/to/image-1.jpg /path/to/image-2.jpg
```

生产环境会拒绝执行实时 API 测试，避免测试账号和迁移操作进入生产数据库。实时套件暂时排除两项已记录的历史直调单测：它们把 FastAPI 的 `Query(None)` 默认对象直接传给接口函数，与本次 OCR 改造无关。
