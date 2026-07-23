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
| 部署 | Docker Compose | 4 容器：API / Worker / PostgreSQL / Redis |

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

- Docker + Docker Compose
- 可访问 MiniMax API 的服务器

### 快速启动

```bash
cd backend

# 1. 配置环境变量
cp .env.example .env
# 编辑 .env，填入：
#   LLM_API_KEY=sk-xxx          （OpenAI 兼容 API Key）
#   LLM_API_BASE=https://api.openai.com/v1
#   MINIMAX_API_KEY=...         （MiniMax Token Plan Key）
#   MINIMAX_API_HOST=https://api.minimaxi.com  （国内 Key）
#   WECHAT_APP_ID=wxXXX         （小程序 AppID，生产环境必填）
#   WECHAT_APP_SECRET=xxx       （小程序 Secret，生产环境必填）

# 2. 构建并启动所有服务（Dockerfile 或模板变化后必须重新 build）
docker compose build api worker
docker compose up -d

# 3. 运行数据库迁移
docker compose exec api alembic upgrade head

# 4. 验证
curl http://localhost:8000/health
# → {"status": "ok"}
```

Dockerfile 安装 Debian 系统包时默认使用清华镜像 `https://mirrors.tuna.tsinghua.edu.cn`。如需临时切换到其他兼容镜像，可在构建时覆盖 `DEBIAN_MIRROR`：

```bash
docker compose build --build-arg DEBIAN_MIRROR=https://mirrors.aliyun.com worker
```

镜像地址应提供标准的 `/debian` 和 `/debian-security` 仓库路径；覆盖参数只影响本次镜像构建。

本版本的 PDF 镜像会安装 Noto CJK 中文字体。仅执行 `docker compose restart` 不会更新镜像；从 Git 拉取本次改动后必须重新执行上面的 `build` 和 `up -d`。历史错题如果没有 `instruction`、`prompt_text` 结构化字段，出卷接口会返回 422，需要重新上传识别后再生成错题集。

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

### 多模态识别坐标格式

MiniMax 每个识别项的 `bbox` 使用归一化角点坐标：

```text
[left, top, right, bottom]
```

坐标必须满足 `0 <= left < right <= 1` 和 `0 <= top < bottom <= 1`。后端不转换该数组，写入 `crop_region.bbox` 时同时记录 `bbox_format: "normalized_ltrb"`。现有业务尚未使用该字段裁图；未来消费者必须根据 `bbox_format` 解释坐标。

### 红色批改标记与错题粒度

图片中存在红圈、红叉、红色删除线或纠错批注时，只识别与标记关联的最小可独立作答单元。对于看词语写拼音、看拼音写词语等词语类练习，即使红色标记只覆盖一个汉字或拼音音节，也按完整词语格组保存 `raw_text`、`prompt_text`、`answer`、`question_type` 和 `bbox`；例如“课文”不能只保存 `kè`，“hé zuò”写成“合做”不能只保存“做”。完整词语范围不得扩展到相邻未标记词语。同一道编号大题中有多个标记时分别生成多条，未标记的兄弟小题不写入。落在同一作答单元上的红圈、红叉和纠正笔迹视为同一标记组。图片中没有明确红色错误标记时，识别全部独立作答单元，但仍不把整道编号大题合并成一条记录。

### 结构化题干与错题集

每个新识别项同时保存：学生实际作答 `raw_text`、原练习要求 `instruction`、干净提示材料 `prompt_text`、正确答案 `answer` 和稳定题型 `question_type`。错题详情可以展示学生错答和模型参考，但 PDF 只根据 `instruction`、`prompt_text`、`question_type` 重建练习题，不打印学生错答和答案。

`POST /api/sheets` 的 `derived_per_original` 支持 0 至 3，默认 0。值为 0 时仅生成原题且不依赖 LLM；值为 1 至 3 时需要配置 `LLM_API_KEY`，并对结构化衍生题执行非空、非原题复制和同组去重校验。PDF 使用单栏分组布局，不生成答案页。

衍生题在 API 请求内同步生成，因此 `LLM_API_KEY`、`LLM_API_BASE`、`LLM_MODEL` 必须注入 `api` 容器；项目的 Docker Compose 已同时向 API 注入这三项配置。

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
| `MINIMAX_API_KEY` | MiniMax Token Plan Key | - |
| `MINIMAX_API_HOST` | 与 Key 地区匹配的 API Host | - |
| `MINIMAX_VISION_TIMEOUT_SECONDS` | 图片理解请求超时秒数 | `60` |
| `MINIMAX_VISION_MAX_RETRIES` | 瞬时错误最大重试次数 | `2` |
| `MINIMAX_VISION_RETRY_DELAY_SECONDS` | 重试等待秒数 | `1` |
| `MINIMAX_CONFIDENCE_THRESHOLD` | 自动确认最低置信度 | `0.85` |
| `MINIMAX_IMAGE_MAX_EDGE` | 预处理图片最长边像素数 | `2048` |
| `MINIMAX_IMAGE_JPEG_QUALITY` | 预处理 JPEG 质量 | `90` |
| `DEV_MODE` | 开发模式（启用 dev-login） | `false` |
| `WECHAT_APP_ID` | 小程序 AppID | - |
| `WECHAT_APP_SECRET` | 小程序 Secret | - |
