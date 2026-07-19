# 错题本 — Wrong Homework Collection App

面向小学生家长的错题管理应用。拍照录入错题 → OCR 自动识别分类 → 生成 A4 分栏错题集（原题 + 衍生题），可打印练习。

---

## 项目架构

```
┌─────────────────────┐       HTTP/JWT       ┌────────────────────────────────┐
│   微信小程序（前端）   │ ←──────────────────→ │   Python FastAPI 后端           │
│                     │                      │                                │
│  · 拍照录入          │                      │  · 账号体系（微信登录 + JWT）     │
│  · 错题库管理        │                      │  · OCR 引擎（PaddleOCR）         │
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
| 异步队列 | Celery + Redis | OCR 和 LLM 调用走异步任务 |
| OCR | PaddleOCR | 开源免费，版面分析 + 题目切分 |
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
│   │   ├── services/       # OCR, Segmenter, LLM, Derivative, PDF
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
- 2C4G 云服务器（推荐，PaddleOCR 需要一定内存）

### 快速启动

```bash
cd backend

# 1. 配置环境变量
cp .env.example .env
# 编辑 .env，填入：
#   LLM_API_KEY=sk-xxx          （OpenAI 兼容 API Key）
#   LLM_API_BASE=https://api.openai.com/v1
#   WECHAT_APP_ID=wxXXX         （小程序 AppID，生产环境必填）
#   WECHAT_APP_SECRET=xxx       （小程序 Secret，生产环境必填）

# 2. 启动所有服务
docker-compose up -d

# 3. 运行数据库迁移
docker-compose exec api alembic upgrade head

# 4. 验证
curl http://localhost:8000/health
# → {"status": "ok"}
```

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
| 🔍 错题详情 | 修正 OCR 结果 | 编辑文字/调整标签/修改难度/删除 |
| 📝 出卷 | 生成错题集 | 配置衍生题数+难度 → 生成 A4 PDF → 预览/分享 |
| 👤 我的 | 个人设置 | 年级/册别设定、手机绑定、统计信息 |

---

## API 接口一览

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
| `DEV_MODE` | 开发模式（启用 dev-login） | `false` |
| `WECHAT_APP_ID` | 小程序 AppID | - |
| `WECHAT_APP_SECRET` | 小程序 Secret | - |
