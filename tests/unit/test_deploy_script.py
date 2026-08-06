import os
import shutil
import subprocess
from pathlib import Path

import pytest


BACKEND_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = BACKEND_ROOT / "scripts" / "deploy.sh"
DEV_SCRIPT_PATH = BACKEND_ROOT / "scripts" / "dev_deploy.sh"
COMMON_SCRIPT_PATH = BACKEND_ROOT / "scripts" / "deploy_common.sh"


def test_deploy_script_is_present():
    assert SCRIPT_PATH.is_file(), "scripts/deploy.sh is missing"


def test_dev_deploy_script_is_present():
    assert DEV_SCRIPT_PATH.is_file(), "scripts/dev_deploy.sh is missing"


def _write_executable(path: Path, content: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        file.write(content)
    path.chmod(0o755)


@pytest.fixture
def deploy_project(tmp_path):
    if not SCRIPT_PATH.is_file():
        pytest.skip("deploy script is not implemented yet")

    project_dir = tmp_path / "backend"
    scripts_dir = project_dir / "scripts"
    fake_bin = tmp_path / "bin"
    bash_env = tmp_path / "bash-env"
    scripts_dir.mkdir(parents=True)
    fake_bin.mkdir()

    shutil.copy2(SCRIPT_PATH, scripts_dir / "deploy.sh")
    if DEV_SCRIPT_PATH.is_file():
        shutil.copy2(DEV_SCRIPT_PATH, scripts_dir / "dev_deploy.sh")
    if COMMON_SCRIPT_PATH.is_file():
        shutil.copy2(COMMON_SCRIPT_PATH, scripts_dir / "deploy_common.sh")
    shutil.copy2(BACKEND_ROOT / ".env.example", project_dir / ".env.example")
    shutil.copy2(BACKEND_ROOT / "docker-compose.yml", project_dir / "docker-compose.yml")

    _write_executable(
        fake_bin / "docker",
        """#!/usr/bin/env bash
set -eu
printf '%s\n' "$*" >> "$DOCKER_LOG"
printf 'DEV_MODE=%s|JWT_SECRET=%s\n' \
  "${DEV_MODE:-}" "${JWT_SECRET:-}" >> "$DOCKER_ENV_LOG"
if [[ "$*" == "compose exec -T db pg_isready -U wb_user -d wrong_book" ]]; then
  count=0
  if [[ -f "$DB_READY_COUNT_FILE" ]]; then
    count="$(cat "$DB_READY_COUNT_FILE")"
  fi
  count=$((count + 1))
  printf '%s' "$count" > "$DB_READY_COUNT_FILE"
  if (( count < FAKE_DB_READY_AFTER )); then
    exit 1
  fi
fi
""",
    )
    _write_executable(
        bash_env,
        """curl() {
set -eu
count=0
if [[ -f "$CURL_COUNT_FILE" ]]; then
  count="$(cat "$CURL_COUNT_FILE")"
fi
count=$((count + 1))
printf '%s' "$count" > "$CURL_COUNT_FILE"
if (( count >= FAKE_CURL_SUCCEED_AFTER )); then
  printf '{"status":"ok"}\n'
  return 0
fi
return 22
}
""",
    )

    return {
        "project_dir": project_dir,
        "fake_bin": fake_bin,
        "bash_env": bash_env,
        "docker_log": tmp_path / "docker.log",
        "docker_env_log": tmp_path / "docker-env.log",
        "curl_count": tmp_path / "curl-count",
        "db_ready_count": tmp_path / "db-ready-count",
    }


def _valid_env():
    return {
        "APP_ENV": "production",
        "DEV_MODE": "false",
        "LLM_API_KEY": "test-llm-key",
        "MINIMAX_API_KEY": "test-minimax-key",
        "MINIMAX_API_HOST": "https://api.minimaxi.example",
        "WECHAT_APP_ID": "wx-test-production-app",
        "WECHAT_APP_SECRET": "test-wechat-secret",
        "JWT_SECRET": "test-jwt-secret-with-sufficient-entropy",
        "AES_KEY": "test-aes-secret-with-sufficient-entropy",
        "PHONE_HMAC_SECRET": "test-phone-hmac-secret-with-sufficient-entropy",
    }


def _valid_dev_env():
    values = _valid_env()
    values.update(
        {
            "APP_ENV": "development",
            "DEV_MODE": "true",
            "DEV_LOGIN_IDENTITY": "test-stable-dev-account",
        }
    )
    return values


def _write_env(project_dir: Path, values) -> None:
    content = "".join(f"{key}={value}\n" for key, value in values.items())
    with (project_dir / ".env").open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        file.write(content)


def _run_deploy(
    deploy_project,
    *,
    script_name="deploy.sh",
    curl_succeed_after="1",
    extra_env=None,
):
    bash = os.environ.get("TEST_BASH") or shutil.which("bash")
    if not bash:
        pytest.skip("bash is required for deploy script tests")

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{deploy_project['fake_bin']}{os.pathsep}{env.get('PATH', '')}",
            "BASH_ENV": str(deploy_project["bash_env"]),
            "DOCKER_LOG": str(deploy_project["docker_log"]),
            "DOCKER_ENV_LOG": str(deploy_project["docker_env_log"]),
            "CURL_COUNT_FILE": str(deploy_project["curl_count"]),
            "DB_READY_COUNT_FILE": str(deploy_project["db_ready_count"]),
            "FAKE_CURL_SUCCEED_AFTER": curl_succeed_after,
            "FAKE_DB_READY_AFTER": "1",
            "DEPLOY_HEALTH_INTERVAL_SECONDS": "0",
            "DEPLOY_DATABASE_INTERVAL_SECONDS": "0",
        }
    )
    if extra_env:
        env.update(extra_env)

    return subprocess.run(
        [bash, f"scripts/{script_name}"],
        cwd=deploy_project["project_dir"],
        env=env,
        text=True,
        capture_output=True,
        encoding="utf-8",
    )


def test_first_run_creates_env_and_stops_before_docker(deploy_project):
    result = _run_deploy(deploy_project)

    assert result.returncode != 0
    assert (deploy_project["project_dir"] / ".env").is_file()
    assert "请先编辑" in result.stdout
    assert not deploy_project["docker_log"].exists()


def test_dev_first_run_creates_env_and_stops_before_docker(deploy_project):
    if not DEV_SCRIPT_PATH.is_file():
        pytest.skip("development deploy script is not implemented yet")

    result = _run_deploy(deploy_project, script_name="dev_deploy.sh")

    assert result.returncode != 0
    assert (deploy_project["project_dir"] / ".env").is_file()
    assert "开发环境" in result.stdout
    assert "MINIMAX_API_KEY" in result.stdout
    assert not deploy_project["docker_log"].exists()


@pytest.mark.parametrize(
    ("key", "invalid_value"),
    [
        ("LLM_API_KEY", None),
        ("LLM_API_KEY", "sk-xxx"),
        ("MINIMAX_API_KEY", None),
        ("MINIMAX_API_KEY", "..."),
        ("MINIMAX_API_HOST", None),
        ("WECHAT_APP_ID", None),
        ("WECHAT_APP_ID", "wxXXX"),
        ("WECHAT_APP_SECRET", None),
        ("WECHAT_APP_SECRET", "xxx"),
        ("JWT_SECRET", "change-me-in-production"),
        ("AES_KEY", "change-me-32bytes-secret-key-ok!"),
        ("PHONE_HMAC_SECRET", "change-me-phone-hmac-secret"),
        ("APP_ENV", "development"),
        ("DEV_MODE", "true"),
    ],
)
def test_invalid_production_config_stops_before_docker(
    deploy_project,
    key,
    invalid_value,
):
    values = _valid_env()
    if invalid_value is None:
        values.pop(key)
    else:
        values[key] = invalid_value
    _write_env(deploy_project["project_dir"], values)

    result = _run_deploy(deploy_project)

    assert result.returncode != 0
    assert key in f"{result.stdout}\n{result.stderr}"
    if invalid_value and key.endswith(("KEY", "SECRET")):
        assert invalid_value not in f"{result.stdout}\n{result.stderr}"
    assert not deploy_project["docker_log"].exists()


@pytest.mark.parametrize(
    ("key", "invalid_value"),
    [
        ("MINIMAX_API_KEY", None),
        ("MINIMAX_API_KEY", "..."),
        ("MINIMAX_API_HOST", None),
        ("DEV_LOGIN_IDENTITY", None),
        ("JWT_SECRET", "change-me-in-production"),
        ("AES_KEY", "change-me-32bytes-secret-key-ok!"),
        ("PHONE_HMAC_SECRET", "change-me-phone-hmac-secret"),
        ("APP_ENV", "production"),
        ("DEV_MODE", "false"),
    ],
)
def test_invalid_development_config_stops_before_docker(
    deploy_project,
    key,
    invalid_value,
):
    if not DEV_SCRIPT_PATH.is_file():
        pytest.skip("development deploy script is not implemented yet")

    values = _valid_dev_env()
    if invalid_value is None:
        values.pop(key)
    else:
        values[key] = invalid_value
    _write_env(deploy_project["project_dir"], values)

    result = _run_deploy(deploy_project, script_name="dev_deploy.sh")

    assert result.returncode != 0
    assert key in f"{result.stdout}\n{result.stderr}"
    if invalid_value and key.endswith(("KEY", "SECRET")):
        assert invalid_value not in f"{result.stdout}\n{result.stderr}"
    assert not deploy_project["docker_log"].exists()


def test_valid_config_runs_deployment_in_order(deploy_project):
    _write_env(deploy_project["project_dir"], _valid_env())

    result = _run_deploy(deploy_project)

    assert result.returncode == 0, result.stderr
    commands = deploy_project["docker_log"].read_text(encoding="utf-8").splitlines()
    assert commands == [
        "compose version",
        "compose config",
        "compose build api worker beat",
        "compose up -d db redis",
        "compose exec -T db pg_isready -U wb_user -d wrong_book",
        "compose run --rm api alembic upgrade head",
        "compose up -d api worker beat",
        "compose ps",
    ]
    assert "部署完成" in result.stdout


def test_valid_dev_config_runs_deployment_in_order(deploy_project):
    _write_env(deploy_project["project_dir"], _valid_dev_env())

    result = _run_deploy(deploy_project, script_name="dev_deploy.sh")

    assert result.returncode == 0, result.stderr
    commands = deploy_project["docker_log"].read_text(encoding="utf-8").splitlines()
    assert commands == [
        "compose version",
        "compose config",
        "compose build api worker beat",
        "compose up -d db redis",
        "compose exec -T db pg_isready -U wb_user -d wrong_book",
        "compose run --rm api alembic upgrade head",
        "compose up -d api worker beat",
        "compose ps",
    ]
    assert "部署完成" in result.stdout


def test_dev_deploy_warns_and_continues_without_llm(deploy_project):
    values = _valid_dev_env()
    values.pop("LLM_API_KEY")
    _write_env(deploy_project["project_dir"], values)

    result = _run_deploy(deploy_project, script_name="dev_deploy.sh")

    assert result.returncode == 0, result.stderr
    assert "衍生题功能不可用" in result.stdout
    assert deploy_project["docker_log"].exists()


@pytest.mark.parametrize("missing_key", ["WECHAT_APP_ID", "WECHAT_APP_SECRET"])
def test_dev_deploy_warns_and_continues_without_wechat(
    deploy_project,
    missing_key,
):
    values = _valid_dev_env()
    values.pop(missing_key)
    _write_env(deploy_project["project_dir"], values)

    result = _run_deploy(deploy_project, script_name="dev_deploy.sh")

    assert result.returncode == 0, result.stderr
    assert "真实微信登录和手机号能力不可用" in result.stdout
    assert deploy_project["docker_log"].exists()


def test_database_readiness_retries_before_migration(deploy_project):
    _write_env(deploy_project["project_dir"], _valid_env())

    result = _run_deploy(
        deploy_project,
        extra_env={
            "FAKE_DB_READY_AFTER": "2",
            "DEPLOY_DATABASE_ATTEMPTS": "2",
        },
    )

    assert result.returncode == 0, result.stderr
    commands = deploy_project["docker_log"].read_text(encoding="utf-8").splitlines()
    readiness = "compose exec -T db pg_isready -U wb_user -d wrong_book"
    migration = "compose run --rm api alembic upgrade head"
    assert commands.count(readiness) == 2
    assert commands.index(readiness) < commands.index(migration)


def test_health_timeout_prints_application_logs(deploy_project):
    _write_env(deploy_project["project_dir"], _valid_env())

    result = _run_deploy(
        deploy_project,
        curl_succeed_after="999",
        extra_env={
            "DEPLOY_HEALTH_ATTEMPTS": "2",
            "DEPLOY_HEALTH_INTERVAL_SECONDS": "0",
        },
    )

    assert result.returncode != 0
    commands = deploy_project["docker_log"].read_text(encoding="utf-8").splitlines()
    assert commands[-1] == "compose logs --tail=100 api worker beat"
    assert "健康检查失败" in f"{result.stdout}\n{result.stderr}"


def test_validated_env_values_override_stale_shell_values(deploy_project):
    values = _valid_env()
    _write_env(deploy_project["project_dir"], values)

    result = _run_deploy(
        deploy_project,
        extra_env={
            "DEV_MODE": "true",
            "JWT_SECRET": "stale-shell-jwt-secret",
        },
    )

    assert result.returncode == 0, result.stderr
    received_values = deploy_project["docker_env_log"].read_text(
        encoding="utf-8"
    ).splitlines()
    assert received_values
    assert set(received_values) == {
        f"DEV_MODE=false|JWT_SECRET={values['JWT_SECRET']}"
    }


def test_dev_validated_env_values_override_stale_shell_values(deploy_project):
    values = _valid_dev_env()
    _write_env(deploy_project["project_dir"], values)

    result = _run_deploy(
        deploy_project,
        script_name="dev_deploy.sh",
        extra_env={
            "DEV_MODE": "false",
            "JWT_SECRET": "stale-shell-jwt-secret",
        },
    )

    assert result.returncode == 0, result.stderr
    received_values = deploy_project["docker_env_log"].read_text(
        encoding="utf-8"
    ).splitlines()
    assert received_values
    assert set(received_values) == {
        f"DEV_MODE=true|JWT_SECRET={values['JWT_SECRET']}"
    }


def test_api_container_uses_production_security_values_from_env():
    compose = (BACKEND_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    api = compose.split("  api:", 1)[1].split("  worker:", 1)[0]

    assert "JWT_SECRET: ${JWT_SECRET}" in api
    assert 'DEV_MODE: "${DEV_MODE:-false}"' in api
    assert "JWT_SECRET: change-me-in-production" not in api
    assert 'DEV_MODE: "true"' not in api
