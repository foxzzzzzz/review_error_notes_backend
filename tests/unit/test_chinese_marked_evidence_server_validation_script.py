import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path, PurePosixPath
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "chinese_marked_evidence_server_validation.sh"
RUNBOOK = ROOT.parent / "docs" / "chinese-marked-evidence-server-validation-20260915.md"


def run_script(*args: str, access_token: str = "test-secret-token"):
    env = os.environ.copy()
    env["ACCESS_TOKEN"] = access_token
    return subprocess.run(
        ["bash", SCRIPT.as_posix(), *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def as_msys_path(path: Path) -> str:
    resolved = path.resolve()
    if not resolved.drive:
        return resolved.as_posix()
    return f"/{resolved.drive[0].lower()}{resolved.as_posix()[2:]}"


class ServerValidationScriptTest(unittest.TestCase):
    @staticmethod
    def _write_executable(path: Path, content: str):
        content = content.replace("#!/usr/bin/env bash", "#!/usr/bin/bash", 1)
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
        path.chmod(0o755)

    def _write_hermetic_path_support(self, fake_bin: Path):
        self._write_executable(
            fake_bin / "env",
            "#!/usr/bin/bash\nexec \"$@\"\n",
        )
        for command_name in (
            "awk", "basename", "cat", "cp", "date", "dirname", "grep", "gzip", "mkdir",
            "realpath", "sha256sum", "sleep", "tar", "tee",
        ):
            command_path = subprocess.run(
                ["bash", "-lc", f"command -v {command_name}"],
                capture_output=True,
                check=True,
                text=True,
            ).stdout.strip()
            self._write_executable(
                fake_bin / command_name,
                f'#!/usr/bin/bash\nexec "{command_path}" "$@"\n',
            )

    def test_as_msys_path_preserves_linux_absolute_path(self):
        with patch.object(
            Path,
            "resolve",
            return_value=PurePosixPath("/tmp/validation/bin"),
        ):
            self.assertEqual(
                as_msys_path(Path("ignored")),
                "/tmp/validation/bin",
            )

    def test_report_uses_persisted_pre_commit_timing_and_ocr_text(self):
        script = SCRIPT.read_text(encoding="utf-8")

        self.assertIn(
            "{evidence_timing,pre_commit,elapsed_seconds}",
            script,
        )
        self.assertIn("{evidence_timing,pre_commit,exhausted}", script)
        self.assertIn("id, ocr_text, ocr_answer, question_type", script)
        self.assertNotIn("id, question_text, ocr_answer, question_type", script)

    def test_raw_comparison_runs_as_module_from_backend_root(self):
        script = SCRIPT.read_text(encoding="utf-8")

        self.assertIn(
            "-m scripts.deepseek_raw_page_comparison",
            script,
        )
        self.assertNotIn(
            "/app/scripts/deepseek_raw_page_comparison.py",
            script,
        )

    def test_return_package_contract_locks_primary_evidence_and_replay_hashes(self):
        script = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("return-package-contract.json", script)
        for required_path in (
            "stage-audit/<page>/candidate-ledger.json",
            "raw-direct/<page>/response.json",
            "raw-direct-bbox/<page>/response.json",
            "prepared-direct/<page>/response.json",
            "replay-hashes.json",
            "legacy-mode-comparison.json",
            "legacy-mode/stage-audit/<page>/candidate-ledger.json",
            "effective-config.json",
        ):
            self.assertIn(required_path, script)
        for required_hash_input in (
            "app/services/vision_recognition.py",
            "app/tasks/process_image.py",
            "config/deepseek-marked-page-primary-prompt.md",
            "docker-compose.yml",
            "CHINESE_QUESTION_DISPLAY_BBOX_SCALE",
        ):
            self.assertIn(required_hash_input, script)
        self.assertIn("required replay hash input missing", script)
        self.assertIn("legacy-mode/stage-audit/$label/candidate-ledger.json", script)
        self.assertIn("container_env_value", script)
        self.assertIn("json.dump", script)
        contract = json.loads(
            re.search(
                r"return-package-contract\.json\" <<'EOF'\n(.*?)\nEOF",
                script,
                re.DOTALL,
            ).group(1)
        )
        self.assertEqual(
            contract["ocr_allowed_states"],
            ["support", "conflict", "missing", "unavailable"],
        )
        self.assertEqual(
            contract["cv_availability_states"],
            ["available", "unavailable"],
        )
        self.assertEqual(
            contract["cv_coverage_semantics"],
            {"covered": "regions with CV support", "uncovered": "regions without CV support", "indeterminate": "coverage cannot be determined"},
        )

    def test_runbook_starts_primary_and_legacy_shadows_with_explicit_modes(self):
        runbook = RUNBOOK.read_text(encoding="utf-8")

        for setting in (
            "-e CHINESE_DEEPSEEK_PAGE_PRIMARY_ENABLED=true",
            "-e CHINESE_DEEPSEEK_PAGE_PROMPT_PATH=/app/config/deepseek-marked-page-primary-prompt.md",
            "-e CHINESE_LOCAL_CV_AUDIT_ENABLED=true",
            "-e CHINESE_MARKED_EVIDENCE_STAGE_AUDIT_ENABLED=true",
            "--name evidence-shadow-legacy-api -p 18001:8000",
            "--name evidence-shadow-legacy-worker",
            "-e CHINESE_DEEPSEEK_PAGE_PRIMARY_ENABLED=false",
        ):
            self.assertIn(setting, runbook)
        self.assertGreaterEqual(runbook.count("--legacy-base-url 'http://127.0.0.1:18001'"), 2)
        self.assertGreaterEqual(runbook.count("--legacy-api-container 'evidence-shadow-legacy-api'"), 2)
        self.assertGreaterEqual(runbook.count("--legacy-worker-container 'evidence-shadow-legacy-worker'"), 2)

    def test_shadow_database_can_be_discovered_without_timestamp_placeholder(self):
        script = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("discover_shadow_db", script)
        self.assertIn("wrong_book_evidence_%", script)
        self.assertNotIn("你的时间戳", script)

    def test_primary_shadow_preflight_requires_worker_flags_and_readable_prompt(self):
        script = SCRIPT.read_text(encoding="utf-8")

        for variable, expected in (
            ("CHINESE_MARKED_EVIDENCE_ENABLED", "true"),
            ("CHINESE_DEEPSEEK_PAGE_PRIMARY_ENABLED", "true"),
            ("CHINESE_MARKED_EVIDENCE_STAGE_AUDIT_ENABLED", "true"),
            ("CHINESE_LOCAL_CV_AUDIT_ENABLED", "true"),
            ("VISION_PROVIDER", "deepseek"),
        ):
            self.assertIn(
                f'container_env_value evidence-shadow-worker {variable}', script
            )
            self.assertIn(f'== "{expected}"', script)
        self.assertIn("sudo docker exec evidence-shadow-worker test -r", script)
        self.assertIn("subjects_include_chinese", script)

    def test_check_inputs_accepts_three_existing_images_without_printing_token(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            images = []
            for name in ("old.jpg", "new.jpg", "protected.jpg"):
                image = temp_path / name
                image.write_bytes(b"fixture")
                images.append(image)

            secret = "must-not-appear-in-output"
            result = run_script(
                "--check-inputs-only",
                "--old-image",
                images[0].as_posix(),
                "--new-image",
                images[1].as_posix(),
                "--protected-image",
                images[2].as_posix(),
                "--base-url",
                "http://127.0.0.1:18000/",
                "--shadow-db",
                "wrong_book_evidence_test",
                "--grade",
                "3",
                "--semester",
                "1",
                "--old-truth-count",
                "11",
                "--new-truth-count",
                "9",
                "--protected-truth-count",
                "6",
                access_token=secret,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Input validation passed", result.stdout)
            self.assertIn("OLD_TRUTH_COUNT=11", result.stdout)
            self.assertIn("NEW_TRUTH_COUNT=9", result.stdout)
            self.assertIn("PROTECTED_TRUTH_COUNT=6", result.stdout)
            self.assertNotIn(secret, result.stdout)
            self.assertNotIn(secret, result.stderr)

    def test_check_inputs_rejects_missing_full_page_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            existing = temp_path / "existing.jpg"
            existing.write_bytes(b"fixture")

            result = run_script(
                "--check-inputs-only",
                "--old-image",
                existing.as_posix(),
                "--new-image",
                (temp_path / "missing.jpg").as_posix(),
                "--protected-image",
                existing.as_posix(),
                "--base-url",
                "http://127.0.0.1:18000",
                "--shadow-db",
                "wrong_book_evidence_test",
                "--grade",
                "3",
                "--semester",
                "1",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("NEW_IMAGE is not a file", result.stderr)

    def _run_host_python_command_probe(self, fake_python_commands):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_bin = temp_path / "bin"
            fake_bin.mkdir()
            self._write_hermetic_path_support(fake_bin)
            for name in ("old.jpg", "new.jpg", "protected.jpg"):
                (temp_path / name).write_bytes(b"fixture")
            for command_name in (
                "curl", "jq", "docker", "tar", "git", "grep", "tee", "awk", "sha256sum"
            ):
                self._write_executable(
                    fake_bin / command_name,
                    "#!/usr/bin/env bash\nexit 0\n",
                )
            self._write_executable(
                fake_bin / "sudo",
                "#!/usr/bin/env bash\nprintf 'host-python-selection-reached\\n' >&2\nexit 31\n",
            )
            for command_name in fake_python_commands:
                self._write_executable(
                    fake_bin / command_name,
                    "#!/usr/bin/env bash\nexit 32\n",
                )
            python_path_assertion = (
                "command -v python >/dev/null"
                if "python" in fake_python_commands
                else "! command -v python >/dev/null"
            )
            env = os.environ.copy()
            env["ACCESS_TOKEN"] = "test-secret-token"
            return subprocess.run(
                [
                    "bash",
                    "-c",
                    f'export PATH="$1"; shift; {python_path_assertion}; exec "$BASH" "$@"',
                    "validation-test",
                    as_msys_path(fake_bin),
                    as_msys_path(SCRIPT),
                    "--old-image", (temp_path / "old.jpg").as_posix(),
                    "--new-image", (temp_path / "new.jpg").as_posix(),
                    "--protected-image", (temp_path / "protected.jpg").as_posix(),
                    "--shadow-db", "wrong_book_evidence_test",
                ],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

    def test_host_python_selection_falls_back_to_python_when_python3_is_unavailable(self):
        result = self._run_host_python_command_probe(("python",))

        self.assertEqual(result.returncode, 31, result.stderr)
        self.assertIn("host-python-selection-reached", result.stderr)

    def test_host_python_selection_reports_clear_error_when_neither_command_is_available(self):
        result = self._run_host_python_command_probe(())

        self.assertEqual(result.returncode, 2)
        self.assertIn("required command not found: python3 or python", result.stderr)

    def test_automatic_run_rejects_shadow_worker_on_production_redis(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_bin = temp_path / "bin"
            fake_bin.mkdir()
            self._write_hermetic_path_support(fake_bin)
            images = []
            for name in ("old.jpg", "new.jpg", "protected.jpg"):
                image = temp_path / name
                image.write_bytes(b"fixture")
                images.append(image)

            self._write_executable(
                fake_bin / "sudo",
                """#!/usr/bin/env bash
set -euo pipefail
if [[ "$*" == "docker compose version" ]]; then exit 0; fi
if [[ "$*" == "docker compose ps -q worker" ]]; then
  printf '%s\n' production-worker
  exit 0
fi
if [[ "$*" == *"docker inspect"* ]]; then
  printf '%s\n' 'REDIS_URL=redis://redis:6379/0'
  exit 0
fi
if [[ "$*" == *"docker logs"* ]]; then exit 0; fi
printf 'unexpected sudo command: %s\n' "$*" >&2
exit 9
""",
            )
            self._write_executable(fake_bin / "docker", "#!/usr/bin/env bash\nexit 0\n")
            self._write_executable(fake_bin / "curl", "#!/usr/bin/env bash\nexit 99\n")
            self._write_executable(fake_bin / "jq", "#!/usr/bin/env bash\nexit 99\n")
            self._write_executable(
                fake_bin / "git",
                "#!/usr/bin/env bash\nprintf '%s\\n' 313051fef267528891a0a5666f32d612107395b4\n",
            )

            env = os.environ.copy()
            env["ACCESS_TOKEN"] = "test-secret-token"
            command = [
                "bash",
                "-c",
                'export PATH="$1:$PATH"; shift; exec bash "$@"',
                "validation-test",
                as_msys_path(fake_bin),
                as_msys_path(SCRIPT),
                "--old-image",
                images[0].as_posix(),
                "--new-image",
                images[1].as_posix(),
                "--protected-image",
                images[2].as_posix(),
                "--shadow-db",
                "wrong_book_evidence_test",
                "--output-dir",
                (temp_path / "results").as_posix(),
                "--skip-human-review",
            ]
            replay_source = ROOT / "config" / "deepseek-raw-page-comparison-bbox-prompt.md"
            unavailable_replay_source = replay_source.with_suffix(".test-hidden")
            os.replace(replay_source, unavailable_replay_source)
            try:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
            finally:
                os.replace(unavailable_replay_source, replay_source)

            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("shares its Redis broker with the production worker", result.stderr)
            self.assertNotIn("required return artifact missing", result.stderr)
            self.assertNotIn("required replay hash input missing", result.stderr)
            self.assertNotIn("write_return_package_contract_best_effort: command not found", result.stderr)
            archives = list((temp_path / "results").glob("*.tar.gz"))
            self.assertEqual(len(archives), 1)
            with tarfile.open(archives[0], "r:gz") as archive:
                summary = next(
                    archive.extractfile(member).read()
                    for member in archive.getmembers()
                    if member.name.endswith("run-summary.json")
                )
                packaging_warnings = next(
                    archive.extractfile(member).read().decode("utf-8")
                    for member in archive.getmembers()
                    if member.name.endswith("packaging-warnings.log")
                )
            self.assertEqual(json.loads(summary)["exit_code"], 2)
            self.assertIn("required replay hash input missing", packaging_warnings)

    def _run_automatic_packaging_with_host_python(self, host_python_command):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_bin = temp_path / "bin"
            fake_bin.mkdir()
            self._write_hermetic_path_support(fake_bin)
            images = []
            for name in ("old.jpg", "new.jpg", "protected.jpg"):
                image = temp_path / name
                image.write_bytes(b"fixture")
                images.append(image)

            self._write_executable(
                fake_bin / "curl",
                """#!/usr/bin/env bash
set -euo pipefail
args="$*"
[[ -z "${CURL_LOG:-}" ]] || printf '%s\n' "$args" >> "$CURL_LOG"
if [[ "$args" == *"/api/upload/images/status"* ]]; then
  printf '%s\n' '[{"image_id":"11111111-1111-1111-1111-111111111111","status":"needs_review","question_count":1},{"image_id":"22222222-2222-2222-2222-222222222222","status":"needs_review","question_count":1},{"image_id":"33333333-3333-3333-3333-333333333333","status":"needs_review","question_count":1}]'
elif [[ "$args" == *"/api/upload/image"* ]]; then
  if [[ "$args" == *"old.jpg"* ]]; then id=11111111-1111-1111-1111-111111111111
  elif [[ "$args" == *"new.jpg"* ]]; then id=22222222-2222-2222-2222-222222222222
  else id=33333333-3333-3333-3333-333333333333
  fi
  printf '{"image_id":"%s","status":"pending"}\n' "$id"
elif [[ "$args" == *"/api/questions/review/images/"*"/decisions"* ]]; then
  printf '%s\n' '{"status":"confirmed"}'
else
  printf '%s\n' '[{"image_id":"11111111-1111-1111-1111-111111111111","group_type":"questions","question_count":1,"issue_code":null,"questions":[{"id":"aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"}]},{"image_id":"22222222-2222-2222-2222-222222222222","group_type":"questions","question_count":1,"issue_code":null,"questions":[{"id":"bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"}]}]'
fi
""",
            )
            self._write_executable(
                fake_bin / "sudo",
                """#!/usr/bin/env bash
set -euo pipefail
if [[ "$*" == "docker compose version" ]]; then exit 0; fi
if [[ "$*" == "docker compose ps -q worker" ]]; then
  printf '%s\n' production-worker
  exit 0
fi
if [[ "$*" == *"docker inspect"*"evidence-shadow-api" ]]; then
  printf '%s\n' 'REDIS_URL=redis://redis:6379/15' 'CHINESE_DEEPSEEK_PAGE_PRIMARY_ENABLED=true'
  exit 0
fi
if [[ "$*" == *"docker inspect"*"evidence-shadow-worker" ]]; then
  worker_env=(
    'REDIS_URL=redis://redis:6379/15'
    'CHINESE_MARKED_EVIDENCE_ENABLED=true'
    'CHINESE_DEEPSEEK_PAGE_PRIMARY_ENABLED=true'
    'CHINESE_MARKED_EVIDENCE_STAGE_AUDIT_ENABLED=true'
    'CHINESE_LOCAL_CV_AUDIT_ENABLED=true'
    'CHINESE_MARKED_EVIDENCE_SUBJECTS=math, chinese'
    'VISION_PROVIDER=deepseek'
    'CHINESE_DEEPSEEK_PAGE_PROMPT_PATH=/app/config/deepseek-marked-page-primary-prompt.md'
  )
  for item in "${worker_env[@]}"; do
    key="${item%%=*}"
    if [[ "${PREFLIGHT_BAD:-}" == "$key" ]]; then
      if [[ "$key" == CHINESE_MARKED_EVIDENCE_SUBJECTS ]]; then printf '%s\n' 'CHINESE_MARKED_EVIDENCE_SUBJECTS=chinese_language';
      elif [[ "$key" == CHINESE_DEEPSEEK_PAGE_PROMPT_PATH ]]; then printf '%s\n' 'CHINESE_DEEPSEEK_PAGE_PROMPT_PATH=/missing/prompt.md';
      else printf '%s=false\n' "$key"; fi
    else
      printf '%s\n' "$item"
    fi
  done
  exit 0
fi
if [[ "$*" == *"docker inspect"*"evidence-shadow-legacy-api"* || "$*" == *"docker inspect"*"evidence-shadow-legacy-worker"* ]]; then
  printf '%s\n' 'REDIS_URL=redis://redis:6379/14' 'CHINESE_DEEPSEEK_PAGE_PRIMARY_ENABLED=false'
  exit 0
fi
if [[ "$*" == *"docker inspect"*"production-worker" ]]; then
  printf '%s\n' 'REDIS_URL=redis://redis:6379/0'
  exit 0
fi
if [[ "$*" == *"docker exec evidence-shadow-worker test -r"* ]]; then
  [[ "${PREFLIGHT_BAD:-}" == prompt_unreadable || "${PREFLIGHT_BAD:-}" == CHINESE_DEEPSEEK_PAGE_PROMPT_PATH ]] && exit 1
  exit 0
fi
if [[ "$*" == *" psql "* && "$*" == *" -At "* ]]; then printf '0\n'; exit 0; fi
if [[ "$*" == *" psql "* ]]; then printf 'fake database report\n'; exit 0; fi
if [[ "$*" == *"docker logs"* ]]; then printf 'evidence_recognition_to_commit fake\n'; exit 0; fi
if [[ "$*" == *"docker compose run"*"scripts.deepseek_raw_page_comparison"* ]]; then
  previous=''
  host_output=''
  comparison_output=''
  for argument in "$@"; do
    if [[ "$previous" == '-v' && "$argument" == *':/comparison' ]]; then
      host_output="${argument%:/comparison}"
    elif [[ "$previous" == '--output-dir' ]]; then
      comparison_output="$argument"
    fi
    previous="$argument"
  done
  group_name="${comparison_output#/comparison/}"
  [[ -n "$host_output" && -n "$group_name" ]] || exit 7
  mkdir -p "$host_output/$group_name"
  printf '%s\n' '# fake raw comparison' > "$host_output/$group_name/summary.md"
  for page in old new protected; do
    mkdir -p "$host_output/$group_name/$page"
    printf '%s\n' '{"choices":[]}' > "$host_output/$group_name/$page/response.json"
  done
  if [[ "$group_name" == prepared-direct ]]; then
    mkdir -p "$host_output/$group_name/P003"
    printf '%s\n' 'fake submitted jpeg' > "$host_output/$group_name/P003/input.jpg"
  fi
  exit 0
fi
if [[ "$*" == *"docker compose run"*"chinese_marked_evidence_stage_audit.py"* ]]; then
  host_output=''; previous=''
  for argument in "$@"; do
    if [[ "$previous" == '-v' && "$argument" == *':/audit' ]]; then host_output="${argument%:/audit}"; fi
    previous="$argument"
  done
  audit_dir="$host_output/stage-audit"
  [[ "$*" == *"legacy-mode-stage-audit"* ]] && audit_dir="$host_output/legacy-mode/stage-audit"
      for page in old new protected; do
        mkdir -p "$audit_dir/$page"
        printf '%s\n' '{}' > "$audit_dir/$page/audit.json"
        if [[ "${OMIT_LEGACY_LEDGER:-}" != true || "$audit_dir" != */legacy-mode/stage-audit || "$page" != protected ]]; then
          printf '%s\n' '[]' > "$audit_dir/$page/candidate-ledger.json"
        fi
    printf '%s\n' '```json\n{"wrong_questions":[]}\n```' > "$audit_dir/$page/primary-raw-response.md"
  done
  printf '%s\n' '{}' > "$audit_dir/summary.json"
  exit 0
fi
if [[ "$*" == *"docker rm"* ]]; then exit 0; fi
printf 'unexpected sudo command: %s\n' "$*" >&2
exit 9
""",
            )
            self._write_executable(fake_bin / "docker", "#!/usr/bin/env bash\nexit 0\n")
            self._write_executable(
                fake_bin / "git",
                "#!/usr/bin/env bash\nprintf '%s\\n' 313051fef267528891a0a5666f32d612107395b4\n",
            )
            host_python_log = temp_path / "host-python.log"
            self._write_executable(
                fake_bin / host_python_command,
                """#!/usr/bin/env bash
printf '%q ' "$@" >> "${HOST_PYTHON_LOG:?}"
printf '\\n' >> "${HOST_PYTHON_LOG:?}"
exec "${REAL_PYTHON3:?}" "$@"
""",
            )
            self._write_executable(
                fake_bin / "jq",
                f"#!/usr/bin/env {host_python_command}\n" + r'''import json
import sys

args = sys.argv[1:]
variables = {}
i = 0
while i < len(args):
    if args[i] == "--arg":
        variables[args[i + 1]] = args[i + 2]
        del args[i:i + 3]
    elif args[i] == "--argjson":
        variables[args[i + 1]] = json.loads(args[i + 2])
        del args[i:i + 3]
    elif args[i] in {"-e", "-r", "-c", "-n", "-er"}:
        i += 1
    else:
        i += 1
flags = {arg for arg in args if arg in {"-e", "-r", "-c", "-n", "-er"}}
positional = [arg for arg in args if arg not in flags]
query = positional[0]
input_path = positional[1] if len(positional) > 1 else None

def load():
    stream = open(input_path, encoding="utf-8") if input_path else sys.stdin
    try:
        return json.load(stream)
    finally:
        if input_path:
            stream.close()

if ".image_id | strings" in query:
    print(load()["image_id"])
elif "checked_at:$checked_at" in query:
    print(json.dumps({"checked_at": variables["checked_at"], "images": load()}))
elif '"  \\(.image_id)' in query:
    for item in load():
        print(f'  {item["image_id"]} {item["status"]} questions={item.get("question_count", 0)}')
elif "length == 3" in query:
    data = load()
    sys.exit(0 if len(data) == 3 and all(x["status"] not in {"pending", "segmented"} for x in data) else 1)
elif '.status == "failed"' in query:
    sys.exit(0 if any(x["status"] == "failed" for x in load()) else 1)
elif "select(.image_id == $old" in query:
    wanted = {variables["old"], variables["new"], variables["protected"]}
    print(json.dumps([x for x in load() if x["image_id"] in wanted]))
elif "old_label" in query:
    print(json.dumps([
        {"label": variables["old_label"], "image_id": variables["old_id"], "image_path": "/audit-inputs/old-image", "truth_count": variables.get("old_truth")},
        {"label": variables["new_label"], "image_id": variables["new_id"], "image_path": "/audit-inputs/new-image", "truth_count": variables.get("new_truth")},
        {"label": variables["protected_label"], "image_id": variables["protected_id"], "image_path": "/audit-inputs/protected-image", "truth_count": variables.get("protected_truth")},
    ]))
elif query.lstrip().startswith("[.[] | {"):
    print(json.dumps(load()))
elif "any(.[]; .image_id == $image_id" in query:
    data = load()
    ok = any(
        group["image_id"] == variables["image_id"]
        and any(question["id"] == variables["question_id"] for question in group.get("questions", []))
        for group in data
    )
    sys.exit(0 if ok else 1)
elif "decisions:" in query:
    print(json.dumps({"decisions": [{
        "question_id": variables["question_id"],
        "decision": "collect",
        "correct_answer": variables["correct_answer"],
        "question_type": variables["question_type"],
        "instruction": variables["instruction"],
        "prompt_text": variables["prompt_text"],
        }]}))
elif "primary_mode" in query and "legacy_mode" in query:
    print(json.dumps({"status": "completed"}))
elif query == '.status == "completed"':
    sys.exit(0 if load().get("status") == "completed" else 1)
else:
    print(f"unsupported jq query: {query}", file=sys.stderr)
    sys.exit(8)
''',
            )

            output_dir = temp_path / "results"
            env = os.environ.copy()
            env["ACCESS_TOKEN"] = "archive-secret-token"
            env["HOST_PYTHON_LOG"] = as_msys_path(host_python_log)
            env["REAL_PYTHON3"] = as_msys_path(Path(sys.executable))
            python_path_assertion = (
                "! command -v python >/dev/null"
                if host_python_command == "python3"
                else "command -v python >/dev/null"
            )
            command = [
                "bash",
                "-c",
                f'export PATH="$1"; shift; {python_path_assertion}; exec "$BASH" "$@"',
                "validation-test",
                as_msys_path(fake_bin),
                as_msys_path(SCRIPT),
            ]
            result = subprocess.run(
                command
                + [
                    "--old-image",
                    images[0].as_posix(),
                    "--new-image",
                    images[1].as_posix(),
                    "--protected-image",
                    images[2].as_posix(),
                    "--shadow-db",
                    "wrong_book_evidence_test",
                    "--legacy-base-url",
                    "http://127.0.0.1:18001",
                    "--output-dir",
                    output_dir.as_posix(),
                    "--skip-human-review",
                ],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            host_python_invocations = host_python_log.read_text(encoding="utf-8")
            self.assertIn("replay-hashes.tsv", host_python_invocations)
            self.assertIn("effective-config.tsv", host_python_invocations)
            archives = list(output_dir.glob("*.tar.gz"))
            self.assertEqual(len(archives), 1)
            with tarfile.open(archives[0], "r:gz") as archive:
                names = archive.getnames()
                self.assertTrue(any(name.endswith("review-images.json") for name in names))
                self.assertTrue(any(name.endswith("automatic-candidates.txt") for name in names))
                self.assertTrue(any(name.endswith("comparison-index.md") for name in names))
                self.assertTrue(any(name.endswith("raw-direct/summary.md") for name in names))
                self.assertTrue(any(name.endswith("raw-direct-bbox/summary.md") for name in names))
                self.assertTrue(any(name.endswith("prepared-direct/summary.md") for name in names))
                self.assertTrue(any(name.endswith("prepared-direct/P003/input.jpg") for name in names))
                self.assertTrue(any(name.endswith("effective-config.json") for name in names))
                self.assertTrue(any(name.endswith("effective-config.sha256") for name in names))
                self.assertTrue(any(name.endswith("replay-hashes.json") for name in names))
                self.assertTrue(any(name.endswith("legacy-mode-comparison.json") for name in names))
                for page in ("old", "new", "protected"):
                    self.assertTrue(any(name.endswith(f"stage-audit/{page}/audit.json") for name in names))
                    self.assertTrue(any(name.endswith(f"stage-audit/{page}/candidate-ledger.json") for name in names))
                    self.assertTrue(any(name.endswith(f"stage-audit/{page}/primary-raw-response.md") for name in names))
                    self.assertTrue(any(name.endswith(f"legacy-mode/stage-audit/{page}/audit.json") for name in names))
                    self.assertTrue(any(name.endswith(f"legacy-mode/stage-audit/{page}/candidate-ledger.json") for name in names))
                    for group in ("raw-direct", "raw-direct-bbox", "prepared-direct"):
                        self.assertTrue(any(name.endswith(f"{group}/{page}/response.json") for name in names))
                members = {
                    member.name: archive.extractfile(member).read()
                    for member in archive.getmembers()
                    if member.isfile()
                }
                effective_config_name = next(name for name in members if name.endswith("effective-config.json"))
                effective_config = json.loads(members[effective_config_name])
                self.assertNotIn("ACCESS_TOKEN", json.dumps(effective_config))
                contract_name = next(name for name in members if name.endswith("return-package-contract.json"))
                self.assertEqual(json.loads(members[contract_name])["schema_version"], 1)
                effective_hash_name = next(name for name in members if name.endswith("effective-config.sha256"))
                expected_effective_hash = members[effective_hash_name].decode("utf-8").split()[0]
                self.assertEqual(hashlib.sha256(members[effective_config_name]).hexdigest(), expected_effective_hash)
                replay_hash_name = next(name for name in members if name.endswith("replay-hashes.json"))
                for replay in json.loads(members[replay_hash_name])["files"]:
                    matching_name = next(name for name in members if name.endswith(replay["path"]))
                    self.assertEqual(hashlib.sha256(members[matching_name]).hexdigest(), replay["sha256"])
                legacy_name = next(name for name in members if name.endswith("legacy-mode-comparison.json"))
                self.assertEqual(json.loads(members[legacy_name])["status"], "completed")
                zero_candidate_ledger = next(
                    content for name, content in members.items()
                    if name.endswith("stage-audit/protected/candidate-ledger.json")
                )
                self.assertEqual(json.loads(zero_candidate_ledger), [])
                zero_candidate_raw = next(
                    content for name, content in members.items()
                    if name.endswith("stage-audit/protected/primary-raw-response.md")
                )
                self.assertIn(b'{"wrong_questions":[]}', zero_candidate_raw)
                index = next(
                    member for member in archive.getmembers()
                    if member.name.endswith("comparison-index.md")
                )
                index_text = archive.extractfile(index).read().decode("utf-8")
                self.assertIn("raw-direct-bbox/", index_text)
                self.assertIn("prepared-direct/", index_text)
                for member in archive.getmembers():
                    if member.isfile():
                        content = archive.extractfile(member).read()
                        self.assertNotIn(b"archive-secret-token", content)

            failure_env = env.copy()
            failure_env["OMIT_LEGACY_LEDGER"] = "true"
            failure_output = temp_path / "missing-artifact-results"
            failure = subprocess.run(
                command
                + [
                    "--old-image", images[0].as_posix(),
                    "--new-image", images[1].as_posix(),
                    "--protected-image", images[2].as_posix(),
                    "--shadow-db", "wrong_book_evidence_test",
                    "--legacy-base-url", "http://127.0.0.1:18001",
                    "--output-dir", failure_output.as_posix(),
                    "--skip-human-review",
                ],
                cwd=ROOT,
                env=failure_env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(failure.returncode, 0)
            self.assertIn("legacy-mode/stage-audit/protected/candidate-ledger.json", failure.stderr)
            self.assertEqual(list(failure_output.glob("*.tar.gz")), [])

            for required_value in (
                "CHINESE_MARKED_EVIDENCE_ENABLED",
                "CHINESE_DEEPSEEK_PAGE_PRIMARY_ENABLED",
                "CHINESE_MARKED_EVIDENCE_STAGE_AUDIT_ENABLED",
                "CHINESE_LOCAL_CV_AUDIT_ENABLED",
                "CHINESE_MARKED_EVIDENCE_SUBJECTS",
                "VISION_PROVIDER",
                "CHINESE_DEEPSEEK_PAGE_PROMPT_PATH",
                "prompt_unreadable",
            ):
                preflight_env = env.copy()
                preflight_env["PREFLIGHT_BAD"] = required_value
                curl_log = temp_path / f"{required_value}.curl.log"
                preflight_env["CURL_LOG"] = curl_log.as_posix()
                preflight = subprocess.run(
                    command
                    + [
                        "--old-image", images[0].as_posix(),
                        "--new-image", images[1].as_posix(),
                        "--protected-image", images[2].as_posix(),
                        "--shadow-db", "wrong_book_evidence_test",
                        "--output-dir", (temp_path / f"{required_value}-results").as_posix(),
                        "--skip-human-review",
                    ],
                    cwd=ROOT,
                    env=preflight_env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertNotEqual(preflight.returncode, 0, required_value)
                self.assertFalse(curl_log.exists(), required_value)

    def test_automatic_run_collects_and_packs_results_with_python3_only(self):
        self._run_automatic_packaging_with_host_python("python3")

    def test_automatic_run_collects_and_packs_results_with_python_only(self):
        self._run_automatic_packaging_with_host_python("python")


if __name__ == "__main__":
    unittest.main()
