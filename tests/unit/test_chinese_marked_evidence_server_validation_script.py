import os
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path, PurePosixPath
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "chinese_marked_evidence_server_validation.sh"


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
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
        path.chmod(0o755)

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

    def test_automatic_run_rejects_shadow_worker_on_production_redis(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_bin = temp_path / "bin"
            fake_bin.mkdir()
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
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("shares its Redis broker with the production worker", result.stderr)

    def test_automatic_run_collects_and_packs_results_without_token(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_bin = temp_path / "bin"
            fake_bin.mkdir()
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
  printf '%s\n' '[{"image_id":"11111111-1111-1111-1111-111111111111","group_type":"questions","question_count":1,"issue_code":null,"questions":[{"id":"aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"}]},{"image_id":"22222222-2222-2222-2222-222222222222","group_type":"questions","question_count":1,"issue_code":null,"questions":[{"id":"bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"}]},{"image_id":"33333333-3333-3333-3333-333333333333","group_type":"questions","question_count":1,"issue_code":null,"questions":[{"id":"cccccccc-cccc-cccc-cccc-cccccccccccc"}]}]'
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
  printf '%s\n' 'REDIS_URL=redis://redis:6379/15'
  exit 0
fi
if [[ "$*" == *"docker inspect"*"evidence-shadow-worker" ]]; then
  printf '%s\n' 'REDIS_URL=redis://redis:6379/15' 'CHINESE_MARKED_EVIDENCE_STAGE_AUDIT_ENABLED=true'
  exit 0
fi
if [[ "$*" == *"docker inspect"*"production-worker" ]]; then
  printf '%s\n' 'REDIS_URL=redis://redis:6379/0'
  exit 0
fi
if [[ "$*" == *" psql "* && "$*" == *" -At "* ]]; then printf '0\n'; exit 0; fi
if [[ "$*" == *" psql "* ]]; then printf 'fake database report\n'; exit 0; fi
if [[ "$*" == *"docker logs"* ]]; then printf 'evidence_recognition_to_commit fake\n'; exit 0; fi
if [[ "$*" == *"docker compose run"*"deepseek_raw_page_comparison.py"* ]]; then
  previous=''
  for argument in "$@"; do
    if [[ "$previous" == '-v' && "$argument" == *':/comparison' ]]; then
      host_output="${argument%:/comparison}"
      mkdir -p "$host_output/raw-direct"
      printf '%s\n' '# fake raw comparison' > "$host_output/raw-direct/summary.md"
      exit 0
    fi
    previous="$argument"
  done
  exit 7
fi
if [[ "$*" == *"docker compose run"*"chinese_marked_evidence_stage_audit.py"* ]]; then exit 0; fi
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
            self._write_executable(
                fake_bin / "jq",
                r'''#!/usr/bin/env python
import json
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
elif "old_label" in query and "protected_truth" in query:
    print(json.dumps([
        {"label": variables["old_label"], "image_id": variables["old_id"], "image_path": "/audit-inputs/old-image", "truth_count": variables["old_truth"]},
        {"label": variables["new_label"], "image_id": variables["new_id"], "image_path": "/audit-inputs/new-image", "truth_count": variables["new_truth"]},
        {"label": variables["protected_label"], "image_id": variables["protected_id"], "image_path": "/audit-inputs/protected-image", "truth_count": variables["protected_truth"]},
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
else:
    print(f"unsupported jq query: {query}", file=sys.stderr)
    sys.exit(8)
''',
            )

            output_dir = temp_path / "results"
            env = os.environ.copy()
            env["ACCESS_TOKEN"] = "archive-secret-token"
            command = [
                "bash",
                "-c",
                'export PATH="$1:$PATH"; shift; exec bash "$@"',
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
            archives = list(output_dir.glob("*.tar.gz"))
            self.assertEqual(len(archives), 1)
            with tarfile.open(archives[0], "r:gz") as archive:
                names = archive.getnames()
                self.assertTrue(any(name.endswith("review-images.json") for name in names))
                self.assertTrue(any(name.endswith("automatic-candidates.txt") for name in names))
                self.assertTrue(any(name.endswith("comparison-index.md") for name in names))
                self.assertTrue(any(name.endswith("raw-direct/summary.md") for name in names))
                for member in archive.getmembers():
                    if member.isfile():
                        content = archive.extractfile(member).read()
                        self.assertNotIn(b"archive-secret-token", content)


if __name__ == "__main__":
    unittest.main()
