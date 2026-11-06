"""Every prompt template renders, and every template has a builder."""

from infrastructure.worker import prompts
from infrastructure.worker.prompts import PROMPT_DIR


def _render_all(tmp_path):
    log = tmp_path / "build.log"
    log.write_text("\n".join(f"line {i}" for i in range(60)))
    return {
        "reproduce_start.md": prompts.build_reproduce_prompt(
            paper_md_path="/home/user/paper/p.md",
            paper_images_dir="/home/user/paper/images",
            paper_pdf_path="/home/user/paper/p.pdf",
            workspace_path="/home/user",
            gpu_budget=30.0,
            max_gpus=2,
            job_time_limit="24:00:00",
            shared_datasets_path="/home/user/shared/datasets",
            shared_models_path="/home/user/shared/models",
            scaffold="codex",
        ),
        "on_job_completion.md": prompts.build_job_completion_prompt("run-1", "/o", "/e", 0),
        "on_job_progress.md": prompts.build_job_progress_prompt("run-1", "/o", "/e", elapsed_min=7),
        "on_job_cancelled.md": prompts.build_job_cancelled_prompt("run-1", "/o", "/e"),
        "on_container_build_failure.md": prompts.build_container_build_failure_prompt(
            str(log), phase="setup", failure_class="agent_fixable",
            failure_signature="sig", failure_hint="hint",
            env_comparison="unchanged", same_signature=True,
        ),
        "on_budget_exhausted.md": prompts.build_budget_exhausted_prompt(12.5, "core_claim"),
        "on_wrap_up.md": prompts.build_wrap_up_prompt("core_claim"),
        "on_continue.md": prompts.build_continue_prompt(),
        "on_iterate.md": prompts.build_iterate_prompt("core_claim", "method_runs"),
        "on_verify.md": prompts.build_verify_prompt("core_claim"),
        "on_packager.md": prompts.build_packager_prompt("/home/user", scaffold="claude-code"),
        "on_designer.md": prompts.build_designer_prompt(
            "/home/user", "/home/user/paper/p.md", "core_claim"),
        "on_designer_stage_1.md": prompts.build_designer_prompt(
            "/home/user", "/home/user/paper/p.md", "core_claim", stage=1),
        "on_designer_stage_2.md": prompts.build_designer_prompt(
            "/home/user", "/home/user/paper/p.md", "core_claim", stage=2),
        "on_designer_stage_3.md": prompts.build_designer_prompt(
            "/home/user", "/home/user/paper/p.md", "core_claim", stage=3),
        "on_validation_failure.md": prompts.build_validation_failure_prompt("FAIL: x", 1, 3),
        "on_invalid_milestone.md": prompts.build_invalid_milestone_prompt("made_up", 1, 3),
        "on_push_forward.md": prompts.build_push_forward_prompt("method_runs", "core_claim", 1, 1),
    }


def test_every_template_renders(tmp_path):
    rendered = _render_all(tmp_path)
    for name, text in rendered.items():
        assert isinstance(text, str) and text.strip(), name

    assert "30.0" in rendered["reproduce_start.md"]
    assert "24:00:00" in rendered["reproduce_start.md"]
    assert "You are running as a Codex agent" in rendered["reproduce_start.md"]
    assert "You are running as a Claude Code agent" in rendered["on_packager.md"]
    assert "run-1" in rendered["on_job_progress.md"]
    assert "line 59" in rendered["on_container_build_failure.md"]
    assert "line 19" not in rendered["on_container_build_failure.md"]  # only the last 40 lines
    assert "made_up" in rendered["on_invalid_milestone.md"]
    # The budget-exhausted prompt ends with the wrap-up prompt.
    assert rendered["on_budget_exhausted.md"].endswith(rendered["on_wrap_up.md"])


def test_every_template_file_has_a_builder(tmp_path):
    shipped = {p.name for p in PROMPT_DIR.glob("*.md")}
    assert shipped == set(_render_all(tmp_path))


def test_tools_section_exists_for_each_supported_scaffold():
    assert set(prompts.TOOLS_BY_SCAFFOLD) == {"claude-code", "codex"}


def test_prompts_use_the_release_project_name():
    former_name = "Research" + "Gym"
    for path in PROMPT_DIR.glob("*.md"):
        assert former_name not in path.read_text(), path.name
    for path in (PROMPT_DIR.parent / "workspace_template").rglob("*"):
        if path.is_file() and path.suffix in {".md", ".py", ".sh", ".yaml", ".json", ".def"}:
            assert former_name not in path.read_text(), path.name
