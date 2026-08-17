"""Guards on the root entry point.

RunPod's GitHub integration greps the repository for a runpod.serverless.start()
call and refuses to consider the endpoint valid without one. These tests keep that
call where the scan finds it and in the shape the scan matches, so a later cleanup
cannot silently break deployment.
"""

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENTRY_POINT = ROOT / "rp_handler.py"


def test_the_entry_point_is_at_the_repository_root():
    assert ENTRY_POINT.is_file()


def test_the_runpod_call_is_on_a_single_line():
    lines = ENTRY_POINT.read_text().splitlines()
    matching = [line for line in lines if "runpod.serverless.start(" in line]
    assert len(matching) == 1
    # A multi-line call is what made RunPod report "handler not found".
    assert matching[0].strip().endswith(")")


def test_the_entry_point_exposes_main_without_starting_anything():
    # Importing must not bootstrap the engine: no GPU exists here.
    import rp_handler

    assert callable(rp_handler.main)


def test_the_entry_point_delegates_to_the_worker_modules():
    import rp_handler

    from handler import bootstrap, concurrency_modifier, run_job

    assert rp_handler.bootstrap is bootstrap
    assert rp_handler.run_job is run_job
    assert rp_handler.concurrency_modifier is concurrency_modifier
