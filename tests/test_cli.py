from app import cli, jobqueue, synth


def test_folder_ingest_is_idempotent_and_reports_rejected_files(tmp_path, ns):
    for i in range(3):
        (tmp_path / f"inv{i}.png").write_bytes(synth.generate_document(i + 40000, "A")[0])
    (tmp_path / "readme.txt").write_text("not a document")
    (tmp_path / "subdir").mkdir()
    first = cli._ingest_folder(tmp_path, ns, "folder-test")
    again = cli._ingest_folder(tmp_path, ns, "folder-test")          # what --watch does on every poll
    assert first == {"queued": 3, "duplicate": 0, "rejected": 1}
    assert again == {"queued": 0, "duplicate": 3, "rejected": 1}      # re-polling a folder never double-queues
    assert jobqueue.stats(ns)["queued"] == 3
