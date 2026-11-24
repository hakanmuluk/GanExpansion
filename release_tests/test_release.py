"""Release-specific checks for asset integrity and runnable-case selection."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import pytest
from scripts.download_assets import install_bundle, resolve_bundles, safe_path, sha256
from examples.run import specification


def archive(tmp_path, *, path="artifacts/final/model.pt", content=b"model bytes"):
    p=tmp_path/"bundle.zip"
    with zipfile.ZipFile(p,"w") as z:z.writestr(path,content)
    return {"archive":p.name,"url":p.as_uri(),"bytes":p.stat().st_size,"sha256":sha256(p),
            "files":[{"path":path,"bytes":len(content),"sha256":hashlib.sha256(content).hexdigest()}]}


def test_download_extract_verify_and_reuse(tmp_path):
    bundle=archive(tmp_path);root=tmp_path/"checkout"
    install_bundle(bundle,root)
    assert (root/bundle["files"][0]["path"]).read_bytes()==b"model bytes"
    # No network is needed when all installed members match.
    bundle["url"]="http://invalid.invalid/missing"
    install_bundle(bundle,root,verify_only=True)


def test_reject_corrupt_archive_without_replacing_existing_model(tmp_path):
    bundle=archive(tmp_path);root=tmp_path/"checkout"
    target=root/bundle["files"][0]["path"];target.parent.mkdir(parents=True);target.write_bytes(b"previous")
    bundle["sha256"]="0"*64
    with pytest.raises(ValueError,match="checksum"):
        install_bundle(bundle,root,force=True)
    assert target.read_bytes()==b"previous"


def test_reject_corrupt_member_before_install(tmp_path):
    bundle=archive(tmp_path);bundle["files"][0]["sha256"]="0"*64;root=tmp_path/"checkout"
    with pytest.raises(ValueError,match="member checksum"):
        install_bundle(bundle,root)
    assert not (root/bundle["files"][0]["path"]).exists()


@pytest.mark.parametrize("name",["../outside","/absolute","foo/../../outside","foo\\outside"])
def test_reject_unsafe_member_paths(tmp_path,name):
    with pytest.raises(ValueError,match="Unsafe"):
        safe_path(tmp_path,name)


def test_reject_symlink_escape(tmp_path):
    root=tmp_path/"root";root.mkdir();outside=tmp_path/"outside";outside.mkdir()
    (root/"link").symlink_to(outside,target_is_directory=True)
    with pytest.raises(ValueError,match="leaves"):
        safe_path(root,"link/model.pt")


def test_modified_model_requires_explicit_force(tmp_path):
    bundle=archive(tmp_path);root=tmp_path/"root";target=root/bundle["files"][0]["path"]
    target.parent.mkdir(parents=True);target.write_bytes(b"user model")
    with pytest.raises(ValueError,match="--force"):
        install_bundle(bundle,root)
    install_bundle(bundle,root,force=True)
    assert target.read_bytes()==b"model bytes"


def test_release_dependencies():
    root=Path(__file__).resolve().parents[1];manifest=json.loads((root/"configs/assets.json").read_text())
    names=resolve_bundles(manifest,"cube-agreement")
    assert names==["cube-models","cube-agreement-models","benchmark-inputs"]
    all_names=resolve_bundles(manifest,"all")
    assert len(all_names)==len(set(all_names))==4


def args(env,case,method=None,model_seed=None):
    return argparse.Namespace(env=env,case=case,method=method,model_seed=model_seed,
                              count=4,codes=2,stratum="supported_conflict")


@pytest.mark.parametrize("env",["tworoom","cube"])
@pytest.mark.parametrize("case",["base","keepout","waypoint","compose"])
def test_every_documented_case_has_explicit_assets(env,case):
    s=specification(args(env,case))
    assert s["required_files"] and s["download_bundles"]
    assert s["model_seed"]==(43 if env=="cube" and case=="compose" else 42)
    if env=="cube" and case=="compose":
        assert "action_adapter_v3_s43" in s["actor"]
        assert s["base"]=="cube/artifacts/final/base_policy.pt"


def test_invalid_method_and_model_seed_rejected():
    with pytest.raises(ValueError):specification(args("tworoom","compose","route"))
    with pytest.raises(ValueError):specification(args("cube","compose","task-aware"))
    with pytest.raises(ValueError):specification(args("cube","base",model_seed=123))


def test_manifest_installed_assets_when_present():
    root=Path(__file__).resolve().parents[1];j=json.loads((root/"configs/assets.json").read_text())
    if not (root/"artifacts/final/base_policy.pt").exists():
        pytest.skip("Run scripts/download_assets.py --bundle all to check checkpoints")
    for bundle in j["bundles"].values():
        for entry in bundle["files"]:
            p=root/entry["path"]
            assert p.is_file() and p.stat().st_size==entry["bytes"] and sha256(p)==entry["sha256"],p
