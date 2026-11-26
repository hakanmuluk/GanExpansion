"""Download versioned models/benchmark inputs into their original relative paths."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_path(root, relative):
    """Reject archive traversal and existing symlinks that leave the root."""
    pure = PurePosixPath(relative)
    if not relative or pure.is_absolute() or ".." in pure.parts or "\\" in relative:
        raise ValueError(f"Unsafe asset path: {relative!r}")
    root = Path(root).resolve()
    target = root.joinpath(*pure.parts)
    if not target.resolve().is_relative_to(root):
        raise ValueError(f"Asset path leaves destination: {relative!r}")
    if target.is_symlink():
        raise ValueError(f"Refusing a symlink asset: {relative!r}")
    return target


def resolve_bundles(manifest, selection):
    result, active = [], set()
    def visit(name):
        if name in active:
            raise ValueError("Cyclic asset dependencies")
        if name in result:
            return
        active.add(name)
        for dependency in manifest["bundles"][name].get("requires", []):
            visit(dependency)
        active.remove(name)
        result.append(name)
    for name in manifest["selections"][selection]:
        visit(name)
    return result


def matches(root, entry):
    path = safe_path(root, entry["path"])
    return path.is_file() and path.stat().st_size == entry["bytes"] and sha256(path) == entry["sha256"]


def install_bundle(bundle, root, *, base_url=None, force=False, verify_only=False):
    root = Path(root).resolve()
    missing = [e for e in bundle["files"] if not matches(root, e)]
    if not missing:
        print(f"Verified: {bundle['archive']} ({len(bundle['files'])} files)")
        return
    if verify_only:
        raise ValueError("Missing or modified assets: " + ", ".join(e["path"] for e in missing))
    for entry in missing:
        destination = safe_path(root, entry["path"])
        if destination.exists() and not force:
            raise ValueError(f"Existing asset has a different checksum: {destination}. Use --force to replace it.")
    root.mkdir(parents=True, exist_ok=True)
    cache = root / ".cache" / "ganexpansion"
    cache.mkdir(parents=True, exist_ok=True)
    url = (base_url.rstrip("/") + "/" + bundle["archive"]) if base_url else bundle["url"]
    print(f"Downloading {bundle['archive']} ({bundle['bytes'] / 1e6:.1f} MB)", flush=True)
    with tempfile.TemporaryDirectory(dir=cache) as folder:
        temporary = Path(folder)
        archive = temporary / "download.zip"
        request = urllib.request.Request(url, headers={"User-Agent": "GanExpansion/1.0"})
        with urllib.request.urlopen(request, timeout=60) as source, archive.open("wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        if archive.stat().st_size != bundle["bytes"] or sha256(archive) != bundle["sha256"]:
            raise ValueError(f"Download checksum mismatch: {bundle['archive']}")
        expected = {e["path"]: e for e in bundle["files"]}
        staged = temporary / "verified"
        with zipfile.ZipFile(archive) as z:
            members = z.infolist()
            if len(members) != len(expected) or {m.filename for m in members} != set(expected):
                raise ValueError("Archive members do not match the release manifest")
            for member in members:
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode) or member.is_dir():
                    raise ValueError(f"Unexpected non-file archive member: {member.filename}")
                entry = expected[member.filename]
                if member.file_size != entry["bytes"]:
                    raise ValueError(f"Archive member size mismatch: {member.filename}")
                destination = safe_path(staged, member.filename)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with z.open(member) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)
                if sha256(destination) != entry["sha256"]:
                    raise ValueError(f"Archive member checksum mismatch: {member.filename}")
        # Verify the complete bundle before replacing any destination file.
        for entry in missing:
            destination = safe_path(root, entry["path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(safe_path(staged, entry["path"]), destination)
    print(f"Installed and verified {len(missing)} files", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default="all", help="Selection in --manifest (default: all, tworoom, cube, cube-agreement)")
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/assets.json")
    parser.add_argument("--root", type=Path, default=ROOT, help="Destination checkout root")
    parser.add_argument("--base-url", help="Optional release mirror containing the same archive filenames")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="Replace modified files only after verifying the downloaded archive")
    parser.add_argument("--list", action="store_true", help="List bundles without downloading")
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    if args.bundle not in manifest["selections"]:
        parser.error("Unknown bundle; choose " + ", ".join(manifest["selections"]))
    names = resolve_bundles(manifest, args.bundle)
    if args.list:
        for name in names:
            bundle = manifest["bundles"][name]
            print(f"{name}: {len(bundle['files'])} files, {bundle['bytes']/1e6:.1f} MB, {bundle['url']}")
        return
    try:
        for name in names:
            install_bundle(manifest["bundles"][name], args.root, base_url=args.base_url,
                           force=args.force, verify_only=args.verify_only)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Asset download failed: {error}\n")


if __name__ == "__main__":
    main()
