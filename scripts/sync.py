#! /usr/bin/env python3

import os
import duckdb
import tempfile
from pathlib import Path
from dotenv import load_dotenv
import pyarrow.parquet as pq


def _strip(path) -> str:
    """single-quote a path for inlining into SQL"""
    return str(path).replace("'", "''")


def _merge_sql(target_file: Path, staging_glob: Path) -> str:
    if target_file.exists():
        source = f"""
            SELECT *, 1 AS __pri FROM read_parquet('{_strip(staging_glob)}', hive_partitioning=false)
            UNION ALL BY NAME
            SELECT *, 0 AS __pri FROM read_parquet('{_strip(target_file)}', hive_partitioning=false)
        """
        qualify = "QUALIFY row_number() OVER (PARTITION BY tick ORDER BY __pri DESC) = 1"
    else:
        source = f"SELECT *, 0 AS __pri FROM read_parquet('{_strip(staging_glob)}', hive_partitioning=false)"
        qualify = "QUALIFY row_number() OVER (PARTITION BY tick) = 1"

    return f"SELECT * EXCLUDE (__pri) FROM ({source}) {qualify} ORDER BY tick"


def compact_and_merge():
    src_env = os.getenv("SpiderDataset")
    tgt_env = os.getenv("SyncDataset")
    if not src_env or not tgt_env:
        print("Error: SpiderDataset or SyncDataset environment variable is not set.")
        return

    source_path = Path(src_env).expanduser()
    target_path = Path(tgt_env).expanduser()

    if not source_path.exists():
        print(f"Source root {source_path} does not exist. Nothing to compact.")
        return

    # source_path/year=2026/quarter=Q2/sid=300059/date=202606
    sub_dirs = sorted(set(f.parent for f in source_path.rglob("*.parquet")))
    if not sub_dirs:
        print("No new parquet files found in staging.")
        return

    con = duckdb.connect()
    con.execute("SET preserve_insertion_order=true;")

    for sdir in sub_dirs:
        staging_files = list(sdir.glob("*.parquet"))
        if not staging_files:
            continue

        rel_path = sdir.relative_to(source_path)
        tdir = target_path / rel_path
        tdir.mkdir(parents=True, exist_ok=True)

        print(f"Processing partition: {rel_path}")

        target_file = tdir / "part-0.parquet"
        staging_glob = sdir / "*.parquet"  # 👈 补充通配符
        merge_sql = _merge_sql(target_file, staging_glob)

        lake_schema = pq.read_schema(staging_files[0])

        fd, tmp_name = tempfile.mkstemp(dir=tdir, prefix="compacting_", suffix=".parquet")
        os.close(fd)
        temp_file_path = Path(tmp_name)
        
        try:
            merged = con.execute(merge_sql).fetch_arrow_table()
            merged = merged.select(lake_schema.names).cast(lake_schema)
            written = merged.num_rows
            pq.write_table(merged, temp_file_path, compression="snappy")
            # 原子替换
            temp_file_path.replace(target_file)
        except Exception:
            temp_file_path.unlink(missing_ok=True)
            raise

        # 成功后再清理 staging 文件
        for f in staging_files:
            f.unlink(missing_ok=True)
            
        print(f" -> Merged {written} rows into {target_file}")

    cleanup_empty_dirs(source_path)
    print("Compaction finished successfully.")


def cleanup_empty_dirs(path: Path):
    if not path.exists():
        return
    for p in sorted(path.rglob("*"), reverse=True):
        if p.is_dir() and not any(p.iterdir()):
            try:
                p.rmdir()
            except OSError:
                pass


if __name__ == "__main__":
    load_dotenv()
    compact_and_merge()
