from pathlib import Path
import hashlib
import re

package = Path(__file__).resolve().parent.parent
names = ["StringIds", "META_DATA_EXPORT", "TARGET_INFO_SESSION_START_TIME",
         "CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY",
         "CUPTI_ACTIVITY_KIND_MEMSET", "CUPTI_ACTIVITY_KIND_RUNTIME",
         "NVTX_EVENTS", "OSRT_API"]
versions = [("2024.2", "3.11.0"), ("2025.3", "3.20.2")]
lines = ["// Code generated from pinned official schema SQL by generate_schema.py; DO NOT EDIT.",
         "//go:generate python3 testdata/generate_schema.py",
         "package nsightsystems",
         "var tableNames = []string{" + ",".join('"' + n + '"' for n in names) + "}",
         "var schemas = map[string]map[string][]Column{"]
hashes = {}
for release, schema in versions:
    body = (package / ("testdata/schema-" + release + ".sql")).read_bytes()
    hashes[schema] = hashlib.sha256(body).hexdigest()
    lines.append('"' + schema + '": {')
    for name in names:
        match = re.search(r"CREATE TABLE " + name + r" \(([\s\S]*?)\n\);", body.decode())
        assert match, name
        columns = []
        for line in match[1].splitlines():
            col = re.match(r"\s*(\w+)\s+(INTEGER|TEXT|REAL)\b(.*)", line)
            if col:
                nullable = "false" if "NOT NULL" in col[3] else "true"
                columns.append('{Name:"' + col[1] + '",Type:"' + col[2] + '",Nullable:' + nullable + '}')
        assert columns, name
        lines.append('"' + name + '": {' + ",".join(columns) + '},')
    lines.append('},')
lines += ['}', 'var schemaSHA256 = map[string]string{']
for version, digest in hashes.items():
    lines.append('"' + version + '":"' + digest + '",')
lines.append('}')
(package / "schema.go").write_text("\n".join(lines) + "\n")
import subprocess
subprocess.run(["gofmt", "-w", str(package / "schema.go")], check=True)
