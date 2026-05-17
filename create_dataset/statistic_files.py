from pathlib import Path

ROOT_DIR = "."

# Ngưỡng size
MB_50 = 50 * 1024 * 1024
MB_100 = 100 * 1024 * 1024

files_over_50 = []
files_over_100 = []

for file_path in Path(ROOT_DIR).rglob("*"):
    if file_path.is_file():
        try:
            size = file_path.stat().st_size

            if size > MB_50:
                files_over_50.append((file_path, size))

            if size > MB_100:
                files_over_100.append((file_path, size))

        except Exception as e:
            print(f"Error reading {file_path}: {e}")


def format_size(size_bytes):
    return f"{size_bytes / (1024 * 1024):.2f} MB"

print("\n=== Files > 50MB ===")
for path, size in sorted(files_over_50, key=lambda x: x[1], reverse=True):
    print(f"{format_size(size):>10}  |  {path}")

print("\n=== Files > 100MB ===")
for path, size in sorted(files_over_100, key=lambda x: x[1], reverse=True):
    print(f"{format_size(size):>10}  |  {path}")