import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    description="Draw GEMM benchmark line chart",
    formatter_class=argparse.RawTextHelpFormatter,
)
parser.add_argument(
    "csv_files",
    nargs="+",
    help="CSV files to plot (e.g. naive_gemm.csv)\n"
         "Each file should have columns: size, time_ms, gflops\n"
         "The filename (without .csv) is used as the legend label",
)
args = parser.parse_args()

plt.figure(figsize=(15, 9))

for csv_path in args.csv_files:
    path = Path(csv_path)
    label = path.stem
    sizes, gflops = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sizes.append(int(row["size"]))
            gflops.append(float(row["gflops"]))
    plt.plot(sizes, gflops, marker='o', linewidth=2, label=label)

plt.xlabel('Matrix Size')
plt.ylabel('GFLOPS')
plt.title('GEMM Kernel Benchmark (RTX 4090)')
plt.xticks(sizes, rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig('benchmark.png', dpi=150)
print("Saved benchmark.png")
