import pandas as pd
import matplotlib.pyplot as plt


# Load model comparison results
metrics = pd.read_csv(
    r"C:\Users\firos\Desktop\M.Tech Geoinformatics\AL and ML for Geospatial Systems\Assignment 2\output\model_comparison.csv"
)


# Short model names for better visualization
models = ["MLR", "KNNR (K=5)"]


# --------------------------------------------------
# 1. EXECUTION TIME COMPARISON
# --------------------------------------------------

plt.figure(figsize=(7, 5))

plt.bar(
    models,
    metrics["Execution_Time_Seconds"]
)

plt.ylabel("Execution Time (seconds)")
plt.title("Execution Time Comparison of MLR and KNNR")

plt.tight_layout()

plt.savefig(
    r"C:\Users\firos\Desktop\M.Tech Geoinformatics\AL and ML for Geospatial Systems\Assignment 2\output\execution_time_comparison.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# 2. MEMORY CHANGE COMPARISON
# --------------------------------------------------

plt.figure(figsize=(7, 5))

plt.bar(
    models,
    metrics["Memory_Change_MB"]
)

plt.ylabel("Memory Change (MB)")
plt.title("Memory Utilization Comparison of MLR and KNNR")

plt.tight_layout()

plt.savefig(
    r"C:\Users\firos\Desktop\M.Tech Geoinformatics\AL and ML for Geospatial Systems\Assignment 2\output\memory_comparison.png",
    dpi=300
)

plt.show()


print("\nResource comparison graphs created successfully.")
