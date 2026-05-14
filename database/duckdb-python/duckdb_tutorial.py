"""
DuckDB + Python — Complete Tutorial Code
=========================================
SQL Analytics at Lightning Speed with DuckDB

Requirements: pip install duckdb pandas polars pyarrow numpy

This script covers:
  1. Basic DuckDB connection and SQL queries
  2. Querying CSV files directly (no import needed!)
  3. DuckDB vs Pandas performance comparison
  4. Querying Parquet files
  5. Window functions for ranking
  6. Hybrid workflow: DuckDB → Pandas → Polars
  7. Persistent databases (.duckdb files)
  8. Exporting results to CSV and Parquet
"""
import duckdb
import pandas as pd
import polars as pl
import numpy as np
import time
import os

print(f"DuckDB version: {duckdb.__version__}")

# ═══════════════════════════════════════════════════════════════
# 1. GENERATE SAMPLE DATA
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("GENERATING SAMPLE DATA (500K rows)")
print("=" * 60)

np.random.seed(42)
n = 500_000

regions = ["North", "South", "East", "West"]
products = ["Widget A", "Widget B", "Gadget X", "Gadget Y", "Doohickey Z"]
categories = ["Electronics", "Home", "Office", "Electronics", "Office"]

df_sales = pd.DataFrame({
    "order_id": range(1, n + 1),
    "region": np.random.choice(regions, n),
    "product": np.random.choice(products, n),
    "category": np.random.choice(categories, n),
    "quantity": np.random.randint(1, 20, n),
    "unit_price": np.round(np.random.uniform(5, 500, n), 2),
    "order_date": pd.date_range("2025-01-01", periods=n, freq="90s"),
})

df_sales["total_amount"] = df_sales["quantity"] * df_sales["unit_price"]
df_sales["customer_id"] = np.random.randint(1000, 5000, n)

csv_path = "sales_data.csv"
parquet_path = "sales_data.parquet"
df_sales.to_csv(csv_path, index=False)
df_sales.to_parquet(parquet_path, index=False)

csv_size = os.path.getsize(csv_path) / (1024 * 1024)
pq_size = os.path.getsize(parquet_path) / (1024 * 1024)
print(f"CSV saved:  {csv_size:.1f} MB ({n:,} rows)")
print(f"Parquet saved: {pq_size:.1f} MB ({n:,} rows)")

# ═══════════════════════════════════════════════════════════════
# 2. BASIC DUCKDB: IN-MEMORY CONNECTION
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("BASIC DUCKDB: Creating tables & querying")
print("=" * 60)

conn = duckdb.connect()  # in-memory database

conn.execute("""
    CREATE TABLE employees (
        id INTEGER,
        name VARCHAR,
        department VARCHAR,
        salary DECIMAL(10, 2)
    )
""")

conn.execute("""
    INSERT INTO employees VALUES
    (1, 'Alice', 'Engineering', 95000),
    (2, 'Bob', 'Engineering', 87000),
    (3, 'Charlie', 'Marketing', 72000),
    (4, 'Diana', 'Marketing', 78000),
    (5, 'Eve', 'Engineering', 105000),
    (6, 'Frank', 'Sales', 65000),
    (7, 'Grace', 'Sales', 71000)
""")

print("\nAll employees (ordered by salary):")
print(conn.execute("SELECT * FROM employees ORDER BY salary DESC").fetchdf())

print("\nAverage salary by department:")
print(conn.execute("""
    SELECT department,
           ROUND(AVG(salary), 2) AS avg_salary,
           COUNT(*) AS headcount
    FROM employees
    GROUP BY department
    ORDER BY avg_salary DESC
""").fetchdf())

# ═══════════════════════════════════════════════════════════════
# 3. QUERY CSV DIRECTLY — THE KILLER FEATURE
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("QUERYING CSV DIRECTLY (No pd.read_csv() needed!)")
print("=" * 60)

t0 = time.time()
result = conn.execute(f"""
    SELECT
        region,
        category,
        COUNT(*) AS num_orders,
        ROUND(SUM(total_amount), 2) AS revenue,
        ROUND(AVG(total_amount), 2) AS avg_order_value
    FROM read_csv('{csv_path}', AUTO_DETECT=TRUE)
    GROUP BY region, category
    ORDER BY revenue DESC
    LIMIT 10
""").fetchdf()
duckdb_time = time.time() - t0
print(f"DuckDB direct CSV query: {duckdb_time:.3f}s")
print(result)

# ═══════════════════════════════════════════════════════════════
# 4. DUCKDB vs PANDAS — PERFORMANCE SHOWDOWN
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DUCKDB vs PANDAS — Same query, who wins?")
print("=" * 60)

t0 = time.time()
df = pd.read_csv(csv_path)
pandas_result = (df.groupby(["region", "category"])
                 .agg(
                     num_orders=("order_id", "count"),
                     revenue=("total_amount", "sum"),
                     avg_order_value=("total_amount", "mean")
                 )
                 .sort_values("revenue", ascending=False)
                 .head(10)
                 .round(2))
pandas_time = time.time() - t0

print(f"Pandas read_csv + groupby: {pandas_time:.3f}s")
print(f"DuckDB direct query:        {duckdb_time:.3f}s")
print(f"Speedup: {pandas_time/duckdb_time:.1f}x faster with DuckDB!")

# ═══════════════════════════════════════════════════════════════
# 5. QUERY PARQUET FILES
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("QUERYING PARQUET FILES")
print("=" * 60)

t0 = time.time()
result = conn.execute(f"""
    SELECT
        product,
        ROUND(SUM(total_amount), 2) AS total_revenue,
        COUNT(*) AS units_sold,
        ROUND(AVG(quantity), 1) AS avg_qty_per_order
    FROM read_parquet('{parquet_path}')
    GROUP BY product
    ORDER BY total_revenue DESC
""").fetchdf()
pq_time = time.time() - t0
print(f"Parquet query: {pq_time:.3f}s")
print(result)

# ═══════════════════════════════════════════════════════════════
# 6. WINDOW FUNCTIONS — Top 3 products per region
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("WINDOW FUNCTIONS — Top 3 Products per Region")
print("=" * 60)

result = conn.execute(f"""
    WITH ranked AS (
        SELECT
            region,
            product,
            ROUND(SUM(total_amount), 2) AS revenue,
            ROW_NUMBER() OVER (
                PARTITION BY region
                ORDER BY SUM(total_amount) DESC
            ) AS rank
        FROM read_parquet('{parquet_path}')
        GROUP BY region, product
    )
    SELECT * FROM ranked WHERE rank <= 3
    ORDER BY region, rank
""").fetchdf()
print(result)

# ═══════════════════════════════════════════════════════════════
# 7. HYBRID WORKFLOW: DuckDB → Pandas → Polars
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("HYBRID WORKFLOW: DuckDB → Pandas → Polars")
print("=" * 60)

# Step 1: DuckDB does the heavy aggregation
print("Step 1: DuckDB aggregates 500K rows → summary...")
t0 = time.time()
summary = conn.execute(f"""
    SELECT
        region,
        category,
        DATE_TRUNC('month', order_date) AS month,
        COUNT(*) AS order_count,
        ROUND(SUM(total_amount), 2) AS monthly_revenue
    FROM read_parquet('{parquet_path}')
    GROUP BY region, category, DATE_TRUNC('month', order_date)
""").fetchdf()
print(f"   Done in {time.time() - t0:.3f}s → {len(summary)} rows")

# Step 2: Pandas for pivot table
print("\nStep 2: Pandas pivot table...")
t0 = time.time()
pivot = summary.pivot_table(
    index="month",
    columns="region",
    values="monthly_revenue",
    aggfunc="sum"
).round(2)
print(f"   Done in {time.time() - t0:.3f}s")
print(pivot.head(6))

# Step 3: Polars for final polish
print("\nStep 3: Polars for final formatting...")
t0 = time.time()
pl_df = pl.from_pandas(summary)
top_month = (pl_df
             .group_by("region")
             .agg(pl.col("monthly_revenue").max().alias("best_month_revenue"))
             .sort("best_month_revenue", descending=True))
print(f"   Done in {time.time() - t0:.3f}s")
print(top_month)

# ═══════════════════════════════════════════════════════════════
# 8. PERSISTENT DATABASE
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PERSISTENT DATABASE — Save to .duckdb file")
print("=" * 60)

db_path = "analytics.duckdb"
persistent_conn = duckdb.connect(db_path)

persistent_conn.execute(f"""
    CREATE OR REPLACE TABLE sales AS
    SELECT * FROM read_parquet('{parquet_path}')
""")

row_count = persistent_conn.execute("SELECT COUNT(*) FROM sales").fetchone()[0]
db_size = os.path.getsize(db_path) / (1024 * 1024)
print(f"Database file: {db_path} ({db_size:.1f} MB)")
print(f"Sales table: {row_count:,} rows persisted")

print("\nTop 5 customers by lifetime value:")
print(persistent_conn.execute("""
    SELECT
        customer_id,
        COUNT(*) AS orders,
        ROUND(SUM(total_amount), 2) AS lifetime_value
    FROM sales
    GROUP BY customer_id
    ORDER BY lifetime_value DESC
    LIMIT 5
""").fetchdf())

persistent_conn.close()

# ═══════════════════════════════════════════════════════════════
# 9. EXPORT RESULTS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXPORTING RESULTS")
print("=" * 60)

conn.execute(f"""
    COPY (
        SELECT region, product, ROUND(SUM(total_amount), 2) AS revenue
        FROM read_parquet('{parquet_path}')
        GROUP BY region, product
        ORDER BY revenue DESC
    ) TO 'revenue_summary.csv' (HEADER, DELIMITER ',')
""")

conn.execute(f"""
    COPY (
        SELECT region, product, ROUND(SUM(total_amount), 2) AS revenue
        FROM read_parquet('{parquet_path}')
        GROUP BY region, product
        ORDER BY revenue DESC
    ) TO 'revenue_summary.parquet' (FORMAT PARQUET)
""")

print("Exported: revenue_summary.csv")
print("Exported: revenue_summary.parquet")

exported = pd.read_csv("revenue_summary.csv")
print(f"\nExported CSV preview ({len(exported)} rows):")
print(exported.head())

# ═══════════════════════════════════════════════════════════════
# CLEANUP
# ═══════════════════════════════════════════════════════════════
conn.close()

print("\n" + "=" * 60)
print("DONE! All examples completed successfully.")
print("=" * 60)
