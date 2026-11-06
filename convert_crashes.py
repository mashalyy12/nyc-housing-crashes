import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os

source = "data/crashes_clean.csv"
target = "data/crashes_clean.parquet"

writer = None

for chunk in pd.read_csv(source, chunksize=100000):
    table = pa.Table.from_pandas(chunk, preserve_index=False)

    if writer is None:
        writer = pq.ParquetWriter(
            target,
            table.schema,
            compression="zstd"
        )

    writer.write_table(table)

if writer is not None:
    writer.close()

size = os.path.getsize(target) / (1024 * 1024)

print(f"Parquet created successfully: {size:.2f} MB")