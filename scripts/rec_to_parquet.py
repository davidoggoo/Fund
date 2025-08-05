#!/usr/bin/env python3
"""
Convert RTAI recording files to Parquet format using DuckDB
High-performance batch conversion for offline analysis
"""

import duckdb
import gzip
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from loguru import logger

def convert_rec_to_parquet(
    rec_file: Path,
    output_file: Optional[Path] = None,
    batch_size: int = 10000
) -> Path:
    """
    Convert .rec.gz file to Parquet using DuckDB
    
    Args:
        rec_file: Path to .rec.gz input file
        output_file: Path to .parquet output file (auto-generated if None)
        batch_size: Number of records to process in each batch
        
    Returns:
        Path to output Parquet file
    """
    
    if not rec_file.exists():
        raise FileNotFoundError(f"Recording file not found: {rec_file}")
    
    # Auto-generate output filename
    if output_file is None:
        output_file = rec_file.with_suffix('.parquet')
    
    logger.info(f"🔄 Converting {rec_file} → {output_file}")
    start_time = time.time()
    
    # Connect to DuckDB (in-memory for processing)
    conn = duckdb.connect()
    
    try:
        # Read compressed JSON lines using DuckDB's built-in support
        logger.info("📖 Reading compressed JSONL data...")
        
        # Use DuckDB's read_json_auto with compression detection
        conn.execute(f"""
            CREATE TABLE raw_data AS 
            SELECT * FROM read_json_auto('{rec_file}', 
                                       compression='gzip',
                                       format='newline_delimited',
                                       maximum_object_size=1048576)
        """)
        
        # Check record count
        record_count = conn.execute("SELECT COUNT(*) FROM raw_data").fetchone()[0]
        logger.info(f"📊 Loaded {record_count:,} records")
        
        if record_count == 0:
            logger.warning("⚠️  No records found in file")
            return output_file
        
        # Inspect schema to understand the data structure
        schema_info = conn.execute("DESCRIBE raw_data").fetchall()
        columns = [row[0] for row in schema_info]
        logger.info(f"📋 Detected columns: {columns}")
        
        # Create optimized table with proper data types
        logger.info("🔧 Optimizing data types...")
        
        # Build CREATE TABLE statement with optimized types
        optimized_columns = []
        for col in columns:
            if col in ['timestamp', 'event_time', 'ts']:
                optimized_columns.append(f"{col} DOUBLE")  # Unix timestamp
            elif col in ['price', 'volume', 'quantity', 'value']:
                optimized_columns.append(f"{col} DOUBLE")  # Numeric values
            elif col in ['side', 'symbol', 'indicator_name', 'signal_type']:
                optimized_columns.append(f"{col} VARCHAR")  # String values
            elif col in ['is_buyer_maker', 'is_signal']:
                optimized_columns.append(f"{col} BOOLEAN")  # Boolean values
            else:
                optimized_columns.append(f"{col} VARCHAR")  # Default to string
        
        create_sql = f"""
            CREATE TABLE optimized_data AS
            SELECT 
                {', '.join(f'CAST({col.split()[0]} AS {col.split()[1]}) AS {col.split()[0]}' for col in optimized_columns)}
            FROM raw_data
        """
        
        logger.debug(f"Optimization SQL: {create_sql}")
        conn.execute(create_sql)
        
        # Add derived columns for analysis
        logger.info("📈 Adding derived columns...")
        
        derived_columns = []
        
        # Add timestamp-based columns if timestamp exists
        if 'timestamp' in columns:
            derived_columns.extend([
                "DATE_TRUNC('minute', TO_TIMESTAMP(timestamp)) AS minute_bucket",
                "DATE_TRUNC('hour', TO_TIMESTAMP(timestamp)) AS hour_bucket", 
                "DATE_TRUNC('day', TO_TIMESTAMP(timestamp)) AS day_bucket",
                "EXTRACT('hour' FROM TO_TIMESTAMP(timestamp)) AS hour_of_day",
                "EXTRACT('dow' FROM TO_TIMESTAMP(timestamp)) AS day_of_week"
            ])
        
        # Add price-based columns if price exists
        if 'price' in columns:
            derived_columns.extend([
                "ROUND(price, 2) AS price_rounded",
                "FLOOR(price / 1000) * 1000 AS price_k_bucket"
            ])
        
        # Add volume-based columns if volume exists
        if 'volume' in columns:
            derived_columns.extend([
                "CASE WHEN volume > 1.0 THEN 'large' WHEN volume > 0.1 THEN 'medium' ELSE 'small' END AS volume_category",
                "ROUND(volume, 4) AS volume_rounded"
            ])
        
        if derived_columns:
            alter_sql = f"""
                CREATE TABLE final_data AS
                SELECT *,
                       {', '.join(derived_columns)}
                FROM optimized_data
            """
            conn.execute(alter_sql)
            table_name = "final_data"
        else:
            table_name = "optimized_data"
        
        # Export to Parquet with compression and optimization
        logger.info("💾 Writing Parquet file...")
        
        conn.execute(f"""
            COPY {table_name} TO '{output_file}' 
            (FORMAT PARQUET, 
             COMPRESSION 'SNAPPY',
             ROW_GROUP_SIZE 50000)
        """)
        
        # Get final statistics
        final_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        final_columns = len(conn.execute(f"DESCRIBE {table_name}").fetchall())
        
        elapsed = time.time() - start_time
        input_size = rec_file.stat().st_size / (1024*1024)  # MB
        output_size = output_file.stat().st_size / (1024*1024)  # MB
        
        logger.success(f"✅ Conversion completed in {elapsed:.1f}s")
        logger.info(f"📊 Records: {final_count:,}")
        logger.info(f"📋 Columns: {final_columns}")
        logger.info(f"📏 Size: {input_size:.1f}MB → {output_size:.1f}MB ({output_size/input_size:.1f}x)")
        logger.info(f"⚡ Speed: {final_count/elapsed:,.0f} records/sec")
        
        return output_file
        
    except Exception as e:
        logger.error(f"❌ Conversion failed: {e}")
        raise
    finally:
        conn.close()

def batch_convert_directory(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    pattern: str = "*.rec.gz"
) -> List[Path]:
    """
    Batch convert all recording files in a directory
    
    Args:
        input_dir: Directory containing .rec.gz files
        output_dir: Output directory for .parquet files (defaults to input_dir)
        pattern: File pattern to match
        
    Returns:
        List of created Parquet files
    """
    
    if output_dir is None:
        output_dir = input_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all matching files
    rec_files = list(input_dir.glob(pattern))
    
    if not rec_files:
        logger.warning(f"⚠️  No files matching {pattern} in {input_dir}")
        return []
    
    logger.info(f"🔄 Batch converting {len(rec_files)} files...")
    
    converted_files = []
    total_start = time.time()
    
    for i, rec_file in enumerate(rec_files, 1):
        try:
            output_file = output_dir / rec_file.with_suffix('.parquet').name
            logger.info(f"[{i}/{len(rec_files)}] Processing {rec_file.name}")
            
            result = convert_rec_to_parquet(rec_file, output_file)
            converted_files.append(result)
            
        except Exception as e:
            logger.error(f"❌ Failed to convert {rec_file}: {e}")
            continue
    
    total_elapsed = time.time() - total_start
    success_count = len(converted_files)
    
    logger.success(f"🎉 Batch conversion completed!")
    logger.info(f"✅ Successfully converted: {success_count}/{len(rec_files)}")
    logger.info(f"⏱️  Total time: {total_elapsed:.1f}s")
    
    return converted_files

def analyze_parquet_file(parquet_file: Path) -> Dict[str, Any]:
    """
    Analyze converted Parquet file and return summary statistics
    
    Args:
        parquet_file: Path to Parquet file
        
    Returns:
        Dictionary of analysis results
    """
    
    logger.info(f"📊 Analyzing {parquet_file}")
    
    conn = duckdb.connect()
    
    try:
        # Load Parquet file
        conn.execute(f"CREATE TABLE data AS SELECT * FROM '{parquet_file}'")
        
        # Basic statistics
        total_records = conn.execute("SELECT COUNT(*) FROM data").fetchone()[0]
        
        # Column information
        columns_info = conn.execute("DESCRIBE data").fetchall()
        columns = [row[0] for row in columns_info]
        
        analysis = {
            'file': str(parquet_file),
            'total_records': total_records,
            'columns': columns,
            'column_count': len(columns)
        }
        
        # Time range analysis if timestamp exists
        if 'timestamp' in columns:
            time_stats = conn.execute("""
                SELECT 
                    MIN(timestamp) as min_ts,
                    MAX(timestamp) as max_ts,
                    MAX(timestamp) - MIN(timestamp) as duration_seconds
                FROM data
            """).fetchone()
            
            analysis['time_range'] = {
                'start': time_stats[0],
                'end': time_stats[1], 
                'duration_hours': time_stats[2] / 3600 if time_stats[2] else 0
            }
        
        # Record type distribution
        if 'type' in columns:
            type_dist = conn.execute("""
                SELECT type, COUNT(*) as count
                FROM data 
                GROUP BY type
                ORDER BY count DESC
            """).fetchall()
            
            analysis['record_types'] = {row[0]: row[1] for row in type_dist}
        
        # Symbol distribution
        if 'symbol' in columns:
            symbol_dist = conn.execute("""
                SELECT symbol, COUNT(*) as count
                FROM data
                GROUP BY symbol
                ORDER BY count DESC
                LIMIT 10
            """).fetchall()
            
            analysis['symbols'] = {row[0]: row[1] for row in symbol_dist}
        
        logger.info(f"📈 Analysis complete: {total_records:,} records, {len(columns)} columns")
        return analysis
        
    finally:
        conn.close()

def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert RTAI recording files to Parquet")
    parser.add_argument("input", help="Input file or directory")
    parser.add_argument("-o", "--output", help="Output file or directory")
    parser.add_argument("-a", "--analyze", action="store_true", help="Analyze output file")
    parser.add_argument("-b", "--batch", action="store_true", help="Batch convert directory")
    parser.add_argument("--pattern", default="*.rec.gz", help="File pattern for batch mode")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        logger.error(f"❌ Input path not found: {input_path}")
        sys.exit(1)
    
    try:
        if args.batch or input_path.is_dir():
            # Batch mode
            output_dir = Path(args.output) if args.output else input_path
            converted_files = batch_convert_directory(input_path, output_dir, args.pattern)
            
            if args.analyze and converted_files:
                logger.info("🔍 Analyzing converted files...")
                for pfile in converted_files[:3]:  # Analyze first 3 files
                    analysis = analyze_parquet_file(pfile)
                    print(json.dumps(analysis, indent=2))
        else:
            # Single file mode
            output_file = Path(args.output) if args.output else None
            result = convert_rec_to_parquet(input_path, output_file)
            
            if args.analyze:
                analysis = analyze_parquet_file(result)
                print(json.dumps(analysis, indent=2))
    
    except Exception as e:
        logger.error(f"❌ Operation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
