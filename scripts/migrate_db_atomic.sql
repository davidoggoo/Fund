/* RTAI Atomic Indicators Database Migration
 * Extends live.features_1m with atomic indicators and Z-band columns
 * Run once: duckdb data/db/rtai.duckdb < scripts/migrate_db_atomic.sql
 */

-- Extend live.features_1m with atomic + band columns
ALTER TABLE live.features_1m
  ADD COLUMN liq_usd          DOUBLE DEFAULT 0,
  ADD COLUMN liq_cnt          INTEGER DEFAULT 0,
  ADD COLUMN wall_ratio       DOUBLE DEFAULT 0,
  ADD COLUMN basis            DOUBLE DEFAULT 0,
  ADD COLUMN tobi             DOUBLE DEFAULT 0,
  ADD COLUMN trade_imb        DOUBLE DEFAULT 0,
  ADD COLUMN liq_z            DOUBLE,
  ADD COLUMN wall_ratio_z     DOUBLE,
  ADD COLUMN basis_z          DOUBLE,
  ADD COLUMN trade_imb_z      DOUBLE;

-- Create state persistence table for Z-band buffers
CREATE TABLE IF NOT EXISTS live.z_state (
    indicator_name VARCHAR,
    timestamp TIMESTAMP,
    value DOUBLE,
    PRIMARY KEY (indicator_name, timestamp)
);

-- Create index for efficient state loading
CREATE INDEX IF NOT EXISTS idx_z_state_indicator_time 
ON live.z_state (indicator_name, timestamp DESC);

-- Verification query
SELECT column_name, data_type, is_nullable, column_default 
FROM information_schema.columns 
WHERE table_name = 'features_1m' 
  AND table_schema = 'live'
  AND column_name IN ('liq_usd', 'liq_cnt', 'wall_ratio', 'basis', 'tobi', 'trade_imb', 
                      'liq_z', 'wall_ratio_z', 'basis_z', 'trade_imb_z')
ORDER BY ordinal_position;
