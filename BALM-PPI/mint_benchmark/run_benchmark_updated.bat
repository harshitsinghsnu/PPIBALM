@echo off
REM Benchmark MINT against BALM-PPI

REM Set paths with corrected MINT CONFIG path
set DATA_PATH=D:\BALM_Fineclone\BALM-PPI\scripts\notebooks\Data.csv
set BALM_CONFIG=D:\BALM_Fineclone\BALM-PPI\default_configs\balm_peft.yaml
set MINT_CHECKPOINT=D:\BALM_Fineclone\BALM-PPI\Benchmark\models\mint.ckpt
set MINT_CONFIG=D:\BALM_Fineclone\BALM-PPI\Benchmark\scripts\mint\data\esm2_t33_650M_UR50D.json
set OUTPUT_DIR=mint_benchmark_results

REM Create the output directory
mkdir %OUTPUT_DIR% 2>nul

REM Step 1: Convert data to MINT format
echo Converting BALM data to MINT format...
python convert_balm_data.py --input_file %DATA_PATH% --output_dir %OUTPUT_DIR%\data --split_ratio 0.2

REM Step 2: Run the benchmark with the updated script
echo Running MINT benchmark...
python benchmark_mint_vs_balm_updated.py --data_path %DATA_PATH% --balm_config %BALM_CONFIG% --mint_checkpoint %MINT_CHECKPOINT% --mint_config %MINT_CONFIG% --output_dir %OUTPUT_DIR% --batch_size 4 --max_seq_len 1024 --data_dir %OUTPUT_DIR%\data

echo Benchmark complete. Results are saved in %OUTPUT_DIR%