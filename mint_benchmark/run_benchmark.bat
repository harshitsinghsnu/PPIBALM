@echo off
REM run_benchmark_updated.bat

REM Set correct file paths based on directory structure
set DATA_PATH=D:\BALM_Fineclone\BALM-PPI\scripts\notebooks\Data.csv
set BALM_CONFIG=D:\BALM_Fineclone\BALM-PPI\default_configs\balm_peft.yaml
set BALM_CHECKPOINT=D:\BALM_Fineclone\BALM-PPI\outputs\latest_checkpoint.pth
set MINT_CONFIG=D:\BALM_Fineclone\BALM-PPI\Benchmark\scripts\mint\data\esm2_t33_650M_UR50D.json  
set MINT_CHECKPOINT=D:\BALM_Fineclone\BALM-PPI\Benchmark\models\mint.ckpt
set OUTPUT_DIR=D:\BALM_Fineclone\mint_benchmark\mint_benchmark_results

REM Make sure directory exists for data
mkdir %OUTPUT_DIR%\data

REM Add paths to PYTHONPATH
set PYTHONPATH=%PYTHONPATH%;D:\BALM_Fineclone\BALM-PPI;D:\BALM_Fineclone\BALM-PPI\Benchmark\scripts\mint

REM Convert data format
python convert_balm_data.py --input "%DATA_PATH%" --output "%OUTPUT_DIR%\data\train.csv"

REM Run full benchmark with both models
python benchmark_mint_vs_balm.py --data_path "%OUTPUT_DIR%\data\train.csv" --balm_config "%BALM_CONFIG%" --balm_checkpoint "%BALM_CHECKPOINT%" --mint_config "%MINT_CONFIG%" --mint_checkpoint "%MINT_CHECKPOINT%" --output_dir "%OUTPUT_DIR%" --seed 42

echo Benchmark completed. Results are in %OUTPUT_DIR%