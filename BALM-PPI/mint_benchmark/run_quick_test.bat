@echo off
REM Quick test run of extended benchmarking for BALM-PPI and MINT

REM Set paths
set DATA_PATH=D:\BALM_Fineclone\BALM-PPI\scripts\notebooks\Data.csv
set BALM_CHECKPOINT=D:\BALM_Fineclone\BALM-PPI\outputs\latest_checkpoint.pth
set BALM_CONFIG=D:\BALM_Fineclone\BALM-PPI\default_configs\balm_peft.yaml
set MINT_CHECKPOINT=D:\BALM_Fineclone\BALM-PPI\Benchmark\models\mint.ckpt
set MINT_CONFIG=D:\BALM_Fineclone\BALM-PPI\Benchmark\scripts\mint\data\esm2_t33_650M_UR50D.json
set OUTPUT_DIR=quick_test_results

REM Create the output directory
mkdir %OUTPUT_DIR% 2>nul

REM Run a single test with random split and 1 run for speed
echo Running quick benchmark test...
python extended_benchmark.py --data_path %DATA_PATH% --balm_checkpoint %BALM_CHECKPOINT% --balm_config %BALM_CONFIG% --mint_checkpoint %MINT_CHECKPOINT% --mint_config %MINT_CONFIG% --output_dir %OUTPUT_DIR% --batch_size 2 --n_runs 1 --split_type random --test_ratio 0.1

echo Quick test complete. Results are saved in %OUTPUT_DIR%
pause