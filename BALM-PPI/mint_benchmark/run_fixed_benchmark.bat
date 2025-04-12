@echo off
REM Fixed comprehensive benchmark for protein-protein binding affinity prediction models

REM Set paths
set DATA_PATH=D:\BALM_Fineclone\BALM-PPI\scripts\notebooks\Data.csv
set BALM_CHECKPOINT=D:\BALM_Fineclone\BALM-PPI\outputs\latest_checkpoint.pth
set BALM_CONFIG=D:\BALM_Fineclone\BALM-PPI\default_configs\balm_peft.yaml
set OUTPUT_DIR=fixed_benchmark_results

REM Create the output directory
mkdir %OUTPUT_DIR% 2>nul

REM Run all splits with a single command
echo Running comprehensive benchmark with all splits and 3 seeds...
python fixed_comprehensive_benchmark.py --data_path %DATA_PATH% ^
    --balm_checkpoint %BALM_CHECKPOINT% ^
    --balm_config %BALM_CONFIG% ^
    --output_dir %OUTPUT_DIR% ^
    --batch_size 2 ^
    --num_runs 3 ^
    --seeds 42,123,456 ^
    --run_all_splits ^
    --generate_plots

echo Comprehensive benchmark complete. Results are saved in %OUTPUT_DIR%
pause