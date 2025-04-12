@echo off
REM Comprehensive benchmark for protein-protein binding affinity prediction models

REM Set paths
set DATA_PATH=D:\BALM_Fineclone\BALM-PPI\scripts\notebooks\Data.csv
set BALM_CHECKPOINT=D:\BALM_Fineclone\BALM-PPI\outputs\latest_checkpoint.pth
set BALM_CONFIG=D:\BALM_Fineclone\BALM-PPI\default_configs\balm_peft.yaml
set MINT_CHECKPOINT=D:\BALM_Fineclone\BALM-PPI\Benchmark\models\mint.ckpt
set MINT_CONFIG=D:\BALM_Fineclone\BALM-PPI\Benchmark\scripts\mint\data\esm2_t33_650M_UR50D.json
set OUTPUT_DIR=comprehensive_benchmark_results

REM Create the output directory
mkdir %OUTPUT_DIR% 2>nul

REM Run random split benchmark
echo Running random split benchmark (3 seeds)...
python comprehensive_benchmark.py --data_path %DATA_PATH% ^
    --balm_checkpoint %BALM_CHECKPOINT% ^
    --balm_config %BALM_CONFIG% ^
    --mint_checkpoint %MINT_CHECKPOINT% ^
    --mint_config %MINT_CONFIG% ^
    --output_dir %OUTPUT_DIR%\random_split ^
    --batch_size 2 ^
    --num_runs 3 ^
    --seeds 42,123,456 ^
    --split_type random ^
    --train_ratio 0.2 ^
    --generate_plots

REM Run cold target benchmark
echo Running cold target benchmark (3 seeds)...
python comprehensive_benchmark.py --data_path %DATA_PATH% ^
    --balm_checkpoint %BALM_CHECKPOINT% ^
    --balm_config %BALM_CONFIG% ^
    --mint_checkpoint %MINT_CHECKPOINT% ^
    --mint_config %MINT_CONFIG% ^
    --output_dir %OUTPUT_DIR%\cold_target_split ^
    --batch_size 2 ^
    --num_runs 3 ^
    --seeds 42,123,456 ^
    --split_type cold_target ^
    --train_ratio 0.2 ^
    --generate_plots

REM Run sequence similarity benchmark
echo Running sequence similarity benchmark (3 seeds)...
python comprehensive_benchmark.py --data_path %DATA_PATH% ^
    --balm_checkpoint %BALM_CHECKPOINT% ^
    --balm_config %BALM_CONFIG% ^
    --mint_checkpoint %MINT_CHECKPOINT% ^
    --mint_config %MINT_CONFIG% ^
    --output_dir %OUTPUT_DIR%\seq_similarity_split ^
    --batch_size 2 ^
    --num_runs 3 ^
    --seeds 42,123,456 ^
    --split_type seq_similarity ^
    --train_ratio 0.2 ^
    --generate_plots

echo Comprehensive benchmark complete. Results are saved in %OUTPUT_DIR%
pause