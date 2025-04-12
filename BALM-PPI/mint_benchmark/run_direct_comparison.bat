@echo off
REM Direct comparison between BALM-PPI and MINT

REM Set paths
set TEST_DATA=D:\BALM_Fineclone\BALM-PPI\scripts\notebooks\Data.csv
set BALM_CHECKPOINT=D:\BALM_Fineclone\BALM-PPI\outputs\latest_checkpoint.pth
set BALM_CONFIG=D:\BALM_Fineclone\BALM-PPI\default_configs\balm_peft.yaml
set MINT_CHECKPOINT=D:\BALM_Fineclone\BALM-PPI\Benchmark\models\mint.ckpt
set MINT_CONFIG=D:\BALM_Fineclone\BALM-PPI\Benchmark\scripts\mint\data\esm2_t33_650M_UR50D.json
set OUTPUT_DIR=direct_comparison_results

REM Create the output directory
mkdir %OUTPUT_DIR% 2>nul

REM Run the comparison
echo Running direct comparison between BALM-PPI and MINT...
python direct_model_comparison.py --test_data %TEST_DATA% --balm_checkpoint %BALM_CHECKPOINT% --balm_config %BALM_CONFIG% --mint_checkpoint %MINT_CHECKPOINT% --mint_config %MINT_CONFIG% --output_dir %OUTPUT_DIR% --batch_size 2

echo Comparison complete. Results are saved in %OUTPUT_DIR%
pause
