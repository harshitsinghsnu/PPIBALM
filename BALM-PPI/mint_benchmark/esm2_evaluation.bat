@echo off
REM ESM2 Evaluation

REM Set paths
set TEST_DATA=D:\BALM_Fineclone\BALM-PPI\scripts\notebooks\Data.csv
set OUTPUT_DIR=esm2_comparison_results

REM Create the output directory
mkdir %OUTPUT_DIR% 2>nul

REM Run the evaluation
echo Running ESM2 evaluation...
python esm2_evaluation.py --test_data %TEST_DATA% --output_dir %OUTPUT_DIR% --batch_size 2

echo Evaluation complete. Results are saved in %OUTPUT_DIR%
pause