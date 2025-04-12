@echo off
REM Create pretty visualizations from benchmark results

REM Set paths
set RESULTS_DIR=fixed_benchmark_results
set OUTPUT_DIR=benchmark_visualizations

REM Create the output directory
mkdir %OUTPUT_DIR% 2>nul

REM Run the visualization script
echo Creating pretty visualizations from benchmark results...
python visualize_benchmark_results.py --results_dir %RESULTS_DIR% --output_dir %OUTPUT_DIR% --dpi 300

echo Visualizations complete. Results are saved in %OUTPUT_DIR%
pause