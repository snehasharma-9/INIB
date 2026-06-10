"""
main.py - Main pipeline to run the complete fake news detection project
"""
import subprocess
import sys
import os
import time

def run_script(script_path, description, cwd=None):
    """Run a Python script and handle errors"""
    print("\n" + "="*80)
    print(f"STEP: {description}")
    print("="*80)
    print(f"Running: {script_path}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, '-B', script_path],
            capture_output=False,
            text=True,
            cwd=cwd,
            timeout=3600
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"\nOK {description} completed successfully in {elapsed:.1f}s")
            return True
        else:
            print(f"\nERROR {description} failed! Return code: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"\nERROR {description} timed out after 3600 seconds!")
        return False
    except Exception as e:
        print(f"\nERROR {description} error: {str(e)}")
        return False

def main():
    """Execute the complete pipeline"""
    
    print("\n" + "="*80)
    print("FAKE NEWS DETECTION - COMPLETE PIPELINE")
    print("="*80)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Project root: {base_dir}")
    
    scripts = [
        ('01_explore_data.py', 'Data Exploration'),
        ('02_preprocess_data.py', 'Data Preprocessing'),
        ('03_feature_engineering.py', 'Feature Engineering'),
        ('04_train_models.py', 'Model Training'),
        ('05_evaluate_models.py', 'Model Evaluation'),
    ]
    
    completed = 0
    
    for script, description in scripts:
        script_path = os.path.join(base_dir, script)
        if os.path.exists(script_path):
            if run_script(script_path, description, cwd=base_dir):
                completed += 1
            else:
                print(f"\nStopping pipeline due to error in {description}")
                break
        else:
            print(f"\nERROR {script_path} not found!")
    
    print("\n" + "="*80)
    print("PIPELINE SUMMARY")
    print("="*80)
    print(f"Completed: {completed}/{len(scripts)} steps")
    
    if completed == len(scripts):
        print("\nOK All steps completed successfully!")
        print("\nYou can now:")
        print("  - View results in CSV files (model_comparison.csv, detailed_metrics.csv)")
        print("  - View plots (model_comparison.png, confusion_matrix_*.png)")
        print("  - Make predictions using 06_predict.py")
    else:
        print("\nERROR Pipeline incomplete due to errors")
    
    print("="*80)

if __name__ == "__main__":
    main()
