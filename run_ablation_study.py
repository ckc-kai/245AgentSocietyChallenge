"""
Script to run all ablation variants for RecAgent2 (recommendation task)

This script will run the following variants:
1. M2-Full: Complete agent with all components
2. M2-No-Memory: Remove EfficientMemory module
3. M2-No-Profiling: Remove statistical profiling
4. M2-No-Consistency: Replace self-consistency with simple COT

Each variant will be run on the specified number of tasks and results will be saved.
"""

import subprocess
import os
import json
import sys

# Configuration
# Note: "full" variant uses existing Agent2 results, no need to rerun
VARIANTS = ["no_memory", "no_profiling", "no_consistency"]
NUM_TASKS = 400  # Change to 10 for quick testing
TASK_SET = "amazon"

def run_ablation_variant(variant):
    """Run a single ablation variant"""
    print(f"\n{'='*60}")
    print(f"Running ablation variant: {variant}")
    print(f"{'='*60}\n")
    
    # Read the ablation file
    ablation_file = "example/RecAgent2_ablation.py"
    with open(ablation_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace the ABLATION_VARIANT setting
    content_modified = content.replace(
        'ABLATION_VARIANT = "full"',
        f'ABLATION_VARIANT = "{variant}"'
    )
    
    # Also replace number_of_tasks
    content_modified = content_modified.replace(
        'number_of_tasks=10',
        f'number_of_tasks={NUM_TASKS}'
    )
    
    # Write temporary file
    temp_file = f"example/RecAgent2_ablation_temp.py"
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(content_modified)
    
    try:
        # Run the script with real-time output
        print(f"Progress will be shown below:\n")
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=False,  # Show output in real-time
            text=True,
            timeout=None
        )
        
        print()  # Add newline after output
        
        if result.returncode != 0:
            print(f"\n✗ ERROR: Variant {variant} failed with return code {result.returncode}")
            return False
        
        # Check if results file was created
        results_file = f'./results/evaluation/evaluation_results_track2_{TASK_SET}_agent2_{variant}.json'
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            print(f"\n✓ Variant {variant} completed successfully")
            print(f"  Results saved to: {results_file}")
            return True
        else:
            print(f"✗ Results file not found: {results_file}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"ERROR: Variant {variant} timed out after 2 hours")
        return False
    except Exception as e:
        print(f"ERROR running variant {variant}: {e}")
        return False
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)

def summarize_results():
    """Summarize all ablation results"""
    print(f"\n{'='*60}")
    print("ABLATION STUDY SUMMARY")
    print(f"{'='*60}\n")
    
    results_summary = []
    
    # Add the "full" variant from existing Agent2 results
    full_results_file = f'./results/evaluation/evaluation_results_track2_{TASK_SET}_agent2.json'
    if os.path.exists(full_results_file):
        with open(full_results_file, 'r') as f:
            content = f.read()
            # Handle old format with two separate JSON objects
            if content.count('{') > 1 and '"time"' in content:
                # Skip first JSON object (time info)
                parts = content.split('}\n{')
                if len(parts) == 2:
                    results = json.loads('{' + parts[1])
                else:
                    results = json.loads(content)
            else:
                results = json.loads(content)
        
        if 'metrics' in results:
            metrics = results['metrics']
            results_summary.append({
                'variant': 'full',
                'hr_1': metrics.get('top_1_hit_rate', 0),
                'hr_3': metrics.get('top_3_hit_rate', 0),
                'hr_5': metrics.get('top_5_hit_rate', 0),
                'avg_hr': metrics.get('average_hit_rate', 0)
            })
            print("✓ Using existing Agent2 results for 'full' variant")
    else:
        print("Warning: Agent2 results file not found for 'full' variant")
    
    for variant in VARIANTS:
        results_file = f'./results/evaluation/evaluation_results_track2_{TASK_SET}_agent2_{variant}.json'
        
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                # Ablation files have combined JSON (time + results)
                results = json.load(f)
            
            if 'metrics' in results:
                metrics = results['metrics']
                results_summary.append({
                    'variant': variant,
                    'hr_1': metrics.get('top_1_hit_rate', 0),
                    'hr_3': metrics.get('top_3_hit_rate', 0),
                    'hr_5': metrics.get('top_5_hit_rate', 0),
                    'avg_hr': metrics.get('average_hit_rate', 0)
                })
        else:
            print(f"Warning: Results file not found for variant: {variant}")
    
    if results_summary:
        print(f"{'Variant':<20} {'HR@1':<10} {'HR@3':<10} {'HR@5':<10} {'Avg HR':<10}")
        print("-" * 60)
        for r in results_summary:
            print(f"{r['variant']:<20} {r['hr_1']:<10.4f} {r['hr_3']:<10.4f} {r['hr_5']:<10.4f} {r['avg_hr']:<10.4f}")
        
        # Generate LaTeX table
        print("\n" + "="*60)
        print("LaTeX Table Format:")
        print("="*60 + "\n")
        print("\\begin{tabular}{lcccc}")
        print("\\toprule")
        print("Variant & HR@1 & HR@3 & HR@5 & Average HR \\\\")
        print("\\midrule")
        for r in results_summary:
            variant_name = r['variant'].replace('_', ' ').title()
            print(f"{variant_name} & {r['hr_1']:.4f} & {r['hr_3']:.4f} & {r['hr_5']:.4f} & {r['avg_hr']:.4f} \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")
    else:
        print("No results found!")

if __name__ == "__main__":
    print(f"Starting ablation study for RecAgent2 (Recommendation Task)")
    print(f"Task Set: {TASK_SET}")
    print(f"Number of Tasks: {NUM_TASKS}")
    print(f"Variants to run: {', '.join(VARIANTS)}")
    print(f"Note: 'full' variant will use existing Agent2 results")
    
    # Create results directory if it doesn't exist
    os.makedirs('./results/evaluation', exist_ok=True)
    
    # Run each variant
    success_count = 0
    for variant in VARIANTS:
        if run_ablation_variant(variant):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Completed {success_count}/{len(VARIANTS)} ablation variants successfully")
    print(f"(Plus 'full' variant from existing Agent2 results)")
    print(f"{'='*60}\n")
    
    # Summarize all results
    summarize_results()

