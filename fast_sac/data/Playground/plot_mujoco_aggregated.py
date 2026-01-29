import json
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
import os

def load_dsac_data(csv_path, max_step=50000):
    steps = []
    tasks = {} # task_name -> list of lists (seeds)
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            
            # Map indices to task names
            # Col 0 is Step
            # Others: "{task}__{exp}__{seed} - eval_avg_return"
            
            task_col_indices = {} # index -> task_name
            
            step_idx = 0
            for i, col in enumerate(header):
                if i == 0: 
                    if 'Step' not in col: print("Warning: First col is not Step?")
                    continue
                
                if 'eval_avg_return' in col and 'MIN' not in col and 'MAX' not in col:
                    parts = col.split('__')
                    if len(parts) > 1:
                        task_name = parts[0].strip('"')
                        task_col_indices[i] = task_name
                        if task_name not in tasks:
                            tasks[task_name] = [] 

            # Initialize lists for each task (we need to know how many seeds? 
            # Or just accumulate and reshape later. 
            # Easier: tasks[task_name] is a dict of seed -> list of vals?
            # Or just a list of lists, assuming order holds.
            # Let's use a dict of lists for now: task -> seed -> list
            
            # Actually, we need to handle row by row
            # Step is index 0
            
            # We need to buffer data then convert to array
            # task_data_buffer: task -> { seed_identifier -> [values] }
            # To simplify, we rely on cols being distinct seeds.
            # We can map ColIndex -> (Task, Seed_ID)
            
            col_map = {} # i -> (task, seed)
            
            # Re-parse header specifically for seeds
            for i, col in enumerate(header):
                if 'eval_avg_return' in col and 'MIN' not in col and 'MAX' not in col:
                     parts = col.split('__')
                     if len(parts) > 1:
                         t_name = parts[0].strip('"')
                         s_name = parts[-1].split()[0] # get seed part? e.g. "888 - eval..." -> "888"
                         col_map[i] = (t_name, s_name)
            
            temp_data = {} # (task, seed) -> list of vals
            
            for row in reader:
                if not row: continue
                try:
                    step = float(row[0])
                    if step > max_step: continue
                    
                    steps.append(step)
                    
                    for i, val_str in enumerate(row):
                        if i in col_map:
                            t, s = col_map[i]
                            key = (t, s)
                            if key not in temp_data: temp_data[key] = []
                            temp_data[key].append(float(val_str))
                            
                except ValueError:
                    continue
            
            # Reconstruct into expected format: steps, tasks = {task: [array_seed1, array_seed2...]}
            for (t, s), vals in temp_data.items():
                if t not in tasks: tasks[t] = []
                tasks[t].append(np.array(vals))
                
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None, None

    return np.array(steps), tasks

def normalize(values, min_val, max_val):
    if max_val == min_val: return values * 0
    return 100 * (values - min_val) / (max_val - min_val)

def main():
    csv_path = 'fast_sac/data/Playground/mujoco_FastDSAC.csv'
    json_path = 'fast_sac/data/Playground/playground_result.json'
    
    # Load FastDSAC (limit 50k)
    dsac_steps, dsac_task_data = load_dsac_data(csv_path, max_step=50000)
    
    # Load FastTD3
    with open(json_path, 'r') as f:
        json_data = json.load(f)
        
    dsac_norm_curves = []
    td3_norm_curves = []
    
    # Process each task
    for task_name, seed_returns in dsac_task_data.items():
        # Clean task name if needed (CSV matches JSON keys based on inspection)
        json_key = task_name 
        
        if json_key not in json_data:
            print(f"Key {json_key} not in JSON")
            continue
            
        td3_raw = json_data[json_key]['FastTD3']['return']
        # Assume FastTD3 has 10 points corresponding to 0..50k roughly?
        # FastDSAC has steps: 5000, 10000... 50000 (10 points)
        # FastTD3 usually includes 0. Let's check length.
        
        td3_vals = np.array(td3_raw)
        
        # Calculate Baseline Min/Max for normalization
        base_min = np.min(td3_vals)
        base_max = np.max(td3_vals)
        
        # Normalize FastDSAC
        # Average seeds first? Or normalize each seed then average? 
        # Usually average seeds first to get "Method Performance on Task"
        dsac_mean_curve = np.mean(np.array(seed_returns), axis=0)
        dsac_norm = normalize(dsac_mean_curve, base_min, base_max)
        dsac_norm_curves.append(dsac_norm)
        
        # Normalize FastTD3
        # FastTD3 steps might be slightly different count?
        # DSAC has 10 points (5k to 50k).
        # TD3 has 10 points. If TD3 corresponds to same grid, we assume 1-1 mapping.
        # Check lengths
        if len(td3_vals) != len(dsac_norm):
            # Align them. 
            # If TD3 has 0, maybe slice?
            # User said "FastTD3都跑了5w步".
            # If TD3 array is longer, we might need interpolation.
            # But earlier checking showed 10 points.
            if len(td3_vals) > len(dsac_norm):
                td3_vals = td3_vals[:len(dsac_norm)] # Truncate if needed
            elif len(td3_vals) < len(dsac_norm):
                # Interpolate DSAC to TD3 or vice versa?
                # Let's align to DSAC steps
                pass
        
        td3_norm = normalize(td3_vals, base_min, base_max)
        td3_norm_curves.append(td3_norm)

    # Aggregate
    if not dsac_norm_curves:
        print("No matched tasks found")
        return

    # Stack
    dsac_stack = np.vstack(dsac_norm_curves)
    td3_stack = np.vstack(td3_norm_curves)
    
    # Compute stats across TASKS
    dsac_mean = np.mean(dsac_stack, axis=0)
    dsac_stderr = np.std(dsac_stack, axis=0) / np.sqrt(dsac_stack.shape[0])
    
    td3_mean = np.mean(td3_stack, axis=0)
    td3_stderr = np.std(td3_stack, axis=0) / np.sqrt(td3_stack.shape[0])
    
    # X-Axis
    steps = dsac_steps
    # Ensure TD3 uses same steps for plotting if counts match
    
    # Plotting
    sns.set(style="whitegrid")
    plt.figure(figsize=(6, 4))
    
    # FastTD3 (Black)
    plt.plot(steps, td3_mean, label='FastTD3', color='black', linewidth=3)
    plt.fill_between(steps, td3_mean - td3_stderr, td3_mean + td3_stderr, color='black', alpha=0.2)
    
    # FastDSAC (Blue)
    plt.plot(steps, dsac_mean, label='FastDSAC', color='#007FFF', linewidth=3) # Azure-ish blue
    plt.fill_between(steps, dsac_mean - dsac_stderr, dsac_mean + dsac_stderr, color='#007FFF', alpha=0.2)
    
    plt.title('MuJoCo Playground (4 Tasks)', fontsize=14)
    plt.xlabel('Environment Steps', fontsize=12)
    plt.ylabel('Average Normalized Return', fontsize=12)
    plt.ylim(bottom=0)
    plt.xlim(0, 50000)
    
    # Formatter
    def format_func(x, pos):
        if x >= 1e3: return f'{x*1e-3:.0f}k'
        return f'{x:.0f}'
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
    
    plt.legend(loc='upper left') # Or wherever fits
    plt.tight_layout()
    
    plt.savefig('mujoco_aggregated.png', dpi=300)
    print("Saved mujoco_aggregated.png")

if __name__ == "__main__":
    main()
