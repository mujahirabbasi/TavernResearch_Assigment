import pandas as pd
import numpy as np

def calculate_treatment_effects():
    """
    Calculate treatment effects for each video based on RCT data.
    
    Returns:
        DataFrame with video_id, control_approval, video_approval, and treatment_effect
    """
    
    # Load the RCT data
    print("Loading RCT data...")
    df = pd.read_csv('rct_dummy_data.csv')
    
    # Display basic info about the data
    print(f"Total observations: {len(df)}")
    print(f"Control group (treated=0): {len(df[df['treated'] == 0])}")
    print(f"Treatment group (treated=1): {len(df[df['treated'] == 1])}")
    print(f"Unique video IDs: {df['video_id'].nunique()}")
    
    # 1. Calculate baseline control approval rate
    print("\n1. Calculating baseline control approval rate...")
    control_group = df[df['treated'] == 0]
    control_approval_rate = control_group['trump_approval'].mean()
    control_approval_count = control_group['trump_approval'].sum()
    control_total = len(control_group)
    
    print(f"Control group approval rate: {control_approval_rate:.4f} ({control_approval_count}/{control_total})")
    
    # 2. Calculate approval rate for each video
    print("\n2. Calculating approval rates for each video...")
    
    # Get unique video IDs (excluding NA)
    video_ids = df[df['video_id'].notna()]['video_id'].unique()
    video_ids = sorted(video_ids)
    
    print(f"Found {len(video_ids)} unique video IDs: {video_ids[:10]}...")
    
    results = []
    
    for video_id in video_ids:
        # Get data for this specific video (treated=1 and specific video_id)
        video_data = df[(df['treated'] == 1) & (df['video_id'] == video_id)]
        
        if len(video_data) > 0:
            video_approval_rate = video_data['trump_approval'].mean()
            video_approval_count = video_data['trump_approval'].sum()
            video_total = len(video_data)
            
            # 3. Calculate treatment effect
            treatment_effect = video_approval_rate - control_approval_rate
            
            results.append({
                'video_id': int(video_id),
                'control_approval': round(control_approval_rate, 4),
                'video_approval': round(video_approval_rate, 4),
                'treatment_effect': round(treatment_effect, 4),
                'video_approval_count': video_approval_count,
                'video_total': video_total
            })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by video_id
    results_df = results_df.sort_values('video_id')
    
    # 4. Display summary statistics
    print(f"\n3. Summary of treatment effects:")
    print(f"Total videos analyzed: {len(results_df)}")
    print(f"Average treatment effect: {results_df['treatment_effect'].mean():.4f}")
    print(f"Treatment effect range: {results_df['treatment_effect'].min():.4f} to {results_df['treatment_effect'].max():.4f}")
    
    # Count positive and negative effects
    positive_effects = len(results_df[results_df['treatment_effect'] > 0])
    negative_effects = len(results_df[results_df['treatment_effect'] < 0])
    no_effects = len(results_df[results_df['treatment_effect'] == 0])
    
    print(f"Positive effects: {positive_effects}")
    print(f"Negative effects: {negative_effects}")
    print(f"No effect: {no_effects}")
    
    return results_df

def display_top_effects(results_df, n=10):
    """Display top positive and negative treatment effects"""
    
    print(f"\nTop {n} positive treatment effects:")
    top_positive = results_df.nlargest(n, 'treatment_effect')[['video_id', 'control_approval', 'video_approval', 'treatment_effect']]
    print(top_positive.to_string(index=False))
    
    print(f"\nTop {n} negative treatment effects:")
    top_negative = results_df.nsmallest(n, 'treatment_effect')[['video_id', 'control_approval', 'video_approval', 'treatment_effect']]
    print(top_negative.to_string(index=False))

def save_results(results_df, filename='treatment_effects_results.csv'):
    """Save results to CSV file"""
    # Select only the main columns for output
    output_df = results_df[['video_id', 'control_approval', 'video_approval', 'treatment_effect']]
    output_df.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")

if __name__ == "__main__":
    print("RCT Treatment Effect Analysis")
    print("=" * 40)
    
    # Calculate treatment effects
    results = calculate_treatment_effects()
    
    # Display top effects
    display_top_effects(results)
    
    # Save results
    save_results(results)
    
    print("\nAnalysis complete!")
