# 📓 RCT + LLM Persuasion Analysis (using precomputed treatment effects)

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import requests
import json
import time

# Initialize Ollama client (make sure Ollama is running locally)
OLLAMA_BASE_URL = "http://localhost:11434"

# ============================================
# 1. Load precomputed treatment effects + transcripts
# ============================================

print("Loading precomputed treatment effects...")
effects = pd.read_csv("treatment_effects_results.csv")
print(f"Loaded {len(effects)} treatment effects")

print("Loading video transcripts...")
maxdiff = pd.read_csv("maxdiff_dummy_data.csv")
print(f"Loaded {len(maxdiff)} video transcripts")

# Merge to attach transcript text
df = pd.merge(effects, maxdiff[["video_id", "text"]], on="video_id")
print(f"Merged dataset: {df.shape}")
print("First few rows:")
print(df.head())

# ============================================
# 2. Extract features with LLM (BATCHED + CACHED)
# ============================================

def extract_features_batch_ollama(texts_batch, video_ids_batch, model="llama3.1:8b", retries=3):
    """Process multiple transcripts in a single Ollama request for speed."""
    
    # Create numbered list of messages
    messages_text = ""
    for i, (vid, text) in enumerate(zip(video_ids_batch, texts_batch)):
        messages_text += f"{i+1}. (video_id: {vid}) {text}\n"
    
    prompt = f"""
    Classify the following political campaign messages. 
    Return a JSON list, where each element has video_id, topic, tone, target, style.
    
    Messages:
    {messages_text}
    
    Return only valid JSON array, no other text. Each element should have:
    - video_id: the video ID number
    - topic: one of [economy, healthcare, immigration, energy, democracy, other]
    - tone: one of [positive, negative, neutral]
    - target: one of [Trump, Biden, Both, Neither]
    - style: one of [emotional, factual, personal_story, slogan]
    """

    for attempt in range(retries):
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9
                    }
                },
                timeout=60  # Longer timeout for batch processing
            )
            
            if response.status_code == 200:
                content = response.json()["response"]
                # Clean the response to extract just the JSON
                try:
                    start_idx = content.find('[')
                    end_idx = content.rfind(']') + 1
                    if start_idx != -1 and end_idx != -1:
                        json_str = content[start_idx:end_idx]
                        return json.loads(json_str)
                    else:
                        raise ValueError("No JSON array found in response")
                except json.JSONDecodeError:
                    print(f"Invalid JSON response (attempt {attempt+1}): {content[:100]}...")
                    time.sleep(2)
                    continue
            else:
                print(f"HTTP error {response.status_code} (attempt {attempt+1})")
                time.sleep(2)
                
        except Exception as e:
            print(f"Error (attempt {attempt+1}):", e)
            time.sleep(2)

    # Return default values if all attempts fail
    default_features = []
    for vid in video_ids_batch:
        default_features.append({
            "video_id": vid,
            "topic": "other",
            "tone": "neutral", 
            "target": "Neither",
            "style": "factual"
        })
    return default_features

def process_all_transcripts_batched(df, batch_size=5, model="llama3.1:8b"):
    """Process all transcripts in batches for much faster processing."""
    
    # Check if cached features exist
    cache_file = "llm_features.csv"
    if os.path.exists(cache_file):
        print(f"🎯 Loading cached features from {cache_file}")
        cached_features = pd.read_csv(cache_file)
        print(f"   Loaded {len(cached_features)} cached feature sets")
        return cached_features
    
    print(f"🚀 Processing {len(df)} transcripts in batches of {batch_size}...")
    
    all_features = []
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    for i in range(0, len(df), batch_size):
        batch_end = min(i + batch_size, len(df))
        batch_df = df.iloc[i:batch_end]
        
        batch_num = (i // batch_size) + 1
        print(f"   Processing batch {batch_num}/{total_batches} (videos {i+1}-{batch_end})...")
        
        # Extract batch features
        batch_features = extract_features_batch_ollama(
            batch_df["text"].tolist(),
            batch_df["video_id"].tolist(),
            model=model
        )
        
        # Add to results
        all_features.extend(batch_features)
        
        # Small delay between batches to be nice to Ollama
        time.sleep(0.5)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)
    
    # Cache the results
    print(f"💾 Caching features to {cache_file}")
    features_df.to_csv(cache_file, index=False)
    
    return features_df

# Check if we need to import os
import os

# Process all transcripts (with caching)
print("Testing Ollama with first batch...")
features_df = process_all_transcripts_batched(df, batch_size=5)

print(f"\nExtracted features for {len(features_df)} videos")
print("Feature columns:", features_df.columns.tolist())
print("\nFirst few feature sets:")
print(features_df.head())

# Merge features back into main dataframe
df_full = pd.merge(df, features_df, on="video_id")
print("Dataframe with features:")
print(df_full.head())

# Load MaxDiff data and merge
print("\nLoading MaxDiff data...")
maxdiff_data = pd.read_csv("maxdiff_dummy_data.csv")
print(f"MaxDiff data shape: {maxdiff_data.shape}")

# Merge MaxDiff mean scores
df_full = pd.merge(df_full, maxdiff_data[["video_id", "maxdiff_mean"]], on="video_id", how="left")
print(f"Final merged dataset shape: {df_full.shape}")

# Check for any missing MaxDiff values
missing_md = df_full["maxdiff_mean"].isnull().sum()
print(f"Missing MaxDiff values: {missing_md}")

print("\nFinal dataset columns:")
print(df_full.columns.tolist())
print("\nFirst few rows of complete dataset:")
print(df_full[["video_id", "treatment_effect", "maxdiff_mean", "topic", "tone", "target", "style"]].head())

# ============================================
# 3. Encode features + fit regression model
# ============================================

# Prepare features for both models
X = pd.get_dummies(df_full[["topic","tone","target","style"]], drop_first=True)
print(f"Feature matrix shape: {X.shape}")
print("Features included:")
print(X.columns.tolist())

# Model 1: RCT (Treatment Effects)
print("\n" + "="*50)
print("MODEL 1: RCT TREATMENT EFFECTS")
print("="*50)
y_rct = df_full["treatment_effect"]
X_rct = sm.add_constant(X)
model_rct = sm.OLS(y_rct, X_rct).fit()
print(model_rct.summary())

# Model 2: MaxDiff (Survey Responses)
print("\n" + "="*50)
print("MODEL 2: MAXDIFF SURVEY RESPONSES")
print("="*50)
y_md = df_full["maxdiff_mean"]
X_md = sm.add_constant(X)
model_md = sm.OLS(y_md, X_md).fit()
print(model_md.summary())

# ============================================
# 4. Model Comparison & Insights
# ============================================

print("\n" + "="*60)
print("MODEL COMPARISON & INSIGHTS")
print("="*60)

# Compare coefficients between models
coef_comparison = pd.DataFrame({
    'Feature': model_rct.params.index,
    'RCT_Coef': model_rct.params.values,
    'MaxDiff_Coef': model_md.params.values,
    'RCT_Pvalue': model_rct.pvalues.values,
    'MaxDiff_Pvalue': model_md.pvalues.values
})

# Calculate difference and significance
coef_comparison['Coef_Difference'] = coef_comparison['RCT_Coef'] - coef_comparison['MaxDiff_Coef']
coef_comparison['RCT_Significant'] = coef_comparison['RCT_Pvalue'] < 0.05
coef_comparison['MaxDiff_Significant'] = coef_comparison['MaxDiff_Pvalue'] < 0.05

print("\nCoefficient Comparison:")
print(coef_comparison.round(4))

# Identify key insights
print("\n" + "-"*40)
print("KEY INSIGHTS:")
print("-"*40)

# Find features that work in reality but not in surveys
reality_works = coef_comparison[
    (coef_comparison['RCT_Significant'] == True) & 
    (coef_comparison['MaxDiff_Significant'] == False) &
    (abs(coef_comparison['RCT_Coef']) > 0.01)
]

if len(reality_works) > 0:
    print("\n🎯 FEATURES THAT WORK IN REALITY BUT NOT IN SURVEYS:")
    for _, row in reality_works.iterrows():
        print(f"  • {row['Feature']}: RCT effect = {row['RCT_Coef']:.4f} (p={row['RCT_Pvalue']:.4f})")
else:
    print("\n✅ No significant mismatches found")

# Find features that work in both
both_work = coef_comparison[
    (coef_comparison['RCT_Significant'] == True) & 
    (coef_comparison['MaxDiff_Significant'] == True) &
    (coef_comparison['RCT_Coef'] * coef_comparison['MaxDiff_Coef'] > 0)  # Same direction
]

if len(both_work) > 0:
    print("\n✅ FEATURES THAT WORK IN BOTH REALITY AND SURVEYS:")
    for _, row in both_work.iterrows():
        print(f"  • {row['Feature']}: RCT = {row['RCT_Coef']:.4f}, MaxDiff = {row['MaxDiff_Coef']:.4f}")

# ============================================
# 5. Visualize treatment effects
# ============================================

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Treatment effects by video
ax1.scatter(df_full["video_id"], df_full["treatment_effect"], alpha=0.6)
ax1.axhline(0, color="red", linestyle="--", alpha=0.7)
ax1.set_xlabel("Video ID")
ax1.set_ylabel("Treatment Effect (Trump Approval Shift)")
ax1.set_title("RCT Treatment Effects by Video")
ax1.grid(True, alpha=0.3)

# Plot 2: MaxDiff scores by video
ax2.scatter(df_full["video_id"], df_full["maxdiff_mean"], alpha=0.6, color='green')
ax2.set_xlabel("Video ID")
ax2.set_ylabel("MaxDiff Mean Score")
ax2.set_title("MaxDiff Survey Scores by Video")
ax2.grid(True, alpha=0.3)

# Plot 3: Correlation between RCT and MaxDiff
ax3.scatter(df_full["treatment_effect"], df_full["maxdiff_mean"], alpha=0.6)
ax3.set_xlabel("RCT Treatment Effect")
ax3.set_ylabel("MaxDiff Mean Score")
ax3.set_title("RCT vs MaxDiff Correlation")
ax3.grid(True, alpha=0.3)

# Plot 4: Coefficient comparison
significant_features = coef_comparison[coef_comparison['RCT_Significant'] | coef_comparison['MaxDiff_Significant']]
if len(significant_features) > 0:
    x_pos = range(len(significant_features))
    ax4.bar([i-0.2 for i in x_pos], significant_features['RCT_Coef'], width=0.4, label='RCT', alpha=0.7)
    ax4.bar([i+0.2 for i in x_pos], significant_features['MaxDiff_Coef'], width=0.4, label='MaxDiff', alpha=0.7)
    ax4.set_xlabel("Features")
    ax4.set_ylabel("Coefficient Value")
    ax4.set_title("Feature Coefficients: RCT vs MaxDiff")
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(significant_features['Feature'], rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print correlation
correlation = df_full["treatment_effect"].corr(df_full["maxdiff_mean"])
print(f"\n📊 Correlation between RCT and MaxDiff: {correlation:.4f}")
print(f"   This shows how well survey responses align with experimental effects")
