import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def analyze_dataset():
    """Analyze the aircraft detection dataset for multi-class classification"""
    
    # Load the dataset
    df = pd.read_csv('dataset/sample_meta.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Basic statistics
    print("\n=== BASIC STATISTICS ===")
    print(f"Total recordings: {len(df)}")
    print(f"Aircraft recordings (class=1): {len(df[df['class'] == 1])}")
    print(f"No aircraft recordings (class=0): {len(df[df['class'] == 0])}")
    
    # Target columns for classification
    target_cols = ['engtype', 'engnum', 'model', 'engmodel', 'fueltype']
    
    print("\n=== TARGET COLUMNS ANALYSIS ===")
    for col in target_cols:
        print(f"\n{col.upper()}:")
        print(f"  Unique values: {df[col].nunique()}")
        print(f"  Missing values: {df[col].isna().sum()}")
        
        # Show distribution for non-missing values
        non_null = df[col].dropna()
        if len(non_null) > 0:
            print(f"  Non-null values: {len(non_null)}")
            print(f"  Top 10 values:")
            value_counts = non_null.value_counts()
            for val, count in value_counts.head(10).items():
                print(f"    {val}: {count}")
    
    # Analyze only aircraft recordings
    aircraft_df = df[df['class'] == 1].copy()
    print(f"\n=== AIRCRAFT RECORDINGS ANALYSIS ===")
    print(f"Total aircraft recordings: {len(aircraft_df)}")
    
    for col in target_cols:
        print(f"\n{col.upper()} (aircraft only):")
        non_null = aircraft_df[col].dropna()
        if len(non_null) > 0:
            print(f"  Non-null values: {len(non_null)}")
            print(f"  Unique values: {non_null.nunique()}")
            print(f"  Distribution:")
            value_counts = non_null.value_counts()
            for val, count in value_counts.head(10).items():
                print(f"    {val}: {count}")
    
    # Check for correlations between target variables
    print(f"\n=== CORRELATION ANALYSIS ===")
    aircraft_numeric = aircraft_df[['engnum']].copy()
    
    # Convert categorical to numeric for correlation
    for col in ['engtype', 'fueltype']:
        if col in aircraft_df.columns:
            aircraft_numeric[col] = pd.Categorical(aircraft_df[col]).codes
    
    correlation_matrix = aircraft_numeric.corr()
    print("Correlation matrix:")
    print(correlation_matrix)
    
    # Data quality assessment
    print(f"\n=== DATA QUALITY ASSESSMENT ===")
    print("Missing data pattern:")
    missing_data = df[target_cols].isnull().sum()
    print(missing_data)
    
    # Check if missing data is related to class
    print(f"\nMissing data by class:")
    for col in target_cols:
        missing_by_class = df.groupby('class')[col].apply(lambda x: x.isnull().sum())
        print(f"{col}: {dict(missing_by_class)}")
    
    # Recommendations for multi-class classification
    print(f"\n=== RECOMMENDATIONS FOR MULTI-CLASS CLASSIFICATION ===")
    
    # Filter to only aircraft recordings with complete data
    complete_aircraft = aircraft_df.dropna(subset=target_cols)
    print(f"Complete aircraft recordings: {len(complete_aircraft)}")
    
    if len(complete_aircraft) > 0:
        print(f"\nAvailable classes for each target:")
        for col in target_cols:
            unique_vals = complete_aircraft[col].unique()
            print(f"{col}: {len(unique_vals)} classes")
            print(f"  Values: {list(unique_vals)}")
    
    # Check class balance
    print(f"\n=== CLASS BALANCE ANALYSIS ===")
    for col in target_cols:
        if col in complete_aircraft.columns:
            class_counts = complete_aircraft[col].value_counts()
            print(f"\n{col} class distribution:")
            print(f"  Total classes: {len(class_counts)}")
            print(f"  Most common: {class_counts.index[0]} ({class_counts.iloc[0]} samples)")
            print(f"  Least common: {class_counts.index[-1]} ({class_counts.iloc[-1]} samples)")
            print(f"  Balance ratio: {class_counts.iloc[0] / class_counts.iloc[-1]:.2f}")
    
    return df, aircraft_df, complete_aircraft

def analyze_missing_data(df):
    """Analyze missing data patterns in detail"""
    
    print("=== DETAILED MISSING DATA ANALYSIS ===")
    
    # Target columns for analysis
    target_cols = ['engtype', 'engnum', 'model', 'engmodel', 'fueltype']
    
    # Calculate missing data statistics
    missing_stats = {}
    for col in target_cols:
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            total_count = len(df)
            missing_percentage = (missing_count / total_count) * 100
            missing_stats[col] = {
                'missing_count': missing_count,
                'total_count': total_count,
                'missing_percentage': missing_percentage,
                'present_count': total_count - missing_count
            }
    
    # Print detailed statistics
    for col, stats in missing_stats.items():
        print(f"\n{col.upper()}:")
        print(f"  Missing: {stats['missing_count']}/{stats['total_count']} ({stats['missing_percentage']:.1f}%)")
        print(f"  Present: {stats['present_count']}/{stats['total_count']} ({100 - stats['missing_percentage']:.1f}%)")
    
    # Analyze missing data patterns
    print(f"\n=== MISSING DATA PATTERNS ===")
    
    # Check if missing data is related to class
    if 'class' in df.columns:
        print(f"\nMissing data by class:")
        for col in target_cols:
            if col in df.columns:
                missing_by_class = df.groupby('class')[col].apply(lambda x: x.isnull().sum())
                total_by_class = df.groupby('class').size()
                print(f"\n{col}:")
                for class_val, missing_count in missing_by_class.items():
                    total_in_class = total_by_class[class_val]
                    percentage = (missing_count / total_in_class) * 100
                    class_label = "Aircraft" if class_val == 1 else "No Aircraft"
                    print(f"  {class_label} (class={class_val}): {missing_count}/{total_in_class} ({percentage:.1f}%)")
    
    # Check for complete vs incomplete records
    complete_records = df.dropna(subset=target_cols)
    incomplete_records = df[df[target_cols].isnull().any(axis=1)]
    
    print(f"\n=== RECORD COMPLETENESS ===")
    print(f"Complete records: {len(complete_records)} ({len(complete_records)/len(df)*100:.1f}%)")
    print(f"Incomplete records: {len(incomplete_records)} ({len(incomplete_records)/len(df)*100:.1f}%)")
    
    # Analyze which columns are most commonly missing together
    print(f"\n=== MISSING DATA COMBINATIONS ===")
    missing_combinations = df[target_cols].isnull().sum(axis=1).value_counts().sort_index()
    for missing_count, record_count in missing_combinations.items():
        percentage = (record_count / len(df)) * 100
        print(f"Records missing {missing_count} columns: {record_count} ({percentage:.1f}%)")
    
    return missing_stats

def create_visualizations(df, aircraft_df, complete_aircraft):
    """Create visualizations for the dataset analysis"""
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Aircraft Detection Dataset Analysis', fontsize=16)
    
    # 1. Class distribution
    class_counts = df['class'].value_counts()
    axes[0, 0].pie(class_counts.values, labels=['No Aircraft', 'Aircraft'], autopct='%1.1f%%')
    axes[0, 0].set_title('Overall Class Distribution')
    
    # 2. Engine type distribution (aircraft only)
    if 'engtype' in aircraft_df.columns:
        engtype_counts = aircraft_df['engtype'].dropna().value_counts()
        axes[0, 1].bar(engtype_counts.index, engtype_counts.values)
        axes[0, 1].set_title('Engine Type Distribution (Aircraft Only)')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Engine number distribution (aircraft only)
    if 'engnum' in aircraft_df.columns:
        engnum_counts = aircraft_df['engnum'].value_counts()
        axes[0, 2].bar(engnum_counts.index, engnum_counts.values)
        axes[0, 2].set_title('Engine Number Distribution (Aircraft Only)')
    
    # 4. Fuel type distribution (aircraft only)
    if 'fueltype' in aircraft_df.columns:
        fuel_counts = aircraft_df['fueltype'].dropna().value_counts()
        axes[1, 0].pie(fuel_counts.values, labels=fuel_counts.index, autopct='%1.1f%%')
        axes[1, 0].set_title('Fuel Type Distribution (Aircraft Only)')
    
    # 5. Top aircraft models
    if 'model' in aircraft_df.columns:
        model_counts = aircraft_df['model'].dropna().value_counts().head(10)
        axes[1, 1].barh(range(len(model_counts)), model_counts.values)
        axes[1, 1].set_yticks(range(len(model_counts)))
        axes[1, 1].set_yticklabels(model_counts.index)
        axes[1, 1].set_title('Top 10 Aircraft Models')
    
    # 6. Missing data heatmap - improved version
    # Only show target columns for missing data analysis (exclude 'class' as it's the target)
    target_cols_for_missing = ['engtype', 'engnum', 'model', 'engmodel', 'fueltype']
    missing_data = df[target_cols_for_missing].isnull()
    
    # Create a better missing data visualization
    if missing_data.sum().sum() > 0:  # Only create heatmap if there are missing values
        # Sample the data if too many rows to avoid overwhelming visualization
        if len(missing_data) > 1000:
            # Take a random sample of 1000 rows for better visualization
            sample_indices = np.random.choice(len(missing_data), 1000, replace=False)
            missing_data_sample = missing_data.iloc[sample_indices]
            title_suffix = " (1000 random samples)"
        else:
            missing_data_sample = missing_data
            title_suffix = ""
        
        sns.heatmap(missing_data_sample, 
                    cbar=True, 
                    ax=axes[1, 2],
                    cmap='viridis',
                    yticklabels=False)  # Hide y-axis labels for cleaner look
        axes[1, 2].set_title(f'Missing Data Pattern{title_suffix}')
        axes[1, 2].set_xlabel('Features')
        axes[1, 2].set_ylabel('Samples')
        
        # Add missing data summary as text
        missing_summary = missing_data.sum()
        total_samples = len(missing_data)
        summary_text = f"Missing values:\n"
        for col, missing_count in missing_summary.items():
            percentage = (missing_count / total_samples) * 100
            summary_text += f"{col}: {missing_count}/{total_samples} ({percentage:.1f}%)\n"
        
        # Add text box with summary
        axes[1, 2].text(0.02, 0.98, summary_text, 
                        transform=axes[1, 2].transAxes, 
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        fontsize=8)
    else:
        # If no missing data, show a message
        axes[1, 2].text(0.5, 0.5, 'No Missing Data\nin Target Columns', 
                        ha='center', va='center', transform=axes[1, 2].transAxes,
                        fontsize=14, fontweight='bold')
        axes[1, 2].set_title('Missing Data Pattern')
        axes[1, 2].set_xticks([])
        axes[1, 2].set_ylabel([])
    
    plt.tight_layout()
    plt.savefig('dataset_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Run the analysis
    df, aircraft_df, complete_aircraft = analyze_dataset()
    
    # Run detailed missing data analysis
    missing_stats = analyze_missing_data(df)
    
    # Create visualizations
    create_visualizations(df, aircraft_df, complete_aircraft)
    
    # Target columns for classification
    target_cols = ['engtype', 'engnum', 'model', 'engmodel', 'fueltype']
    
    print(f"\n=== SUMMARY ===")
    print(f"Total recordings: {len(df)}")
    print(f"Aircraft recordings: {len(aircraft_df)}")
    print(f"Complete aircraft recordings: {len(complete_aircraft)}")
    print(f"Target columns: {target_cols}")
    
    if len(complete_aircraft) > 0:
        print(f"\nFor multi-class classification, you have:")
        for col in target_cols:
            if col in complete_aircraft.columns:
                n_classes = complete_aircraft[col].nunique()
                print(f"  {col}: {n_classes} classes")
