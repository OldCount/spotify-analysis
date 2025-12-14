"""
Spotify Correlation Analysis
This script explores whether audio features (danceability, energy, valence, etc.)
correlate with song popularity. Spoiler: they don't strongly correlate,
which led us to focus on genre efficiency analysis instead.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


# =============================================================================
# Configuration Constants
# =============================================================================

# Correlation strength thresholds for classification
STRONG_CORRELATION_THRESHOLD = 0.3
MODERATE_CORRELATION_THRESHOLD = 0.1

# Key features to include in correlation matrix visualization
KEY_FEATURES = [
    'popularity', 'valence', 'energy', 'danceability',
    'acousticness', 'instrumentalness', 'speechiness',
    'loudness', 'tempo', 'duration_minutes', 'liveness'
]


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_dataset(filepath):
    """
    Load the cleaned Spotify dataset from a CSV file.

    Parameters:
        filepath (str): Path to the CSV file

    Returns:
        DataFrame: Dataset loaded from CSV, or None if loading fails
    """
    print("Loading data...")

    try:
        df = pd.read_csv(filepath)
        if df.empty:
            print("Error: CSV file is empty")
            return None
        print(f"Loaded {len(df):,} rows successfully")
        return df
    except FileNotFoundError:
        print("File not found")
    except pd.errors.EmptyDataError:
        print("CSV has no data")
    except pd.errors.ParserError:
        print("CSV file is corrupted")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return None


# =============================================================================
# Correlation Analysis Functions
# =============================================================================

def get_numeric_features(df):
    """
    Extract numeric column names, excluding popularity and index.

    Parameters:
        df (DataFrame): Spotify dataset

    Returns:
        list: Names of numeric columns suitable for correlation analysis
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove columns that shouldn't be correlated
    columns_to_exclude = ['popularity', 'index', 'Unnamed: 0']
    for col in columns_to_exclude:
        if col in numeric_cols:
            numeric_cols.remove(col)

    return numeric_cols


def calculate_popularity_correlations(df, numeric_cols):
    """
    Calculate correlation coefficients between all numeric features and popularity.

    Parameters:
        df (DataFrame): Spotify dataset
        numeric_cols (list): List of numeric column names to correlate

    Returns:
        Series: Correlation values sorted from highest to lowest
    """
    correlations = df[numeric_cols].corrwith(df['popularity']).sort_values(ascending=False)
    return correlations


def classify_correlation_strength(correlation_value):
    """
    Classify a correlation coefficient by its strength.

    Parameters:
        correlation_value (float): Correlation coefficient between -1 and 1

    Returns:
        str: Classification label ('STRONG', 'MODERATE', or 'WEAK')
    """
    abs_corr = abs(correlation_value)

    if abs_corr > STRONG_CORRELATION_THRESHOLD:
        return "STRONG"
    elif abs_corr > MODERATE_CORRELATION_THRESHOLD:
        return "MODERATE"
    else:
        return "WEAK"


def display_correlation_summary(correlations):
    """
    Print a formatted summary of correlations with visual bars.

    Parameters:
        correlations (Series): Correlation values indexed by feature name
    """
    print("\nCORRELATIONS WITH POPULARITY:")
    print("-" * 70)
    print(f"{'Feature':<20} {'Correlation':>12}  {'Visual':^50}  {'Strength':<10}")
    print("-" * 70)

    for feature, corr in correlations.items():
        # Create visual bar proportional to correlation strength
        bar_length = int(abs(corr) * 50)
        bar = '█' * bar_length

        strength = classify_correlation_strength(corr)
        print(f"{feature:<20} {corr:>+.3f}        {bar:<50}  [{strength}]")

    print("-" * 70)
    print("\nKey insight: No features show STRONG correlation with popularity.")
    print("This suggests audio features alone don't determine song success.")


# =============================================================================
# Visualization Functions
# =============================================================================

def get_correlation_color(correlation_value):
    """
    Determine bar color based on correlation strength and direction.

    Parameters:
        correlation_value (float): Correlation coefficient

    Returns:
        str: Color name for matplotlib
    """
    abs_corr = abs(correlation_value)

    if abs_corr > STRONG_CORRELATION_THRESHOLD:
        return 'darkgreen' if correlation_value > 0 else 'darkred'
    elif abs_corr > MODERATE_CORRELATION_THRESHOLD:
        return 'lightgreen' if correlation_value > 0 else 'lightcoral'
    else:
        return 'lightgray'


def plot_correlation_strength(correlations):
    """
    Create a horizontal bar chart showing correlation strength with popularity.

    Color-codes bars by correlation strength and direction:
    - Dark green/red: Strong correlations (|r| > 0.3)
    - Light green/coral: Moderate correlations (|r| > 0.1)
    - Gray: Weak correlations

    Parameters:
        correlations (Series): Correlation values indexed by feature name
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = [get_correlation_color(corr) for corr in correlations.values]

    correlations.plot(kind='barh', ax=ax, color=colors, alpha=0.8, edgecolor='black')

    ax.set_xlabel('Correlation with Popularity', fontsize=12, fontweight='bold')
    ax.set_title('How Each Feature Relates to Popularity', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.grid(axis='x', alpha=0.3)

    # Create legend explaining color coding
    legend_elements = [
        Patch(facecolor='darkgreen', label='Strong Positive (>0.3)'),
        Patch(facecolor='lightgreen', label='Moderate Positive (0.1-0.3)'),
        Patch(facecolor='lightgray', label='Weak (-0.1 to 0.1)'),
        Patch(facecolor='lightcoral', label='Moderate Negative (-0.3 to -0.1)'),
        Patch(facecolor='darkred', label='Strong Negative (<-0.3)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig('correlation_strength.png', dpi=300, bbox_inches='tight')
    print("\nSaved: correlation_strength.png")


def plot_correlation_matrix(df):
    """
    Create a heatmap showing correlations between key audio features.

    Displays correlation coefficients as colored cells with numeric labels.
    Uses red-yellow-green colormap where green indicates positive correlation.

    Parameters:
        df (DataFrame): Spotify dataset containing the key features
    """
    # Filter to only features that exist in the dataframe
    available_features = [f for f in KEY_FEATURES if f in df.columns]
    corr_matrix = df[available_features].corr()

    fig, ax = plt.subplots(figsize=(12, 10))

    # Create heatmap using imshow
    im = ax.imshow(corr_matrix, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)

    # Configure axis labels
    ax.set_xticks(np.arange(len(available_features)))
    ax.set_yticks(np.arange(len(available_features)))
    ax.set_xticklabels(available_features, rotation=45, ha='right')
    ax.set_yticklabels(available_features)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Strength', fontsize=12, fontweight='bold')

    # Add correlation values as text annotations
    for i in range(len(available_features)):
        for j in range(len(available_features)):
            ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                    ha='center', va='center', color='black', fontsize=9)

    ax.set_title('Correlation Matrix: How Features Relate to Each Other',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("Saved: correlation_matrix.png")


def plot_correlation_lollipop(correlations):
    """
    Create a lollipop chart showing correlation strength and direction.

    Lollipop charts are effective for showing magnitude with clear
    directional indicators. Features are sorted by absolute correlation.

    Parameters:
        correlations (Series): Correlation values indexed by feature name
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Sort by absolute value while preserving sign
    correlations_sorted = correlations.reindex(
        correlations.abs().sort_values(ascending=True).index
    )

    y_pos = np.arange(len(correlations_sorted))
    colors = ['green' if x > 0 else 'red' for x in correlations_sorted.values]

    # Draw stems (lines from zero to point)
    ax.hlines(y=y_pos, xmin=0, xmax=correlations_sorted.values,
              color=colors, alpha=0.6, linewidth=3)

    # Draw lollipop heads (circles at end of stems)
    ax.scatter(correlations_sorted.values, y_pos, color=colors,
               s=100, alpha=0.8, edgecolors='black', linewidth=1.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(correlations_sorted.index)
    ax.set_xlabel('Correlation with Popularity', fontsize=12, fontweight='bold')
    ax.set_title('Feature Correlation Strength (Lollipop Chart)',
                 fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('correlation_lollipop.png', dpi=300, bbox_inches='tight')
    print("Saved: correlation_lollipop.png")


def plot_absolute_correlation(correlations):
    """
    Create a bar chart showing absolute correlation strength.

    Ignores direction to focus purely on how strongly each feature
    relates to popularity. Includes reference lines for strength thresholds.

    Parameters:
        correlations (Series): Correlation values indexed by feature name
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Convert to absolute values and sort
    abs_correlations = correlations.abs().sort_values(ascending=True)

    abs_correlations.plot(kind='barh', ax=ax, color='steelblue',
                          alpha=0.7, edgecolor='black')

    ax.set_xlabel('Absolute Correlation (Strength)', fontsize=12, fontweight='bold')
    ax.set_title('Feature Correlation Strength (Regardless of Direction)',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add threshold reference lines
    ax.axvline(x=STRONG_CORRELATION_THRESHOLD, color='red',
               linestyle='--', alpha=0.5, label=f'Strong (>{STRONG_CORRELATION_THRESHOLD})')
    ax.axvline(x=MODERATE_CORRELATION_THRESHOLD, color='orange',
               linestyle='--', alpha=0.5, label=f'Moderate (>{MODERATE_CORRELATION_THRESHOLD})')
    ax.legend()

    plt.tight_layout()
    plt.savefig('correlation_absolute.png', dpi=300, bbox_inches='tight')
    print("Saved: correlation_absolute.png")


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """
    Main execution function that orchestrates the correlation analysis workflow.

    This function:
    1. Loads the cleaned Spotify dataset
    2. Calculates correlations between audio features and popularity
    3. Displays summary statistics
    4. Generates four different correlation visualizations
    """
    print("=" * 70)
    print("SPOTIFY CORRELATION ANALYSIS")
    print("=" * 70)
    print("\nHypothesis: Audio features (energy, danceability, valence)")
    print("            may predict song popularity.")

    # Step 1: Load data
    df = load_dataset('spotify_cleaned.csv')
    if df is None:
        return

    # Step 2: Calculate correlations
    print("\n" + "=" * 70)
    print("CALCULATING CORRELATIONS")
    print("=" * 70)

    numeric_cols = get_numeric_features(df)
    correlations = calculate_popularity_correlations(df, numeric_cols)

    # Step 3: Display summary
    display_correlation_summary(correlations)

    # Step 4: Generate visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    plot_correlation_strength(correlations)
    plot_correlation_matrix(df)
    plot_correlation_lollipop(correlations)
    plot_absolute_correlation(correlations)

# Execute main function when script is run directly
if __name__ == "__main__":
    main()