"""
Spotify Dataset Analysis
This script analyzes a Spotify dataset to identify patterns in music popularity,
focusing on genre and artist performance metrics.
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


# =============================================================================
# Configuration Constants
# =============================================================================

# Define threshold for classifying songs as "high popularity"
POPULARITY_THRESHOLD = 70

# Minimum number of high-popularity songs required for a genre to be included in analysis
# This prevents statistical noise from genres with too few data points
MIN_HIGH_POP_SONGS = 20

# Minimum number of songs an artist must have to be included in analysis
# Ensures we're comparing artists with sufficient sample sizes
MIN_ARTIST_SONGS = 5


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_dataset(filepath):
    """
    Load the Spotify dataset from a CSV file.

    Parameters:
        filepath (str): Path to the CSV file

    Returns:
        DataFrame: Raw dataset loaded from CSV, or exits program if file not found
    """
    print("Loading data...")

    try:
        data_raw = pd.read_csv(filepath)
        if data_raw.empty:
            print("Error: CSV file is empty")
    except FileNotFoundError:
        print("File not found")
    except pd.errors.EmptyDataError:
        print("CSV has no data")
    except pd.errors.ParserError:
        print("CSV file is corrupted")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return data_raw

def explore_raw_data(data_raw):
    """
    Display initial exploration statistics for the raw dataset.

    Shows dataset dimensions, missing values, statistical summaries, and preview rows
    to understand data quality and structure before cleaning.

    Parameters:
        data_raw (DataFrame): Raw dataset to explore
    """
    print(f"Original dataset size: {data_raw.shape[0]} rows, {data_raw.shape[1]} columns")

    missing_values = data_raw.isnull().sum()
    print("\nMissing Values per Column:")
    print(missing_values[missing_values > 0] if missing_values.any() else "  None found")

    print("\nBasic Statistics (Numeric Columns):")
    print(data_raw.describe().T)

    print("\nPreview of Original Data (first 5 rows):")
    print(data_raw.head())


# =============================================================================
# Data Cleaning Functions
# =============================================================================

def clean_spotify_data(data_raw):
    """
    Clean the Spotify dataset by removing missing values, duplicates, and irrelevant data.

    This function performs the following operations:
    1. Removes rows containing any missing values
    2. Removes duplicate tracks (keeping first occurrence)
    3. Drops unnecessary columns to reduce memory usage
    4. Filters out songs with zero popularity (unreleased/untracked songs)

    Parameters:
        data_raw (DataFrame): The raw dataset loaded from CSV file

    Returns:
        DataFrame: Cleaned dataset ready for analysis
    """

    print("\nRemoving missing values...")
    data_cleaned = data_raw.dropna()
    rows_removed_missing = len(data_raw) - len(data_cleaned)
    print(f"  Removed {rows_removed_missing} rows with missing values")

    # Use track_id as unique identifier to avoid counting same song multiple times
    # keep='first' retains the first occurrence of each duplicate
    print("\nRemoving duplicate tracks...")
    before_duplicates = len(data_cleaned)
    data_cleaned = data_cleaned.drop_duplicates(subset=['track_id'], keep='first')
    duplicates_removed = before_duplicates - len(data_cleaned)
    print(f"  Removed {duplicates_removed} duplicate tracks")

    print(f"\nCleaned dataset size: {data_cleaned.shape[0]} rows, {data_cleaned.shape[1]} columns")
    print(f"Total rows removed: {len(data_raw) - len(data_cleaned)}")

    # Drop columns not needed for popularity analysis to improve performance
    # errors='ignore' prevents crashes if a column doesn't exist
    columns_to_drop = ['index', 'track_id', 'album_name', 'duration_ms', 'time_signature',
        'explicit', 'danceability', 'energy', 'key', 'loudness', 'mode',
        'speechiness', 'acousticness', 'instrumentalness', 'liveness', 
        'valence', 'tempo']
    data_cleaned = data_cleaned.drop(columns=columns_to_drop, errors='ignore')

    # Filter out songs with 0 popularity to only show songs with traction
    # .copy() prevents SettingWithCopyWarning in downstream operations
    data_cleaned = data_cleaned[data_cleaned['popularity'] > 0].copy()

    return data_cleaned


def display_cleaned_data_info(data_cleaned):
    """
    Display summary information about the cleaned dataset.

    Parameters:
        data_cleaned (DataFrame): Cleaned Spotify dataset
    """
    print("\nPreview of cleaned data (first 5 rows):")
    print(data_cleaned.head())

    print("\nDataset info:")
    print(f"Total songs: {len(data_cleaned)}")
    print(f"Number of columns: {len(data_cleaned.columns)}")
    print(f"Genres in dataset: {data_cleaned['track_genre'].nunique()}")
    print(f"Artists in dataset: {data_cleaned['artists'].nunique()}")
    print()


# =============================================================================
# Basic Analysis Functions
# =============================================================================

def analyze_genre_popularity(data_cleaned):
    """
    Analyze which music genres have the highest average popularity on Spotify.

    Groups all songs by genre and calculates aggregate statistics to identify
    the most popular genres based on average popularity score.

    Parameters:
        data_cleaned (DataFrame): Cleaned Spotify dataset

    Returns:
        DataFrame: Top 10 genres with their mean popularity and song count
    """
    # Use .agg() to calculate multiple statistics at once for efficiency
    genre_pop = data_cleaned.groupby('track_genre')['popularity'].agg(['mean', 'count'])

    genre_pop = genre_pop.sort_values('mean', ascending=False).head(10)

    return genre_pop


def analyze_artist_popularity(data_cleaned, min_songs=MIN_ARTIST_SONGS):
    """
    Analyze which artists have the highest average popularity on Spotify.

    Groups songs by artist and filters to include only artists with sufficient
    data (minimum 5 songs by default) to ensure statistical reliability.

    Parameters:
        data_cleaned (DataFrame): Cleaned Spotify dataset
        min_songs (int): Minimum number of songs required for an artist to be included

    Returns:
        DataFrame: Top 10 artists with their mean popularity and song count
    """
    artist_pop = data_cleaned.groupby('artists')['popularity'].agg(['mean', 'count'])

    # Filter by count to avoid statistical noise from one-hit wonders
    artist_pop = artist_pop[artist_pop['count'] >= min_songs]

    top_artists = artist_pop.sort_values('mean', ascending=False).head(10)

    return top_artists


# =============================================================================
# Visualization Utility Functions
# =============================================================================

def setup_dark_theme(fig, ax):
    """
    Apply consistent dark theme styling to matplotlib plots.

    Sets background colors, text colors, and axis styling to match
    a professional dark presentation theme.

    Parameters:
        fig (Figure): Matplotlib figure object
        ax (Axes): Matplotlib axes object
    """
    fig.patch.set_facecolor('#1a1d29')
    ax.set_facecolor('#242937')
    ax.tick_params(colors='white', which='both')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.2, color='gray')


# =============================================================================
# Basic Visualization Functions
# =============================================================================

def plot_genre_popularity(genre_data):
    """
    Create a vertical bar chart showing the top 10 most popular genres.

    Visualizes genre popularity with a purple-themed color scheme on a dark background.

    Parameters:
        genre_data (DataFrame): DataFrame containing genre popularity statistics
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    setup_dark_theme(fig, ax)

    ax.bar(genre_data.index, genre_data['mean'], color='#B8A4E8',
           edgecolor='#2C3E50', alpha=0.85)

    # Rotate labels to prevent overlap with long genre names
    ax.set_xticklabels(genre_data.index, rotation=45, ha='right')

    ax.set_xlabel('Genre', fontsize=12, fontweight='bold', color='white')
    ax.set_ylabel('Average Popularity', fontsize=12, fontweight='bold', color='white')
    ax.set_title('Top 10 Most Popular Music Genres on Spotify',
                 fontsize=14, fontweight='bold', color='white')

    plt.tight_layout()
    plt.savefig('genre_popularity.png', dpi=300)


def plot_artist_popularity(artist_data):
    """
    Create a horizontal bar chart showing the top 10 most popular artists.

    Uses horizontal bars to better accommodate long artist names.
    Only includes artists with at least 5 songs for statistical reliability.

    Parameters:
        artist_data (DataFrame): DataFrame containing artist popularity statistics
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    setup_dark_theme(fig, ax)

    ax.barh(artist_data.index, artist_data['mean'], color='#4ECDC4',
            edgecolor='#2C3E50', alpha=0.85)

    ax.set_xlabel('Average Popularity Score', fontsize=12, fontweight='bold', color='white')
    ax.set_ylabel('Artist', fontsize=12, fontweight='bold', color='white')
    ax.set_title('Top 10 Most Popular Artists on Spotify (min. 5 songs)',
                 fontsize=14, fontweight='bold', color='white')

    # Invert y-axis so highest popularity appears at top
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig('artist_popularity.png', dpi=300)


# =============================================================================
# Deep Dive Analysis Functions
# =============================================================================

def create_high_popularity_subset(data_cleaned, threshold=POPULARITY_THRESHOLD):
    """
    Create a subset of songs that meet the high popularity threshold.

    Parameters:
        data_cleaned (DataFrame): Cleaned Spotify dataset
        threshold (int): Minimum popularity score to be considered "high popularity"

    Returns:
        DataFrame: Subset containing only high-popularity songs
    """
    high_pop_songs = data_cleaned[data_cleaned['popularity'] >= threshold]

    print("\n" + "=" * 70)
    print("GENRE ANALYSIS: HIGH POPULARITY SONGS")
    print("=" * 70)
    print(f"\nTotal songs with popularity >= {threshold}: {len(high_pop_songs):,}")
    print(f"Total songs in dataset: {len(data_cleaned):,}")
    print(f"Percentage of high-popularity songs: {len(high_pop_songs)/len(data_cleaned)*100:.1f}%")

    return high_pop_songs


def analyze_genre_distribution(high_pop_songs, threshold=POPULARITY_THRESHOLD):
    """
    Analyze the distribution of genres within high-popularity songs.

    Parameters:
        high_pop_songs (DataFrame): Subset of high-popularity songs
        threshold (int): The popularity threshold used

    Returns:
        Series: Count of songs per genre in high-popularity subset
    """
    genre_counts_high_pop = high_pop_songs['track_genre'].value_counts()

    print(f"\nGENRE DISTRIBUTION IN HIGH POPULARITY SONGS (>= {threshold}):")
    print("-" * 70)
    print(f"{'Genre':<30} {'Count':>10} {'% of High-Pop':>15}")
    print("-" * 70)

    for genre, count in genre_counts_high_pop.head(20).items():
        percentage = (count / len(high_pop_songs)) * 100
        print(f"{genre:<30} {count:>10} {percentage:>14.1f}%")

    print(f"\n... and {len(genre_counts_high_pop) - 20} more genres")

    return genre_counts_high_pop


def create_comparison_dataframe(data_cleaned, high_pop_songs, min_songs=MIN_HIGH_POP_SONGS):
    """
    Create a comprehensive comparison of genre representation in dataset vs high-popularity songs.

    Calculates overperformance ratios to identify which genres are more or less likely
    to produce hit songs compared to their overall representation in the dataset.

    Parameters:
        data_cleaned (DataFrame): Full cleaned dataset
        high_pop_songs (DataFrame): Subset of high-popularity songs
        min_songs (int): Minimum high-popularity songs required for genre inclusion

    Returns:
        DataFrame: Comparison data with overperformance metrics for each genre
    """
    all_genre_counts = data_cleaned['track_genre'].value_counts()
    high_genre_counts = high_pop_songs['track_genre'].value_counts()

    comparison_data = []

    for genre in all_genre_counts.index:
        pct_in_dataset = (all_genre_counts[genre] / len(data_cleaned)) * 100

        # Use .get() with default 0 to handle genres not present in high-popularity subset
        high_pop_count = high_genre_counts.get(genre, 0)
        pct_in_high_pop = (high_pop_count / len(high_pop_songs)) * 100

        if high_pop_count >= min_songs:
            # Overperformance ratio: ratio > 1 means genre is overrepresented in hits
            # ratio < 1 means genre is underrepresented in hits
            ratio = pct_in_high_pop / pct_in_dataset if pct_in_dataset > 0 else 0

            comparison_data.append({
                'genre': genre,
                'pct_dataset': pct_in_dataset,
                'pct_high_pop': pct_in_high_pop,
                'ratio': ratio,
                'high_pop_count': high_pop_count,
                'total_count': all_genre_counts[genre]
            })

    comparison_df = pd.DataFrame(comparison_data).sort_values('ratio', ascending=False)

    return comparison_df


def display_overperformance_analysis(comparison_df):
    """
    Display detailed overperformance analysis for genres.

    Shows which genres are overrepresented in high-popularity songs compared
    to their overall presence in the dataset.

    Parameters:
        comparison_df (DataFrame): Comparison dataframe with overperformance metrics
    """
    print(f"\nGENRE OVERPERFORMANCE ANALYSIS")
    print("-" * 70)
    print(f"{'Genre':<30} {'% of Dataset':>15} {'% of High-Pop':>15} {'Ratio':>10}")
    print("-" * 70)

    for _, row in comparison_df.head(15).iterrows():
        print(f"{row['genre']:<30} {row['pct_dataset']:>14.1f}% {row['pct_high_pop']:>14.1f}% {row['ratio']:>9.2f}x")

    print("\n(Ratio > 1 indicates genre is overrepresented in high-popularity songs)")

    print(f"\nTOP 3 OVERPERFORMING GENRES:")
    print("-" * 70)

    for i, (_, row) in enumerate(comparison_df.head(3).iterrows(), 1):
        print(f"\n{i}. {row['genre'].upper()}:")
        print(f"   - Represents {row['pct_dataset']:.1f}% of all songs in dataset")
        print(f"   - Comprises {row['pct_high_pop']:.1f}% of high-popularity songs")
        print(f"   - Overperformance factor: {row['ratio']:.1f}x")


# =============================================================================
# Deep Dive Visualization Functions
# =============================================================================

def plot_high_pop_genre_breakdown(genre_counts_high_pop, threshold=POPULARITY_THRESHOLD):
    """
    Create horizontal bar chart showing top genres in high-popularity songs.

    Parameters:
        genre_counts_high_pop (Series): Genre counts in high-popularity subset
        threshold (int): Popularity threshold used (for display purposes)
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    setup_dark_theme(fig, ax)

    top_genres_high_pop = genre_counts_high_pop.head(15)

    top_genres_high_pop.plot(kind='barh', ax=ax, color='#B8A4E8',
                             edgecolor='#2C3E50', alpha=0.85)

    ax.set_xlabel('Number of Songs', fontsize=12, fontweight='bold', color='white')
    ax.set_ylabel('Genre', fontsize=12, fontweight='bold', color='white')
    ax.set_title(f'Top 15 Genres in High Popularity Songs (>= {threshold})',
                 fontsize=14, fontweight='bold', color='white')

    ax.invert_yaxis()

    # Add count labels at end of each bar for precise values
    for i, (genre, count) in enumerate(top_genres_high_pop.items()):
        ax.text(count + 5, i, f'{count}', va='center', fontweight='bold', color='white')

    plt.tight_layout()
    plt.savefig('high_pop_genres_breakdown.png', dpi=300)
    print("\nSaved: high_pop_genres_breakdown.png")


def plot_genre_comparison(comparison_df):
    """
    Create side-by-side bar chart comparing genre representation in dataset vs high-popularity songs.

    Parameters:
        comparison_df (DataFrame): Comparison dataframe with genre metrics
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    setup_dark_theme(fig, ax)

    # Sort by high-popularity percentage for better visual flow
    top_15_comparison = comparison_df.head(20).sort_values('pct_high_pop', ascending=True)

    x_positions = np.arange(len(top_15_comparison))
    bar_width = 0.35

    # Create grouped bars: teal shows overall dataset, purple shows high-popularity subset
    bars_dataset = ax.barh(x_positions - bar_width/2, top_15_comparison['pct_dataset'],
                           bar_width, label='% of Total Dataset',
                           color='#4ECDC4', edgecolor='#2C3E50', alpha=0.85)

    bars_high_pop = ax.barh(x_positions + bar_width/2, top_15_comparison['pct_high_pop'],
                            bar_width, label='% of High-Popularity Songs',
                            color='#B8A4E8', edgecolor='#2C3E50', alpha=0.85)

    ax.set_yticks(x_positions)
    ax.set_yticklabels(top_15_comparison['genre'])
    ax.set_xlabel('Percentage', fontsize=12, fontweight='bold', color='white')
    ax.set_title('Genre Representation: Dataset vs High-Popularity Songs',
                 fontsize=14, fontweight='bold', color='white')

    legend = ax.legend(fontsize=11, facecolor='#2C3E50', edgecolor='white')
    for text in legend.get_texts():
        text.set_color('white')

    # Add percentage labels for both bars
    for i, (_, row) in enumerate(top_15_comparison.iterrows()):
        ax.text(row['pct_dataset'] + 0.1, i - bar_width/2,
                f"{row['pct_dataset']:.1f}%", va='center', fontsize=9, color='white')
        ax.text(row['pct_high_pop'] + 0.1, i + bar_width/2,
                f"{row['pct_high_pop']:.1f}%", va='center', fontsize=9, fontweight='bold', color='white')

    plt.savefig('genre_size_vs_hits.png', dpi=300, bbox_inches='tight')
    print("\nSaved: genre_size_vs_hits.png")


def plot_overperformers_underperformers(comparison_df):
    """
    Create side-by-side comparison of top overperforming and underperforming genres.

    Parameters:
        comparison_df (DataFrame): Comparison dataframe sorted by overperformance ratio
    """
    top_10_over = comparison_df.head(10)
    bottom_10_under = comparison_df.tail(10).sort_values('ratio', ascending=True)

    y_positions = np.arange(10)

    # Use gridspec_kw to add spacing between subplots for better readability
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7),
                                   gridspec_kw={'wspace': 0.3})

    fig.patch.set_facecolor('#1a1d29')
    ax1.set_facecolor('#242937')
    ax2.set_facecolor('#242937')

    # Calculate x-axis limits with 15% padding for label space
    overperformers_limit = top_10_over['ratio'].max() * 1.15
    underperformers_limit = bottom_10_under['ratio'].max() * 1.15

    # Left subplot: Overperformers (green = success)
    bars_over = ax1.barh(y_positions, top_10_over['ratio'],
                         color='#2ECC71', edgecolor='#2C3E50', alpha=0.85)
    ax1.set_yticks(y_positions)
    ax1.set_yticklabels(top_10_over['genre'])
    ax1.set_xlabel('Overperformance Ratio', fontweight='bold', color='white')
    ax1.set_title('Top Overperformers', fontsize=12, fontweight='bold', color='white')

    # Add reference line at ratio=1.0 (proportional representation)
    ax1.axvline(x=1, color='#FFD700', linestyle='--', linewidth=2, alpha=0.8)
    ax1.invert_yaxis()
    ax1.set_xlim(0, overperformers_limit)

    ax1.tick_params(colors='white', which='both')
    ax1.spines['bottom'].set_color('white')
    ax1.spines['left'].set_color('white')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='x', alpha=0.2, color='gray')

    for i, ratio in enumerate(top_10_over['ratio']):
        ax1.text(ratio + overperformers_limit * 0.01, i, f'{ratio:.1f}x',
                 va='center', fontsize=9, fontweight='bold', color='white')

    # Right subplot: Underperformers (pink = underperformance)
    bars_under = ax2.barh(y_positions, bottom_10_under['ratio'],
                          color='#FF6B9D', edgecolor='#2C3E50', alpha=0.85)
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(bottom_10_under['genre'])
    ax2.set_xlabel('Underperformance Ratio', fontweight='bold', color='white')
    ax2.set_title('Bottom Underperformers', fontsize=12, fontweight='bold', color='white')

    ax2.axvline(x=1, color='#FFD700', linestyle='--', linewidth=2, alpha=0.8)
    ax2.invert_yaxis()
    ax2.set_xlim(0, underperformers_limit)

    ax2.tick_params(colors='white', which='both')
    ax2.spines['bottom'].set_color('white')
    ax2.spines['left'].set_color('white')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='x', alpha=0.2, color='gray')

    for i, ratio in enumerate(bottom_10_under['ratio']):
        ax2.text(ratio + underperformers_limit * 0.01, i, f'{ratio:.2f}x',
                 va='center', fontsize=9, fontweight='bold', color='white')

    fig.suptitle('Genre Performance: Overperformers vs Underperformers',
                 fontsize=15, fontweight='bold', color='white', y=0.98)
    plt.savefig('genre_champions_vs_underperformers.png', dpi=300, bbox_inches='tight')
    print("Saved: genre_champions_vs_underperformers.png")


def plot_genre_representation_scatter(comparison_df, threshold=POPULARITY_THRESHOLD):
    """
    Create scatter plot showing genre representation in dataset vs high-popularity songs.
    Uses color gradient to indicate overperformance ratio and annotates top overperformers.
    
    Parameters:
    comparison_df (DataFrame): Comparison dataframe with genre metrics
    threshold (int): Popularity threshold used (for display purposes)
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    setup_dark_theme(fig, ax)
    
    colors = ['#FF6B9D', '#FFB142', '#2ECC71']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('coral_to_green', colors, N=n_bins)
    
    x_values = comparison_df['pct_dataset']
    y_values = comparison_df['pct_high_pop']
    colors_by_ratio = comparison_df['ratio']
    
    scatter = ax.scatter(x_values, y_values, c=colors_by_ratio, s=200, alpha=0.7,
                        cmap=cmap, edgecolors='#2C3E50', linewidth=1.5)
    
    ax.plot([0, 2], [0, 2], color='#45B7D1', linestyle='--', linewidth=2, alpha=0.6,
            label='Equal representation')
    
    # Annotate top overperformers and bottom underperformers
    top_performers = comparison_df.head(11) # 11 to include k-pop as an example
    bottom_performers = comparison_df.tail(3)
    genres_to_annotate = pd.concat([top_performers, bottom_performers])
    
    # Vary label positions to reduce overlap
    offsets = [(10, 10), (-40, 10), (10, -20), (-40, -20), (10, 25), 
               (-40, 25), (10, -35), (-40, -10)]
    
    for idx, (_, row) in enumerate(genres_to_annotate.iterrows()):
        if row['pct_dataset'] <= 2.0:  # Skip if too far right
            offset = offsets[idx % len(offsets)]
            
            ax.annotate(row['genre'],
                       (row['pct_dataset'], row['pct_high_pop']),
                       fontsize=9, fontweight='bold',
                       xytext=offset, textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor='#6C5B7B', alpha=0.9),
                       arrowprops=dict(arrowstyle='->', color='#6C5B7B', 
                                     lw=1.0, alpha=0.7))
    
    ax.set_xlabel('% of Total Dataset', fontsize=12, fontweight='bold', color='white')
    ax.set_ylabel(f'% of High Popularity Songs (>= {threshold})',
                 fontsize=12, fontweight='bold', color='white')
    ax.set_title('Genre Representation: Dataset vs High-Popularity Songs',
                fontsize=14, fontweight='bold', color='white')
    ax.set_xlim(0, 1.6)
    
    legend = ax.legend(facecolor='#2C3E50', edgecolor='white')
    for text in legend.get_texts():
        text.set_color('white')
    
    colorbar = plt.colorbar(scatter, ax=ax)
    colorbar.set_label('Overperformance Ratio', fontweight='bold', color='white')
    colorbar.ax.yaxis.set_tick_params(color='white')
    colorbar.outline.set_edgecolor('white')
    plt.setp(plt.getp(colorbar.ax.axes, 'yticklabels'), color='white')
    
    plt.savefig('genre_representation_scatter.png', dpi=300, bbox_inches='tight')
    print("Saved: genre_representation_scatter.png")

# =============================================================================
# Main Execution
# =============================================================================

def main():
    """
    Main execution function that orchestrates the entire analysis workflow.

    This function:
    1. Loads and explores the raw data
    2. Cleans the dataset
    3. Performs basic popularity analysis (genres and artists)
    4. Conducts deep dive analysis on high-popularity songs
    5. Generates all visualizations
    """

    # Step 1: Load and explore raw data
    data_raw = load_dataset('dataset.csv')
    explore_raw_data(data_raw)

    # Step 2: Clean the dataset
    data_cleaned = clean_spotify_data(data_raw)
    display_cleaned_data_info(data_cleaned)

    # Step 3: Basic popularity analysis
    print("=" * 70)
    print("BASIC POPULARITY ANALYSIS")
    print("=" * 70)

    genre_data = analyze_genre_popularity(data_cleaned)
    print("\nTop 10 Genres by Popularity:")
    print(genre_data)
    plot_genre_popularity(genre_data)

    artist_data = analyze_artist_popularity(data_cleaned)
    print("\nTop 10 Artists by Popularity:")
    print(artist_data)
    plot_artist_popularity(artist_data)

    # Step 4: Deep dive analysis on high-popularity songs
    high_pop_songs = create_high_popularity_subset(data_cleaned)
    if high_pop_songs.empty:
      print("Warning: No high-popularity songs found")
      return
    genre_counts_high_pop = analyze_genre_distribution(high_pop_songs)

    # Step 5: Create comprehensive comparison analysis
    comparison_df = create_comparison_dataframe(data_cleaned, high_pop_songs)
    display_overperformance_analysis(comparison_df)

    # Step 6: Generate all deep dive visualizations
    plot_high_pop_genre_breakdown(genre_counts_high_pop)
    plot_genre_comparison(comparison_df)
    plot_overperformers_underperformers(comparison_df)
    plot_genre_representation_scatter(comparison_df)

    print("\nAnalysis complete. All visualizations saved successfully.")



# Execute main function when script is run directly
if __name__ == "__main__":
    main()

