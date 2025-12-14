# Spotify Dataset Analysis: What Makes a Hit Song?

An exploratory data analysis project examining patterns in music popularity using a dataset of 114,000+ Spotify tracks. We analyze genre performance, artist popularity, and discover which genres punch above their weight in producing hit songs.

## Key Findings

- **Dance music dominates**: With a 12.6x overperformance ratio, dance is the most efficient genre at producing hits—only 0.6% of the dataset, but 7.9% of high-popularity songs
- **Presence ≠ Success**: Larger genres (anime, synth-pop, garage) often underperform relative to their dataset size
- **Audio features don't predict popularity**: Danceability, energy, valence, etc. showed weak correlations with song success—genre and artist matter more

## Visualizations

The analysis generates several visualizations with a consistent dark theme:

| Visualization | Description |
|--------------|-------------|
| `genre_popularity.png` | Top 10 genres by average popularity score |
| `artist_popularity.png` | Top 10 artists (min. 5 songs) by popularity |
| `high_pop_genres_breakdown.png` | Genre distribution in high-popularity songs (≥70) |
| `genre_representation_scatter.png` | Dataset representation vs hit production |
| `genre_champions_vs_underperformers.png` | Overperformers vs underperformers comparison |

## Dataset

The Spotify dataset contains 114,000+ tracks with:
- **Metadata**: Artist, album, genre
- **Audio features**: Danceability, energy, valence, tempo, loudness, etc.
- **Metrics**: Popularity score (0-100), duration, explicit flag

Source: Source: [Spotify Tracks Attributes and Popularity](https://www.kaggle.com/datasets/melissamonfared/spotify-tracks-attributes-and-popularity/data) on Kaggle

## Methodology

We initially hypothesized that audio features (danceability, energy, valence) would predict popularity. 
Correlation analysis showed weak relationships (see `exploratory/correlation_analysis.py`), 
which led us to focus on genre efficiency instead.

## Installation

```bash
# Clone the repository
git clone https://github.com/OldCount/spotify-analysis.git
cd spotify-analysis

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. 1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/melissamonfared/spotify-tracks-attributes-and-popularity/data) and save it as `dataset.csv` in the project directory
2. Run the analysis:

```bash
python spotify_popularity.py
```

The script will:
- Load and clean the data
- Display exploration statistics
- Generate all visualizations as PNG files
- Print detailed analysis to console

## Configuration

Key parameters can be adjusted at the top of `spotify_popularity.py`:

```python
POPULARITY_THRESHOLD = 70    # Score threshold for "hit" songs
MIN_HIGH_POP_SONGS = 20      # Minimum hits for genre inclusion
MIN_ARTIST_SONGS = 5         # Minimum songs for artist analysis
```

## Project Structure

```
spotify-analysis/
├── .gitignore
├── README.md
├── requirements.txt
├── spotify_popularity.py         # Main analysis
├── exploratory/                  
│   ├── correlation_analysis.py              # Initial song 'mood' exploration.
│   └── exploratory_visualizations/          # Initial correlation charts
│        ├── correlation_absolute.png
│        ├── correlation_lollipop.png
│        ├── correlation_matrix.jpg
│        └── correlation_strength.png
├── docs/
│   └── presentation.pdf     # Project presentation
└── outputs/                 # Final generated visualizations
    ├── genre_popularity.png
    ├── artist_popularity.png
    ├── high_pop_genres_breakdown.png
    ├── genre_representation_scatter.png
    └── genre_champions_vs_underperformers.png
```

## Requirements

- Python 3.7+
- pandas
- matplotlib
- numpy

## Team

- Patrick Verschoor
- Camdon Watson
- Ionuț Marian

## License

This project was created for educational purposes as part of the Data Science and Artificial Intelligence minor at Leiden University.
