"""
Smart Prioritization Engine

Calculates SEO Value Score based on:
- Relevancy (0.40): AI semantic analysis
- Traffic Potential (0.35): Search volume + keyword difficulty
- Ranking Opportunity (0.25): Current position + improvement potential

Formula: SEO_Value_Score = (Relevancy × 0.40) + (Traffic × 0.35) + (Ranking × 0.25)
"""

import pandas as pd
from typing import Optional, Dict
import math


class PrioritizationEngine:
    """Calculate SEO Value Scores for intelligent keyword prioritization."""

    # Configurable weights (sum must equal 1.0)
    RELEVANCY_WEIGHT = 0.40
    TRAFFIC_WEIGHT = 0.35
    RANKING_WEIGHT = 0.25

    def __init__(
        self,
        relevancy_weight: float = 0.40,
        traffic_weight: float = 0.35,
        ranking_weight: float = 0.25
    ):
        """Initialize prioritization engine with custom weights.

        Args:
            relevancy_weight: Weight for semantic relevancy (default: 0.40)
            traffic_weight: Weight for traffic potential (default: 0.35)
            ranking_weight: Weight for ranking opportunity (default: 0.25)
        """
        # Validate weights sum to 1.0
        total = relevancy_weight + traffic_weight + ranking_weight
        if not math.isclose(total, 1.0, abs_tol=0.01):
            raise ValueError(f"Weights must sum to 1.0, got {total}")

        self.relevancy_weight = relevancy_weight
        self.traffic_weight = traffic_weight
        self.ranking_weight = ranking_weight

    def calculate_relevancy_score(
        self,
        ai_relevancy: Optional[float] = None,
        in_title: bool = False,
        in_h1: bool = False,
        in_content: bool = False
    ) -> float:
        """Calculate relevancy score (0-100).

        If AI relevancy score is available, use it.
        Otherwise, use keyword presence in HTML elements.

        Args:
            ai_relevancy: AI semantic relevancy score (0-100)
            in_title: Whether keyword appears in title
            in_h1: Whether keyword appears in H1
            in_content: Whether keyword appears in content

        Returns:
            Relevancy score (0-100)
        """
        if ai_relevancy is not None:
            # Use AI score directly
            return max(0, min(100, ai_relevancy))

        # Fallback: keyword presence scoring
        score = 0

        if in_title:
            score += 40  # Title is most important
        if in_h1:
            score += 30  # H1 is second most important
        if in_content:
            score += 30  # Content presence

        return min(100, score)

    def calculate_traffic_potential_score(
        self,
        search_volume: Optional[int] = None,
        keyword_difficulty: Optional[int] = None
    ) -> float:
        """Calculate traffic potential score (0-100).

        High volume + low difficulty = high score
        Low volume + high difficulty = low score

        Args:
            search_volume: Monthly search volume
            keyword_difficulty: Keyword difficulty (0-100)

        Returns:
            Traffic potential score (0-100)
        """
        if search_volume is None or keyword_difficulty is None:
            # No data available
            return 50  # Neutral score

        # Normalize search volume (logarithmic scale)
        # 10 searches = 10, 100 = 30, 1000 = 50, 10000 = 70, 100000 = 90
        if search_volume <= 0:
            volume_score = 0
        else:
            volume_score = min(100, 10 + (math.log10(search_volume) * 20))

        # Invert keyword difficulty (lower is better)
        difficulty_score = 100 - keyword_difficulty

        # Combined score: 60% volume, 40% difficulty
        traffic_score = (volume_score * 0.6) + (difficulty_score * 0.4)

        return max(0, min(100, traffic_score))

    def calculate_ranking_opportunity_score(
        self,
        current_position: Optional[float] = None,
        clicks: Optional[int] = None,
        impressions: Optional[int] = None
    ) -> float:
        """Calculate ranking opportunity score (0-100).

        Better positions closer to top get higher scores.
        Consider CTR improvement potential.

        Args:
            current_position: Current average position
            clicks: Current clicks
            impressions: Current impressions

        Returns:
            Ranking opportunity score (0-100)
        """
        if current_position is None:
            return 50  # Neutral score

        # Position-based scoring
        # Positions 1-3: Already ranking well, less opportunity
        # Positions 4-10: High opportunity (page 1)
        # Positions 11-20: Good opportunity (striking distance)
        # Positions 21+: Lower priority

        if current_position <= 3:
            position_score = 60  # Good position, but less improvement potential
        elif current_position <= 10:
            position_score = 90  # Best opportunity - page 1
        elif current_position <= 20:
            position_score = 75  # Striking distance
        elif current_position <= 30:
            position_score = 50  # Moderate opportunity
        else:
            # Gradual decline for lower positions
            position_score = max(10, 50 - ((current_position - 30) * 1.5))

        # CTR improvement potential
        if clicks is not None and impressions is not None and impressions > 0:
            current_ctr = (clicks / impressions) * 100

            # Expected CTR for position 1-3 is ~20-30%
            # Expected CTR for position 4-10 is ~5-15%
            # Calculate improvement potential

            if current_position <= 3:
                expected_ctr = 25
            elif current_position <= 10:
                expected_ctr = 10
            elif current_position <= 20:
                expected_ctr = 3
            else:
                expected_ctr = 1

            # If current CTR is much lower than expected, higher opportunity
            if current_ctr < expected_ctr * 0.5:
                ctr_boost = 20
            elif current_ctr < expected_ctr * 0.75:
                ctr_boost = 10
            else:
                ctr_boost = 0

            position_score = min(100, position_score + ctr_boost)

        return max(0, min(100, position_score))

    def calculate_seo_value_score(
        self,
        relevancy_score: float,
        traffic_score: float,
        ranking_score: float
    ) -> float:
        """Calculate final SEO Value Score.

        Args:
            relevancy_score: Relevancy score (0-100)
            traffic_score: Traffic potential score (0-100)
            ranking_score: Ranking opportunity score (0-100)

        Returns:
            SEO Value Score (0-100)
        """
        seo_score = (
            (relevancy_score * self.relevancy_weight) +
            (traffic_score * self.traffic_weight) +
            (ranking_score * self.ranking_weight)
        )

        return round(seo_score, 2)

    def prioritize_dataframe(
        self,
        df: pd.DataFrame,
        ai_relevancy_col: str = 'ai_relevancy_score',
        volume_col: str = 'search_volume',
        difficulty_col: str = 'keyword_difficulty',
        position_col: str = 'position',
        clicks_col: str = 'clicks',
        impressions_col: str = 'impressions',
        title_col: str = 'in_title',
        h1_col: str = 'in_h1',
        content_col: str = 'in_content'
    ) -> pd.DataFrame:
        """Add prioritization scores to a dataframe.

        Args:
            df: Input dataframe
            ai_relevancy_col: Column name for AI relevancy scores
            volume_col: Column name for search volume
            difficulty_col: Column name for keyword difficulty
            position_col: Column name for current position
            clicks_col: Column name for clicks
            impressions_col: Column name for impressions
            title_col: Column name for keyword in title flag
            h1_col: Column name for keyword in H1 flag
            content_col: Column name for keyword in content flag

        Returns:
            Dataframe with added prioritization columns
        """
        df = df.copy()

        # Calculate component scores
        relevancy_scores = []
        traffic_scores = []
        ranking_scores = []
        seo_value_scores = []

        for idx, row in df.iterrows():
            # Relevancy score
            ai_rel = row.get(ai_relevancy_col) if ai_relevancy_col in df.columns else None
            in_title = row.get(title_col, False) if title_col in df.columns else False
            in_h1 = row.get(h1_col, False) if h1_col in df.columns else False
            in_content = row.get(content_col, False) if content_col in df.columns else False

            rel_score = self.calculate_relevancy_score(
                ai_relevancy=ai_rel,
                in_title=in_title,
                in_h1=in_h1,
                in_content=in_content
            )
            relevancy_scores.append(rel_score)

            # Traffic potential score
            volume = row.get(volume_col) if volume_col in df.columns else None
            difficulty = row.get(difficulty_col) if difficulty_col in df.columns else None

            traffic_score = self.calculate_traffic_potential_score(
                search_volume=volume,
                keyword_difficulty=difficulty
            )
            traffic_scores.append(traffic_score)

            # Ranking opportunity score
            position = row.get(position_col) if position_col in df.columns else None
            clicks = row.get(clicks_col) if clicks_col in df.columns else None
            impressions = row.get(impressions_col) if impressions_col in df.columns else None

            rank_score = self.calculate_ranking_opportunity_score(
                current_position=position,
                clicks=clicks,
                impressions=impressions
            )
            ranking_scores.append(rank_score)

            # SEO Value Score
            seo_score = self.calculate_seo_value_score(
                relevancy_score=rel_score,
                traffic_score=traffic_score,
                ranking_score=rank_score
            )
            seo_value_scores.append(seo_score)

        # Add columns to dataframe
        df['relevancy_score'] = relevancy_scores
        df['traffic_potential_score'] = traffic_scores
        df['ranking_opportunity_score'] = ranking_scores
        df['seo_value_score'] = seo_value_scores

        # Sort by SEO Value Score (highest first)
        df = df.sort_values('seo_value_score', ascending=False).reset_index(drop=True)

        return df

    def get_top_opportunities(
        self,
        df: pd.DataFrame,
        n: int = 20,
        min_seo_score: float = 60.0
    ) -> pd.DataFrame:
        """Get top N keyword opportunities.

        Args:
            df: Dataframe with prioritization scores
            n: Number of top opportunities to return
            min_seo_score: Minimum SEO Value Score threshold

        Returns:
            Top N opportunities
        """
        if 'seo_value_score' not in df.columns:
            raise ValueError("Dataframe must have 'seo_value_score' column. Run prioritize_dataframe first.")

        # Filter by minimum score
        filtered_df = df[df['seo_value_score'] >= min_seo_score].copy()

        # Return top N
        return filtered_df.head(n)


# Convenience function
def add_prioritization_scores(
    df: pd.DataFrame,
    relevancy_weight: float = 0.40,
    traffic_weight: float = 0.35,
    ranking_weight: float = 0.25
) -> pd.DataFrame:
    """Quick function to add prioritization scores to a dataframe.

    Args:
        df: Input dataframe
        relevancy_weight: Weight for relevancy (default: 0.40)
        traffic_weight: Weight for traffic potential (default: 0.35)
        ranking_weight: Weight for ranking opportunity (default: 0.25)

    Returns:
        Dataframe with prioritization scores
    """
    engine = PrioritizationEngine(
        relevancy_weight=relevancy_weight,
        traffic_weight=traffic_weight,
        ranking_weight=ranking_weight
    )

    return engine.prioritize_dataframe(df)
