import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def dataset_summary(df: pd.DataFrame) -> dict:
    """Basic stats for the first-three-column rating table."""
    users = df.iloc[:, 0].nunique()
    items = df.iloc[:, 1].nunique()
    n = len(df)
    cells = users * items if users and items else 0
    sparsity = 1.0 - (n / cells) if cells else 0.0
    return {
        "n_ratings": n,
        "n_users": int(users),
        "n_items": int(items),
        "sparsity": float(sparsity),
    }


class RecommenderEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df.columns = ["user", "item", "rating"]
        self.pivot = self.df.pivot_table(index="user", columns="item", values="rating")

    def _user_similarity_df(self) -> pd.DataFrame:
        sim_matrix = cosine_similarity(self.pivot.fillna(0))
        return pd.DataFrame(sim_matrix, index=self.pivot.index, columns=self.pivot.index)

    def _item_similarity_df(self) -> pd.DataFrame:
        sim_matrix = cosine_similarity(self.pivot.T.fillna(0))
        return pd.DataFrame(sim_matrix, index=self.pivot.columns, columns=self.pivot.columns)

    def user_based_scores_raw(self, target_user) -> pd.Series:
        """All item scores for user-based CF (including items already rated)."""
        if target_user not in self.pivot.index:
            return pd.Series(dtype=float)
        user_sim_df = self._user_similarity_df()
        similar_users = user_sim_df[target_user].sort_values(ascending=False)[1:11]
        if similar_users.empty:
            return pd.Series(dtype=float)
        neighbor_ratings = self.pivot.loc[similar_users.index].fillna(0)
        weights = similar_users.values
        scores = np.dot(weights, neighbor_ratings)
        return pd.Series(scores, index=self.pivot.columns)

    def get_user_based(self, target_user, n=5) -> pd.Series:
        """User-based CF: neighbors who rate similarly, weighted sum of their ratings."""
        scores = self.user_based_scores_raw(target_user)
        if scores.empty:
            return scores
        already_rated = self.pivot.loc[target_user].dropna().index
        return scores.drop(already_rated, errors="ignore").sort_values(ascending=False).head(n)

    def get_similar_users(self, target_user, k: int = 10) -> pd.DataFrame:
        """Top-k neighbors by cosine similarity (excludes self)."""
        if target_user not in self.pivot.index:
            return pd.DataFrame(columns=["neighbor", "cosine_similarity"])
        user_sim_df = self._user_similarity_df()
        sims = user_sim_df[target_user].drop(target_user, errors="ignore").sort_values(ascending=False).head(k)
        return pd.DataFrame({"neighbor": sims.index, "cosine_similarity": sims.values})

    def get_item_based(self, target_user, n=5) -> pd.Series:
        """Item-based CF: similarity-weighted sum over the user's rated items."""
        if target_user not in self.pivot.index:
            return pd.Series(dtype=float)
        user_history = self.pivot.loc[target_user].dropna()
        if user_history.empty:
            return pd.Series(dtype=float)
        item_sim_df = self._item_similarity_df()
        scores = item_sim_df[user_history.index].dot(user_history)
        return scores.drop(user_history.index, errors="ignore").sort_values(ascending=False).head(n)

    def item_based_contributions(self, target_user, recommended_item, top_k: int = 5) -> pd.DataFrame:
        """Which rated items contributed most to this recommendation score."""
        if target_user not in self.pivot.index or recommended_item not in self.pivot.columns:
            return pd.DataFrame(columns=["rated_item", "contribution"])
        user_history = self.pivot.loc[target_user].dropna()
        if user_history.empty:
            return pd.DataFrame(columns=["rated_item", "contribution"])
        item_sim_df = self._item_similarity_df()
        if recommended_item not in item_sim_df.index:
            return pd.DataFrame(columns=["rated_item", "contribution"])
        contribs = item_sim_df.loc[recommended_item, user_history.index] * user_history
        contribs = contribs.sort_values(ascending=False).head(top_k)
        return pd.DataFrame({"rated_item": contribs.index, "contribution": contribs.values})

    def get_item_cooccurrence(self, target_user, n=5) -> pd.Series:
        """
        Binary co-occurrence: sum over the user's rated items of how often other items
        appear together with those items in any user's profile (not Apriori / lift).
        """
        if target_user not in self.pivot.index:
            return pd.Series(dtype=float)
        basket = self.pivot.notna().astype(int)
        user_items = basket.loc[target_user]
        items_bought = user_items[user_items == 1].index
        if items_bought.empty:
            return pd.Series(dtype=float)
        co_matrix = basket.T.dot(basket)
        scores = co_matrix[items_bought].sum(axis=1)
        return scores.drop(items_bought, errors="ignore").sort_values(ascending=False).head(n)

    def get_market_basket(self, target_user, n=5) -> pd.Series:
        """Backward-compatible alias for get_item_cooccurrence."""
        return self.get_item_cooccurrence(target_user, n=n)

    def cooccurrence_breakdown(self, target_user, recommended_item, top_k: int = 5) -> pd.DataFrame:
        """Co-occurrence counts of recommended_item with each of the user's rated items."""
        if target_user not in self.pivot.index or recommended_item not in self.pivot.columns:
            return pd.DataFrame(columns=["rated_item", "cooccurrence_count"])
        basket = self.pivot.notna().astype(int)
        user_items = basket.loc[target_user]
        rated = user_items[user_items == 1].index
        if rated.empty:
            return pd.DataFrame(columns=["rated_item", "cooccurrence_count"])
        co_matrix = basket.T.dot(basket)
        if recommended_item not in co_matrix.index:
            return pd.DataFrame(columns=["rated_item", "cooccurrence_count"])
        counts = co_matrix.loc[recommended_item, rated].sort_values(ascending=False).head(top_k)
        return pd.DataFrame({"rated_item": counts.index, "cooccurrence_count": counts.values})

    @staticmethod
    def evaluate_user_based_holdout(
        df: pd.DataFrame,
        random_state: int = 42,
        max_tests: int = 100,
        test_fraction: float = 0.2,
    ) -> tuple[float | None, float | None, int, str | None]:
        """
        Hold-out random interactions: rebuild engine without each test row and predict
        that item's user-based score. Returns (rmse, mae, n_predictions, error_message).
        """
        work = df.copy()
        work.columns = ["user", "item", "rating"]
        work = work.reset_index(drop=True)
        n = len(work)
        if n < 8:
            return None, None, 0, "Need at least 8 ratings for a stable hold-out check."
        rng = np.random.default_rng(random_state)
        n_test = min(max_tests, max(1, int(n * test_fraction)))
        positions = rng.choice(n, size=n_test, replace=False)

        sq_err = []
        abs_err = []
        for pos in positions:
            row = work.iloc[int(pos)]
            u, it, r = row["user"], row["item"], float(row["rating"])
            train = work.drop(index=work.index[int(pos)])
            if u not in train["user"].values or it not in train["item"].values:
                continue
            try:
                eng = RecommenderEngine(train)
            except Exception:
                continue
            scores = eng.user_based_scores_raw(u)
            if it not in scores.index or pd.isna(scores.loc[it]):
                continue
            pred = float(scores.loc[it])
            sq_err.append((pred - r) ** 2)
            abs_err.append(abs(pred - r))

        if not sq_err:
            return None, None, 0, "No overlapping hold-out predictions (try more data)."

        rmse = float(np.sqrt(np.mean(sq_err)))
        mae = float(np.mean(abs_err))
        return rmse, mae, len(sq_err), None
