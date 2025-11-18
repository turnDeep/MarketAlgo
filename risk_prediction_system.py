"""
Hybrid ML Ensemble for Short-Horizon Market Risk Prediction
Based on: Ranjan (2025) - Causal and Predictive Modeling
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# import yfinance as yf
import pandas_datareader as pdr
import time
# from curl_cffi import requests as curl_requests
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import skew, kurtosis
from scipy.special import rel_entr
import xgboost as xgb
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ===============================
# 1. データ取得とロード
# ===============================

class DataLoader_Market:
    """Market data loader for cross-asset universe"""

    def __init__(self, start_date: str = '2005-01-01', end_date: str = '2025-01-01', session=None):
        self.start_date = start_date
        self.end_date = end_date
        self.session = session

        # Investment Universe (Table 1 from paper)
        self.universe = {
            'equities': ['SPY', 'QQQ', 'IWM', 'TLT'],
            'volatility': ['^VIX'],
            'commodities': ['GLD', 'CL=F'],
            'fx': ['DX-Y.NYB', 'EURUSD=X', 'JPYUSD=X'],
            'treasuries': ['^TNX', '^IRX']
        }

        self.all_symbols = [sym for category in self.universe.values() for sym in category]

    def fetch_data(self) -> pd.DataFrame:
        """Fetch historical price data"""
        print("Fetching market data...")
        data_dict = {}

        for symbol in self.all_symbols:
            try:
                df = pdr.get_data_yahoo(symbol, start=self.start_date, end=self.end_date)
                data_dict[symbol] = df['Close']
                print(f"✓ {symbol}")
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                print(f"✗ {symbol}: {e}")

        prices = pd.DataFrame(data_dict)
        prices = prices.ffill().bfill() # Forward fill then backward fill

        return prices

    def calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Calculate logarithmic returns (Equation 1)"""
        returns = np.log(prices / prices.shift(1))
        return returns.iloc[1:] # Drop first NaN row

# ===============================
# 2. 特徴量エンジニアリング
# ===============================

class FeatureEngine:
    """Feature engineering pipeline following paper specifications"""

    def __init__(self, returns: pd.DataFrame):
        self.returns = returns
        self.features = pd.DataFrame(index=returns.index)

    def compute_all_features(self) -> pd.DataFrame:
        """Compute all 178 engineered features"""
        print("\nEngineering features...")

        # 1. Time-Series Moments (Equations 3-6)
        self._add_statistical_features(windows=[21, 63])

        # 2. Hurst Exponent
        self._add_hurst_exponent(scales=[16, 64, 256])

        # 3. Cross-Asset Relations (Equations 7-8)
        self._add_cross_asset_features(windows=[21, 63])

        # 4. Information-Theoretic Measures (Equation 9)
        self._add_kl_divergence(current_window=21, reference_window=126)

        # Remove NaN rows from rolling calculations
        self.features = self.features.dropna()

        print(f"Created {len(self.features.columns)} features")
        print(f"Sample size: {len(self.features)} observations")

        return self.features

    def _add_statistical_features(self, windows: List[int]):
        """Add volatility, skewness, kurtosis, entropy"""
        for asset in self.returns.columns:
            for w in windows:
                # Volatility (Equation 3)
                self.features[f'{asset}_vol_{w}d'] = self.returns[asset].rolling(w).std()

                # Skewness (Equation 4)
                self.features[f'{asset}_skew_{w}d'] = self.returns[asset].rolling(w).apply(
                    lambda x: skew(x, nan_policy='omit'), raw=False
                )

                # Kurtosis (Equation 5)
                self.features[f'{asset}_kurt_{w}d'] = self.returns[asset].rolling(w).apply(
                    lambda x: kurtosis(x, nan_policy='omit'), raw=False
                )

                # Shannon Entropy (Equation 6)
                self.features[f'{asset}_entropy_{w}d'] = self.returns[asset].rolling(w).apply(
                    self._shannon_entropy, raw=False
                )

    def _shannon_entropy(self, x: pd.Series, bins: int = 30) -> float:
        """Calculate Shannon entropy"""
        if len(x) < 2:
            return np.nan
        counts, _ = np.histogram(x, bins=bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0] # Remove zeros
        return -np.sum(probs * np.log(probs))

    def _add_hurst_exponent(self, scales: List[int]):
        """Compute Hurst exponent using R/S analysis"""
        scale_names = {16: 'short', 64: 'medium', 256: 'long'}

        for asset in self.returns.columns:
            for scale in scales:
                self.features[f'{asset}_hurst_{scale_names[scale]}'] = (
                    self.returns[asset].rolling(scale).apply(
                        self._calculate_hurst, raw=False
                    )
                )

    def _calculate_hurst(self, series: pd.Series) -> float:
        """Calculate Hurst exponent via rescaled range"""
        if len(series) < 10:
            return np.nan

        series = np.array(series)
        n = len(series)

        # Mean-centered cumulative sum
        mean_centered = series - np.mean(series)
        cumsum = np.cumsum(mean_centered)

        # Range
        R = np.max(cumsum) - np.min(cumsum)

        # Standard deviation
        S = np.std(series, ddof=1)

        if S == 0 or R == 0:
            return np.nan

        # R/S ratio
        rs = R / S

        # Hurst exponent
        hurst = np.log(rs) / np.log(n)

        return hurst

    def _add_cross_asset_features(self, windows: List[int]):
        """Add beta and correlation with SPY (Equations 7-8)"""
        if 'SPY' not in self.returns.columns:
            return

        spy_returns = self.returns['SPY']

        for asset in self.returns.columns:
            if asset == 'SPY':
                continue

            for w in windows:
                # Rolling Beta (Equation 7)
                self.features[f'{asset}_beta_SPY_{w}d'] = (
                    self.returns[asset].rolling(w).cov(spy_returns) /
                    spy_returns.rolling(w).var()
                )

                # Rolling Correlation (Equation 8)
                self.features[f'{asset}_corr_SPY_{w}d'] = (
                    self.returns[asset].rolling(w).corr(spy_returns)
                )

    def _add_kl_divergence(self, current_window: int, reference_window: int):
        """Add KL divergence (Equation 9)"""
        bins = 30

        for asset in self.returns.columns:
            kl_values = []

            for i in range(len(self.returns)):
                if i < reference_window:
                    kl_values.append(np.nan)
                    continue

                # Current distribution
                current = self.returns[asset].iloc[i-current_window:i]
                # Reference distribution
                reference = self.returns[asset].iloc[i-reference_window:i-current_window]

                # Histogram
                p_curr, _ = np.histogram(current, bins=bins, density=True)
                p_ref, _ = np.histogram(reference, bins=bins, density=True)

                # Add small constant to avoid log(0)
                p_curr = p_curr + 1e-10
                p_ref = p_ref + 1e-10

                # KL divergence
                kl = np.sum(rel_entr(p_curr, p_ref))
                kl_values.append(kl)

            self.features[f'{asset}_kl_{current_window}_{reference_window}'] = kl_values

# ===============================
# 3. ターゲット変数生成
# ===============================

class TargetGenerator:
    """Generate binary crash target (Equation 2)"""

    def __init__(self, returns: pd.DataFrame, horizon: int = 5, threshold: float = 0.01):
        self.returns = returns
        self.horizon = horizon # h=5 days
        self.threshold = threshold # δ=1%

    def generate_target(self) -> pd.Series:
        """
        y_t = 1 if sum(r_SPY,t+k for k=1 to h) <= -δ
        """
        if 'SPY' not in self.returns.columns:
            raise ValueError("SPY returns required for target generation")

        spy_returns = self.returns['SPY']

        # Forward-looking cumulative return
        forward_returns = spy_returns.rolling(self.horizon).sum().shift(-self.horizon)

        # Binary target: 1 if crash, 0 otherwise
        target = (forward_returns <= -self.threshold).astype(int)

        print(f"\nTarget statistics:")
        print(f"Total samples: {len(target)}")
        print(f"Crash weeks (1): {target.sum()} ({100*target.mean():.1f}%)")
        print(f"Non-crash weeks (0): {(1-target).sum()} ({100*(1-target.mean()):.1f}%)")

        return target

# ===============================
# 4. 特徴選択
# ===============================

class FeatureSelector:
    """Feature selection via mutual information (Section 2.2)"""

    def __init__(self, top_k: int = 80):
        self.top_k = top_k
        self.selected_features = None
        self.mi_scores = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        1. Remove low variance features
        2. Remove highly correlated features
        3. Select top k by mutual information
        """
        print("\nFeature selection...")

        # 1. Low variance removal (< 1e-4)
        variances = X.var()
        high_var = variances[variances >= 1e-4].index
        X_filtered = X[high_var]
        print(f"After variance filter: {len(X_filtered.columns)} features")

        # 2. High correlation removal (>= 0.95)
        corr_matrix = X_filtered.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] >= 0.95)]
        X_filtered = X_filtered.drop(columns=to_drop)
        print(f"After correlation filter: {len(X_filtered.columns)} features")

        # 3. Mutual information ranking (Equation 10)
        # Align X and y indices
        common_index = X_filtered.index.intersection(y.index)
        X_aligned = X_filtered.loc[common_index]
        y_aligned = y.loc[common_index]

        self.mi_scores = mutual_info_classif(
            X_aligned, y_aligned,
            random_state=42,
            n_neighbors=5
        )

        mi_df = pd.DataFrame({
            'feature': X_aligned.columns,
            'mi_score': self.mi_scores
        }).sort_values('mi_score', ascending=False)

        # Select top k
        self.selected_features = mi_df.head(self.top_k)['feature'].tolist()

        print(f"Selected top {self.top_k} features")
        print(f"\nTop 13 features (Table 2):")
        for i, row in mi_df.head(13).iterrows():
            print(f" {row['feature']}: {row['mi_score']:.6f}")

        # Visualization (Figure 1)
        self._plot_mi_scores(mi_df)

        return X_filtered[self.selected_features]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature selection"""
        return X[self.selected_features]

    def _plot_mi_scores(self, mi_df: pd.DataFrame):
        """Plot MI scores (Figure 1)"""
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(mi_df)), mi_df['mi_score'].values)
        plt.xlabel('Feature rank')
        plt.ylabel('Mutual Information')
        plt.title('Mutual Information Scores (Top-to-Bottom Order)')
        plt.tight_layout()
        plt.savefig('mi_scores.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: mi_scores.png")

# ===============================
# 5. PyTorch Dataset
# ===============================

class MarketDataset(Dataset):
    """PyTorch dataset for time series"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ===============================
# 6. Multi-Layer Perceptron (MLP)
# ===============================

class MLP_Classifier(nn.Module):
    """Shallow MLP for temporal dependencies (Section 2.3.1)"""

    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64], dropout: float = 0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 2)) # Binary classification

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class MLPTrainer:
    """MLP training wrapper"""

    def __init__(self, input_dim: int, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = MLP_Classifier(input_dim).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray,
            epochs: int = 50, batch_size: int = 128):
        """Train MLP"""

        train_dataset = MarketDataset(X_train, y_train)
        val_dataset = MarketDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs, y_batch)
                    val_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions"""
        self.model.eval()
        dataset = MarketDataset(X, np.zeros(len(X))) # Dummy labels
        loader = DataLoader(dataset, batch_size=256)

        probs = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                probs.append(torch.softmax(outputs, dim=1).cpu().numpy())

        return np.vstack(probs)

# ===============================
# 7. Hybrid Ensemble
# ===============================

class HybridEnsemble:
    """Soft voting ensemble: MLP + XGBoost + CatBoost (Section 2.3)"""

    def __init__(self, input_dim: int):
        self.models = {}
        self.input_dim = input_dim

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray):
        """Train all base learners"""

        print("\n" + "="*50)
        print("Training Hybrid Ensemble")
        print("="*50)

        # 1. MLP
        print("\n[1/3] Training MLP...")
        self.models['mlp'] = MLPTrainer(self.input_dim)
        self.models['mlp'].fit(X_train, y_train, X_val, y_val, epochs=50)

        # 2. XGBoost
        print("\n[2/3] Training XGBoost...")
        self.models['xgb'] = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        self.models['xgb'].fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # 3. CatBoost
        print("\n[3/3] Training CatBoost...")
        self.models['catboost'] = CatBoostClassifier(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=False
        )
        self.models['catboost'].fit(X_train, y_train, eval_set=(X_val, y_val))

        print("\n✓ Ensemble training complete")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Soft voting: average probabilities (Section 2.3.3)"""

        # Get probabilities from each model
        mlp_probs = self.models['mlp'].predict_proba(X)
        xgb_probs = self.models['xgb'].predict_proba(X)
        catboost_probs = self.models['catboost'].predict_proba(X)

        # Average (soft voting)
        ensemble_probs = (mlp_probs + xgb_probs + catboost_probs) / 3

        return ensemble_probs

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Binary predictions"""
        probs = self.predict_proba(X)
        return (probs[:, 1] >= threshold).astype(int)

# ===============================
# 8. バックテストシステム
# ===============================

class BacktestEngine:
    """Backtesting framework for trading strategy (Section 4)"""

    def __init__(self, returns: pd.DataFrame, predictions: np.ndarray,
                 probabilities: np.ndarray, dates: pd.DatetimeIndex):
        self.returns = returns['SPY'] # SPY returns for P&L
        self.predictions = predictions
        self.probabilities = probabilities
        self.dates = dates

        # Align all data
        common_index = self.returns.index.intersection(dates)
        self.returns = self.returns.loc[common_index]

    def run_backtest(self, threshold: float = 0.5) -> Dict:
        """
        Execute long/short strategy:
        - Long if P(crash) < threshold
        - Short if P(crash) >= threshold
        """

        # Generate signals
        positions = np.where(self.probabilities[:, 1] < threshold, 1, -1) # 1=long, -1=short

        # Calculate strategy returns
        strategy_returns = positions * self.returns.values

        # Cumulative returns
        cum_returns_spy = (1 + self.returns).cumprod()
        cum_returns_strategy = (1 + pd.Series(strategy_returns, index=self.returns.index)).cumprod()

        # Performance metrics
        metrics = self._calculate_metrics(strategy_returns, self.returns.values)

        # Visualization
        self._plot_backtest_results(cum_returns_spy, cum_returns_strategy, positions)
        self._plot_return_distribution(strategy_returns, self.returns.values)

        return metrics

    def _calculate_metrics(self, strategy_returns: np.ndarray,
                           spy_returns: np.ndarray) -> Dict:
        """Calculate performance metrics (Table 4)"""

        # Annualization factor
        trading_days = 252

        # Returns
        total_return_strategy = (1 + strategy_returns).prod() - 1
        total_return_spy = (1 + spy_returns).prod() - 1

        years = len(strategy_returns) / trading_days
        ann_return_strategy = (1 + total_return_strategy) ** (1/years) - 1
        ann_return_spy = (1 + total_return_spy) ** (1/years) - 1

        # Volatility
        ann_vol_strategy = np.std(strategy_returns) * np.sqrt(trading_days)
        ann_vol_spy = np.std(spy_returns) * np.sqrt(trading_days)

        # Sharpe Ratio (assuming rf=0)
        sharpe_strategy = ann_return_strategy / ann_vol_strategy
        sharpe_spy = ann_return_spy / ann_vol_spy

        # Information Ratio
        active_return = strategy_returns - spy_returns
        information_ratio = (np.mean(active_return) / np.std(active_return)) * np.sqrt(trading_days)

        # Maximum Drawdown
        cum_returns = (1 + strategy_returns).cumprod()
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # CAPM Alpha and Beta
        covariance = np.cov(strategy_returns, spy_returns)[0, 1]
        variance_spy = np.var(spy_returns)
        beta = covariance / variance_spy

        alpha_daily = np.mean(strategy_returns) - beta * np.mean(spy_returns)
        alpha_annual = alpha_daily * trading_days

        # T-stat for alpha
        residuals = strategy_returns - (alpha_daily + beta * spy_returns)
        se_alpha = np.std(residuals) / np.sqrt(len(strategy_returns))
        t_stat = alpha_daily / se_alpha

        metrics = {
            'Sharpe Ratio': sharpe_strategy,
            'Information Ratio vs SPY': information_ratio,
            'Maximum Drawdown': max_drawdown,
            'Annualized Return': ann_return_strategy,
            'Annualized Volatility': ann_vol_strategy,
            'CAPM Alpha (daily)': alpha_daily,
            'CAPM Alpha (annual)': alpha_annual,
            'CAPM Beta': beta,
            'T-stat Alpha': t_stat
        }

        # Print results (Table 4 format)
        print("\n" + "="*50)
        print("BACKTEST PERFORMANCE METRICS")
        print("="*50)
        for key, value in metrics.items():
            if 'Ratio' in key or 'Beta' in key or 'T-stat' in key:
                print(f"{key:30s}: {value:.2f}")
            elif 'Alpha' in key:
                print(f"{key:30s}: {value:.5f}")
            else:
                print(f"{key:30s}: {value:.2%}")

        return metrics

    def _plot_backtest_results(self, cum_spy: pd.Series, cum_strategy: pd.Series,
                               positions: np.ndarray):
        """Plot cumulative returns with positions (Figure 7)"""

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                        gridspec_kw={'height_ratios': [3, 1]})

        # Cumulative returns
        ax1.plot(cum_spy.index, cum_spy.values, label='SPY (Buy & Hold)',
                 linewidth=2, alpha=0.7)
        ax1.plot(cum_strategy.index, cum_strategy.values, label='Strategy',
                 linewidth=2, alpha=0.7)
        ax1.set_ylabel('Cumulative Return')
        ax1.set_title('Backtest: Strategy vs SPY')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Positions
        position_series = pd.Series(positions, index=self.returns.index)
        colors = ['red' if p == -1 else 'green' for p in positions]
        ax2.scatter(position_series.index, position_series.values,
                    c=colors, alpha=0.5, s=10)
        ax2.set_ylabel('Position')
        ax2.set_yticks([-1, 1])
        ax2.set_yticklabels(['Short', 'Long'])
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('backtest_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("\nSaved: backtest_results.png")

    def _plot_return_distribution(self, strategy_returns: np.ndarray,
                                  spy_returns: np.ndarray):
        """Plot return distributions (Figure 8)"""

        plt.figure(figsize=(10, 6))
        plt.hist(spy_returns, bins=100, alpha=0.5, label='SPY', density=True, color='gray')
        plt.hist(strategy_returns, bins=100, alpha=0.5, label='Strategy',
                 density=True, color='green')
        plt.xlabel('Daily Return')
        plt.ylabel('Density')
        plt.title('Distribution of Daily Returns: Strategy vs SPY')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('return_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: return_distribution.png")

# ===============================
# 9. メイン実行
# ===============================

def main():
    """Main execution pipeline"""

    print("="*70)
    print("HYBRID ML ENSEMBLE FOR SHORT-HORIZON MARKET RISK PREDICTION")
    print("Based on: Ranjan (2025)")
    print("="*70)

    # 1. データ取得
    # Use standard yfinance session
    # session = curl_requests.Session(impersonate="chrome110")
    loader = DataLoader_Market(start_date='2015-11-18', end_date='2025-11-18', session=None)
    prices = loader.fetch_data()
    returns = loader.calculate_returns(prices)

    # 2. 特徴量エンジニアリング
    feature_engine = FeatureEngine(returns)
    features = feature_engine.compute_all_features()

    # 3. ターゲット生成
    target_gen = TargetGenerator(returns, horizon=5, threshold=0.01)
    target = target_gen.generate_target()

    # 4. データ整合性確保
    common_index = features.index.intersection(target.index)
    features = features.loc[common_index]
    target = target.loc[common_index]

    print(f"\nFinal dataset: {len(features)} samples, {len(features.columns)} features")

    # 5. 特徴選択
    selector = FeatureSelector(top_k=80)
    X_selected = selector.fit(features, target)

    # 6. Train/Test split (時系列を考慮)
    split_idx = int(0.8 * len(X_selected))

    X_train = X_selected.iloc[:split_idx]
    X_test = X_selected.iloc[split_idx:]
    y_train = target.iloc[:split_idx]
    y_test = target.iloc[split_idx:]

    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Validation split from training
    val_split = int(0.9 * len(X_train_scaled))
    X_train_final = X_train_scaled[:val_split]
    X_val = X_train_scaled[val_split:]
    y_train_final = y_train.values[:val_split]
    y_val = y_train.values[val_split:]

    print(f"\nTrain: {len(X_train_final)}, Val: {len(X_val)}, Test: {len(X_test_scaled)}")

    # 7. モデル訓練
    ensemble = HybridEnsemble(input_dim=X_train_final.shape[1])
    ensemble.fit(X_train_final, y_train_final, X_val, y_val)

    # 8. 予測
    print("\n" + "="*50)
    print("GENERATING PREDICTIONS")
    print("="*50)

    # Full dataset predictions for backtest
    X_full_scaled = scaler.transform(X_selected)
    y_pred_proba_full = ensemble.predict_proba(X_full_scaled)
    y_pred_full = ensemble.predict(X_full_scaled)

    # Test set evaluation
    y_pred_proba = ensemble.predict_proba(X_test_scaled)
    y_pred = ensemble.predict(X_test_scaled)

    # 9. 分類性能評価
    print("\n" + "="*50)
    print("CLASSIFICATION PERFORMANCE (Test Set)")
    print("="*50)

    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    print(f"ROC-AUC: {roc_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Non-Crash', 'Crash'],
                                digits=2))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Test Set)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: confusion_matrix.png")

    # 10. バックテスト
    print("\n" + "="*50)
    print("RUNNING BACKTEST")
    print("="*50)

    backtest = BacktestEngine(
        returns=returns,
        predictions=y_pred_full,
        probabilities=y_pred_proba_full,
        dates=X_selected.index
    )

    metrics = backtest.run_backtest(threshold=0.5)

    # 11. リスク分位分析 (Figure 5, 6)
    analyze_risk_quantiles(returns, y_pred_proba_full, X_selected.index)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print(" - mi_scores.png")
    print(" - confusion_matrix.png")
    print(" - backtest_results.png")
    print(" - return_distribution.png")
    print(" - risk_quantiles.png")

def analyze_risk_quantiles(returns: pd.DataFrame, probabilities: np.ndarray,
                           dates: pd.DatetimeIndex):
    """Analyze returns by predicted risk quintiles (Section 3.3)"""

    # Align data
    spy_returns = returns['SPY'].loc[dates]
    crash_probs = probabilities[:, 1]

    # Quintiles
    quintiles = pd.qcut(crash_probs, q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

    # Average returns by quintile
    df = pd.DataFrame({
        'return': spy_returns.values,
        'quintile': quintiles
    })

    avg_returns = df.groupby('quintile')['return'].mean()

    # Plot (Figure 5)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Average returns
    ax1.bar(range(5), avg_returns.values, color='salmon')
    ax1.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax1.set_xlabel('Predicted Risk Quintile (0=low, 4=high)')
    ax1.set_ylabel('Avg SPY 5-day Return')
    ax1.set_title('Average SPY Return by Predicted Risk Quantile')
    ax1.set_xticks(range(5))
    ax1.set_xticklabels(['0', '1', '2', '3', '4'])
    ax1.grid(True, alpha=0.3)

    # Distribution: High vs Low risk (Figure 6)
    high_risk = df[df['quintile'] == 'Q5']['return']
    low_risk = df[df['quintile'].isin(['Q1', 'Q2', 'Q3', 'Q4'])]['return']

    ax2.hist(low_risk, bins=50, alpha=0.7, label='Low Risk', color='green', density=True)
    ax2.hist(high_risk, bins=50, alpha=0.7, label='High Risk', color='red', density=True)
    ax2.set_xlabel('5-Day SPY Return')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Forward 5-Day SPY Returns During High vs Low Risk Periods')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('risk_quantiles.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nSaved: risk_quantiles.png")

    # Statistics
    print(f"\nRisk Quintile Analysis:")
    print(f"Average 5-day SPY return during HIGH-risk periods (Q5): {high_risk.mean():.2%}")
    print(f"Average 5-day SPY return during LOW-risk periods (Q1-Q4): {low_risk.mean():.2%}")

if __name__ == '__main__':
    main()
