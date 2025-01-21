from typing import List, Tuple, Dict
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy import stats
from sqlalchemy.orm import Session
from app.database import Prediction

@dataclass
class ChangePoint:
    timestamp: datetime
    confidence: float
    metric: str
    before_params: Dict
    after_params: Dict

class BayesianDriftDetector:
    """Bayesian drift detection for sentiment analysis models"""

    def __init__(self,
                 window_size: timedelta = timedelta(hours=24),
                 min_probability: float = 0.95):
        self.window_size = window_size
        self.min_probability = min_probability

    def detect_sentiment_drift(self,
                               predictions: List[Prediction]) -> List[ChangePoint]:
        """
        Detect changes in sentiment probability distribution using
        Bayesian inference with Beta conjugate priors
        """
        if not predictions:
            return []
        
        #sort predictions by timestamp
        sorted_preds = sorted(predictions, key=lambda x: x.created_at)

        #init changepoints list
        changepoints: List[ChangePoint] = []

        #sliding window analysis
        window_start = sorted_preds[0].created_at
        while window_start < sorted_preds[-1].created_at:
            window_end = window_start + self.window_size

            #get predictions in current window
            window_preds = [p for p in sorted_preds
                            if window_start <= p.created_at < window_end]
            
            if len(window_preds) < 50: #check for minimum sample size
                window_start += self.window_size
                continue

            #calculate prior parameters (Beta distribution)
            prior_positive = sum(1 for p in window_preds
                                 if p.sentiment == 'POSITIVE')
            prior_negative = len(window_preds) - prior_positive

            #prior parameters (Beta(alpha, beta))
            alpha_prior = prior_positive + 1 #add 1 for Laplace smoothing
            beta_prior = prior_negative + 1

            #check next window
            next_window_end = window_end + self.window_size
            next_window_preds = [p for p in sorted_preds
                                 if window_end <= p.created_at < next_window_end]
            
            if len(next_window_preds) < 50: #check for minimum sample size
                window_start += self.window_size
                continue

            #calculate posterior parameters
            post_positive = sum(1 for p in next_window_preds
                                if p.sentiment == 'POSITIVE')
            post_negative = len(next_window_preds) - post_positive

            #posterior parameters
            alpha_post = post_positive + 1 #add 1 for Laplace smoothing
            beta_post = post_negative + 1

            #calculate Bayes factor using Beta distributions
            prior_dist = stats.beta(alpha_prior, beta_prior)
            post_dist = stats.beta(alpha_post, beta_post)

            #calculate probability of change
            change_prob = self.calculate_change_probability(prior_dist, post_dist)

            if change_prob >= self.min_probability:
                changepoints.append(
                    ChangePoint(
                        timestamp=window_end,
                        confidence=change_prob,
                        metric="sentiment_distribution",
                        before_params={
                            "alpha": alpha_prior,
                            "beta": beta_prior,
                            "positive_rate": prior_positive/len(window_preds),
                        },
                        after_params={
                            "alpha": alpha_post,
                            "beta": beta_post,
                            "positive_rate": post_positive/len(next_window_preds)
                        }
                    )
                )
            window_start += self.window_size
        return changepoints

    def calculate_change_probability(self,
                                     prior_dist: stats.beta,
                                     post_dist: stats.beta,
                                     n_samples: int = 10000) -> float:
        """Calculate probability distribution using Monte Carlo"""
        prior_samples = prior_dist.rvs(n_samples)
        post_samples = post_dist.rvs(n_samples)
        return np.mean(post_samples > prior_samples)

    def detect_confidence_drift(self, 
                                predictions: List[Prediction]) -> List[ChangePoint]:
        """
        Detect changes in confidence score distribution using
        Bayesian inference with Normal-Gamma conjugate prior
        """

        if not predictions:
            return []
        
        sorted_preds = sorted(predictions, key=lambda x: x.created_at)
        changepoints: List[ChangePoint] = []

        #sliding window analysis
        window_start = sorted_preds[0].created_at
        while window_start < sorted_preds[-1].created_at:
            window_end = window_start + self.window_size

            #get predictions in current window
            window_preds = [p for p in sorted_preds
                            if window_start <= p.created_at < window_end]
            
            if len(window_preds) < 50: #check for minimum sample size
                window_start += self.window_size
                continue
            
            #calculate statistics for current window
            conf_scores = [p.confidence for p in window_preds]
            prior_mean = np.mean(conf_scores)
            prior_var = np.var(conf_scores)

            #check next window
            next_window_end = window_end + self.window_size
            next_window_preds = [p for p in sorted_preds
                                 if window_end <= p.created_at < next_window_end]
            
            if len(next_window_preds) < 50: #check for minimum sample size
                window_start += self.window_size
                continue

            #calculate statistics for next window
            next_conf_scores = [p.confidence for p in next_window_preds]
            post_mean = np.mean(next_conf_scores)
            post_var = np.var(next_conf_scores)

            #use normal distribution approximation for large samples
            change_prob = self.calculate_normal_change_probability(
                prior_mean, prior_var, len(conf_scores),
                post_mean, post_var, len(next_conf_scores)
            )

            if change_prob >= self.min_probability:
                changepoints.append(
                    ChangePoint(
                        timestamp=window_end,
                        confidence=change_prob,
                        metric="confidence_distribution",
                        before_params={
                            "mean": prior_mean,
                            "variance": prior_var,
                            "n_samples": len(conf_scores)
                        },
                        after_params={
                            "mean": post_mean,
                            "variance": post_var,
                            "n_samples": len(next_conf_scores)
                        }
                    )
                )

            window_start += self.window_size
        return changepoints
    
    def calculate_normal_change_probability(self,
                                            prior_mean: float,
                                            prior_var: float,
                                            prior_n: int,
                                            post_mean: float,
                                            post_var: float,
                                            post_n: int) -> float:
        """Calculate probability of change for normal distributions"""
        #use Welch's t-test for unequal variances
        t_stat, p_value = stats.ttest_ind_from_stats(
            prior_mean, np.sqrt(prior_var), prior_n,
            post_mean, np.sqrt(post_var), post_n, equal_var=False
        )
        return 1 - p_value #convert p-value to probability of change

