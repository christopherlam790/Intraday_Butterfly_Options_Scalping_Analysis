import pandas as pd
import numpy as np


import download_raw_data

"""
Helper func to filter time regimes by time zone
@param df: pd.DataFrame - input DataFrame
@param start_time_till_eod: int - Start time in minutes until end of day
@param end_time_till_eof: int - End time in minutes until end of day
@returns: pd.DataFrame
"""
def filter_regime_time_zone(df: pd.DataFrame, start_time_till_eod: int, end_time_till_eod: int) -> pd.DataFrame:

    try:
        assert start_time_till_eod <= 390 and start_time_till_eod > 0, "Start time must be over 0"
        assert end_time_till_eod < 390 and end_time_till_eod >= 0, "End time must be less than 390"
        assert start_time_till_eod > end_time_till_eod, "End time must be less than start time" 
        
        filtered_df = df[(df["time_till_eod"] >= end_time_till_eod) & (df["time_till_eod"] <= start_time_till_eod)]

        filtered_df["day"] = filtered_df.index.astype(str).str[:10]

        return filtered_df

    except:
        raise Exception("Invalid time zones; Try again")


"""
General heuristc func for momentum ta indicators (rsi & roc) - returns heuristic score of requested momentum indicator
@param table_name: str - Name of table
@param ta_indicator: str - Momentum indicator to get heuristic on ('rsi' or 'roc')
@param start_time_till_eod: int - Start time in minutes until end of day
@param end_time_till_eof: int - End time in minutes until end of day
@returns: pd.DataFrame

"""
def heuristic_sticky_momentum_tas(table_name: str, ta_indicator: str, start_time_till_eod: int, end_time_till_eod: int) -> float:


    """
    General heuristic func for RSI - returns overall heuristic score of RSI
    @param heuristic_weights: dict - dict of weights for heuristic
    @param rsi_period_weights: list - list of weights for rsi periods
    @returns: float
    """
    def heuristic_sticky_rsi_tas(
    heuristic_weights: dict = {
        'range_compression': 0.4,      # Your original S_range
        'mean_reversion_zone': 0.3,    # How close to RSI=50
        'volatility_compression': 0.2, # RSI std dev
        'extreme_avoidance': 0.1       # Avoid oversold/overbought
    },
    rsi_period_weights: list = [0.2, 0.3, 0.5]
) -> float:
        
        assert abs(sum(heuristic_weights.values()) - 1.0) < 0.001, "Heuristic weights must sum to 1"
        assert abs(sum(rsi_period_weights) - 1.0) < 0.001, "RSI period weights must sum to 1"
        
        """
        Heuristic for range compression of RSI - Measures if RSI is in a tight range or not
        @param group: col to group by
        @param col_name: col name to operate on (rsi_3, rsi_6, rsi_12)
        @param weight: weight of heuristic
        @param min_required: minutes required to calculate - Nan otherwise
        @returns: float
        """
        def heuristic_range_compression(group:str, col_name: str,  min_required: int, weight: float = 1.0,) -> float:
            rsi_values = group[col_name].dropna()
            if len(rsi_values) >= min_required:
                rsi_range = rsi_values.max() - rsi_values.min()
                if pd.notna(rsi_range):
                    return (1 - (rsi_range / 100)) * weight
            return np.nan
        
        
        """
        Heuristic for mean reversion of RSI - Measures if RSI is constantly near 50 (neutral levels)
        @param group: col to group by
        @param col_name: col name to operate on (rsi_3, rsi_6, rsi_12)
        @param weight: weight of heuristic
        @param min_required: minutes required to calculate - Nan otherwise
        @returns: float
        """ 
        def heuristic_rsi_mean_reversion_zone(group: str, col_name: str, min_required:int=12, weight: float = 1.0) -> float:
            rsi_values = group[col_name].dropna()
            
            if len(rsi_values) >= min_required:
                distance_from_neutral = np.abs(rsi_values - 50).mean()
                # Convert to score: closer to 50 = higher score
                score = 1 - (distance_from_neutral / 50)  # Normalize to 0-1
                return score * weight
            return np.nan 
        
        
        """
        Heuristic for range compression of RSI - Measures if RSI is in a 'squeeze' / contracting vol
        @param group: col to group by
        @param col_name: col name to operate on (rsi_3, rsi_6, rsi_12)
        @param weight: weight of heuristic
        @param min_required: minutes required to calculate - Nan otherwise
        @returns: float
        """    
        def heuristic_rsi_volatility_compression(group: str, col_name: str, min_required:int=12, weight: float = 1.0):
        
            rsi_values = group[col_name].dropna()
            
            if len(rsi_values) >= min_required:
                rsi_std = rsi_values.std()
                score = 1 - (rsi_std / 30)
                score = max(min(score, 1), 0)  # Clamp to 0-1
                return score * weight
            return np.nan
        
        
        
        """
        Heuristic for range compression of RSI - Penalize if RSI frequently overbought (>70) or oversold (<30)
        @param group: col to group by
        @param col_name: col name to operate on (rsi_3, rsi_6, rsi_12)
        @param weight: weight of heuristic
        @param min_required: minutes required to calculate - Nan otherwise
        @returns: float
        """
        def heuristic_rsi_extreme_avoidance(group:str, col_name:str, min_required:int=12, weight: float = 1.0):
            """Penalizes days where RSI hits extreme levels"""
            rsi_values = group[col_name].dropna()
            
            if len(rsi_values) >= min_required:
                # Count bars in extreme zones
                extreme_bars = ((rsi_values > 70) | (rsi_values < 30)).sum()
                extreme_pct = extreme_bars / len(rsi_values)
                
                # Score: fewer extremes = better
                score = 1 - extreme_pct
                return score * weight
            return np.nan
        
        df = download_raw_data.get_raw_df_from_sql(
            table_name, 
            fields=["rsi_3_isolated", "rsi_6_isolated", "rsi_12_isolated", "time_till_eod"]
        )
        
        filtered_df = filter_regime_time_zone(
            df, 
            start_time_till_eod=start_time_till_eod, 
            end_time_till_eod=end_time_till_eod
        )
                
        daily_scores = pd.DataFrame(index=filtered_df.groupby('day').size().index)
        
        # For each RSI period (3, 6, 12)
        for idx, (col_name, period_weight, min_req) in enumerate([
            ('rsi_3_isolated', rsi_period_weights[0], 3),
            ('rsi_6_isolated', rsi_period_weights[1], 6),
            ('rsi_12_isolated', rsi_period_weights[2], 12)
        ]):
            
            # Calculate each heuristic
            h1_range = filtered_df.groupby('day').apply(
                lambda g: heuristic_range_compression(g, col_name, min_req, weight=1.0,)
            )
            
            h2_zone = filtered_df.groupby('day').apply(
                lambda g: heuristic_rsi_mean_reversion_zone(g, col_name, min_req, weight=1.0)
            )
            
            h3_vol = filtered_df.groupby('day').apply(
                lambda g: heuristic_rsi_volatility_compression(g, col_name, min_req, weight=1.0)
            )
            
            h4_extreme = filtered_df.groupby('day').apply(
                lambda g: heuristic_rsi_extreme_avoidance(g, col_name, min_req, weight=1.0)
            )
            
            # Combine heuristics with their weights
            combined_heuristic = (
                h1_range * heuristic_weights['range_compression'] +
                h2_zone * heuristic_weights['mean_reversion_zone'] +
                h3_vol * heuristic_weights['volatility_compression'] +
                h4_extreme * heuristic_weights['extreme_avoidance']
            )
            
            # Apply RSI period weight
            daily_scores[f'rsi_{idx}_combined'] = combined_heuristic * period_weight
        
        # Sum across RSI periods
        daily_scores['final_rsi_heuristic'] = daily_scores[
            ['rsi_0_combined', 'rsi_1_combined', 'rsi_2_combined']
        ].sum(axis=1, skipna=True)
        
        # Handle all-NaN rows
        daily_scores.loc[
            daily_scores[['rsi_0_combined', 'rsi_1_combined', 'rsi_2_combined']].isna().all(axis=1),
            'final_rsi_heuristic'
        ] = np.nan
        
        overall_score = np.nanmedian(daily_scores['final_rsi_heuristic'])
                
        return overall_score




    def heuristic_sticky_roc_tas(
        weights: list = [0.2, 0.3, 0.5],
        heuristic_weights: dict = {
            'range_compression': 0.35,
            'zero_gravity': 0.30,
            'volatility_compression': 0.20,
            'neutrality': 0.15
        }
    ) -> float:

        def heuristic_roc_range_compression(group, col_name, weight, min_required):
            roc_values = group[col_name].dropna()
            if len(roc_values) >= min_required:
                roc_range = roc_values.max() - roc_values.min()
                if pd.notna(roc_range):
                    score = 1 - (roc_range / 0.902)
                    score = max(min(score, 1), 0)
                    return score * weight
            return np.nan

        def heuristic_roc_zero_gravity(group, col_name, weight, min_required):
            roc_values = group[col_name].dropna()
            if len(roc_values) >= min_required:
                avg_distance = np.abs(roc_values).mean()
                score = 1 - (avg_distance / 0.348)
                score = max(min(score, 1), 0)
                return score * weight
            return np.nan

        def heuristic_roc_volatility_compression(group, col_name, weight, min_required):
            roc_values = group[col_name].dropna()
            if len(roc_values) >= min_required:
                roc_std = roc_values.std()
                score = 1 - (roc_std / 0.278)
                score = max(min(score, 1), 0)
                return score * weight
            return np.nan

        def heuristic_roc_neutrality(group, col_name, weight, min_required):
            roc_values = group[col_name].dropna()
            if len(roc_values) >= min_required:
                roc_mean = roc_values.mean()
                bias = abs(roc_mean)
                score = 1 - (bias / 0.304)
                score = max(min(score, 1), 0)
                return score * weight
            return np.nan
        
            
        # Validation
        assert abs(sum(heuristic_weights.values()) - 1.0) < 0.001, "Heuristic weights must sum to 1"
        assert abs(sum(weights) - 1.0) < 0.001, "RSI period weights must sum to 1"
        
        
        # Load data
        df = download_raw_data.get_raw_df_from_sql(
            table_name, 
            fields=["roc_3_isolated", "roc_6_isolated", "roc_12_isolated", "time_till_eod"]
        )
        
        # Filter to specific time window
        filtered_df = filter_regime_time_zone(
            df, 
            start_time_till_eod=start_time_till_eod, 
            end_time_till_eod=end_time_till_eod
        )
        
        
        # Initialize results
        daily_scores = pd.DataFrame(index=filtered_df.groupby('day').size().index)
        
        # Process each ROC period
        for idx, (col_name, period_weight, min_req) in enumerate([
            ('roc_3_isolated', weights[0], 3),
            ('roc_6_isolated', weights[1], 6),
            ('roc_12_isolated', weights[2], 12)
        ]):
            
            # Calculate each heuristic
            h1_range = filtered_df.groupby('day').apply(
                lambda g: heuristic_roc_range_compression(g, col_name, 1.0, min_req)
            )
            
            h2_zero = filtered_df.groupby('day').apply(
                lambda g: heuristic_roc_zero_gravity(g, col_name, 1.0, min_req)
            )
            
            h3_vol = filtered_df.groupby('day').apply(
                lambda g: heuristic_roc_volatility_compression(g, col_name, 1.0, min_req)
            )
            
            h4_neutral = filtered_df.groupby('day').apply(
                lambda g: heuristic_roc_neutrality(g, col_name, 1.0, min_req)
            )
            
            # Combine heuristics with their weights
            combined_heuristic = (
                h1_range * heuristic_weights['range_compression'] +
                h2_zero * heuristic_weights['zero_gravity'] +
                h3_vol * heuristic_weights['volatility_compression'] +
                h4_neutral * heuristic_weights['neutrality']
            )
            
            # Apply period weight
            daily_scores[f'roc_{idx}_combined'] = combined_heuristic * period_weight
        
        # Sum across ROC periods
        daily_scores['final_roc_heuristic'] = daily_scores[
            ['roc_0_combined', 'roc_1_combined', 'roc_2_combined']
        ].sum(axis=1, skipna=True)
        
        # Handle all-NaN rows
        daily_scores.loc[
            daily_scores[['roc_0_combined', 'roc_1_combined', 'roc_2_combined']].isna().all(axis=1),
            'final_roc_heuristic'
        ] = np.nan
        
        # Calculate overall score
        overall_score = np.nanmedian(daily_scores['final_roc_heuristic'])
        
        print(overall_score)



        return overall_score




    if ta_indicator == "rsi":
        return heuristic_sticky_rsi_tas()
    elif ta_indicator == "roc":
        return heuristic_sticky_roc_tas()
    else:
        
        raise Exception("Invalid ta indicator; Must be 'rsi' or 'roc'")



def heuristic_sticky_tas(table_name: str) -> float:
    pass



if __name__ == "__main__":


    heuristic_sticky_momentum_tas("spy_2025_5_minute_annual", ta_indicator="roc", start_time_till_eod=60, end_time_till_eod=0)
    

    
    pass
