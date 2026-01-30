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



    """
    General heuristic func for RSI - returns overall heuristic score of ROC
    @param heuristic_weights: dict - dict of weights for heuristic
    @param weights: list - list of weights for roc periods
    @returns: float
    """
    def heuristic_sticky_roc_tas(
        weights: list = [0.2, 0.3, 0.5],
        heuristic_weights: dict = {
            'range_compression': 0.35,
            'zero_gravity': 0.30,
            'volatility_compression': 0.20,
            'neutrality': 0.15
        }
    ) -> float:

        """
        Heuristic for range compression of ROC - Measures if ROC is in a tight range or not
        @param group: col to group by
        @param col_name: col name to operate on (roc_3, roc_6, roc_12)
        @param weight: weight of heuristic
        @param min_required: minutes required to calculate - Nan otherwise
        @returns: float
        """
        def heuristic_roc_range_compression(group, col_name, weight, min_required):
            roc_values = group[col_name].dropna()
            if len(roc_values) >= min_required:
                roc_range = roc_values.max() - roc_values.min()
                if pd.notna(roc_range):
                    score = 1 - (roc_range / 0.902)
                    score = max(min(score, 1), 0)
                    return score * weight
            return np.nan


        """
        Heuristic for zero gravity of ROC - Measures if ROC stays close to 0
        @param group: col to group by
        @param col_name: col name to operate on (roc_3, roc_6, roc_12)
        @param weight: weight of heuristic
        @param min_required: minutes required to calculate - Nan otherwise
        @returns: float
        """ 
        def heuristic_roc_zero_gravity(group, col_name, weight, min_required):
            roc_values = group[col_name].dropna()
            if len(roc_values) >= min_required:
                avg_distance = np.abs(roc_values).mean()
                score = 1 - (avg_distance / 0.348)
                score = max(min(score, 1), 0)
                return score * weight
            return np.nan



        """
        Heuristic for std for ROC
        @param group: col to group by
        @param col_name: col name to operate on (roc_3, roc_6, roc_12)
        @param weight: weight of heuristic
        @param min_required: minutes required to calculate - Nan otherwise
        @returns: float
        """
        def heuristic_roc_volatility_compression(group, col_name, weight, min_required):
            roc_values = group[col_name].dropna()
            if len(roc_values) >= min_required:
                roc_std = roc_values.std()
                score = 1 - (roc_std / 0.278)
                score = max(min(score, 1), 0)
                return score * weight
            return np.nan


        """
        Heuristic for if ROC has directional bias
        @param group: col to group by
        @param col_name: col name to operate on (roc_3, roc_6, roc_12)
        @param weight: weight of heuristic
        @param min_required: minutes required to calculate - Nan otherwise
        @returns: float
        """
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

        return overall_score


    if ta_indicator == "rsi":
        return heuristic_sticky_rsi_tas()
    elif ta_indicator == "roc":
        return heuristic_sticky_roc_tas()
    else:
        
        raise Exception("Invalid ta indicator; Must be 'rsi' or 'roc'")



def heuristic_sticky_volume_tas(table_name: str, ta_indicator:str, start_time_till_eod: int, end_time_till_eod:int) -> float:
        
    def heuristic_sticky_vwap_tas(
        heuristic_weights: dict = {
            'price_clustering': 0.30,      # How tight around VWAP
            'range_compression': 0.25,     # Range of deviations
            'slope_stability': 0.20,       # VWAP flatness
            'crossover_frequency': 0.15,   # Healthy oscillation
            'final_distance': 0.10         # Close to VWAP at end
        },
        vwap_period_weights = [0.2,0.3, 0.5]
    ) -> float:

        def heuristic_vwap_price_clustering(group, col_name, weight=1.0, min_required=12):
            prices = group["close"].dropna()
            vwap_values = group[col_name].dropna()
            
            # Ensure we have matching data
            valid_idx = prices.index.intersection(vwap_values.index)
            if len(valid_idx) < min_required:
                return np.nan
            
            prices = prices.loc[valid_idx]
            vwap_values = vwap_values.loc[valid_idx]
            
            # Calculate percentage deviation from VWAP
            pct_deviation = np.abs((prices - vwap_values) / vwap_values) * 100
            
            # Average deviation
            avg_deviation = pct_deviation.mean()
            
            # Convert to score: lower deviation = higher score
            # Typical intraday deviation: 0.1% - 1.0%
            # 0.1% deviation → score 0.90
            # 0.5% deviation → score 0.50
            # 1.0% deviation → score 0.00
            score = max(1 - avg_deviation, 0)
            
            return score * weight


        def heuristic_vwap_slope_stability(group, col_name, weight=1.0, min_required=12):
            """
            Measures VWAP slope - flatter = more stable anchor point
            """
            vwap_values = group[col_name].dropna()
            
            if len(vwap_values) < min_required:
                return np.nan
            
            # Calculate slope using linear regression
            x = np.arange(len(vwap_values))
            y = vwap_values.values
            
            # Slope of best-fit line
            slope = np.polyfit(x, y, 1)[0]
            
            # Normalize slope as percentage of mean VWAP
            mean_vwap = y.mean()
            slope_pct = abs(slope / mean_vwap) * 100 * len(vwap_values)  # Slope over window
            
            # Convert to score: lower slope = higher score
            # Typical intraday VWAP slope: 0.05% - 0.5%
            score = max(1 - (slope_pct / 0.5), 0)
            
            return score * weight
        
        def heuristic_vwap_range_compression(group, col_name, weight=1.0, min_required=12):
            """
            Measures the range of (price - VWAP) deviations
            Tighter range = better mean reversion setup
            """
            prices = group["close"].dropna()
            vwap_values = group[col_name].dropna()
            
            valid_idx = prices.index.intersection(vwap_values.index)
            if len(valid_idx) < min_required:
                return np.nan
            
            prices = prices.loc[valid_idx]
            vwap_values = vwap_values.loc[valid_idx]
            
            # Calculate deviations
            deviations = ((prices - vwap_values) / vwap_values) * 100
            
            # Range of deviations
            dev_range = deviations.max() - deviations.min()
            
            # Convert to score: tighter range = higher score
            # Typical range: 0.5% - 2.0%
            # 0.5% range → score 0.75
            # 1.0% range → score 0.50
            # 2.0% range → score 0.00
            score = max(1 - (dev_range / 2.0), 0)
            
            return score * weight
        
        def heuristic_vwap_crossover_frequency(group, col_name, weight=1.0, min_required=12):
            """
            Measures optimal number of VWAP crossovers
            Too many = choppy, too few = trending away from VWAP
            """
            prices = group["close"].dropna()
            vwap_values = group[col_name].dropna()
            
            valid_idx = prices.index.intersection(vwap_values.index)
            if len(valid_idx) < min_required:
                return np.nan
            
            prices = prices.loc[valid_idx]
            vwap_values = vwap_values.loc[valid_idx]
            
            # Count crossovers
            above_vwap = prices > vwap_values
            crossovers = (above_vwap != above_vwap.shift(1)).sum()
            
            # Optimal: 2-4 crossovers in window (healthy oscillation)
            if 2 <= crossovers <= 4:
                score = 1.0
            elif crossovers < 2:
                score = 0.5  # Trending away from VWAP
            else:
                score = max(1 - ((crossovers - 4) * 0.1), 0)  # Too choppy
            
            return score * weight

        
        def heuristic_vwap_final_distance(group, col_name, weight=1.0, min_required=12):
            """
            Measures how close price is to VWAP at end of window
            Closer = better setup for mean reversion butterfly
            """
            prices = group["close"].dropna()
            vwap_values = group[col_name].dropna()
            
            valid_idx = prices.index.intersection(vwap_values.index)
            if len(valid_idx) < min_required:
                return np.nan
            
            # Get final values
            final_price = prices.loc[valid_idx].iloc[-1]
            final_vwap = vwap_values.loc[valid_idx].iloc[-1]
            
            # Calculate final deviation
            final_deviation = abs((final_price - final_vwap) / final_vwap) * 100
            
            # Convert to score: closer = higher score
            # Within 0.2% → score 0.90+
            # Within 0.5% → score 0.50
            # > 1.0% → score 0.00
            score = max(1 - (final_deviation / 1.0), 0)
            
            return score * weight
        
        
        assert abs(sum(heuristic_weights.values()) - 1.0) < 0.001, "Heuristic weights must sum to 1"
        assert abs(sum(vwap_period_weights) - 1.0) < 0.001, "VWAP weight must sum to 1"
        

        # Load data
        df = download_raw_data.get_raw_df_from_sql(
            table_name, 
            fields=["vwap_3_isolated", "vwap_6_isolated", "vwap_12_isolated", "time_till_eod", "close"]
        )
        
        # Filter to specific time window
        filtered_df = filter_regime_time_zone(
            df, 
            start_time_till_eod=start_time_till_eod, 
            end_time_till_eod=end_time_till_eod
        )
        
        # Initialize results
        daily_scores = pd.DataFrame(index=filtered_df.groupby('day').size().index)
        
        # For each VWAP period (3, 6, 12)
        for idx, (col_name, period_weight, min_req) in enumerate([
            ('vwap_3_isolated', vwap_period_weights[0], 3),
            ('vwap_6_isolated', vwap_period_weights[1], 6),
            ('vwap_12_isolated', vwap_period_weights[2], 12)
        ]):
            
            # Calculate each heuristic
            h1_clustering = filtered_df.groupby('day').apply(
                lambda g: heuristic_vwap_price_clustering(g, col_name, 1.0, min_req)
            )
            
            h2_range = filtered_df.groupby('day').apply(
                lambda g: heuristic_vwap_range_compression(g, col_name, 1.0, min_req)
            )
            
            h3_slope = filtered_df.groupby('day').apply(
                lambda g: heuristic_vwap_slope_stability(g, col_name, 1.0, min_req)
            )
            
            h4_crossover = filtered_df.groupby('day').apply(
                lambda g: heuristic_vwap_crossover_frequency(g, col_name, 1.0, min_req)
            )
            
            h5_final_dist = filtered_df.groupby('day').apply(
                lambda g: heuristic_vwap_final_distance(g, col_name, 1.0, min_req)
            )
            
            # Combine with weights
            combined_heuristic = (
                h1_clustering * heuristic_weights['price_clustering'] +
                h2_range * heuristic_weights['range_compression'] +
                h3_slope * heuristic_weights['slope_stability'] +
                h4_crossover * heuristic_weights['crossover_frequency'] +
                h5_final_dist * heuristic_weights['final_distance']
            )
            
            # Apply VWAP period weight
            daily_scores[f'vwap_{idx}_combined'] = combined_heuristic * period_weight
            
    
    
        # Sum across RSI periods
        daily_scores['final_vwap_heuristic'] = daily_scores[
            ['vwap_0_combined', 'vwap_1_combined', 'vwap_2_combined']
        ].sum(axis=1, skipna=True)
        
        # Handle all-NaN rows
        daily_scores.loc[
            daily_scores[['vwap_0_combined', 'vwap_1_combined', 'vwap_2_combined']].isna().all(axis=1),
            'final_vwap_heuristic'
        ] = np.nan
        
        overall_score = np.nanmedian(daily_scores['final_vwap_heuristic'])
        
        print(overall_score)
        
         
        return overall_score
    
        
    def heuristic_sticky_cmf_tas(
        heuristic_weights: dict = {
            'range_compression': 0.35,      # CMF range stability
            'neutrality': 0.30,             # How close to zero
            'volatility_compression': 0.20, # CMF std dev
            'extreme_avoidance': 0.15       # Avoid strong directional flow
        },
        cmf_period_weights: list = [0.2,0.3, 0.5],  # Weights for CMF_6_rolling, CMF_12_rolling, CMF_3/6/12_iso
    ) -> float:
                
        def heuristic_cmf_range_compression(group, col_name, weight, min_required):
            cmf_values = group[col_name].dropna()
            if len(cmf_values) >= min_required:
                cmf_range = cmf_values.max() - cmf_values.min()
                if pd.notna(cmf_range):
                    return (1 - (cmf_range / 2.0)) * weight
            return np.nan


        def heuristic_cmf_neutrality(group, col_name, weight, min_required):
            cmf_values = group[col_name].dropna()
            if len(cmf_values) >= min_required:
                distance_from_neutral = np.abs(cmf_values).mean()
                score = 1 - (distance_from_neutral / 0.5)
                score = max(min(score, 1), 0)
                return score * weight
            return np.nan


        def heuristic_cmf_volatility_compression(group, col_name, weight, min_required):
            cmf_values = group[col_name].dropna()
            if len(cmf_values) >= min_required:
                cmf_std = cmf_values.std()
                score = 1 - (cmf_std / 0.4)
                score = max(min(score, 1), 0)
                return score * weight
            return np.nan


        def heuristic_cmf_extreme_avoidance(group, col_name, weight, min_required):
            cmf_values = group[col_name].dropna()
            if len(cmf_values) >= min_required:
                extreme_bars = (np.abs(cmf_values) > 0.25).sum()
                extreme_pct = extreme_bars / len(cmf_values)
                score = 1 - extreme_pct
                return score * weight
            return np.nan
        
        
        # Validation
        assert abs(sum(heuristic_weights.values()) - 1.0) < 0.001, "Heuristic weights must sum to 1"
        assert abs(sum(cmf_period_weights) - 1.0) < 0.001, "CMF period weights must sum to 1"
        assert len(cmf_period_weights) == 3, "Must have exactly 3 CMF period weights"
        
        # Load data
        df = download_raw_data.get_raw_df_from_sql(
            table_name, 
            fields=["cmf_3_isolated", "cmf_6_isolated", "cmf_12_isolated", "time_till_eod"]
        )

        
        # Filter to specific time window
        filtered_df = filter_regime_time_zone(
            df, 
            start_time_till_eod=start_time_till_eod, 
            end_time_till_eod=end_time_till_eod
        )
        
        
        # Initialize results dataframe
        daily_scores = pd.DataFrame(index=filtered_df.groupby('day').size().index)
        
        # For each CMF period (10, 20, 30)
        for idx, (col_name, period_weight, min_req) in enumerate([
            ('cmf_3_isolated', cmf_period_weights[0], 3),
            ('cmf_6_isolated', cmf_period_weights[1], 6),
            ('cmf_12_isolated', cmf_period_weights[2], 12)
        ]):
            
            # Calculate each heuristic
            h1_range = filtered_df.groupby('day').apply(
                lambda g: heuristic_cmf_range_compression(g, col_name, 1.0, min_req)
            )
            
            h2_neutrality = filtered_df.groupby('day').apply(
                lambda g: heuristic_cmf_neutrality(g, col_name, 1.0, min_req)
            )
            
            h3_vol = filtered_df.groupby('day').apply(
                lambda g: heuristic_cmf_volatility_compression(g, col_name, 1.0, min_req)
            )
            
            h4_extreme = filtered_df.groupby('day').apply(
                lambda g: heuristic_cmf_extreme_avoidance(g, col_name, 1.0, min_req)
            )
            
            # Combine heuristics with their weights
            combined_heuristic = (
                h1_range * heuristic_weights['range_compression'] +
                h2_neutrality * heuristic_weights['neutrality'] +
                h3_vol * heuristic_weights['volatility_compression'] +
                h4_extreme * heuristic_weights['extreme_avoidance']
            )
            
            # Apply CMF period weight
            daily_scores[f'cmf_{idx}_combined'] = combined_heuristic * period_weight
        
        # Sum across CMF periods
        daily_scores['final_cmf_heuristic'] = daily_scores[
            ['cmf_0_combined', 'cmf_1_combined', 'cmf_2_combined']
        ].sum(axis=1, skipna=True)
        
        # Handle all-NaN rows
        daily_scores.loc[
            daily_scores[['cmf_0_combined', 'cmf_1_combined', 'cmf_2_combined']].isna().all(axis=1),
            'final_cmf_heuristic'
        ] = np.nan
        
        # Calculate overall score
        overall_score = np.nanmedian(daily_scores['final_cmf_heuristic'])
        
        print(overall_score)
        
        return overall_score
    
    
    
    if ta_indicator == "vwap":
        return heuristic_sticky_vwap_tas()
    elif ta_indicator == "cmf":
        return heuristic_sticky_cmf_tas()
    else:
        
        raise Exception("Invalid ta indicator; Must be 'vwap' or 'cmf'")
 
    

def heuristic_sticky_volatility_tas(table_name: str, ta_indicator:str, start_time_till_eod: int, end_time_till_eod:int) -> float:

    def heuristic_sticky_atr_tas(
    weights: list = [0.4, 0.4, 0.2],  # Compression, Stability, Contraction
    atr_period_weights: list = [0.4, 0.6]  # atr_6 vs atr_12
) -> float:
            
        def heuristic_atr_compression(group, col_name, weight, min_required=6,):
            """
            Measures ATR relative to price - lower ATR = higher score
            Similar to your RSI range compression, but inverted logic
            """
            atr_values = group[col_name].dropna()
            price_values = group["close"].dropna()
            
            if len(atr_values) >= min_required and len(price_values) > 0:
                # Calculate ATR as % of price (normalized)
                avg_price = price_values.mean()
                avg_atr = atr_values.mean()
                
                if avg_price > 0 and pd.notna(avg_atr):
                    # ATR as percentage of price
                    atr_pct = (avg_atr / avg_price) * 100
                    
                    # Typical intraday ATR: 0.3% - 1.5% of price
                    # Lower ATR% = higher score for butterflies
                    # Normalize: 0.2% ATR = 1.0 score, 1.5% ATR = 0.0 score
                    score = 1 - ((atr_pct - 0.2) / 1.3)  # Linear mapping
                    score = max(min(score, 1.0), 0.0)  # Clamp 0-1
                    
                    return score * weight
            return np.nan
        
        def heuristic_atr_stability(group, col_name, weight, min_required=6):
            atr_values = group[col_name].dropna()
            
            if len(atr_values) >= min_required:
                atr_std = atr_values.std()
                atr_mean = atr_values.mean()
                
                if atr_mean > 0 and pd.notna(atr_std):
                    # Coefficient of variation (CV) - normalized volatility
                    cv = atr_std / atr_mean
                    
                    # Lower CV = more stable = better for butterflies
                    # Typical CV: 0.1 - 0.5
                    # CV < 0.15 = excellent, CV > 0.4 = poor
                    score = 1 - (cv / 0.5)
                    score = max(min(score, 1.0), 0.0)
                    
                    return score * weight
            return np.nan
        
        def heuristic_atr_contraction(group, col_name, weight, min_required=6):
            atr_values = group[col_name].dropna()
            
            if len(atr_values) >= min_required:
                # Simple linear regression slope
                x = np.arange(len(atr_values))
                y = atr_values.values
                
                # Calculate slope
                slope = np.polyfit(x, y, 1)[0]
                
                # Normalize slope by mean ATR
                atr_mean = atr_values.mean()
                if atr_mean > 0:
                    normalized_slope = slope / atr_mean
                    
                    # Negative slope (decreasing ATR) = good
                    # -0.05 or lower = excellent (1.0 score)
                    # 0.0 = neutral (0.5 score)
                    # +0.05 or higher = poor (0.0 score)
                    if normalized_slope <= -0.05:
                        score = 1.0
                    elif normalized_slope >= 0.05:
                        score = 0.0
                    else:
                        # Linear interpolation between -0.05 and 0.05
                        score = 0.5 - (normalized_slope * 5)
                    
                    return score * weight
            return np.nan
        
        
        assert abs(sum(weights) - 1.0) < 0.001, "Heuristic weights must sum to 1"
        assert abs(sum(atr_period_weights) - 1.0) < 0.001, "ATR period weights must sum to 1"
        assert len(weights) == 3, "Must have exactly 3 heuristic weights"
        assert len(atr_period_weights) == 2, "Must have exactly 2 ATR period weights"
        
        # Load data
        df = download_raw_data.get_raw_df_from_sql(
            table_name, 
            fields=["atr_6_isolated", "atr_12_isolated", "close", "time_till_eod"]
        )
        
        filtered_df = filter_regime_time_zone(
            df, 
            start_time_till_eod=start_time_till_eod, 
            end_time_till_eod=end_time_till_eod
        )
        

        
        # Initialize results
        daily_scores = pd.DataFrame(index=filtered_df.groupby('day').size().index)
        
        # Process each ATR period
        for idx, (col_name, period_weight, min_req) in enumerate([
            ('atr_6_isolated', atr_period_weights[0], 6),
            ('atr_12_isolated', atr_period_weights[1], 12)
        ]):
            
            # Calculate each heuristic
            h1_compression = filtered_df.groupby('day').apply(
                lambda g: heuristic_atr_compression(g, col_name, 1.0, min_req)
            )
            
            h2_stability = filtered_df.groupby('day').apply(
                lambda g: heuristic_atr_stability(g, col_name, 1.0, min_req)
            )
            
            h3_contraction = filtered_df.groupby('day').apply(
                lambda g: heuristic_atr_contraction(g, col_name, 1.0, min_req)
            )
            
            # Combine heuristics with their weights
            combined_heuristic = (
                h1_compression * weights[0] +
                h2_stability * weights[1] +
                h3_contraction * weights[2]
            )
            
            # Apply ATR period weight
            daily_scores[f'atr_{idx}_combined'] = combined_heuristic * period_weight
        
        # Sum across ATR periods
        daily_scores['final_atr_heuristic'] = daily_scores[
            ['atr_0_combined', 'atr_1_combined']
        ].sum(axis=1, skipna=True)
        
        # Handle all-NaN rows
        daily_scores.loc[
            daily_scores[['atr_0_combined', 'atr_1_combined']].isna().all(axis=1),
            'final_atr_heuristic'
        ] = np.nan
        
        # Calculate overall score
        overall_score = np.nanmedian(daily_scores['final_atr_heuristic'])
        
        print(overall_score)
        
        return overall_score



    def heuristic_sticky_bb_width_tas(
        weights: list = [0.2, 0.3, 0.5],
    ) -> float:

        
        assert abs(sum(weights) - 1.0) < 0.001, "Weights must sum to 1"
        assert len(weights) == 3, "Must have exactly 3 weights"
        
        # Fetch data
        df = download_raw_data.get_raw_df_from_sql(
            table_name, 
            fields=["bb_width_3_isolated", "bb_width_6_isolated", "bb_width_12_isolated", "time_till_eod"]
        )
        
        filtered_df = filter_regime_time_zone(
            df, 
            start_time_till_eod=start_time_till_eod, 
            end_time_till_eod=end_time_till_eod
        )

        # Calculate normalization parameters based on method
        # Use percentile rank normalization (more robust)
        normalization_params = {}
        for col in ['bb_width_3_isolated', 'bb_width_6_isolated', 'bb_width_12_isolated']:
            normalization_params[col] = {
                'method': 'percentile',
                'data': filtered_df[col].dropna()
            }

        
        # Helper function to calculate heuristic with validation
        def calculate_bb_heuristic_with_validation(group, col_name, weight, min_required, norm_params):
            """
            Calculate BB width heuristic for one period
            
            Lower BB width = Higher score (better for butterflies)
            """
            bb_values = group[col_name].dropna()
            
            # Require minimum number of valid observations
            if len(bb_values) >= min_required:
                # Calculate mean BB width for this day
                mean_bb_width = bb_values.mean()
                
                if pd.notna(mean_bb_width):
                    # Normalize based on method
                    if norm_params['method'] == 'percentile':
                        # Calculate percentile rank (0-1)
                        # Lower percentile = narrower width = higher score
                        percentile_rank = (norm_params['data'] < mean_bb_width).sum() / len(norm_params['data'])
                        normalized_width = percentile_rank
                    else:  # historical_max
                        # Normalize by historical max
                        normalized_width = mean_bb_width / norm_params['max_value']
                        normalized_width = min(normalized_width, 1.0)  # Cap at 1.0
                    
                    # Convert to score: narrow width = high score
                    score = 1 - normalized_width
                    return score * weight
            
            return np.nan
        
        # Initialize results dataframe
        daily_scores = pd.DataFrame(index=filtered_df.groupby('day').size().index)
        
        # Calculate heuristic for each BB width period
        daily_scores['bb_width_3_heuristic'] = filtered_df.groupby('day').apply(
            lambda g: calculate_bb_heuristic_with_validation(
                g, 'bb_width_3_isolated', weights[0], min_required=3, 
                norm_params=normalization_params['bb_width_3_isolated']
            )
        ).values
        
        daily_scores['bb_width_6_heuristic'] = filtered_df.groupby('day').apply(
            lambda g: calculate_bb_heuristic_with_validation(
                g, 'bb_width_6_isolated', weights[1], min_required=6,
                norm_params=normalization_params['bb_width_6_isolated']
            )
        ).values
        
        daily_scores['bb_width_12_heuristic'] = filtered_df.groupby('day').apply(
            lambda g: calculate_bb_heuristic_with_validation(
                g, 'bb_width_12_isolated', weights[2], min_required=12,
                norm_params=normalization_params['bb_width_12_isolated']
            )
        ).values
        
        # Combine weighted scores
        daily_scores['general_bb_width_heuristic'] = daily_scores[
            ['bb_width_3_heuristic', 'bb_width_6_heuristic', 'bb_width_12_heuristic']
        ].sum(axis=1, skipna=True)
        
        # Handle all-NaN rows
        daily_scores.loc[
            daily_scores[['bb_width_3_heuristic', 'bb_width_6_heuristic', 'bb_width_12_heuristic']].isna().all(axis=1),
            'general_bb_width_heuristic'
        ] = np.nan
        
        # Calculate overall score
        overall_score = np.nanmedian(daily_scores['general_bb_width_heuristic'])
        
        # Diagnostics
        valid_days = daily_scores['general_bb_width_heuristic'].notna().sum()
        total_days = len(daily_scores)
        
        print(overall_score)

        return overall_score



    if ta_indicator == "atr":
        return heuristic_sticky_atr_tas()
    elif ta_indicator == "bb_width":
        return heuristic_sticky_bb_width_tas()
    else:
        
        raise Exception("Invalid ta indicator; Must be 'atr' or 'bb_width'")
 

def heuristic_sticky_trend_tas(table_name: str, ta_indicator:str, start_time_till_eod: int, end_time_till_eod:int) -> float:
        
    def heuristic_sticky_adx_tas(weights: list = [0.4, 0.6]) -> float:
        """
        ADX heuristic targeting optimal range (15-25) for butterflies
        """
        
        assert abs(sum(weights) - 1.0) < 0.001, "Weights must sum to 1"
        assert len(weights) == 2, "Must have exactly 2 weights"
        
        df = download_raw_data.get_raw_df_from_sql(
            table_name, 
            fields=["adx_6_isolated", "adx_12_isolated", "time_till_eod"]
        )
        
        filtered_df = filter_regime_time_zone(
            df, 
            start_time_till_eod=start_time_till_eod, 
            end_time_till_eod=end_time_till_eod
        )
        
        
        def optimal_adx_score(x, min_len):
            """Score ADX: peak at 15-25 range"""
            x_clean = x.dropna()
            if len(x_clean) < min_len:
                return np.nan
            
            adx_mean = x_clean.mean()
            
            # Optimal zone: 15-25
            if 15 <= adx_mean <= 25:
                return 1.0
            elif adx_mean < 15:
                return adx_mean / 15  # Penalize choppy markets
            else:
                return max(0, 1 - ((adx_mean - 25) / 25))  # Penalize trends
        
        daily_scores = filtered_df.groupby('day').agg({
            'adx_6_isolated': lambda x: optimal_adx_score(x, 6) * weights[0],
            'adx_12_isolated': lambda x: optimal_adx_score(x, 12) * weights[1]
        }).rename(columns={
            'adx_6_isolated': 'adx_6_heuristic',
            'adx_12_isolated': 'adx_12_heuristic'
        })
        
        daily_scores['general_adx_heuristic'] = daily_scores[
            ['adx_6_heuristic', 'adx_12_heuristic']
        ].sum(axis=1, skipna=True)
        
        daily_scores.loc[
            daily_scores[['adx_6_heuristic', 'adx_12_heuristic']].isna().all(axis=1),
            'general_adx_heuristic'
        ] = np.nan
        
        overall_score = np.nanmedian(daily_scores['general_adx_heuristic'])
        
        print(overall_score)

        return overall_score
    
    
        
    def heuristic_sticky_ema_tas(
        heuristic_weights: dict = {
            'oscillation_stability': 0.4,        # How close EMA_3 and EMA_6 are
            'cross_frequency': 0.2,    # Optimal crossing pattern
            'slope_stability': 0.2,    # Flat EMAs = good
            'range_compression': 0.2   # Tight EMA range = good
        },
        ema_period_weights: list = [0.4, 0.6]  # [EMA_3 weight, EMA_6 weight]
    ) -> float:

        def heuristic_ema_oscillation_stability(group, weight, min_required=12):
            """
            Measures stability of the EMA_3-EMA_6 spread
            Stable spread = consolidation, volatile spread = trending/choppy
            """
            ema_3 = group['ema_3_isolated'].dropna()
            ema_6 = group['ema_6_isolated'].dropna()
            
            if len(ema_3) >= min_required and len(ema_6) >= min_required:
                valid_idx = ema_3.index.intersection(ema_6.index)
                if len(valid_idx) < min_required:
                    return np.nan
                
                ema_3_valid = ema_3.loc[valid_idx]
                ema_6_valid = ema_6.loc[valid_idx]
                
                # Calculate the spread between EMAs
                spread = ema_3_valid - ema_6_valid
                
                # Normalize by price level
                price_level = ema_6_valid.mean()
                if price_level == 0:
                    return np.nan
                
                spread_pct = (spread / price_level) * 100
                
                # Measure volatility of the spread (not absolute value)
                # Low spread volatility = consolidation
                # High spread volatility = trending or whipsaw
                spread_std = spread_pct.std()
                
                # Based on your data, spread std ranges ~0.01-0.05%
                max_spread_std = 0.05
                
                score = 1 - min(spread_std / max_spread_std, 1.0)
                
                return score * weight
            
            return np.nan

        def heuristic_ema_cross_frequency(group, weight, min_required=6):
            """
            Measures crossing frequency between EMA_3 and EMA_6
            CALIBRATED: Your median is 4, mode is 3-4
            """
            ema_3 = group['ema_3_isolated'].dropna()
            ema_6 = group['ema_6_isolated'].dropna()
            
            if len(ema_3) >= min_required and len(ema_6) >= min_required:
                valid_idx = ema_3.index.intersection(ema_6.index)
                if len(valid_idx) < min_required:
                    return np.nan
                
                ema_3_valid = ema_3.loc[valid_idx]
                ema_6_valid = ema_6.loc[valid_idx]
                
                above = ema_3_valid > ema_6_valid
                crosses = (above != above.shift(1)).sum()
                
                # CALIBRATED: Based on your distribution
                # 3-5 crosses is most common (consolidation with healthy oscillation)
                # 1-2 = trending (not ideal)
                # 6+ = choppy (not ideal)
                if 3 <= crosses <= 5:
                    score = 1.0  # Optimal range
                elif 2 <= crosses <= 6:
                    score = 0.85  # Good range
                elif crosses == 1 or crosses == 7:
                    score = 0.6  # Marginal
                else:
                    # 0 crosses = strong trend, 8+ = whipsaw
                    score = max(0.4 - (abs(crosses - 4) * 0.05), 0)
                
                return score * weight
            
            return np.nan


        def heuristic_ema_slope_stability(group, weight, min_required=6):
            """
            Measures EMA slope - flatter = better for butterflies
            CALIBRATED to your data
            """
            ema_6 = group['ema_6_isolated'].dropna()
            
            if len(ema_6) >= min_required:
                price_level = ema_6.mean()
                if price_level == 0:
                    return np.nan
                
                slope = ema_6.diff().abs()
                avg_slope_pct = (slope.mean() / price_level) * 100
                
                # CALIBRATED: Your 95th percentile is 0.1161%
                max_slope = 0.12  # Based on your data
                
                score = 1 - min(avg_slope_pct / max_slope, 1.0)
                return score * weight
            
            return np.nan


        def heuristic_ema_range_compression(group, ema_col, weight, min_required=6):
            """
            Measures range of a single EMA over the period
            CALIBRATED to your data
            """
            ema_values = group[ema_col].dropna()
            
            if len(ema_values) >= min_required:
                ema_range = ema_values.max() - ema_values.min()
                price_level = ema_values.mean()
                
                if price_level == 0:
                    return np.nan
                
                range_pct = (ema_range / price_level) * 100
                
                # CALIBRATED: Your 95th percentile is ~1.0% for both EMA_3 and EMA_6
                max_range = 1.0  # Based on your data
                
                score = 1 - min(range_pct / max_range, 1.0)
                return score * weight
            
            return np.nan
        
        
        
        
        assert abs(sum(heuristic_weights.values()) - 1.0) < 0.001, "Heuristic weights must sum to 1"
        assert abs(sum(ema_period_weights) - 1.0) < 0.001, "EMA period weights must sum to 1"
        
        # Load data
        df = download_raw_data.get_raw_df_from_sql(
            table_name, 
            fields=["ema_3_isolated", "ema_6_isolated", "time_till_eod"]
        )
        
        # Filter to your time window
        filtered_df = filter_regime_time_zone(
            df, 
            start_time_till_eod=start_time_till_eod, 
            end_time_till_eod=end_time_till_eod
        )

        # Initialize results
        daily_scores = pd.DataFrame(index=filtered_df.groupby('day').size().index)
        
        # NEW: Oscillation stability (better than convergence)
        h1_oscillation = filtered_df.groupby('day').apply(
            lambda g: heuristic_ema_oscillation_stability(g, weight=1.0, min_required=12),
            include_groups=False
        )
        
        # Heuristic 2: Cross frequency (uses both EMAs together)
        h2_crosses = filtered_df.groupby('day').apply(
            lambda g: heuristic_ema_cross_frequency(g, weight=1.0, min_required=6)
        )
        

        h3_slope = filtered_df.groupby('day').apply(
            lambda g: heuristic_ema_slope_stability(g, weight=1.0, min_required=6),
            include_groups=False
        )
        # Apply maturity penalty based on number of bars in window
        num_bars = filtered_df.groupby('day').size()
        maturity_factors = num_bars.apply(
            lambda n: 0.6 if n < 12 else (0.8 if n < 24 else (0.9 if n < 36 else 1.0))
        )
        h3_slope = h3_slope * maturity_factors

        
        # Heuristic 4: Range compression for each EMA
        h4_ema3_range = filtered_df.groupby('day').apply(
            lambda g: heuristic_ema_range_compression(g, 'ema_3_isolated', weight=1.0, min_required=3)
        )
        
        h4_ema6_range = filtered_df.groupby('day').apply(
            lambda g: heuristic_ema_range_compression(g, 'ema_6_isolated', weight=1.0, min_required=6)
        )
        
        # Combine range compression scores with EMA period weights
        h4_combined_range = (
            h4_ema3_range * ema_period_weights[0] + 
            h4_ema6_range * ema_period_weights[1]
        )
        
        # Apply heuristic weights and combine
        daily_scores['ema_heuristic'] = (
            h1_oscillation * heuristic_weights['oscillation_stability'] +
            h2_crosses * heuristic_weights['cross_frequency'] +
            h3_slope * heuristic_weights['slope_stability'] +
            h4_combined_range * heuristic_weights['range_compression']
        )
        
        # Handle all-NaN days
        daily_scores.loc[daily_scores['ema_heuristic'].isna(), 'ema_heuristic'] = np.nan
        
        # Calculate overall score
        overall_score = np.nanmedian(daily_scores['ema_heuristic'])

        print(overall_score)        

        return overall_score
    
    
    
    if ta_indicator == "adx":
        return heuristic_sticky_adx_tas()
    elif ta_indicator == "ema":
        return heuristic_sticky_ema_tas()
    else:
        
        raise Exception("Invalid ta indicator; Must be 'adx' or 'ema'")
 



def heuristic_sticky_tas(table_name: str, ta_indicator:str, start_time_till_eod:int, end_time_till_eod:int) -> float:
    pass



if __name__ == "__main__":


    heuristic_sticky_trend_tas("spy_2025_5_minute_annual", ta_indicator="ema", start_time_till_eod=60, end_time_till_eod=0)
    



    
    pass
