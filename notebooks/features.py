import ta
import pandas as pd
from utils import load_csv_with_dates, get_apple_stock

def add_cpi(df):
    copy = df.copy()
    
    cpi = load_csv_with_dates('datasets/CPIAUCSL_FRED_19470101-20190401.csv')
    idx_date = pd.date_range(cpi.index[0], cpi.index[-1])
    cpi_full = cpi.reindex(idx_date)
    cpi_full.fillna(method='ffill', inplace=True)
    copy = pd.merge(copy, cpi_full, left_index=True, right_index=True)
    
    return copy

def add_vix(df):
    copy = df.copy()
    
    vix = load_csv_with_dates('datasets/VIX_yahoo-finance_19900102-20190531.csv')
    copy = pd.merge(copy, 
                    vix['Close'], 
                    left_index=True, 
                    right_index=True, 
                    suffixes=('', '_vix'))
    
    return copy

def add_ndx(df):
    copy = df.copy()
    
    ndx = load_csv_with_dates('datasets/NDX_yahoo-finance_19851001-20190531.csv')
    copy = pd.merge(copy, 
                    ndx['Close'], 
                    left_index=True, 
                    right_index=True, 
                    suffixes=('', '_ndx'))
    
    return copy

def add_technical_indicators(df, dropna=False):
    copy = df.copy()
    
    # Momentum Indicators 
    # https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html#momentum-indicators
    copy['ao'] = ta.momentum.ao(copy['High'], copy['Low'], s=5, len=34, fillna=False)
    copy['mfi_14'] = ta.momentum.money_flow_index(copy['High'], copy['Low'], copy['Close'], copy['Volume'], n=14, fillna=False)
    copy['rsi_14'] = ta.momentum.rsi(copy['Close'], n=14, fillna=False)
    copy['so_14'] = ta.momentum.stoch(copy['High'], copy['Low'], copy['Close'], n=14, fillna=False)
    copy['so_sig_14'] = ta.momentum.stoch_signal(copy['High'], copy['Low'], copy['Close'], n=14, d_n=3, fillna=False)
    copy['tsi'] = ta.momentum.tsi(copy['Close'], r=25, s=13, fillna=False)
    copy['uo'] = ta.momentum.uo(copy['High'], copy['Low'], copy['Close'], s=7, m=14, len=28, ws=4.0, wm=2.0, wl=1.0, fillna=False)
    copy['wr'] = ta.momentum.wr(copy['High'], copy['Low'], copy['Close'], lbp=14, fillna=False)
    
    # Volume Indicators
    # https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html#volume-indicators
    copy['adi'] = ta.volume.acc_dist_index(copy['High'], copy['Low'], copy['Close'], copy['Volume'], fillna=False)
    copy['cmf'] = ta.volume.chaikin_money_flow(copy['High'], copy['Low'], copy['Close'], copy['Volume'], n=20, fillna=False)
    copy['eom_20'] = ta.volume.ease_of_movement(copy['High'], copy['Low'], copy['Close'], copy['Volume'], n=20, fillna=False)
    copy['fi_2'] = ta.volume.force_index(copy['Close'], copy['Volume'], n=2, fillna=False)
    copy['nvi'] = ta.volume.negative_volume_index(copy['Close'], copy['Volume'], fillna=False)
#     copy['obv'] = ta.volume.on_balance_volume(copy['Close'], copy['Volume'], fillna=False)
    copy['vpt'] = ta.volume.volume_price_trend(copy['Close'], copy['Volume'], fillna=False)
    
    # Volatility Indicators
    # https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html#volatility-indicators
    copy['atr_14'] = ta.volatility.average_true_range(copy['High'], copy['Low'], copy['Close'], n=14, fillna=False)
    copy['b_hband_20'] = ta.volatility.bollinger_hband(copy['Close'], n=20, ndev=2, fillna=False)
    copy['b_hband_ind_20'] = ta.volatility.bollinger_hband_indicator(copy['Close'], n=20, ndev=2, fillna=False)
    copy['b_lband_20'] = ta.volatility.bollinger_lband(copy['Close'], n=20, ndev=2, fillna=False)
    copy['b_lband_ind_20'] = ta.volatility.bollinger_lband_indicator(copy['Close'], n=20, ndev=2, fillna=False)
    copy['mavg_10'] = ta.volatility.bollinger_mavg(copy['Close'], n=10, fillna=False)
    copy['mavg_20'] = ta.volatility.bollinger_mavg(copy['Close'], n=20, fillna=False)
    copy['mavg_50'] = ta.volatility.bollinger_mavg(copy['Close'], n=50, fillna=False)
    copy['mavg_200'] = ta.volatility.bollinger_mavg(copy['Close'], n=200, fillna=False)
    copy['dc_hband_20'] = ta.volatility.donchian_channel_hband(copy['Close'], n=20, fillna=False)
    copy['dc_hband_ind_20'] = ta.volatility.donchian_channel_hband_indicator(copy['Close'], n=20, fillna=False)
    copy['dc_lband_20'] = ta.volatility.donchian_channel_lband(copy['Close'], n=20, fillna=False)
    copy['dc_lband_ind_20'] = ta.volatility.donchian_channel_lband_indicator(copy['Close'], n=20, fillna=False)
    copy['kc_10'] = ta.volatility.keltner_channel_central(copy['High'], copy['Low'], copy['Close'], n=10, fillna=False)
    copy['kc_hband_10'] = ta.volatility.keltner_channel_hband(copy['High'], copy['Low'], copy['Close'], n=10, fillna=False)
    copy['kc_hband_ind_10'] = ta.volatility.keltner_channel_hband_indicator(copy['High'], copy['Low'], copy['Close'], n=10, fillna=False)
    copy['kc_lband_10'] = ta.volatility.keltner_channel_lband(copy['High'], copy['Low'], copy['Close'], n=10, fillna=False)
    copy['kc_lband_ind_10'] = ta.volatility.keltner_channel_lband_indicator(copy['High'], copy['Low'], copy['Close'], n=10, fillna=False)

    # Trend Indicators
    # https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html#trend-indicators
#     copy['adx_14'] = ta.trend.adx(copy['High'], copy['Low'], copy['Close'], n=14, fillna=True)
#     copy['adx_neg_14'] = ta.trend.adx_neg(copy['High'], copy['Low'], copy['Close'], n=14, fillna=True)
#     copy['adx_pos_14'] = ta.trend.adx_pos(copy['High'], copy['Low'], copy['Close'], n=14, fillna=True)
    copy['ai_down25'] = ta.trend.aroon_down(copy['Close'], n=25, fillna=False)
    copy['ai_up25'] = ta.trend.aroon_up(copy['Close'], n=25, fillna=False)
    copy['cci20'] = ta.trend.cci(copy['High'], copy['Low'], copy['Close'], n=20, c=0.015, fillna=False)
    copy['dpo_20'] = ta.trend.dpo(copy['Close'], n=20, fillna=False)
    copy['ema_12'] = ta.trend.ema_indicator(copy['Close'], n=12, fillna=False)
    copy['ema_26'] = ta.trend.ema_indicator(copy['Close'], n=26, fillna=False)
    copy['ichimoku_a'] = ta.trend.ichimoku_a(copy['High'], copy['Low'], n1=9, n2=26, visual=False, fillna=False)
    copy['ichimoku_b'] = ta.trend.ichimoku_b(copy['High'], copy['Low'], n2=26, n3=52, visual=False, fillna=False)
    copy['kst'] = ta.trend.kst(copy['Close'], r1=10, r2=15, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15, fillna=False)
    copy['kst_sig'] = ta.trend.kst_sig(copy['Close'], r1=10, r2=15, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15, nsig=9, fillna=False)
    copy['macd'] = ta.trend.macd(copy['Close'], n_fast=12, n_slow=26, fillna=False)
    copy['macd_diff'] = ta.trend.macd_diff(copy['Close'], n_fast=12, n_slow=26, n_sign=9, fillna=False)
    copy['macd_sig'] = ta.trend.macd_signal(copy['Close'], n_fast=12, n_slow=26, n_sign=9, fillna=False)
    copy['mi'] = ta.trend.mass_index(copy['High'], copy['Low'], n=9, n2=25, fillna=False)
    copy['trix_15'] = ta.trend.trix(copy['Close'], n=15, fillna=False)
    copy['vi_neg_14'] = ta.trend.vortex_indicator_neg(copy['High'], copy['Low'], copy['Close'], n=14, fillna=False)
    copy['vi_pos_14'] = ta.trend.vortex_indicator_pos(copy['High'], copy['Low'], copy['Close'], n=14, fillna=False)
    
    # Others Indicators
    # https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html#others-indicators
    copy['cr'] = ta.others.cumulative_return(copy['Close'], fillna=False)
    copy['dlr'] = ta.others.daily_log_return(copy['Close'], fillna=False)
    copy['dr'] = ta.others.daily_return(copy['Close'], fillna=False)
    
    if dropna is True:
        return copy.dropna() # we drop all the rows that contain at least one NaN
    
    return copy

def get_apple_stock_with_features(columns=None, dropna=False):
    df = get_apple_stock()
    
    df = add_technical_indicators(df, dropna)
    df = add_cpi(df)
    df = add_vix(df)
    df = add_ndx(df)
    
    if columns is not None:
        return df[columns]
    
    return df
    
    