# --- Do not remove these libs ---
import pandas_ta as pta
import copy
import logging
import pathlib
import rapidjson
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair, timeframe_to_minutes
from freqtrade.exchange import timeframe_to_prev_date
from pandas import DataFrame, Series, concat
from functools import reduce
import math
from typing import Dict
from freqtrade.persistence import Trade
from datetime import datetime, timedelta
from technical.util import resample_to_interval, resampled_merge
from technical.indicators import RMI, zema, VIDYA, ichimoku
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)
import time

log = logging.getLogger(__name__)

# --------------------------------
def ha_typical_price(bars):
    res = (bars['ha_high'] + bars['ha_low'] + bars['ha_close']) / 3.
    return Series(index=bars.index, data=res)

# Volume Weighted Moving Average
def vwma(dataframe: DataFrame, length: int = 10):
    """Indicator: Volume Weighted Moving Average (VWMA)"""
    # Calculate Result
    pv = dataframe['close'] * dataframe['volume']
    vwma = Series(ta.SMA(pv, timeperiod=length) / ta.SMA(dataframe['volume'], timeperiod=length))
    return vwma

# Modified Elder Ray Index
def moderi(dataframe: DataFrame, len_slow_ma: int = 32) -> Series:
    slow_ma = Series(ta.EMA(vwma(dataframe, length=len_slow_ma), timeperiod=len_slow_ma))
    return slow_ma >= slow_ma.shift(1)  # we just need true & false for ERI trend

def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['low'] * 100
    return emadif

def SROC(dataframe, roclen=21, emalen=13, smooth=21):
    df = dataframe.copy()

    roc = ta.ROC(df, timeperiod=roclen)
    ema = ta.EMA(df, timeperiod=emalen)
    sroc = ta.ROC(ema, timeperiod=smooth)

    return sroc

def range_percent_change(dataframe: DataFrame, method, length: int) -> float:
        """
        Rolling Percentage Change Maximum across interval.

        :param dataframe: DataFrame The original OHLC dataframe
        :param method: High to Low / Open to Close
        :param length: int The length to look back
        """
        if method == 'HL':
            return (dataframe['high'].rolling(length).max() - dataframe['low'].rolling(length).min()) / dataframe['low'].rolling(length).min()
        elif method == 'OC':
            return (dataframe['open'].rolling(length).max() - dataframe['close'].rolling(length).min()) / dataframe['close'].rolling(length).min()
        else:
            raise ValueError(f"Method {method} not defined!")

# Williams %R
def williams_r(dataframe: DataFrame, period: int = 14) -> Series:
    """Williams %R, or just %R, is a technical analysis oscillator showing the current closing price in relation to the high and low
        of the past N days (for a given N). It was developed by a publisher and promoter of trading materials, Larry Williams.
        Its purpose is to tell whether a stock or commodity market is trading near the high or the low, or somewhere in between,
        of its recent trading range.
        The oscillator is on a negative scale, from âˆ’100 (lowest) up to 0 (highest).
    """

    highest_high = dataframe["high"].rolling(center=False, window=period).max()
    lowest_low = dataframe["low"].rolling(center=False, window=period).min()

    WR = Series(
        (highest_high - dataframe["close"]) / (highest_high - lowest_low),
        name=f"{period} Williams %R",
        )

    return WR * -100

# Chaikin Money Flow
def chaikin_money_flow(dataframe, n=20, fillna=False) -> Series:
    """Chaikin Money Flow (CMF)
    It measures the amount of Money Flow Volume over a specific period.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chaikin_money_flow_cmf
    Args:
        dataframe(pandas.Dataframe): dataframe containing ohlcv
        n(int): n period.
        fillna(bool): if fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """
    mfv = ((dataframe['close'] - dataframe['low']) - (dataframe['high'] - dataframe['close'])) / (dataframe['high'] - dataframe['low'])
    mfv = mfv.fillna(0.0)  # float division by zero
    mfv *= dataframe['volume']
    cmf = (mfv.rolling(n, min_periods=0).sum()
           / dataframe['volume'].rolling(n, min_periods=0).sum())
    if fillna:
        cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
    return Series(cmf, name='cmf')

class BB_RPB_TSL_Tranz(IStrategy):
    INTERFACE_VERSION = 3
    '''
        BB_RPB_TSL
        @author jilv220
        Simple bollinger brand strategy inspired by this blog  ( https://hacks-for-life.blogspot.com/2020/12/freqtrade-notes.html )
        RPB, which stands for Real Pull Back, taken from ( https://github.com/GeorgeMurAlkh/freqtrade-stuff/blob/main/user_data/strategies/TheRealPullbackV2.py )
        The trailing custom stoploss taken from BigZ04_TSL from Perkmeister ( modded by ilya )
        I modified it to better suit my taste and added Hyperopt for this strategy.
    '''

    # (1) sell rework

    ##########################################################################

    # Hyperopt result area

    # buy space
    buy_params = {
        "max_slip": 0.983,
        ##
        "buy_bb_width_1h": 0.954,
        "buy_roc_1h": 86,
        ##
        "buy_threshold": 0.003,
        "buy_bb_factor": 0.999,
        #
        "buy_bb_delta": 0.025,
        "buy_bb_width": 0.095,
        ##
        "buy_cci": -116,
        "buy_cci_length": 25,
        "buy_rmi": 49,
        "buy_rmi_length": 17,
        "buy_srsi_fk": 32,
        ##
        "buy_closedelta": 17.922,
        "buy_ema_diff": 0.026,
        ##
        "buy_ema_high": 0.968,
        "buy_ema_low": 0.935,
        "buy_ewo": -5.001,
        "buy_rsi": 23,
        "buy_rsi_fast": 44,
        ##
        "buy_ema_high_2": 1.087,
        "buy_ema_low_2": 0.970,
        "buy_ewo_high_2": 4.179,
        "buy_rsi_ewo_2": 35,
        "buy_rsi_fast_ewo_2": 45,
        ##
        "buy_closedelta_local_dip": 12.044,
        "buy_ema_diff_local_dip": 0.024,
        "buy_ema_high_local_dip": 1.014,
        "buy_rsi_local_dip": 21,
        ##
        "buy_r_deadfish_bb_factor": 1.014,
        "buy_r_deadfish_bb_width": 0.299,
        "buy_r_deadfish_ema": 1.054,
        "buy_r_deadfish_volume_factor": 1.59,
        "buy_r_deadfish_cti": -0.115,
        "buy_r_deadfish_r14": -44.34,
        ##
        "buy_clucha_bbdelta_close": 0.049,
        "buy_clucha_bbdelta_tail": 1.146,
        "buy_clucha_close_bblower": 0.018,
        "buy_clucha_closedelta_close": 0.017,
        "buy_clucha_rocr_1h": 0.526,
        ##
        "buy_adx": 13,
        "buy_cofi_r14": -85.016,
        "buy_cofi_cti": -0.892,
        "buy_ema_cofi": 1.147,
        "buy_ewo_high": 8.594,
        "buy_fastd": 28,
        "buy_fastk": 39,
        ##
        "buy_gumbo_ema": 1.121,
        "buy_gumbo_ewo_low": -9.442,
        "buy_gumbo_cti": -0.374,
        "buy_gumbo_r14": -51.971,
        ##
        "buy_sqzmom_ema": 0.981,
        "buy_sqzmom_ewo": -3.966,
        "buy_sqzmom_r14": -45.068,
        ##
        "buy_nfix_39_ema": 0.912,
        ##
        "buy_nfix_49_cti": -0.105,
        "buy_nfix_49_r14": -81.827,
    }

    # sell space
    sell_params = {
        ##
        "sell_cmf": -0.046,
        "sell_ema": 0.988,
        "sell_ema_close_delta": 0.022,
        ##
        "sell_deadfish_profit": -0.063,
        "sell_deadfish_bb_factor": 0.954,
        "sell_deadfish_bb_width": 0.043,
        "sell_deadfish_volume_factor": 2.37,
        ##
        "sell_cti_r_cti": 0.844,
        "sell_cti_r_r": -19.99,
    }

    minimal_roi = {
        "0": 0.205,
        "81": 0.038,
        "292": 0.005,
    }

    # Optimal timeframe for the strategy
    timeframe = '5m'
    inf_1h = '1h'
    inf_15m = '15m'
    info_timefame_1d = 'none'
    info_timeframe_1h = '1h'
    info_timeframe_15m = '15m'
    res_timeframe = 'none'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # Disabled
    stoploss = -0.99

    # Custom stoploss
    use_custom_stoploss = True
    use_sell_signal = True

    ############################################################################

    ## Buy params

    is_optimize_dip = False
    buy_rmi = IntParameter(30, 50, default=35, optimize= is_optimize_dip)
    buy_cci = IntParameter(-135, -90, default=-133, optimize= is_optimize_dip)
    buy_srsi_fk = IntParameter(30, 50, default=25, optimize= is_optimize_dip)
    buy_cci_length = IntParameter(25, 45, default=25, optimize = is_optimize_dip)
    buy_rmi_length = IntParameter(8, 20, default=8, optimize = is_optimize_dip)

    is_optimize_break = False
    buy_bb_width = DecimalParameter(0.065, 0.135, default=0.095, optimize = is_optimize_break)
    buy_bb_delta = DecimalParameter(0.018, 0.035, default=0.025, optimize = is_optimize_break)

    is_optimize_local_uptrend = False
    buy_ema_diff = DecimalParameter(0.022, 0.027, default=0.025, optimize = is_optimize_local_uptrend)
    buy_bb_factor = DecimalParameter(0.990, 0.999, default=0.995, optimize = False)
    buy_closedelta = DecimalParameter(12.0, 18.0, default=15.0, optimize = is_optimize_local_uptrend)

    is_optimize_local_dip = False
    buy_ema_diff_local_dip = DecimalParameter(0.022, 0.027, default=0.025, optimize = is_optimize_local_dip)
    buy_ema_high_local_dip = DecimalParameter(0.90, 1.2, default=0.942 , optimize = is_optimize_local_dip)
    buy_closedelta_local_dip = DecimalParameter(12.0, 18.0, default=15.0, optimize = is_optimize_local_dip)
    buy_rsi_local_dip = IntParameter(15, 45, default=28, optimize = is_optimize_local_dip)
    buy_crsi_local_dip = IntParameter(10, 18, default=10, optimize = False)

    is_optimize_ewo = False
    buy_rsi_fast = IntParameter(35, 50, default=45, optimize = is_optimize_ewo)
    buy_rsi = IntParameter(15, 35, default=35, optimize = is_optimize_ewo)
    buy_ewo = DecimalParameter(-6.0, 5, default=-5.585, optimize = is_optimize_ewo)
    buy_ema_low = DecimalParameter(0.9, 0.99, default=0.942 , optimize = is_optimize_ewo)
    buy_ema_high = DecimalParameter(0.95, 1.2, default=1.084 , optimize = is_optimize_ewo)

    is_optimize_ewo_2 = False
    buy_rsi_fast_ewo_2 = IntParameter(15, 50, default=45, optimize = is_optimize_ewo_2)
    buy_rsi_ewo_2 = IntParameter(15, 50, default=35, optimize = is_optimize_ewo_2)
    buy_ema_low_2 = DecimalParameter(0.90, 1.2, default=0.970 , optimize = is_optimize_ewo_2)
    buy_ema_high_2 = DecimalParameter(0.90, 1.2, default=1.087 , optimize = is_optimize_ewo_2)
    buy_ewo_high_2 = DecimalParameter(2, 12, default=4.179, optimize = is_optimize_ewo_2)

    is_optimize_r_deadfish = False
    buy_r_deadfish_ema = DecimalParameter(0.90, 1.2, default=1.087 , optimize = is_optimize_r_deadfish)
    buy_r_deadfish_bb_width = DecimalParameter(0.03, 0.75, default=0.05 , optimize = is_optimize_r_deadfish)
    buy_r_deadfish_bb_factor = DecimalParameter(0.90, 1.2, default=1.0 , optimize = is_optimize_r_deadfish)
    buy_r_deadfish_volume_factor = DecimalParameter(1, 2.5, default=1.0 , optimize = is_optimize_r_deadfish)

    is_optimize_r_deadfish_protection = False
    buy_r_deadfish_cti = DecimalParameter(-0.6, -0.0, default=-0.5 , optimize = is_optimize_r_deadfish_protection)
    buy_r_deadfish_r14 = DecimalParameter(-60, -44, default=-60 , optimize = is_optimize_r_deadfish_protection)

    is_optimize_clucha = False
    buy_clucha_bbdelta_close = DecimalParameter(0.01,0.05, default=0.02206, optimize = is_optimize_clucha)
    buy_clucha_bbdelta_tail = DecimalParameter(0.7, 1.2, default=1.02515, optimize = is_optimize_clucha)
    buy_clucha_closedelta_close = DecimalParameter(0.001, 0.05, default=0.04401, optimize = is_optimize_clucha)
    buy_clucha_rocr_1h = DecimalParameter(0.1, 1.0, default=0.47782, optimize = is_optimize_clucha)

    is_optimize_cofi = False
    buy_ema_cofi = DecimalParameter(0.94, 1.2, default=0.97 , optimize = is_optimize_cofi)
    buy_fastk = IntParameter(0, 40, default=20, optimize = is_optimize_cofi)
    buy_fastd = IntParameter(0, 40, default=20, optimize = is_optimize_cofi)
    buy_adx = IntParameter(0, 30, default=30, optimize = is_optimize_cofi)
    buy_ewo_high = DecimalParameter(2, 12, default=3.553, optimize = is_optimize_cofi)

    is_optimize_cofi_protection = False
    buy_cofi_cti = DecimalParameter(-0.9, -0.0, default=-0.5 , optimize = is_optimize_cofi_protection)
    buy_cofi_r14 = DecimalParameter(-100, -44, default=-60 , optimize = is_optimize_cofi_protection)

    is_optimize_gumbo = False
    buy_gumbo_ema = DecimalParameter(0.9, 1.2, default=0.97 , optimize = is_optimize_gumbo)
    buy_gumbo_ewo_low = DecimalParameter(-12.0, 5, default=-5.585, optimize = is_optimize_gumbo)

    is_optimize_gumbo_protection = False
    buy_gumbo_cti = DecimalParameter(-0.9, -0.0, default=-0.5 , optimize = is_optimize_gumbo_protection)
    buy_gumbo_r14 = DecimalParameter(-100, -44, default=-60 , optimize = is_optimize_gumbo_protection)

    is_optimize_sqzmom_protection = False
    buy_sqzmom_ema = DecimalParameter(0.9, 1.2, default=0.97 , optimize = is_optimize_sqzmom_protection)
    buy_sqzmom_ewo = DecimalParameter(-12 , 12, default= 0 , optimize = is_optimize_sqzmom_protection)
    buy_sqzmom_r14 = DecimalParameter(-100, -22, default=-50 , optimize = is_optimize_sqzmom_protection)

    is_optimize_nfix_39 = True
    buy_nfix_39_ema = DecimalParameter(0.9, 1.2, default=0.97 , optimize = is_optimize_nfix_39)

    is_optimize_nfix_49_protection = False
    buy_nfix_49_cti = DecimalParameter(-0.9, -0.0, default=-0.5 , optimize = is_optimize_nfix_49_protection)
    buy_nfix_49_r14 = DecimalParameter(-100, -44, default=-60 , optimize = is_optimize_nfix_49_protection)

    is_optimize_btc_safe = False
    buy_btc_safe = IntParameter(-300, 50, default=-200, optimize = is_optimize_btc_safe)
    buy_btc_safe_1d = DecimalParameter(-0.075, -0.025, default=-0.05, optimize = is_optimize_btc_safe)
    buy_threshold = DecimalParameter(0.003, 0.012, default=0.008, optimize = is_optimize_btc_safe)

    is_optimize_check = False
    buy_roc_1h = IntParameter(-25, 200, default=10, optimize = is_optimize_check)
    buy_bb_width_1h = DecimalParameter(0.3, 2.0, default=0.3, optimize = is_optimize_check)
    
    #BB MODDED
    is_optimize_ctt15_protection = False
    buy_ema_open_mult_15 = DecimalParameter(0.01, 0.03, default=0.024, optimize = is_optimize_ctt15_protection)
    buy_ma_offset_15 = DecimalParameter(0.93, 0.99, default=0.958, optimize = is_optimize_ctt15_protection)
    buy_rsi_15 = DecimalParameter(20.0, 36.0, default=28.0, optimize = is_optimize_ctt15_protection)
    buy_ema_rel_15 = DecimalParameter(0.97, 0.999, default=0.974, optimize = is_optimize_ctt15_protection)
    
    is_optimize_ctt25_protection = False
    buy_25_ma_offset = DecimalParameter(0.90, 0.99, default=0.922, optimize = is_optimize_ctt25_protection)
    buy_25_rsi_4 = DecimalParameter(26.0, 40.0, default=38.0, optimize = is_optimize_ctt25_protection)
    buy_25_cti = DecimalParameter(-0.99, -0.4, default=-0.76, optimize = is_optimize_ctt25_protection)


    ## Slippage params

    is_optimize_slip = False
    max_slip = DecimalParameter(0.33, 1.00, default=0.33, decimals=3, optimize=is_optimize_slip , space='buy', load=True)

    ## Sell params

    sell_btc_safe = IntParameter(-400, -300, default=-365, optimize = False)

    is_optimize_sell_stoploss = False
    sell_cmf = DecimalParameter(-0.4, 0.0, default=0.0, optimize = is_optimize_sell_stoploss)
    sell_ema_close_delta = DecimalParameter(0.022, 0.027, default= 0.024, optimize = is_optimize_sell_stoploss)
    sell_ema = DecimalParameter(0.97, 0.99, default=0.987 , optimize = is_optimize_sell_stoploss)

    is_optimize_deadfish = False
    sell_deadfish_bb_width = DecimalParameter(0.03, 0.75, default=0.05 , optimize = is_optimize_deadfish)
    sell_deadfish_profit = DecimalParameter(-0.15, -0.05, default=-0.05 , optimize = is_optimize_deadfish)
    sell_deadfish_bb_factor = DecimalParameter(0.90, 1.20, default=1.0 , optimize = is_optimize_deadfish)
    sell_deadfish_volume_factor = DecimalParameter(1, 2.5, default=1.0 , optimize = is_optimize_deadfish)

    is_optimize_bleeding = False
    sell_bleeding_cti = DecimalParameter(-0.9, -0.0, default=-0.5 , optimize = is_optimize_bleeding)
    sell_bleeding_r14 = DecimalParameter(-100, -44, default=-60 , optimize = is_optimize_bleeding)
    sell_bleeding_volume_factor = DecimalParameter(1, 2.5, default=1.0 , optimize = is_optimize_bleeding)

    is_optimize_cti_r = False
    sell_cti_r_cti = DecimalParameter(0.55, 1, default=0.5 , optimize = is_optimize_cti_r)
    sell_cti_r_r = DecimalParameter(-15, 0, default=-20 , optimize = is_optimize_cti_r)
    
    ############################################################################

    def informative_pairs(self):
        
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]
        informative_pairs.extend = [(pair, '15m') for pair in pairs]

        return informative_pairs

    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_1h)
        
        # RSI
        informative_1h['rsi_14'] = ta.RSI(informative_1h, timeperiod=14)

        # SMA
        informative_1h['sma_200'] = ta.SMA(informative_1h, timeperiod=200)

        informative_1h['sma_200_dec_20'] = informative_1h['sma_200'] < informative_1h['sma_200'].shift(20)
        informative_1h['sma_200_dec_24'] = informative_1h['sma_200'] < informative_1h['sma_200'].shift(24) 

        # EMA
        informative_1h['ema_8'] = ta.EMA(informative_1h, timeperiod=8)
        informative_1h['ema_50'] = ta.EMA(informative_1h, timeperiod=50)
        informative_1h['ema_100'] = ta.EMA(informative_1h, timeperiod=100)
        informative_1h['ema_200'] = ta.EMA(informative_1h, timeperiod=200)
        informative_1h['ema_20'] = ta.EMA(informative_1h, timeperiod=20)
        informative_1h['ema_26'] = ta.EMA(informative_1h, timeperiod=26)
        informative_1h['ema_12'] = ta.EMA(informative_1h, timeperiod=12)
        informative_1h['ema_25'] = ta.EMA(informative_1h, timeperiod=25)
        informative_1h['ema_35'] = ta.EMA(informative_1h, timeperiod=35)

        # CTI
        informative_1h['cti'] = pta.cti(informative_1h["close"], length=20)
        informative_1h['cti_40'] = pta.cti(informative_1h["close"], length=40)

        # BB
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(informative_1h), window=20, stds=2)
        informative_1h['bb20_2_low'] = bollinger['lower']
        informative_1h['bb20_2_mid'] = bollinger['mid']
        informative_1h['bb20_2_upp'] = bollinger['upper']
        informative_1h['bb20_width'] = ((informative_1h['bb20_2_upp'] - informative_1h['bb20_2_low']) / informative_1h['bb20_2_mid'])

        # CRSI (3, 2, 100)
        crsi_closechange = informative_1h['close'] / informative_1h['close'].shift(1)
        crsi_updown = np.where(crsi_closechange.gt(1), 1.0, np.where(crsi_closechange.lt(1), -1.0, 0.0))
        informative_1h['crsi'] =  (ta.RSI(informative_1h['close'], timeperiod=3) + ta.RSI(crsi_updown, timeperiod=2) + ta.ROC(informative_1h['close'], 100)) / 3

        # Williams %R
        informative_1h['r_96'] = williams_r(informative_1h, period=96)
        informative_1h['r_480'] = williams_r(informative_1h, period=480)

        # Bollinger bands
        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(informative_1h), window=20, stds=2)
        informative_1h['bb_lowerband2'] = bollinger2['lower']
        informative_1h['bb_middleband2'] = bollinger2['mid']
        informative_1h['bb_upperband2'] = bollinger2['upper']
        informative_1h['bb_width'] = ((informative_1h['bb_upperband2'] - informative_1h['bb_lowerband2']) / informative_1h['bb_middleband2'])

        # ROC
        informative_1h['roc'] = ta.ROC(dataframe, timeperiod=9)

        # MOMDIV
        mom = momdiv(informative_1h)
        informative_1h['momdiv_buy'] = mom['momdiv_buy']
        informative_1h['momdiv_sell'] = mom['momdiv_sell']
        informative_1h['momdiv_coh'] = mom['momdiv_coh']
        informative_1h['momdiv_col'] = mom['momdiv_col']

        # RSI
        informative_1h['rsi'] = ta.RSI(informative_1h, timeperiod=14)

        # CMF
        informative_1h['cmf'] = chaikin_money_flow(informative_1h, 20)

        # Heikin Ashi
        inf_heikinashi = qtpylib.heikinashi(informative_1h)
        informative_1h['ha_close'] = inf_heikinashi['close']
        informative_1h['rocr'] = ta.ROCR(informative_1h['ha_close'], timeperiod=168)

        # T3 Average
        informative_1h['T3'] = T3(informative_1h)

        # Elliot
        informative_1h['EWO'] = EWO(informative_1h, 50, 200)

        # nfi 37
        informative_1h['hl_pct_change_5'] = range_percent_change(informative_1h, 'HL', 5)
        informative_1h['low_5'] = informative_1h['low'].shift().rolling(5).min()
        informative_1h['safe_dump_50'] = ((informative_1h['hl_pct_change_5'] < 0.66) | (informative_1h['close'] < informative_1h['low_5']) | (informative_1h['close'] > informative_1h['open']))

        # Pump protections
        #informative_1h['hl_pct_change_48'] = range_percent_change(informative_1h, 'HL', length=48)
        #informative_1h['hl_pct_change_36'] = range_percent_change(informative_1h, 'HL', length=36)
        #informative_1h['hl_pct_change_24'] = range_percent_change(informative_1h, 'HL', length=24)
        #informative_1h['hl_pct_change_12'] = range_percent_change(informative_1h, 'HL', length=12)
        #informative_1h['hl_pct_change_6'] = range_percent_change(informative_1h, 'HL', length=6)

        return informative_1h

    def informative_15m_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        tik = time.perf_counter()
        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative_15m = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.info_timeframe_15m)

        # RSI
        informative_15m['rsi_14'] = ta.RSI(informative_15m, timeperiod=14)

        # EMAs
        informative_15m['ema_12'] = ta.EMA(informative_15m, timeperiod=12)
        informative_15m['ema_16'] = ta.EMA(informative_15m, timeperiod=16)
        informative_15m['ema_20'] = ta.EMA(informative_15m, timeperiod=20)
        informative_15m['ema_25'] = ta.EMA(informative_15m, timeperiod=25)
        informative_15m['ema_26'] = ta.EMA(informative_15m, timeperiod=26)
        informative_15m['ema_50'] = ta.EMA(informative_15m, timeperiod=50)
        informative_15m['ema_100'] = ta.EMA(informative_15m, timeperiod=100)
        informative_15m['ema_200'] = ta.EMA(informative_15m, timeperiod=200)

        # SMA
        informative_15m['sma_15'] = ta.SMA(informative_15m, timeperiod=15)
        informative_15m['sma_30'] = ta.SMA(informative_15m, timeperiod=30)
        informative_15m['sma_200'] = ta.SMA(informative_15m, timeperiod=200)

        informative_15m['sma_200_dec_20'] = informative_15m['sma_200'] < informative_15m['sma_200'].shift(20)
        
                # BB
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(informative_15m), window=20, stds=2)
        informative_15m['bb20_2_low'] = bollinger['lower']
        informative_15m['bb20_2_mid'] = bollinger['mid']
        informative_15m['bb20_2_upp'] = bollinger['upper']

        # BB 40 - STD2
        bb_40_std2 = qtpylib.bollinger_bands(informative_15m['close'], window=40, stds=2)
        informative_15m['bb40_2_low'] = bb_40_std2['lower']
        informative_15m['bb40_2_mid'] = bb_40_std2['mid']
        informative_15m['bb40_2_delta'] = (bb_40_std2['mid'] - informative_15m['bb40_2_low']).abs()
        informative_15m['closedelta'] = (informative_15m['close'] - informative_15m['close'].shift()).abs()
        informative_15m['tail'] = (informative_15m['close'] - informative_15m['bb40_2_low']).abs()

        # CMF
        informative_15m['cmf'] = chaikin_money_flow(informative_15m, 20)

        # CTI
        informative_15m['cti'] = pta.cti(informative_15m["close"], length=20)

        # Williams %R
        informative_15m['r_14'] = williams_r(informative_15m, period=14)
        informative_15m['r_64'] = williams_r(informative_15m, period=64)
        informative_15m['r_96'] = williams_r(informative_15m, period=96)

        # EWO
        informative_15m['ewo'] = ewo(informative_15m, 50, 200)

        # CCI
        informative_15m['cci'] = ta.CCI(informative_15m, source='hlc3', timeperiod=20)

        # CRSI (3, 2, 100)
        crsi_closechange = informative_15m['close'] / informative_15m['close'].shift(1)
        crsi_updown = np.where(crsi_closechange.gt(1), 1.0, np.where(crsi_closechange.lt(1), -1.0, 0.0))
        informative_15m['crsi'] =  (ta.RSI(informative_15m['close'], timeperiod=3) + ta.RSI(crsi_updown, timeperiod=2) + ta.ROC(informative_15m['close'], 100)) / 3

        tok = time.perf_counter()
        log.debug(f"[{metadata['pair']}] informative_1h_indicators took: {tok - tik:0.4f} seconds.")

        return informative_15m
        
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        sl_new = 1

        if (current_profit > 0.2):
            sl_new = 0.05
        elif (current_profit > 0.1):
            sl_new = 0.03
        elif (current_profit > 0.06):
            sl_new = 0.02
        elif (current_profit > 0.03):
            sl_new = 0.015

        return sl_new

    # From NFIX
    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        last_candle = dataframe.iloc[-1]
        previous_candle_1 = dataframe.iloc[-2]
        previous_candle_2 = dataframe.iloc[-3]

        max_profit = ((trade.max_rate - trade.open_rate) / trade.open_rate)
        max_loss = ((trade.open_rate - trade.min_rate) / trade.min_rate)

        buy_tag = 'empty'
        if hasattr(trade, 'buy_tag') and trade.buy_tag is not None:
            buy_tag = trade.buy_tag
        buy_tags = buy_tag.split()

        # sell trail
        if 0.012 > current_profit >= 0.0:
            if (max_profit > (current_profit + 0.045)) and (last_candle['rsi'] < 46.0):
                return f"sell_profit_t_0_1( {buy_tag})"
            elif (max_profit > (current_profit + 0.025)) and (last_candle['rsi'] < 32.0):
                return f"sell_profit_t_0_2( {buy_tag})"
            elif (max_profit > (current_profit + 0.05)) and (last_candle['rsi'] < 48.0):
                return f"sell_profit_t_0_3( {buy_tag})"
        elif 0.02 > current_profit >= 0.012:
            if (max_profit > (current_profit + 0.01)) and (last_candle['rsi'] < 39.0):
                return f"sell_profit_t_1_1( {buy_tag})"
            elif (max_profit > (current_profit + 0.035)) and (last_candle['rsi'] < 45.0) and (last_candle['cmf'] < -0.0) and (last_candle['cmf_1h'] < -0.0):
                return f"sell_profit_t_1_2( {buy_tag})"
            elif (max_profit > (current_profit + 0.02)) and (last_candle['rsi'] < 40.0) and (last_candle['cmf'] < -0.0) and (last_candle['cti_1h'] > 0.8):
                return f"sell_profit_t_1_4( {buy_tag})"
            elif (max_profit > (current_profit + 0.04)) and (last_candle['rsi'] < 49.0) and (last_candle['cmf_1h'] < -0.0):
                return f"sell_profit_t_1_5( {buy_tag})"
            elif (max_profit > (current_profit + 0.06)) and (last_candle['rsi'] < 43.0) and (last_candle['cmf'] < -0.0):
                return f"sell_profit_t_1_7( {buy_tag})"
            elif (max_profit > (current_profit + 0.025)) and (last_candle['rsi'] < 40.0) and (last_candle['cmf'] < -0.1) and (last_candle['rsi_1h'] < 50.0):
                return f"sell_profit_t_1_9( {buy_tag})"
            elif (max_profit > (current_profit + 0.025)) and (last_candle['rsi'] < 46.0) and (last_candle['cmf'] < -0.0) and (last_candle['r_480_1h'] > -20.0):
                return f"sell_profit_t_1_10( {buy_tag})"
            elif (max_profit > (current_profit + 0.025)) and (last_candle['rsi'] < 42.0):
                return f"sell_profit_t_1_11( {buy_tag})"
            elif (max_profit > (current_profit + 0.01)) and (last_candle['rsi'] < 44.0) and (last_candle['cmf'] < -0.25):
                return f"sell_profit_t_1_12( {buy_tag})"

        # sell cti_r
        if 0.012 > current_profit >= 0.0 :
            if (last_candle['cti'] > self.sell_cti_r_cti.value) and (last_candle['r_14'] > self.sell_cti_r_r.value):
                return f"sell_profit_t_cti_r_0_1( {buy_tag})"

        # main sell
        if current_profit > 0.02:
            if (last_candle['momdiv_sell_1h'] == True):
                return f"signal_profit_q_momdiv_1h( {buy_tag})"
            if (last_candle['momdiv_sell'] == True):
                return f"signal_profit_q_momdiv( {buy_tag})"
            if (last_candle['momdiv_coh'] == True):
                return f"signal_profit_q_momdiv_coh( {buy_tag})"

        # sell bear
        if last_candle['close'] < last_candle['ema_200']:
            if 0.02 > current_profit >= 0.01:
                if (last_candle['rsi'] < 34.0) and (last_candle['cmf'] < 0.0):
                    return f"sell_profit_u_bear_1_1( {buy_tag})"
                elif (last_candle['rsi'] < 44.0) and (last_candle['cmf'] < -0.4):
                    return f"sell_profit_u_bear_1_2( {buy_tag})"

        # sell quick
        if (0.06 > current_profit > 0.02) and (last_candle['rsi'] > 80.0):
            return f"signal_profit_q_1( {buy_tag})"

        if (0.06 > current_profit > 0.02) and (last_candle['cti'] > 0.95):
            return f"signal_profit_q_2( {buy_tag})"

        if (0.06 > current_profit > 0.02) and (last_candle['pm'] <= last_candle['pmax_thresh']) and (last_candle['close'] > last_candle['sma_21'] * 1.1):
            return f"signal_profit_q_pmax_bull( {buy_tag})"
        if (0.06 > current_profit > 0.02) and (last_candle['pm'] > last_candle['pmax_thresh']) and (last_candle['close'] > last_candle['sma_21'] * 1.016):
            return f"signal_profit_q_pmax_bear( {buy_tag})"

        # sell scalp
        if (current_profit > 0 and buy_tag in [ 'nfix_39 ']):
            if (
                    (current_profit > 0)
                    and (last_candle['fisher'] > 0.39075)
                    and (last_candle['ha_high'] <= previous_candle_1['ha_high'])
                    and (previous_candle_1['ha_high'] <= previous_candle_2['ha_high'])
                    and (last_candle['ha_close'] <= previous_candle_1['ha_close'])
                    and (last_candle['ema_4'] > last_candle['ha_close'])
                    and (last_candle['ha_close'] * 0.99754 > last_candle['bb_middleband2'])
                ):
                return f"sell_scalp( {buy_tag})"

        if (
                (current_profit < -0.05)
                and (last_candle['close'] < last_candle['ema_200'] * 0.988)
                and (last_candle['cmf'] < -0.046)
                and (((last_candle['ema_200'] - last_candle['close']) / last_candle['close']) < 0.022)
                and last_candle['rsi'] > previous_candle_1['rsi']
                and (last_candle['rsi'] > (last_candle['rsi_1h'] + 10.0))
            ):
            return f"sell_stoploss_u_e_1( {buy_tag})"

        # stoploss - deadfish
        if (    (current_profit < self.sell_deadfish_profit.value)
                and (last_candle['close'] < last_candle['ema_200'])
                and (last_candle['bb_width'] < self.sell_deadfish_bb_width.value)
                and (last_candle['close'] > last_candle['bb_middleband2'] * self.sell_deadfish_bb_factor.value)
                and (last_candle['volume_mean_12'] < last_candle['volume_mean_24'] * self.sell_deadfish_volume_factor.value)
            ):
            return f"sell_stoploss_deadfish( {buy_tag})"

        # stoploss - bleeding
        #if (    (current_profit < -0.05)
                #and (last_candle['close'] < last_candle['ema_200'])
                #and (last_candle['cti_mean_24'] < self.sell_bleeding_cti.value)
                #and (last_candle['r_14_mean_24'] < self.sell_bleeding_r14.value)
                #and (last_candle['volume_mean_12'] < last_candle['volume_mean_24'] * self.sell_bleeding_volume_factor.value)
            #):
            #return f"sell_stoploss_bleeding( {buy_tag})"

        return None

    ## Confirm Entry
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, **kwargs) -> bool:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        max_slip = self.max_slip.value

        if(len(dataframe) < 1):
            return False

        dataframe = dataframe.iloc[-1].squeeze()
        if ((rate > dataframe['close'])) :

            slippage = ( (rate / dataframe['close']) - 1 ) * 100

            if slippage < max_slip:
                return True
            else:
                return False

        return True

    ############################################################################

    def normal_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    
        # RSI
        dataframe['rsi_4'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_20'] = ta.RSI(dataframe, timeperiod=20)
        
        # Zero-Lag EMA
        dataframe['zema_61'] = zema(dataframe, period=61)
        
        # Bollinger bands
        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband2'] = bollinger2['lower']
        dataframe['bb_middleband2'] = bollinger2['mid']
        dataframe['bb_upperband2'] = bollinger2['upper']

        bollinger3 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=3)
        dataframe['bb_lowerband3'] = bollinger3['lower']
        dataframe['bb_middleband3'] = bollinger3['mid']
        dataframe['bb_upperband3'] = bollinger3['upper']

        # BB 40 - STD2
        bb_40_std2 = qtpylib.bollinger_bands(dataframe['close'], window=40, stds=2)
        dataframe['bb40_2_low'] = bb_40_std2['lower']
        dataframe['bb40_2_mid'] = bb_40_std2['mid']
        dataframe['bb40_2_delta'] = (bb_40_std2['mid'] - dataframe['bb40_2_low']).abs()
        dataframe['closedelta'] = (dataframe['close'] - dataframe['close'].shift()).abs()
        dataframe['tail'] = (dataframe['close'] - dataframe['bb40_2_low']).abs()

        # BB 20 - STD2
        bb_20_std2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb20_2_low'] = bb_20_std2['lower']
        dataframe['bb20_2_mid'] = bb_20_std2['mid']
        dataframe['bb20_2_upp'] = bb_20_std2['upper']

        # BB 20 - STD3
        bb_20_std3 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=3)
        dataframe['bb20_3_low'] = bb_20_std3['lower']
        dataframe['bb20_3_mid'] = bb_20_std3['mid']
        dataframe['bb20_3_upp'] = bb_20_std3['upper']

        ### Other BB checks
        dataframe['bb_width'] = ((dataframe['bb_upperband2'] - dataframe['bb_lowerband2']) / dataframe['bb_middleband2'])
        dataframe['bb_delta'] = ((dataframe['bb_lowerband2'] - dataframe['bb_lowerband3']) / dataframe['bb_lowerband2'])

        # CCI hyperopt
        for val in self.buy_cci_length.range:
            dataframe[f'cci_length_{val}'] = ta.CCI(dataframe, val)

        dataframe['cci'] = ta.CCI(dataframe, 26)
        dataframe['cci_long'] = ta.CCI(dataframe, 170)

        # RMI hyperopt
        for val in self.buy_rmi_length.range:
            dataframe[f'rmi_length_{val}'] = RMI(dataframe, length=val, mom=4)

        # SRSI hyperopt
        stoch = ta.STOCHRSI(dataframe, 15, 20, 2, 2)
        dataframe['srsi_fk'] = stoch['fastk']
        dataframe['srsi_fd'] = stoch['fastd']

        # BinH
        dataframe['closedelta'] = (dataframe['close'] - dataframe['close'].shift()).abs()

        # SMA
        dataframe['bb9'] = ta.SMA(dataframe, timeperiod=9)
        dataframe['sma_15'] = ta.SMA(dataframe, timeperiod=15)
        dataframe['sma_20'] = ta.SMA(dataframe, timeperiod=20)
        dataframe['sma_21'] = ta.SMA(dataframe, timeperiod=21)
        dataframe['sma_28'] = ta.SMA(dataframe, timeperiod=28)
        dataframe['sma_30'] = ta.SMA(dataframe, timeperiod=30)
        dataframe['sma_75'] = ta.SMA(dataframe, timeperiod=75)

        # CTI
        dataframe['cti'] = pta.cti(dataframe["close"], length=20)
        
        # CMF
        dataframe['cmf'] = chaikin_money_flow(dataframe, 20)

        # CRSI (3, 2, 100)
        crsi_closechange = dataframe['close'] / dataframe['close'].shift(1)
        crsi_updown = np.where(crsi_closechange.gt(1), 1.0, np.where(crsi_closechange.lt(1), -1.0, 0.0))
        dataframe['crsi'] =  (ta.RSI(dataframe['close'], timeperiod=3) + ta.RSI(crsi_updown, timeperiod=2) + ta.ROC(dataframe['close'], 100)) / 3

        # EMA
        dataframe['ema_4'] = ta.EMA(dataframe, timeperiod=4)
        dataframe['ema_8'] = ta.EMA(dataframe, timeperiod=8)
        dataframe['ema_12'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema_13'] = ta.EMA(dataframe, timeperiod=13)
        dataframe['ema_16'] = ta.EMA(dataframe, timeperiod=16)
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_26'] = ta.EMA(dataframe, timeperiod=26)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        # Elliot
        dataframe['EWO'] = EWO(dataframe, 50, 200)

        # Williams %R
        dataframe['r_14'] = williams_r(dataframe, period=14)
        dataframe['r_32'] = williams_r(dataframe, period=32)
        dataframe['r_64'] = williams_r(dataframe, period=64)
        dataframe['r_96'] = williams_r(dataframe, period=96)
        dataframe['r_480'] = williams_r(dataframe, period=480)

        # Volume
        dataframe['volume_mean_4'] = dataframe['volume'].rolling(4).mean().shift(1)
        dataframe['volume_mean_12'] = dataframe['volume'].rolling(12).mean().shift(1)
        dataframe['volume_mean_24'] = dataframe['volume'].rolling(24).mean().shift(1)

        # MFI
        dataframe['mfi'] = ta.MFI(dataframe)

        # Heiken Ashi
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        ## BB 40
        bollinger2_40 = qtpylib.bollinger_bands(ha_typical_price(dataframe), window=40, stds=2)
        dataframe['bb_lowerband2_40'] = bollinger2_40['lower']
        dataframe['bb_middleband2_40'] = bollinger2_40['mid']
        dataframe['bb_upperband2_40'] = bollinger2_40['upper']

        # ClucHA
        dataframe['bb_delta_cluc'] = (dataframe['bb_middleband2_40'] - dataframe['bb_lowerband2_40']).abs()
        dataframe['ha_closedelta'] = (dataframe['ha_close'] - dataframe['ha_close'].shift()).abs()
        dataframe['tail'] = (dataframe['ha_close'] - dataframe['ha_low']).abs()
        dataframe['ema_slow'] = ta.EMA(dataframe['ha_close'], timeperiod=50)
        dataframe['rocr'] = ta.ROCR(dataframe['ha_close'], timeperiod=28)

        # Cofi
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        dataframe['adx'] = ta.ADX(dataframe)

        # Profit Maximizer - PMAX
        dataframe['pm'], dataframe['pmx'] = pmax(heikinashi, MAtype=1, length=9, multiplier=27, period=10, src=3)
        dataframe['source'] = (dataframe['high'] + dataframe['low'] + dataframe['open'] + dataframe['close'])/4
        dataframe['pmax_thresh'] = ta.EMA(dataframe['source'], timeperiod=9)

        # MOMDIV
        mom = momdiv(dataframe)
        dataframe['momdiv_buy'] = mom['momdiv_buy']
        dataframe['momdiv_sell'] = mom['momdiv_sell']
        dataframe['momdiv_coh'] = mom['momdiv_coh']
        dataframe['momdiv_col'] = mom['momdiv_col']

        # T3 Average
        dataframe['T3'] = T3(dataframe)

        # True range
        dataframe['trange'] = ta.TRANGE(dataframe)

        # KC
        dataframe['range_ma_28'] = ta.SMA(dataframe['trange'], 28)
        dataframe['kc_upperband_28_1'] = dataframe['sma_28'] + dataframe['range_ma_28']
        dataframe['kc_lowerband_28_1'] = dataframe['sma_28'] - dataframe['range_ma_28']

        # KC 20
        dataframe['range_ma_20'] = ta.SMA(dataframe['trange'], 20)
        dataframe['kc_upperband_20_2'] = dataframe['sma_20'] + dataframe['range_ma_20'] * 2
        dataframe['kc_lowerband_20_2'] = dataframe['sma_20'] - dataframe['range_ma_20'] * 2
        dataframe['kc_bb_delta'] =  ( dataframe['kc_lowerband_20_2'] - dataframe['bb_lowerband2'] ) / dataframe['bb_lowerband2'] * 100

        # Linreg
        dataframe['hh_20'] = ta.MAX(dataframe['high'], 20)
        dataframe['ll_20'] = ta.MIN(dataframe['low'], 20)
        dataframe['avg_hh_ll_20'] = (dataframe['hh_20'] + dataframe['ll_20']) / 2
        dataframe['avg_close_20'] = ta.SMA(dataframe['close'], 20)
        dataframe['avg_val_20'] = (dataframe['avg_hh_ll_20'] + dataframe['avg_close_20']) / 2
        dataframe['linreg_val_20'] = ta.LINEARREG(dataframe['close'] - dataframe['avg_val_20'], 20, 0)

        # fisher
        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe["fisher"] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

        # Modified Elder Ray Index
        dataframe['moderi_96'] = moderi(dataframe, 96)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # The indicators for the 1h informative timeframe
        informative_1h = self.informative_1h_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(dataframe, informative_1h, self.timeframe, self.inf_1h, ffill=True)

        # The indicators for the normal (5m) timeframe
        dataframe = self.normal_tf_indicators(dataframe, metadata)
        
        informative_15m = self.informative_15m_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(dataframe, informative_15m, self.timeframe, self.inf_15m, ffill=True)
        
        return dataframe
        
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []
        dataframe.loc[:, 'buy_tag'] = ''
        
        is_additional_check = (
                (dataframe['roc_1h'] < self.buy_roc_1h.value) &
                (dataframe['bb_width_1h'] < self.buy_bb_width_1h.value)
            )        

        is_dip = (
                (dataframe[f'rmi_length_{self.buy_rmi_length.value}'] < self.buy_rmi.value) &
                (dataframe[f'cci_length_{self.buy_cci_length.value}'] <= self.buy_cci.value) &
                (dataframe['srsi_fk'] < self.buy_srsi_fk.value)
            )

        is_sqzOff = (
                (dataframe['bb_lowerband2'] < dataframe['kc_lowerband_28_1']) &
                (dataframe['bb_upperband2'] > dataframe['kc_upperband_28_1'])
            )

        is_break = (

                (dataframe['bb_delta'] > self.buy_bb_delta.value) &
                (dataframe['bb_width'] > self.buy_bb_width.value) &
                (dataframe['closedelta'] > dataframe['close'] * self.buy_closedelta.value / 1000 ) &    # from BinH
                (dataframe['close'] < dataframe['bb_lowerband3'] * self.buy_bb_factor.value)
            )

        is_local_uptrend = (  
                is_additional_check &        # from NFI next gen, credit goes to @iterativ
                (dataframe['ema_26'] > dataframe['ema_12']) &
                (dataframe['ema_26'] - dataframe['ema_12'] > dataframe['open'] * self.buy_ema_diff.value) &
                (dataframe['ema_26'].shift() - dataframe['ema_12'].shift() > dataframe['open'] / 100) &
                (dataframe['close'] < dataframe['bb_lowerband2'] * self.buy_bb_factor.value) &
                (dataframe['closedelta'] > dataframe['close'] * self.buy_closedelta.value / 1000 )
            )

        is_local_dip = (
                is_additional_check &        
                (dataframe['ema_26'] > dataframe['ema_12']) &
                (dataframe['ema_26'] - dataframe['ema_12'] > dataframe['open'] * self.buy_ema_diff_local_dip.value) &
                (dataframe['ema_26'].shift() - dataframe['ema_12'].shift() > dataframe['open'] / 100) &
                (dataframe['close'] < dataframe['ema_20'] * self.buy_ema_high_local_dip.value) &
                (dataframe['rsi'] < self.buy_rsi_local_dip.value) &
                (dataframe['crsi'] > self.buy_crsi_local_dip.value) &
                (dataframe['closedelta'] > dataframe['close'] * self.buy_closedelta_local_dip.value / 1000 )
            )

        is_ewo = (        
                is_additional_check &        # from SMA offset
                (dataframe['rsi_fast'] < self.buy_rsi_fast.value) &
                (dataframe['close'] < dataframe['ema_8'] * self.buy_ema_low.value) &
                (dataframe['EWO'] > self.buy_ewo.value) &
                (dataframe['close'] < dataframe['ema_16'] * self.buy_ema_high.value) &
                (dataframe['rsi'] < self.buy_rsi.value)
            )

        is_ewo_2 = (
                is_additional_check &        
                (dataframe['ema_200_1h'] > dataframe['ema_200_1h'].shift(12)) &
                (dataframe['ema_200_1h'].shift(12) > dataframe['ema_200_1h'].shift(24)) &
                (dataframe['rsi_fast'] < self.buy_rsi_fast_ewo_2.value) &
                (dataframe['close'] < dataframe['ema_8'] * self.buy_ema_low_2.value) &
                (dataframe['EWO'] > self.buy_ewo_high_2.value) &
                (dataframe['close'] < dataframe['ema_16'] * self.buy_ema_high_2.value) &
                (dataframe['rsi'] < self.buy_rsi_ewo_2.value)
            )

        is_r_deadfish = (  
                is_additional_check &        # reverse deadfish
                (dataframe['ema_100'] < dataframe['ema_200'] * self.buy_r_deadfish_ema.value) &
                (dataframe['bb_width'] > self.buy_r_deadfish_bb_width.value) &
                (dataframe['close'] < dataframe['bb_middleband2'] * self.buy_r_deadfish_bb_factor.value) &
                (dataframe['volume_mean_12'] > dataframe['volume_mean_24'] * self.buy_r_deadfish_volume_factor.value) &
                (dataframe['cti'] < self.buy_r_deadfish_cti.value) &
                (dataframe['r_14'] < self.buy_r_deadfish_r14.value)
            )

        is_clucHA = (
                (dataframe['rocr_1h'] > self.buy_clucha_rocr_1h.value ) &
                (
                        (dataframe['bb_lowerband2_40'].shift() > 0) &
                        (dataframe['bb_delta_cluc'] > dataframe['ha_close'] * self.buy_clucha_bbdelta_close.value) &
                        (dataframe['ha_closedelta'] > dataframe['ha_close'] * self.buy_clucha_closedelta_close.value) &
                        (dataframe['tail'] < dataframe['bb_delta_cluc'] * self.buy_clucha_bbdelta_tail.value) &
                        (dataframe['ha_close'] < dataframe['bb_lowerband2_40'].shift()) &
                        (dataframe['ha_close'] < dataframe['ha_close'].shift())
                )
            )

        is_cofi = (    
                is_additional_check &        # Modified from cofi, credit goes to original author "slack user CofiBit"
                (dataframe['open'] < dataframe['ema_8'] * self.buy_ema_cofi.value) &
                (qtpylib.crossed_above(dataframe['fastk'], dataframe['fastd'])) &
                (dataframe['fastk'] < self.buy_fastk.value) &
                (dataframe['fastd'] < self.buy_fastd.value) &
                (dataframe['adx'] > self.buy_adx.value) &
                (dataframe['EWO'] > self.buy_ewo_high.value) &
                (dataframe['cti'] < self.buy_cofi_cti.value) &
                (dataframe['r_14'] < self.buy_cofi_r14.value)
            )

        is_gumbo = (    
                is_additional_check &        # Modified from gumbo1, creadit goes to original author @raph92
                (dataframe['EWO'] < self.buy_gumbo_ewo_low.value) &
                (dataframe['bb_middleband2_1h'] >= dataframe['T3_1h']) &
                (dataframe['T3'] <= dataframe['ema_8'] * self.buy_gumbo_ema.value) &
                (dataframe['cti'] < self.buy_gumbo_cti.value) &
                (dataframe['r_14'] < self.buy_gumbo_r14.value)
            )

        is_sqzmom = (     
                is_additional_check &        # Modified from squeezeMomentum, credit goes to original author @LazyBear of TradingView
                (is_sqzOff) &
                (dataframe['linreg_val_20'].shift(2) > dataframe['linreg_val_20'].shift(1)) &
                (dataframe['linreg_val_20'].shift(1) < dataframe['linreg_val_20']) &
                (dataframe['linreg_val_20'] < 0) &
                (dataframe['close'] < dataframe['ema_13'] * self.buy_sqzmom_ema.value) &
                (dataframe['EWO'] < self.buy_sqzmom_ewo.value) &
                (dataframe['r_14'] < self.buy_sqzmom_r14.value)
            )

        # NFI quick mode, credit goes to @iterativ
        is_nfi_13 = (
                is_additional_check &        
                (dataframe['ema_50_1h'] > dataframe['ema_100_1h']) &
                (dataframe['close'] < dataframe['sma_30'] * 0.99) &
                (dataframe['cti'] < -0.92) &
                (dataframe['EWO'] < -5.585) &
                (dataframe['cti_1h'] < -0.88) &
                (dataframe['crsi_1h'] > 10.0)
            )

        is_nfi_32 = (
                is_additional_check &        
                (dataframe['rsi_slow'] < dataframe['rsi_slow'].shift(1)) &
                (dataframe['rsi_fast'] < 46) &
                (dataframe['rsi'] > 25.0) &
                (dataframe['close'] < dataframe['sma_15'] * 0.93) &
                (dataframe['cti'] < -0.9)
            )

        is_nfi_33 = (
                is_additional_check &        
                (dataframe['close'] < (dataframe['ema_13'] * 0.978)) &
                (dataframe['EWO'] > 8) &
                (dataframe['cti'] < -0.88) &
                (dataframe['rsi'] < 32) &
                (dataframe['r_14'] < -98.0) &
                (dataframe['volume'] < (dataframe['volume_mean_4'] * 2.5))
            )

        is_nfi_38 = (
                is_additional_check &
                (dataframe['pm'] > dataframe['pmax_thresh']) &
                (dataframe['close'] < dataframe['sma_75'] * 0.98) &
                (dataframe['EWO'] < -4.4) &
                (dataframe['cti'] < -0.95) &
                (dataframe['r_14'] < -97) &
                (dataframe['crsi_1h'] > 0.5)
            )

        is_nfix_5 = (
                is_additional_check &
                (dataframe['ema_200_1h'] > dataframe['ema_200_1h'].shift(12)) &
                (dataframe['ema_200_1h'].shift(12) > dataframe['ema_200_1h'].shift(24)) &
                (dataframe['close'] < dataframe['sma_75'] * 0.932) &
                (dataframe['EWO'] > 3.6) &
                (dataframe['cti'] < -0.9) &
                (dataframe['r_14'] < -97.0)
            )

        is_nfix_39 = (
                is_additional_check &        
                (dataframe['ema_200_1h'] > dataframe['ema_200_1h'].shift(12)) &
                (dataframe['ema_200_1h'].shift(12) > dataframe['ema_200_1h'].shift(24)) &
                (dataframe['bb_lowerband2_40'].shift().gt(0)) &
                (dataframe['bb_delta_cluc'].gt(dataframe['close'] * 0.056)) &
                (dataframe['closedelta'].gt(dataframe['close'] * 0.01)) &
                (dataframe['tail'].lt(dataframe['bb_delta_cluc'] * 0.5)) &
                (dataframe['close'].lt(dataframe['bb_lowerband2_40'].shift())) &
                (dataframe['close'].le(dataframe['close'].shift())) &
                (dataframe['close'] > dataframe['ema_13'] * self.buy_nfix_39_ema.value)
            )

        is_nfix_49 = (
                is_additional_check &        
                (dataframe['ema_26'].shift(3) > dataframe['ema_12'].shift(3)) &
                (dataframe['ema_26'].shift(3) - dataframe['ema_12'].shift(3) > dataframe['open'].shift(3) * 0.032) &
                (dataframe['ema_26'].shift(9) - dataframe['ema_12'].shift(9) > dataframe['open'].shift(3) / 100) &
                (dataframe['close'].shift(3) < dataframe['ema_20'].shift(3) * 0.916) &
                (dataframe['rsi'].shift(3) < 32.5) &
                (dataframe['crsi'].shift(3) > 18.0) &
                (dataframe['cti'] < self.buy_nfix_49_cti.value) &
                (dataframe['r_14'] < self.buy_nfix_49_r14.value)
            )

        is_nfi7_33 = (
                is_additional_check &        
                (dataframe['moderi_96']) &
                (dataframe['cti'] < -0.88) &
                (dataframe['close'] < (dataframe['ema_13'] * 0.988)) &
                (dataframe['EWO'] > 6.4) &
                (dataframe['rsi'] < 32.0) &
                (dataframe['volume'] < (dataframe['volume_mean_4'] * 2.0))
            )

        is_nfi7_37 = (
                is_additional_check &        
                (dataframe['pm'] > dataframe['pmax_thresh']) &
                (dataframe['close'] < dataframe['sma_75'] * 0.98) &
                (dataframe['EWO'] > 9.8) &
                (dataframe['rsi'] < 56.0) &
                (dataframe['cti'] < -0.7) &
                (dataframe['safe_dump_50_1h'])
            )
            ## BB MODDED
        is_nfi_ctt35 = (
                is_additional_check &        
                (dataframe['pm'] <= dataframe['pmax_thresh']) &
                (dataframe['close'] < dataframe['sma_75'] * 0.984) &
                (dataframe['EWO'] > 9.6) &
                (dataframe['rsi_14'] < 32.0) &
                (dataframe['cti'] < -0.5)
            )
    
        is_nfi_ctt25 = (
                is_additional_check &        
                (dataframe['rsi_20'] < dataframe['rsi_20'].shift()) &
                (dataframe['rsi_4'] < self.buy_25_rsi_4.value) &
                (dataframe['ema_20_1h'] > dataframe['ema_26_1h']) &
                (dataframe['close'] < (dataframe['sma_20'] * self.buy_25_ma_offset.value)) &
                (dataframe['open'] > (dataframe['sma_20'] * self.buy_25_ma_offset.value)) &
                (
                    (dataframe['open'] < dataframe['ema_20_1h']) & (dataframe['low'] < dataframe['ema_20_1h']) |
                    (dataframe['open'] > dataframe['ema_20_1h']) & (dataframe['low'] > dataframe['ema_20_1h'])
                ) &
                (dataframe['cti'] < self.buy_25_cti.value)
            )
                
        is_nfi_ctt15 = (
                is_additional_check &        
                (dataframe['close'] > dataframe['ema_200_1h'] * self.buy_ema_rel_15.value) &
                (dataframe['ema_26'] > dataframe['ema_12']) &
                ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_ema_open_mult_15.value)) &
                ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open'] / 100)) &
                (dataframe['rsi_14'] < self.buy_rsi_15.value) &
                (dataframe['close'] < dataframe['ema_20'] * self.buy_ma_offset_15.value)
            )
            
        is_nfi_9 = (
                is_additional_check &        
                (dataframe['ema_50'] > dataframe['ema_200']) &
                (dataframe['close'] < dataframe['ema_20'] * 0.968) &
                (dataframe['close'] < dataframe['bb20_2_low'] * 0.982) &
                (dataframe['mfi'] < 50.0) &
                (dataframe['cti'] < -0.85) &
                (dataframe['r_14'] < -94.0) &
                (dataframe['rsi_14_1h'] > 20.0) &
                (dataframe['rsi_14_1h'] < 88.0) &
                (dataframe['crsi_1h'] > 21.0)    
            )
            
        is_nfi_10 = (
                is_additional_check &        
                (dataframe['ema_50_1h'] > dataframe['ema_100_1h']) &
                (dataframe['close'] < dataframe['sma_30'] * 0.94) &
                (dataframe['close'] < dataframe['bb20_2_low'] * 0.984) &
                (dataframe['r_14'] < -88.0) &
                (dataframe['cti_1h'] > -0.5) &
                (dataframe['cti_1h'] < 0.94)
            )
            
        is_nfi_26 = (
                is_additional_check &        
                (dataframe['close'] < (dataframe['zema_61'] * 0.9405)) &
                (dataframe['cti'] < -0.72) &
                (dataframe['cci'] < -166.0) &
                (dataframe['r_14'] < -98.0) &
                (dataframe['cti_1h'] < 0.95) &
                (dataframe['volume'] < (dataframe['volume_mean_4'] * 2.0))
            )
            
        is_nfix_53 = (
                is_additional_check &        
                (dataframe['ema_200_1h'] > dataframe['ema_200_1h'].shift(12)) &
                (dataframe['ema_200_1h'].shift(12) > dataframe['ema_200_1h'].shift(24)) &
                (dataframe['ema_200_1h'].shift(24) > dataframe['ema_200_1h'].shift(36)) &
                (dataframe['ema_26_15m'] > dataframe['ema_12_15m']) &
                ((dataframe['ema_26_15m'] - dataframe['ema_12_15m']) > (dataframe['open_15m'] * 0.02)) &
                ((dataframe['ema_26_15m'].shift(3) - dataframe['ema_12_15m'].shift(3)) > (dataframe['open_15m'] / 100)) &
                (dataframe['close_15m'] < (dataframe['bb20_2_low_15m'] * 0.99)) &
                (dataframe['r_14'] < -90.0) &
                (dataframe['cti_1h'] > -0.7)
                    
            )
        is_nfix_52 = (
                is_additional_check &        
                (dataframe['ema_26_15m'] > dataframe['ema_12_15m']) &
                ((dataframe['ema_26_15m'] - dataframe['ema_12_15m']) > (dataframe['open_15m'] * 0.032)) &
                ((dataframe['ema_26_15m'].shift(3) - dataframe['ema_12_15m'].shift(3)) > (dataframe['open_15m'] / 100)) &
                (dataframe['close_15m'] < (dataframe['bb20_2_low_15m'] * 0.998)) &
                (dataframe['crsi_1h'] > 10.0)
            )
                
        is_nfix_51 = (
                is_additional_check &        
                (dataframe['close_15m'] < (dataframe['ema_16_15m'] * 0.944)) &
                (dataframe['ewo_15m'] < -1.0) &
                (dataframe['rsi_14_15m'] > 28.0) &
                (dataframe['cti_15m'] < -0.84) &
                (dataframe['r_14_15m'] < -94.0) &
                (dataframe['rsi_14'] > 30.0) &
                (dataframe['crsi_1h'] > 1.0)
            )
                
        is_nfix_48 = (
                is_additional_check &        
                (dataframe['close_15m'].shift(3) < (dataframe['sma_15_15m'].shift(3) * 0.95)) &
                (dataframe['close_15m'] > (dataframe['open_15m'].shift(3))) &
                (dataframe['ewo_15m'] > 2.8) &
                (dataframe['cti_15m'] < -0.75) &
                (dataframe['r_14_15m'].shift(3) < -94.0) &
                (dataframe['cti'] < -0.5) &
                (dataframe['cti_1h'] < 0.1) &
                (dataframe['crsi_1h'] > 18.0)
            )
                
        is_nfix_47 = (
                is_additional_check &        
                (dataframe['rsi_14_15m'] < dataframe['rsi_14_15m'].shift(3)) &
                (dataframe['ema_20_1h'] > dataframe['ema_25_1h']) &
                (dataframe['close_15m'] < (dataframe['sma_15_15m'] * 0.95)) &
                (
                    ((dataframe['open_15m'] < dataframe['ema_20_1h']) & (dataframe['low_15m'] < dataframe['ema_20_1h'])) |
                    ((dataframe['open_15m'] > dataframe['ema_20_1h']) & (dataframe['low_15m'] > dataframe['ema_20_1h']))
                ) &
                (dataframe['cti_15m'] < -0.9) &
                (dataframe['r_14_15m'] < -90.0) &
                (dataframe['r_14'] < -97.0) &
                (dataframe['cti_1h'] < 0.1) &
                (dataframe['crsi_1h'] > 8.0)
            )
                
        is_nfix_41 = (
                is_additional_check &        
                (dataframe['ema_12_15m'] > dataframe['ema_200_1h']) &
                (dataframe['ema_26_15m'] > dataframe['ema_12_15m']) &
                ((dataframe['ema_26_15m'] - dataframe['ema_12_15m']) > (dataframe['open_15m'] * 0.03)) &
                ((dataframe['ema_26_15m'].shift(3) - dataframe['ema_12_15m'].shift(3)) > (dataframe['open_15m'] / 100)) &
                (dataframe['close_15m'] < (dataframe['bb20_2_low_15m'] * 0.99))
            )

        is_nfix_38 = (
                is_additional_check &        
                (dataframe['ema_200'] > (dataframe['ema_200'].shift(12) * 1.01)) &
                (dataframe['ema_26'] > dataframe['ema_12']) &
                ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * 0.02)) &
                ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open'] / 100)) &
                (dataframe['mfi'] < 34.5) &
                (dataframe['r_64'] < -65.0) &
                (dataframe['r_96'] < -50.0) &
                (dataframe['r_480_1h'] < -1.0)
            )
                
        is_nfix_36 = (
                is_additional_check &        
                (dataframe['ema_200'] > (dataframe['ema_200'].shift(36) * 1.035)) &
                (dataframe['close'] < dataframe['ema_20'] * 0.956) &
                (dataframe['rsi_14'] < 34.0) &
                (dataframe['r_64'] < -80.0) &
                (dataframe['cti'] < -0.5) &
                (dataframe['r_480_1h'] < -30.0)
            )
                
        is_nfix_204 = (
                is_additional_check &        
                (dataframe['pm'] > dataframe['pmax_thresh']) &
                (dataframe['close'] < dataframe['sma_75'] * 0.98) &
                (dataframe['EWO'] < -4.4) &
                (dataframe['cti'] < -0.95) &
                (dataframe['r_14'] < -97.0) &
                (dataframe['crsi_1h'] > 0.5)
            )
                
        is_nfix_203 = (
                is_additional_check &        
                (dataframe['ema_50_1h'] > dataframe['ema_100_1h']) &
                (dataframe['close'] < dataframe['sma_30'] * 0.99) &
                (dataframe['cti'] < -0.92) &
                (dataframe['EWO'] < -6.0) &
                (dataframe['cti_1h'] < -0.88) &
                (dataframe['crsi_1h'] > 10.0)
            )
                
        is_nfix_202 = (
                is_additional_check &        
                (dataframe['close'] > (dataframe['ema_200_1h'] * 0.84)) &
                (dataframe['ema_26'] > dataframe['ema_12']) &
                ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * 0.02)) &
                ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open'] / 100)) &
                (dataframe['close'] < (dataframe['bb20_2_low'] * 0.999)) &
                (dataframe['cti'] < -0.5) &
                (dataframe['rsi_14'] > 25.0) &
                (dataframe['mfi'] > 18.0) &
                (dataframe['r_14'] < -94.0) &
                (dataframe['r_14'].shift(1) < -94.0) &
                (dataframe['crsi_1h'] > 12.0) &
                (dataframe['volume'] < (dataframe['volume_mean_4'] * 1.6))
            )
                
        is_nfix_201 = (
                is_additional_check &        
                (dataframe['rsi_20'] < dataframe['rsi_20'].shift()) &
                (dataframe['rsi_4'] < 30.0) &
                (dataframe['ema_20_1h'] > dataframe['ema_26_1h']) &
                (dataframe['close'] < dataframe['sma_15'] * 0.953) &
                (dataframe['cti'] < -0.78) &
                (dataframe['cci'] < -200.0)
            )
                
        is_nfix_34 = (
                is_additional_check &        
                (dataframe['close'] < dataframe['ema_50']) &
                (dataframe['close'] < (dataframe['bb20_2_low'] * 0.972)) &
                (dataframe['cti'] < -0.8) &
                (dataframe['rsi_14'] < 18.0)
            )
                
        is_nfix_28 = (
                is_additional_check &        
                (dataframe['close'] < dataframe['sma_75'] * 0.96) &
                (dataframe['EWO'] < -8.0) &
                (dataframe['cti'] < -0.9) &
                (dataframe['r_14'] < -97.0) &
                (dataframe['crsi_1h'] > 14.0)
            )
                
        is_nfix_27 = (
                is_additional_check &        
                (dataframe['close'] < dataframe['sma_75'] * 0.934) &
                (dataframe['EWO'] > 6.4) &
                (dataframe['rsi_14'] < 32.0) &
                (dataframe['cti'] < -0.8) &
                (dataframe['r_14'] < -96.0)
            )
                
        is_nfix_19 = (
                is_additional_check &        
                (dataframe['ema_200_1h'] > dataframe['ema_200_1h'].shift(12)) &
                (dataframe['ema_200_1h'].shift(12) > dataframe['ema_200_1h'].shift(24)) &
                (dataframe['bb40_2_low'].shift().gt(0)) &
                (dataframe['bb40_2_delta'].gt(dataframe['close'] * 0.045)) &
                (dataframe['closedelta'].gt(dataframe['close'] * 0.02)) &
                (dataframe['tail'].lt(dataframe['bb40_2_delta'] * 0.28)) &
                (dataframe['close'].lt(dataframe['bb40_2_low'].shift())) &
                (dataframe['close'].le(dataframe['close'].shift())) &
                (dataframe['cti'] < -0.9) &
                (dataframe['cti_1h'] > -0.75) &
                (dataframe['cti_1h'] < 0.25) 
            )
                
        is_nfix_11 = (
                is_additional_check &        
                (dataframe['ema_26'] > dataframe['ema_12']) &
                ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * 0.027)) &
                ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open'] / 100)) &
                (dataframe['close'] < dataframe['ema_20'] * 0.932) &
                (dataframe['rsi_14'] < 25.0)
            )
            
        is_nfix_9 = (
                is_additional_check &        
                (dataframe['ema_50_1h'] > dataframe['ema_100_1h']) &
                (dataframe['close'] < dataframe['sma_30'] * 0.99) &
                (dataframe['cti'] < -0.92) &
                (dataframe['EWO'] < -5.0) &
                (dataframe['cti_1h'] < -0.88) &
                (dataframe['crsi_1h'] > 20.0)
            )
            
        ## Additional Check
        is_BB_checked = is_dip & is_break

        ## Condition Append
        conditions.append(is_BB_checked)                                           # ~2.32 / 91.1% / 46.27%      D
        dataframe.loc[is_BB_checked, 'buy_tag'] += 'bb '

        conditions.append(is_local_uptrend)                                        # ~3.28 / 92.4% / 69.72%
        dataframe.loc[is_local_uptrend, 'buy_tag'] += 'local_uptrend '

        conditions.append(is_local_dip)                                            # ~0.76 / 91.1% / 15.54%
        dataframe.loc[is_local_dip, 'buy_tag'] += 'local_dip '

        conditions.append(is_ewo)                                                  # ~0.92 / 92.0% / 43.74%      D
        dataframe.loc[is_ewo, 'buy_tag'] += 'ewo '

        conditions.append(is_ewo_2)                                                 # ~2.86 / 91.5% / 33.31%     D
        dataframe.loc[is_ewo_2, 'buy_tag'] += 'ewo2 '

        conditions.append(is_r_deadfish)                                           # ~0.99 / 86.9% / 21.93%      D
        dataframe.loc[is_r_deadfish, 'buy_tag'] += 'r_deadfish '

        conditions.append(is_clucHA)                                               # ~7.2 / 92.5% / 97.98%       D
        dataframe.loc[is_clucHA, 'buy_tag'] += 'clucHA '

        conditions.append(is_cofi)                                                 # ~0.4 / 94.4% / 9.59%        D
        dataframe.loc[is_cofi, 'buy_tag'] += 'cofi '

        conditions.append(is_gumbo)                                                # ~2.63 / 90.6% / 41.49%      D
        dataframe.loc[is_gumbo, 'buy_tag'] += 'gumbo '

        conditions.append(is_sqzmom)                                               # ~3.14 / 92.4% / 64.14%      D
        dataframe.loc[is_sqzmom, 'buy_tag'] += 'sqzmom '

        conditions.append(is_nfi_13)                                               # ~0.4 / 100%                 D
        dataframe.loc[is_nfi_13, 'buy_tag'] += 'nfi_13 '

        conditions.append(is_nfi_32)                                               # ~0.78 / 92.0 % / 37.41%     D
        dataframe.loc[is_nfi_32, 'buy_tag'] += 'nfi_32 '

        conditions.append(is_nfi_33)                                               # ~0.11 / 100%                D
        dataframe.loc[is_nfi_33, 'buy_tag'] += 'nfi_33 '

        conditions.append(is_nfi_38)                                               # ~1.13 / 88.5% / 31.34%      D
        dataframe.loc[is_nfi_38, 'buy_tag'] += 'nfi_38 '

        conditions.append(is_nfix_5)                                               # ~0.25 / 97.7% / 6.53%       D
        dataframe.loc[is_nfix_5, 'buy_tag'] += 'nfix_5 '

        conditions.append(is_nfix_39)                                              # ~5.33 / 91.8% / 58.57%      D
        dataframe.loc[is_nfix_39, 'buy_tag'] += 'nfix_39 '

        conditions.append(is_nfix_49)                                              # ~0.33 / 100% / 0%           D
        dataframe.loc[is_nfix_49, 'buy_tag'] += 'nfix_49 '

        conditions.append(is_nfi7_33)                                              # ~0.71 / 91.3% / 28.94%      D
        dataframe.loc[is_nfi7_33, 'buy_tag'] += 'nfi7_33 '

        conditions.append(is_nfi7_37)                                              # ~0.46 / 92.6% / 17.05%      D
        dataframe.loc[is_nfi7_37, 'buy_tag'] += 'nfi7_37 '
        
        conditions.append(is_nfi_ctt35)                                           # ~2.32 / 91.1% / 46.27%      D
        dataframe.loc[is_nfi_ctt35, 'buy_tag'] += 'nfi_ctt35 '

        conditions.append(is_nfi_ctt25)                                        # ~3.28 / 92.4% / 69.72%
        dataframe.loc[is_nfi_ctt25, 'buy_tag'] += 'nfi_ctt25 '

        conditions.append(is_nfi_ctt15)                                            # ~0.76 / 91.1% / 15.54%
        dataframe.loc[is_nfi_ctt15, 'buy_tag'] += 'nfi_ctt15 '

        conditions.append(is_nfi_10)                                                  # ~0.92 / 92.0% / 43.74%      D
        dataframe.loc[is_nfi_10, 'buy_tag'] += 'nfi_10 '

        conditions.append(is_nfix_53)                                               # ~7.2 / 92.5% / 97.98%       D
        dataframe.loc[is_nfix_53, 'buy_tag'] += 'nfix_53 '

        conditions.append(is_nfix_52)                                                 # ~0.4 / 94.4% / 9.59%        D
        dataframe.loc[is_nfix_52, 'buy_tag'] += 'nfix_52 '

        conditions.append(is_nfix_51)                                                # ~2.63 / 90.6% / 41.49%      D
        dataframe.loc[is_nfix_51, 'buy_tag'] += 'nfix_51 '

        conditions.append(is_nfix_48)                                               # ~3.14 / 92.4% / 64.14%      D
        dataframe.loc[is_nfix_48, 'buy_tag'] += 'nfix_48 '

        conditions.append(is_nfix_47)                                               # ~0.4 / 100%                 D
        dataframe.loc[is_nfix_47, 'buy_tag'] += 'nfix_47 '

        conditions.append(is_nfix_41)                                               # ~0.78 / 92.0 % / 37.41%     D
        dataframe.loc[is_nfix_41, 'buy_tag'] += 'nfix_41 '

        conditions.append(is_nfix_38)                                               # ~0.11 / 100%                D
        dataframe.loc[is_nfix_38, 'buy_tag'] += 'nfix_38 '

        conditions.append(is_nfix_36)                                               # ~1.13 / 88.5% / 31.34%      D
        dataframe.loc[is_nfix_36, 'buy_tag'] += 'nfix_36 '

        conditions.append(is_nfix_204)                                               # ~0.25 / 97.7% / 6.53%       D
        dataframe.loc[is_nfix_204, 'buy_tag'] += 'nfix_204 '

        conditions.append(is_nfix_203)                                              # ~5.33 / 91.8% / 58.57%      D
        dataframe.loc[is_nfix_203, 'buy_tag'] += 'nfix_203 '

        conditions.append(is_nfix_202)                                              # ~0.33 / 100% / 0%           D
        dataframe.loc[is_nfix_202, 'buy_tag'] += 'nfix_202 '

        conditions.append(is_nfix_201)                                              # ~0.71 / 91.3% / 28.94%      D
        dataframe.loc[is_nfix_201, 'buy_tag'] += 'nfix_201 '

        conditions.append(is_nfix_34)                                              # ~0.46 / 92.6% / 17.05%      D
        dataframe.loc[is_nfix_34, 'buy_tag'] += 'nfix_34 '
        
        conditions.append(is_nfix_28)                                               # ~0.25 / 97.7% / 6.53%       D
        dataframe.loc[is_nfix_28, 'buy_tag'] += 'nfix_28 '

        conditions.append(is_nfix_27)                                              # ~5.33 / 91.8% / 58.57%      D
        dataframe.loc[is_nfix_27, 'buy_tag'] += 'nfix_27 '

        conditions.append(is_nfix_19)                                              # ~0.33 / 100% / 0%           D
        dataframe.loc[is_nfix_19, 'buy_tag'] += 'nfix_19 '

        conditions.append(is_nfix_11)                                              # ~0.71 / 91.3% / 28.94%      D
        dataframe.loc[is_nfix_11, 'buy_tag'] += 'nfix_11 '

        conditions.append(is_nfix_9)                                              # ~0.46 / 92.6% / 17.05%      D
        dataframe.loc[is_nfix_9, 'buy_tag'] += 'nfix_9 '

        if conditions:
            dataframe.loc[
                            is_additional_check
                            &
                            reduce(lambda x, y: x | y, conditions)

                        , 'buy' ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[ (dataframe['volume'] > 0), 'sell' ] = 0

        return dataframe
        
class BB_RPB_TSL_Tranz_TrailingBuy(BB_RPB_TSL_Tranz):
    # Original idea by @MukavaValkku, code by @tirail and @stash86
    #
    # This class is designed to inherit from yours and starts trailing buy with your buy signals
    # Trailing buy starts at any buy signal and will move to next candles if the trailing still active
    # Trailing buy stops  with BUY if : price decreases and rises again more than trailing_buy_offset
    # Trailing buy stops with NO BUY : current price is > initial price * (1 +  trailing_buy_max) OR custom_sell tag
    # IT IS NOT COMPATIBLE WITH BACKTEST/HYPEROPT
    #

    process_only_new_candles = True

    custom_info_trail_buy = dict()

    # Trailing buy parameters
    trailing_buy_order_enabled = True
    trailing_expire_seconds = 1800

    # If the current candle goes above min_uptrend_trailing_profit % before trailing_expire_seconds_uptrend seconds, buy the coin
    trailing_buy_uptrend_enabled = False
    trailing_expire_seconds_uptrend = 90
    min_uptrend_trailing_profit = 0.02

    debug_mode = True
    trailing_buy_max_stop = 0.02  # stop trailing buy if current_price > starting_price * (1+trailing_buy_max_stop)
    trailing_buy_max_buy = 0.000  # buy if price between uplimit (=min of serie (current_price * (1 + trailing_buy_offset())) and (start_price * 1+trailing_buy_max_buy))

    init_trailing_dict = {
        'trailing_buy_order_started': False,
        'trailing_buy_order_uplimit': 0,
        'start_trailing_price': 0,
        'buy_tag': None,
        'start_trailing_time': None,
        'offset': 0,
        'allow_trailing': False,
    }

    def trailing_buy(self, pair, reinit=False):
        # returns trailing buy info for pair (init if necessary)
        if not pair in self.custom_info_trail_buy:
            self.custom_info_trail_buy[pair] = dict()
        if (reinit or not 'trailing_buy' in self.custom_info_trail_buy[pair]):
            self.custom_info_trail_buy[pair]['trailing_buy'] = self.init_trailing_dict.copy()
        return self.custom_info_trail_buy[pair]['trailing_buy']

    def trailing_buy_info(self, pair: str, current_price: float):
        # current_time live, dry run
        current_time = datetime.now(timezone.utc)
        if not self.debug_mode:
            return
        trailing_buy = self.trailing_buy(pair)

        duration = 0
        try:
            duration = (current_time - trailing_buy['start_trailing_time'])
        except TypeError:
            duration = 0
        finally:
            logger.info(
                f"pair: {pair} : "
                f"start: {trailing_buy['start_trailing_price']:.4f}, "
                f"duration: {duration}, "
                f"current: {current_price:.4f}, "
                f"uplimit: {trailing_buy['trailing_buy_order_uplimit']:.4f}, "
                f"profit: {self.current_trailing_profit_ratio(pair, current_price)*100:.2f}%, "
                f"offset: {trailing_buy['offset']}")

    def current_trailing_profit_ratio(self, pair: str, current_price: float) -> float:
        trailing_buy = self.trailing_buy(pair)
        if trailing_buy['trailing_buy_order_started']:
            return (trailing_buy['start_trailing_price'] - current_price) / trailing_buy['start_trailing_price']
        else:
            return 0

    def trailing_buy_offset(self, dataframe, pair: str, current_price: float):
        # return rebound limit before a buy in % of initial price, function of current price
        # return None to stop trailing buy (will start again at next buy signal)
        # return 'forcebuy' to force immediate buy
        # (example with 0.5%. initial price : 100 (uplimit is 100.5), 2nd price : 99 (no buy, uplimit updated to 99.5), 3price 98 (no buy uplimit updated to 98.5), 4th price 99 -> BUY
        current_trailing_profit_ratio = self.current_trailing_profit_ratio(pair, current_price)
        default_offset = 0.005

        trailing_buy = self.trailing_buy(pair)
        if not trailing_buy['trailing_buy_order_started']:
            return default_offset

        # example with duration and indicators
        # dry run, live only
        last_candle = dataframe.iloc[-1]
        current_time = datetime.now(timezone.utc)
        trailing_duration = current_time - trailing_buy['start_trailing_time']
        if trailing_duration.total_seconds() > self.trailing_expire_seconds:
            if ((current_trailing_profit_ratio > 0) and (last_candle['buy'] == 1)):
                # more than 1h, price under first signal, buy signal still active -> buy
                return 'forcebuy'
            else:
                # wait for next signal
                return None
        elif (self.trailing_buy_uptrend_enabled and (trailing_duration.total_seconds() < self.trailing_expire_seconds_uptrend) and (current_trailing_profit_ratio < (-1 * self.min_uptrend_trailing_profit))):
            # less than 90s and price is rising, buy
            return 'forcebuy'

        if current_trailing_profit_ratio < 0:
            # current price is higher than initial price
            return default_offset

        trailing_buy_offset = {
            0.06: 0.02,
            0.03: 0.01,
            0: default_offset,
        }

        for key in trailing_buy_offset:
            if current_trailing_profit_ratio > key:
                return trailing_buy_offset[key]

        return default_offset

    # end of trailing buy parameters
    # -----------------------------------------------------

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)
        self.trailing_buy(metadata['pair'])
        return dataframe

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, **kwargs) -> bool:
        val = super().confirm_trade_entry(pair, order_type, amount, rate, time_in_force, **kwargs)
        
        if val:
            if self.trailing_buy_order_enabled and self.config['runmode'].value in ('live', 'dry_run'):
                val = False
                dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                if(len(dataframe) >= 1):
                    last_candle = dataframe.iloc[-1].squeeze()
                    current_price = rate
                    trailing_buy = self.trailing_buy(pair)
                    trailing_buy_offset = self.trailing_buy_offset(dataframe, pair, current_price)

                    if trailing_buy['allow_trailing']:
                        if (not trailing_buy['trailing_buy_order_started'] and (last_candle['buy'] == 1)):
                            # start trailing buy
                            
                            # self.custom_info_trail_buy[pair]['trailing_buy']['trailing_buy_order_started'] = True
                            # self.custom_info_trail_buy[pair]['trailing_buy']['trailing_buy_order_uplimit'] = last_candle['close']
                            # self.custom_info_trail_buy[pair]['trailing_buy']['start_trailing_price'] = last_candle['close']
                            # self.custom_info_trail_buy[pair]['trailing_buy']['buy_tag'] = f"initial_buy_tag (strat trail price {last_candle['close']})"
                            # self.custom_info_trail_buy[pair]['trailing_buy']['start_trailing_time'] = datetime.now(timezone.utc)
                            # self.custom_info_trail_buy[pair]['trailing_buy']['offset'] = 0

                            trailing_buy['trailing_buy_order_started'] = True
                            trailing_buy['trailing_buy_order_uplimit'] = last_candle['close']
                            trailing_buy['start_trailing_price'] = last_candle['close']
                            trailing_buy['buy_tag'] = last_candle['buy_tag']
                            trailing_buy['start_trailing_time'] = datetime.now(timezone.utc)
                            trailing_buy['offset'] = 0
                            
                            self.trailing_buy_info(pair, current_price)
                            logger.info(f'start trailing buy for {pair} at {last_candle["close"]}')

                        elif trailing_buy['trailing_buy_order_started']:
                            if trailing_buy_offset == 'forcebuy':
                                # buy in custom conditions
                                val = True
                                ratio = "%.2f" % ((self.current_trailing_profit_ratio(pair, current_price)) * 100)
                                self.trailing_buy_info(pair, current_price)
                                logger.info(f"price OK for {pair} ({ratio} %, {current_price}), order may not be triggered if all slots are full")

                            elif trailing_buy_offset is None:
                                # stop trailing buy custom conditions
                                self.trailing_buy(pair, reinit=True)
                                logger.info(f'STOP trailing buy for {pair} because "trailing buy offset" returned None')

                            elif current_price < trailing_buy['trailing_buy_order_uplimit']:
                                # update uplimit
                                old_uplimit = trailing_buy["trailing_buy_order_uplimit"]
                                self.custom_info_trail_buy[pair]['trailing_buy']['trailing_buy_order_uplimit'] = min(current_price * (1 + trailing_buy_offset), self.custom_info_trail_buy[pair]['trailing_buy']['trailing_buy_order_uplimit'])
                                self.custom_info_trail_buy[pair]['trailing_buy']['offset'] = trailing_buy_offset
                                self.trailing_buy_info(pair, current_price)
                                logger.info(f'update trailing buy for {pair} at {old_uplimit} -> {self.custom_info_trail_buy[pair]["trailing_buy"]["trailing_buy_order_uplimit"]}')
                            elif current_price < (trailing_buy['start_trailing_price'] * (1 + self.trailing_buy_max_buy)):
                                # buy ! current price > uplimit && lower thant starting price
                                val = True
                                ratio = "%.2f" % ((self.current_trailing_profit_ratio(pair, current_price)) * 100)
                                self.trailing_buy_info(pair, current_price)
                                logger.info(f"current price ({current_price}) > uplimit ({trailing_buy['trailing_buy_order_uplimit']}) and lower than starting price price ({(trailing_buy['start_trailing_price'] * (1 + self.trailing_buy_max_buy))}). OK for {pair} ({ratio} %), order may not be triggered if all slots are full")

                            elif current_price > (trailing_buy['start_trailing_price'] * (1 + self.trailing_buy_max_stop)):
                                # stop trailing buy because price is too high
                                self.trailing_buy(pair, reinit=True)
                                self.trailing_buy_info(pair, current_price)
                                logger.info(f'STOP trailing buy for {pair} because of the price is higher than starting price * {1 + self.trailing_buy_max_stop}')
                            else:
                                # uplimit > current_price > max_price, continue trailing and wait for the price to go down
                                self.trailing_buy_info(pair, current_price)
                                logger.info(f'price too high for {pair} !')

                    else:
                        logger.info(f"Wait for next buy signal for {pair}")

                if (val == True):
                    self.trailing_buy_info(pair, rate)
                    self.trailing_buy(pair, reinit=True)
                    logger.info(f'STOP trailing buy for {pair} because I buy it')
        
        return val

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)

        if self.trailing_buy_order_enabled and self.config['runmode'].value in ('live', 'dry_run'): 
            last_candle = dataframe.iloc[-1].squeeze()
            trailing_buy = self.trailing_buy(metadata['pair'])
            if (last_candle['buy'] == 1):
                if not trailing_buy['trailing_buy_order_started']:
                    open_trades = Trade.get_trades([Trade.pair == metadata['pair'], Trade.is_open.is_(True), ]).all()
                    if not open_trades:
                        logger.info(f"Set 'allow_trailing' to True for {metadata['pair']} to start trailing!!!")
                        # self.custom_info_trail_buy[metadata['pair']]['trailing_buy']['allow_trailing'] = True
                        trailing_buy['allow_trailing'] = True
                        initial_buy_tag = last_candle['buy_tag'] if 'buy_tag' in last_candle else 'buy signal'
                        dataframe.loc[:, 'buy_tag'] = f"{initial_buy_tag} (start trail price {last_candle['close']})"
            else:
                if (trailing_buy['trailing_buy_order_started'] == True):
                    logger.info(f"Continue trailing for {metadata['pair']}. Manually trigger buy signal!!")
                    dataframe.loc[:,'buy'] = 1
                    dataframe.loc[:, 'buy_tag'] = trailing_buy['buy_tag']
                    # dataframe['buy'] = 1

        return dataframe

# Elliot Wave Oscillator
def ewo(dataframe, sma1_length=5, sma2_length=35):
    sma1 = ta.EMA(dataframe, timeperiod=sma1_length)
    sma2 = ta.EMA(dataframe, timeperiod=sma2_length)
    smadif = (sma1 - sma2) / dataframe['close'] * 100
    return smadif

# Exponential moving average of a volume weighted simple moving average
def ema_vwma_osc(dataframe, len_slow_ma):
    slow_ema = Series(ta.EMA(vwma(dataframe, len_slow_ma), len_slow_ma))
    return ((slow_ema - slow_ema.shift(1)) / slow_ema.shift(1)) * 100

def pivot_points(dataframe: DataFrame, mode = 'fibonacci') -> Series:
    hlc3_pivot = (dataframe['high'] + dataframe['low'] + dataframe['close']).shift(1) / 3
    hl_range = (dataframe['high'] - dataframe['low']).shift(1)
    if mode == 'simple':
        res1 = hlc3_pivot * 2 - dataframe['low'].shift(1)
        sup1 = hlc3_pivot * 2 - dataframe['high'].shift(1)
        res2 = hlc3_pivot + (dataframe['high'] - dataframe['low']).shift()
        sup2 = hlc3_pivot - (dataframe['high'] - dataframe['low']).shift()
        res3 = hlc3_pivot * 2 + (dataframe['high'] - 2 * dataframe['low']).shift()
        sup3 = hlc3_pivot * 2 - (2 * dataframe['high'] - dataframe['low']).shift()
    elif mode == 'fibonacci':
        res1 = hlc3_pivot + 0.382 * hl_range
        sup1 = hlc3_pivot - 0.382 * hl_range
        res2 = hlc3_pivot + 0.618 * hl_range
        sup2 = hlc3_pivot - 0.618 * hl_range
        res3 = hlc3_pivot + 1 * hl_range
        sup3 = hlc3_pivot - 1 * hl_range

    return hlc3_pivot, res1, res2, res3, sup1, sup2, sup3

def heikin_ashi(dataframe, smooth_inputs = False, smooth_outputs = False, length = 10):
    df = dataframe[['open','close','high','low']].copy().fillna(0)
    if smooth_inputs:
        df['open_s']  = ta.EMA(df['open'], timeframe = length)
        df['high_s']  = ta.EMA(df['high'], timeframe = length)
        df['low_s']   = ta.EMA(df['low'],  timeframe = length)
        df['close_s'] = ta.EMA(df['close'],timeframe = length)

        open_ha  = (df['open_s'].shift(1) + df['close_s'].shift(1)) / 2
        high_ha  = df.loc[:, ['high_s', 'open_s', 'close_s']].max(axis=1)
        low_ha   = df.loc[:, ['low_s', 'open_s', 'close_s']].min(axis=1)
        close_ha = (df['open_s'] + df['high_s'] + df['low_s'] + df['close_s'])/4
    else:
        open_ha  = (df['open'].shift(1) + df['close'].shift(1)) / 2
        high_ha  = df.loc[:, ['high', 'open', 'close']].max(axis=1)
        low_ha   = df.loc[:, ['low', 'open', 'close']].min(axis=1)
        close_ha = (df['open'] + df['high'] + df['low'] + df['close'])/4

    open_ha = open_ha.fillna(0)
    high_ha = high_ha.fillna(0)
    low_ha  = low_ha.fillna(0)
    close_ha = close_ha.fillna(0)

    if smooth_outputs:
        open_sha  = ta.EMA(open_ha, timeframe = length)
        high_sha  = ta.EMA(high_ha, timeframe = length)
        low_sha   = ta.EMA(low_ha, timeframe = length)
        close_sha = ta.EMA(close_ha, timeframe = length)

        return open_sha, close_sha, low_sha
    else:
        return open_ha, close_ha, low_ha

# PMAX
def pmax(df, period, multiplier, length, MAtype, src):

    period = int(period)
    multiplier = int(multiplier)
    length = int(length)
    MAtype = int(MAtype)
    src = int(src)

    mavalue = f'MA_{MAtype}_{length}'
    atr = f'ATR_{period}'
    pm = f'pm_{period}_{multiplier}_{length}_{MAtype}'
    pmx = f'pmX_{period}_{multiplier}_{length}_{MAtype}'

    # MAtype==1 --> EMA
    # MAtype==2 --> DEMA
    # MAtype==3 --> T3
    # MAtype==4 --> SMA
    # MAtype==5 --> VIDYA
    # MAtype==6 --> TEMA
    # MAtype==7 --> WMA
    # MAtype==8 --> VWMA
    # MAtype==9 --> zema
    if src == 1:
        masrc = df["close"]
    elif src == 2:
        masrc = (df["high"] + df["low"]) / 2
    elif src == 3:
        masrc = (df["high"] + df["low"] + df["close"] + df["open"]) / 4

    if MAtype == 1:
        mavalue = ta.EMA(masrc, timeperiod=length)
    elif MAtype == 2:
        mavalue = ta.DEMA(masrc, timeperiod=length)
    elif MAtype == 3:
        mavalue = ta.T3(masrc, timeperiod=length)
    elif MAtype == 4:
        mavalue = ta.SMA(masrc, timeperiod=length)
    elif MAtype == 5:
        mavalue = VIDYA(df, length=length)
    elif MAtype == 6:
        mavalue = ta.TEMA(masrc, timeperiod=length)
    elif MAtype == 7:
        mavalue = ta.WMA(df, timeperiod=length)
    elif MAtype == 8:
        mavalue = vwma(df, length)
    elif MAtype == 9:
        mavalue = zema(df, period=length)

    df[atr] = ta.ATR(df, timeperiod=period)
    df['basic_ub'] = mavalue + ((multiplier/10) * df[atr])
    df['basic_lb'] = mavalue - ((multiplier/10) * df[atr])


    basic_ub = df['basic_ub'].values
    final_ub = np.full(len(df), 0.00)
    basic_lb = df['basic_lb'].values
    final_lb = np.full(len(df), 0.00)

    for i in range(period, len(df)):
        final_ub[i] = basic_ub[i] if (
            basic_ub[i] < final_ub[i - 1]
            or mavalue[i - 1] > final_ub[i - 1]) else final_ub[i - 1]
        final_lb[i] = basic_lb[i] if (
            basic_lb[i] > final_lb[i - 1]
            or mavalue[i - 1] < final_lb[i - 1]) else final_lb[i - 1]

    df['final_ub'] = final_ub
    df['final_lb'] = final_lb

    pm_arr = np.full(len(df), 0.00)
    for i in range(period, len(df)):
        pm_arr[i] = (
            final_ub[i] if (pm_arr[i - 1] == final_ub[i - 1]
                                    and mavalue[i] <= final_ub[i])
        else final_lb[i] if (
            pm_arr[i - 1] == final_ub[i - 1]
            and mavalue[i] > final_ub[i]) else final_lb[i]
        if (pm_arr[i - 1] == final_lb[i - 1]
            and mavalue[i] >= final_lb[i]) else final_ub[i]
        if (pm_arr[i - 1] == final_lb[i - 1]
            and mavalue[i] < final_lb[i]) else 0.00)

    pm = Series(pm_arr)

    # Mark the trend direction up/down
    pmx = np.where((pm_arr > 0.00), np.where((mavalue < pm_arr), 'down',  'up'), np.NaN)

    return pm, pmx

# Mom DIV
def momdiv(dataframe: DataFrame, mom_length: int = 10, bb_length: int = 20, bb_dev: float = 2.0, lookback: int = 30) -> DataFrame:
    mom: Series = ta.MOM(dataframe, timeperiod=mom_length)
    upperband, middleband, lowerband = ta.BBANDS(mom, timeperiod=bb_length, nbdevup=bb_dev, nbdevdn=bb_dev, matype=0)
    buy = qtpylib.crossed_below(mom, lowerband)
    sell = qtpylib.crossed_above(mom, upperband)
    hh = dataframe['high'].rolling(lookback).max()
    ll = dataframe['low'].rolling(lookback).min()
    coh = dataframe['high'] >= hh
    col = dataframe['low'] <= ll
    df = DataFrame({
            "momdiv_mom": mom,
            "momdiv_upperb": upperband,
            "momdiv_lowerb": lowerband,
            "momdiv_buy": buy,
            "momdiv_sell": sell,
            "momdiv_coh": coh,
            "momdiv_col": col,
        }, index=dataframe['close'].index)
    return df

def T3(dataframe, length=5):
    """
    T3 Average by HPotter on Tradingview
    https://www.tradingview.com/script/qzoC9H1I-T3-Average/
    """
    df = dataframe.copy()

    df['xe1'] = ta.EMA(df['close'], timeperiod=length)
    df['xe2'] = ta.EMA(df['xe1'], timeperiod=length)
    df['xe3'] = ta.EMA(df['xe2'], timeperiod=length)
    df['xe4'] = ta.EMA(df['xe3'], timeperiod=length)
    df['xe5'] = ta.EMA(df['xe4'], timeperiod=length)
    df['xe6'] = ta.EMA(df['xe5'], timeperiod=length)
    b = 0.7
    c1 = -b * b * b
    c2 = 3 * b * b + 3 * b * b * b
    c3 = -6 * b * b - 3 * b - 3 * b * b * b
    c4 = 1 + 3 * b + b * b * b + 3 * b * b
    df['T3Average'] = c1 * df['xe6'] + c2 * df['xe5'] + c3 * df['xe4'] + c4 * df['xe3']

    return df['T3Average']