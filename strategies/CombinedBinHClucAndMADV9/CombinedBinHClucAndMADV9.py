"""
CombinedBinHClucAndMADV9 strategy

Updated for Freqtrade 2025 compatibility.  The core trading logic is kept
intact but the strategy now supports futures trading, new order types,
dynamic trailing stops and additional hooks introduced with recent Freqtrade
versions.  The parameters remain Hyperopt compatible.
"""

# Updated imports for Freqtrade 2025
from datetime import datetime, timedelta
from typing import Optional

import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.persistence import Trade
from freqtrade.strategy import (
    IStrategy,
    merge_informative_pair,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
)
from pandas import DataFrame


# --------------------------------------------------------------------------------------------------
# Indicator helpers
# --------------------------------------------------------------------------------------------------

def ssl_channels(dataframe: DataFrame, length: int = 7) -> tuple[DataFrame, DataFrame]:
    """SSL Channels indicator"""
    df = dataframe.copy()
    df["ATR"] = ta.ATR(df, timeperiod=14)
    df["smaHigh"] = df["high"].rolling(length).mean() + df["ATR"]
    df["smaLow"] = df["low"].rolling(length).mean() - df["ATR"]
    df["hlv"] = np.where(df["close"] > df["smaHigh"], 1, np.where(df["close"] < df["smaLow"], -1, np.nan))
    df["hlv"] = df["hlv"].ffill()
    df["sslDown"] = np.where(df["hlv"] < 0, df["smaHigh"], df["smaLow"])
    df["sslUp"] = np.where(df["hlv"] < 0, df["smaLow"], df["smaHigh"])
    return df["sslDown"], df["sslUp"]


# --------------------------------------------------------------------------------------------------
# Strategy class
# --------------------------------------------------------------------------------------------------


class CombinedBinHClucAndMADV9(IStrategy):
    """CombinedBinHClucAndMADV9 with futures and dynamic trailing stop"""

    INTERFACE_VERSION = 3  # Still valid for 2025

    # --- Futures configuration -------------------------------------------------------------------
    position = "long"
    can_short = False
    margin_mode = "isolated"  # Futures isolated margin

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        side: str,
        **kwargs,
    ) -> float:
        """Always use 10x leverage"""
        return 10.0

    # --- ROI table --------------------------------------------------------------------------------
    minimal_roi = {
        "0": 0.028,  # I feel lucky!
        "10": 0.018,
        "40": 0.005,
    }

    stoploss = -0.10  # Base stoploss, actual values controlled via custom_stoploss

    timeframe = "5m"
    inf_1h = "1h"

    # --- Exit signal configuration (renamed from sell*) ------------------------------------------
    use_exit_signal = True
    exit_profit_only = False
    exit_profit_offset = 0.001  # Minimal profit to consider
    ignore_roi_if_entry_signal = False

    # --- Trailing stop configuration --------------------------------------------------------------
    trailing_stop = True  # Enable built-in trailing stop
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.025

    # --- Custom stoploss -------------------------------------------------------------------------
    use_custom_stoploss = True

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    # --- Order types (updated for Freqtrade 2025) -------------------------------------------------
    order_types = {
        "entry": "market",
        "exit": "market",
        "emergency_exit": "market",
        "force_entry": "market",
        "force_exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": True,
    }

    order_time_in_force = {
        "entry": "gtc",
        "exit": "gtc",
    }

    # --- Hyperopt parameters ---------------------------------------------------------------------
    buy_condition_0_enable = CategoricalParameter([True, False], default=True, space="buy", optimize=False, load=True)
    buy_condition_1_enable = CategoricalParameter([True, False], default=True, space="buy", optimize=False, load=True)
    buy_condition_2_enable = CategoricalParameter([True, False], default=True, space="buy", optimize=False, load=True)
    buy_condition_3_enable = CategoricalParameter([True, False], default=True, space="buy", optimize=False, load=True)
    buy_condition_4_enable = CategoricalParameter([True, False], default=True, space="buy", optimize=False, load=True)
    buy_condition_5_enable = CategoricalParameter([True, False], default=True, space="buy", optimize=False, load=True)
    buy_condition_6_enable = CategoricalParameter([True, False], default=True, space="buy", optimize=False, load=True)
    buy_condition_7_enable = CategoricalParameter([True, False], default=True, space="buy", optimize=False, load=True)
    buy_condition_8_enable = CategoricalParameter([True, False], default=True, space="buy", optimize=False, load=True)
    buy_condition_9_enable = CategoricalParameter([True, False], default=True, space="buy", optimize=False, load=True)
    buy_condition_10_enable = CategoricalParameter([True, False], default=True, space="buy", optimize=False, load=True)

    buy_bb20_close_bblowerband_safe_1 = DecimalParameter(0.7, 1.1, default=0.99, space="buy", optimize=False, load=True)
    buy_bb20_close_bblowerband_safe_2 = DecimalParameter(0.7, 1.1, default=0.982, space="buy", optimize=False, load=True)

    buy_volume_pump_1 = DecimalParameter(0.1, 0.9, default=0.4, space="buy", decimals=1, optimize=False, load=True)
    buy_volume_drop_1 = DecimalParameter(1, 10, default=4, space="buy", decimals=1, optimize=False, load=True)

    buy_rsi_1h_1 = DecimalParameter(10.0, 40.0, default=16.5, space="buy", decimals=1, optimize=False, load=True)
    buy_rsi_1h_2 = DecimalParameter(10.0, 40.0, default=15.0, space="buy", decimals=1, optimize=False, load=True)
    buy_rsi_1h_3 = DecimalParameter(10.0, 40.0, default=20.0, space="buy", decimals=1, optimize=False, load=True)
    buy_rsi_1h_4 = DecimalParameter(10.0, 40.0, default=35.0, space="buy", decimals=1, optimize=False, load=True)

    buy_rsi_1 = DecimalParameter(10.0, 40.0, default=28.0, space="buy", decimals=1, optimize=False, load=True)
    buy_rsi_2 = DecimalParameter(7.0, 40.0, default=10.0, space="buy", decimals=1, optimize=False, load=True)
    buy_rsi_3 = DecimalParameter(7.0, 40.0, default=14.2, space="buy", decimals=1, optimize=False, load=True)

    buy_macd_1 = DecimalParameter(0.01, 0.09, default=0.02, space="buy", decimals=2, optimize=False, load=True)
    buy_macd_2 = DecimalParameter(0.01, 0.09, default=0.03, space="buy", decimals=2, optimize=False, load=True)

    # --------------------------------------------------------------------------------------------------
    # Custom stoploss with dynamic trailing
    # --------------------------------------------------------------------------------------------------

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        """Dynamic stoploss in addition to built-in trailing stop"""

        # Profit based trailing - allow profit to run but protect at 50%
        if current_profit > 0.05:
            return current_profit * 0.5
        if current_profit > 0.02:
            return 0.02

        # Original drawdown protection from legacy strategy
        trade_time_50 = current_time - timedelta(minutes=50)
        if trade_time_50 > trade.open_date_utc:
            try:
                number_of_candle_shift = int((trade_time_50 - trade.open_date_utc).total_seconds() / 300)
                dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                candle = dataframe.iloc[-number_of_candle_shift].squeeze()
                if current_rate * 1.015 < candle["open"]:
                    return 0.01
            except IndexError:
                return 0.01

        return self.stoploss

    # --------------------------------------------------------------------------------------------------
    # Informative pairs / indicators
    # --------------------------------------------------------------------------------------------------

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, "1h") for pair in pairs]
        return informative_pairs

    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        informative_1h = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=self.inf_1h)
        informative_1h["ema_50"] = ta.EMA(informative_1h, timeperiod=50)
        informative_1h["ema_200"] = ta.EMA(informative_1h, timeperiod=200)
        informative_1h["rsi"] = ta.RSI(informative_1h, timeperiod=14)
        ssl_down_1h, ssl_up_1h = ssl_channels(informative_1h, 20)
        informative_1h["ssl_down"] = ssl_down_1h
        informative_1h["ssl_up"] = ssl_up_1h
        return informative_1h

    def normal_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_middleband"] = bollinger["mid"]
        dataframe["bb_upperband"] = bollinger["upper"]
        dataframe["volume_mean_slow"] = dataframe["volume"].rolling(window=30).mean()
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["ema_26"] = ta.EMA(dataframe, timeperiod=26)
        dataframe["ema_12"] = ta.EMA(dataframe, timeperiod=12)
        dataframe["sma_5"] = ta.EMA(dataframe, timeperiod=5)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        informative_1h = self.informative_1h_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(dataframe, informative_1h, self.timeframe, self.inf_1h, ffill=True)
        dataframe = self.normal_tf_indicators(dataframe, metadata)
        return dataframe

    # --------------------------------------------------------------------------------------------------
    # Entry / Exit
    # --------------------------------------------------------------------------------------------------

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["buy"] = 0

        # --- Condition 1 -------------------------------------------------------------------------
        cond1 = (
            self.buy_condition_1_enable.value
            & (dataframe["close"] > dataframe["ema_200"])
            & (dataframe["close"] > dataframe["ema_200_1h"])
            & (dataframe["close"] < dataframe["bb_lowerband"] * self.buy_bb20_close_bblowerband_safe_1.value)
            & (dataframe["volume_mean_slow"] > dataframe["volume_mean_slow"].shift(30) * self.buy_volume_pump_1.value)
            & (dataframe["volume"] < (dataframe["volume"].shift() * self.buy_volume_drop_1.value))
            & (dataframe["open"] - dataframe["close"] < dataframe["bb_upperband"].shift(2) - dataframe["bb_lowerband"].shift(2))
            & (dataframe["volume"] > 0)
        )
        dataframe.loc[cond1, ["buy", "buy_tag"]] = (1, "bc1")

        # --- Condition 2 -------------------------------------------------------------------------
        cond2 = (
            self.buy_condition_2_enable.value
            & (dataframe["close"] > dataframe["ema_200"])
            & (dataframe["close"] < dataframe["bb_lowerband"] * self.buy_bb20_close_bblowerband_safe_2.value)
            & (dataframe["volume_mean_slow"] > dataframe["volume_mean_slow"].shift(30) * self.buy_volume_pump_1.value)
            & (dataframe["volume"] < (dataframe["volume"].shift() * self.buy_volume_drop_1.value))
            & (dataframe["open"] - dataframe["close"] < dataframe["bb_upperband"].shift(2) - dataframe["bb_lowerband"].shift(2))
            & (dataframe["volume"] > 0)
        )
        dataframe.loc[cond2, ["buy", "buy_tag"]] = (1, "bc2")

        # --- Condition 3 -------------------------------------------------------------------------
        cond3 = (
            self.buy_condition_3_enable.value
            & (dataframe["close"] > dataframe["ema_200_1h"])
            & (dataframe["close"] < dataframe["bb_lowerband"])
            & (dataframe["rsi"] < self.buy_rsi_3.value)
            & (dataframe["volume"] < (dataframe["volume"].shift() * self.buy_volume_drop_1.value))
            & (dataframe["volume"] > 0)
        )
        dataframe.loc[cond3, ["buy", "buy_tag"]] = (1, "bc3")

        # --- Condition 4 -------------------------------------------------------------------------
        cond4 = (
            self.buy_condition_4_enable.value
            & (dataframe["rsi_1h"] < self.buy_rsi_1h_1.value)
            & (dataframe["close"] < dataframe["bb_lowerband"])
            & (dataframe["volume"] < (dataframe["volume"].shift() * self.buy_volume_drop_1.value))
            & (dataframe["volume"] > 0)
        )
        dataframe.loc[cond4, ["buy", "buy_tag"]] = (1, "bc4")

        # --- Condition 5 -------------------------------------------------------------------------
        cond5 = (
            self.buy_condition_5_enable.value
            & (dataframe["close"] > dataframe["ema_200"])
            & (dataframe["close"] > dataframe["ema_200_1h"])
            & (dataframe["ema_26"] > dataframe["ema_12"])
            & ((dataframe["ema_26"] - dataframe["ema_12"]) > (dataframe["open"] * self.buy_macd_1.value))
            & ((dataframe["ema_26"].shift() - dataframe["ema_12"].shift()) > (dataframe["open"] / 100))
            & (dataframe["close"] < dataframe["bb_lowerband"])
            & (dataframe["volume"] < (dataframe["volume"].shift() * self.buy_volume_drop_1.value))
            & (dataframe["volume_mean_slow"] > dataframe["volume_mean_slow"].shift(30) * self.buy_volume_pump_1.value)
            & (dataframe["volume"] > 0)
        )
        dataframe.loc[cond5, ["buy", "buy_tag"]] = (1, "bc5")

        # --- Condition 6 -------------------------------------------------------------------------
        cond6 = (
            self.buy_condition_6_enable.value
            & (dataframe["ema_26"] > dataframe["ema_12"])
            & ((dataframe["ema_26"] - dataframe["ema_12"]) > (dataframe["open"] * self.buy_macd_2.value))
            & ((dataframe["ema_26"].shift() - dataframe["ema_12"].shift()) > (dataframe["open"] / 100))
            & (dataframe["close"] < dataframe["bb_lowerband"])
            & (dataframe["volume"] < (dataframe["volume"].shift() * self.buy_volume_drop_1.value))
            & (dataframe["volume"] > 0)
        )
        dataframe.loc[cond6, ["buy", "buy_tag"]] = (1, "bc6")

        # --- Condition 7 -------------------------------------------------------------------------
        cond7 = (
            self.buy_condition_7_enable.value
            & (dataframe["rsi_1h"] < self.buy_rsi_1h_2.value)
            & (dataframe["ema_26"] > dataframe["ema_12"])
            & ((dataframe["ema_26"] - dataframe["ema_12"]) > (dataframe["open"] * self.buy_macd_1.value))
            & ((dataframe["ema_26"].shift() - dataframe["ema_12"].shift()) > (dataframe["open"] / 100))
            & (dataframe["volume"] < (dataframe["volume"].shift() * self.buy_volume_drop_1.value))
            & (dataframe["volume_mean_slow"] > dataframe["volume_mean_slow"].shift(30) * self.buy_volume_pump_1.value)
            & (dataframe["volume"] > 0)
        )
        dataframe.loc[cond7, ["buy", "buy_tag"]] = (1, "bc7")

        # --- Condition 8 -------------------------------------------------------------------------
        cond8 = (
            self.buy_condition_8_enable.value
            & (dataframe["rsi_1h"] < self.buy_rsi_1h_3.value)
            & (dataframe["rsi"] < self.buy_rsi_1.value)
            & (dataframe["volume"] < (dataframe["volume"].shift() * self.buy_volume_drop_1.value))
            & (dataframe["volume_mean_slow"] > dataframe["volume_mean_slow"].shift(30) * self.buy_volume_pump_1.value)
            & (dataframe["volume"] > 0)
        )
        dataframe.loc[cond8, ["buy", "buy_tag"]] = (1, "bc8")

        # --- Condition 9 -------------------------------------------------------------------------
        cond9 = (
            self.buy_condition_9_enable.value
            & (dataframe["rsi_1h"] < self.buy_rsi_1h_4.value)
            & (dataframe["rsi"] < self.buy_rsi_2.value)
            & (dataframe["volume"] < (dataframe["volume"].shift() * self.buy_volume_drop_1.value))
            & (dataframe["volume_mean_slow"] > dataframe["volume_mean_slow"].shift(30) * self.buy_volume_pump_1.value)
            & (dataframe["volume"] > 0)
        )
        dataframe.loc[cond9, ["buy", "buy_tag"]] = (1, "bc9")

        # --- Condition 10 ------------------------------------------------------------------------
        cond10 = (
            self.buy_condition_10_enable.value
            & (dataframe["close"] < dataframe["sma_5"])
            & (dataframe["ssl_up_1h"] > dataframe["ssl_down_1h"])
            & (dataframe["ema_50_1h"] > dataframe["ema_200_1h"])
            & (dataframe["rsi"] < dataframe["rsi_1h"] - 43.276)
            & (dataframe["volume"] > 0)
        )
        dataframe.loc[cond10, ["buy", "buy_tag"]] = (1, "bc10")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["sell"] = 0
        cond_exit = (
            (dataframe["close"] > dataframe["bb_middleband"] * 1.01)
            & (dataframe["volume"] > 0)
        )
        dataframe.loc[cond_exit, ["sell", "sell_reason"]] = (1, "bb_mid")
        return dataframe

    # --------------------------------------------------------------------------------------------------
    # Optional hooks for new strategy interface (2025) -------------------------------------------------
    # --------------------------------------------------------------------------------------------------

    def custom_exit_price(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[float]:
        """No custom exit price used, provided for compatibility"""
        return None

    def min_roi_entry(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        entry_tag: str,
        **kwargs,
    ) -> Optional[float]:
        """Use default minimal ROI but hook exists for future customisation"""
        return None

    def trade_adjust_exit(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[float]:
        """No extra exit adjustment"""
        return None

