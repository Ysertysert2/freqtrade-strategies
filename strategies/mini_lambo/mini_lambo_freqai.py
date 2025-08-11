import logging
from datetime import datetime, timezone
from functools import reduce
from typing import Dict

import numpy as np
import pandas_ta as pta
import talib.abstract as ta
from pandas import DataFrame, Series

from freqtrade.persistence import Trade
from freqtrade.strategy import DecimalParameter, IntParameter
from freqtrade.freqai.freqai_interface import FreqaiStrategy

logger = logging.getLogger(__name__)


class MiniLamboFreqAI(FreqaiStrategy):
    """FreqAI-enabled version of MiniLambo strategy.

    Keeps original protections, custom stoploss and trailing-buy logic
    while replacing manual entry conditions with FreqAI predictions.
    """

    INTERFACE_VERSION = 3

    # Protection hyperspace params
    protection_params = {
        "protection_cooldown_period": 2,
        "protection_maxdrawdown_lookback_period": 35,
        "protection_maxdrawdown_max_allowed_drawdown": 0.097,
        "protection_maxdrawdown_stop_duration": 1,
        "protection_maxdrawdown_trade_limit": 6,
        "protection_stoplossguard_lookback_period": 16,
        "protection_stoplossguard_stop_duration": 29,
        "protection_stoplossguard_trade_limit": 3,
    }

    protection_cooldown_period = IntParameter(
        low=1,
        high=48,
        default=protection_params["protection_cooldown_period"],
        space="protection",
        optimize=True,
    )
    protection_maxdrawdown_lookback_period = IntParameter(
        low=1,
        high=48,
        default=protection_params["protection_maxdrawdown_lookback_period"],
        space="protection",
        optimize=True,
    )
    protection_maxdrawdown_trade_limit = IntParameter(
        low=1,
        high=8,
        default=protection_params["protection_maxdrawdown_trade_limit"],
        space="protection",
        optimize=True,
    )
    protection_maxdrawdown_stop_duration = IntParameter(
        low=1,
        high=48,
        default=protection_params["protection_maxdrawdown_stop_duration"],
        space="protection",
        optimize=True,
    )
    protection_maxdrawdown_max_allowed_drawdown = DecimalParameter(
        low=0.01,
        high=0.20,
        default=protection_params["protection_maxdrawdown_max_allowed_drawdown"],
        space="protection",
        optimize=True,
    )
    protection_stoplossguard_lookback_period = IntParameter(
        low=1,
        high=48,
        default=protection_params["protection_stoplossguard_lookback_period"],
        space="protection",
        optimize=True,
    )
    protection_stoplossguard_trade_limit = IntParameter(
        low=1,
        high=8,
        default=protection_params["protection_stoplossguard_trade_limit"],
        space="protection",
        optimize=True,
    )
    protection_stoplossguard_stop_duration = IntParameter(
        low=1,
        high=48,
        default=protection_params["protection_stoplossguard_stop_duration"],
        space="protection",
        optimize=True,
    )

    @property
    def protections(self):
        return [
            {"method": "CooldownPeriod", "stop_duration": self.protection_cooldown_period.value},
            {
                "method": "MaxDrawdown",
                "lookback_period": self.protection_maxdrawdown_lookback_period.value,
                "trade_limit": self.protection_maxdrawdown_trade_limit.value,
                "stop_duration": self.protection_maxdrawdown_stop_duration.value,
                "max_allowed_drawdown": self.protection_maxdrawdown_max_allowed_drawdown.value,
            },
            {
                "method": "StoplossGuard",
                "lookback_period": self.protection_stoplossguard_lookback_period.value,
                "trade_limit": self.protection_stoplossguard_trade_limit.value,
                "stop_duration": self.protection_stoplossguard_stop_duration.value,
                "only_per_pair": False,
            },
        ]

    # ROI hyperspace params
    roi_t1 = IntParameter(240, 720, default=400, space="roi", optimize=True)
    roi_t2 = IntParameter(120, 240, default=154, space="roi", optimize=True)
    roi_t3 = IntParameter(90, 120, default=112, space="roi", optimize=True)
    roi_t4 = IntParameter(60, 90, default=81, space="roi", optimize=True)
    roi_t5 = IntParameter(30, 60, default=51, space="roi", optimize=True)
    roi_t6 = IntParameter(1, 30, default=15, space="roi", optimize=True)

    @property
    def minimal_roi(self) -> Dict[int, float]:
        return {
            0: 0.05,
            self.roi_t6.value: 0.04,
            self.roi_t5.value: 0.03,
            self.roi_t4.value: 0.02,
            self.roi_t3.value: 0.01,
            self.roi_t2.value: 0.0001,
            self.roi_t1.value: -10,
        }

    # Stoploss
    stoploss = -0.10

    # Trailing stop
    trailing_stop = False
    trailing_stop_positive = 0.3207
    trailing_stop_positive_offset = 0.3849
    trailing_only_offset_is_reached = False

    timeframe = "1m"
    use_sell_signal = False
    sell_profit_only = False
    ignore_roi_if_buy_signal = False
    use_custom_stoploss = True
    process_only_new_candles = True
    startup_candle_count = 200

    plot_config = {
        "main_plot": {
            "ema_14": {"color": "#888b48", "type": "line"},
            "buy_sell": {
                "sell_tag": {"color": "red"},
                "buy_tag": {"color": "blue"},
            },
        },
        "subplots": {
            "rsi4": {"rsi_4": {"color": "#888b48", "type": "line"}},
            "rsi14": {"rsi_14": {"color": "#888b48", "type": "line"}},
            "cti": {"cti": {"color": "#573892", "type": "line"}},
            "ewo": {"EWO": {"color": "#573892", "type": "line"}},
            "pct_change": {"pct_change": {"color": "#26782f", "type": "line"}},
        },
    }

    # Buy hyperspace params (kept for backwards compatibility, not used directly)
    buy_params = {
        "lambo2_pct_change_high_period": 109,
        "lambo2_pct_change_high_ratio": -0.235,
        "lambo2_pct_change_low_period": 20,
        "lambo2_pct_change_low_ratio": -0.06,
        "lambo2_ema_14_factor": 0.981,
        "lambo2_rsi_14_limit": 39,
        "lambo2_rsi_4_limit": 44,
    }
    lambo2_ema_14_factor = DecimalParameter(0.8, 1.2, decimals=3, default=buy_params["lambo2_ema_14_factor"], space="buy", optimize=True)
    lambo2_rsi_4_limit = IntParameter(5, 60, default=buy_params["lambo2_rsi_4_limit"], space="buy", optimize=True)
    lambo2_rsi_14_limit = IntParameter(5, 60, default=buy_params["lambo2_rsi_14_limit"], space="buy", optimize=True)
    lambo2_pct_change_low_period = IntParameter(1, 60, default=buy_params["lambo2_pct_change_low_period"], space="buy", optimize=True)
    lambo2_pct_change_low_ratio = DecimalParameter(low=-0.20, high=-0.01, decimals=3, default=buy_params["lambo2_pct_change_low_ratio"], space="buy", optimize=True)
    lambo2_pct_change_high_period = IntParameter(1, 180, default=buy_params["lambo2_pct_change_high_period"], space="buy", optimize=True)
    lambo2_pct_change_high_ratio = DecimalParameter(low=-0.30, high=-0.01, decimals=3, default=buy_params["lambo2_pct_change_high_ratio"], space="buy", optimize=True)

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, "1h") for pair in pairs]
        informative_pairs += [("BTC/USDT", "1m")]
        informative_pairs += [("BTC/USDT", "1d")]
        return informative_pairs

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:
        sl_new = 1
        if current_profit > 0.2:
            sl_new = 0.05
        elif current_profit > 0.1:
            sl_new = 0.03
        elif current_profit > 0.06:
            sl_new = 0.02
        elif current_profit > 0.03:
            sl_new = 0.015
        elif current_profit > 0.015:
            sl_new = 0.0075
        return sl_new

    # ----------------------------------------------------------------------
    # FreqAI feature & target generation
    # ----------------------------------------------------------------------
    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ) -> DataFrame:
        """Features auto-expanded on indicator periods."""
        dataframe[f"%-ema{period}"] = ta.EMA(dataframe, timeperiod=period)
        dataframe[f"%-rsi{period}"] = ta.RSI(dataframe, timeperiod=period)
        dataframe[f"%-cti{period}"] = pta.cti(dataframe["close"], length=period)
        dataframe[f"%-adx{period}"] = ta.ADX(dataframe, timeperiod=period)
        dataframe[f"%-mfi{period}"] = ta.MFI(dataframe, timeperiod=period)
        dataframe[f"%-roc{period}"] = ta.ROC(dataframe, timeperiod=period)
        bb_upper, bb_middle, bb_lower = ta.BBANDS(dataframe["close"], timeperiod=period)
        dataframe[f"%-bbwidth{period}"] = bb_upper - bb_lower
        dataframe[f"%-close_bb_lower{period}"] = dataframe["close"] / bb_lower
        dataframe[f"%-relvol{period}"] = dataframe["volume"] / dataframe["volume"].rolling(period).mean()
        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-volume"] = dataframe["volume"]
        dataframe["%-price"] = dataframe["close"]
        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        df = dataframe.copy()
        df["%ewo"] = (ta.EMA(df, timeperiod=50) - ta.EMA(df, timeperiod=200)) / df["low"] * 100
        df["%perc"] = (df["high"] - df["low"]) / df["low"] * 100
        df["%perc_norm"] = (
            df["%perc"] - df["%perc"].rolling(50).min()
        ) / (df["%perc"].rolling(50).max() - df["%perc"].rolling(50).min())
        df["%atr14"] = ta.ATR(df, timeperiod=14)
        bb_upper, bb_middle, bb_lower = ta.BBANDS(df["close"], timeperiod=20)
        df["%bbwidth"] = bb_upper - bb_lower
        df["%volume_norm"] = (
            (df["volume"] - df["volume"].rolling(50).min())
            / (df["volume"].rolling(50).max() - df["volume"].rolling(50).min())
        )
        df["%-day_of_week"] = (df["date"].dt.dayofweek + 1) / 7
        df["%-hour_of_day"] = (df["date"].dt.hour + 1) / 25
        features = [c for c in df.columns if c.startswith("%")]
        df = df.dropna(subset=features).reset_index(drop=True)
        return df

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Create classification target using configured label period."""
        label_period = self.freqai_info["feature_parameters"]["label_period_candles"]
        future = dataframe["close"].shift(-label_period)
        change = (future - dataframe["close"]) / dataframe["close"]
        dataframe["&-target"] = 0
        dataframe.loc[change > 0.005, "&-target"] = 1
        dataframe.loc[change < -0.005, "&-target"] = -1
        return dataframe

    def freqai_select_signals(self, dataframe: DataFrame) -> DataFrame:
        dataframe["buy"] = 0
        dataframe["sell"] = 0
        dataframe["buy_tag"] = ""
        dataframe["sell_tag"] = ""
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        prediction = dataframe.get("freqai_prediction") or dataframe.get("prediction")
        if prediction is not None:
            dataframe.loc[prediction > 0.55, ["buy", "buy_tag", "enter_long"]] = [1, "freqai-long", 1]
            dataframe.loc[prediction < 0.45, ["sell", "sell_tag", "enter_short"]] = [1, "freqai-short", 1]
        return dataframe

    # ----------------------------------------------------------------------
    # Indicator population and entry/exit logic with trailing buy support
    # ----------------------------------------------------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if getattr(self, "freqai", None):
            if hasattr(self.freqai, "process"):
                dataframe = self.freqai.process(dataframe, metadata, self)
            else:  # pragma: no cover
                dataframe = self.freqai.start(dataframe, metadata, self)
        self.trailing_buy(metadata["pair"])
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.freqai_select_signals(dataframe)
        if self.trailing_buy_order_enabled and self.config["runmode"].value in ("live", "dry_run"):
            last_candle = dataframe.iloc[-1].squeeze()
            trailing_buy = self.trailing_buy(metadata["pair"])
            if last_candle["buy"] == 1:
                if not trailing_buy["trailing_buy_order_started"]:
                    open_trades = Trade.get_trades([
                        Trade.pair == metadata["pair"],
                        Trade.is_open.is_(True),
                    ]).all()
                    if not open_trades:
                        logger.info(f"Set 'allow_trailing' to True for {metadata['pair']} to start trailing!!!")
                        trailing_buy["allow_trailing"] = True
                        initial_buy_tag = last_candle.get("buy_tag", "buy signal")
                        dataframe.loc[:, "buy_tag"] = f"{initial_buy_tag} (start trail price {last_candle['close']})"
            else:
                if trailing_buy["trailing_buy_order_started"]:
                    logger.info(f"Continue trailing for {metadata['pair']}. Manually trigger buy signal!!")
                    dataframe.loc[:, "buy"] = 1
                    dataframe.loc[:, "buy_tag"] = trailing_buy["buy_tag"]
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        trade.sell_reason = f"{sell_reason} ({trade.buy_tag})"
        return True

    # ----------------------------------------------------------------------
    # Trailing buy helpers (unchanged from original strategy)
    # ----------------------------------------------------------------------
    custom_info_trail_buy: dict = {}
    trailing_buy_order_enabled = True
    trailing_expire_seconds = 1800
    trailing_buy_uptrend_enabled = False
    trailing_expire_seconds_uptrend = 90
    min_uptrend_trailing_profit = 0.02
    debug_mode = True
    trailing_buy_max_stop = 0.02
    trailing_buy_max_buy = 0.000

    init_trailing_dict = {
        "trailing_buy_order_started": False,
        "trailing_buy_order_uplimit": 0,
        "start_trailing_price": 0,
        "buy_tag": None,
        "start_trailing_time": None,
        "offset": 0,
        "allow_trailing": False,
    }

    def trailing_buy(self, pair, reinit: bool = False):
        if pair not in self.custom_info_trail_buy:
            self.custom_info_trail_buy[pair] = {}
        if reinit or "trailing_buy" not in self.custom_info_trail_buy[pair]:
            self.custom_info_trail_buy[pair]["trailing_buy"] = self.init_trailing_dict.copy()
        return self.custom_info_trail_buy[pair]["trailing_buy"]

    def trailing_buy_info(self, pair: str, current_price: float):
        if not self.debug_mode:
            return
        current_time = datetime.now(timezone.utc)
        trailing_buy = self.trailing_buy(pair)
        try:
            duration = current_time - trailing_buy["start_trailing_time"]
        except TypeError:
            duration = 0
        logger.info(
            f"pair: {pair} : start: {trailing_buy['start_trailing_price']:.4f}, "
            f"duration: {duration}, current: {current_price:.4f}, "
            f"uplimit: {trailing_buy['trailing_buy_order_uplimit']:.4f}, "
            f"profit: {self.current_trailing_profit_ratio(pair, current_price) * 100:.2f}%, "
            f"offset: {trailing_buy['offset']}"
        )

    def current_trailing_profit_ratio(self, pair: str, current_price: float) -> float:
        trailing_buy = self.trailing_buy(pair)
        if trailing_buy["trailing_buy_order_started"]:
            return (trailing_buy["start_trailing_price"] - current_price) / trailing_buy["start_trailing_price"]
        return 0

    def trailing_buy_offset(self, dataframe: DataFrame, pair: str, current_price: float):
        current_trailing_profit_ratio = self.current_trailing_profit_ratio(pair, current_price)
        last_candle = dataframe.iloc[-1]
        adapt = abs(last_candle["%perc_norm"])
        default_offset = 0.004 * (1 + adapt)
        trailing_buy = self.trailing_buy(pair)
        if not trailing_buy["trailing_buy_order_started"]:
            return default_offset
        last_candle = dataframe.iloc[-1]
        current_time = datetime.now(timezone.utc)
        trailing_duration = current_time - trailing_buy["start_trailing_time"]
        if trailing_duration.total_seconds() > self.trailing_expire_seconds:
            if current_trailing_profit_ratio > 0 and last_candle["buy"] == 1:
                return "forcebuy"
            return None
        elif (
            self.trailing_buy_uptrend_enabled
            and trailing_duration.total_seconds() < self.trailing_expire_seconds_uptrend
            and current_trailing_profit_ratio < (-1 * self.min_uptrend_trailing_profit)
        ):
            return "forcebuy"
        if current_trailing_profit_ratio < 0:
            return default_offset
        trailing_buy_offset = {0.06: 0.02, 0.03: 0.01, 0: default_offset}
        for key in trailing_buy_offset:
            if current_trailing_profit_ratio > key:
                return trailing_buy_offset[key]
        return default_offset

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, **kwargs) -> bool:
        val = super().confirm_trade_entry(pair, order_type, amount, rate, time_in_force, **kwargs)
        if val:
            if self.trailing_buy_order_enabled and self.config["runmode"].value in ("live", "dry_run"):
                val = False
                dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                if len(dataframe) >= 1:
                    last_candle = dataframe.iloc[-1].squeeze()
                    current_price = rate
                    trailing_buy = self.trailing_buy(pair)
                    trailing_buy_offset = self.trailing_buy_offset(dataframe, pair, current_price)
                    if trailing_buy["allow_trailing"]:
                        if not trailing_buy["trailing_buy_order_started"] and last_candle["buy"] == 1:
                            trailing_buy["trailing_buy_order_started"] = True
                            trailing_buy["trailing_buy_order_uplimit"] = last_candle["close"]
                            trailing_buy["start_trailing_price"] = last_candle["close"]
                            trailing_buy["buy_tag"] = last_candle["buy_tag"]
                            trailing_buy["start_trailing_time"] = datetime.now(timezone.utc)
                            trailing_buy["offset"] = 0
                            self.trailing_buy_info(pair, current_price)
                            logger.info(f"start trailing buy for {pair} at {last_candle['close']}")
                        elif trailing_buy["trailing_buy_order_started"]:
                            if trailing_buy_offset == "forcebuy":
                                val = True
                                ratio = f"{self.current_trailing_profit_ratio(pair, current_price) * 100:.2f}"
                                self.trailing_buy_info(pair, current_price)
                                logger.info(
                                    f"price OK for {pair} ({ratio} %, {current_price}), order may not be triggered if all slots are full"
                                )
                            elif trailing_buy_offset is None:
                                self.trailing_buy(pair, reinit=True)
                                logger.info(f"STOP trailing buy for {pair} because \"trailing buy offset\" returned None")
                            elif current_price < trailing_buy["trailing_buy_order_uplimit"]:
                                old_uplimit = trailing_buy["trailing_buy_order_uplimit"]
                                self.custom_info_trail_buy[pair]["trailing_buy"]["trailing_buy_order_uplimit"] = min(
                                    current_price * (1 + trailing_buy_offset),
                                    self.custom_info_trail_buy[pair]["trailing_buy"]["trailing_buy_order_uplimit"],
                                )
                                self.custom_info_trail_buy[pair]["trailing_buy"]["offset"] = trailing_buy_offset
                                self.trailing_buy_info(pair, current_price)
                                logger.info(
                                    f"update trailing buy for {pair} at {old_uplimit} -> {self.custom_info_trail_buy[pair]['trailing_buy']['trailing_buy_order_uplimit']}"
                                )
                            elif current_price < trailing_buy["start_trailing_price"] * (1 + self.trailing_buy_max_buy):
                                val = True
                                ratio = f"{self.current_trailing_profit_ratio(pair, current_price) * 100:.2f}"
                                self.trailing_buy_info(pair, current_price)
                                logger.info(
                                    f"current price ({current_price}) > uplimit ({trailing_buy['trailing_buy_order_uplimit']}) and lower than starting price ({trailing_buy['start_trailing_price'] * (1 + self.trailing_buy_max_buy)}). OK for {pair} ({ratio} %), order may not be triggered if all slots are full"
                                )
                            elif current_price > trailing_buy["start_trailing_price"] * (1 + self.trailing_buy_max_stop):
                                self.trailing_buy(pair, reinit=True)
                                self.trailing_buy_info(pair, current_price)
                                logger.info(
                                    f"STOP trailing buy for {pair} because of the price is higher than starting price * {1 + self.trailing_buy_max_stop}"
                                )
                            else:
                                self.trailing_buy_info(pair, current_price)
                                logger.info(f"price too high for {pair} !")
                    else:
                        logger.info(f"Wait for next buy signal for {pair}")
                if val:
                    self.trailing_buy_info(pair, rate)
                    self.trailing_buy(pair, reinit=True)
                    logger.info(f"STOP trailing buy for {pair} because I buy it")
        return val


# Helper indicator

def EWO(dataframe: DataFrame, ema_length: int = 5, ema2_length: int = 35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df["low"] * 100
    return emadif
