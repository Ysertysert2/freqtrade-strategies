import logging
from datetime import UTC, datetime

import pandas_ta as pta
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import DecimalParameter, IntParameter
from freqtrade.strategy.interface import IStrategy


logger = logging.getLogger(__name__)

class MiniLamboFreqAI(IStrategy):
    """
    FreqAI-enabled version of MiniLambo strategy.
    Trailing BUY (long) et SELL (short) intelligent, protections, custom stoploss.
    """

    INTERFACE_VERSION = 3
    can_short = True

    # ---------------- TRAILING BUY (LONG) ----------------
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

    # ---------------- TRAILING SELL (SHORT) ----------------
    custom_info_trail_sell: dict = {}
    trailing_sell_order_enabled = True
    trailing_sell_expire_seconds = 1800
    trailing_sell_downtrend_enabled = False
    trailing_expire_seconds_downtrend = 90
    min_downtrend_trailing_profit = 0.02
    trailing_sell_max_stop = 0.02
    trailing_sell_max_sell = 0.000
    init_trailing_sell_dict = {
        "trailing_sell_order_started": False,
        "trailing_sell_order_downlimit": 0,
        "start_trailing_price": 0,
        "sell_tag": None,
        "start_trailing_time": None,
        "offset": 0,
        "allow_trailing": False,
    }

    # -------------- PROTECTIONS -----------------
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
        low=1, high=48, default=protection_params["protection_cooldown_period"], space="protection", optimize=True  # noqa: E501
    )
    protection_maxdrawdown_lookback_period = IntParameter(
        low=1, high=48, default=protection_params["protection_maxdrawdown_lookback_period"], space="protection", optimize=True  # noqa: E501
    )
    protection_maxdrawdown_trade_limit = IntParameter(
        low=1, high=8, default=protection_params["protection_maxdrawdown_trade_limit"], space="protection", optimize=True  # noqa: E501
    )
    protection_maxdrawdown_stop_duration = IntParameter(
        low=1, high=48, default=protection_params["protection_maxdrawdown_stop_duration"], space="protection", optimize=True  # noqa: E501
    )
    protection_maxdrawdown_max_allowed_drawdown = DecimalParameter(
        low=0.01, high=0.20, default=protection_params["protection_maxdrawdown_max_allowed_drawdown"], space="protection", optimize=True  # noqa: E501
    )
    protection_stoplossguard_lookback_period = IntParameter(
        low=1, high=48, default=protection_params["protection_stoplossguard_lookback_period"], space="protection", optimize=True  # noqa: E501
    )
    protection_stoplossguard_trade_limit = IntParameter(
        low=1, high=8, default=protection_params["protection_stoplossguard_trade_limit"], space="protection", optimize=True  # noqa: E501
    )
    protection_stoplossguard_stop_duration = IntParameter(
        low=1, high=48, default=protection_params["protection_stoplossguard_stop_duration"], space="protection", optimize=True  # noqa: E501
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

    # -------------- ROI & STOPLOSS -----------------
    roi_t1 = IntParameter(240, 720, default=400, space="roi", optimize=True)
    roi_t2 = IntParameter(120, 240, default=154, space="roi", optimize=True)
    roi_t3 = IntParameter(90, 120, default=112, space="roi", optimize=True)
    roi_t4 = IntParameter(60, 90, default=81, space="roi", optimize=True)
    roi_t5 = IntParameter(30, 60, default=51, space="roi", optimize=True)
    roi_t6 = IntParameter(1, 30, default=15, space="roi", optimize=True)

    minimal_roi = {
        "0": 0.05,
        "15": 0.04,
        "51": 0.03,
        "81": 0.02,
        "112": 0.01,
        "154": 0.0001,
        "400": -10,
    }
    stoploss = -0.10

        # ROI short hyperspace params
    roi_short_t1 = IntParameter(240, 720, default=350, space="roi_short", optimize=True)
    roi_short_t2 = IntParameter(120, 240, default=111, space="roi_short", optimize=True)
    roi_short_t3 = IntParameter(90, 120, default=80, space="roi_short", optimize=True)
    roi_short_t4 = IntParameter(60, 90, default=61, space="roi_short", optimize=True)
    roi_short_t5 = IntParameter(30, 60, default=35, space="roi_short", optimize=True)
    roi_short_t6 = IntParameter(1, 30, default=15, space="roi_short", optimize=True)

    minimal_roi_short = {
        "0": 0.045,
        "15": 0.032,
        "35": 0.023,
        "61": 0.014,
        "80": 0.01,
        "111": 0.0001,
        "350": -10,
    }

    stoploss_short = -0.07  # Stoploss short moins large pour scalping ou si marché très volatile


    trailing_stop = False
    trailing_stop_positive = 0.3207
    trailing_stop_positive_offset = 0.3849
    trailing_only_offset_is_reached = False

    timeframe = "1m"
    use_exit_signal = False
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    use_custom_stoploss = True
    process_only_new_candles = True
    startup_candle_count = 300

    # -------------- INDICATEURS -----------------
    plot_config = {
        "main_plot": {
            "ema_14": {"color": "#888b48", "type": "line"},
            "buy_sell": {"sell_tag": {"color": "red"}, "buy_tag": {"color": "blue"}},
        },
        "subplots": {
            "rsi4": {"rsi_4": {"color": "#888b48", "type": "line"}},
            "rsi14": {"rsi_14": {"color": "#888b48", "type": "line"}},
            "cti": {"cti": {"color": "#573892", "type": "line"}},
            "ewo": {"EWO": {"color": "#573892", "type": "line"}},
            "pct_change": {"pct_change": {"color": "#26782f", "type": "line"}},
        },
    }

    def minimal_roi_for_trade(self, trade: Trade) -> dict:
        is_short = getattr(trade, "is_short", False)
        if is_short:
            return self.minimal_roi_short
        return self.minimal_roi


    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, "1h") for pair in pairs]
        informative_pairs += [("BTC/USDT:USDT", "1m")]
        informative_pairs += [("BTC/USDT:USDT", "1d")]
        return informative_pairs

    def custom_stoploss(self, pair, trade, current_time, current_rate, current_profit, **kwargs) -> float:  # noqa: E501
        if getattr(trade, "is_short", False):
        # Stop short (plus serré)
            if current_profit > 0.10:
                return 0.01
            elif current_profit > 0.03:
             return 0.005
            return self.stoploss_short
        else:
        # Stop long (classique)
            if current_profit > 0.2:
                return 0.05
            elif current_profit > 0.1:
                return 0.03
            elif current_profit > 0.06:
                return 0.02
            elif current_profit > 0.03:
                return 0.015
            elif current_profit > 0.015:
                return 0.0075
        return self.stoploss


    # --------- FREQAI FEATURE & TARGET ---------
    def feature_engineering_expand_all(self, dataframe: DataFrame, period, metadata, **kwargs) -> DataFrame:  # noqa: E501
    # Toutes les features doivent être préfixées %
        dataframe[f"%-ema{period}"] = ta.EMA(dataframe, timeperiod=period)
        dataframe[f"%-rsi{period}"] = ta.RSI(dataframe, timeperiod=period)
        dataframe[f"%-cti{period}"] = pta.cti(dataframe["close"], length=period)
        dataframe[f"%-adx{period}"] = ta.ADX(dataframe, timeperiod=period)
        dataframe[f"%-mfi{period}"] = ta.MFI(dataframe, timeperiod=period)
        dataframe[f"%-roc{period}"] = ta.ROC(dataframe, timeperiod=period)
        # BBands width et ratio
        bb_upper, bb_middle, bb_lower = ta.BBANDS(dataframe["close"], timeperiod=period)
        dataframe[f"%-bbwidth{period}"] = bb_upper - bb_lower
        dataframe[f"%-close_bb_lower{period}"] = dataframe["close"] / bb_lower
        # Relative volume
        dataframe[f"%-relvol{period}"] = dataframe["volume"] / dataframe["volume"].rolling(period).mean()  # noqa: E501
        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata, **kwargs) -> DataFrame:  # noqa: E501
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-volume"] = dataframe["volume"]
        dataframe["%-price"] = dataframe["close"]
        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, metadata, **kwargs) -> DataFrame:
        df = dataframe.copy()
        df["%ewo"] = (ta.EMA(df, timeperiod=50) - ta.EMA(df, timeperiod=200)) / df["low"] * 100
        df["%perc"] = (df["high"] - df["low"]) / df["low"] * 100
        df["%perc_norm"] = (
        df["%perc"] - df["%perc"].rolling(50).min()
        ) / (df["%perc"].rolling(50).max() - df["%perc"].rolling(50).min())
        df["%atr14"] = ta.ATR(df, timeperiod=14)
        bb_upper, bb_middle, bb_lower = ta.BBANDS(df['close'], timeperiod=20)
        df["%bbwidth"] = bb_upper - bb_lower
        df["%volume_norm"] = (
        (df["volume"] - df["volume"].rolling(50).min())
        / (df["volume"].rolling(50).max() - df["volume"].rolling(50).min())
        )
        # Exemple de feature exotique :  # noqa: RUF003
        df["%-day_of_week"] = (df["date"].dt.dayofweek + 1) / 7
        df["%-hour_of_day"] = (df["date"].dt.hour + 1) / 25

        # DROPNA : obligatoire pour éviter le bug "0 feature(s)" de FreqAI  # noqa: RUF003
        features = [c for c in df.columns if c.startswith('%')]
        df = df.dropna(subset=features).reset_index(drop=True)
        return df

    @staticmethod
    def _ewo(dataframe: DataFrame, ema1_length: int, ema2_length: int) -> DataFrame:
        ema1 = ta.EMA(dataframe, timeperiod=ema1_length)
        ema2 = ta.EMA(dataframe, timeperiod=ema2_length)
        return (ema1 - ema2) / dataframe["low"] * 100

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        label_period = 10  # adapte si tu veux label_period_candles=10 dans config
        future = dataframe["close"].shift(-label_period)
        change = (future - dataframe["close"]) / dataframe["close"]
        dataframe["&-target"] = 0
        dataframe.loc[change > 0.005, "&-target"] = 1
        dataframe.loc[change < -0.005, "&-target"] = -1
        return dataframe



    # --------- TRAILING BUY/SELL INFOS ---------
    def trailing_buy(self, pair, reinit: bool = False):
        if pair not in self.custom_info_trail_buy:
            self.custom_info_trail_buy[pair] = {}
        if reinit or "trailing_buy" not in self.custom_info_trail_buy[pair]:
            self.custom_info_trail_buy[pair]["trailing_buy"] = self.init_trailing_dict.copy()
        return self.custom_info_trail_buy[pair]["trailing_buy"]

    def trailing_sell(self, pair, reinit: bool = False):
        if pair not in self.custom_info_trail_sell:
            self.custom_info_trail_sell[pair] = {}
        if reinit or "trailing_sell" not in self.custom_info_trail_sell[pair]:
            self.custom_info_trail_sell[pair]["trailing_sell"] = self.init_trailing_sell_dict.copy()
        return self.custom_info_trail_sell[pair]["trailing_sell"]

    def trailing_buy_info(self, pair: str, current_price: float):
        if not self.debug_mode:
            return
        t = self.trailing_buy(pair)
        duration = datetime.now(UTC) - (t["start_trailing_time"] or datetime.now(UTC))
        logger.info(f"[LONG] {pair} start {t['start_trailing_price']:.4f}, duration {duration}, "
                    f"uplimit {t['trailing_buy_order_uplimit']:.4f}, profit {(t['start_trailing_price'] - current_price)/t['start_trailing_price']*100:.2f}%, offset {t['offset']}")  # noqa: E501

    def trailing_sell_info(self, pair: str, current_price: float):
        if not self.debug_mode:
            return
        t = self.trailing_sell(pair)
        duration = datetime.now(UTC) - (t["start_trailing_time"] or datetime.now(UTC))
        logger.info(f"[SHORT] {pair} start {t['start_trailing_price']:.4f}, duration {duration}, "
                    f"downlimit {t['trailing_sell_order_downlimit']:.4f}, profit {(current_price - t['start_trailing_price'])/t['start_trailing_price']*100:.2f}%, offset {t['offset']}")  # noqa: E501

    def current_trailing_profit_ratio(self, pair: str, current_price: float) -> float:
        trailing_buy = self.trailing_buy(pair)
        if trailing_buy["trailing_buy_order_started"]:
            return (trailing_buy["start_trailing_price"] - current_price) / trailing_buy["start_trailing_price"]  # noqa: E501
        return 0

    def current_trailing_profit_ratio_short(self, pair: str, current_price: float) -> float:
        trailing_sell = self.trailing_sell(pair)
        if trailing_sell["trailing_sell_order_started"]:
            return (current_price - trailing_sell["start_trailing_price"]) / trailing_sell["start_trailing_price"]  # noqa: E501
        return 0

    def trailing_buy_offset(self, dataframe: DataFrame, pair: str, current_price: float):
        current_trailing_profit_ratio = self.current_trailing_profit_ratio(pair, current_price)
        last_candle = dataframe.iloc[-1]
        adapt = abs(last_candle["perc_norm"])
        default_offset = 0.004 * (1 + adapt)
        trailing_buy = self.trailing_buy(pair)
        if not trailing_buy["trailing_buy_order_started"]:
            return default_offset
        current_time = datetime.now(UTC)
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

    def trailing_sell_offset(self, dataframe: DataFrame, pair: str, current_price: float):
        ratio = self.current_trailing_profit_ratio_short(pair, current_price)
        last_candle = dataframe.iloc[-1]
        adapt = abs(last_candle["perc_norm"])
        default_offset = 0.004 * (1 + adapt)
        trailing_sell = self.trailing_sell(pair)
        if not trailing_sell["trailing_sell_order_started"]:
            return default_offset
        current_time = datetime.now(UTC)
        trailing_duration = current_time - trailing_sell["start_trailing_time"]
        if trailing_duration.total_seconds() > self.trailing_sell_expire_seconds:
            if ratio < 0 and last_candle["sell"] == 1:
                return "forcesell"
            return None
        elif (
            self.trailing_sell_downtrend_enabled
            and trailing_duration.total_seconds() < self.trailing_expire_seconds_downtrend
            and ratio > self.min_downtrend_trailing_profit
        ):
            return "forcesell"
        if ratio > 0:
            return default_offset
        trailing_sell_offset = {-0.06: 0.02, -0.03: 0.01, 0: default_offset}
        for key in sorted(trailing_sell_offset.keys()):
            if ratio < key:
                return trailing_sell_offset[key]
        return default_offset

    # --------- FreqAI SIGNALS ---------
    def freqai_select_signals(self, dataframe: DataFrame) -> DataFrame:
        dataframe["buy"] = 0; dataframe["buy_tag"] = ""  # noqa: E702
        dataframe["sell"] = 0; dataframe["sell_tag"] = ""  # noqa: E702
        dataframe["enter_long"] = 0; dataframe["enter_short"] = 0  # noqa: E702

        dataframe.loc[dataframe["freqai_prediction"] > 0.55, ["buy", "buy_tag", "enter_long"]] = [1, "freqai-long", 1]  # noqa: E501
        dataframe.loc[dataframe["freqai_prediction"] < 0.45, ["sell", "sell_tag", "enter_short"]] = [1, "freqai-short", 1]  # noqa: E501
        return dataframe

    # --------- INDICATORS POPULATION ---------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if hasattr(self, "freqai"):
            dataframe = self.freqai.start(dataframe, metadata, self)
    # tes trailing infos ou protections ici…
        return dataframe



    # --------- ENTRY TRENDS (long + short) ---------
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Utilisation des signaux générés par FreqAI
        if "freqai_prediction" in dataframe.columns:
            dataframe = self.freqai_select_signals(dataframe)
        pair = metadata["pair"]
        last = dataframe.iloc[-1]

        # Trailing BUY (long)
        if self.trailing_buy_order_enabled and self.config["runmode"].value in ("live", "dry_run"):
            t = self.trailing_buy(pair)
            if last["buy"] == 1 and not t["trailing_buy_order_started"]:
                open_trades = Trade.get_trades([Trade.pair == pair, Trade.is_open.is_(True)]).all()
                if not open_trades:
                    logger.info(f"Start trailing BUY for {pair}")
                    t["allow_trailing"] = True
                    dataframe.loc[:, "buy_tag"] = f"{last['buy_tag']} (start trail {last['close']})"
            elif t["trailing_buy_order_started"]:
                logger.info(f"Continuing trailing BUY for {pair}")
                dataframe.loc[:, "buy"] = 1; dataframe.loc[:, "buy_tag"] = t["buy_tag"]  # noqa: E702

        # Trailing SELL (short)
        if self.trailing_sell_order_enabled and self.config["runmode"].value in ("live", "dry_run"):
            t2 = self.trailing_sell(pair)
            if last["sell"] == 1 and not t2["trailing_sell_order_started"]:
                open_trades = Trade.get_trades([Trade.pair == pair, Trade.is_open.is_(True)]).all()
                if not open_trades:
                    logger.info(f"Start trailing SELL for {pair}")
                    t2["allow_trailing"] = True
                    dataframe.loc[:, "sell_tag"] = f"{last['sell_tag']} (start trail {last['close']})"  # noqa: E501
            elif t2["trailing_sell_order_started"]:
                logger.info(f"Continuing trailing SELL for {pair}")
                dataframe.loc[:, "sell"] = 1; dataframe.loc[:, "sell_tag"] = t2["sell_tag"]  # noqa: E702

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    # --------- CONFIRM TRADE ENTRY (long + short trailing logic) ---------
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, **kwargs) -> bool:  # noqa: C901, E501
        val = super().confirm_trade_entry(pair, order_type, amount, rate, time_in_force, **kwargs)

        # ----- Trailing BUY -----
        if val and self.trailing_buy_order_enabled and self.config["runmode"].value in ("live", "dry_run"):  # noqa: E501
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
                        trailing_buy["start_trailing_time"] = datetime.now(UTC)
                        trailing_buy["offset"] = 0
                        self.trailing_buy_info(pair, current_price)
                        logger.info(f"start trailing buy for {pair} at {last_candle['close']}")
                    elif trailing_buy["trailing_buy_order_started"]:
                        if trailing_buy_offset == "forcebuy":
                            val = True
                            ratio = f"{self.current_trailing_profit_ratio(pair, current_price) * 100:.2f}"  # noqa: E501
                            self.trailing_buy_info(pair, current_price)
                            logger.info(f"price OK for {pair} ({ratio} %, {current_price}), order may not be triggered if all slots are full")  # noqa: E501
                        elif trailing_buy_offset is None:
                            self.trailing_buy(pair, reinit=True)
                            logger.info(f"STOP trailing buy for {pair} because \"trailing buy offset\" returned None")  # noqa: E501
                        elif current_price < trailing_buy["trailing_buy_order_uplimit"]:
                            old_uplimit = trailing_buy["trailing_buy_order_uplimit"]
                            self.custom_info_trail_buy[pair]["trailing_buy"]["trailing_buy_order_uplimit"] = min(  # noqa: E501
                                current_price * (1 + trailing_buy_offset),
                                self.custom_info_trail_buy[pair]["trailing_buy"]["trailing_buy_order_uplimit"],
                            )
                            self.custom_info_trail_buy[pair]["trailing_buy"]["offset"] = trailing_buy_offset  # noqa: E501
                            self.trailing_buy_info(pair, current_price)
                            logger.info(f"update trailing buy for {pair} at {old_uplimit} -> {self.custom_info_trail_buy[pair]['trailing_buy']['trailing_buy_order_uplimit']}")  # noqa: E501
                        elif current_price < trailing_buy["start_trailing_price"] * (1 + self.trailing_buy_max_buy):  # noqa: E501
                            val = True
                            ratio = f"{self.current_trailing_profit_ratio(pair, current_price) * 100:.2f}"  # noqa: E501
                            self.trailing_buy_info(pair, current_price)
                            logger.info(f"current price ({current_price}) > uplimit ({trailing_buy['trailing_buy_order_uplimit']}) and lower than starting price ({trailing_buy['start_trailing_price'] * (1 + self.trailing_buy_max_buy)}). OK for {pair} ({ratio} %), order may not be triggered if all slots are full")  # noqa: E501
                        elif current_price > trailing_buy["start_trailing_price"] * (1 + self.trailing_buy_max_stop):  # noqa: E501
                            self.trailing_buy(pair, reinit=True)
                            self.trailing_buy_info(pair, current_price)
                            logger.info(f"STOP trailing buy for {pair} because of the price is higher than starting price * {1 + self.trailing_buy_max_stop}")  # noqa: E501
                        else:
                            self.trailing_buy_info(pair, current_price)
                            logger.info(f"price too high for {pair} !")
                else:
                    logger.info(f"Wait for next buy signal for {pair}")
            if val:
                self.trailing_buy_info(pair, rate)
                self.trailing_buy(pair, reinit=True)
                logger.info(f"STOP trailing buy for {pair} because I buy it")

        # ----- Trailing SELL -----
        if val and self.trailing_sell_order_enabled and self.config["runmode"].value in ("live", "dry_run"):  # noqa: E501
            val = False
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if len(dataframe) >= 1:
                last_candle = dataframe.iloc[-1].squeeze()
                current_price = rate
                trailing_sell = self.trailing_sell(pair)
                trailing_sell_offset = self.trailing_sell_offset(dataframe, pair, current_price)
                if trailing_sell["allow_trailing"]:
                    if not trailing_sell["trailing_sell_order_started"] and last_candle["sell"] == 1:  # noqa: E501
                        trailing_sell["trailing_sell_order_started"] = True
                        trailing_sell["trailing_sell_order_downlimit"] = last_candle["close"]
                        trailing_sell["start_trailing_price"] = last_candle["close"]
                        trailing_sell["sell_tag"] = last_candle["sell_tag"]
                        trailing_sell["start_trailing_time"] = datetime.now(UTC)
                        trailing_sell["offset"] = 0
                        self.trailing_sell_info(pair, current_price)
                        logger.info(f"start trailing sell for {pair} at {last_candle['close']}")
                    elif trailing_sell["trailing_sell_order_started"]:
                        if trailing_sell_offset == "forcesell":
                            val = True
                            ratio = f"{self.current_trailing_profit_ratio_short(pair, current_price) * 100:.2f}"  # noqa: E501
                            self.trailing_sell_info(pair, current_price)
                            logger.info(f"price OK for {pair} ({ratio} %, {current_price}), short order may not be triggered if all slots are full")  # noqa: E501
                        elif trailing_sell_offset is None:
                            self.trailing_sell(pair, reinit=True)
                            logger.info(f"STOP trailing sell for {pair} because \"trailing sell offset\" returned None")  # noqa: E501
                        elif current_price > trailing_sell["trailing_sell_order_downlimit"]:
                            old_downlimit = trailing_sell["trailing_sell_order_downlimit"]
                            self.custom_info_trail_sell[pair]["trailing_sell"]["trailing_sell_order_downlimit"] = max(  # noqa: E501
                                current_price * (1 - trailing_sell_offset),
                                self.custom_info_trail_sell[pair]["trailing_sell"]["trailing_sell_order_downlimit"],
                            )
                            self.custom_info_trail_sell[pair]["trailing_sell"]["offset"] = trailing_sell_offset  # noqa: E501
                            self.trailing_sell_info(pair, current_price)
                            logger.info(f"update trailing sell for {pair} at {old_downlimit} -> {self.custom_info_trail_sell[pair]['trailing_sell']['trailing_sell_order_downlimit']}")  # noqa: E501
                        elif current_price > trailing_sell["start_trailing_price"] * (1 - self.trailing_sell_max_sell):  # noqa: E501
                            val = True
                            ratio = f"{self.current_trailing_profit_ratio_short(pair, current_price) * 100:.2f}"  # noqa: E501
                            self.trailing_sell_info(pair, current_price)
                            logger.info(f"current price ({current_price}) < downlimit ({trailing_sell['trailing_sell_order_downlimit']}) and higher than starting price ({trailing_sell['start_trailing_price'] * (1 - self.trailing_sell_max_sell)}). OK for {pair} ({ratio} %), order may not be triggered if all slots are full")  # noqa: E501
                        elif current_price < trailing_sell["start_trailing_price"] * (1 - self.trailing_sell_max_stop):  # noqa: E501
                            self.trailing_sell(pair, reinit=True)
                            self.trailing_sell_info(pair, current_price)
                            logger.info(f"STOP trailing sell for {pair} because of the price is lower than starting price * {1 - self.trailing_sell_max_stop}")  # noqa: E501
                        else:
                            self.trailing_sell_info(pair, current_price)
                            logger.info(f"price too low for {pair} !")
                else:
                    logger.info(f"Wait for next sell signal for {pair}")
            if val:
                self.trailing_sell_info(pair, rate)
                self.trailing_sell(pair, reinit=True)
                logger.info(f"STOP trailing sell for {pair} because I short it")

        return val

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:  # noqa: F811
        return dataframe

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float, rate: float, time_in_force: str, sell_reason: str, current_time: datetime, **kwargs) -> bool:  # noqa: E501
        logger.info(
            f"[EXIT] {pair} - Reason: {sell_reason}, Buy tag: {getattr(trade, 'buy_tag', None)}, Profit: {getattr(trade, 'close_profit', None)}"  # noqa: E501
        )
        trade.custom_exit_reason = f'{sell_reason} ({getattr(trade, "buy_tag", None)})'
        return True

# Helper indicator
def EWO(dataframe: DataFrame, ema_length: int = 5, ema2_length: int = 35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df["low"] * 100
    return emadif
