from typing import Dict

import pandas_ta as pta
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.freqai.freqai_interface import FreqaiStrategy


class MiniLamboAI(FreqaiStrategy):
    """FreqAI classifier based on MiniLambo indicators.

    This strategy demonstrates how to reuse the MiniLambo indicator set
    with FreqAI models. Train and predict via::

        freqtrade train-predict --strategy MiniLamboAI --config config.json

    Ensure the ``freqai`` section of ``config.json`` enables FreqAI and sets
    ``target_column`` to ``"target"``.
    """

    INTERFACE_VERSION = 3
    timeframe = "1m"
    startup_candle_count = 200  # longest indicator period (EWO 200)

    minimal_roi: Dict[int, float] = {}
    stoploss = 0.0
    use_exit_signal = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Populate indicators (same as feature generation)."""
        return self.freqai_generate_features(dataframe)

    def freqai_generate_features(self, dataframe: DataFrame) -> DataFrame:
        """Generate features for the ML model."""
        dataframe["ema_14"] = ta.EMA(dataframe, timeperiod=14)
        dataframe["rsi_4"] = ta.RSI(dataframe, timeperiod=4)
        dataframe["rsi_14"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["rsi_100"] = ta.RSI(dataframe, timeperiod=100)
        dataframe["cti"] = pta.cti(dataframe["close"], length=20)
        dataframe["ewo"] = self._ewo(dataframe, 50, 200)
        return dataframe

    @staticmethod
    def _ewo(dataframe: DataFrame, ema1_length: int, ema2_length: int) -> DataFrame:
        ema1 = ta.EMA(dataframe, timeperiod=ema1_length)
        ema2 = ta.EMA(dataframe, timeperiod=ema2_length)
        return (ema1 - ema2) / dataframe["low"] * 100

    def set_freqai_targets(self, dataframe: DataFrame) -> DataFrame:
        """Create classification targets: price change > 0.5%% in next 10 candles."""
        future_close = dataframe["close"].shift(-10)
        change = (future_close - dataframe["close"]) / dataframe["close"]
        dataframe["target"] = (change > 0.005).astype(int)
        return dataframe

    def freqai_select_signals(self, dataframe: DataFrame) -> DataFrame:
        """Use model predictions to generate buy signals."""
        dataframe["buy"] = 0
        dataframe.loc[dataframe["freqai_prediction"] > 0.5, "buy"] = 1
        dataframe["sell"] = 0
        return dataframe
