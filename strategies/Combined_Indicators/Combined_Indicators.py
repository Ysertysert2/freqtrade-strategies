# --- Do not remove these libs ---
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
# --------------------------------
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame


def bollinger_bands(stock_price, window_size, num_of_std):
    rolling_mean = stock_price.rolling(window=window_size).mean()
    rolling_std = stock_price.rolling(window=window_size).std()
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return np.nan_to_num(rolling_mean), np.nan_to_num(lower_band)


class Combined_Indicators(IStrategy):
    INTERFACE_VERSION = 3
    # Based on a backtesting:
    # - the best perfomance is reached with "max_open_trades" = 2 (in average for any market),
    #   so it is better to increase "stake_amount" value rather then "max_open_trades" to get more profit
    # - if the market is constantly green(like in JAN 2018) the best performance is reached with
    #   "max_open_trades" = 2 and minimal_roi = 0.01
      # ROI table:
    # ROI table:
    # ROI table:
    minimal_roi = {
        "0": 0.015
    }

    # Stoploss:
    stoploss = -0.0658

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.0198
    trailing_stop_positive_offset = 0.03082
    trailing_only_offset_is_reached = True

    timeframe = '1m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # strategy BinHV45
        mid, lower = bollinger_bands(dataframe['close'], window_size=40, num_of_std=2)
        dataframe['lower'] = lower
        dataframe['bbdelta'] = (mid - dataframe['lower']).abs()
        dataframe['closedelta'] = (dataframe['close'] - dataframe['close'].shift()).abs()
        dataframe['tail'] = (dataframe['close'] - dataframe['low']).abs()
        # strategy ClucMay72018
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=30).mean()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (  # strategy BinHV45
                    dataframe['lower'].shift().gt(0) &
                    dataframe['bbdelta'].gt(dataframe['close'] * 0.00703) &
                    dataframe['closedelta'].gt(dataframe['close'] * 0.01476) &
                    dataframe['tail'].lt(dataframe['bbdelta'] * 0.03632) &
                    dataframe['close'].lt(dataframe['lower'].shift()) &
                    dataframe['close'].le(dataframe['close'].shift())
            ) |
            (  # strategy ClucMay72018
                    (dataframe['close'] < dataframe['ema_slow']) &
                    (dataframe['close'] < 0.98863 * dataframe['bb_lowerband']) &
                    (dataframe['volume'] < (dataframe['volume_mean_slow'].shift(1) * 34))
            ),
            'buy'
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        """
        dataframe.loc[
            (dataframe['close'] > dataframe['bb_middleband']),
            'sell'
        ] = 1

        return dataframe