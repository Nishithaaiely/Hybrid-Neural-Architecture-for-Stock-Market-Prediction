import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import streamlit as st

def aroon_indicator(high_prices, low_prices, period=14):
    '''
    Function:

    def aroon_indicator(high_prices, low_prices, period=14):
    Calculates the Aroon Indicator.
    Parameters:
    - high_prices (pd.Series): Series of high prices.
    - low_prices (pd.Series): Series of low prices.
    - period (int): The period over which to calculate the Aroon Indicator. Default is 14.

    Returns:
    - pd.DataFrame: DataFrame containing Aroon Up and Aroon Down values.
    
    Example usage:
    aroon_indicator(high_prices, low_prices, period=14)

    # This will calculate the Aroon Up and Aroon Down indicators for the given high and low prices over a period of 14 days.
    '''
    days_since_high = []
    days_since_low = []

    high_rolling_max = high_prices.rolling(period).max()
    low_rolling_min = low_prices.rolling(period).min()

    for i in range(len(high_prices)):
        where_high = np.where(high_rolling_max == high_prices[i])[0]
        if where_high.size > 0:
            days_since_high.append(where_high[-1] + 1)
        else:
            days_since_high.append(0)

        where_low = np.where(low_rolling_min == low_prices[i])[0]
        if where_low.size > 0:
            days_since_low.append(where_low[-1] + 1)
        else:
            days_since_low.append(0)

    aroon_up = ((period - np.array(days_since_high)) / period) * 100
    aroon_down = ((period - np.array(days_since_low)) / period) * 100

    aroon_df = pd.DataFrame({
        'Aroon Up': aroon_up,
        'Aroon Down': aroon_down
    }, index=np.arange(len(high_prices)))

    return aroon_df

# Add your other functions here

def average_high_low(high_prices, low_prices, period):
    '''
    Function:
    def average_high_low(high_prices, low_prices, period):
    Calculates the average high and low prices over a specified period and their difference.
    
    Parameters:
    - high_prices (pd.Series): Series of high prices.
    - low_prices (pd.Series): Series of low prices.
    - period (int): The period over which to calculate the averages.

    Returns:
    - None: Displays the DataFrame in Streamlit.
    
    Example usage:
    average_high_low(high_prices, low_prices, period=14)

    # This will calculate and display the average high and low prices and their difference over a period of 14 days.
    '''
    high_prices = pd.Series(high_prices)
    low_prices = pd.Series(low_prices)

    avg_high = high_prices.rolling(period).mean()
    avg_low = low_prices.rolling(period).mean()
    diff = avg_high - avg_low
    result = pd.DataFrame({
        'Average High': avg_high,
        'Average Low': avg_low,
        'Difference': diff
    })
    st.write(result)


def vortex_indicator(high_prices, low_prices, close_prices, period=14):
    '''
    Function:
    def vortex_indicator_plot(high_prices, low_prices, close_prices, period=14):
    Calculates and plots the Vortex Indicator (VI).
    
    Parameters:
    - high_prices (pd.Series): Series of high prices.
    - low_prices (pd.Series): Series of low prices.
    - close_prices (pd.Series): Series of close prices.
    - period (int): The period over which to calculate the Vortex Indicator. Default is 14.

    Returns:
    - None: Displays the plot in Streamlit.
    
    Example usage:
    vortex_indicator_plot(high_prices, low_prices, close_prices, period=14)

    # This will calculate and plot the Vortex Indicator VI+ and VI- for the given high, low, and close prices over a period of 14 days.
    '''
    # Calculate True Range
    true_range = np.maximum(high_prices - low_prices, np.maximum(high_prices - close_prices, low_prices - close_prices))

    # Calculate VM+ and VM-
    vm_plus = np.abs(high_prices - low_prices.rolling(period).min())
    vm_minus = np.abs(low_prices - high_prices.rolling(period).max())

    # Sum TR and VM
    sum_trn = true_range.rolling(period).sum()
    sum_vm_plus = vm_plus.rolling(period).sum()
    sum_vm_minus = vm_minus.rolling(period).sum()

    # Calculate VI+ and VI-
    vi_plus = sum_vm_plus / sum_trn
    vi_minus = sum_vm_minus / sum_trn

    # Create DataFrame with VI values
    result = pd.DataFrame({
        'VI+': vi_plus,
        'VI-': vi_minus
    })

    # Plotting
    plt.figure(figsize=(14, 7))
    plt.plot(result['VI+'], label='VI+', color='green')
    plt.plot(result['VI-'], label='VI-', color='red')
    plt.title('Vortex Indicator (VI)')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    st.dataframe(result)
    st.pyplot(plt)


def bull_bear_power(high_prices, low_prices, close_prices, period: int):
    '''
    Function:
    def bull_bear_power_plot(high_prices, low_prices, close_prices, period: int): 
    Calculates and plots the Bull and Bear Power indicators.
    
    Parameters:
    - high_prices : List of high prices.
    - low_prices : List of low prices.
    - close_prices : List of close prices.
    - period (int): The period over which to calculate the EMA.

    Returns:
    - None: Displays the plot in Streamlit.
    
    Example usage:
    bull_bear_power(high_prices, low_prices, close_prices, period=14)

    # This will calculate and plot the Bull Power, Bear Power, and Bull Bear Power indicators for the given high, low, and close prices over a period of 14 days.
    '''
    # Convert lists to Pandas Series
    high_prices_series = pd.Series(high_prices)
    low_prices_series = pd.Series(low_prices)
    close_prices_series = pd.Series(close_prices)
    
    # Calculate the Exponential Moving Average (EMA)
    ema = close_prices_series.ewm(span=period, adjust=False).mean()
    
    # Calculate Bull Power and Bear Power
    bull_power = high_prices_series - ema
    bear_power = ema - low_prices_series
    
    # Calculate Bull Bear Power
    bbp = bull_power + bear_power
    
    # Create a DataFrame to store the results
    result = pd.DataFrame({
        'Bull Power': bull_power,
        'Bear Power': bear_power,
        'Bull Bear Power': bbp
    })
    
    # Plotting
    plt.figure(figsize=(14, 7))
    plt.plot(result['Bull Power'], label='Bull Power', color='green')
    plt.plot(result['Bear Power'], label='Bear Power', color='red')
    plt.plot(result['Bull Bear Power'], label='Bull Bear Power', color='blue')
    plt.title('Bull and Bear Power Indicators')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    st.write(result)
    st.pyplot(plt)


def dmi(high_prices, low_prices, close_prices, period=14):
    ''' 
    Function:
    def dmi(high_prices, low_prices, close_prices, period=14): 
    Calculates and plots the Directional Movement Index (DMI) and Average Directional Index (ADX).
    
    Parameters:
        high_prices (pd.Series): Series of high prices.
        low_prices (pd.Series): Series of low prices.
        close_prices (pd.Series): Series of close prices.
        period (int): The period over which to calculate the DMI and ADX.

    Returns:
        None: Displays the plot in Streamlit.
    
    Example usage:
    dmi(high_prices, low_prices, close_prices, period=14)

    # This will calculate and plot the DI+, DI-, and ADX values.
    '''
    high_prices = pd.Series(high_prices)
    low_prices = pd.Series(low_prices)
    close_prices = pd.Series(close_prices)
    
    # Calculate the directional movements
    up_move = high_prices - high_prices.shift(1)
    down_move = low_prices.shift(1) - low_prices
    dm_plus = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=high_prices.index)
    dm_minus = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=low_prices.index)
    
    # Calculate the directional indicators
    di_plus = 100 * dm_plus.ewm(span=period, adjust=False).mean()
    di_minus = 100 * dm_minus.ewm(span=period, adjust=False).mean()
    
    # Calculate the ADX
    adx = 100 * np.abs(di_plus - di_minus).ewm(span=period, adjust=False).mean() / (di_plus + di_minus).ewm(span=period, adjust=False).mean()
    
    # Create a DataFrame to store the results
    result = pd.DataFrame({
        'DI+': di_plus,
        'DI-': di_minus,
        'ADX': adx
    })
    
    # Plotting
    plt.figure(figsize=(14, 7))
    plt.plot(result['DI+'], label='DI+', color='g')
    plt.plot(result['DI-'], label='DI-', color='r')
    plt.plot(result['ADX'], label='ADX', color='b')
    plt.title('Directional Movement Index (DMI) and Average Directional Index (ADX)')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    st.write(result)
    st.pyplot(plt)