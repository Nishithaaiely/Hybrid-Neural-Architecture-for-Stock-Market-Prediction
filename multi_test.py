import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objs as go
from eodhd import APIClient
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from utils import query_raven, build_raven_prompt
from stock_indicators import aroon_indicator, average_high_low, dmi, vortex_indicator, bull_bear_power

# Function to fetch historical data for multiple tickers
def get_historical_data(api_client, tickers, start_date, end_date, interval):
    """Fetch historical data for given tickers within a specified date range."""
    combined_data = pd.DataFrame()
    try:
        for ticker in tickers:
            response = api_client.get_historical_data(ticker, interval, start_date, end_date)
            df = pd.DataFrame(response)
            df['ticker'] = ticker
            combined_data = pd.concat([combined_data, df], ignore_index=True)
        return combined_data
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return pd.DataFrame()

def get_minute_data(api_client, tickers, start_date, end_date, interval):
    """Fetch historical data for given tickers within a specified date range at minute intervals."""
    combined_data = pd.DataFrame()
    try:
        for ticker in tickers:
            response = api_client.get_historical_data(ticker, interval, start_date, end_date)
            df = pd.DataFrame(response)
            df['ticker'] = ticker
            combined_data = pd.concat([combined_data, df], ignore_index=True)
        return combined_data
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return pd.DataFrame()

def GoldenCrossverSignal(data, data_point):
    data['20_SMA'] = data['close'].rolling(window=20, min_periods=1).mean()
    data['50_SMA'] = data['close'].rolling(window=50, min_periods=1).mean()
    data['Signal'] = 0
    data['Signal'] = np.where(data['20_SMA'] > data['50_SMA'], 1, 0)
    data['Position'] = data['Signal'].diff()
    
    df_pos = data.iloc[-data_point:][(data.iloc[-data_point:]['Position'] == 1) | (data['Position'] == -1)].copy()
    df_pos['Position'] = df_pos['Position'].apply(lambda x: 'Buy' if x == 1 else 'Sell')
    
    trace_close = go.Scatter(x=data.index, y=data['close'], mode='lines', name='Close Price', line=dict(color='black'))
    trace_20sma = go.Scatter(x=data.index, y=data['20_SMA'], mode='lines', name='20-day SMA', line=dict(color='blue'))
    trace_50sma = go.Scatter(x=data.index, y=data['50_SMA'], mode='lines', name='50-day SMA', line=dict(color='green'))
    trace_buy = go.Scatter(x=data.iloc[-data_point:][data.iloc[-data_point:]['Position'] == 1].index, 
                           y=data.iloc[-data_point:]['20_SMA'][data.iloc[-data_point:]['Position'] == 1], 
                           mode='markers', name='Buy', marker=dict(symbol='triangle-up', size=15, color='green'))
    trace_sell = go.Scatter(x=data.iloc[-data_point:][data.iloc[-data_point:]['Position'] == -1].index, 
                            y=data.iloc[-data_point:]['20_SMA'][data.iloc[-data_point:]['Position'] == -1], 
                            mode='markers', name='Sell', marker=dict(symbol='triangle-down', size=15, color='red'))
    
    layout = go.Layout(
        title="Golden Crossover Signal",
        xaxis=dict(title='Date'),
        yaxis=dict(title='Price in Rupees'),
        showlegend=True
    )
    
    fig = go.Figure(data=[trace_close, trace_20sma, trace_50sma, trace_buy, trace_sell], layout=layout)
    
    return fig, df_pos

# Main Streamlit app function
def main():
    st.title("Historical Data Fetcher and Signal Generator")

    # API key input
    api_key = st.text_input("Enter your API key", "67a9c28e974fc3.81787843")
    api_client = APIClient(api_key) if api_key else None

    if api_client:
        asset_type = st.selectbox("Select asset type", ["stock", "crypto"])
        tickers_input = st.text_input("Enter the ticker symbols (comma separated)")
        tickers = [ticker.strip() for ticker in tickers_input.split(',')]
        start_date = st.text_input("Enter the start date (YYYY-MM-DD)")
        end_date = st.text_input("Enter the end date (YYYY-MM-DD)")
        interval = st.selectbox("Select the data interval", ["1m", "5m", "1h", "d", "w"])

        if st.button("Fetch Data"):
            if interval in ["d", "w"]:
                historical_data = get_historical_data(api_client, tickers, start_date, end_date, interval)
            else:
                historical_data = get_minute_data(api_client, tickers, start_date, end_date, interval)

            if not historical_data.empty:
                st.session_state['historical_data'] = historical_data  # Save to session state
                st.write(historical_data)
            else:
                st.info("No data available for the specified range or tickers.")

        # Check if historical data is stored in session state
        if 'historical_data' in st.session_state:
            data = pd.DataFrame(st.session_state['historical_data'])
            customer_input = st.text_input("Enter your analysis request", "What is the highest closing price of this data?")
            symbol = data['ticker']
            interval = data['interval']
            open_prices = data['open']
            high_prices = data['high']
            low_prices = data['low']
            close_prices = data['close']
            adjusted_close_prices = data['adjusted_close']
            volume = data['volume']
            period = 14

            # Processing prompt for data analysis
            if st.button("Analyze Data"):
                user_query = f"""function for {customer_input} and pass data from available variables high_prices,low_prices,close_prices and period"""

                function_list = [aroon_indicator, average_high_low, dmi, vortex_indicator, bull_bear_power]
                raven_prompt = build_raven_prompt(function_list, user_query)
                raven_call = query_raven(raven_prompt)
                st.write(raven_call)
                surya2 = exec(raven_call)
                st.write(surya2)
                print(surya2)
                
            # Generate Golden Crossover Signal
            if st.button("Generate Golden Crossover Signal"):
                fig, df_pos = GoldenCrossverSignal(data, 300)
                st.plotly_chart(fig)
                st.subheader("Buy/Sell Signals")
                st.dataframe(df_pos[['close', 'Position']].reset_index())

if __name__ == "__main__":
    main()
