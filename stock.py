import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import plotly.graph_objs as go
from eodhd import APIClient
from utils import query_raven, build_raven_prompt
from stock_indicators import aroon_indicator, average_high_low, dmi, vortex_indicator, bull_bear_power
import mplfinance as mpf
from datetime import datetime, timedelta
import json
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class StockAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Market Analysis Suite")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.api_client = None
        self.historical_data = None
        
        # Create notebook for multiple screens
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create frames for different screens
        self.main_screen = ttk.Frame(self.notebook)
        self.portfolio_screen = ttk.Frame(self.notebook)
        self.prediction_screen = ttk.Frame(self.notebook)
        self.screener_screen = ttk.Frame(self.notebook)
        
        # Add screens to notebook
        self.notebook.add(self.main_screen, text="Market Analysis")
        self.notebook.add(self.portfolio_screen, text="Portfolio Tracker")
        self.notebook.add(self.prediction_screen, text="Price Prediction")
        self.notebook.add(self.screener_screen, text="Stock Screener")
        
        # Initialize all screens
        self._init_main_screen()
        self._init_portfolio_screen()
        self._init_prediction_screen()
        self._init_screener_screen()

    def _init_main_screen(self):
        """Initialize the main market analysis screen"""
        # Create main container
        main_container = ttk.Frame(self.main_screen)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create and pack the components
        self._create_header(main_container)
        self._create_input_section(main_container)
        self._create_chart_section(main_container)
        self._create_analysis_section(main_container)

    def _init_portfolio_screen(self):
        """Initialize the portfolio tracking screen"""
        # Portfolio Header
        header_frame = ttk.Frame(self.portfolio_screen)
        header_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(header_frame, text="Portfolio Tracker", 
                 font=("Helvetica", 20, "bold")).pack()
        
        # Add Stock Frame
        add_frame = ttk.LabelFrame(self.portfolio_screen, text="Add Stock", padding=10)
        add_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Stock Entry Fields
        ttk.Label(add_frame, text="Symbol:").grid(row=0, column=0, padx=5, pady=5)
        self.stock_symbol = ttk.Entry(add_frame)
        self.stock_symbol.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(add_frame, text="Quantity:").grid(row=0, column=2, padx=5, pady=5)
        self.stock_quantity = ttk.Entry(add_frame)
        self.stock_quantity.grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Label(add_frame, text="Buy Price:").grid(row=0, column=4, padx=5, pady=5)
        self.stock_price = ttk.Entry(add_frame)
        self.stock_price.grid(row=0, column=5, padx=5, pady=5)
        
        ttk.Button(add_frame, text="Add to Portfolio", 
                  command=self._add_to_portfolio).grid(row=0, column=6, padx=5, pady=5)
        
        # Portfolio Table
        table_frame = ttk.Frame(self.portfolio_screen)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        columns = ('Symbol', 'Quantity', 'Buy Price', 'Current Price', 'P/L', 'P/L %')
        self.portfolio_table = ttk.Treeview(table_frame, columns=columns, show='headings')
        
        # Set column headings
        for col in columns:
            self.portfolio_table.heading(col, text=col)
            self.portfolio_table.column(col, width=100)
        
        self.portfolio_table.pack(fill=tk.BOTH, expand=True)
        
        # Summary Frame
        summary_frame = ttk.LabelFrame(self.portfolio_screen, text="Portfolio Summary", padding=10)
        summary_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.total_value_label = ttk.Label(summary_frame, text="Total Value: $0")
        self.total_value_label.pack(side=tk.LEFT, padx=10)
        
        self.total_pl_label = ttk.Label(summary_frame, text="Total P/L: $0")
        self.total_pl_label.pack(side=tk.LEFT, padx=10)

    def _init_prediction_screen(self):
        """Initialize the price prediction screen"""
        # Prediction Header
        header_frame = ttk.Frame(self.prediction_screen)
        header_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(header_frame, text="Stock Price Prediction", 
                 font=("Helvetica", 20, "bold")).pack()
        
        # Input Frame
        input_frame = ttk.LabelFrame(self.prediction_screen, text="Prediction Parameters", padding=10)
        input_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(input_frame, text="Symbol:").grid(row=0, column=0, padx=5, pady=5)
        self.pred_symbol = ttk.Entry(input_frame)
        self.pred_symbol.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="Prediction Days:").grid(row=0, column=2, padx=5, pady=5)
        self.pred_days = ttk.Entry(input_frame)
        self.pred_days.grid(row=0, column=3, padx=5, pady=5)
        self.pred_days.insert(0, "30")
        
        ttk.Button(input_frame, text="Generate Prediction", 
                  command=self._generate_prediction).grid(row=0, column=4, padx=5, pady=5)
        
        # Prediction Chart Frame
        self.pred_chart_frame = ttk.LabelFrame(self.prediction_screen, text="Prediction Chart", padding=10)
        self.pred_chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create a canvas for matplotlib
        self.pred_fig = Figure(figsize=(12, 6))
        self.pred_canvas = FigureCanvasTkAgg(self.pred_fig, master=self.pred_chart_frame)
        self.pred_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _init_screener_screen(self):
        """Initialize the stock screener screen with improved layout"""
        # Screener Header
        header_frame = ttk.Frame(self.screener_screen)
        header_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(header_frame, text="Stock Screener", 
                font=("Helvetica", 20, "bold")).pack()
        
        # Criteria Frame
        criteria_frame = ttk.LabelFrame(self.screener_screen, text="Screening Criteria", padding=10)
        criteria_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Market Cap
        ttk.Label(criteria_frame, text="Market Cap (USD):").grid(row=0, column=0, padx=5, pady=5)
        self.min_market_cap = ttk.Entry(criteria_frame, width=15)
        self.min_market_cap.grid(row=0, column=1, padx=5, pady=5)
        self.min_market_cap.insert(0, "0")
        ttk.Label(criteria_frame, text="to").grid(row=0, column=2, padx=5, pady=5)
        self.max_market_cap = ttk.Entry(criteria_frame, width=15)
        self.max_market_cap.grid(row=0, column=3, padx=5, pady=5)
        
        # P/E Ratio
        ttk.Label(criteria_frame, text="P/E Ratio:").grid(row=1, column=0, padx=5, pady=5)
        self.min_pe = ttk.Entry(criteria_frame, width=15)
        self.min_pe.grid(row=1, column=1, padx=5, pady=5)
        self.min_pe.insert(0, "0")
        ttk.Label(criteria_frame, text="to").grid(row=1, column=2, padx=5, pady=5)
        self.max_pe = ttk.Entry(criteria_frame, width=15)
        self.max_pe.grid(row=1, column=3, padx=5, pady=5)
        
        # Price Range
        ttk.Label(criteria_frame, text="Price (USD):").grid(row=2, column=0, padx=5, pady=5)
        self.min_price = ttk.Entry(criteria_frame, width=15)
        self.min_price.grid(row=2, column=1, padx=5, pady=5)
        self.min_price.insert(0, "0")
        ttk.Label(criteria_frame, text="to").grid(row=2, column=2, padx=5, pady=5)
        self.max_price = ttk.Entry(criteria_frame, width=15)
        self.max_price.grid(row=2, column=3, padx=5, pady=5)
        
        # Run button with better styling
        run_button = ttk.Button(criteria_frame, text="Run Screener", 
                            command=self._run_screener,
                            style="Accent.TButton")
        run_button.grid(row=3, column=0, columnspan=4, pady=10)
        
        # Results Table
        table_frame = ttk.Frame(self.screener_screen)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Add scrollbars
        y_scrollbar = ttk.Scrollbar(table_frame)
        y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        x_scrollbar = ttk.Scrollbar(table_frame, orient='horizontal')
        x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Create table with scrollbars
        columns = ('Symbol', 'Company', 'Market Cap', 'P/E Ratio', 'Price', 'Volume')
        self.screener_table = ttk.Treeview(table_frame, columns=columns, show='headings',
                                        yscrollcommand=y_scrollbar.set,
                                        xscrollcommand=x_scrollbar.set)
        
        # Configure scrollbars
        y_scrollbar.config(command=self.screener_table.yview)
        x_scrollbar.config(command=self.screener_table.xview)
        
        # Set column headings and widths
        column_widths = {
            'Symbol': 100,
            'Company': 200,
            'Market Cap': 150,
            'P/E Ratio': 100,
            'Price': 100,
            'Volume': 120
        }
        
        for col in columns:
            self.screener_table.heading(col, text=col)
            self.screener_table.column(col, width=column_widths.get(col, 100))
        
        self.screener_table.pack(fill=tk.BOTH, expand=True)

    def _add_to_portfolio(self):
        """Add a stock to the portfolio with improved error handling"""
        try:
            symbol = self.stock_symbol.get().upper()
            quantity = float(self.stock_quantity.get())
            buy_price = float(self.stock_price.get())
            
            if not symbol or not quantity or not buy_price:
                raise ValueError("All fields must be filled")
                
            # Get current price using yfinance with better error handling
            try:
                stock = yf.Ticker(symbol)
                stock_info = stock.info
                
                # Try multiple price fields in case some are missing
                current_price = None
                price_fields = ['regularMarketPrice', 'currentPrice', 'lastPrice']
                
                for field in price_fields:
                    if field in stock_info and stock_info[field] is not None:
                        current_price = stock_info[field]
                        break
                        
                if current_price is None:
                    # Fallback to last available price from history
                    hist = stock.history(period="1d")
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                    else:
                        raise ValueError(f"Could not retrieve current price for {symbol}")
                
                # Calculate P/L
                pl = (current_price - buy_price) * quantity
                pl_percent = (pl / (buy_price * quantity)) * 100
                
                # Add to table with formatted values
                self.portfolio_table.insert('', 'end', values=(
                    symbol,
                    quantity,  # Store as number, not formatted string
                    buy_price,  # Store as number, not formatted string
                    current_price,  # Store as number, not formatted string
                    pl,  # Store as number, not formatted string
                    f"{pl_percent:.2f}"  # Only format the percentage
                ))
                
                # Clear entries
                self.stock_symbol.delete(0, tk.END)
                self.stock_quantity.delete(0, tk.END)
                self.stock_price.delete(0, tk.END)
                
                # Update summary
                self._update_portfolio_summary()
                
                messagebox.showinfo("Success", f"Added {symbol} to portfolio successfully!")
                
            except yf.exceptions.YFinanceError as ye:
                messagebox.showerror("Error", f"Failed to fetch data for {symbol}: {str(ye)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process stock data: {str(e)}")
                
        except ValueError as ve:
            messagebox.showerror("Error", str(ve))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add stock: {str(e)}")


    def _update_portfolio_summary(self):
        """Update the portfolio summary labels with improved error handling"""
        try:
            total_value = 0
            total_pl = 0
            
            for item in self.portfolio_table.get_children():
                values = self.portfolio_table.item(item)['values']
                
                # Values are now stored as numbers, no need for string conversion
                current_price = float(values[3])  # Current price is at index 3
                quantity = float(values[1])       # Quantity is at index 1
                pl = float(values[4])            # P/L is at index 4
                
                current_value = current_price * quantity
                total_value += current_value
                total_pl += pl
            
            # Update labels with formatted values
            self.total_value_label.config(text=f"Total Value: ${total_value:,.2f}")
            self.total_pl_label.config(text=f"Total P/L: ${total_pl:,.2f}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update portfolio summary: {str(e)}")

    def _format_portfolio_display(self):
        """Update the display of the portfolio table with formatted values"""
        for item in self.portfolio_table.get_children():
            values = self.portfolio_table.item(item)['values']
            
            # Format the values for display
            formatted_values = (
                values[0],  # Symbol stays as is
                f"{float(values[1]):,.0f}",  # Format quantity
                f"${float(values[2]):,.2f}",  # Format buy price
                f"${float(values[3]):,.2f}",  # Format current price
                f"${float(values[4]):,.2f}",  # Format P/L
                values[5]  # Percentage stays as is
            )
            
            # Update the item with formatted values
            self.portfolio_table.item(item, values=formatted_values)

    def _generate_prediction(self):
        """Generate price prediction using LSTM"""
        try:
            symbol = self.pred_symbol.get().upper()
            days = int(self.pred_days.get())
            
            # Get historical data
            stock = yf.Ticker(symbol)
            data = stock.history(period='1y')
            
            # Prepare data for LSTM
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
            
            # Create sequences
            sequence_length = 60
            sequences = []
            for i in range(len(scaled_data) - sequence_length):
                sequences.append(scaled_data[i:(i + sequence_length)])
            
            # Split into training and testing sets
            train_size = int(len(sequences) * 0.8)
            train_sequences = np.array(sequences[:train_size])
            test_sequences = np.array(sequences[train_size:])
            
            # Create and train LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mse')
            model.fit(train_sequences, train_sequences[:, -1, :], epochs=50, batch_size=32, verbose=0)
            
            # Generate predictions
            last_sequence = scaled_data[-sequence_length:]
            predictions = []
            
            for _ in range(days):
                next_pred = model.predict(last_sequence.reshape(1, sequence_length, 1))
                predictions.append(next_pred[0, 0])
                last_sequence = np.roll(last_sequence, -1)
                last_sequence[-1] = next_pred
            
            # Inverse transform predictions
            predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
            
            # Plot results
            self.pred_fig.clear()
            ax = self.pred_fig.add_subplot(111)
            
            # Plot historical data
            ax.plot(data.index[-60:], data['Close'].values[-60:], 
                   label='Historical Data', color='blue')
            
            # Plot predictions
            future_dates = pd.date_range(
                start=data.index[-1], 
                periods=days+1, 
                freq='D'
            )[1:]
            ax.plot(future_dates, predictions, 
                   label='Predictions', color='red', linestyle='--')
            
            ax.set_title(f'{symbol} Price Prediction')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            ax.grid(True)
            
            self.pred_canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate prediction: {str(e)}")

    def _run_screener(self):
        """Run the stock screener with specified criteria and improved error handling"""
        try:
            # Get criteria values with validation
            try:
                min_market_cap = float(self.min_market_cap.get() or 0)
                max_market_cap = float(self.max_market_cap.get() or float('inf'))
                min_pe = float(self.min_pe.get() or 0)
                max_pe = float(self.max_pe.get() or float('inf'))
                min_price = float(self.min_price.get() or 0)
                max_price = float(self.max_price.get() or float('inf'))
            except ValueError as ve:
                messagebox.showerror("Input Error", "Please enter valid numbers for criteria")
                return

            # Clear existing results
            for item in self.screener_table.get_children():
                self.screener_table.delete(item)

            # Show progress
            progress_label = ttk.Label(self.screener_screen, text="Screening stocks... Please wait.")
            progress_label.pack(pady=5)
            self.screener_screen.update()

            try:
                # Get S&P 500 symbols
                sp500_url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
                sp500_stocks = pd.read_csv(sp500_url)
                total_stocks = len(sp500_stocks)
                stocks_processed = 0
                stocks_found = 0

                for symbol in sp500_stocks['Symbol']:
                    try:
                        # Update progress
                        stocks_processed += 1
                        progress_label.config(text=f"Processing: {stocks_processed}/{total_stocks} stocks... Found: {stocks_found}")
                        self.screener_screen.update()

                        # Get stock info
                        stock = yf.Ticker(symbol)
                        info = stock.info

                        # Get required values with proper fallbacks
                        market_cap = info.get('marketCap', 0)
                        if market_cap == 0:
                            market_cap = info.get('totalMarketCap', 0)

                        pe_ratio = info.get('forwardPE', 0)
                        if pe_ratio == 0:
                            pe_ratio = info.get('trailingPE', 0)

                        price = info.get('regularMarketPrice', 0)
                        if price == 0:
                            price = info.get('currentPrice', 0)

                        volume = info.get('regularMarketVolume', 0)
                        if volume == 0:
                            volume = info.get('volume', 0)

                        company_name = info.get('shortName', info.get('longName', 'N/A'))

                        # Debug print
                        print(f"Processing {symbol}: Cap={market_cap}, PE={pe_ratio}, Price={price}")

                        # Apply screening criteria with validation
                        if (market_cap > 0 and pe_ratio > 0 and price > 0 and
                            min_market_cap <= market_cap <= max_market_cap and
                            min_pe <= pe_ratio <= max_pe and
                            min_price <= price <= max_price):

                            stocks_found += 1
                            
                            # Format values for display
                            formatted_values = (
                                symbol,
                                company_name,
                                f"${market_cap:,.0f}" if market_cap else "N/A",
                                f"{pe_ratio:.2f}" if pe_ratio else "N/A",
                                f"${price:.2f}" if price else "N/A",
                                f"{volume:,}" if volume else "N/A"
                            )

                            self.screener_table.insert('', 'end', values=formatted_values)

                    except Exception as stock_error:
                        print(f"Error processing {symbol}: {str(stock_error)}")
                        continue

                # Remove progress label
                progress_label.destroy()

                if stocks_found == 0:
                    messagebox.showinfo("Results", "No stocks matched your criteria. Try adjusting the parameters.")
                else:
                    messagebox.showinfo("Success", f"Screening completed! Found {stocks_found} matching stocks.")

            except Exception as data_error:
                messagebox.showerror("Data Error", f"Failed to fetch stock data: {str(data_error)}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to run screener: {str(e)}")
            if 'progress_label' in locals():
                progress_label.destroy()

    def _connect_api(self):
        """Connect to the EODHD API"""
        api_key = self.api_key_entry.get()
        try:
            self.api_client = APIClient(api_key)
            messagebox.showinfo("Success", "API connected successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to connect to API: {str(e)}")

    def _create_header(self, container):
        """Create the header section with title and API input"""
        header_frame = ttk.Frame(container)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = ttk.Label(
            header_frame, 
            text="Market Analysis", 
            font=("Helvetica", 24, "bold")
        )
        title_label.pack(pady=10)
        
        # API Key input
        api_frame = ttk.Frame(header_frame)
        api_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(api_frame, text="API Key:").pack(side=tk.LEFT, padx=5)
        self.api_key_entry = ttk.Entry(api_frame, width=40)
        self.api_key_entry.pack(side=tk.LEFT, padx=5)
        self.api_key_entry.insert(0, "67ac2201be0840.46891599")  # Default API key
        
        ttk.Button(
            api_frame, 
            text="Connect", 
            command=self._connect_api,
            style="Accent.TButton"
        ).pack(side=tk.LEFT, padx=5)

    def _fetch_data(self):
        """Fetch historical data based on user inputs"""
        if not self.api_client:
            messagebox.showerror("Error", "Please connect to API first")
            return
            
        try:
            tickers = [t.strip() for t in self.tickers_entry.get().split(',')]
            start_date = self.start_date_entry.get()
            end_date = self.end_date_entry.get()
            interval = self.interval.get()
            
            if interval in ["d", "w"]:
                self.historical_data = self._get_historical_data(
                    tickers, start_date, end_date, interval
                )
            else:
                self.historical_data = self._get_minute_data(
                    tickers, start_date, end_date, interval
                )
                
            if not self.historical_data.empty:
                self._plot_data()
                messagebox.showinfo("Success", "Data fetched successfully!")
            else:
                messagebox.showwarning("Warning", "No data available for the specified range")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to fetch data: {str(e)}")

    def _get_historical_data(self, tickers, start_date, end_date, interval):
        """Fetch historical data for daily/weekly intervals"""
        combined_data = pd.DataFrame()
        for ticker in tickers:
            response = self.api_client.get_historical_data(
                ticker, interval, start_date, end_date
            )
            df = pd.DataFrame(response)
            df['ticker'] = ticker
            combined_data = pd.concat([combined_data, df], ignore_index=True)
        return combined_data

    def _get_minute_data(self, tickers, start_date, end_date, interval):
        """Fetch historical data for minute intervals"""
        combined_data = pd.DataFrame()
        for ticker in tickers:
            response = self.api_client.get_historical_data(
                ticker, interval, start_date, end_date
            )
            df = pd.DataFrame(response)
            df['ticker'] = ticker
            combined_data = pd.concat([combined_data, df], ignore_index=True)
        return combined_data

    def _plot_data(self):
        """Plot the fetched data using matplotlib"""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        data = self.historical_data
        ax.plot(data.index, data['close'], label='Close Price')
        ax.set_title('Stock Price History')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True)
        
        self.canvas.draw()

    def _create_input_section(self, container):
        """Create the input section for stock data parameters"""
        input_frame = ttk.LabelFrame(container, text="Data Parameters", padding=10)
        input_frame.pack(fill=tk.X, pady=10)
        
        # Create a grid layout
        # Row 1
        ttk.Label(input_frame, text="Asset Type:").grid(row=0, column=0, padx=5, pady=5)
        self.asset_type = ttk.Combobox(input_frame, values=["stock", "crypto"])
        self.asset_type.set("stock")
        self.asset_type.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="Tickers:").grid(row=0, column=2, padx=5, pady=5)
        self.tickers_entry = ttk.Entry(input_frame)
        self.tickers_entry.grid(row=0, column=3, padx=5, pady=5)
        
        # Row 2
        ttk.Label(input_frame, text="Start Date:").grid(row=1, column=0, padx=5, pady=5)
        self.start_date_entry = ttk.Entry(input_frame)
        self.start_date_entry.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="End Date:").grid(row=1, column=2, padx=5, pady=5)
        self.end_date_entry = ttk.Entry(input_frame)
        self.end_date_entry.grid(row=1, column=3, padx=5, pady=5)
        
        # Row 3
        ttk.Label(input_frame, text="Interval:").grid(row=2, column=0, padx=5, pady=5)
        self.interval = ttk.Combobox(input_frame, values=["1m", "5m", "1h", "d", "w"])
        self.interval.set("d")
        self.interval.grid(row=2, column=1, padx=5, pady=5)
        
        # Fetch button
        ttk.Button(
            input_frame, 
            text="Fetch Data",
            command=self._fetch_data,
            style="Accent.TButton"
        ).grid(row=2, column=2, columnspan=2, pady=10)

    def _create_chart_section(self, container):
        """Create the chart section for displaying visualizations"""
        self.chart_frame = ttk.LabelFrame(container, text="Charts", padding=10)
        self.chart_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create a canvas for matplotlib
        self.fig = Figure(figsize=(12, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _analyze_data(self):
        """Analyze data based on user query"""
        if self.historical_data is None:
            messagebox.showerror("Error", "Please fetch data first")
            return
            
        try:
            user_query = self.query_entry.get()
            data = self.historical_data
            
            function_list = [
                aroon_indicator, average_high_low, dmi, 
                vortex_indicator, bull_bear_power
            ]
            
            raven_prompt = build_raven_prompt(
                function_list,
                f"function for {user_query} and pass data from available variables"
            )
            
            raven_call = query_raven(raven_prompt)
            
            # Execute the analysis
            high_prices = data['high']
            low_prices = data['low']
            close_prices = data['close']
            period = 14
            
            # Update results text
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Analyzing: {user_query}\n\n")
            
            try:
                result = eval(raven_call)
                self.results_text.insert(tk.END, str(result))
            except Exception as e:
                self.results_text.insert(tk.END, f"Analysis error: {str(e)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")

    def _generate_golden_crossover(self):
        """Generate and display Golden Crossover signals"""
        if self.historical_data is None:
            messagebox.showerror("Error", "Please fetch data first")
            return
            
        try:
            data = self.historical_data.copy()
            
            # Calculate moving averages
            data['20_SMA'] = data['close'].rolling(window=20, min_periods=1).mean()
            data['50_SMA'] = data['close'].rolling(window=50, min_periods=1).mean()
            data['Signal'] = 0
            data['Signal'] = np.where(data['20_SMA'] > data['50_SMA'], 1, 0)
            data['Position'] = data['Signal'].diff()
            
            # Clear previous plot
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            
            # Plot price and moving averages
            ax.plot(data.index, data['close'], label='Close Price', color='black')
            ax.plot(data.index, data['20_SMA'], label='20-day SMA', color='blue')
            ax.plot(data.index, data['50_SMA'], label='50-day SMA', color='red')
            
            # Add buy signals
            buy_signals = data[data['Position'] == 1]
            if not buy_signals.empty:
                ax.scatter(buy_signals.index, buy_signals['close'], 
                         color='green', marker='^', s=100, label='Buy Signal')
            
            # Add sell signals
            sell_signals = data[data['Position'] == -1]
            if not sell_signals.empty:
                ax.scatter(sell_signals.index, sell_signals['close'], 
                         color='red', marker='v', s=100, label='Sell Signal')
            
            ax.set_title('Golden Crossover Analysis')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            ax.grid(True)
            
            self.canvas.draw()
            
            # Update results text with signal summary
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Golden Crossover Analysis:\n\n")
            self.results_text.insert(tk.END, f"Total Buy Signals: {len(buy_signals)}\n")
            self.results_text.insert(tk.END, f"Total Sell Signals: {len(sell_signals)}\n")
            
            if not buy_signals.empty:
                self.results_text.insert(tk.END, "\nLatest Buy Signal:\n")
                self.results_text.insert(tk.END, f"Date: {buy_signals.index[-1]}\n")
                self.results_text.insert(tk.END, f"Price: ${buy_signals['close'].iloc[-1]:.2f}\n")
            
            if not sell_signals.empty:
                self.results_text.insert(tk.END, "\nLatest Sell Signal:\n")
                self.results_text.insert(tk.END, f"Date: {sell_signals.index[-1]}\n")
                self.results_text.insert(tk.END, f"Price: ${sell_signals['close'].iloc[-1]:.2f}\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate Golden Crossover: {str(e)}")

    def _create_analysis_section(self, container):
        """Create the analysis section for user queries and results"""
        analysis_frame = ttk.LabelFrame(container, text="Analysis", padding=10)
        analysis_frame.pack(fill=tk.X, pady=10)
        
        # Query input
        ttk.Label(analysis_frame, text="Analysis Query:").pack(anchor=tk.W)
        self.query_entry = ttk.Entry(analysis_frame, width=60)
        self.query_entry.pack(fill=tk.X, pady=5)
        
        # Buttons frame
        button_frame = ttk.Frame(analysis_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(
            button_frame,
            text="Analyze Data",
            command=self._analyze_data,
            style="Accent.TButton"
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Generate Golden Crossover",
            command=self._generate_golden_crossover,
            style="Accent.TButton"
        ).pack(side=tk.LEFT, padx=5)
        
        # Results text area
        self.results_text = scrolledtext.ScrolledText(analysis_frame, height=6)
        self.results_text.pack(fill=tk.X, pady=5)

if __name__ == "__main__":
    root = ttk.Window(themename="cosmo")
    app = StockAnalysisApp(root)
    root.mainloop()