# Stock Market Price Prediction 📈

This project is a Streamlit-based web app for predicting stock prices using historical data and a Long Short-Term Memory (LSTM) model.

## Features

- Fetches 10 years of historical stock data using [Yahoo Finance](https://finance.yahoo.com/).
- Visualizes the closing price history.
- Trains an LSTM model to predict stock prices based on historical data.
- Displays predictions and the Root Mean Square Error (RMSE).
- Provides the next day's predicted stock price.

## Demo
![Stock_market_price_predictor - Google Chrome 19-01-2025 01_29_39 AM](https://github.com/user-attachments/assets/33bb40e1-8dc4-431a-a6ee-1e4bbd50a572)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repository.git
   cd your-repository
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

4. Open the app in your browser at [http://localhost:8501](http://localhost:8501).

## Usage

1. Enter a stock symbol (e.g., `AAPL` for Apple, `RELIANCE.NS` for Reliance Industries).
2. View historical stock data and a visualization of closing prices.
3. Train the LSTM model to make predictions.
4. View predictions compared to actual values.
5. Get the predicted price for the next day.

## Dependencies

- `streamlit`
- `yfinance`
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `tensorflow`

Install all dependencies using the command:
```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── Stock_market_price_predictor.py                 # Streamlit app code
├── Stock_Market_Price_Prediction.ipynb # Jupyter notebook with detailed steps
├── README.md              # Project description
```

## Example

1. **Input**: Stock symbol, e.g., `AAPL`.
2. **Output**: Predicted stock price for the next day.

## Screenshots

### Closing Price History
![Editing Stock-Market-Price-Predictor-using-ML_README md at main · biprojeetchaudhury_Stock-Market-Price-Predictor-using-ML - Google Chrome 19-01-2025 01_31_32 AM](https://github.com/user-attachments/assets/00c5ef29-c1dc-413b-ab65-7057079db976)


### Model Predictions & Future Prediction
![Editing Stock-Market-Price-Predictor-using-ML_README md at main · biprojeetchaudhury_Stock-Market-Price-Predictor-using-ML - Google Chrome 19-01-2025 01_32_36 AM](https://github.com/user-attachments/assets/548eb518-4c7b-49d7-aa67-d8ad3ccc3ed6)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue.

## License

This project is licensed under the [MIT License](LICENSE).
