    import os
    import sys
    from datetime import datetime, timedelta
    from dateutil.relativedelta import relativedelta
    import pickle
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    import torch
    import torch.nn as NN
    from torch.utils.data import DataLoader, TensorDataset

    from alpaca.data import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()


    # Function 1: Fetch data
    def fetch_stock_data(stock, start_date, end_date):
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        client = StockHistoricalDataClient(api_key, secret_key)
        request_params = StockBarsRequest(
            symbol_or_symbols=[stock],
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date
        )
        data = client.get_stock_bars(request_params).df
        return data[['close']]


    # Function 2: Preprocess data
    def preprocess_data(data):
        data.reset_index(inplace=True)
        data['close'] = pd.to_numeric(data['close'], errors='coerce')
        data.dropna(inplace=True)
        scaler = MinMaxScaler()
        scaled_data = pd.DataFrame(
            scaler.fit_transform(data[['close']]),
            columns=['close'],
            index=data.index
        )
        # Save the scaler as a pickle file
        with open('lstm_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        return scaled_data, scaler


    # Function 3: Create sequences
    def create_sequences(data, window_size):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data.iloc[i:i + window_size].values)
            y.append(data.iloc[i + window_size]['close'])
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


    # Function 4: Define the LSTM model
    class StockPriceLSTM(NN.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super(StockPriceLSTM, self).__init__()
            self.lstm = NN.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = NN.Linear(hidden_size, output_size)

        def forward(self, x):
            _, (h_n, _) = self.lstm(x)
            out = self.fc(h_n[-1])
            return out


    # Function 5: Train the model
    def train_model(model, train_loader, num_epochs, criterion, optimizer):
        for epoch in range(num_epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

            # Save the trained model as a pickle file
            with open('lstm_trained_model.pkl', 'wb') as f:
                pickle.dump(model, f)
        return model


    # Function 6: Evaluate the model
    def evaluate_model(model, X_test, y_test, scaler):
        model.eval()
        with torch.no_grad():
            predictions = model(X_test).squeeze().numpy()
            actual = y_test.numpy()
        predicted_prices = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        actual_prices = scaler.inverse_transform(actual.reshape(-1, 1)).flatten()
        rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
        mae = mean_absolute_error(actual_prices, predicted_prices)
        r2 = r2_score(actual_prices, predicted_prices)
        return predicted_prices, actual_prices, rmse, mae, r2


    # Function 7: Plot results
    def plot_results(time_index, actual_prices, predicted_prices, title="Stock Price Prediction"):
        plt.figure(figsize=(14, 8))
        plt.plot(time_index, actual_prices, label='Actual Prices', color='blue', linewidth=2)
        plt.plot(time_index, predicted_prices, label='Predicted Prices', color='orange', linestyle='--', linewidth=2)
        plt.title(title, fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Close Price (USD)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

    # Function 8: Perform inference
    def make_inference(model, new_data, window_size, scaler):
        """
        Perform inference using the trained model on new data.
        """
        model.eval()
        with torch.no_grad():
            # Scale the new data
            scaled_new_data = scaler.transform(new_data[['close']])

            # Convert to DataFrame for consistency
            scaled_new_data = pd.DataFrame(scaled_new_data, columns=['close'])

            # Generate sequences
            X_new, _ = create_sequences(scaled_new_data, window_size)

            # Make predictions
            predictions = model(X_new).squeeze().numpy()

            # Inverse scale predictions
            predicted_prices = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

        return predicted_prices


    # Main pipeline
    if __name__ == "__main__":
        if len(sys.argv) < 2:
            print("Usage: python lstm.py <stock> <input_size> <hidden_size> <num_layers> <num_epochs>")
            sys.exit(1)

        # Parse arguments
        stock = sys.argv[1] if len(sys.argv) > 2 else "AAPL"
        input_size = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        hidden_size = int(sys.argv[3]) if len(sys.argv) > 3 else 100
        num_layers = int(sys.argv[4]) if len(sys.argv) > 4 else 2
        num_epochs = int(sys.argv[5]) if len(sys.argv) > 5 else 20
        window_size = int(sys.argv[6]) if len(sys.argv) > 6 else 7

        # Fetch and preprocess data
        today = datetime.now()
        end_date = (today - timedelta(days=1)).date()
        print(end_date)
        start_date = end_date - relativedelta(months=120)
        print(start_date)
        data = fetch_stock_data(stock, start_date, end_date)
        scaled_data, scaler = preprocess_data(data)

        # Create sequences
        X, y = create_sequences(scaled_data, window_size)

        # Split data into training and testing
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

        # Initialize and train the model
        model = StockPriceLSTM(input_size, hidden_size, num_layers, output_size=1)
        criterion = NN.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model = train_model(model, train_loader, num_epochs, criterion, optimizer)

        # Evaluate the model
        predicted_prices, actual_prices, rmse, mae, r2 = evaluate_model(model, X_test, y_test, scaler)
        print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R^2 Score: {r2:.2f}")

        # Plot results
        time_index = data.index[-len(actual_prices):]
        plot_results(time_index, actual_prices, predicted_prices, f"{stock} Stock Price Prediction")

        # Perform inference on the last window of the test set
    print("\nPerforming Inference...")
    new_data = data[-(window_size + len(y_test)):]  # Use part of the test data or new unseen data
    predicted_future_prices = make_inference(model, new_data, window_size, scaler)

    # Extract the last predicted date, actual price, and predicted price
    last_date = new_data.index[-3:]  # The last date in the new_data
    last_actual_price = new_data['close'].values[-3:]  # The actual price on the last date
    last_predicted_price = predicted_future_prices[-3:]  # The last predicted price

    # Print the last three days' data
    print("\nLast 3 Days Prices:")
    for i in range(3):
        print(f"Date: {last_date[i]}, Actual Price: {last_actual_price[i]:.2f}, Predicted Price: {last_predicted_price[i]:.2f}")
    # Plot the actual vs predicted prices for the inference
    time_index_future = new_data.index[-len(predicted_future_prices):]

    plt.figure(figsize=(14, 8))
    plt.plot(time_index_future, new_data['close'].values[-len(predicted_future_prices):], 
            label="Actual Prices", color="blue", linewidth=2)
    plt.plot(time_index_future, predicted_future_prices, 
            label="Predicted Prices", color="orange", linestyle="--", linewidth=2)
    plt.title(f"{stock} Stock Price Inference", fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Close Price (USD)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
