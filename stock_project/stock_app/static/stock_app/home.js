// Author: Isaac Lindegren Ternbom
//variaböes to track selected stock and indicator
let selectedStock = null;
let selectedIndicator = null;

// Updated function to set and validate both stock and strategy
function setupSelectionListeners() {
    // Listen for stock selection
    const stockListButtons = document.querySelectorAll(".stock-list li button");
    stockListButtons.forEach((button) => {
        button.addEventListener("click", function () {
            selectedStock = button.getAttribute("stock-name");
            console.log(`Selected stock: ${selectedStock}`);
            checkAndRunInference();
        });
    });

    // Listen for indicator (strategy) selection
    const indicatorButtons = document.querySelectorAll(".top-section .indicators button");
    indicatorButtons.forEach((button) => {
        button.addEventListener("click", function () {
            selectedIndicator = button.textContent.trim();
            console.log(`Selected strategy: ${selectedIndicator}`);
            checkAndRunInference();
        });
    });
}

// Event listener function for stock list
function setupStockListButtons() {
    // Getters
    const stockListButtons = document.querySelectorAll(".stock-list li button");
    const stockHeader = document.getElementById("stock-header");

    // Add event listener for each button
    stockListButtons.forEach((button) => {
        button.addEventListener("click", function () {
            // Update selected button color state
            stockListButtons.forEach((btn) => btn.classList.remove("selected"));
            button.classList.add("selected");

            // Get the stock name from the button
            const selectedStock = button.getAttribute("stock-name");

            // Update the stockHeader with the selected stock name
            stockHeader.textContent = selectedStock;

            // Update the TradingView widget with the selected stock
            // Map stock names to TradingView symbols
            const stockSymbols = {
                "Nvidia": "NASDAQ:NVDA",
                "Apple": "NASDAQ:AAPL",
                "Microsoft": "NASDAQ:MSFT",
                "Amazon": "NASDAQ:AMZN",
                "Google": "NASDAQ:GOOGL",
                "Meta": "NASDAQ:META",
                "Tesla": "NASDAQ:TSLA",
                "Berkshire Hathaway": "NYSE:BRK.B",
                "Taiwan Semiconductors": "NYSE:TSM",
                "Broadcom": "NASDAQ:AVGO"
            };

            //
            const stockSymbol = stockSymbols[selectedStock];
            loadStockChart(stockSymbol);
        });
    });
}
// Call the function to set up the event listener
setupStockListButtons();

// Function to set the selected indicator
function setIndicator(indicator) {
    const infoBox = document.getElementById("info-box-text");
    const infoHeader = document.getElementById("info-header");
    const indicatorButtons = document.querySelectorAll(".top-section .indicators button");

    indicatorButtons.forEach((button) => {
        if (button.textContent === indicator) {
            button.classList.add("selected");
        } else {
            button.classList.remove("selected");
        }
    });

    selectedIndicator = indicator; // Set the selected indicator
    infoHeader.textContent = indicator;
    infoBox.value = `Selected Indicator: ${indicator}`;

    console.log(`Selected indicator: ${selectedIndicator}`);
}

// Function to initialize and update the Stock Chart
function loadStockChart(symbol) {
    // Clear existing widget
    const container = document.getElementById("tradingview");
    container.innerHTML = ""; 

    // Initialize new widget (JSON Format)
    // To add indicators to chart 
    //      "studies": ["MACD@tv-basicstudies"]
    //      "studies": ["RSI@tv-basicstudies"]
    //      "studies": ["EMA@tv-basicstudies"]

    // width 875, height 400
    new TradingView.widget({
        "width": "100%",
        "height": "100%",
        "symbol": symbol,
        "interval": "D",
        "range": "12M",
        "theme": "light",
        "style": "1",
        "enable_publishing": false,
        "withdateranges": true,
        "hide_top_toolbar": true,
        "hide_side_toolbar": true,
        "allow_symbol_change": false,
        "container_id": "tradingview"
    });
}

// Function for setting the text in the info box when choosing an indicator (RSI, MACD, EMA)
function setIndicator(indicator) {
    // Getters
    const infoBox = document.getElementById("info-box-text");
    const infoHeader = document.getElementById("info-header");
    const indicatorButtons = document.querySelectorAll(".top-section .indicators button");

    // Update the selected color state for the indicator buttons
    indicatorButtons.forEach((button) => {
        if (button.textContent == indicator) {
            button.classList.add("selected");
        } else {
            button.classList.remove("selected");
        }
    });

    // Update the info box depending on what indicator is selected
    if (indicator == 'RSI') {
        infoHeader.textContent = "RSI";
        infoBox.value = "The RSI (Relative Strength Index) is a momentum oscillator that measures the speed and magnitude of recent price movements on a scale of 0 to 100. Values above 70 typically indicate overbought conditions, signaling a potential price reversal or pullback, while values below 30 suggest oversold conditions, indicating a possible upward price correction. Traders use RSI to identify potential entry and exit points and to confirm trends.";
    } else if (indicator == 'MACD') {
        infoHeader.textContent = "MACD";
        infoBox.value = "The MACD (Moving Average Convergence Divergence) is a trend-following momentum indicator that compares the difference between a short-term moving average (typically 12 periods) and a long-term moving average (typically 26 periods). When the MACD line crosses above the signal line, it suggests bullish momentum, while a crossover below indicates bearish momentum. MACD is used to identify trend changes, assess strength, and generate buy or sell signals.";
    } else if (indicator == 'EMA') {
        infoHeader.textContent = "EMA";
        infoBox.value = "The EMA (Exponential Moving Average) is a type of moving average that gives more weight to recent price data, making it more responsive to recent changes compared to a simple moving average. Traders use the EMA to identify the direction of the trend and to spot potential support or resistance levels. Crossovers between shorter-term and longer-term EMAs often serve as signals for entry or exit points in a trade.";
    }
    
}

//Function to call the backend API
function runInference(stock, indicator) {
    console.log("runInference called with:", stock, indicator);
    const apiUrl = `/predict/${indicator}/${stock}/`;
    fetch(apiUrl)
        .then((response) => {
            if (!response.ok) {
                throw new Error(`Error: ${response.statusText}`);
            }
            return response.json();
        })
        .then((data) => {
            console.log("Inference result:", data);
            displayInferenceResult(data);
        })
        .catch((error) => {
            console.error("Error during inference:", error);
            alert("Failed to run inference. Please try again.");
        });
}

//Function to check and run inference
function checkAndRunInference() {
    if(selectedStock && selectedIndicator) {
        console.log(`Running inference for stock: ${selectedStock} and indicator: ${selectedIndicator}`);
        runInference(selectedStock, selectedIndicator);
    } else {
        console.log("Both stock and indicator need to be selected before running inference.");
    }
}

setupSelectionListeners();


// Function to display the inference result
function displayInferenceResult(data) {
    const resultBox = document.getElementById("result-box");
    const prediction = data.prediction;

    console.log("Prediction received:", prediction);

    // Highlight corresponding prediction (buy/hold/sell)
    const buyButton = document.getElementById("buy-button");
    const holdButton = document.getElementById("hold-button");
    const sellButton = document.getElementById("sell-button");

    console.log("Buy Button:", buyButton);
    console.log("Hold Button:", holdButton);
    console.log("Sell Button:", sellButton);

    // Check if buttons exist
    if (!buyButton || !holdButton || !sellButton) {
        console.error("One or more prediction buttons are missing from the DOM.");
        resultBox.textContent = "Prediction buttons not found. Please check the HTML.";
        return;
    }

    // Normalize the prediction
    const normalizedPrediction = prediction.toLowerCase();

    // Reset button styles
    [buyButton, holdButton, sellButton].forEach((button) =>
        button.classList.remove("highlight")
    );

    // Highlight based on prediction
    if (["Buy", "Hold", "Sell"].includes(prediction)) {
        if (prediction === "Buy") {
            buyButton.classList.add("highlight");
        } else if (prediction === "Hold") {
            holdButton.classList.add("highlight");
        } else if (prediction === "Sell") {
            sellButton.classList.add("highlight");
        }
    } else {
        console.error("Unexpected prediction:", prediction);
    }
}
