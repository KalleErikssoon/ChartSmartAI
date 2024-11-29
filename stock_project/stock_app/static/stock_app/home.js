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
        infoBox.value = "Relative Strength Index";
    } else if (indicator == 'MACD') {
        infoHeader.textContent = "MACD";
        infoBox.value = "The MACD (Moving Average Convergence Divergence) is a momentum indicator used to analyze price trends by comparing two moving averages of a stock's price. Our machine learning model uses MACD patterns, to predict short-term trends and recommend Buy, Hold, or Sell decisions based on expected price movements over the next t+3 days. This approach combines technical analysis with AI to provide actionable insights for confident trading."
    } else if (indicator == 'EMA') {
        infoHeader.textContent = "EMA";
        infoBox.value = "EMA"
    }
}