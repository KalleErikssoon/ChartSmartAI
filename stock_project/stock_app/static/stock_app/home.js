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
        });
    });
}
// Call the function to set up the event listener
setupStockListButtons();


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