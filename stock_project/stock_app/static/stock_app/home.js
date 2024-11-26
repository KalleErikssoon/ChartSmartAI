// Get
const stockListButtons = document.querySelectorAll(".stock-list li button");
const stockHeader = document.getElementById("stock-header");

// Add event listener
stockListButtons.forEach((button) => {
    button.addEventListener("click", function () {
        // Get the stock name from the button
        const selectedStock = button.getAttribute("stock-name");

        // Update the stockHeader with the selected stock name
        stockHeader.textContent = selectedStock;
    });
});