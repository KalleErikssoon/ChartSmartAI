//aate Validation
function validateDates() {
    const startDateInput = document.getElementById('start-date');
    const endDateInput = document.getElementById('end-date');
    const validationMessage = document.getElementById('date-validation-message');

    if (startDateInput.value && endDateInput.value) {
        const startDate = new Date(startDateInput.value);
        const endDate = new Date(endDateInput.value);

        if (startDate >= endDate) {
            validationMessage.style.display = 'block';
            return false;
        } else {
            validationMessage.style.display = 'none';
            return true;
        }
    }
    validationMessage.style.display = 'none';
    return false;
}


function getSelectedStrategies() {
    const checkboxes = document.querySelectorAll('.strategy-options input[type="checkbox"]');
    const selectedStrategies = [];
    checkboxes.forEach(checkbox => {
        if (checkbox.checked) {
            selectedStrategies.push(checkbox.value);
        }
    });
    return selectedStrategies;
}

// Confirm  button for Retrain
function confirmRetrain() {
    const startDateInput = document.getElementById('start-date');
    const endDateInput = document.getElementById('end-date');
    const dateValidationMessage = document.getElementById('date-validation-message');

    //get selected strategies
    const strategies = Array.from(document.querySelectorAll('.strategy-options input[type="checkbox"]:checked'))
        .map(checkbox => checkbox.value);

    //validate dates
    const startDate = startDateInput.value;
    const endDate = endDateInput.value;
    if (!startDate || !endDate) {
        dateValidationMessage.textContent = 'Both Start Date and End Date must be entered.';
        dateValidationMessage.style.display = 'block';
        return;
    }
    if (new Date(startDate) >= new Date(endDate)) {
        dateValidationMessage.textContent = 'Start Date must be before End Date.';
        dateValidationMessage.style.display = 'block';
        return;
    }
    dateValidationMessage.style.display = 'none'; // Clear validation message

    //validate strategies
    if (strategies.length === 0) {
        alert('Please select at least one strategy.');
        return;
    }

    //show confirmation modal
    showModal(
        'Confirm Retrain',
        `Are you sure you want to retrain the model with strategies: ${strategies.join(', ')}?`,
        () => {
            startRetraining(strategies);
        }
    );
}

function startRetraining(strategies) {
    closeModal();
    const progressBar = document.getElementById('progress-bar');
    const logs = document.getElementById('logs');
    const progressBarInner = document.getElementById('progress-bar-inner');

    progressBar.style.display = 'block';
    logs.style.display = 'block';
    logs.innerHTML = ''; 

    // Training bar
    let progress = 0;
    const interval = setInterval(() => {
        progress += 10;
        progressBarInner.style.width = progress + '%';
        
        // Overwrite the log content with the current progress
        logs.innerHTML = `Training progress: ${progress}% with strategies: ${strategies.join(', ')}`;
        logs.scrollTop = logs.scrollHeight;

        if (progress >= 100) {
            clearInterval(interval);
            logs.innerHTML = 'Training completed successfully.';
            progressBar.style.display = 'none';
        }
    }, 500);
}



function verifyModel() {
    const verificationResult = document.getElementById('verification-result');
    verificationResult.style.display = 'block';
    verificationResult.innerText = 'Model verification in progress...';

    setTimeout(() => {
        verificationResult.innerText = 'New model meets performance thresholds.';
        document.getElementById('accuracy').innerText = '95%';
        document.getElementById('precision').innerText = '93%';
        document.getElementById('recall').innerText = '92%';
        document.getElementById('f1-score').innerText = '94%';

    }, 2000);
}

// confirm deployment
function deployModel() {
    showModal('Confirm Deployment', 'Are you sure you want to deploy the new model?', () => {
        closeModal();
        alert('New model deployed successfully.');
        // deployment logic here
    });
}

// Confirm Rollback
function confirmRollback() {
    const selectedVersion = document.getElementById('model-version').value;
    showModal('Confirm Rollback', `Are you sure you want to rollback to ${selectedVersion}?`, () => {
        closeModal();
        alert(`Model rolled back to ${selectedVersion}.`);
        //rollback logic here
    });
}

function showModal(title, message, confirmCallback) {
    document.getElementById('modal-title').innerText = title;
    document.getElementById('modal-message').innerText = message;
    document.getElementById('confirmation-modal').style.display = 'block';
    document.getElementById('modal-confirm-button').onclick = confirmCallback;
}

function closeModal() {
    document.getElementById('confirmation-modal').style.display = 'none';
}

//event Listeners for Date Inputs
document.getElementById('start-date').addEventListener('change', validateDates);
document.getElementById('end-date').addEventListener('change', validateDates);


// window.onload = function() {
//     // Get the context of the canvas element we want to select
//     var ctx = document.getElementById('performance-chart').getContext('2d');

//     // For the purpose of this example, we will use some dummy data
//     // In practice, you would retrieve these values from the server or calculate them
//     var accuracy = 85;
//     var precision = 80;
//     var recall = 75;
//     var f1Score = 77;

//     // Update the DOM elements with the actual values
//     document.getElementById('accuracy').textContent = accuracy + '%';
//     document.getElementById('precision').textContent = precision + '%';
//     document.getElementById('recall').textContent = recall + '%';
//     document.getElementById('f1-score').textContent = f1Score + '%';

//     // Create the chart
//     var myChart = new Chart(ctx, {
//         type: 'bar',
//         data: {
//             labels: ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
//             datasets: [{
//                 label: 'Model Performance (%)',
//                 data: [accuracy, precision, recall, f1Score],
//                 backgroundColor: [
//                     'rgba(75, 192, 192, 0.2)', // Accuracy
//                     'rgba(54, 162, 235, 0.2)', // Precision
//                     'rgba(255, 206, 86, 0.2)', // Recall
//                     'rgba(153, 102, 255, 0.2)' // F1 Score
//                 ],
//                 borderColor: [
//                     'rgba(75, 192, 192, 1)', // Accuracy
//                     'rgba(54, 162, 235, 1)', // Precision
//                     'rgba(255, 206, 86, 1)', // Recall
//                     'rgba(153, 102, 255, 1)' // F1 Score
//                 ],
//                 borderWidth: 1
//             }]
//         },
//         options: {
//             scales: {
//                 y: {
//                     beginAtZero: true,
//                     max: 100
//                 }
//             }
//         }
//     });
// };
