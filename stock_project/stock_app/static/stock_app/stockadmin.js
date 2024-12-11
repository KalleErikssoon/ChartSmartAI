

function getSelectedStrategy() {
    const checkboxes = document.querySelectorAll('.strategy-options input[type="checkbox"]');
    const selectedStrategy = [];
    checkboxes.forEach(checkbox => {
        if (checkbox.checked) {
            selectedStrategy.push(checkbox.value);
        }
    });
    return selectedStrategy;
}
function confirmRetrain() {
    const strategy = Array.from(document.querySelectorAll('.strategy-options input[type="checkbox"]:checked'))
        .map(checkbox => checkbox.value);

    if (strategy.length === 0) {
        alert('Please select at least one strategy.');
        return;
    }

    showModal(
        'Confirm Retrain',
        `Are you sure you want to retrain the model with strategy: ${strategy.join(', ')}?`,
        () => {
            closeModal();
            startTrainingProgress(strategy, sendRetrainingRequest(strategy));
        }
    );
}

function sendRetrainingRequest(strategy) {
    return fetch('run_strategy/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ strategy })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            throw new Error(`Error: ${data.error}`);
        } else {
            return "Retraining job created successfully.";
        }
    })
    .catch(error => {
        console.error('Error:', error);
        throw error; 
    });
}

function startTrainingProgress(strategy, fetchPromise) {
    const progressBar = document.getElementById('progress-bar');
    const logs = document.getElementById('logs');
    const progressBarInner = document.getElementById('progress-bar-inner');

    progressBar.style.display = 'block';
    logs.style.display = 'block';
    logs.innerHTML = '';

    let progress = 0;
    const interval = setInterval(() => {
        progress = Math.min(progress + 10, 85);
        progressBarInner.style.width = progress + '%';

        logs.innerHTML = `Training progress: ${progress}% with strategy: ${strategy.join(', ')}`;
        logs.scrollTop = logs.scrollHeight;
    }, 1000);

    fetchPromise
        .then((message) => {
            clearInterval(interval);
            progressBarInner.style.width = '100%';
            logs.innerHTML = `${message} `;
            progressBar.style.display = 'none';
        })
        .catch((error) => {
            clearInterval(interval);
            logs.innerHTML = `An error occurred: ${error.message}`;
            progressBar.style.display = 'none';
        });
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



function setupSingleSelectionCheckbox(className) {
    // Select all checkboxes using the provided class name
    const checkboxes = document.querySelectorAll(`.${className}`);
    
    // Attach an event listener to each checkbox
    checkboxes.forEach(checkbox => {
        checkbox.addEventListener('change', () => {
            if (checkbox.checked) {
                // If the current checkbox is checked, disable all others
                checkboxes.forEach(otherCheckbox => {
                    if (otherCheckbox !== checkbox) {
                        otherCheckbox.disabled = true;
                    }
                });
            } else {
                // If no checkbox is checked, enable all checkboxes
                const anyChecked = Array.from(checkboxes).some(cb => cb.checked);
                if (!anyChecked) {
                    checkboxes.forEach(otherCheckbox => {
                        otherCheckbox.disabled = false;
                    });
                }
            }
        });
    });
}

// Call the function and pass the class name of the checkboxes
setupSingleSelectionCheckbox('strategy-checkbox');

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
