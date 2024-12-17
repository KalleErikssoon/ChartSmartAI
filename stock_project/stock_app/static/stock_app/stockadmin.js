// Author: Mehmet Asim Altinisik
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
            get_model_version_list();
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
    // verificationResult.innerText = 'Model verification in progress...';

    const modelVersionDropdown = document.getElementById('model-version-evaluation');
    const selectedModel = modelVersionDropdown.value;

    fetch('get_performance/')
        .then(response => response.json())
        .then(data => {
            // Find the performance data for the selected model
            const performanceData = data.performance_data.find(item => item.model_pickle === selectedModel);
            if (performanceData) {
                const accuracy = performanceData.accuracy;
                const macroAvg = performanceData.macro_avg;

                // verificationResult.innerText = 'New model meets performance thresholds.';
                document.getElementById('accuracy').innerText = `${(accuracy * 100).toFixed(2)}%`;
                document.getElementById('precision').innerText = `${(macroAvg.precision * 100).toFixed(2)}%`;
                document.getElementById('recall').innerText = `${(macroAvg.recall * 100).toFixed(2)}%`;
                document.getElementById('f1-score').innerText = `${(macroAvg["f1-score"] * 100).toFixed(2)}%`;
            } else {
                verificationResult.innerText = 'Selected model data not found.';
            }
        })
        .catch(error => {
            verificationResult.innerText = 'Error fetching performance data.';
            console.error('Error:', error);
        });
}

function get_model_version_list() {
    fetch('get_models/')
        .then(response => response.json())
        .then(data => {
            // Check if the response has files
            if (data.files && Array.isArray(data.files)) {
                const selectElement = document.getElementById('model-version');
                const selectPerformance = document.getElementById('model-version-evaluation');

                // Clear existing options
                selectElement.innerHTML = '';
                selectPerformance.innerHTML = '';

                // Add an empty placeholder option to "Select Model"
                const placeholderOption1 = document.createElement('option');
                placeholderOption1.value = '';
                placeholderOption1.textContent = 'Select a model';
                placeholderOption1.disabled = true;
                placeholderOption1.selected = true;
                selectElement.appendChild(placeholderOption1);

                // Add an empty placeholder option to "Model Evaluation"
                const placeholderOption2 = document.createElement('option');
                placeholderOption2.value = '';
                placeholderOption2.textContent = 'Select a model';
                placeholderOption2.disabled = true;
                placeholderOption2.selected = true;
                selectPerformance.appendChild(placeholderOption2);

                // Populate the select dropdowns with file options
                data.files.forEach(file => {
                    const option1 = document.createElement('option');
                    option1.value = file; // Set the value to the file name
                    option1.textContent = file; // Display the file name
                    selectElement.appendChild(option1);

                    const option2 = document.createElement('option');
                    option2.value = file; // Set the value to the file name
                    option2.textContent = file; // Display the file name
                    selectPerformance.appendChild(option2);
                });
            } else {
                console.error('No files found in the response');
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
}

// Confirm Rollback
function confirmRollback() {
    const selectedVersion = document.getElementById('model-version').value;
    const chosen_model = selectedVersion;  
    const strategyMatch = selectedVersion.match(/_(ema|macd|rsi)_/i);
    const chosen_strategy = strategyMatch ? strategyMatch[1].toUpperCase() : "UNKNOWN";
    

    showModal('Confirm Rollback', `Are you sure you want to rollback to ${selectedVersion}?`, () => {
        closeModal();
        send_chosen_model(chosen_strategy, chosen_model);
    });
}
function send_chosen_model(chosen_strategy, chosen_model) {
    fetch('change_model/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ chosen_strategy: chosen_strategy, chosen_model: chosen_model })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            console.error('Error:', data.error);
            alert(`Error: ${data.error}`);
        } else {
            console.log('Success:', data.message);
            alert(`Model rolled back to ${chosen_model}.`);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while rolling back the model.');
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
    const checkboxes = document.querySelectorAll(`.${className}`);
    
    // event listener in each checkbox
    checkboxes.forEach(checkbox => {
        checkbox.addEventListener('change', () => {
            if (checkbox.checked) {
                //iff the current checkbox is checked disable all others
                checkboxes.forEach(otherCheckbox => {
                    if (otherCheckbox !== checkbox) {
                        otherCheckbox.disabled = true;
                    }
                });
            } else {
                //if no checkbox is checked enable all checkboxes
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
// this is for selections of strategies
setupSingleSelectionCheckbox('strategy-checkbox');

document.addEventListener('DOMContentLoaded', function() {
    get_model_version_list();
});