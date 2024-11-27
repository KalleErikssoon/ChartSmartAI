// Date Validation
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

// Get Selected Strategies
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

// Confirm Retrain Model
function confirmRetrain() {
    if (!validateDates()) {
        return;
    }
    const strategies = getSelectedStrategies();
    if (strategies.length === 0) {
        alert('Please select at least one strategy.');
        return;
    }
    showModal('Confirm Retrain', `Are you sure you want to retrain the model with strategies: ${strategies.join(', ')}?`, () => {
        startRetraining(strategies);
    });
}

// Start Retraining Process
function startRetraining(strategies) {
    closeModal();
    const progressBar = document.getElementById('progress-bar');
    const logs = document.getElementById('logs');
    const progressBarInner = document.getElementById('progress-bar-inner');

    progressBar.style.display = 'block';
    logs.style.display = 'block';
    logs.innerHTML = '';

    // Simulate training progress
    let progress = 0;
    const interval = setInterval(() => {
        progress += 10;
        progressBarInner.style.width = progress + '%';
        logs.innerHTML += `Training progress: ${progress}% with strategies: ${strategies.join(', ')}\n`;
        logs.scrollTop = logs.scrollHeight;

        if (progress >= 100) {
            clearInterval(interval);
            logs.innerHTML += 'Training completed successfully.\n';
            progressBar.style.display = 'none';
        }
    }, 500);
}

// Verify Model
function verifyModel() {
    // Simulate model verification
    const verificationResult = document.getElementById('verification-result');
    verificationResult.style.display = 'block';
    verificationResult.innerText = 'Model verification in progress...';

    setTimeout(() => {
        verificationResult.innerText = 'New model meets performance thresholds.';
        // Update metrics
        document.getElementById('accuracy').innerText = '95%';
        document.getElementById('precision').innerText = '93%';
        document.getElementById('recall').innerText = '92%';
        document.getElementById('f1-score').innerText = '94%';

    }, 2000);
}

// Deploy Model
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
        // Add rollback logic here
    });
}

// Modal Functions
function showModal(title, message, confirmCallback) {
    document.getElementById('modal-title').innerText = title;
    document.getElementById('modal-message').innerText = message;
    document.getElementById('confirmation-modal').style.display = 'block';
    document.getElementById('modal-confirm-button').onclick = confirmCallback;
}

function closeModal() {
    document.getElementById('confirmation-modal').style.display = 'none';
}

// Event Listeners for Date Inputs
document.getElementById('start-date').addEventListener('change', validateDates);
document.getElementById('end-date').addEventListener('change', validateDates);
