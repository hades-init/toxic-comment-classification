async function classifyComment() {
    const comment = document.getElementById('textbox').value.trim();
    if (!comment) {
        errorMessage.textContent = '* Please enter some text before submitting.';
        return;
    }

    try {
        // Send POST request to backend
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: comment }),
        });

        if (!response.ok) throw new Error('Backend error');

        const data = await response.json();
        displayResults(data);
    } catch (error) {
        console.error('Error:', error);
        const scoresList = document.getElementById('scores')
        scoresList.textContent = 'Failed to get prediction! Check console for details.'
    }
}

function displayResults(data) {
    const resultDiv = document.getElementById('result');
    const scoresList = document.getElementById('scores');
    scoresList.innerHTML = ''; // Clear previous results

    // Add each label and score to the list
    for (const [label, score] of Object.entries(data)) {
        const li = document.createElement('li');
        li.textContent = `${label}: ${score.toFixed(4)}`;
        scoresList.appendChild(li);
    }

    resultDiv.style.display = 'block'; // Show results
}

function clearText() {
    const textBox = document.getElementById('textbox');
    const errorMessage = document.getElementById('errorMessage');
    const scoresList = document.getElementById('scores');
    textBox.value = '';
    errorMessage.textContent = '';
    scoresList.innerHTML = ''
}

function clearError() {
    const errorMessage = document.getElementById('errorMessage');
    errorMessage.textContent = ''; // Clear error message when user starts typing
}