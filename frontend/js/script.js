// Theme toggling functionality
document.addEventListener('DOMContentLoaded', function () {
    const themeToggle = document.getElementById('themeToggle');
    const bodyElement = document.body;
    const htmlElement = document.documentElement;

    // Check for saved theme preference or use preferred color scheme
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark' || (!savedTheme && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
        bodyElement.classList.add('is-dark-mode');
        htmlElement.classList.add('is-dark-mode');
        themeToggle.querySelector('i').className = 'fas fa-sun';
    }

    // Initialize toggle buttons for collapsible sections
    document.querySelectorAll('.toggle-button').forEach(button => {
        button.addEventListener('click', () => {
            const targetId = button.getAttribute('data-target');
            const targetElement = document.getElementById(targetId);

            if (targetElement.style.display === 'none') {
                targetElement.style.display = 'block';
                button.querySelector('i').className = 'fas fa-chevron-down';
            } else {
                targetElement.style.display = 'none';
                button.querySelector('i').className = 'fas fa-chevron-right';
            }
        });
    });

    themeToggle.addEventListener('click', function () {
        const isDarkMode = bodyElement.classList.contains('is-dark-mode');

        if (isDarkMode) {
            bodyElement.classList.remove('is-dark-mode');
            htmlElement.classList.remove('is-dark-mode');
            themeToggle.querySelector('i').className = 'fas fa-moon';
            localStorage.setItem('theme', 'light');
        } else {
            bodyElement.classList.add('is-dark-mode');
            htmlElement.classList.add('is-dark-mode');
            themeToggle.querySelector('i').className = 'fas fa-sun';
            localStorage.setItem('theme', 'dark');
        }
    });
});

// Function to analyze sarcasm
async function analyzeSarcasm() {
    const inputText = document.getElementById('inputText').value.trim();

    if (!inputText) {
        // Create Bulma notification
        const notification = document.createElement('div');
        notification.className = 'notification is-warning';
        notification.innerHTML = '<button class="delete"></button>Please enter some text to analyze';

        // Add to page
        document.querySelector('.container').insertBefore(notification, document.querySelector('.box'));

        // Add event listener to close button
        notification.querySelector('.delete').addEventListener('click', function () {
            notification.remove();
        });

        // Auto remove after 3 seconds
        setTimeout(() => notification.remove(), 3000);
        return;
    }

    // Show loading spinner
    document.getElementById('loadingSpinner').classList.remove('is-hidden');
    document.getElementById('resultsSection').classList.add('is-hidden');

    // Scroll to loading spinner
    document.getElementById('loadingSpinner').scrollIntoView({ behavior: 'smooth' });

    try {
        // Make API request
        const response = await fetch('/infer?n_embedding=10', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: inputText })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();
        displayResults(data);

        // Scroll to results after a short delay
        setTimeout(() => {
            document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
        }, 300);

    } catch (error) {
        console.error('Error analyzing text:', error);
        // Create Bulma notification for error
        const notification = document.createElement('div');
        notification.className = 'notification is-danger';
        notification.innerHTML = `<button class="delete"></button>Error analyzing text: ${error.message}`;

        // Add to page
        document.querySelector('.container').insertBefore(notification, document.querySelector('.box'));

        // Add event listener to close button
        notification.querySelector('.delete').addEventListener('click', function () {
            notification.remove();
        });

        // Auto remove after 5 seconds
        setTimeout(() => notification.remove(), 5000);
    } finally {
        // Hide loading spinner
        document.getElementById('loadingSpinner').classList.add('is-hidden');
    }
}

// Function to display results
function displayResults(data) {
    // Update verdict
    const isSarcastic = data.final_verdict.is_sarcastic;
    const confidence = data.final_verdict.confidence;

    document.getElementById('verdictText').textContent = isSarcastic ? 'Sarcastic' : 'Not Sarcastic';
    document.getElementById('confidenceBadge').textContent = `${(confidence * 100).toFixed(0)}% Confidence`;

    // Set badge color based on confidence
    const confidenceBadge = document.getElementById('confidenceBadge');
    if (confidence > 0.7) {
        confidenceBadge.className = 'modern-tag is-success';
    } else if (confidence > 0.4) {
        confidenceBadge.className = 'modern-tag is-warning';
    } else {
        confidenceBadge.className = 'modern-tag is-danger';
    }

    // Update explanation
    document.getElementById('explanationText').textContent = data.final_verdict.explanation;

    // Update predictions
    document.getElementById('llmPrediction').innerHTML = data.is_llm_predicted_sarcasm ?
        '<i class="fas fa-check-circle"></i>' :
        '<i class="fas fa-times-circle"></i>';
    document.getElementById('embeddingPrediction').innerHTML = data.is_embedded_predicted_sarcasm ?
        '<i class="fas fa-check-circle"></i>' :
        '<i class="fas fa-times-circle"></i>';

    // Update reply
    document.getElementById('replyText').textContent = data.reply || 'No reply generated';

    // Update analysis
    document.getElementById('reasoningText').textContent = data.analysis.reasoning;
    document.getElementById('culturalContextText').textContent = data.analysis.cultural_context;
    document.getElementById('toneAnalysisText').textContent = data.analysis.tone_analysis;

    // Update similar texts
    const similarTextsContainer = document.getElementById('similarTextsContainer');
    similarTextsContainer.innerHTML = '';

    // Add similar texts with staggered animation
    data.similar_texts.forEach((text, index) => {
        const scorePercentage = (text.score * 100).toFixed(1);
        const scorePosition = `${text.score * 100}%`;

        const cardHtml = `
            <div class="column is-one-third" style="animation-delay: ${index * 0.1}s">
                <div class="modern-card similar-text-card">
                    <div class="similar-text-header">
                        <span class="similar-text-number">Text ${index + 1}</span>
                        <span class="modern-tag ${text.polarity ? 'is-success' : 'is-danger'}">
                            ${text.polarity ? 'Sarcastic' : 'Not Sarcastic'}
                        </span>
                    </div>
                    <p class="similar-text-content bengali-text">${text.text}</p>
                    ${text.reply ? `<div class="similar-text-reply bengali-text">${text.reply}</div>` : ''}
                    <div class="mt-4">
                        <div class="is-flex is-justify-content-space-between">
                            <small>Similarity</small>
                            <small>${scorePercentage}%</small>
                        </div>
                        <div class="score-indicator" style="--score-width: ${scorePercentage}%">
                            <div class="score-marker" style="left: ${scorePosition}"></div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        similarTextsContainer.innerHTML += cardHtml;
    });

    // Add animation class to each similar text card
    setTimeout(() => {
        const cards = document.querySelectorAll('.similar-text-card');
        cards.forEach((card, index) => {
            setTimeout(() => {
                card.classList.add('fade-in');
            }, index * 100);
        });
    }, 300);

    // Show results section
    document.getElementById('resultsSection').classList.remove('is-hidden');
} 