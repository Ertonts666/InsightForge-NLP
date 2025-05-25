/**
 * NLP Insights Engine UI
 * JavaScript for handling UI interactions and API calls
 */

// API endpoint configuration
const API_BASE_URL = 'http://localhost:8000';

// DOM elements
document.addEventListener('DOMContentLoaded', function() {
    // Navigation
    const navLinks = document.querySelectorAll('.nav-link');
    const pages = document.querySelectorAll('.page');
    
    // Sentiment Analysis
    const sentimentForm = document.getElementById('sentiment-form');
    const sentimentResultsCard = document.getElementById('sentiment-results-card');
    const sentimentResults = document.getElementById('sentiment-results');
    
    // Question Answering
    const qaForm = document.getElementById('qa-form');
    const qaResultsCard = document.getElementById('qa-results-card');
    const qaResults = document.getElementById('qa-results');
    const useContextCheckbox = document.getElementById('use-context');
    const contextContainer = document.getElementById('context-container');
    
    // Document Management
    const documentForm = document.getElementById('document-form');
    
    // Initialize UI
    initializeNavigation();
    initializeSentimentAnalysis();
    initializeQuestionAnswering();
    initializeDocumentManagement();
    
    /**
     * Initialize navigation between pages
     */
    function initializeNavigation() {
        navLinks.forEach(link => {
            link.addEventListener('click', function(e) {
                e.preventDefault();
                
                // Update active nav link
                navLinks.forEach(l => l.classList.remove('active'));
                this.classList.add('active');
                
                // Show selected page
                const targetPage = this.getAttribute('data-page');
                pages.forEach(page => {
                    page.style.display = 'none';
                });
                document.getElementById(`${targetPage}-page`).style.display = 'block';
            });
        });
    }
    
    /**
     * Initialize sentiment analysis functionality
     */
    function initializeSentimentAnalysis() {
        sentimentForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const text = document.getElementById('sentiment-text').value.trim();
            const language = document.getElementById('sentiment-language').value;
            const aspectsInput = document.getElementById('sentiment-aspects').value.trim();
            
            if (!text) {
                alert('Please enter text to analyze');
                return;
            }
            
            // Show loading state
            sentimentResults.innerHTML = createLoadingSpinner();
            sentimentResultsCard.style.display = 'block';
            
            try {
                let result;
                
                // Check if aspects are provided
                if (aspectsInput) {
                    const aspects = aspectsInput.split(',').map(aspect => aspect.trim());
                    result = await analyzeAspectSentiment(text, aspects, language);
                    displayAspectSentimentResults(result);
                } else {
                    result = await analyzeSentiment(text, language);
                    displaySentimentResults(result);
                }
            } catch (error) {
                displayError(sentimentResults, error);
            }
        });
    }
    
    /**
     * Initialize question answering functionality
     */
    function initializeQuestionAnswering() {
        // Toggle context input visibility
        useContextCheckbox.addEventListener('change', function() {
            contextContainer.style.display = this.checked ? 'block' : 'none';
        });
        
        // Handle question form submission
        qaForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const question = document.getElementById('question-input').value.trim();
            const useContext = useContextCheckbox.checked;
            const context = useContext ? document.getElementById('context-input').value.trim() : null;
            
            if (!question) {
                alert('Please enter a question');
                return;
            }
            
            // Show loading state
            qaResults.innerHTML = createLoadingSpinner();
            qaResultsCard.style.display = 'block';
            
            try {
                const result = await answerQuestion(question, context);
                displayQuestionResults(result);
            } catch (error) {
                displayError(qaResults, error);
            }
        });
    }
    
    /**
     * Initialize document management functionality
     */
    function initializeDocumentManagement() {
        documentForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const text = document.getElementById('document-text').value.trim();
            const title = document.getElementById('document-title').value.trim();
            const category = document.getElementById('document-category').value.trim();
            
            if (!text) {
                alert('Please enter document text');
                return;
            }
            
            try {
                const result = await addDocument(text, title, category);
                alert(`Document added successfully with ID: ${result.document_id}`);
                
                // Clear form
                document.getElementById('document-text').value = '';
                document.getElementById('document-title').value = '';
                document.getElementById('document-category').value = '';
            } catch (error) {
                alert(`Error adding document: ${error.message}`);
            }
        });
    }
    
    /**
     * API call to analyze sentiment
     */
    async function analyzeSentiment(text, language) {
        const response = await fetch(`${API_BASE_URL}/sentiment`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                language: language === 'multilingual' ? null : language
            }),
        });
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        
        return await response.json();
    }
    
    /**
     * API call to analyze aspect-based sentiment
     */
    async function analyzeAspectSentiment(text, aspects, language) {
        const response = await fetch(`${API_BASE_URL}/sentiment/aspects`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                aspects: aspects,
                language: language === 'multilingual' ? null : language
            }),
        });
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        
        return await response.json();
    }
    
    /**
     * API call to answer a question
     */
    async function answerQuestion(question, context = null) {
        const response = await fetch(`${API_BASE_URL}/question`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: question,
                context: context,
                top_k: 5,
                threshold: 0.01
            }),
        });
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        
        return await response.json();
    }
    
    /**
     * API call to add a document to the knowledge base
     */
    async function addDocument(text, title, category) {
        const response = await fetch(`${API_BASE_URL}/documents`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                metadata: {
                    title: title || 'Untitled',
                    category: category || 'Uncategorized',
                    source: 'user-submitted'
                }
            }),
        });
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        
        return await response.json();
    }
    
    /**
     * Display sentiment analysis results
     */
    function displaySentimentResults(result) {
        const sentimentClass = getSentimentClass(result.sentiment);
        const scorePercentage = Math.round(result.score * 100);
        
        const html = `
            <div class="sentiment-result ${sentimentClass}">
                <h4>Sentiment: <span class="badge ${getBadgeClass(result.sentiment)}">${capitalizeFirstLetter(result.sentiment)}</span></h4>
                <p>Confidence: ${(result.confidence * 100).toFixed(2)}%</p>
                <p>Language: ${result.language}</p>
                
                <div class="mt-3">
                    <label>Sentiment Score:</label>
                    <div class="progress">
                        <div class="progress-bar ${getProgressBarClass(result.sentiment)}" 
                             role="progressbar" 
                             style="width: ${scorePercentage}%" 
                             aria-valuenow="${scorePercentage}" 
                             aria-valuemin="0" 
                             aria-valuemax="100">
                            ${scorePercentage}%
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        sentimentResults.innerHTML = html;
    }
    
    /**
     * Display aspect-based sentiment analysis results
     */
    function displayAspectSentimentResults(result) {
        const overall = result.overall;
        const aspects = result.aspects;
        
        const overallSentimentClass = getSentimentClass(overall.sentiment);
        const overallScorePercentage = Math.round(overall.score * 100);
        
        let aspectsHtml = '';
        for (const [aspect, sentiment] of Object.entries(aspects)) {
            if (sentiment.mentions > 0) {
                const aspectSentimentClass = getSentimentClass(sentiment.sentiment);
                const aspectScorePercentage = Math.round(sentiment.score * 100);
                
                aspectsHtml += `
                    <div class="aspect-item">
                        <h6>${aspect} <span class="badge ${getBadgeClass(sentiment.sentiment)}">${capitalizeFirstLetter(sentiment.sentiment)}</span></h6>
                        <div class="progress">
                            <div class="progress-bar ${getProgressBarClass(sentiment.sentiment)}" 
                                 role="progressbar" 
                                 style="width: ${aspectScorePercentage}%" 
                                 aria-valuenow="${aspectScorePercentage}" 
                                 aria-valuemin="0" 
                                 aria-valuemax="100">
                                ${aspectScorePercentage}%
                            </div>
                        </div>
                        <small>Mentions: ${sentiment.mentions} | Confidence: ${(sentiment.confidence * 100).toFixed(2)}%</small>
                    </div>
                `;
            } else {
                aspectsHtml += `
                    <div class="aspect-item">
                        <h6>${aspect} <span class="badge bg-secondary">Not Mentioned</span></h6>
                    </div>
                `;
            }
        }
        
        const html = `
            <div class="sentiment-result ${overallSentimentClass}">
                <h4>Overall Sentiment: <span class="badge ${getBadgeClass(overall.sentiment)}">${capitalizeFirstLetter(overall.sentiment)}</span></h4>
                <p>Confidence: ${(overall.confidence * 100).toFixed(2)}%</p>
                <p>Language: ${overall.language}</p>
                
                <div class="mt-3">
                    <label>Overall Score:</label>
                    <div class="progress">
                        <div class="progress-bar ${getProgressBarClass(overall.sentiment)}" 
                             role="progressbar" 
                             style="width: ${overallScorePercentage}%" 
                             aria-valuenow="${overallScorePercentage}" 
                             aria-valuemin="0" 
                             aria-valuemax="100">
                            ${overallScorePercentage}%
                        </div>
                    </div>
                </div>
                
                <h5 class="mt-4">Aspect Analysis:</h5>
                <div class="aspects-container">
                    ${aspectsHtml}
                </div>
            </div>
        `;
        
        sentimentResults.innerHTML = html;
    }
    
    /**
     * Display question answering results
     */
    function displayQuestionResults(result) {
        if (result.has_answer) {
            const confidencePercentage = Math.round(result.confidence * 100);
            
            let sourcesHtml = '';
            if (result.sources && result.sources.length > 0) {
                sourcesHtml = '<h5 class="mt-3">Sources:</h5><div class="sources-container">';
                
                result.sources.forEach((source, index) => {
                    if (source && Object.keys(source).length > 0) {
                        sourcesHtml += `
                            <div class="source-item">
                                <strong>${source.metadata.title || 'Untitled'}</strong>
                                ${source.metadata.category ? `<span class="badge bg-info text-dark ms-2">${source.metadata.category}</span>` : ''}
                                ${source.metadata.source ? `<small class="text-muted ms-2">(${source.metadata.source})</small>` : ''}
                            </div>
                        `;
                    }
                });
                
                sourcesHtml += '</div>';
            }
            
            const html = `
                <div class="answer-container">
                    <div class="answer-text">${result.answer}</div>
                    
                    <div class="mt-3">
                        <label>Confidence:</label>
                        <div class="confidence-bar">
                            <div class="confidence-level" style="width: ${confidencePercentage}%"></div>
                        </div>
                        <small>${confidencePercentage}%</small>
                    </div>
                    
                    ${sourcesHtml}
                </div>
            `;
            
            qaResults.innerHTML = html;
        } else {
            qaResults.innerHTML = `
                <div class="alert alert-warning">
                    <i class="bi bi-exclamation-triangle"></i> 
                    ${result.message || "Couldn't find an answer to your question."}
                </div>
            `;
        }
    }
    
    /**
     * Display error message
     */
    function displayError(container, error) {
        container.innerHTML = `
            <div class="alert alert-danger">
                <i class="bi bi-exclamation-circle"></i> 
                Error: ${error.message || 'Something went wrong'}
            </div>
        `;
    }
    
    /**
     * Create loading spinner HTML
     */
    function createLoadingSpinner() {
        return `
            <div class="spinner-container">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
            </div>
        `;
    }
    
    /**
     * Get CSS class for sentiment
     */
    function getSentimentClass(sentiment) {
        switch (sentiment.toLowerCase()) {
            case 'positive':
                return 'sentiment-positive';
            case 'negative':
                return 'sentiment-negative';
            default:
                return 'sentiment-neutral';
        }
    }
    
    /**
     * Get Bootstrap badge class for sentiment
     */
    function getBadgeClass(sentiment) {
        switch (sentiment.toLowerCase()) {
            case 'positive':
                return 'bg-success';
            case 'negative':
                return 'bg-danger';
            default:
                return 'bg-secondary';
        }
    }
    
    /**
     * Get Bootstrap progress bar class for sentiment
     */
    function getProgressBarClass(sentiment) {
        switch (sentiment.toLowerCase()) {
            case 'positive':
                return 'bg-success';
            case 'negative':
                return 'bg-danger';
            default:
                return 'bg-secondary';
        }
    }
    
    /**
     * Capitalize first letter of a string
     */
    function capitalizeFirstLetter(string) {
        return string.charAt(0).toUpperCase() + string.slice(1);
    }
});
