function analyzeSentiment() {
    const text = document.getElementById("userInput").value;
    
    fetch("http://127.0.0.1:8080/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "Accept": "application/json"
        },
        body: JSON.stringify({ text: text })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("result").innerText = "Sentiment: " + data.sentiment;
    })
    .catch(error => {
        console.error("Error:", error);
    });
}
