const reviews = [
  
    "Battery drains way too fast, even on standby.",
    "I have to charge my phone twice a day, battery life is terrible.",
    "Battery life dropped significantly after the latest update.",
    "The device gets hot and battery drains within hours.",
    "Battery percentage jumps around and dies suddenly at 20%.",
    "Even with power saving on, the battery drains quickly.",
    "Battery doesn't last more than 5-6 hours of regular use.",
    "I wake up with 30% less battery without using the phone overnight.",
    "Battery life is very poor, needs serious improvement.",
    "Within a few months, battery backup got really bad.",


    "App crashes every time I try to upload a photo.",
    "Keeps getting stuck while uploading a white image.",
    "I can't upload anything — app freezes on the upload screen.",
    "Trying to post a photo leads to instant crash.",
    "Every time I upload a white image, the app gets stuck.",
    "The app just closes when I tap on the upload button.",
    "Photos won't upload — stuck on processing forever.",
    "White photos break the upload process completely.",
    "Uploading crashes the app more than half the time.",
    "App becomes unresponsive during file uploads.",

    "Takes forever to load after unlocking.",
    "Laggy UI when switching between tabs.",
    "The app is slow and unresponsive most of the time.",
    "Typing has a delay and it's really annoying.",
    "Switching apps causes a short freeze every time.",
    "Scrolling stutters and doesn't feel smooth.",
    "Performance is bad even for basic tasks.",
    "Multitasking causes noticeable lag.",
    "Animations are slow and not fluid at all.",
    "The whole system feels sluggish.",


    "Even at full volume, I can barely hear the audio.",
    "Speakers are too quiet — not suitable for videos.",
    "Volume is too low, especially on calls.",
    "Sound quality is good, but it's not loud enough.",
    "Volume drops randomly while playing music.",
    "Can't hear anything unless I'm in a silent room.",
    "Max volume still feels like 50%.",
    "Audio becomes faint and muffled.",
    "Volume control doesn't actually increase sound much.",


    "Wi-Fi keeps disconnecting randomly.",
    "Poor network reception in my home.",
    "Bluetooth drops connection every few minutes.",
    "Signal bars are full but no internet.",
    "Wi-Fi reconnects every 10 minutes, very unstable.",
    "Calls drop frequently when switching networks.",
    "Device doesn't reconnect to Wi-Fi after sleep.",
    "4G keeps switching to 3G even in good coverage.",
    "Bluetooth range is too short, disconnects easily.",
    "Mobile data doesn't work unless toggled manually.",
    
   
    "Battery life is excellent, lasts me all day with heavy use.",
    "The app works perfectly every time, never had a crash.",
    "Phone performance is lightning fast, no lag whatsoever.",
    "Audio quality is amazing, speakers are loud and clear.",
    "Wi-Fi and mobile connections are rock solid everywhere I go.",
    ]
;
function formatProblemStatement(problemData) {
  const boxes = problemData.map(({ cluster, summary }) => `
    <div class="cluster-box">
      <div class="cluster-header">Problem ${cluster}</div>
      <div class="cluster-summary">${summary}</div>
    </div>
  `).join("");

  return `
    <div class="summary-grid">
      ${boxes}
    </div>
  `;
}





// Function to load sentiment chart with sample reviews
async function loadSentimentChart() {
  try {
    // Step 1: Get sentiment chart data
    const chartResponse = await fetch("/sentiment-chart", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ raw_reviews: reviews })
    });

    const chartData = await chartResponse.json();

    new Chart("myChart", {
      type: "doughnut",
      data: {
        labels: chartData.labels,
        datasets: [{
          backgroundColor: chartData.colors,
          data: chartData.data
        }]
      },
      options: {
        responsive: false,
        maintainAspectRatio: false,
        circumference: 180,
        rotation: -90,
        plugins: {
          legend: {
            display: true,
            position: 'bottom'
          }
        }
      }
    });

    // Step 2: Fetch the problem statement from backend
    const problemResponse = await fetch("/generate-problem", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ raw_reviews: reviews })
    });

    const problemData = await problemResponse.json();
    console.log("works till here");
    document.querySelector(".top").innerHTML = formatProblemStatement(problemData.problem_statements);
    
    // Render bar chart for cluster counters in bottom-left
    if (problemData.counters && Array.isArray(problemData.counters)) {
      const ctx = document.getElementById("barChart").getContext("2d");
      // Destroy previous chart instance if exists
      if (window.barChartInstance) {
        window.barChartInstance.destroy();
      }
      window.barChartInstance = new Chart(ctx, {
        type: "bar",
        data: {
          labels: problemData.counters.map((_, i) => `Cluster ${i+1}`),
          datasets: [{
            label: "Cluster Size",
            data: problemData.counters,
            backgroundColor: "#007bff",
            borderColor: "#0056b3",
            borderWidth: 1,
            borderRadius: 6
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            title: { display: false }
          },
          scales: {
            x: {
              title: { display: true, text: "Cluster" },
              grid: { display: false }
            },
            y: {
              beginAtZero: true,
              title: { display: true, text: "Count" },
              grid: { display: true, color: "#e0e0e0" }
            }
          }
        }
      });
    }
  } catch (err) {
    console.error("Error loading data", err);
    document.querySelector(".top").innerText = "Failed to load insights.";
  }
}

window.addEventListener("DOMContentLoaded", () => {
  document.getElementById("inputForm").addEventListener("submit", function(e) {
    e.preventDefault();
    const userInput = document.getElementById("userInput").value;
    // Split input using ~ as delimiter, trim each entry, and filter out empty ones
    const userReviews = userInput.split("~").map(s => s.trim()).filter(Boolean);
    loadSentimentChartWithInput(userReviews);
  });
});

function loadSentimentChartWithInput(userReviews) {
  // Step 1: Get sentiment chart data
  fetch("/sentiment-chart", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ raw_reviews: userReviews })
  })
    .then(res => res.json())
    .then(chartData => {
      new Chart("myChart", {
        type: "doughnut",
        data: {
          labels: chartData.labels,
          datasets: [{
            backgroundColor: chartData.colors,
            data: chartData.data
          }]
        },
        options: {
          responsive: false,
          maintainAspectRatio: false,
          circumference: 180,
          rotation: -90,
          plugins: {
            legend: {
              display: true,
              position: 'bottom'
            }
          }
        }
      });
      // Step 2: Fetch the problem statement from backend
      return fetch("/generate-problem", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ raw_reviews: userReviews })
      });
    })
    .then(res => res.json())
    .then(problemData => {
      document.querySelector(".top").innerHTML = formatProblemStatement(problemData.problem_statements);
      // Render bar chart for cluster counters in bottom-left
      if (problemData.counters && Array.isArray(problemData.counters)) {
        const ctx = document.getElementById("barChart").getContext("2d");
        if (window.barChartInstance) {
          window.barChartInstance.destroy();
        }
        window.barChartInstance = new Chart(ctx, {
          type: "bar",
          data: {
            labels: problemData.counters.map((_, i) => `Cluster ${i+1}`),
            datasets: [{
              label: "Cluster Size",
              data: problemData.counters,
              backgroundColor: "#007bff",
              borderColor: "#0056b3",
              borderWidth: 1,
              borderRadius: 6
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: { display: false },
              title: { display: false }
            },
            scales: {
              x: {
                title: { display: true, text: "Cluster" },
                grid: { display: false }
              },
              y: {
                beginAtZero: true,
                title: { display: true, text: "Count" },
                grid: { display: true, color: "#e0e0e0" }
              }
            }
          }
        });
      }
    })
    .catch(err => {
      console.error("Error loading data", err);
      document.querySelector(".top").innerText = "Failed to load insights.";
    });
}
