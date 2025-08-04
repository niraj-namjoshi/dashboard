This project is an AI-powered Review Intelligence Dashboard designed to help companies automatically collect, analyze, and act upon customer feedback at scale. 
The system fetches reviews from multiple sources—such as app stores, e-commerce platforms, or social media—using secure OAuth integrations. 
Once ingested, reviews are processed through a Roberta-based sentiment analysis model, which classifies each comment as positive, neutral, or negative. 
To identify recurring themes and issues, the platform applies K-Means clustering, combining techniques like the silhouette score and the elbow method to determine the optimal number of clusters. 
Within each cluster, representative comments are selected by calculating similarity indices, ensuring that only the most relevant examples are passed to a Large Language Model (LLM) for summarization. 
The LLM then generates clear, concise problem descriptions and notifications, highlighting the most critical issues customers face. 
These insights are surfaced in an interactive frontend dashboard that includes sentiment gauges, problem analysis cards, and bar charts displaying the distribution of identified problems. 
The platform can also automatically report issues and bugs to company representatives, making it a comprehensive solution for continuous customer experience monitoring and improvement.

