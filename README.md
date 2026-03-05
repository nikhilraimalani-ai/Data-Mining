🏧 FinTrust ATM Intelligence Platform
A data mining dashboard that turns raw ATM transaction data into something actually useful.

What is it?
An interactive web app built with Streamlit that helps bank managers understand ATM cash
demand patterns. Instead of manually digging through spreadsheets trying to figure out which
ATM needs refilling and when, this app does all the heavy lifting visually.
Upload your transaction CSV and within seconds you have charts, cluster groups, anomaly flags,
and demand filters all in one place. It covers every stage of the FA-2 pipeline — exploratory
data analysis, K-Means clustering, anomaly detection, and an interactive demand planner — all
wrapped inside a dark-themed dashboard that actually looks good enough to present.
The idea behind it is simple: cash management at scale is hard. You have 50 ATMs spread across
different locations, different footfall patterns, weather conditions, holidays, events — and
someone has to decide how much cash goes where every single day. This app turns that guesswork
into data-driven decisions.

What's inside
🌐 Dashboard
The first thing you see when you open the app. It shows total withdrawals, average daily demand,
how many ATMs are being tracked, and how much more cash gets pulled on holidays vs normal days.
Below the KPIs there's a live alert system that automatically classifies every ATM into critical,
warning, or normal based on how its average demand compares to the rest of the network. If an
ATM is in the red, you know before it becomes a problem.

📊 EDA Explorer
Five tabs covering different angles of the data. Distribution histograms and box plots to
understand the shape of the data, time trend charts with a 7-day moving average, bar charts
showing how demand shifts by day of the week and time of day, holiday and event impact
comparisons, weather condition breakdowns, and a full correlation heatmap. Each chart has a
short observation underneath explaining what the pattern actually means in plain English.

🔵 Cluster Analysis
Groups all 50 ATMs into segments based on their withdrawal behavior, deposit levels, and
location type. Uses K-Means with an elbow method and silhouette score chart to help you pick
the right number of clusters. The result is a 3D scatter plot where you can rotate the cluster
space and see how ATMs group together. Segments get labeled meaningfully — high demand, steady
demand, low demand — so the output is interpretable, not just colored dots.

⚠️ Anomaly Radar
Detects unusual withdrawal spikes that don't fit normal patterns. Switch between three detection
methods — Z-Score, IQR, and Isolation Forest — depending on how sensitive you need the detection
to be. Anomalies get marked in red directly on the withdrawal timeline so you can see exactly
when they happened. There's also a breakdown by location type and a full table of every flagged
record with its anomaly score attached.

🎛️ Demand Planner
A filter-driven page for scenario planning. Filter by day of the week, time of day, location
type, and weather condition and the charts update live to show the demand picture for that
specific scenario. Good for answering things like "how much should we load into supermarket ATMs
on rainy Friday afternoons?" The top 10 ATMs for the filtered view are listed so you know exactly
which machines to prioritise.

🗺️ ATM Heatmap
Two maps — a density heatmap showing where demand concentrates geographically, and a bubble map
where each ATM is sized by its average withdrawal volume. Toggle between a dark and light map
style, and switch the color metric between average withdrawals, holiday rate, or total records.
