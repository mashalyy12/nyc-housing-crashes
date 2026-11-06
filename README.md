# NYC Housing & Traffic Crashes

**Data Engineering and Visualization – Milestone 1**  
German International University (GIU) – Winter 2026

## Project Overview

This project investigates the relationship between affordable housing development and motor vehicle crashes across New York City.

By combining NYC affordable housing production data with motor vehicle collision records, we explore how housing development relates to traffic crashes, injuries, and fatalities across different ZIP codes and boroughs.

## Live Dashboard

**[Open the NYC Housing & Traffic Crashes Dashboard](https://nyc-housing-crashes-toxi.vercel.app/)**

The interactive dashboard provides visualizations, filtering and search functionality, and CSV report generation.

## Research Questions & Visualizations

The project addresses eight research questions:

### Housing Analysis – Hagar Osama

1. **How are affordable housing units distributed across NYC boroughs?**  
   Visualization: Affordable Housing Units by Borough.

2. **How are affordable housing units distributed across different income levels and boroughs?**  
   Visualization: Affordable Housing Units by Income Level and Borough.

### Traffic Crash Analysis – Mariam Hatem

3. **Which NYC borough has the highest number of reported motor vehicle collisions?**  
   Visualization: Motor Vehicle Crashes by Borough.

4. **At what time of day do motor vehicle collisions occur most frequently?**  
   Visualization: Motor Vehicle Crashes by Hour of Day.

### Integrated Housing & Traffic Analysis – Marwan Moataz

5. **How do affordable housing units relate to motor vehicle crashes across matched ZIP codes?**  
   Visualization: Housing Units vs. Crashes by Matched ZIP Code.

6. **How do affordable housing units and motor vehicle crashes compare across NYC boroughs?**  
   Visualization: Housing Units and Motor Vehicle Crashes by Borough.

### Additional Integrated Analysis – Yassin Mashaly

7. **How do affordable housing units relate to traffic injuries across matched NYC ZIP codes?**  
   Visualization: Affordable Housing Units vs. Traffic Injuries Across NYC ZIP Codes.

8. **How do affordable housing units relate to traffic fatalities across matched NYC ZIP codes?**  
   Visualization: Affordable Housing Units vs. Traffic Fatalities Across NYC ZIP Codes.

*Note: These analyses explore associations and distributions; they do not establish that housing development causes traffic crashes or injuries.*

## Datasets

The project uses two publicly available datasets from NYC Open Data.

**1. Affordable Housing Production by Building**

- Dataset ID: `hg8x-zxpr`
- [NYC Open Data – Housing Dataset](https://data.cityofnewyork.us/d/hg8x-zxpr)

**2. Motor Vehicle Collisions – Crashes**

- Dataset ID: `h9gi-nx95`
- [NYC Open Data – Traffic Crashes Dataset](https://data.cityofnewyork.us/d/h9gi-nx95)

## Data Engineering & Integration

The datasets were cleaned and preprocessed before analysis.

The integration process included:

- Cleaning and standardizing geographic fields.
- Aggregating housing data by borough and ZIP code.
- Aggregating traffic crash data by borough and ZIP code.
- Combining both datasets using geographic identifiers.
- Using an outer join to retain records from both datasets.
- Filling missing numerical aggregate values with zero where appropriate.
- Identifying matched and unmatched geographic records.

The resulting integrated dataset contains housing and traffic statistics for geographic comparisons.

## Interactive Dashboard Features

The dashboard includes:

- Eight interactive visualizations.
- Housing and traffic analysis sections.
- Integrated geographic comparisons.
- Dashboard filtering and search functionality.
- KPI summaries.
- CSV report generation.

## Team Members & Contributions

1. **Hagar Osama:** Housing dataset cleaning and preprocessing, housing analysis, and housing visualizations (Q1, Q2).

2. **Mariam Hatem:** Traffic crashes dataset cleaning and preprocessing, crash analysis, and traffic visualizations (Q3, Q4).

3. **Marwan Moataz:** Integration of housing and traffic datasets, geographic analysis, and combined visualizations (Q5, Q6).

4. **Yassin Mashaly:** Additional integrated analysis and visualizations (Q7, Q8), interactive dashboard development, dashboard filters and search functionality, CSV report generation, and project documentation.

## Technologies Used

- Python
- Pandas
- NumPy
- Plotly
- Dash
- PyArrow
- Git and GitHub
- Vercel

## Project Setup

### 1. Clone the repository

```bash
git clone https://github.com/mashalyy12/nyc-housing-crashes.git
```

### 2. Navigate to the project directory

```bash
cd nyc-housing-crashes
```

### 3. Install the required dependencies

```bash
py -m pip install -r requirements.txt
```

### 4. Run the dashboard

```bash
py dashboard.py
```

### 5. Open the dashboard

Visit:

```text
http://127.0.0.1:8050/
```

## Deployment

The interactive dashboard is deployed using Vercel.

**Live URL:**  
https://nyc-housing-crashes-toxi.vercel.app/

**GitHub Repository:**  
https://github.com/mashalyy12/nyc-housing-crashes

---

