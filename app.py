
from pathlib import Path

import pandas as pd
import plotly.express as px

from dash import Dash, dcc, html, Input, Output


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

HOUSING_PATH = DATA_DIR / "housing.csv"
CRASHES_PATH = DATA_DIR / "crashes.csv"
INTEGRATED_PATH = DATA_DIR / "integrated_clean.csv"


# --------------------------------------------------
# 2. CHECK DATA FILES
# --------------------------------------------------

for file_path in [HOUSING_PATH, CRASHES_PATH, INTEGRATED_PATH]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required dataset not found: {file_path}"
        )


# --------------------------------------------------
# 3. LOAD HOUSING DATA
# --------------------------------------------------

housing = pd.read_csv(
    HOUSING_PATH,
    usecols=["Borough", "All Counted Units"]
)

housing["BOROUGH_KEY"] = (
    housing["Borough"]
    .astype("string")
    .str.strip()
    .str.upper()
)

housing["All Counted Units"] = pd.to_numeric(
    housing["All Counted Units"],
    errors="coerce"
)

housing_summary = (
    housing.dropna(subset=["BOROUGH_KEY"])
    .groupby("BOROUGH_KEY", as_index=False)
    .agg(
        Housing_Units=(
            "All Counted Units",
            lambda x: x.sum(min_count=1)
        )
    )
)


# --------------------------------------------------
# 4. LOAD AND SUMMARIZE CRASH DATA
#    Read in chunks to reduce memory usage.
# --------------------------------------------------

borough_parts = []
hour_parts = []

CRASH_COLUMNS = [
    "BOROUGH",
    "CRASH TIME",
    "NUMBER OF PERSONS INJURED",
    "NUMBER OF PERSONS KILLED"
]

for chunk in pd.read_csv(
    CRASHES_PATH,
    usecols=CRASH_COLUMNS,
    chunksize=150_000,
    low_memory=False
):

    chunk["BOROUGH_DASH"] = (
        chunk["BOROUGH"]
        .astype("string")
        .str.strip()
        .str.upper()
        .fillna("UNKNOWN")
    )

    chunk["INJURED_N"] = pd.to_numeric(
        chunk["NUMBER OF PERSONS INJURED"],
        errors="coerce"
    )

    chunk["KILLED_N"] = pd.to_numeric(
        chunk["NUMBER OF PERSONS KILLED"],
        errors="coerce"
    )

    chunk["CRASH HOUR"] = pd.to_datetime(
        chunk["CRASH TIME"].astype("string").str.strip(),
        format="%H:%M",
        errors="coerce"
    ).dt.hour

    # Crashes and injury totals by borough
    borough_piece = (
        chunk.groupby("BOROUGH_DASH")
        .agg(
            Number_of_Crashes=("BOROUGH_DASH", "size"),
            Total_Injured=("INJURED_N", "sum"),
            Total_Killed=("KILLED_N", "sum")
        )
    )

    borough_parts.append(borough_piece)

    # Crashes by borough and hour
    hour_piece = (
        chunk.dropna(subset=["CRASH HOUR"])
        .groupby(["BOROUGH_DASH", "CRASH HOUR"])
        .size()
        .rename("Number_of_Crashes")
        .to_frame()
    )

    hour_parts.append(hour_piece)


crash_summary = (
    pd.concat(borough_parts)
    .groupby(level=0)
    .sum()
    .reset_index()
)

hourly_summary = (
    pd.concat(hour_parts)
    .groupby(level=[0, 1])
    .sum()
    .reset_index()
)


# --------------------------------------------------
# 5. LOAD INTEGRATED DATA
# --------------------------------------------------

integrated = pd.read_csv(INTEGRATED_PATH)

matched_data = integrated[
    integrated["MATCH_STATUS"] == "Matched"
].copy()

matched_data["Housing_Total_Units"] = pd.to_numeric(
    matched_data["Housing_Total_Units"],
    errors="coerce"
)

matched_data["Number_of_Crashes"] = pd.to_numeric(
    matched_data["Number_of_Crashes"],
    errors="coerce"
)

matched_data = matched_data.dropna(
    subset=["Housing_Total_Units", "Number_of_Crashes"]
)


# --------------------------------------------------
# 6. DASHBOARD SETUP
# --------------------------------------------------

app = Dash(__name__)
app.title = "NYC Housing and Crash Dashboard"


borough_options = [
    {"label": "All Boroughs", "value": "ALL"}
]

borough_options += [
    {"label": b.title(), "value": b}ks
    for b in sorted(housing_summary["BOROUGH_KEY"].unique())
]


def metric_card(title, component_id):
    return html.Div(
        [
            html.P(
                title,
                style={
                    "color": "#64748b",
                    "fontSize": "14px"
                }
            ),
            html.H2(
                id=component_id,
                children="Loading...",
                style={
                    "color": "#172554",
                    "margin": "0"
                }
            )
        ],
        style={
            "backgroundColor": "white",
            "padding": "20px",
            "borderRadius": "12px",
            "boxShadow": "0 2px 8px rgba(0,0,0,0.06)"
        }
    )


app.layout = html.Div(
    [

        # HEADER
        html.Div(
            [
                html.H1(
                    "NYC Housing & Traffic Dashboard",
                    style={"marginBottom": "8px"}
                ),
                html.P(
                    "Explore affordable housing and motor vehicle collisions "
                    "across New York City."
                )
            ],
            style={
                "backgroundColor": "#172554",
                "color": "white",
                "padding": "28px",
                "borderRadius": "14px",
                "marginBottom": "25px"
            }
        ),

        # FILTER
        html.Div(
            [
                html.Label(
                    "Select Borough",
                    style={
                        "fontWeight": "bold",
                        "marginBottom": "8px"
                    }
                ),
                dcc.Dropdown(
                    id="borough-filter",
                    options=borough_options,
                    value="ALL",
                    clearable=False
                )
            ],
            style={
                "maxWidth": "400px",
                "marginBottom": "25px"
            }
        ),

        # SUMMARY CARDS
        html.Div(
            [
                metric_card(
                    "Counted Affordable Housing Units",
                    "housing-kpi"
                ),
                metric_card(
                    "Reported Motor Vehicle Crashes",
                    "crashes-kpi"
                ),
                metric_card(
                    "Recorded Persons Injured",
                    "injuries-kpi"
                ),
                metric_card(
                    "Recorded Persons Killed",
                    "fatalities-kpi"
                )
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(210px, 1fr))",
                "gap": "16px",
                "marginBottom": "25px"
            }
        ),

        # CHARTS
        html.Div(
            [
                html.Div(
                    dcc.Graph(id="housing-chart"),
                    style={
                        "backgroundColor": "white",
                        "borderRadius": "12px",
                        "padding": "12px"
                    }
                ),

                html.Div(
                    dcc.Graph(id="crash-chart"),
                    style={
                        "backgroundColor": "white",
                        "borderRadius": "12px",
                        "padding": "12px"
                    }
                ),

                html.Div(
                    dcc.Graph(id="hour-chart"),
                    style={
                        "backgroundColor": "white",
                        "borderRadius": "12px",
                        "padding": "12px"
                    }
                ),

                html.Div(
                    dcc.Graph(id="integration-chart"),
                    style={
                        "backgroundColor": "white",
                        "borderRadius": "12px",
                        "padding": "12px"
                    }
                )
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(400px, 1fr))",
                "gap": "20px"
            }
        ),

        # FOOTNOTE
        html.P(
            "Source: NYC Housing and Development and NYC Motor Vehicle "
            "Collisions datasets. Crash totals cover the full available "
            "collision dataset. The postcode comparison includes only "
            "matched borough-postcode locations. Counts are not adjusted "
            "for population or traffic volume.",
            style={
                "color": "#64748b",
                "fontSize": "12px",
                "marginTop": "25px"
            }
        )

    ],
    style={
        "fontFamily": "Arial, sans-serif",
        "backgroundColor": "#f1f5f9",
        "minHeight": "100vh",
        "padding": "25px",
        "maxWidth": "1500px",
        "margin": "auto"
    }
)


# --------------------------------------------------
# 7. INTERACTIVE DASHBOARD
# --------------------------------------------------

@app.callback(
    Output("housing-chart", "figure"),
    Output("crash-chart", "figure"),
    Output("hour-chart", "figure"),
    Output("integration-chart", "figure"),
    Output("housing-kpi", "children"),
    Output("crashes-kpi", "children"),
    Output("injuries-kpi", "children"),
    Output("fatalities-kpi", "children"),
    Input("borough-filter", "value")
)

def update_dashboard(selected_borough):

    # Filter housing
    housing_view = housing_summary.copy()

    # Filter crashes
    crash_view = crash_summary.copy()

    # Filter hourly data
    hour_view = hourly_summary.copy()

    # Filter integrated data
    integrated_view = matched_data.copy()

    if selected_borough != "ALL":
        housing_view = housing_view[
            housing_view["BOROUGH_KEY"] == selected_borough
        ]

        crash_view = crash_view[
            crash_view["BOROUGH_DASH"] == selected_borough
        ]

        hour_view = hour_view[
            hour_view["BOROUGH_DASH"] == selected_borough
        ]

        integrated_view = integrated_view[
            integrated_view["BOROUGH_KEY"] == selected_borough
        ]

    # Summary numbers
    housing_total = housing_view["Housing_Units"].sum()

    crashes_total = crash_view["Number_of_Crashes"].sum()

    injuries_total = crash_view["Total_Injured"].sum()

    fatalities_total = crash_view["Total_Killed"].sum()

    # HOUSING CHART
    housing_fig = px.bar(
        housing_view,
        x="BOROUGH_KEY",
        y="Housing_Units",
        title="Affordable Housing Units by Borough",
        labels={
            "BOROUGH_KEY": "Borough",
            "Housing_Units": "Counted Housing Units"
        }
    )

    housing_fig.update_layout(template="plotly_white")

    # CRASH CHART
    crash_view = crash_view[
        crash_view["BOROUGH_DASH"] != "UNKNOWN"
    ]

    crash_fig = px.bar(
        crash_view,
        x="BOROUGH_DASH",
        y="Number_of_Crashes",
        title="Reported Collisions by Borough",
        labels={
            "BOROUGH_DASH": "Borough",
            "Number_of_Crashes": "Reported Crashes"
        }
    )

    crash_fig.update_layout(template="plotly_white")

    # HOURLY CHART
    hourly_data = (
        hour_view.groupby("CRASH HOUR")["Number_of_Crashes"]
        .sum()
        .reindex(range(24), fill_value=0)
        .rename_axis("Hour")
        .reset_index()
    )

    hour_fig = px.bar(
        hourly_data,
        x="Hour",
        y="Number_of_Crashes",
        title="Collisions by Hour of Day",
        labels={
            "Hour": "Hour (24-hour clock)",
            "Number_of_Crashes": "Reported Crashes"
        }
    )

    hour_fig.update_xaxes(tickmode="linear", dtick=1)
    hour_fig.update_layout(template="plotly_white")

    # INTEGRATED DATA CHART
    integration_fig = px.scatter(
        integrated_view,
        x="Housing_Total_Units",
        y="Number_of_Crashes",
        color="BOROUGH_KEY",
        hover_data=["ZIP_KEY"],
        title="Housing Units vs. Crashes by Matched ZIP Code",
        labels={
            "Housing_Total_Units": "Counted Housing Units",
            "Number_of_Crashes": "Reported Crashes",
            "BOROUGH_KEY": "Borough"
        }
    )

    integration_fig.update_layout(template="plotly_white")

    return (
        housing_fig,
        crash_fig,
        hour_fig,
        integration_fig,
        f"{housing_total:,.0f}",
        f"{crashes_total:,.0f}",
        f"{injuries_total:,.0f}",
        f"{fatalities_total:,.0f}"
    )


# --------------------------------------------------
# 8. RUN WEBSITE
# --------------------------------------------------

if __name__ == "__main__":
    print("Starting NYC Housing and Crash Dashboard...")
    app.run(debug=False, host="127.0.0.1", port=8050)
