
from pathlib import Path
import os

import pandas as pd
import pyarrow.parquet as pq
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dash import Dash, dcc, html, Input, Output, State


# ==================================================
# 1. PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

HOUSING_PATH = DATA_DIR / "housing_clean.csv"
CRASHES_PATH = DATA_DIR / "crashes_clean.parquet"
INTEGRATED_PATH = DATA_DIR / "integrated_clean_updated.csv"


# ==================================================
# 2. CHECK DATA FILES
# ==================================================

for file_path in [HOUSING_PATH, CRASHES_PATH, INTEGRATED_PATH]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required dataset not found: {file_path}"
        )


# ==================================================
# 3. LOAD AND PREPARE HOUSING DATA
# ==================================================

housing = pd.read_csv(HOUSING_PATH)

housing["BOROUGH_DASH"] = (
    housing["Borough"]
    .astype("string")
    .str.strip()
    .str.upper()
)

housing["YEAR"] = pd.to_datetime(
    housing["Project Start Date"],
    errors="coerce"
).dt.year

HOUSING_UNIT_COLUMNS = [
    "All Counted Units",
    "Extremely Low Income Units",
    "Very Low Income Units",
    "Low Income Units",
    "Moderate Income Units",
    "Middle Income Units",
    "Other Income Units",
]

INCOME_CATEGORY_COLUMNS = [
    "Extremely Low Income Units",
    "Very Low Income Units",
    "Low Income Units",
    "Moderate Income Units",
    "Middle Income Units",
    "Other Income Units",
]

for col in HOUSING_UNIT_COLUMNS:
    housing[col] = pd.to_numeric(
        housing[col],
        errors="coerce"
    ).fillna(0)

housing_long = housing.melt(
    id_vars=["BOROUGH_DASH", "YEAR"],
    value_vars=HOUSING_UNIT_COLUMNS,
    var_name="Income_Category",
    value_name="Units",
)

housing_summary = (
    housing_long
    .groupby(
        ["BOROUGH_DASH", "YEAR", "Income_Category"],
        as_index=False,
        dropna=False,
    )
    .agg(Housing_Units=("Units", "sum"))
)


# ==================================================
# 4. LOAD AND SUMMARIZE CRASH DATA
#    Read Parquet in batches to reduce memory usage.
# ==================================================

CRASH_COLUMNS = [
    "BOROUGH",
    "CRASH DATE",
    "CRASH TIME",
    "NUMBER OF PERSONS INJURED",
    "NUMBER OF PERSONS KILLED",
    "CONTRIBUTING FACTOR VEHICLE 1",
]

crash_parts = []
hour_parts = []

parquet_file = pq.ParquetFile(CRASHES_PATH)

for batch in parquet_file.iter_batches(
    batch_size=200_000,
    columns=CRASH_COLUMNS,
):
    chunk = batch.to_pandas()

    chunk["BOROUGH_DASH"] = (
        chunk["BOROUGH"]
        .astype("string")
        .str.strip()
        .str.upper()
        .fillna("UNKNOWN")
    )

    chunk["YEAR"] = pd.to_datetime(
        chunk["CRASH DATE"],
        errors="coerce"
    ).dt.year

    crash_time_text = (
        chunk["CRASH TIME"]
        .astype("string")
        .str.strip()
    )

    chunk["CRASH_HOUR"] = pd.to_numeric(
        crash_time_text,
        errors="coerce"
    )

    time_mask = chunk["CRASH_HOUR"].isna()

    if time_mask.any():
        parsed_time = pd.to_datetime(
            crash_time_text[time_mask],
            format="%H:%M",
            errors="coerce",
        )

        chunk.loc[time_mask, "CRASH_HOUR"] = (
            parsed_time.dt.hour
        )

    # Keep only valid hours from 0 to 23.
    chunk.loc[
        ~chunk["CRASH_HOUR"].between(0, 23),
        "CRASH_HOUR"
    ] = pd.NA

    chunk["INJURED_N"] = pd.to_numeric(
        chunk["NUMBER OF PERSONS INJURED"],
        errors="coerce"
    ).fillna(0)

    chunk["KILLED_N"] = pd.to_numeric(
        chunk["NUMBER OF PERSONS KILLED"],
        errors="coerce"
    ).fillna(0)

    chunk["CRASH_FACTOR"] = (
        chunk["CONTRIBUTING FACTOR VEHICLE 1"]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
        .fillna("Unspecified")
    )

    crash_piece = (
        chunk.groupby(
            ["BOROUGH_DASH", "YEAR", "CRASH_FACTOR"],
            dropna=False,
        )
        .agg(
            Number_of_Crashes=("BOROUGH_DASH", "size"),
            Total_Injured=("INJURED_N", "sum"),
            Total_Killed=("KILLED_N", "sum"),
        )
        .reset_index()
    )

    crash_parts.append(crash_piece)

    hour_piece = (
        chunk.dropna(subset=["CRASH_HOUR"])
        .groupby(
            [
                "BOROUGH_DASH",
                "YEAR",
                "CRASH_FACTOR",
                "CRASH_HOUR",
            ]
        )
        .size()
        .rename("Number_of_Crashes")
        .reset_index()
    )

    hour_parts.append(hour_piece)


crash_summary = (
    pd.concat(crash_parts, ignore_index=True)
    .groupby(
        ["BOROUGH_DASH", "YEAR", "CRASH_FACTOR"],
        as_index=False,
        dropna=False,
    )
    .agg(
        Number_of_Crashes=("Number_of_Crashes", "sum"),
        Total_Injured=("Total_Injured", "sum"),
        Total_Killed=("Total_Killed", "sum"),
    )
)

hourly_summary = (
    pd.concat(hour_parts, ignore_index=True)
    .groupby(
        [
            "BOROUGH_DASH",
            "YEAR",
            "CRASH_FACTOR",
            "CRASH_HOUR",
        ],
        as_index=False,
    )
    .agg(Number_of_Crashes=("Number_of_Crashes", "sum"))
)


# ==================================================
# 5. LOAD INTEGRATED DATA
# ==================================================

integrated = pd.read_csv(
    INTEGRATED_PATH,
    dtype={"Postcode": "string"}
)

integrated["Borough"] = (
    integrated["Borough"]
    .astype("string")
    .str.strip()
    .str.upper()
)

integrated["Postcode"] = (
    integrated["Postcode"]
    .astype("string")
    .str.strip()
)

for col in [
    "counted_housing_units",
    "total_crashes",
    "total_injured",
    "total_killed",
]:
    integrated[col] = pd.to_numeric(
        integrated[col],
        errors="coerce"
    ).fillna(0)

matched_data = integrated[
    integrated["match_status"]
    .astype("string")
    .str.strip()
    .str.lower()
    == "matched"
].copy()


# ==================================================
# 6. FILTER OPTIONS
# ==================================================

boroughs = sorted(
    housing_summary["BOROUGH_DASH"]
    .dropna()
    .unique()
)

borough_options = [
    {"label": "All Boroughs", "value": "ALL"}
]

borough_options += [
    {"label": borough.title(), "value": borough}
    for borough in boroughs
]

years = sorted(
    set(
        housing_summary["YEAR"]
        .dropna()
        .astype(int)
    )
    |
    set(
        crash_summary["YEAR"]
        .dropna()
        .astype(int)
    )
)

year_options = [
    {"label": "All Years", "value": "ALL"}
]

year_options += [
    {"label": str(year), "value": year}
    for year in years
]

housing_options = [
    {
        "label": "All Counted Units",
        "value": "All Counted Units"
    },
    {
        "label": "Extremely Low Income",
        "value": "Extremely Low Income Units"
    },
    {
        "label": "Very Low Income",
        "value": "Very Low Income Units"
    },
    {
        "label": "Low Income",
        "value": "Low Income Units"
    },
    {
        "label": "Moderate Income",
        "value": "Moderate Income Units"
    },
    {
        "label": "Middle Income",
        "value": "Middle Income Units"
    },
    {
        "label": "Other Income",
        "value": "Other Income Units"
    },
]

top_factors = (
    crash_summary
    .groupby("CRASH_FACTOR")["Number_of_Crashes"]
    .sum()
    .sort_values(ascending=False)
    .head(12)
    .index.tolist()
)

crash_options = [
    {
        "label": "All Contributing Factors",
        "value": "ALL"
    }
]

crash_options += [
    {"label": factor, "value": factor}
    for factor in top_factors
]


# ==================================================
# 7. DASHBOARD SETUP
# ==================================================

app = Dash(__name__)
server = app.server

app.title = "NYC Housing and Crash Dashboard"


def metric_card(title, component_id):
    return html.Div(
        [
            html.P(
                title,
                style={
                    "color": "#64748b",
                    "fontSize": "14px",
                },
            ),

            html.H2(
                id=component_id,
                children="Loading...",
                style={
                    "color": "#172554",
                    "margin": "0",
                },
            ),
        ],
        style={
            "backgroundColor": "white",
            "padding": "20px",
            "borderRadius": "12px",
            "boxShadow": "0 2px 8px rgba(0,0,0,0.06)",
        },
    )


def chart_card(graph_id):
    return html.Div(
        dcc.Graph(id=graph_id),
        style={
            "backgroundColor": "white",
            "borderRadius": "12px",
            "padding": "12px",
            "boxShadow": "0 2px 8px rgba(0,0,0,0.04)",
        },
    )


def empty_figure(title):
    fig = go.Figure()

    fig.update_layout(
        title=title,
        template="plotly_white",
        annotations=[
            {
                "text": "No data available for the selected filters",
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"size": 15},
            }
        ],
    )

    return fig


# ==================================================
# 8. DASHBOARD LAYOUT
# ==================================================

app.layout = html.Div(
    [
        html.Div(
            [
                html.H1(
                    "NYC Housing & Traffic Dashboard",
                    style={"marginBottom": "8px"},
                ),

                html.P(
                    "Explore affordable housing and motor vehicle "
                    "collisions across New York City."
                ),
            ],
            style={
                "backgroundColor": "#172554",
                "color": "white",
                "padding": "28px",
                "borderRadius": "14px",
                "marginBottom": "25px",
            },
        ),

        # SEARCH
        html.Div(
            [
                html.Label(
                    "Search Borough or ZIP Code",
                    style={"fontWeight": "bold"},
                ),

                dcc.Input(
                    id="search-input",
                    type="text",
                    placeholder="Enter borough or ZIP code...",
                    debounce=False,
                    style={
                        "width": "100%",
                        "padding": "12px",
                        "borderRadius": "8px",
                        "border": "1px solid #cbd5e1",
                        "marginTop": "8px",
                        "boxSizing": "border-box",
                    },
                ),

                html.Small(
                    "Search a borough name or a ZIP code, for example 10001.",
                    style={
                        "color": "#64748b",
                        "display": "block",
                        "marginTop": "6px",
                    },
                ),
            ],
            style={
                "maxWidth": "500px",
                "marginBottom": "20px",
            },
        ),

        # FILTERS
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Select Borough"),

                        dcc.Dropdown(
                            id="borough-filter",
                            options=borough_options,
                            value="ALL",
                            clearable=False,
                        ),
                    ]
                ),

                html.Div(
                    [
                        html.Label("Select Year"),

                        dcc.Dropdown(
                            id="year-filter",
                            options=year_options,
                            value="ALL",
                            clearable=False,
                        ),
                    ]
                ),

                html.Div(
                    [
                        html.Label("Housing Category"),

                        dcc.Dropdown(
                            id="housing-category-filter",
                            options=housing_options,
                            value="All Counted Units",
                            clearable=False,
                        ),
                    ]
                ),

                html.Div(
                    [
                        html.Label("Crash Category"),

                        dcc.Dropdown(
                            id="crash-category-filter",
                            options=crash_options,
                            value="ALL",
                            clearable=False,
                        ),
                    ]
                ),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": (
                    "repeat(auto-fit, minmax(210px, 1fr))"
                ),
                "gap": "16px",
                "marginBottom": "25px",
            },
        ),

        # KPI CARDS
        html.Div(
            [
                metric_card(
                    "Counted Housing Units",
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
                ),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": (
                    "repeat(auto-fit, minmax(210px, 1fr))"
                ),
                "gap": "16px",
                "marginBottom": "25px",
            },
        ),

        # HOUSING
        html.H2(
            "Housing Analysis",
            style={"color": "#172554"}
        ),

        html.Div(
            [
                chart_card("housing-chart"),
                chart_card("income-chart"),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": (
                    "repeat(auto-fit, minmax(400px, 1fr))"
                ),
                "gap": "20px",
                "marginBottom": "25px",
            },
        ),

        # TRAFFIC
        html.H2(
            "Traffic Crash Analysis",
            style={"color": "#172554"}
        ),

        html.Div(
            [
                chart_card("crash-chart"),
                chart_card("hour-chart"),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": (
                    "repeat(auto-fit, minmax(400px, 1fr))"
                ),
                "gap": "20px",
                "marginBottom": "25px",
            },
        ),

        # INTEGRATED ANALYSIS
        html.H2(
            "Integrated Housing and Traffic Analysis",
            style={"color": "#172554"}
        ),

        html.Div(
            [
                chart_card("integration-chart"),
                chart_card("borough-integration-chart"),
                chart_card("injuries-chart"),
                chart_card("fatalities-chart"),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": (
                    "repeat(auto-fit, minmax(400px, 1fr))"
                ),
                "gap": "20px",
            },
        ),

        # REPORT
        html.Div(
            [
                html.Button(
                    "Generate Report (CSV)",
                    id="generate-report",
                    n_clicks=0,
                    style={
                        "backgroundColor": "#172554",
                        "color": "white",
                        "padding": "13px 24px",
                        "border": "none",
                        "borderRadius": "8px",
                        "fontWeight": "bold",
                        "cursor": "pointer",
                        "marginTop": "25px",
                    },
                ),

                dcc.Download(id="download-report"),
            ]
        ),

        # FOOTER
        html.P(
            "Sources: NYC Housing and Development and NYC Motor "
            "Vehicle Collisions datasets. Year filtering uses housing "
            "project start dates and crash dates. Integrated ZIP-code "
            "and borough comparisons use the full available period. "
            "ZIP-level scatter charts use geographic groups matched "
            "in both datasets. Counts are not adjusted for population "
            "or traffic volume; the charts show associations, not causation.",
            style={
                "color": "#64748b",
                "fontSize": "12px",
                "marginTop": "25px",
            },
        ),
    ],
    style={
        "fontFamily": "Arial, sans-serif",
        "backgroundColor": "#f1f5f9",
        "minHeight": "100vh",
        "padding": "25px",
        "maxWidth": "1500px",
        "margin": "auto",
    },
)


# ==================================================
# 9. FILTERING FUNCTION
# ==================================================

def get_filtered_data(
    selected_borough,
    selected_year,
    search_term,
    selected_housing_category,
    selected_crash_factor,
):

    housing_view = housing_summary.copy()

    housing_income_view = housing_summary[
        housing_summary["Income_Category"].isin(
            INCOME_CATEGORY_COLUMNS
        )
    ].copy()

    crash_view = crash_summary.copy()
    hour_view = hourly_summary.copy()

    integrated_view = matched_data.copy()
    borough_integrated_view = integrated.copy()

    # ------------------------------------------------
    # BOROUGH FILTER
    # ------------------------------------------------

    if selected_borough != "ALL":

        housing_view = housing_view[
            housing_view["BOROUGH_DASH"] == selected_borough
        ]

        housing_income_view = housing_income_view[
            housing_income_view["BOROUGH_DASH"] == selected_borough
        ]

        crash_view = crash_view[
            crash_view["BOROUGH_DASH"] == selected_borough
        ]

        hour_view = hour_view[
            hour_view["BOROUGH_DASH"] == selected_borough
        ]

        integrated_view = integrated_view[
            integrated_view["Borough"] == selected_borough
        ]

        borough_integrated_view = borough_integrated_view[
            borough_integrated_view["Borough"] == selected_borough
        ]

    # ------------------------------------------------
    # YEAR FILTER
    # Applies to original housing and crash datasets.
    # Integrated data remains full-period.
    # ------------------------------------------------

    if selected_year != "ALL":

        selected_year = int(selected_year)

        housing_view = housing_view[
            housing_view["YEAR"] == selected_year
        ]

        housing_income_view = housing_income_view[
            housing_income_view["YEAR"] == selected_year
        ]

        crash_view = crash_view[
            crash_view["YEAR"] == selected_year
        ]

        hour_view = hour_view[
            hour_view["YEAR"] == selected_year
        ]

    # ------------------------------------------------
    # HOUSING CATEGORY FILTER
    # ------------------------------------------------

    housing_view = housing_view[
        housing_view["Income_Category"]
        == selected_housing_category
    ]

    # ------------------------------------------------
    # CRASH CATEGORY FILTER
    # ------------------------------------------------

    if selected_crash_factor != "ALL":

        crash_view = crash_view[
            crash_view["CRASH_FACTOR"]
            == selected_crash_factor
        ]

        hour_view = hour_view[
            hour_view["CRASH_FACTOR"]
            == selected_crash_factor
        ]

    # ------------------------------------------------
    # SEARCH: BOROUGH OR ZIP CODE
    # ------------------------------------------------

    search_term = str(
        search_term or ""
    ).strip().upper()

    if search_term:

        # Check if the search matches a borough.
        borough_matches = [
            borough
            for borough in boroughs
            if search_term in borough
        ]

        if borough_matches:

            housing_view = housing_view[
                housing_view["BOROUGH_DASH"].isin(
                    borough_matches
                )
            ]

            housing_income_view = housing_income_view[
                housing_income_view["BOROUGH_DASH"].isin(
                    borough_matches
                )
            ]

            crash_view = crash_view[
                crash_view["BOROUGH_DASH"].isin(
                    borough_matches
                )
            ]

            hour_view = hour_view[
                hour_view["BOROUGH_DASH"].isin(
                    borough_matches
                )
            ]

            integrated_view = integrated_view[
                integrated_view["Borough"].isin(
                    borough_matches
                )
            ]

            borough_integrated_view = borough_integrated_view[
                borough_integrated_view["Borough"].isin(
                    borough_matches
                )
            ]

        else:

            # Otherwise search the ZIP code.
            # String conversion protects against mixed data types.

            integrated_view = integrated_view[
                integrated_view["Postcode"]
                .astype("string")
                .str.contains(
                    search_term,
                    na=False,
                    regex=False,
                )
            ]

            borough_integrated_view = borough_integrated_view[
                borough_integrated_view["Postcode"]
                .astype("string")
                .str.contains(
                    search_term,
                    na=False,
                    regex=False,
                )
            ]

    return (
        housing_view,
        housing_income_view,
        crash_view,
        hour_view,
        integrated_view,
        borough_integrated_view,
    )


# ==================================================
# 10. INTERACTIVE DASHBOARD CALLBACK
# ==================================================

@app.callback(
    Output("housing-chart", "figure"),
    Output("income-chart", "figure"),
    Output("crash-chart", "figure"),
    Output("hour-chart", "figure"),
    Output("integration-chart", "figure"),
    Output("borough-integration-chart", "figure"),
    Output("injuries-chart", "figure"),
    Output("fatalities-chart", "figure"),

    Output("housing-kpi", "children"),
    Output("crashes-kpi", "children"),
    Output("injuries-kpi", "children"),
    Output("fatalities-kpi", "children"),

    Input("borough-filter", "value"),
    Input("year-filter", "value"),
    Input("search-input", "value"),
    Input("housing-category-filter", "value"),
    Input("crash-category-filter", "value"),
)
def update_dashboard(
    selected_borough,
    selected_year,
    search_term,
    selected_housing_category,
    selected_crash_factor,
):

    (
        housing_view,
        housing_income_view,
        crash_view,
        hour_view,
        integrated_view,
        borough_integrated_view,
    ) = get_filtered_data(
        selected_borough,
        selected_year,
        search_term,
        selected_housing_category,
        selected_crash_factor,
    )

    # ------------------------------------------------
    # KPI VALUES
    # ------------------------------------------------

    housing_total = housing_view["Housing_Units"].sum()

    crashes_total = crash_view[
        "Number_of_Crashes"
    ].sum()

    injuries_total = crash_view[
        "Total_Injured"
    ].sum()

    fatalities_total = crash_view[
        "Total_Killed"
    ].sum()

    # ------------------------------------------------
    # Q1: AFFORDABLE HOUSING UNITS BY BOROUGH
    # ------------------------------------------------

    housing_plot = (
        housing_view[
            housing_view["BOROUGH_DASH"] != "UNKNOWN"
        ]
        .groupby("BOROUGH_DASH", as_index=False)
        .agg(
            Housing_Units=("Housing_Units", "sum")
        )
    )

    if housing_plot.empty:

        housing_fig = empty_figure(
            "Affordable Housing Units by Borough"
        )

    else:

        housing_fig = px.bar(
            housing_plot,
            x="BOROUGH_DASH",
            y="Housing_Units",
            title=f"{selected_housing_category} by Borough",
            labels={
                "BOROUGH_DASH": "Borough",
                "Housing_Units": "Housing Units",
            },
            template="plotly_white",
        )

    # ------------------------------------------------
    # Q2: HOUSING BY INCOME LEVEL AND BOROUGH
    # ------------------------------------------------

    income_plot = (
        housing_income_view[
            housing_income_view["BOROUGH_DASH"] != "UNKNOWN"
        ]
        .groupby(
            ["BOROUGH_DASH", "Income_Category"],
            as_index=False,
        )
        .agg(
            Housing_Units=("Housing_Units", "sum")
        )
    )

    if income_plot.empty:

        income_fig = empty_figure(
            "Affordable Housing Units by Income Level and Borough"
        )

    else:

        income_fig = px.bar(
            income_plot,
            x="BOROUGH_DASH",
            y="Housing_Units",
            color="Income_Category",
            barmode="stack",
            category_orders={
                "Income_Category": INCOME_CATEGORY_COLUMNS
            },
            title=(
                "Affordable Housing Units by Income Level "
                "and Borough"
            ),
            labels={
                "BOROUGH_DASH": "NYC Borough",
                "Housing_Units": "Number of Housing Units",
                "Income_Category": "Income Category",
            },
            template="plotly_white",
        )

    # ------------------------------------------------
    # Q3: REPORTED CRASHES BY BOROUGH
    # ------------------------------------------------

    crash_plot = (
        crash_view[
            crash_view["BOROUGH_DASH"] != "UNKNOWN"
        ]
        .groupby("BOROUGH_DASH", as_index=False)
        .agg(
            Number_of_Crashes=("Number_of_Crashes", "sum")
        )
    )

    if crash_plot.empty:

        crash_fig = empty_figure(
            "Reported Collisions by Borough"
        )

    else:

        crash_fig = px.bar(
            crash_plot,
            x="BOROUGH_DASH",
            y="Number_of_Crashes",
            title="Reported Collisions by Borough",
            labels={
                "BOROUGH_DASH": "Borough",
                "Number_of_Crashes": "Reported Crashes",
            },
            template="plotly_white",
        )

    # ------------------------------------------------
    # Q4: CRASHES BY HOUR OF DAY
    # ------------------------------------------------

    hourly_data = (
        hour_view
        .groupby("CRASH_HOUR")["Number_of_Crashes"]
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
            "Number_of_Crashes": "Reported Crashes",
        },
        template="plotly_white",
    )

    hour_fig.update_xaxes(
        tickmode="linear",
        dtick=1
    )

    # ------------------------------------------------
    # Q5: HOUSING UNITS VS CRASHES BY ZIP CODE
    # ------------------------------------------------

    q5_data = integrated_view[
        integrated_view["counted_housing_units"] > 0
    ].copy()

    if q5_data.empty:

        integration_fig = empty_figure(
            "Housing Units vs. Motor Vehicle Crashes by ZIP Code"
        )

    else:

        integration_fig = px.scatter(
            q5_data,
            x="counted_housing_units",
            y="total_crashes",
            color="Borough",
            hover_data=[
                "Postcode",
                "housing_projects",
                "total_injured",
                "total_killed",
            ],
            title=(
                "Housing Units vs. Motor Vehicle Crashes "
                "by ZIP Code"
            ),
            labels={
                "counted_housing_units": "Counted Housing Units",
                "total_crashes": "Number of Crashes",
                "Borough": "NYC Borough",
            },
            template="plotly_white",
        )

        integration_fig.update_traces(
            marker=dict(size=10, opacity=0.7)
        )

    # ------------------------------------------------
    # Q6: HOUSING AND CRASHES BY BOROUGH
    # ------------------------------------------------

    q6_plot = (
        borough_integrated_view[
            borough_integrated_view["Borough"].notna()
            &
            (borough_integrated_view["Borough"] != "UNKNOWN")
        ]
        .groupby("Borough", as_index=False)
        .agg(
            total_housing_units=(
                "counted_housing_units",
                "sum"
            ),
            total_crashes=("total_crashes", "sum"),
            total_injured=("total_injured", "sum"),
            total_killed=("total_killed", "sum"),
        )
        .sort_values("Borough")
    )

    if q6_plot.empty:

        borough_integration_fig = empty_figure(
            "Housing Units and Motor Vehicle Crashes by NYC Borough"
        )

    else:

        borough_integration_fig = make_subplots(
            specs=[[{"secondary_y": True}]]
        )

        borough_integration_fig.add_trace(
            go.Bar(
                x=q6_plot["Borough"],
                y=q6_plot["total_housing_units"],
                name="Housing Units",
            ),
            secondary_y=False,
        )

        borough_integration_fig.add_trace(
            go.Scatter(
                x=q6_plot["Borough"],
                y=q6_plot["total_crashes"],
                name="Motor Vehicle Crashes",
                mode="lines+markers",
            ),
            secondary_y=True,
        )

        borough_integration_fig.update_layout(
            title=(
                "Housing Units and Motor Vehicle Crashes "
                "by NYC Borough"
            ),
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )

        borough_integration_fig.update_xaxes(
            title_text="NYC Borough"
        )

        borough_integration_fig.update_yaxes(
            title_text="Number of Housing Units",
            secondary_y=False,
        )

        borough_integration_fig.update_yaxes(
            title_text="Number of Motor Vehicle Crashes",
            secondary_y=True,
        )

    # ------------------------------------------------
    # Q7: HOUSING UNITS VS TRAFFIC INJURIES
    # ------------------------------------------------

    q7_data = integrated_view.copy()

    if q7_data.empty:

        injuries_fig = empty_figure(
            "Affordable Housing Units vs. Traffic Injuries Across NYC ZIP Codes"
        )

    else:

        injuries_fig = px.scatter(
            q7_data,
            x="counted_housing_units",
            y="total_injured",
            color="Borough",
            hover_data=[
                "Postcode",
                "total_crashes",
            ],
            title=(
                "Affordable Housing Units vs. Traffic Injuries "
                "Across NYC ZIP Codes"
            ),
            labels={
                "counted_housing_units": (
                    "Counted Affordable Housing Units"
                ),
                "total_injured": "Total Traffic-Related Injuries",
                "Borough": "NYC Borough",
            },
            template="plotly_white",
        )

    # ------------------------------------------------
    # Q8: HOUSING UNITS VS TRAFFIC FATALITIES
    # ------------------------------------------------

    q8_data = integrated_view.copy()

    if q8_data.empty:

        fatalities_fig = empty_figure(
            "Affordable Housing Units vs. Traffic Fatalities Across NYC ZIP Codes"
        )

    else:

        fatalities_fig = px.scatter(
            q8_data,
            x="counted_housing_units",
            y="total_killed",
            color="Borough",
            hover_data=[
                "Postcode",
                "total_crashes",
            ],
            title=(
                "Affordable Housing Units vs. Traffic Fatalities "
                "Across NYC ZIP Codes"
            ),
            labels={
                "counted_housing_units": (
                    "Counted Affordable Housing Units"
                ),
                "total_killed": "Total Traffic-Related Fatalities",
                "Borough": "NYC Borough",
            },
            template="plotly_white",
        )

    # ------------------------------------------------
    # COMMON CHART FORMATTING
    # ------------------------------------------------

    all_figures = [
        housing_fig,
        income_fig,
        crash_fig,
        hour_fig,
        integration_fig,
        borough_integration_fig,
        injuries_fig,
        fatalities_fig,
    ]

    for figure in all_figures:

        figure.update_layout(
            margin=dict(
                l=35,
                r=25,
                t=65,
                b=45,
            ),
            height=500,
        )

    # ------------------------------------------------
    # RETURN ALL CHARTS AND KPIs
    # ------------------------------------------------

    return (
        housing_fig,
        income_fig,
        crash_fig,
        hour_fig,
        integration_fig,
        borough_integration_fig,
        injuries_fig,
        fatalities_fig,

        f"{housing_total:,.0f}",
        f"{crashes_total:,.0f}",
        f"{injuries_total:,.0f}",
        f"{fatalities_total:,.0f}",
    )


# ==================================================
# 11. GENERATE CSV REPORT
# ==================================================

@app.callback(
    Output("download-report", "data"),

    Input("generate-report", "n_clicks"),

    State("borough-filter", "value"),
    State("year-filter", "value"),
    State("search-input", "value"),
    State("housing-category-filter", "value"),
    State("crash-category-filter", "value"),

    prevent_initial_call=True,
)
def generate_report(
    n_clicks,
    selected_borough,
    selected_year,
    search_term,
    selected_housing_category,
    selected_crash_factor,
):

    (
        housing_view,
        housing_income_view,
        crash_view,
        hour_view,
        report_data,
        borough_report_data,
    ) = get_filtered_data(
        selected_borough,
        selected_year,
        search_term,
        selected_housing_category,
        selected_crash_factor,
    )

    report_data = report_data.copy()

    report_data["Selected_Year_Filter"] = selected_year

    report_data["Selected_Housing_Category"] = (
        selected_housing_category
    )

    report_data["Selected_Crash_Category"] = (
        selected_crash_factor
    )

    return dcc.send_data_frame(
        report_data.to_csv,
        "NYC_Housing_Crashes_Report.csv",
        index=False,
    )


# ==================================================
# 12. RUN DASHBOARD
# ==================================================

if __name__ == "__main__":

    print("Starting NYC Housing and Crash Dashboard...")

    app.run(
        debug=False,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8050)),
    )
