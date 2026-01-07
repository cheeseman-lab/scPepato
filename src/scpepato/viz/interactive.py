"""Interactive embedding visualizer using Dash.

This module provides a Dash-based web application for exploring
single-cell embeddings with hover tooltips showing cell images.
"""

from pathlib import Path

import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from dash import Dash, Input, Output, dcc, html
except ImportError:
    Dash = None
    px = None
    go = None

from scpepato.viz.images import clear_image_cache, get_cell_composite


class EmbeddingVisualizer:
    """Interactive Dash application for exploring cell embeddings.

    Displays 2D embeddings (UMAP, PHATE, t-SNE) with:
    - Dropdown to select embedding type
    - Dropdown to select color variable
    - Scatter plot with points colored by perturbation
    - Hover tooltip showing composite cell image

    Args:
        embedding_df: DataFrame with embedding coordinates and metadata.
            Required columns: plate, well, tile, cell_bounds_0/1/2/3, gene_symbol_0
            Embedding columns: umap_1, umap_2, phate_1, phate_2, tsne_1, tsne_2
        images_dir: Path to directory containing tile images
        channels: List of channel names (for display)
        channel_colors: RGB colors for each channel in composite
        title: Application title
    """

    def __init__(
        self,
        embedding_df: pd.DataFrame,
        images_dir: str | Path,
        channels: list[str] | None = None,
        channel_colors: list[tuple[float, float, float]] | None = None,
        title: str = "Embedding Visualizer",
    ):
        if Dash is None:
            raise ImportError(
                "dash is required for interactive visualization. Install with: pip install dash"
            )

        self.df = embedding_df.copy()
        self.images_dir = Path(images_dir)
        self.channels = channels or ["DAPI", "Channel2"]
        self.channel_colors = channel_colors
        self.title = title

        # Detect available embeddings
        self.embedding_types = self._detect_embeddings()
        if not self.embedding_types:
            raise ValueError(
                "No embedding columns found. Expected: umap_1/2, phate_1/2, or tsne_1/2"
            )

        # Detect color options
        self.color_options = self._detect_color_options()

        # Get unique genes for highlight dropdown
        self.unique_genes = sorted(self.df["gene_symbol_0"].unique().tolist())

        # Create row index for hover lookup
        self.df = self.df.reset_index(drop=True)
        self.df["_row_idx"] = self.df.index

        # Initialize Dash app
        self.app = Dash(__name__)
        self._setup_layout()
        self._setup_callbacks()

    def _detect_embeddings(self) -> list[str]:
        """Detect which embedding types are available in the data."""
        available = []
        for emb_type in ["umap", "phate", "tsne"]:
            if f"{emb_type}_1" in self.df.columns and f"{emb_type}_2" in self.df.columns:
                available.append(emb_type)
        return available

    def _detect_color_options(self) -> list[str]:
        """Detect which columns can be used for coloring."""
        options = []
        candidate_cols = ["gene_symbol_0", "well", "plate", "class"]
        for col in candidate_cols:
            if col in self.df.columns:
                options.append(col)
        return options if options else ["gene_symbol_0"]

    def _setup_layout(self):
        """Set up the Dash layout."""
        self.app.layout = html.Div(
            [
                # Header
                html.H1(self.title, style={"textAlign": "center", "marginBottom": "20px"}),
                # Controls row
                html.Div(
                    [
                        # Embedding type dropdown
                        html.Div(
                            [
                                html.Label("Embedding:", style={"fontWeight": "bold"}),
                                dcc.Dropdown(
                                    id="embedding-dropdown",
                                    options=[
                                        {"label": emb.upper(), "value": emb}
                                        for emb in self.embedding_types
                                    ],
                                    value=self.embedding_types[0],
                                    clearable=False,
                                    style={"width": "150px"},
                                ),
                            ],
                            style={"display": "inline-block", "marginRight": "30px"},
                        ),
                        # Color by dropdown
                        html.Div(
                            [
                                html.Label("Color by:", style={"fontWeight": "bold"}),
                                dcc.Dropdown(
                                    id="color-dropdown",
                                    options=[
                                        {"label": col.replace("_", " ").title(), "value": col}
                                        for col in self.color_options
                                    ],
                                    value=self.color_options[0],
                                    clearable=False,
                                    style={"width": "180px"},
                                ),
                            ],
                            style={"display": "inline-block", "marginRight": "30px"},
                        ),
                        # Highlight gene dropdown
                        html.Div(
                            [
                                html.Label("Highlight:", style={"fontWeight": "bold"}),
                                dcc.Dropdown(
                                    id="highlight-dropdown",
                                    options=[{"label": "All genes", "value": ""}]
                                    + [
                                        {"label": gene, "value": gene} for gene in self.unique_genes
                                    ],
                                    value="",
                                    clearable=False,
                                    style={"width": "200px"},
                                    placeholder="Select gene to highlight",
                                ),
                            ],
                            style={"display": "inline-block", "marginRight": "30px"},
                        ),
                        # Cell count display
                        html.Div(
                            [
                                html.Label("Cells:", style={"fontWeight": "bold"}),
                                html.Span(
                                    f"{len(self.df):,}",
                                    style={"marginLeft": "10px", "fontSize": "16px"},
                                ),
                            ],
                            style={"display": "inline-block"},
                        ),
                    ],
                    style={
                        "display": "flex",
                        "justifyContent": "center",
                        "alignItems": "center",
                        "marginBottom": "20px",
                    },
                ),
                # Main scatter plot and cell image panel
                html.Div(
                    [
                        # Left: scatter plot
                        html.Div(
                            [
                                dcc.Graph(
                                    id="embedding-plot",
                                    style={"width": "100%", "height": "80vh"},
                                    config={
                                        "displayModeBar": True,
                                        "scrollZoom": True,
                                    },
                                ),
                            ],
                            style={
                                "width": "75%",
                                "display": "inline-block",
                                "verticalAlign": "top",
                            },
                        ),
                        # Right: cell image panel (click to show)
                        html.Div(
                            [
                                html.H4(
                                    "Cell Image",
                                    style={"textAlign": "center", "marginBottom": "10px"},
                                ),
                                html.P(
                                    "Click on a point to view cell",
                                    id="cell-info",
                                    style={"textAlign": "center", "color": "#666"},
                                ),
                                html.Div(id="cell-image-container", style={"textAlign": "center"}),
                            ],
                            style={
                                "width": "24%",
                                "display": "inline-block",
                                "verticalAlign": "top",
                                "padding": "20px",
                                "backgroundColor": "#f8f9fa",
                                "borderRadius": "8px",
                                "marginLeft": "1%",
                            },
                        ),
                    ],
                    style={"display": "flex"},
                ),
            ],
            style={"padding": "20px", "fontFamily": "Arial, sans-serif"},
        )

    def _setup_callbacks(self):
        """Set up Dash callbacks for interactivity."""

        @self.app.callback(
            Output("embedding-plot", "figure"),
            [
                Input("embedding-dropdown", "value"),
                Input("color-dropdown", "value"),
                Input("highlight-dropdown", "value"),
            ],
        )
        def update_plot(embedding_type, color_by, highlight_gene):
            """Update the scatter plot based on dropdown selections."""
            x_col = f"{embedding_type}_1"
            y_col = f"{embedding_type}_2"

            # Include sgRNA if available
            custom_cols = ["_row_idx", "gene_symbol_0", "well", "tile"]
            if "sgRNA_0" in self.df.columns:
                custom_cols.append("sgRNA_0")

            # If highlighting a specific gene, create two traces
            if highlight_gene:
                # Split data into highlighted and background
                mask = self.df["gene_symbol_0"] == highlight_gene
                df_highlight = self.df[mask]
                df_background = self.df[~mask]

                fig = go.Figure()

                # Background points (grey)
                fig.add_trace(
                    go.Scatter(
                        x=df_background[x_col],
                        y=df_background[y_col],
                        mode="markers",
                        marker=dict(size=8, color="lightgrey", opacity=0.4),
                        name="Other",
                        customdata=df_background[custom_cols].values,
                        hoverinfo="none",  # Hide tooltip but keep click detection
                    )
                )

                # Highlighted points (colored)
                fig.add_trace(
                    go.Scatter(
                        x=df_highlight[x_col],
                        y=df_highlight[y_col],
                        mode="markers",
                        marker=dict(size=12, color="red", opacity=0.9),
                        name=highlight_gene,
                        customdata=df_highlight[custom_cols].values,
                        hoverinfo="none",  # Hide tooltip but keep click detection
                    )
                )

                fig.update_layout(
                    title=dict(
                        text=f"{embedding_type.upper()} - {highlight_gene}",
                        x=0.5,
                        font=dict(size=18),
                    ),
                    showlegend=True,
                )
            else:
                # Standard colored scatter plot
                fig = px.scatter(
                    self.df,
                    x=x_col,
                    y=y_col,
                    color=color_by,
                    custom_data=custom_cols,
                    labels={
                        x_col: f"{embedding_type.upper()} 1",
                        y_col: f"{embedding_type.upper()} 2",
                    },
                )

                fig.update_traces(
                    marker=dict(size=10, opacity=0.7),
                    hoverinfo="none",  # Hide tooltip but keep click detection
                )

                fig.update_layout(
                    title=dict(
                        text=f"{embedding_type.upper()} Embedding", x=0.5, font=dict(size=18)
                    ),
                    legend=dict(
                        title=color_by.replace("_", " ").title(),
                        itemsizing="constant",
                    ),
                )

            # Common layout settings
            fig.update_layout(
                xaxis_title=f"{embedding_type.upper()} 1",
                yaxis_title=f"{embedding_type.upper()} 2",
                uirevision="constant",
                hovermode="closest",  # Required for click detection to work
                dragmode="pan",
            )

            return fig

        @self.app.callback(
            [Output("cell-info", "children"), Output("cell-image-container", "children")],
            [Input("embedding-plot", "clickData")],
        )
        def show_cell_image(click_data):
            """Show cell image when clicking on a point."""
            if click_data is None:
                return "Click on a point to view cell", None

            try:
                point = click_data["points"][0]
                row_idx = point["customdata"][0]
                gene = point["customdata"][1]
                well = point["customdata"][2]
                tile = point["customdata"][3]
                # sgRNA is optional (index 4 if present)
                sgrna = point["customdata"][4] if len(point["customdata"]) > 4 else None

                # Get cell data
                row = self.df.iloc[row_idx]
                bounds = (
                    int(row["cell_bounds_0"]),
                    int(row["cell_bounds_1"]),
                    int(row["cell_bounds_2"]),
                    int(row["cell_bounds_3"]),
                )

                # Get composite image
                img_base64 = get_cell_composite(
                    plate=row["plate"],
                    well=row["well"],
                    tile=row["tile"],
                    bounds=bounds,
                    images_dir=self.images_dir,
                    channel_colors=self.channel_colors,
                    padding=15,
                    output_size=(200, 200),
                )

                # Info text with sgRNA if available
                if sgrna:
                    info_text = f"Gene: {gene} | sgRNA: {sgrna} | Well: {well}"
                else:
                    info_text = f"Gene: {gene} | Well: {well} | Tile: {int(tile)}"

                # Image element
                image_element = html.Div(
                    [
                        html.Img(
                            src=img_base64,
                            style={
                                "width": "200px",
                                "height": "200px",
                                "borderRadius": "8px",
                                "border": "2px solid #ddd",
                            },
                        ),
                        html.P(
                            "DAPI (blue) + GLYCORNA (green)",
                            style={"marginTop": "10px", "fontSize": "12px", "color": "#666"},
                        ),
                    ]
                )

                return info_text, image_element

            except Exception as e:
                return f"Error: {e}", None

    def run(self, host: str = "0.0.0.0", port: int = 8050, debug: bool = True):
        """Run the Dash application.

        Args:
            host: Host address to bind to
            port: Port number
            debug: Enable debug mode with auto-reload
        """
        print("\nStarting Embedding Visualizer...")
        print(f"Data: {len(self.df):,} cells")
        print(f"Embeddings: {', '.join(e.upper() for e in self.embedding_types)}")
        print(f"Images: {self.images_dir}")
        print(f"\nOpen in browser: http://localhost:{port}")
        print("Press Ctrl+C to stop\n")

        # Clear image cache before starting
        clear_image_cache()

        self.app.run(host=host, port=port, debug=debug)


def launch_visualizer(
    data_path: str | Path,
    images_dir: str | Path,
    channels: list[str] | None = None,
    channel_colors: list[tuple[float, float, float]] | None = None,
    port: int = 8050,
    debug: bool = True,
):
    """Convenience function to launch the visualizer from a parquet file.

    Args:
        data_path: Path to parquet file with embeddings and metadata
        images_dir: Path to directory containing tile images
        channels: Channel names for display
        channel_colors: RGB colors for composite
        port: Port to run server on
        debug: Enable debug mode
    """
    print(f"Loading data from: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df):,} cells")

    viz = EmbeddingVisualizer(
        embedding_df=df,
        images_dir=images_dir,
        channels=channels,
        channel_colors=channel_colors,
    )

    viz.run(port=port, debug=debug)
