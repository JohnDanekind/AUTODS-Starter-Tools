# tools/visualization.py
from langchain_core.tools import tool, BaseTool
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os



class VisualizationTools:
    # Constructor for dataframe and output directory
    def __init__(self, df: Optional[pd.DataFrame] = None, output_dir: str = "plots"):
        self.df = df
        self.output_dir = output_dir
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def set_dataframe(self, df: pd.DataFrame) -> None:
        """Set the dataframe to use for visualizations"""
        self.df = df

    def create_bar_chart(self, x_column: str, y_column: str, title: str = "Bar Chart") -> str:
        """
        Create a bar chart from the dataframe columns.

        Args:
            x_column: Column name to use for x-axis
            y_column: Column name to use for y-axis
            title: Title of the chart

        Returns:
            Path to the saved chart image
        """
        if self.df is None:
            return "Error: No DataFrame loaded"

        if x_column not in self.df.columns:
            return f"Error: Column '{x_column}' not found in DataFrame"

        if y_column not in self.df.columns:
            return f"Error: Column '{y_column}' not found in DataFrame"

        # Create the bar chart
        fig = px.bar(self.df, x=x_column, y=y_column, title=title)

        # Save the figure
        html_filename = f"{self.output_dir}/{x_column}_{y_column}_bar.html"
        png_filename = f"{self.output_dir}/{x_column}_{y_column}_bar.png"

        fig.write_html(html_filename)
        fig.write_image(png_filename)

        return f"Bar chart created and saved to {html_filename} (interactive) and {png_filename} (static)"

    def create_scatter_plot(self, x_column: str, y_column: str,
                            color_column: Optional[str] = None,
                            title: str = "Scatter Plot") -> str:
        """
        Create a scatter plot from the dataframe columns.

        Args:
            x_column: Column name to use for x-axis
            y_column: Column name to use for y-axis
            color_column: Optional column name to use for coloring points
            title: Title of the chart

        Returns:
            Path to the saved chart image
        """
        if self.df is None:
            return "Error: No DataFrame loaded"

        if x_column not in self.df.columns:
            return f"Error: Column '{x_column}' not found in DataFrame"

        if y_column not in self.df.columns:
            return f"Error: Column '{y_column}' not found in DataFrame"

        if color_column is not None and color_column not in self.df.columns:
            return f"Error: Column '{color_column}' not found in DataFrame"

        # Create the scatter plot
        fig = px.scatter(self.df, x=x_column, y=y_column,
                         color=color_column, title=title)

        # Save the figure
        filename_html = f"{self.output_dir}/{x_column}_{y_column}_scatter.html"
        filename_png = f"{self.output_dir}/{x_column}_{y_column}_scatter.png"
        fig.write_image(filename_png)
        fig.write_html(filename_html)

        return f"Scatter plot created and saved to {filename_html} (interactive) and {filename_png} (static)"

    def create_histogram(self, column_name: str, bins: int = 20,
                         title: str = None, color: str = "blue") -> str:
        """
        Create a histogram from a specified column of the DataFrame

        Args:
             column_name: The name of the column to create the histogram for
             bins: Number of bins for the histogram (default: 20)
             title: Title of the chart (default: auto-generated based on the column name)
             color: Color of the histogram bars (default: blue)

        Returns
            Path to the saved histogram images
        """
        # Check if DataFrame has been loaded
        if self.df is None:
            return "Error: No DataFrame loaded"

        # Check if the specified column name is in the DataFrame
        if column_name not in self.df.columns:
            return f"Error: Column '{column_name}' not found in DataFrame"

        # Check if the column with the specified column name is numeric
        if not pd.api.types.is_numeric_dtype(self.df[column_name]):
            return f"Error: Column '{column_name}' is not numeric. Histograms require numeric data."

        # Create the histogram title if not provided
        if title is None:
            title = f"Distribution of {column_name}"

        # Create the histogram
        fig = px.histogram(
            self.df,
            x=column_name,
            nbins=bins,
            title=title,
            color_discrete_sequence=[color]
        )

        # Add customization
        fig.update_layout(xaxis_title=column_name,
                          yaxis_title="Count",
                          bargap=0.5
                          )

        # Save the file to both html and png
        filename_html = f"{self.output_dir}/{column_name}_histogram.html"
        filename_png = f"{self.output_dir}/{column_name}_histogram.png"
        fig.write_image(filename_png)
        fig.write_html(filename_html)

        return f"Histogram plot created and saved to {filename_html} (interactive) and {filename_png} (static)"


    # Convert methods to tools
    def get_tools(self) -> List[BaseTool]:
        return [
            tool(self.create_bar_chart),
            tool(self.create_scatter_plot),
            tool(self.create_histogram)
        ]


