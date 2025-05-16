import pandas as pd
from rich.console import Console
from rich.table import Table
from typing import Dict, Optional, List, Any
import numpy as np

def display(
    df: pd.DataFrame,
    decimal_places: int = 4,
    num_rows: int = 5,
    highlight_map: Optional[Dict] = None,
    show_index: bool = False
) -> None:
    """Display a pandas DataFrame as a rich formatted table.
    
    Args:
        df: The pandas DataFrame to display.
        decimal_places: Maximum number of decimal places for float values.
        num_rows: Maximum number of rows to display. If None, shows all rows.
        highlight_map: Dictionary specifying highlighting rules for columns.
            Format is {column_key: {'max': 'color', 'min': 'color'}}
            Column keys can be integers, strings, or column names.
        show_index: Whether to display the DataFrame index. Defaults to True.
    """
    # Create a table
    table = Table(show_header=True, header_style="bold")
    
    # Add index column if requested
    if show_index:
        index_name = df.index.name or "Index"
        table.add_column(index_name, style="dim")
    
    # Add data columns
    for column in df.columns:
        table.add_column(str(column))
    
    # Normalize highlight_map keys to strings for consistent handling
    normalized_highlight_map = {}
    if highlight_map:
        for key, value in highlight_map.items():
            normalized_highlight_map[str(key)] = value
    
    # Calculate min/max for columns in highlight_map
    column_stats = {}
    if normalized_highlight_map:
        for col_str in normalized_highlight_map:
            # Try to match the column by string representation
            for col in df.columns:
                if str(col) == col_str:
                    try:
                        column_stats[col_str] = {
                            'min': df[col].min(),
                            'max': df[col].max()
                        }
                    except:
                        pass
    
    # Determine number of rows to display
    if num_rows is None or num_rows > len(df):
        num_rows = len(df)
    
    # Add rows
    for idx, (row_idx, row) in enumerate(df.iterrows()):
        if idx >= num_rows:
            break
            
        # Start with index if requested
        row_values = []
        if show_index:
            row_values.append(str(row_idx))
        
        # Format each value in the row
        for col in df.columns:
            val = row[col]
            
            # Format float values
            if isinstance(val, float):
                formatted_val = f"{val:.{decimal_places}f}"
            else:
                formatted_val = str(val)
            
            # Apply highlighting if specified
            col_str = str(col)
            if col_str in normalized_highlight_map and col_str in column_stats:
                # Check for min value
                if 'min' in normalized_highlight_map[col_str] and np.isclose(val, column_stats[col_str]['min']):
                    formatted_val = f"[{normalized_highlight_map[col_str]['min']}]{formatted_val}[/]"
                # Check for max value
                elif 'max' in normalized_highlight_map[col_str] and np.isclose(val, column_stats[col_str]['max']):
                    formatted_val = f"[{normalized_highlight_map[col_str]['max']}]{formatted_val}[/]"
            
            row_values.append(formatted_val)
        
        table.add_row(*row_values)
    
    # Add a footer if not all rows are displayed
    if num_rows < len(df):
        footer_cells = [""] * (len(df.columns) + (1 if show_index else 0))
        footer_cells[0] = f"[dim]Showing {num_rows} of {len(df)} rows[/]"
        table.add_row(*footer_cells)
    
    # Display the table
    console = Console()
    console.print(table)

def compute_pricing_error(df: pd.DataFrame) -> float:
    """
    Computes the Root Mean Square Error (RMSE) between Black-Scholes and Monte Carlo prices.
    
    Args:
        df: DataFrame containing 'bs_price' and 'mc_price' columns
        
    Returns:
        float: RMSE of the pricing differences
    """
    squared_diff = (df["bs_price"] - df["mc_price"]) ** 2
    mean_squared_error = squared_diff.mean()
    rmse = np.sqrt(mean_squared_error)
    return rmse