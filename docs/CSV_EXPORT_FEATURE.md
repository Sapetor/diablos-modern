# CSV Export Feature for Plot Data

## Overview

The CSV export feature allows users to export simulation plot data from Scope blocks to CSV files for post-processing in external tools like Excel, MATLAB, Python (pandas), R, etc.

## How to Use

### Step 1: Run a Simulation

1. Create your block diagram with one or more **Scope** blocks
2. Connect signals to the Scope blocks
3. Run the simulation
4. View the plots by clicking the **Show Plots** button

### Step 2: Export to CSV

1. The plot window will appear with all your scope signals
2. At the bottom of the plot window, click the **"Export to CSV..."** button
3. A dialog will appear showing all available Scope blocks

### Step 3: Select Scopes

1. Check/uncheck the scopes you want to include in the export
   - By default, all scopes are selected
   - Use **Select All** or **Deselect All** buttons for convenience
2. Click **Export** when ready

### Step 4: Save File

1. Choose a location and filename for your CSV file
2. The default filename includes a timestamp: `plot_data_YYYYMMDD_HHMMSS.csv`
3. Click **Save**

### Step 5: Success

You'll see a confirmation dialog showing:
- The file path where data was saved
- Number of rows (time steps)
- Number of columns (signals + time)

## CSV Format

The exported CSV file uses a **wide format** with:
- First column: `time` (simulation time steps)
- Remaining columns: Signal data with headers from scope labels

### Example CSV Structure

```csv
time,scope0-signal1,scope0-signal2,scope1-signal1
0.00,1.23,4.56,7.89
0.01,1.25,4.58,7.91
0.02,1.27,4.60,7.93
...
```

### Multi-Signal Scopes

If a Scope block has multiple signals (multi-dimensional input), each signal gets its own column:

```csv
time,sine,cosine,tangent,square
0.00,0.000,1.000,0.000,0.0
0.01,0.100,0.995,0.101,0.0
0.02,0.199,0.980,0.203,0.0
...
```

## Use Cases

### Data Analysis
- Import into pandas for statistical analysis
- Plot in MATLAB or GNU Octave
- Analyze in R or Julia

### Reporting
- Import into Excel for creating reports
- Generate publication-quality plots in Python/matplotlib
- Share data with colleagues who don't have DiaBloS

### Verification
- Compare simulation results with analytical solutions
- Validate against experimental data
- Cross-check with other simulation tools

## Technical Details

### Implementation
- Location: `lib/lib.py` - `SignalPlot` class
- Method: `export_to_csv()`
- Dependencies: Python's built-in `csv` module, PyQt5 dialogs

### Data Handling
- Automatically handles multi-dimensional scope signals
- Preserves signal labels from scope parameters
- Handles mismatched data lengths gracefully
- Uses NumPy arrays for efficient data processing

### File Naming
- Default format: `plot_data_YYYYMMDD_HHMMSS.csv`
- Timestamp ensures unique filenames
- User can override with custom name

## Testing

A test script is available to verify the export functionality:

```bash
python test_csv_export.py
```

This creates sample data with multiple scope types and allows testing the export dialog and CSV generation.

## Troubleshooting

### "No Data" Warning
**Problem:** "No plot data available to export" message appears.
**Solution:** Run a simulation first and ensure plots are displayed before attempting export.

### "No Selection" Warning
**Problem:** Export dialog closes with "Please select at least one scope" message.
**Solution:** Check at least one scope checkbox before clicking Export.

### Empty or Malformed CSV
**Problem:** CSV file is created but contains no data or incorrect data.
**Solution:** Check the application log file for error messages. Ensure simulation completed successfully.

### Multi-dimensional Data Issues
**Problem:** Multi-signal scopes don't export correctly.
**Solution:** Ensure scope labels match the number of signals. Default labels will be generated if needed.

## Future Enhancements

Potential improvements for future versions:
- Export to other formats (JSON, MAT, HDF5)
- Include simulation parameters in exported file
- Export metadata (block diagram structure, parameters)
- Batch export all simulations in a session
- Export directly from canvas without showing plots
