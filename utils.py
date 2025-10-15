'''
Utility functions to process, filter and visualize DTS and auxiliary data
'''

import pandas as pd
import numpy as np
import xarray as xr
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import sys
from matplotlib.lines import Line2D

def convert_ddf_to_monthly_csv(in_directory, out_directory, dates_from_filenames=True):
    """
    Converts .ddf files from a given directory into monthly CSV files.

    Parameters:
    - in_directory: str, path to the input directory containing .ddf files.
    - out_directory: str, path to the output directory where CSVs will be saved.
    - dates_from_filenames: bool, if True, extracts date from filenames, otherwise from file headers.
    """

    folder = os.path.join(in_directory, "*", "*", "**", "*.ddf")  # Include all years and months
    file_paths = glob.glob(folder, recursive=True)

    # Dictionary to store DataFrames grouped by (year, month)
    monthly_dfs = {}

    # Loop through each file path
    for file_path in file_paths:
        # Extract year and month from the file path
        parts = file_path.split(os.sep)
        year, month = parts[-3], parts[-2]  # Adjust based on folder structure

        # Read the .ddf file (skip header)
        df = pd.read_csv(file_path, encoding='latin-1', sep='\t', skiprows=25)

        if dates_from_filenames:
            # Extract time information from the file path
            parts = file_path.split(' ')
            time_frame = parts[-3] + ' ' + parts[-1].split('.')[0]  # Adjust indices as needed
        else:
            # Extract time from the file header
            with open(file_path, 'r', encoding='latin-1') as f:
                header_lines = [next(f).strip().split('\t') for _ in range(25)]
            
            header_dict = {line[0].strip(): line[1].strip() if len(line) > 1 else None for line in header_lines}
            date = header_dict.get('date', 'unknown').replace('/', '-')  # YYYY/MM/DD → YYYY-MM-DD
            time = header_dict.get('time', 'unknown')
            time_frame = f"{date} {time}"  # Format: YYYY-MM-DD HH:MM:SS

        # Add time_frame column
        df['time_frame'] = time_frame

        # Store in dictionary grouped by (year, month)
        key = (year, month)
        if key in monthly_dfs:
            monthly_dfs[key].append(df)
        else:
            monthly_dfs[key] = [df]

    # Ensure output directory exists
    os.makedirs(out_directory, exist_ok=True)

    # Process and save each month's data separately
    for (year, month), dfs in monthly_dfs.items():
        combined_df = pd.concat(dfs, ignore_index=True)
        output_filename = f"channel 1 dts {month} {year}.csv"
        output_path = os.path.join(out_directory, output_filename)
        
        combined_df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")

def parse_time_frame(time_frame):
    # Extract date and time parts
    date_part = time_frame.str.split().str[0]
    time_part = time_frame.str.split().str[1].str.extract(r'(\d{5})')[0]
    time_part_in_minutes = pd.to_timedelta(time_part.astype(int) * 30, unit='m')
    
    return pd.to_datetime(date_part, format='%Y%m%d') + time_part_in_minutes

def read_and_combine_dts_files(directory, dates_from_filenames=True):
    csv_files = glob.glob(os.path.join(directory, '*.csv'))

    # Initialize an empty list to store DataFrames
    dataframes = []

    # Process each CSV file
    for file in csv_files:
        # Read the CSV file

        if dates_from_filenames == True:
            df = pd.read_csv(file)

            # Extract and parse the time_frame
            df['time_frame'] = parse_time_frame(df['time_frame'])

        else:
            df = pd.read_csv(file)
            df['time_frame'] = pd.to_datetime(df['time_frame'], format="%Y-%m-%d %H:%M:%S", errors='coerce')

        # Filter the DataFrame
        df_filtered = df.loc[(df['length (m)'] > 60) & (df['length (m)'] < 1940)]
        df_filtered = df_filtered.loc[(df_filtered['temperature (°C)'] > -40) & (df_filtered['temperature (°C)'] < 30)]

        # Extract relevant columns
        df_filtered = df_filtered[['time_frame', 'length (m)', 'temperature (°C)']]
        
        # Append the filtered DataFrame to the list
        dataframes.append(df_filtered)

    # Concatenate all DataFrames into one
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Create the pivot table from the combined DataFrame, aggfunc takes mean if there is duplicates for time AND columns
    df_pivot = combined_df.pivot_table(index='time_frame', columns='length (m)', values='temperature (°C)', aggfunc='mean')

    # Compute the most common time difference (mode of time differences)
    time_diffs = df_pivot.index.to_series().diff().dropna()
    most_common_freq = time_diffs.mode()[0]  # Pick the most frequent difference

    # Generate a full time index using the detected frequency
    full_time_index = pd.date_range(start=df_pivot.index.min(), end=df_pivot.index.max(), freq=most_common_freq)

    # Reindex the DataFrame to include missing timestamps with NaN
    df_pivot = df_pivot.reindex(full_time_index)

    return df_pivot


def read_fmi_meteo_obs(filename, resample=None):
    meteo1 = pd.read_csv(filename)
    meteo1['time'] = pd.to_datetime(meteo1[['Vuosi', 'Kuukausi', 'Päivä']].astype(str).agg('-'.join, axis=1) + ' ' + meteo1['Aika [Paikallinen aika]'])
    meteo1.set_index('time', inplace=True)
    meteo1.drop(columns=['Vuosi', 'Kuukausi', 'Päivä', 'Aika [Paikallinen aika]', 'Havaintoasema'], inplace=True)
    meteo1 = meteo1.rename(columns={'Ilman lämpötila [°C]': 'Tair'})
    if resample:
        resampling_time = resample
        meteo1 = meteo1.resample(resampling_time).mean()
    return meteo1


def plot_2D_dts_colormap(xr_data, meteo_df, time_slice, x_slice, vmin=None, vmax=None, save_fp=None):

    # Filter meteo DataFrame to match the time slice
    meteo_filtered = meteo_df.loc[time_slice]
    time_len = len(meteo_filtered.index)

    if vmin == None:
        meteo_min = min(meteo_filtered.min())
        xr_min = np.nanmin(xr_data.sel(time=time_slice, x=x_slice)['T'].values)
        vmin = min([meteo_min, xr_min])
    if vmax == None:
        meteo_max = max(meteo_filtered.max())
        xr_max = np.nanmax(xr_data.sel(time=time_slice, x=x_slice)['T'].values)
        vmax = max([meteo_max, xr_max])

    # Create subplots with adjusted spacing using constrained_layout
    fig, axes = plt.subplots(1, 4, figsize=(16, 9), gridspec_kw={'width_ratios': [3, 0.1, 0.1, 0.1]})  # Adjusted width ratios

    # Process the xarray dataset to get temperature as a 2D numpy array
    stream_temp_2d = xr_data.sel(time=time_slice, x=x_slice)['T'].values  # Extract the temperature data as a 2D numpy array
    
    # Define the distance array based on the x_slice (assuming the x-coordinate corresponds to distance in meters)
    distance_along_stream = np.linspace(x_slice.start, x_slice.stop, len(stream_temp_2d[0]))
        
    # Plot the temperature along the stream as a 2D array
    cax = axes[0].imshow(
        stream_temp_2d,  # Use the temperature 2D array
        aspect='auto',  
        cmap='RdYlBu_r', vmin=vmin, vmax=vmax,
        extent=[distance_along_stream.min(), distance_along_stream.max(), mdates.date2num(meteo_filtered.index.max()), mdates.date2num(meteo_filtered.index.min())]  # Adjusted to place time on y-axis
    )

    # Set x-axis ticks and labels based on distance
    axes[0].set_xticks(np.linspace(distance_along_stream.min(), distance_along_stream.max(), num=6))  # Set ticks at 6 intervals
    axes[0].set_xticklabels([f"{x:.0f}" for x in np.linspace(distance_along_stream.min(), distance_along_stream.max(), num=6)])  # Label the ticks with distance (m)

    # Set y-axis with date and time formatting
    axes[0].set_yticks(np.linspace(mdates.date2num(meteo_filtered.index.min()), mdates.date2num(meteo_filtered.index.max()), num=len(meteo_filtered.index)//12))  # Set y-ticks for the time axis

    meteo_freq = meteo_df.index.freq
    if meteo_freq == '1D':
        freq = '1D'
    else:
        # Manually format y-tick labels
        if time_len <= 48:
            freq='1H'
        if (time_len > 48) & (time_len <= 96):
            freq='3H'
        if (time_len > 96) & (time_len <= 336):
            freq='6H'
        if (time_len > 336) & (time_len <= 1500):
            freq='1D'
        if (time_len > 1500) & (time_len <= 4000):
            freq='3D'
        if (time_len > 4000) & (time_len <= 10000):
            freq='7D'
        if time_len > 10000:
            freq='1M'
    
    time_ticks = pd.date_range(start=meteo_filtered.index.min(), end=meteo_filtered.index.max(), freq=freq)
    axes[0].set_yticks(mdates.date2num(time_ticks))  # Ensure the ticks match the 3-hour intervals
    axes[0].set_yticklabels(time_ticks.strftime('%Y-%m-%d %H:%M'))  # Format as date and time

    axes[0].set_title('Stream T (°C)')
    axes[0].set_xlabel('Distance Along Stream (m)')
    axes[0].set_ylabel('Time')
    axes[0].invert_yaxis()  # Invert the y-axis to have time from bottom to top

    # Compute the mean temperature along the x-dimension for the data
    mean_temp_x = xr_data.sel(time=time_slice, x=x_slice)['T'].mean(dim='x')

    # Create a 2D array by tiling the mean temperature along the x-dimension
    mean_temp_2d = np.tile(mean_temp_x.values, (len(xr_data['x']), 1)).T

    # Plot the mean temperature along the x-dimension as a 2D strip
    axes[1].imshow(
        mean_temp_2d,  # Use the mean temperature 2D array
        aspect='auto',  
        cmap='RdYlBu_r', vmin=vmin, vmax=vmax,
        extent=[0, 1, mdates.date2num(meteo_filtered.index.max()), mdates.date2num(meteo_filtered.index.min())]  # Adjusted to place time on y-axis
    )
    axes[1].set_title('Stream \nmean (°C)', rotation=90, fontsize=12)
    axes[1].set_xticks([])  # No x-ticks since it's just a strip
    axes[1].set_xlabel('')
    axes[1].set_yticks([])  # Remove y-ticks
    axes[1].set_yticklabels([])  # Remove y-tick labels
    axes[1].invert_yaxis()  # Invert the y-axis to have time from bottom to top

    # Plot the meteo temperature for 'Lompolo' as a vertical strip
    meteo_time = meteo_filtered.index
    meteo_temp = meteo_filtered['Lompolo']

    # Create a 2D array with temperature repeated horizontally to fit the plot
    temp_2d = np.tile(meteo_temp.values, (len(xr_data['x']), 1)).T

    # Plot the vertical strip for Lompolo
    axes[2].imshow(
        temp_2d,  # Use the temperature 2D array
        aspect='auto',  # Stretch to fit
        cmap='RdYlBu_r', vmin=vmin, vmax=vmax,
        extent=[0, 1, mdates.date2num(meteo_time.max()), mdates.date2num(meteo_time.min())]  # Adjusted to place time on y-axis
    )
    axes[2].set_title('Lompolo\n(°C)', rotation=90, fontsize=12)
    axes[2].set_xticks([])  # No x-ticks since it's just a strip
    axes[2].set_xlabel('')
    axes[2].set_yticks([])  # Remove y-ticks
    axes[2].set_yticklabels([])  # Remove y-tick labels
    axes[2].invert_yaxis()  # Invert the y-axis to have time from bottom to top

    # Plot the meteo temperature for 'Kenttarova' as a vertical strip
    meteo_temp_kenttarova = meteo_filtered['Kenttarova']

    # Create a 2D array for Kenttarova
    temp_2d_kenttarova = np.tile(meteo_temp_kenttarova.values, (len(xr_data['x']), 1)).T

    # Use imshow to plot the vertical strip for Kenttarova
    axes[3].imshow(
        temp_2d_kenttarova,  # Use the Kenttarova temperature 2D array
        aspect='auto',  # Stretch to fit
        cmap='RdYlBu_r', vmin=vmin, vmax=vmax,
        extent=[0, 1, mdates.date2num(meteo_time.max()), mdates.date2num(meteo_time.min())]  # Adjusted to place time on y-axis
    )
    axes[3].set_title('Kenttarova\n(°C)', rotation=90, fontsize=12)
    axes[3].set_xticks([])  # No x-ticks since it's just a strip
    axes[3].set_xlabel('')
    axes[3].set_yticks([])  # Remove y-ticks
    axes[3].set_yticklabels([])  # Remove y-tick labels
    axes[3].invert_yaxis()  # Invert the y-axis to have time from bottom to top

    # Add a single shared colorbar for all plots
    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])  # Position for colorbar (x, y, width, height)
    fig.colorbar(plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=plt.Normalize(vmin=vmin, vmax=vmax)), cax=cbar_ax, label='T (°C)')

    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    if save_fp:
        plt.savefig(save_fp, dpi=300)

    plt.show()

def plot_dts_meteo_distributions(xr_data, meteo_df, time_slice, x_slice, save_fp=None):

    # Filter the meteo DataFrame to match the time slice
    meteo_filtered = meteo_df.loc[time_slice]

    # Create subplots with adjusted width for the second and third subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), gridspec_kw={'width_ratios': [3, 0.3, 0.3]})

    # First plot: Temperature along the stream (from xarray)
    for time in xr_data.sel(time=time_slice)['time']:
        xr_data['T'].sel(time=time).plot(ax=axes[0], alpha=0.2, color='tab:blue')

    axes[0].set_title(f'Stream Temperature {time_slice.start} (°C)')
    axes[0].set_xlabel('Distance Along Stream (m)')
    axes[0].set_ylabel('Temperature (°C)')

    # Set the same y-limits for both subplots
    y_min = min(xr_data['T'].sel(time=time_slice).min(), meteo_filtered['Lompolo'].min(), meteo_filtered['Kenttarova'].min())
    y_max = max(xr_data['T'].sel(time=time_slice).max(), meteo_filtered['Lompolo'].max(), meteo_filtered['Kenttarova'].max())

    axes[0].set_ylim(y_min, y_max)
    axes[1].set_ylim(y_min, y_max)
    axes[2].set_ylim(y_min, y_max)

    # Second plot: Boxplot for temperature variation (from meteo) for Lompolo
    sns.boxplot(data=meteo_filtered, y='Lompolo', ax=axes[1], color='tab:blue')

    axes[1].set_title('T range (°C)\nat Lompolo')
    axes[1].set_xlabel('')
    axes[1].set_ylabel('')  # Hide y-axis title
    axes[1].set_yticks([])  # Hide y-axis ticks and labels

    # Third plot: Boxplot for temperature variation (from meteo) for Kenttarova
    sns.boxplot(data=meteo_filtered, y='Kenttarova', ax=axes[2], color='tab:blue')

    axes[2].set_title('T range (°C)\nat Kenttarova')
    axes[2].set_xlabel('')
    axes[2].set_ylabel('')  # Hide y-axis title
    axes[2].set_yticks([])  # Hide y-axis ticks and labels

    # Show the plot
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    #plt.savefig('FIGS/temp_dist_summer.png', dpi=300)
    plt.show()


def histogram_match(data1, data2, lims,  bins=50):
    
    hobs, binobs = np.histogram(data1, bins=bins, range=lims)
    hsim, binsim = np.histogram(data2, bins=bins, range=lims)
    
    hobs=np.float64(hobs)
    hsim=np.float64(hsim)
    
    minima = np.minimum(hsim, hobs)
    gamma = round(np.sum(minima)/np.sum(hobs),2)
    
    return gamma


def plot_monthly_water_temp_contour(xr_data, time_slice, x_slice, save_fp=None):

    import calendar
    
    # 1) Select the time & x-range
    T = xr_data['T'].sel(time=time_slice, x=x_slice)

    # 2) Group by calendar month and average over time
    #    Result dims: (month=1..12, x)
    T_monthly = T.groupby('time.month').mean(dim='time')

    # 3) Extract the numeric arrays for plotting
    months = T_monthly['month'].values                # [1,2,…,12]
    x_pos  = T_monthly['x'].values                    # e.g. 50…2000
    Z      = T_monthly.values                         # shape (12, N_x)

    # 4) Build the grid
    M, X = np.meshgrid(months, x_pos, indexing='xy')  # X: x_pos×month

    # 5) Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Filled contours
    cf = ax.contourf(M, X, Z.T, levels=20, cmap='RdBu_r', extend='both')
    # Overlay contour lines
    cs = ax.contour(M, X, Z.T, levels=10, colors='k', linewidths=0.5)
    ax.clabel(cs, fmt='%1.1f', fontsize=12)

    # Colorbar
    cbar = fig.colorbar(cf, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label(r'$T_w$ (°C)', fontsize=12)

    # 6) Format axes
    ax.set_xlabel('Year-2022', fontsize=12)
    ax.set_ylabel('X, Distance Along Stream (m)', fontsize=12)

    # Replace month numbers with names
    ax.set_xticks(months)
    ax.set_xticklabels([calendar.month_abbr[m] for m in months], fontsize=12)

    # Add a grid
    ax.grid(True, linestyle='--', alpha=0.4)

    # Title
    start, stop = pd.to_datetime(time_slice.start), pd.to_datetime(time_slice.stop)
    # ax.set_title('Monthly Mean Stream Temperature', fontsize=12)

    plt.tight_layout()
    if save_fp:
        plt.savefig(save_fp, dpi=300, bbox_inches='tight')
    plt.show()


# ===== DTS FEATURE EXTRACTION AND VISUALIZATION =====

def load_dts_data(dts_file):

    dts_data = xr.open_dataset(dts_file, engine='netcdf4')
    dts_x = dts_data.x.values
    
    # Determine temperature variable name
    if 'st' in dts_data.variables:
        temp_var = 'st'
    elif 'temperature' in dts_data.variables:
        temp_var = 'temperature'
    elif 'temp' in dts_data.variables:
        temp_var = 'temp'
    else:
        temp_var = list(dts_data.data_vars.keys())[0]
    
    return dts_data, dts_x, temp_var


def create_burned_features(dts_x, correct_poi_distances, esker_regions=None, tolerance=25, output_file='dts_burned_features.csv', verbose=True):

    
    # Define burning values (categorical codes)
    BURN_VALUES = {
        'background': 0,
        'ditch_3': 1,          # Ditch 2 Entering (630m)
        'ditch_2': 2,          # Ditch 1 Entering (700m)
        'ditch_1': 4,          # Tributary entering area (1470m)
        'upland_spring': 8,    # Upland Spring 2 (1860m)
        'upland': 16,          # Upstream spring (1890m)
        'esker_1': 32,         # Episodic ditch (380-450m)
        'esker_2': 64,         # Esker region (1000-1300m)
    }
    
    # Initialize burned vector
    burned_vector = pd.DataFrame({
        'x_dts': dts_x,
        'burned_features': 0
    })
    
    # Burn POI features
    for feature_type, stream_distance in correct_poi_distances.items():
        if feature_type not in BURN_VALUES:
            continue
            
        # Find DTS points near this stream distance
        distances = np.abs(dts_x - stream_distance)
        nearby_indices = np.where(distances <= tolerance)[0]
        
        if len(nearby_indices) > 0:
            burned_vector.loc[nearby_indices, 'burned_features'] = BURN_VALUES[feature_type]
    
    # Burn esker regions if provided
    if esker_regions:
        for esker_name, (start, end) in esker_regions.items():
            if esker_name not in BURN_VALUES:
                continue
                
            esker_mask = (dts_x >= start) & (dts_x <= end)
            esker_indices = np.where(esker_mask)[0]
            
            if len(esker_indices) > 0:
                # Only burn if not already occupied by a POI feature
                available_indices = esker_indices[burned_vector.loc[esker_indices, 'burned_features'] == 0]
                if len(available_indices) > 0:
                    burned_vector.loc[available_indices, 'burned_features'] = BURN_VALUES[esker_name]
    
    # Save
    burned_vector.to_csv(output_file, index=False)
    
    return burned_vector


def plot_temperature_with_features(dts_data, dates, start_hour=0, end_hour=3, 
                                   burned_features_file='dts_burned_features.csv',
                                   xlim=(0, 2000), ylim=(4, 13.2),
                                   figsize=(14, 9), save_fp=None):
       
    # Determine temperature variable
    if 'st' in dts_data.variables:
        temp_var = 'st'
    elif 'temperature' in dts_data.variables:
        temp_var = 'temperature'
    elif 'temp' in dts_data.variables:
        temp_var = 'temp'
    else:
        temp_var = list(dts_data.data_vars.keys())[0]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Load and plot burned features as background
    feature_patches = []
    feature_labels_list = []
    
    try:
        burned_features = pd.read_csv(burned_features_file)
        
        # Single color for all features
        FEATURE_COLOR = '#2F2F2F'
        
        FEATURE_LABELS = {
            1: 'Ditch inflow 1 (630m)',
            2: 'Ditch inflow 2 (700m)',
            4: 'Tributary (1470m)',
            8: 'Upland Spring 2 (1860m)',
            16: 'Upstream spring (1890m)',
            32: 'Episodic ditch (380-450m)',
            64: 'Esker region (1000-1300m)'
        }
        
        for feature_value, feature_label in FEATURE_LABELS.items():
            feature_mask = burned_features['burned_features'] == feature_value
            if feature_mask.sum() > 0:
                feature_x = burned_features.loc[feature_mask, 'x_dts'].values
                min_x, max_x = feature_x.min(), feature_x.max()
                patch = ax.axvspan(min_x, max_x, color=FEATURE_COLOR, alpha=0.12, zorder=0)
                
                # Store for legend
                feature_patches.append(patch)
                feature_labels_list.append(feature_label)
    
        
    except FileNotFoundError:
        pass
    except Exception as e:
        pass
    
    # Generate colors for dates
    n_dates = len(dates)
    if n_dates > 0:
        colors = sns.color_palette("coolwarm_r", n_dates)
    
    # Plot temperature for each date
    for i, date in enumerate(dates):
        try:
            time_slice = slice(f"{date}T{start_hour:02d}:00:00", f"{date}T{end_hour:02d}:59:59")
            selected_data = dts_data.sel(time=time_slice)
            mean_temp = selected_data[temp_var].mean(dim='time')
            
            ax.plot(dts_data.x.values, mean_temp.values,
                   color=colors[i],
                   marker='o',
                   markersize=9,
                   markeredgecolor='black',
                   markeredgewidth=1,
                   linewidth=0,
                   label=f'{date}')
            
        except Exception as e:
            pass
    
    # Styling
    ax.set_xlabel('X, Distance Along Stream (m)', fontsize=26, fontfamily='Arial')
    ax.set_ylabel('Nocturnal Mean $S_t$ (°C)', fontsize=26, fontfamily='Arial')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.tick_params(axis='x', labelsize=26, length=6, width=1)
    ax.tick_params(axis='y', labelsize=26, length=6, width=1)
    
    # Set tick labels to Arial
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('Arial')
        label.set_fontsize(26)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.grid(False)
    
    # Create combined legend with dates and features
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.35)  # More space for two-row legend
    
    # Get date handles and labels
    date_handles, date_labels = ax.get_legend_handles_labels()
    
    # Combine with feature handles and labels
    all_handles = date_handles + feature_patches
    all_labels = date_labels + feature_labels_list
    
    # Create legend with two rows: dates on top, features on bottom
    legend = ax.legend(all_handles, all_labels, 
                      frameon=False, 
                      loc='upper center',
                      bbox_to_anchor=(0.5, -0.12), 
                      ncol=4,  # 4 columns to fit better
                      fontsize=20,
                      prop={'family': 'Arial', 'size': 20},
                      columnspacing=1.5)
    
    if save_fp:
        plt.savefig(save_fp, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_3night_mean_temperature(dts_data, three_night_periods, start_hour=0, end_hour=3,
                                  burned_features_file='dts_burned_features.csv',
                                  xlim=(0, 2000), ylim=(4, 16),
                                  figsize=(14, 9), save_fp=None):
    """
    Plot mean stream temperature averaged over 3 consecutive nights.
    """
    
    # Determine temperature variable
    if 'st' in dts_data.variables:
        temp_var = 'st'
    elif 'T' in dts_data.variables:
        temp_var = 'T'
    elif 'temperature' in dts_data.variables:
        temp_var = 'temperature'
    elif 'temp' in dts_data.variables:
        temp_var = 'temp'
    else:
        temp_var = list(dts_data.data_vars.keys())[0]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Load and plot burned features as background
    feature_patches = []
    feature_labels_list = []
    
    try:
        burned_features = pd.read_csv(burned_features_file)
        
        # Single color for all features
        FEATURE_COLOR = '#2F2F2F'
        
        FEATURE_LABELS = {
            1: 'Ditch inflow 1 (630m)',
            2: 'Ditch inflow 2 (700m)',
            4: 'Tributary (1470m)',
            8: 'Upland Spring 2 (1860m)',
            16: 'Upstream spring (1890m)',
            32: 'Episodic ditch (380-450m)',
            64: 'Esker region (1000-1300m)'
        }
        
        for feature_value, feature_label in FEATURE_LABELS.items():
            feature_mask = burned_features['burned_features'] == feature_value
            if feature_mask.sum() > 0:
                feature_x = burned_features.loc[feature_mask, 'x_dts'].values
                min_x, max_x = feature_x.min(), feature_x.max()
                patch = ax.axvspan(min_x, max_x, color=FEATURE_COLOR, alpha=0.12, zorder=0)
                
                # Store for legend
                feature_patches.append(patch)
                feature_labels_list.append(feature_label)
                
    except FileNotFoundError:
        pass
    except Exception as e:
        pass
    
    # Generate colors for the periods
    n_periods = len(three_night_periods)
    if n_periods > 0:
        colors = sns.color_palette("coolwarm_r", n_periods)
    
    # Plot 3-night mean for each period
    for i, (start_date, end_date, label) in enumerate(three_night_periods):
        try:
            # Select 3 nights between start and end date, hours 0-3
            time_slice = slice(f"{start_date}T{start_hour:02d}:00:00", f"{end_date}T{end_hour:02d}:59:59")
            selected_data = dts_data.sel(time=time_slice)
            
            # Filter to only include hours start_hour to end_hour for each night
            hourly_data = selected_data.where(
                (selected_data.time.dt.hour >= start_hour) & (selected_data.time.dt.hour <= end_hour),
                drop=True
            )
            
            # Compute mean across all selected times
            mean_temp = hourly_data[temp_var].mean(dim='time')
            
            # Plot
            ax.plot(dts_data.x.values, mean_temp.values,
                    color=colors[i],
                    marker='o',
                    markersize=9,
                    markeredgecolor='black',
                    markeredgewidth=1,
                    linewidth=0,
                    label=label)
            
        except Exception as e:
            pass
    
    # Styling (matching plot_temperature_with_features)
    ax.set_xlabel('X, Distance Along Stream (m)', fontsize=26, fontfamily='Arial')
    ax.set_ylabel('Nocturnal Mean $S_t$ (°C)', fontsize=26, fontfamily='Arial')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.tick_params(axis='x', labelsize=26, length=6, width=1)
    ax.tick_params(axis='y', labelsize=26, length=6, width=1)
    
    # Set tick labels to Arial
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('Arial')
        label.set_fontsize(26)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.grid(False)
    
    # Create combined legend with dates and features
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.35)  # More space for two-row legend
    
    # Get date handles and labels
    date_handles, date_labels = ax.get_legend_handles_labels()
    
    # Combine with feature handles and labels
    all_handles = date_handles + feature_patches
    all_labels = date_labels + feature_labels_list
    
    # Create legend with two rows: dates on top, features on bottom
    legend = ax.legend(all_handles, all_labels, 
                      frameon=False, 
                      loc='upper center',
                      bbox_to_anchor=(0.5, -0.12), 
                      ncol=4,  # 4 columns to fit better
                      fontsize=20,
                      prop={'family': 'Arial', 'size': 20},
                      columnspacing=1.5)
    
    if save_fp:
        plt.savefig(save_fp, dpi=300, bbox_inches='tight')
    
    plt.show()


# ===== TWI AND FLOW ACCUMULATION FUNCTIONS =====

def load_twi_data_for_dual_axis():
    """Load TWI data for adding to existing temperature plot"""
    
    base_dir = r'C:\Users\pparvizi24\OneDrive - University of Oulu and Oamk\Parsa-PHD-OneDrive\DTS\Dataset - All\WBT_data\WBT_data'
    resolutions = [8, 16]  # Only keep 8m and 16m resolutions
    methods = ['d8', 'dinf', 'fd8', 'mdinf', 'rho8']
    
    # Colors and styles for TWI data (using same coolwarm palette as temperature)
    twi_method_colors = {
        'd8': '#5a1a35', 'dinf': '#4a2bb8', 'fd8': 'darkblue', 
        'mdinf': '#b8730d', 'rho8': '#0f2d3f'
    }
    resolution_linestyles = {8: '-', 16: '--'}
    
    def load_twi_data(method, resolution):
        """Load TWI data from CSV file"""
        twi_file = os.path.join(base_dir, f'pallas_{resolution}', f'twi_1d_{method}_clipped.csv')
        
        if not os.path.exists(twi_file):
            return None
        
        try:
            # Read TWI data
            df = pd.read_csv(twi_file, index_col=0)
            
            if len(df) == 0:
                return None
            
            # Use smoothed TWI data if available, otherwise use original
            if 'twi_smooth' in df.columns:
                twi_values = df['twi_smooth']
            elif 'twi_value' in df.columns:
                twi_values = df['twi_value']
            else:
                return None
            
            # Create DataFrame with stream_length as index and TWI values
            result_df = pd.DataFrame({
                'twi_value': twi_values
            })
            result_df.index.name = 'stream_length'
            
            return result_df
            
        except Exception as e:
            return None
    
    all_data = {}
    for method in methods:
        for r in resolutions:
            df = load_twi_data(method, r)
            if df is not None:
                all_data[f'{r}m_{method}'] = df
    
    return all_data, twi_method_colors, resolution_linestyles


def load_flow_accumulation_for_dual_axis():
    """Load flow accumulation data for adding to existing temperature plot"""
    
    base_dir = r'C:\Users\pparvizi24\OneDrive - University of Oulu and Oamk\Parsa-PHD-OneDrive\DTS\Dataset - All\WBT_data\WBT_data'
    resolutions = [8, 16]  # Only keep 8m and 16m resolutions
    methods = ['d8', 'dinf', 'fd8', 'mdinf', 'rho8']
    
    # Colors and styles for flow accumulation (darker colors)
    flow_method_colors = {
        'd8': '#5a1a35', 'dinf': '#4a2bb8', 'fd8': 'darkblue', 
        'mdinf': '#b8730d', 'rho8': '#0f2d3f'
    }
    resolution_linestyles = {8: '-', 16: '--'}
    
    def extract_flow_data_km2(method, resolution):
        flow_acc_file = os.path.join(base_dir, f'pallas_{resolution}', 
                                   f'korkeusmalli_{resolution}m_culverts_water_no_deps_flowacc_{method}_clipped.tif')
        stream_l_file = os.path.join(base_dir, f'pallas_{resolution}', 'stream_lengths_burned_clipped.tif')
        
        if not os.path.exists(flow_acc_file) or not os.path.exists(stream_l_file):
            return None
        
        try:
            import rasterio
            with rasterio.open(flow_acc_file) as src:
                flow_acc_sca = src.read(1)
            with rasterio.open(stream_l_file) as src:
                stream_len = src.read(1)
            
            if flow_acc_sca.shape != stream_len.shape:
                return None
            
            stream_len = stream_len.astype('float')
            stream_len[stream_len == 0] = np.nan
            stream_len_1d = np.unique(stream_len)[np.isfinite(np.unique(stream_len))]
            
            data = []
            pixel_size = resolution
            pixel_area = pixel_size * pixel_size
            
            for length in stream_len_1d:
                mask = stream_len == length
                sca_values = flow_acc_sca[mask]
                
                if len(sca_values) > 0:
                    sca_m2_per_m = sca_values[0]
                    pixel_count = (sca_m2_per_m * pixel_size) / pixel_area
                    upslope_area_km2 = (pixel_count * pixel_area) / 1000000
                    
                    data.append({'stream_length': length, 'upslope_area_km2': upslope_area_km2})
            
            # Sort and clean data
            data_sorted = sorted(data, key=lambda x: x['stream_length'], reverse=True)
            cleaned = []
            max_area = -np.inf
            
            for row in data_sorted:
                if row['upslope_area_km2'] >= max_area:
                    cleaned.append(row)
                    max_area = row['upslope_area_km2']
            
            df = pd.DataFrame(cleaned)
            df.set_index('stream_length', inplace=True)
            return df
            
        except Exception as e:
            return None
    
    all_data = {}
    for method in methods:
        for r in resolutions:
            df = extract_flow_data_km2(method, r)
            if df is not None:
                all_data[f'{r}m_{method}'] = df
    
    return all_data, flow_method_colors, resolution_linestyles

def plot_twi_data():
    """Create TWI plot for all methods at 8m and 16m resolutions"""
    
    # Load TWI data
    twi_data, method_colors, resolution_styles = load_twi_data_for_dual_axis()
    
    # Set seaborn theme for professional appearance
    sns.set_theme(style="white")
    
    # Create the plot with single y-axis
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Load burned features for background shading
    try:
        burned_features = pd.read_csv('dts_burned_features.csv')
        
        # Define feature colors using single gray color
        feature_color = '#2F2F2F'
        
        FEATURE_COLORS = {
            0: None,
            1: feature_color,      # Ditch inflow 1 (630m)
            2: feature_color,      # Ditch inflow 2 (700m)  
            4: feature_color,      # Tributary entering area (1470m)
            8: feature_color,      # Upland Spring 2 (1860m)
            16: feature_color,     # Upstream spring (1900m)
            32: feature_color,     # Episodic ditch (450-470m)
            64: feature_color,     # Esker region (1000-1300m)
        }
        
        # Add background shading
        for feature_value, color in FEATURE_COLORS.items():
            if feature_value == 0 or color is None:
                continue
            feature_mask = burned_features['burned_features'] == feature_value
            if feature_mask.sum() > 0:
                feature_x = burned_features.loc[feature_mask, 'x_dts'].values
                min_x, max_x = feature_x.min(), feature_x.max()
                # Add feature shading
                ax1.axvspan(min_x, max_x, color=color, alpha=0.12, zorder=0)
        
        features_loaded = True
        
    except FileNotFoundError:
        features_loaded = False
    except Exception as e:
        features_loaded = False
    
    # Plot TWI data - filter for 8m and 16m resolutions only
    if twi_data:
        # Filter TWI data for only 8m and 16m resolutions
        filtered_twi_data = {}
        for key, df in twi_data.items():
            resolution = int(key.split('m_')[0])
            if resolution in [8, 16]:  # Only include 8m and 16m
                filtered_twi_data[key] = df
        
        # Track methods and resolutions for legend
        plotted_methods = set()
        plotted_resolutions = set()
        
        for key, df in filtered_twi_data.items():
            if len(df) > 0:
                resolution = int(key.split('m_')[0])
                method = key.split('m_')[1]
                
                # Plot TWI data on primary y-axis (left side)
                ax1.plot(df.index, df['twi_value'],
                        linestyle=resolution_styles[resolution],
                        color=method_colors[method],
                        linewidth=3.0,
                        alpha=0.8)
                
                # Track what we've plotted
                plotted_methods.add(method)
                plotted_resolutions.add(resolution)
        
        # Set TWI plot properties on primary axis
        ax1.set_ylim(10, 18)
        ax1.set_ylabel('TWI', fontsize=24, fontfamily='Arial')
        ax1.set_yticks([10, 12, 14, 16, 18])
        ax1.tick_params(axis='y', labelsize=24, length=6, width=1)
        
        # Set TWI plot tick label font to Arial
        for label in ax1.get_yticklabels():
            label.set_fontfamily('Arial')
            label.set_fontsize(24)
    
    # Professional styling for TWI plot
    ax1.set_xlabel('X, Distance Along Stream (m)', fontsize=24, fontfamily='Arial')
    
    # Set axis limits and ticks
    ax1.set_xlim(0, 2000)
    ax1.tick_params(axis='x', labelsize=24, length=6, width=1)
    
    # Set tick label fonts to Arial
    for label in ax1.get_xticklabels():
        label.set_fontfamily('Arial')
        label.set_fontsize(24)
    
    # Remove top and right spines for clean appearance
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    ax1.grid(False)
    
    # TWI legend
    if 'filtered_twi_data' in locals() and len(plotted_methods) > 0:
        
        def format_method_name(method):
            method_names = {
                'dinf': 'D∞',
                'mdinf': 'MD∞', 
                'rho8': 'Rho8',
                'd8': 'D8',
                'fd8': 'FD8'
            }
            return method_names.get(method.lower(), method.upper())
        
        # Method legend handles for TWI
        method_handles = []
        for method in sorted(plotted_methods):
            method_handles.append(Line2D([0], [0], color=method_colors[method], linewidth=4,
                                       label=format_method_name(method)))
        
        # Resolution legend handles for TWI
        resolution_handles = []
        desired_order = [8, 16]  # Only 8m and 16m resolutions
        plotted_in_order = [res for res in desired_order if res in plotted_resolutions]
        for res in plotted_in_order:
            resolution_handles.append(Line2D([0], [0], color='#34495E',
                                           linestyle=resolution_styles[res],
                                           linewidth=3, label=f'{res}m'))
        
        # Create TWI legend with methods in first column, resolutions in second column
        # Create two separate legends side by side
        
        # First create method legend
        method_legend = ax1.legend(method_handles, [h.get_label() for h in method_handles], 
                                 frameon=False, loc='upper right', 
                                 fontsize=20, prop={'family': 'Arial', 'size': 20},
                                 bbox_to_anchor=(0.85, 1.0))
        
        # Add the method legend to the plot
        ax1.add_artist(method_legend)
        
        # Then create resolution legend next to it
        resolution_legend = ax1.legend(resolution_handles, [h.get_label() for h in resolution_handles], 
                                     frameon=False, loc='upper right', 
                                     fontsize=20, prop={'family': 'Arial', 'size': 20},
                                     bbox_to_anchor=(1.0, 1.0))
    
    plt.tight_layout()
    plt.show()


def plot_twi_data_subplots():
    """Create TWI plot with two subplots: 8m resolution (top) and 16m resolution (bottom)"""
    
    # Load TWI data
    twi_data, method_colors, resolution_styles = load_twi_data_for_dual_axis()
    
    # Set seaborn theme for professional appearance
    sns.set_theme(style="white")
    
    # Create the plot with two subplots (vertical arrangement)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 11), sharex=True)
    
    # Load burned features for background shading
    try:
        burned_features = pd.read_csv('dts_burned_features.csv')
        
        # Define feature colors using single gray color
        feature_color = '#2F2F2F'
        
        FEATURE_COLORS = {
            0: None,
            1: feature_color,      # Ditch inflow 1 (630m)
            2: feature_color,      # Ditch inflow 2 (700m)  
            4: feature_color,      # Tributary entering area (1470m)
            8: feature_color,      # Upland Spring 2 (1860m)
            16: feature_color,     # Upstream spring (1900m)
            32: feature_color,     # Episodic ditch (450-470m)
            64: feature_color,     # Esker region (1000-1300m)
        }
        
        # Add background shading to both subplots
        for ax in [ax1, ax2]:
            for feature_value, color in FEATURE_COLORS.items():
                if feature_value == 0 or color is None:
                    continue
                feature_mask = burned_features['burned_features'] == feature_value
                if feature_mask.sum() > 0:
                    feature_x = burned_features.loc[feature_mask, 'x_dts'].values
                    min_x, max_x = feature_x.min(), feature_x.max()
                    # Add feature shading
                    ax.axvspan(min_x, max_x, color=color, alpha=0.12, zorder=0)
        
        features_loaded = True
        
    except FileNotFoundError:
        features_loaded = False
    except Exception as e:
        features_loaded = False
    
    # Plot TWI data for each resolution
    if twi_data:
        # Separate data by resolution
        data_8m = {}
        data_16m = {}
        
        for key, df in twi_data.items():
            resolution = int(key.split('m_')[0])
            if resolution == 8:
                data_8m[key] = df
            elif resolution == 16:
                data_16m[key] = df
        
        # Track methods for legend
        plotted_methods_8m = set()
        plotted_methods_16m = set()
        
        # Plot 8m data in first subplot
        for key, df in data_8m.items():
            if len(df) > 0:
                method = key.split('m_')[1]
                ax1.plot(df.index, df['twi_value'],
                        linestyle='--',  # Solid line for 8m
                        color=method_colors[method],
                        linewidth=3.0,
                        alpha=0.8)
                plotted_methods_8m.add(method)
        
        # Plot 16m data in second subplot
        for key, df in data_16m.items():
            if len(df) > 0:
                method = key.split('m_')[1]
                ax2.plot(df.index, df['twi_value'],
                        linestyle='-',  # Solid line for 16m
                        color=method_colors[method],
                        linewidth=3.0,
                        alpha=0.8)
                plotted_methods_16m.add(method)
        
        # Set TWI plot properties for both subplots
        for ax in [ax1, ax2]:
            ax.set_ylim(11, 17)
            ax.set_ylabel('TWI', fontsize=24, fontfamily='Arial')
            ax.set_yticks([12, 14, 16, 18])
            ax.tick_params(axis='y', labelsize=24, length=6, width=1)
            
            # Set TWI plot tick label font to Arial
            for label in ax.get_yticklabels():
                label.set_fontfamily('Arial')
                label.set_fontsize(24)
            
            # Remove top and right spines for clean appearance
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.yaxis.set_ticks_position('left')
            ax.grid(False)
    
    # Professional styling for TWI plot (only bottom subplot gets x-axis label)
    ax2.set_xlabel('X, Distance Along Stream (m)', fontsize=24, fontfamily='Arial')
    
    # Set axis limits and ticks for both subplots
    for ax in [ax1, ax2]:
        ax.set_xlim(0, 2000)
        ax.tick_params(axis='x', labelsize=24, length=6, width=1)
        ax.xaxis.set_ticks_position('bottom')
        
        # Set tick label fonts to Arial
        for label in ax.get_xticklabels():
            label.set_fontfamily('Arial')
            label.set_fontsize(24)
    
    # Create legends for both subplots
    def format_method_name(method):
        method_names = {
            'dinf': 'D∞',
            'mdinf': 'MD∞', 
            'rho8': 'Rho8',
            'd8': 'D8',
            'fd8': 'FD8'
        }
        return method_names.get(method.lower(), method.upper())
    
    # Add legend to first subplot (8m) with proper title and box
    if len(plotted_methods_8m) > 0:
        method_handles_8m = []
        method_labels_8m = []
        
        # Add method entries
        for method in sorted(plotted_methods_8m):
            method_handles_8m.append(Line2D([0], [0], color=method_colors[method], linewidth=4))
            method_labels_8m.append(format_method_name(method))
        
        legend_8m = ax1.legend(method_handles_8m, method_labels_8m, 
                              frameon=False, loc='upper right', 
                              fontsize=20, prop={'family': 'Arial', 'size': 20},
                              title='8m DEM', title_fontsize=20,
                              ncol=2)
        legend_8m.get_title().set_fontfamily('Arial')
        #legend_8m.get_title().set_fontweight('bold')
    
    # Add legend to second subplot (16m) with proper title and box
    if len(plotted_methods_16m) > 0:
        method_handles_16m = []
        method_labels_16m = []
        
        # Add method entries
        for method in sorted(plotted_methods_16m):
            method_handles_16m.append(Line2D([0], [0], color=method_colors[method], linewidth=4))
            method_labels_16m.append(format_method_name(method))
        
        legend_16m = ax2.legend(method_handles_16m, method_labels_16m, 
                               frameon=False, loc='upper right', 
                               fontsize=20, prop={'family': 'Arial', 'size': 20},
                               title='16m DEM', title_fontsize=20,
                               ncol=2)
        legend_16m.get_title().set_fontfamily('Arial')
        #legend_16m.get_title().set_fontweight('bold')
    
    plt.tight_layout()
    plt.show()


def plot_twi_data_subplots_horizontal():
    """Create TWI plot with two subplots side by side: 8m resolution (left) and 16m resolution (right)"""
    
    # Load TWI data
    twi_data, method_colors, resolution_styles = load_twi_data_for_dual_axis()
    
    # Set seaborn theme for professional appearance
    sns.set_theme(style="white")
    
    # Create the plot with two subplots (horizontal arrangement)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), sharey=False)
    
    # Load burned features for background shading
    try:
        burned_features = pd.read_csv('dts_burned_features.csv')
        
        # Define feature colors using single gray color
        feature_color = '#2F2F2F'
        
        FEATURE_COLORS = {
            0: None,
            1: feature_color,      # Ditch inflow 1 (630m)
            2: feature_color,      # Ditch inflow 2 (700m)  
            4: feature_color,      # Tributary entering area (1470m)
            8: feature_color,      # Upland Spring 2 (1860m)
            16: feature_color,     # Upstream spring (1900m)
            32: feature_color,     # Episodic ditch (450-470m)
            64: feature_color,     # Esker region (1000-1300m)
        }
        
        # Add background shading to both subplots
        for ax in [ax1, ax2]:
            for feature_value, color in FEATURE_COLORS.items():
                if feature_value == 0 or color is None:
                    continue
                feature_mask = burned_features['burned_features'] == feature_value
                if feature_mask.sum() > 0:
                    feature_x = burned_features.loc[feature_mask, 'x_dts'].values
                    min_x, max_x = feature_x.min(), feature_x.max()
                    # Add feature shading
                    ax.axvspan(min_x, max_x, color=color, alpha=0.12, zorder=0)
        
        features_loaded = True
        
    except FileNotFoundError:
        features_loaded = False
    except Exception as e:
        features_loaded = False
    
    # Plot TWI data for each resolution
    if twi_data:
        # Separate data by resolution
        data_8m = {}
        data_16m = {}
        
        for key, df in twi_data.items():
            resolution = int(key.split('m_')[0])
            if resolution == 8:
                data_8m[key] = df
            elif resolution == 16:
                data_16m[key] = df
        
        # Track methods for legend
        plotted_methods_8m = set()
        plotted_methods_16m = set()
        
        # Plot 8m data in first subplot (left)
        #print(f"Adding {len(data_8m)} TWI datasets for 8m resolution...")
        for key, df in data_8m.items():
            if len(df) > 0:
                method = key.split('m_')[1]
                ax1.plot(df.index, df['twi_value'],
                        linestyle='--',  # Dashed line for 8m
                        color=method_colors[method],
                        linewidth=3.0,
                        alpha=0.8)
                plotted_methods_8m.add(method)
        
        # Plot 16m data in second subplot (right)
        #print(f"Adding {len(data_16m)} TWI datasets for 16m resolution...")
        for key, df in data_16m.items():
            if len(df) > 0:
                method = key.split('m_')[1]
                ax2.plot(df.index, df['twi_value'],
                        linestyle='-',  # Solid line for 16m
                        color=method_colors[method],
                        linewidth=3.0,
                        alpha=0.8)
                plotted_methods_16m.add(method)
        
        # Set TWI plot properties for both subplots
        for ax in [ax1, ax2]:
            ax.set_ylim(10, 18)
            
            # Add horizontal grid lines at specific TWI values
            for twi_value in [12, 14, 16]:
                ax.axhline(y=twi_value, color='darkgray', linestyle='--', alpha=0.7, linewidth=1.5, zorder=1)
            
            # Remove top and right spines for clean appearance
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(False)
        
        # Left subplot gets y-axis label and tick values
        ax1.set_ylabel('TWI', fontsize=24, fontfamily='Arial')
        ax1.set_yticks([10, 12, 14, 16, 18])
        ax1.set_yticklabels(['10', '12', '14', '16', '18'])
        ax1.yaxis.set_ticks_position('left')
        ax1.tick_params(axis='y', labelsize=24, length=6, width=1, which='major')
        ax1.yaxis.set_tick_params(labelleft=True)
        
        # Set TWI plot tick label font to Arial for left subplot
        for label in ax1.get_yticklabels():
            label.set_fontfamily('Arial')
            label.set_fontsize(24)
            label.set_visible(True)
        
        # Right subplot - same y-limits but no labels
        ax2.set_ylim(10, 18)
        ax2.set_yticks([10, 12, 14, 16, 18])
        ax2.set_yticklabels([])
        ax2.yaxis.set_ticks_position('none')
        ax2.tick_params(axis='y', length=0, labelsize=0, labelleft=False)
    
    # Professional styling for TWI plot (both subplots get x-axis labels)
    ax1.set_xlabel('X, Distance Along Stream (m)', fontsize=24, fontfamily='Arial')
    ax2.set_xlabel('X, Distance Along Stream (m)', fontsize=24, fontfamily='Arial')
    
    # Set axis limits and ticks for both subplots
    for ax in [ax1, ax2]:
        ax.set_xlim(0, 2000)
        ax.tick_params(axis='x', labelsize=24, length=6, width=1)
        ax.xaxis.set_ticks_position('bottom')
        
        # Set tick label fonts to Arial
        for label in ax.get_xticklabels():
            label.set_fontfamily('Arial')
            label.set_fontsize(24)
    
    # Create legends for both subplots
    def format_method_name(method):
        method_names = {
            'dinf': 'D∞',
            'mdinf': 'MD∞', 
            'rho8': 'Rho8',
            'd8': 'D8',
            'fd8': 'FD8'
        }
        return method_names.get(method.lower(), method.upper())
    
    # Add legend to first subplot (8m)
    if len(plotted_methods_8m) > 0:
        method_handles_8m = []
        method_labels_8m = []
        
        # Add method entries
        for method in sorted(plotted_methods_8m):
            method_handles_8m.append(Line2D([0], [0], color=method_colors[method], linewidth=4))
            method_labels_8m.append(format_method_name(method))
        
        legend_8m = ax1.legend(method_handles_8m, method_labels_8m, 
                              frameon=False, loc='upper right', 
                              fontsize=20, prop={'family': 'Arial', 'size': 20},
                              title='8m DEM', title_fontsize=20,
                              ncol=2)
        legend_8m.get_title().set_fontfamily('Arial')
    
    # Add legend to second subplot (16m)
    if len(plotted_methods_16m) > 0:
        method_handles_16m = []
        method_labels_16m = []
        
        # Add method entries
        for method in sorted(plotted_methods_16m):
            method_handles_16m.append(Line2D([0], [0], color=method_colors[method], linewidth=4))
            method_labels_16m.append(format_method_name(method))
        
        legend_16m = ax2.legend(method_handles_16m, method_labels_16m, 
                               frameon=False, loc='upper right', 
                               fontsize=20, prop={'family': 'Arial', 'size': 20},
                               title='16m DEM', title_fontsize=20,
                               ncol=2)
        legend_16m.get_title().set_fontfamily('Arial')
    
    plt.tight_layout()
    plt.show()
    
    #print(" TWI horizontal subplot plot complete!")
    if twi_data:
        print(f" 8m resolution: {len(data_8m)} datasets, Methods: {', '.join(sorted(plotted_methods_8m))}")
        print(f" 16m resolution: {len(data_16m)} datasets, Methods: {', '.join(sorted(plotted_methods_16m))}")
    else:
        print(" No TWI data available")
    print(" All styling preserved: 24pt Arial fonts, burned feature background, two horizontal subplots")


def plot_temperature_and_flow_acc_side_by_side(data, dates_left, dates_right, start_hour=0, end_hour=3, 
                                                 flow_data=None, method_colors=None, resolution_styles=None,
                                                 temp_ylim_left=None, temp_ylim_right=None):
    """
    Create side-by-side plots: 
    - Left: Temperature only (single-axis)
    - Right: Temperature with flow accumulation (dual-axis)
    """
    
    import numpy as np
    import xarray as xr
    import rasterio
    
   
    # Set default y-axis limits if not provided
    if temp_ylim_left is None:
        temp_ylim_left = (4.8, 13.2)
    if temp_ylim_right is None:
        temp_ylim_right = (0, 7)
    
    # Determine temperature variable name
    if 'st' in data.variables:
        temp_var = 'st'
    elif 'temperature' in data.variables:
        temp_var = 'temperature'
    elif 'temp' in data.variables:
        temp_var = 'temp'
    else:
        temp_var = list(data.data_vars.keys())[0]
    
    # Set seaborn theme for professional appearance
    sns.set_theme(style="white")
    
    # Create the plot with two subplots side by side
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(20, 8), sharey=False)
    
    # Load burned features for background shading
    try:
        burned_features = pd.read_csv('dts_burned_features.csv')
        
        FEATURE_COLORS = {
            0: None,
            1: '#2F2F2F',    # Ditch inflow 1 (630m)
            2: '#2F2F2F',    # Ditch inflow 2 (700m)
            4: '#2F2F2F',    # Tributary entering area (1470m)
            8: '#2F2F2F',    # Upland Spring 2 (1860m)
            16: '#2F2F2F',   # Upstream spring (1900m)
            32: '#2F2F2F',   # Episodic ditch (450-470m)
            64: '#2F2F2F',   # Esker region (1000-1300m)
        }
        
        FEATURE_LABELS = {
            1: 'Ditch inflow 1 (630m)',
            2: 'Ditch inflow 2 (700m)',
            4: 'Tributary entering area (1470m)', 
            8: 'Upland Spring 2 (1860m)',
            16: 'Upstream spring (1900m)',
            32: 'Episodic ditch (450-470m)',
            64: 'Esker region (1000-1300m)'
        }
        
        # Add background shading to both subplots
        # print(" Adding burned stream features as background...")
        
        for ax in [ax_left, ax_right]:
            for feature_value, color in FEATURE_COLORS.items():
                if feature_value == 0 or color is None:
                    continue
                feature_mask = burned_features['burned_features'] == feature_value
                if feature_mask.sum() > 0:
                    feature_x = burned_features.loc[feature_mask, 'x_dts'].values
                    min_x, max_x = feature_x.min(), feature_x.max()
                    ax.axvspan(min_x, max_x, color=color, alpha=0.12, zorder=0)
        
        # print(f" Successfully added {len(FEATURE_LABELS)} burned stream features!")
        
    except Exception as e:
        pass  # Silently handle error
        # print(f"  Error loading burned features: {e}")
    
    # Generate coolwarm colors for LEFT plot dates
    n_dates_left = len(dates_left)
    if n_dates_left > 0:
        colors_left = sns.color_palette("coolwarm_r", n_dates_left)
        # print(f"\n=== LEFT PLOT ===")
        # print(f"Coolwarm_r (reversed) color mapping for {n_dates_left} selected dates:")
        # for i, date in enumerate(dates_left):
        #     print(f"{i+1}. {date} -> coolwarm_r color {i+1}/{n_dates_left}")
    
    # Generate coolwarm colors for RIGHT plot dates
    n_dates_right = len(dates_right)
    if n_dates_right > 0:
        colors_right = sns.color_palette("coolwarm", n_dates_right)
        # print(f"\n=== RIGHT PLOT ===")
        # print(f"Coolwarm color mapping for {n_dates_right} selected dates:")
        # for i, date in enumerate(dates_right):
        #     print(f"{i+1}. {date} -> coolwarm color {i+1}/{n_dates_right}")
    
    # ===== LEFT SUBPLOT: Temperature Only (Single-axis) =====
    
    # Plot temperature data on left subplot
    for i, date in enumerate(dates_left):
        try:
            time_slice = slice(f"{date}T{start_hour:02d}:00:00", f"{date}T{end_hour:02d}:59:59")
            selected_data = data.sel(time=time_slice)
            mean_temp = selected_data[temp_var].mean(dim='time')
            
            ax_left.plot(data.x.values, mean_temp.values, 
                         color=colors_left[i], 
                         marker='o',
                         markersize=9,
                         markeredgecolor='black',
                         markeredgewidth=1,
                         linewidth=0,
                         linestyle='-',
                         label=f'{date}')
            
            # print(f"LEFT - {date}: Temp range {mean_temp.min().values:.2f}°C to {mean_temp.max().values:.2f}°C")
            
        except Exception as e:
            pass  # Silently handle error
            # print(f"Error plotting {date} on left subplot: {e}")
    
    # Style left subplot
    ax_left.set_xlabel('X, Distance Along Stream (m)', fontsize=24, fontfamily='Arial')
    ax_left.set_ylabel('Nocturnal Mean $S_t$ (°C)', fontsize=24, fontfamily='Arial')
    ax_left.set_xlim(0, 2000)
    ax_left.set_ylim(temp_ylim_left)
    ax_left.tick_params(axis='x', labelsize=24, length=6, width=1)
    ax_left.tick_params(axis='y', labelsize=24, length=6, width=1)
    ax_left.spines['top'].set_visible(False)
    ax_left.spines['right'].set_visible(False)
    ax_left.grid(False)
    
    for label in ax_left.get_xticklabels():
        label.set_fontfamily('Arial')
        label.set_fontsize(24)
    for label in ax_left.get_yticklabels():
        label.set_fontfamily('Arial')
        label.set_fontsize(24)
    
    # ===== RIGHT SUBPLOT: Temperature + Flow Accumulation (Dual-axis) =====
    
    # Plot temperature data on right subplot
    
    for i, date in enumerate(dates_right):
        try:
            time_slice = slice(f"{date}T{start_hour:02d}:00:00", f"{date}T{end_hour:02d}:59:59")
            selected_data = data.sel(time=time_slice)
            mean_temp = selected_data[temp_var].mean(dim='time')
            
            ax_right.plot(data.x.values, mean_temp.values, 
                        color=colors_right[i], 
                        marker='o',
                        markersize=9,
                        markeredgecolor='black',
                        markeredgewidth=1,
                        linewidth=0,
                        linestyle='-',
                        label=f'{date}')
            
            # print(f"RIGHT - {date}: Temp range {mean_temp.min().values:.2f}°C to {mean_temp.max().values:.2f}°C")
            
        except Exception as e:
            pass  # Silently handle error
            # print(f"Error plotting {date}: {e}")
    
    # Add flow accumulation on second y-axis (right subplot)
    if flow_data and method_colors and resolution_styles:
        ax_right_twin = ax_right.twinx()
        
                
        plotted_methods = set()
        plotted_resolutions = set()
        
        for key, df in flow_data.items():
            if len(df) > 0:
                resolution = int(key.split('m_')[0])
                method = key.split('m_')[1]
                
                ax_right_twin.plot(df.index, df['upslope_area_km2'],
                                linestyle=resolution_styles[resolution],
                                color=method_colors[method],
                                linewidth=3.0,
                                alpha=0.8)
                
                plotted_methods.add(method)
                plotted_resolutions.add(resolution)
        
        # Set flow accumulation y-axis properties
        ax_right_twin.set_ylim(0, 4.5)
        ax_right_twin.set_ylabel('Upstream Contributing Area (km²)', fontsize=24, fontfamily='Arial', color='darkblue')
        ax_right_twin.set_yticks([0, 1, 2, 3, 4])
        ax_right_twin.tick_params(axis='y', labelsize=24, length=6, width=1, colors='darkblue')
        ax_right_twin.spines['right'].set_color('darkblue')
        ax_right_twin.spines['top'].set_visible(False)
        
        for label in ax_right_twin.get_yticklabels():
            label.set_fontfamily('Arial')
            label.set_fontsize(24)
    
    # Style right subplot
    ax_right.set_xlabel('X, Distance Along Stream (m)', fontsize=24, fontfamily='Arial')
    ax_right.set_ylabel('Nocturnal Mean $S_t$ (°C)', fontsize=24, fontfamily='Arial')
    ax_right.set_xlim(0, 2000)
    ax_right.set_ylim(temp_ylim_right)
    ax_right.tick_params(axis='x', labelsize=24, length=6, width=1)
    ax_right.tick_params(axis='y', labelsize=24, length=6, width=1)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.grid(False)
    
    for label in ax_right.get_xticklabels():
        label.set_fontfamily('Arial')
        label.set_fontsize(24)
    for label in ax_right.get_yticklabels():
        label.set_fontfamily('Arial')
        label.set_fontsize(24)
    
    # ===== CREATE LEGENDS =====
    
    # Get date handles from left subplot (temperature only)
    ax_left_handles, ax_left_labels = ax_left.get_legend_handles_labels()
    date_handles_left = ax_left_handles[:len(dates_left)]
    date_labels_left = ax_left_labels[:len(dates_left)]
    
    # Get date handles from right subplot (temperature + flow accumulation)
    ax_right_handles, ax_right_labels = ax_right.get_legend_handles_labels()
    date_handles_right = ax_right_handles[:len(dates_right)]
    date_labels_right = ax_right_labels[:len(dates_right)]
    
    # Add legends for right subplot if flow data exists
    if flow_data and method_colors and resolution_styles:
        def format_method_name(method):
            method_names = {
                'dinf': 'D∞',
                'mdinf': 'MD∞', 
                'rho8': 'Rho8',
                'd8': 'D8',
                'fd8': 'FD8'
            }
            return method_names.get(method.lower(), method.upper())
        
        # Create legend handles for flow accumulation
        method_handles = []
        for method in sorted(plotted_methods):
            method_handles.append(Line2D([0], [0], color=method_colors[method], linewidth=4,
                                       label=format_method_name(method)))
        
        resolution_handles = []
        desired_order = [8, 16]
        plotted_in_order = [res for res in desired_order if res in plotted_resolutions]
        for res in plotted_in_order:
            resolution_handles.append(Line2D([0], [0], color='#34495E',
                                           linestyle=resolution_styles[res],
                                           linewidth=3, label=f'{res}m'))
        
        # Position legends below both plots at the same distance
        plt.subplots_adjust(bottom=0.22)
        
        # LEFT plot: Simple date legend
        ax_left.legend(date_handles_left, date_labels_left, frameon=False, loc='upper center', 
                      bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=20,
                      prop={'family': 'Arial', 'size': 20})
        
        # RIGHT plot: Dates legend + Methods legend + Resolutions legend
        legend1 = ax_right.legend(date_handles_right, date_labels_right, 
                                bbox_to_anchor=(0, -0.12), loc='upper center',
                                frameon=False, fontsize=20, ncol=1,
                                prop={'family': 'Arial', 'size': 20})
        
        # Methods legend
        legend2 = ax_right.legend(method_handles, [h.get_label() for h in method_handles],
                                bbox_to_anchor=(0.5, -0.12), loc='upper center', 
                                frameon=False, fontsize=20, ncol=3,
                                prop={'family': 'Arial', 'size': 20})
        
        # Resolutions legend
        legend3 = ax_right.legend(resolution_handles, [h.get_label() for h in resolution_handles],
                                bbox_to_anchor=(0.95, -0.12), loc='upper center',
                                frameon=False, fontsize=20, ncol=1,
                                prop={'family': 'Arial', 'size': 20})
        
        ax_right.add_artist(legend1)
        ax_right.add_artist(legend2)
    else:
        # Simple date legends for both plots at the same distance
        plt.subplots_adjust(bottom=0.22)
        
        # LEFT plot legend
        ax_left.legend(date_handles_left, date_labels_left, frameon=False, loc='upper center', 
                      bbox_to_anchor=(0.5, -0.10), ncol=2, fontsize=20,
                      prop={'family': 'Arial', 'size': 20})
        
        # RIGHT plot legend
        ax_right.legend(date_handles_right, date_labels_right, frameon=False, loc='upper center', 
                       bbox_to_anchor=(0.5, -0.10), ncol=2, fontsize=20,
                       prop={'family': 'Arial', 'size': 20})
    
    plt.tight_layout()
    plt.show()
    
    # print("\n Side-by-side temperature plots complete!")
    # print(" Left plot: Temperature only (single-axis)")
    # print(" Right plot: Temperature with flow accumulation (dual-axis)")


def plot_3night_mean_and_flow_acc_side_by_side(data, three_night_periods_left, three_night_periods_right, 
                                                start_hour=0, end_hour=3, 
                                                flow_data=None, method_colors=None, resolution_styles=None,
                                                temp_ylim_left=None, temp_ylim_right=None):
    """
    Create side-by-side plots with 3-night mean temperatures:
    - Left: 3-night mean temperature only (single-axis)
    - Right: 3-night mean temperature with flow accumulation (dual-axis)
    """
    
    import numpy as np
    import xarray as xr
    import rasterio
    
    # print("Creating side-by-side 3-night mean temperature plots...")
    
    # Set default y-axis limits if not provided
    if temp_ylim_left is None:
        temp_ylim_left = (4.8, 13.2)
    if temp_ylim_right is None:
        temp_ylim_right = (0, 7)
    
    # Determine temperature variable name
    if 'st' in data.variables:
        temp_var = 'st'
    elif 'temperature' in data.variables:
        temp_var = 'temperature'
    elif 'temp' in data.variables:
        temp_var = 'temp'
    else:
        temp_var = list(data.data_vars.keys())[0]
    
    # Set seaborn theme for professional appearance
    sns.set_theme(style="white")
    
    # Create the plot with two subplots side by side
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(20, 8), sharey=False)
    
    # Load burned features for background shading
    try:
        burned_features = pd.read_csv('dts_burned_features.csv')
        
        FEATURE_COLORS = {
            0: None,
            1: '#2F2F2F',    # Ditch inflow 1 (630m)
            2: '#2F2F2F',    # Ditch inflow 2 (700m)
            4: '#2F2F2F',    # Tributary entering area (1470m)
            8: '#2F2F2F',    # Upland Spring 2 (1860m)
            16: '#2F2F2F',   # Upstream spring (1900m)
            32: '#2F2F2F',   # Episodic ditch (450-470m)
            64: '#2F2F2F',   # Esker region (1000-1300m)
        }
        
        FEATURE_LABELS = {
            1: 'Ditch inflow 1 (630m)',
            2: 'Ditch inflow 2 (700m)',
            4: 'Tributary entering area (1470m)', 
            8: 'Upland Spring 2 (1860m)',
            16: 'Upstream spring (1900m)',
            32: 'Episodic ditch (450-470m)',
            64: 'Esker region (1000-1300m)'
        }
        
        # Add background shading to both subplots
        # print(" Adding burned stream features as background...")
        
        for ax in [ax_left, ax_right]:
            for feature_value, color in FEATURE_COLORS.items():
                if feature_value == 0 or color is None:
                    continue
                feature_mask = burned_features['burned_features'] == feature_value
                if feature_mask.sum() > 0:
                    feature_x = burned_features.loc[feature_mask, 'x_dts'].values
                    min_x, max_x = feature_x.min(), feature_x.max()
                    ax.axvspan(min_x, max_x, color=color, alpha=0.12, zorder=0)
        
        # print(f" Successfully added {len(FEATURE_LABELS)} burned stream features!")
        
    except Exception as e:
        pass  # Silently handle error
        # print(f"  Error loading burned features: {e}")
    
    # Generate coolwarm colors for LEFT plot periods (REVERSED)
    n_periods_left = len(three_night_periods_left)
    if n_periods_left > 0:
        colors_left = sns.color_palette("coolwarm_r", n_periods_left)
        # print(f"\n=== LEFT PLOT ===")
        # print(f"Coolwarm_r (reversed) color mapping for {n_periods_left} 3-night periods:")
        # for i, (_, _, label) in enumerate(three_night_periods_left):
        #     print(f"{i+1}. {label} -> coolwarm_r color {i+1}/{n_periods_left}")
    
    # Generate coolwarm colors for RIGHT plot periods
    n_periods_right = len(three_night_periods_right)
    if n_periods_right > 0:
        colors_right = sns.color_palette("coolwarm", n_periods_right)
        # print(f"\n=== RIGHT PLOT ===")
        # print(f"Coolwarm color mapping for {n_periods_right} 3-night periods:")
        # for i, (_, _, label) in enumerate(three_night_periods_right):
        #     print(f"{i+1}. {label} -> coolwarm color {i+1}/{n_periods_right}")
    
    # ===== LEFT SUBPLOT: 3-Night Mean Temperature Only (Single-axis) =====
    
    # Plot 3-night mean temperature data on left subplot
    # print(f"\n=== Plotting LEFT subplot data (3-Night Mean Temperature Only) ===")
    for i, (start_date, end_date, label) in enumerate(three_night_periods_left):
        try:
            # Select 3 nights between start and end date, hours start_hour-end_hour
            time_slice = slice(f"{start_date}T{start_hour:02d}:00:00", f"{end_date}T{end_hour:02d}:59:59")
            selected_data = data.sel(time=time_slice)
            
            # Filter to only include hours start_hour to end_hour for each night
            hourly_data = selected_data.where(
                (selected_data.time.dt.hour >= start_hour) & (selected_data.time.dt.hour <= end_hour),
                drop=True
            )
            
            # Compute mean across all selected times
            mean_temp = hourly_data[temp_var].mean(dim='time')
            
            ax_left.plot(data.x.values, mean_temp.values, 
                         color=colors_left[i], 
                         marker='o',
                         markersize=8,
                         markeredgecolor='black',
                         markeredgewidth=1,
                         linewidth=0,
                         linestyle='-',
                         label=label)
            
            # print(f"LEFT - {label}: Temp range {mean_temp.min().values:.2f}°C to {mean_temp.max().values:.2f}°C")
            
        except Exception as e:
            pass  # Silently handle error
            # print(f"Error plotting {label} on left subplot: {e}")
    
    # Style left subplot
    ax_left.set_xlabel('X, Distance Along Stream (m)', fontsize=24, fontfamily='Arial')
    ax_left.set_ylabel('Nocturnal Mean $S_t$ (°C)', fontsize=24, fontfamily='Arial')
    ax_left.set_xlim(0, 2000)
    ax_left.set_ylim(temp_ylim_left)
    ax_left.tick_params(axis='x', labelsize=24, length=6, width=1)
    ax_left.tick_params(axis='y', labelsize=24, length=6, width=1)
    ax_left.spines['top'].set_visible(False)
    ax_left.spines['right'].set_visible(False)
    ax_left.grid(False)
    
    for label in ax_left.get_xticklabels():
        label.set_fontfamily('Arial')
        label.set_fontsize(24)
    for label in ax_left.get_yticklabels():
        label.set_fontfamily('Arial')
        label.set_fontsize(24)
    
    # ===== RIGHT SUBPLOT: 3-Night Mean Temperature + Flow Accumulation (Dual-axis) =====
    
    # Plot 3-night mean temperature data on right subplot
    # print(f"\n=== Plotting RIGHT subplot data (3-Night Mean Temperature + Flow Accumulation) ===")
    for i, (start_date, end_date, label) in enumerate(three_night_periods_right):
        try:
            # Select 3 nights between start and end date, hours start_hour-end_hour
            time_slice = slice(f"{start_date}T{start_hour:02d}:00:00", f"{end_date}T{end_hour:02d}:59:59")
            selected_data = data.sel(time=time_slice)
            
            # Filter to only include hours start_hour to end_hour for each night
            hourly_data = selected_data.where(
                (selected_data.time.dt.hour >= start_hour) & (selected_data.time.dt.hour <= end_hour),
                drop=True
            )
            
            # Compute mean across all selected times
            mean_temp = hourly_data[temp_var].mean(dim='time')
            
            ax_right.plot(data.x.values, mean_temp.values, 
                        color=colors_right[i], 
                        marker='o',
                        markersize=8,
                        markeredgecolor='black',
                        markeredgewidth=1,
                        linewidth=0,
                        linestyle='-',
                        label=label)
            
            # print(f"RIGHT - {label}: Temp range {mean_temp.min().values:.2f}°C to {mean_temp.max().values:.2f}°C")
            
        except Exception as e:
            pass  # Silently handle error
            # print(f"Error plotting {label}: {e}")
    
    # Add flow accumulation on second y-axis (right subplot)
    if flow_data and method_colors and resolution_styles:
        ax_right_twin = ax_right.twinx()
        
        # print(f"Adding {len(flow_data)} flow accumulation datasets to right plot...")
        
        plotted_methods = set()
        plotted_resolutions = set()
        
        for key, df in flow_data.items():
            if len(df) > 0:
                resolution = int(key.split('m_')[0])
                method = key.split('m_')[1]
                
                ax_right_twin.plot(df.index, df['upslope_area_km2'],
                                linestyle=resolution_styles[resolution],
                                color=method_colors[method],
                                linewidth=3.0,
                                alpha=0.8)
                
                plotted_methods.add(method)
                plotted_resolutions.add(resolution)
        
        # Set flow accumulation y-axis properties
        ax_right_twin.set_ylim(0, 4.5)
        ax_right_twin.set_ylabel('Upstream Contributing Area (km²)', fontsize=24, fontfamily='Arial', color='darkblue')
        ax_right_twin.set_yticks([0, 1, 2, 3, 4])
        ax_right_twin.tick_params(axis='y', labelsize=24, length=6, width=1, colors='darkblue')
        ax_right_twin.spines['right'].set_color('darkblue')
        ax_right_twin.spines['top'].set_visible(False)
        
        for label in ax_right_twin.get_yticklabels():
            label.set_fontfamily('Arial')
            label.set_fontsize(24)
    
    # Style right subplot
    ax_right.set_xlabel('X, Distance Along Stream (m)', fontsize=24, fontfamily='Arial')
    ax_right.set_ylabel('Nocturnal Mean $S_t$ (°C)', fontsize=24, fontfamily='Arial')
    ax_right.set_xlim(0, 2000)
    ax_right.set_ylim(temp_ylim_right)
    ax_right.tick_params(axis='x', labelsize=24, length=6, width=1)
    ax_right.tick_params(axis='y', labelsize=24, length=6, width=1)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.grid(False)
    
    for label in ax_right.get_xticklabels():
        label.set_fontfamily('Arial')
        label.set_fontsize(24)
    for label in ax_right.get_yticklabels():
        label.set_fontfamily('Arial')
        label.set_fontsize(24)
    
    # ===== CREATE LEGENDS =====
    
    # Get period handles from left subplot (3-night mean temperature only)
    ax_left_handles, ax_left_labels = ax_left.get_legend_handles_labels()
    period_handles_left = ax_left_handles[:len(three_night_periods_left)]
    period_labels_left = ax_left_labels[:len(three_night_periods_left)]
    
    # Get period handles from right subplot (3-night mean temperature + flow accumulation)
    ax_right_handles, ax_right_labels = ax_right.get_legend_handles_labels()
    period_handles_right = ax_right_handles[:len(three_night_periods_right)]
    period_labels_right = ax_right_labels[:len(three_night_periods_right)]
    
    # Add legends for right subplot if flow data exists
    if flow_data and method_colors and resolution_styles:
        def format_method_name(method):
            method_names = {
                'dinf': 'D∞',
                'mdinf': 'MD∞', 
                'rho8': 'Rho8',
                'd8': 'D8',
                'fd8': 'FD8'
            }
            return method_names.get(method.lower(), method.upper())
        
        # Create legend handles for flow accumulation
        method_handles = []
        for method in sorted(plotted_methods):
            method_handles.append(Line2D([0], [0], color=method_colors[method], linewidth=4,
                                       label=format_method_name(method)))
        
        resolution_handles = []
        desired_order = [8, 16]
        plotted_in_order = [res for res in desired_order if res in plotted_resolutions]
        for res in plotted_in_order:
            resolution_handles.append(Line2D([0], [0], color='#34495E',
                                           linestyle=resolution_styles[res],
                                           linewidth=3, label=f'{res}m'))
        
        # Position legends below both plots at the same distance
        plt.subplots_adjust(bottom=0.22)
        
        # LEFT plot: Simple period legend
        ax_left.legend(period_handles_left, period_labels_left, frameon=False, loc='upper center', 
                      bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=20,
                      prop={'family': 'Arial', 'size': 20})
        
        # RIGHT plot: Periods legend + Methods legend + Resolutions legend
        legend1 = ax_right.legend(period_handles_right, period_labels_right, 
                                bbox_to_anchor=(0, -0.12), loc='upper center',
                                frameon=False, fontsize=20, ncol=1,
                                prop={'family': 'Arial', 'size': 20})
        
        # Methods legend
        legend2 = ax_right.legend(method_handles, [h.get_label() for h in method_handles],
                                bbox_to_anchor=(0.5, -0.12), loc='upper center', 
                                frameon=False, fontsize=20, ncol=3,
                                prop={'family': 'Arial', 'size': 20})
        
        # Resolutions legend
        legend3 = ax_right.legend(resolution_handles, [h.get_label() for h in resolution_handles],
                                bbox_to_anchor=(0.95, -0.12), loc='upper center',
                                frameon=False, fontsize=20, ncol=1,
                                prop={'family': 'Arial', 'size': 20})
        
        ax_right.add_artist(legend1)
        ax_right.add_artist(legend2)
    else:
        # Simple period legends for both plots at the same distance
        plt.subplots_adjust(bottom=0.22)
        
        # LEFT plot legend
        ax_left.legend(period_handles_left, period_labels_left, frameon=False, loc='upper center', 
                      bbox_to_anchor=(0.5, -0.10), ncol=2, fontsize=20,
                      prop={'family': 'Arial', 'size': 20})
        
        # RIGHT plot legend
        ax_right.legend(period_handles_right, period_labels_right, frameon=False, loc='upper center', 
                       bbox_to_anchor=(0.5, -0.10), ncol=2, fontsize=20,
                       prop={'family': 'Arial', 'size': 20})
    
    plt.tight_layout()
    plt.show()
    
    # print("\n Side-by-side 3-night mean temperature plots complete!")
    # print(" Left plot: 3-night mean temperature only (single-axis)")
    # print(" Right plot: 3-night mean temperature with flow accumulation (dual-axis)")


def plot_diurnal_and_amplitude_side_by_side(data, dates, start_hour=12, end_hour=16, 
                                             canopy_csv_path=None):
    """
    Create side-by-side plots: 
    - Left: Diurnal stream temperature along the cable (with optional canopy fraction)
    - Right: Daily temperature amplitudes
    """
    
    import numpy as np
    import xarray as xr
    
    # Determine temperature variable name
    if 'st' in data.variables:
        temp_var = 'st'
    elif 'temperature' in data.variables:
        temp_var = 'temperature'
    elif 'temp' in data.variables:
        temp_var = 'temp'
    else:
        temp_var = list(data.data_vars.keys())[0]
    
    # Load canopy data if path is provided
    canopy_available = False
    if canopy_csv_path:
        try:
            canopy_data = pd.read_csv(canopy_csv_path)
            main_stream_canopy = canopy_data[canopy_data['stream_id'] == 0].copy()
            main_stream_canopy = main_stream_canopy.sort_values('distance_along_stream')
            canopy_distance = main_stream_canopy['distance_along_stream'].values
            canopy_values = main_stream_canopy['canopy_fraction'].values
            canopy_available = True
        except Exception as e:
            canopy_available = False
    
    # Set seaborn theme for professional appearance
    sns.set_theme(style="white")
    
    # Create the plot with two subplots side by side
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(20, 8), sharey=False)
    
    # Load burned features for background shading
    try:
        burned_features = pd.read_csv('dts_burned_features.csv')
        
        FEATURE_COLORS = {
            0: None,
            1: '#2F2F2F',    # Ditch inflow 1 (630m)
            2: '#2F2F2F',    # Ditch inflow 2 (700m)
            4: '#2F2F2F',    # Tributary entering area (1470m)
            8: '#2F2F2F',    # Upland Spring 2 (1860m)
            16: '#2F2F2F',   # Upstream spring (1900m)
            32: '#2F2F2F',   # Episodic ditch (450-470m)
            64: '#2F2F2F',   # Esker region (1000-1300m)
        }
        
        # Add background shading to both subplots
        for ax in [ax_left, ax_right]:
            for feature_value, color in FEATURE_COLORS.items():
                if feature_value == 0 or color is None:
                    continue
                feature_mask = burned_features['burned_features'] == feature_value
                if feature_mask.sum() > 0:
                    feature_x = burned_features.loc[feature_mask, 'x_dts'].values
                    min_x, max_x = feature_x.min(), feature_x.max()
                    ax.axvspan(min_x, max_x, color=color, alpha=0.12, zorder=0)
        
    except Exception as e:
        pass
    
    # Generate coolwarm colors for dates
    n_dates = len(dates)
    if n_dates > 0:
        colors = sns.color_palette("coolwarm", n_dates)
    else:
        colors = []
    
    # ===== LEFT SUBPLOT: Diurnal Mean Temperature =====
    
    for i, date in enumerate(dates):
        try:
            time_slice = slice(f"{date}T{start_hour:02d}:00:00", f"{date}T{end_hour:02d}:59:59")
            selected_data = data.sel(time=time_slice)
            mean_temp = selected_data[temp_var].mean(dim='time')
            
            ax_left.plot(data.x.values, mean_temp.values, 
                        color=colors[i], 
                        marker='o',
                        markersize=9,
                        markeredgecolor='black',
                        markeredgewidth=1,
                        linewidth=1.5,
                        linestyle='-',
                        label=f'{date}')
            
        except Exception as e:
            pass
    
    # Add canopy fraction on secondary y-axis if available
    if canopy_available:
        ax_left_twin = ax_left.twinx()
        
        ax_left_twin.plot(canopy_distance, canopy_values, 
                         color='darkgreen', 
                         marker='s',
                         markersize=6,
                         markeredgecolor='black',
                         markeredgewidth=0.5,
                         linewidth=2,
                         linestyle='--',
                         label='Canopy Fraction',
                         alpha=0.8)
        
        ax_left_twin.set_ylabel('Canopy Fraction', fontsize=24, fontfamily='Arial', color='darkgreen')
        ax_left_twin.tick_params(axis='y', labelcolor='darkgreen', labelsize=24, length=6, width=1)
        ax_left_twin.set_ylim(0, 2)
        ax_left_twin.spines['top'].set_visible(False)
        ax_left_twin.spines['right'].set_color('darkgreen')
        ax_left_twin.spines['right'].set_linewidth(2)
        
        for label in ax_left_twin.get_yticklabels():
            label.set_fontfamily('Arial')
            label.set_fontsize(24)
    
    # Style left subplot
    ax_left.set_xlabel('X, Distance Along Stream (m)', fontsize=24, fontfamily='Arial')
    ax_left.set_ylabel('Diurnal Mean $S_t$ (°C)', fontsize=24, fontfamily='Arial')
    ax_left.set_xlim(0, 2000)
    ax_left.set_xticks([0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000])
    ax_left.set_ylim(0, 17)
    ax_left.tick_params(axis='x', labelsize=24, length=6, width=1)
    ax_left.tick_params(axis='y', labelsize=24, length=6, width=1)
    ax_left.spines['top'].set_visible(False)
    ax_left.spines['right'].set_visible(False)
    ax_left.grid(False)
    
    for label in ax_left.get_xticklabels():
        label.set_fontfamily('Arial')
        label.set_fontsize(24)
    for label in ax_left.get_yticklabels():
        label.set_fontfamily('Arial')
        label.set_fontsize(24)
    
    # ===== RIGHT SUBPLOT: Daily Temperature Amplitude =====
    
    # Use coolwarm_r (reversed) for amplitude plot
    colors_amplitude = sns.color_palette("coolwarm_r", n_dates)
    
    for i, date in enumerate(dates):
        try:
            # Create time slice for the entire day (24 hours)
            time_slice = slice(f"{date}T00:00:00", f"{date}T23:59:59")
            selected_data = data.sel(time=time_slice)
            
            # Calculate daily amplitude (max - min) for each x position
            daily_max = selected_data[temp_var].max(dim='time')
            daily_min = selected_data[temp_var].min(dim='time')
            daily_amplitude = daily_max - daily_min
            
            ax_right.plot(data.x.values, daily_amplitude.values, 
                         color=colors_amplitude[i], 
                         marker='o',
                         markersize=9,
                         markeredgecolor='black',
                         markeredgewidth=1,
                         linewidth=1.5,
                         linestyle='-',
                         label=f'{date}')
            
        except Exception as e:
            pass
    
    # Style right subplot
    ax_right.set_xlabel('X, Distance Along Stream (m)', fontsize=24, fontfamily='Arial')
    ax_right.set_ylabel('Daily Temperature Amplitude (°C)', fontsize=24, fontfamily='Arial')
    ax_right.set_xlim(0, 2000)
    ax_right.set_xticks([0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000])
    ax_right.set_ylim(0, 8)
    ax_right.tick_params(axis='x', labelsize=24, length=6, width=1)
    ax_right.tick_params(axis='y', labelsize=24, length=6, width=1)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.grid(False)
    
    for label in ax_right.get_xticklabels():
        label.set_fontfamily('Arial')
        label.set_fontsize(24)
    for label in ax_right.get_yticklabels():
        label.set_fontfamily('Arial')
        label.set_fontsize(24)
    
    # ===== CREATE LEGENDS =====
    
    # Get handles from both subplots
    ax_left_handles, ax_left_labels = ax_left.get_legend_handles_labels()
    ax_right_handles, ax_right_labels = ax_right.get_legend_handles_labels()
    
    # Calculate legend layout for both subplots
    # Left subplot (with or without canopy)
    if canopy_available:
        canopy_handles, canopy_labels = ax_left_twin.get_legend_handles_labels()
        all_left_handles = ax_left_handles + canopy_handles
        all_left_labels = ax_left_labels + canopy_labels
    else:
        all_left_handles = ax_left_handles
        all_left_labels = ax_left_labels
    
    # Calculate layout for both legends
    ncol_left = min(3, len(all_left_labels))
    ncol_right = min(3, len(dates))
    legend_rows_left = (len(all_left_labels) + ncol_left - 1) // ncol_left
    legend_rows_right = (len(dates) + ncol_right - 1) // ncol_right
    
    # Use the maximum rows to ensure consistent spacing
    max_legend_rows = max(legend_rows_left, legend_rows_right)
    
    # Set bottom adjustment once for the entire figure
    plt.subplots_adjust(bottom=0.15 + (max_legend_rows - 1) * 0.02)
    
    # Use same y-position for both legends (consistent distance from bottom)
    legend_y_pos = -0.12 - (max_legend_rows - 1) * 0.03
    
    # Create legends at the same distance from bottom
    ax_left.legend(all_left_handles, all_left_labels, frameon=False, loc='upper center', 
                  bbox_to_anchor=(0.5, legend_y_pos), ncol=ncol_left, fontsize=20,
                  prop={'family': 'Arial', 'size': 20})
    
    ax_right.legend(frameon=False, loc='upper center', 
                   bbox_to_anchor=(0.5, legend_y_pos), ncol=ncol_right, fontsize=20,
                   prop={'family': 'Arial', 'size': 20})
    
    plt.tight_layout()
    plt.show()


def plot_temperature_and_twi_side_by_side(data, dates_left, dates_right, start_hour=0, end_hour=3):
    """
    Create side-by-side plots with TWI data on both plots, but y-axis ticks only on the right:
    - Left plot: Temperature (left y-axis) + 8m TWI (right y-axis, hidden ticks/labels)
    - Right plot: Temperature (left y-axis) + 16m TWI (right y-axis, visible ticks/labels)
    Both plots include horizontal reference lines at TWI values 12, 14, 16.
    
    """
    
    import numpy as np
    import xarray as xr
    
    # Determine temperature variable name
    if 'st' in data.variables:
        temp_var = 'st'
    elif 'temperature' in data.variables:
        temp_var = 'temperature'
    elif 'temp' in data.variables:
        temp_var = 'temp'
    else:
        temp_var = list(data.data_vars.keys())[0]
    
    # Load TWI data
    twi_data, method_colors, resolution_styles = load_twi_data_for_dual_axis()
    
    # Separate TWI data by resolution
    data_8m = {}
    data_16m = {}
    
    if twi_data:
        for key, df in twi_data.items():
            resolution = int(key.split('m_')[0])
            if resolution == 8:
                data_8m[key] = df
            elif resolution == 16:
                data_16m[key] = df
    
    # Set seaborn theme for professional appearance
    sns.set_theme(style="white")
    
    # Create the plot with two subplots side by side
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(20, 8), sharey=False)
    
    # Load burned features for background shading
    try:
        burned_features = pd.read_csv('dts_burned_features.csv')
        
        FEATURE_COLORS = {
            0: None,
            1: '#2F2F2F',    # Ditch inflow 1 (630m)
            2: '#2F2F2F',    # Ditch inflow 2 (700m)
            4: '#2F2F2F',    # Tributary entering area (1470m)
            8: '#2F2F2F',    # Upland Spring 2 (1860m)
            16: '#2F2F2F',   # Upstream spring (1900m)
            32: '#2F2F2F',   # Episodic ditch (450-470m)
            64: '#2F2F2F',   # Esker region (1000-1300m)
        }
        
        # Add background shading to both subplots
        for ax in [ax_left, ax_right]:
            for feature_value, color in FEATURE_COLORS.items():
                if feature_value == 0 or color is None:
                    continue
                feature_mask = burned_features['burned_features'] == feature_value
                if feature_mask.sum() > 0:
                    feature_x = burned_features.loc[feature_mask, 'x_dts'].values
                    min_x, max_x = feature_x.min(), feature_x.max()
                    ax.axvspan(min_x, max_x, color=color, alpha=0.12, zorder=0)
        
    except Exception as e:
        pass
    
    # Generate coolwarm colors for LEFT plot dates
    n_dates_left = len(dates_left)
    if n_dates_left > 0:
        colors_left = sns.color_palette("coolwarm", n_dates_left)
    
    # Generate coolwarm colors for RIGHT plot dates
    n_dates_right = len(dates_right)
    if n_dates_right > 0:
        colors_right = sns.color_palette("coolwarm", n_dates_right)
    
    # ===== LEFT SUBPLOT: Temperature (left y-axis) + 8m TWI (right y-axis, no ticks) =====
    
    for i, date in enumerate(dates_left):
        try:
            time_slice = slice(f"{date}T{start_hour:02d}:00:00", f"{date}T{end_hour:02d}:59:59")
            selected_data = data.sel(time=time_slice)
            mean_temp = selected_data[temp_var].mean(dim='time')
            
            ax_left.plot(data.x.values, mean_temp.values, 
                        color=colors_left[i], 
                        marker='o',
                        markersize=7,
                        markeredgecolor='black',
                        markeredgewidth=1,
                        linewidth=0,
                        linestyle='-',
                        label=f'{date}')
            
            
        except Exception as e:
            pass
    
    # Add 8m TWI on right y-axis (left subplot) - without visible ticks
    ax_left_twin = ax_left.twinx()
    
    plotted_methods_8m = set()
    if len(data_8m) > 0:
        for key, df in data_8m.items():
            if len(df) > 0:
                method = key.split('m_')[1]
                ax_left_twin.plot(df.index, df['twi_value'],
                                linestyle='--',
                                color=method_colors[method],
                                linewidth=3.0,
                                alpha=0.8)
                plotted_methods_8m.add(method)
    
    # Style left subplot
    ax_left.set_xlabel('X, Distance Along Stream (m)', fontsize=24, fontfamily='Arial')
    ax_left.set_ylabel('Nocturnal Mean $S_t$ (°C)', fontsize=24, fontfamily='Arial')
    ax_left.set_xlim(0, 2000)
    ax_left.set_xticks([0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000])
    ax_left.set_ylim(0, 15)
    ax_left.tick_params(axis='x', labelsize=24, length=6, width=1)
    ax_left.tick_params(axis='y', labelsize=24, length=6, width=1)
    ax_left.spines['top'].set_visible(False)
    ax_left.spines['right'].set_visible(False)
    ax_left.grid(False)
    
    for label in ax_left.get_xticklabels():
        label.set_fontfamily('Arial')
        label.set_fontsize(24)
    for label in ax_left.get_yticklabels():
        label.set_fontfamily('Arial')
        label.set_fontsize(24)
    
    # Style TWI y-axis (right) for left subplot - HIDE ticks and labels
    ax_left_twin.set_ylim(10, 19)
    ax_left_twin.set_yticks([10, 12, 14, 16, 18])
    ax_left_twin.set_yticklabels([])  # Hide tick labels
    ax_left_twin.tick_params(axis='y', length=0, labelsize=0, labelleft=False, labelright=False)
    ax_left_twin.spines['top'].set_visible(False)
    ax_left_twin.spines['right'].set_visible(False)  # Hide right spine
    
    # Add horizontal lines at TWI reference values on left plot (using TWI axis)
    for twi_value in [12, 14, 16]:
        ax_left_twin.axhline(y=twi_value, color='darkgray', linestyle='--', alpha=0.5, linewidth=1.5, zorder=1)
    
    # Add DEM label on left plot (right position)
    ax_left.text(0.95, 0.95, '8m DEM', transform=ax_left.transAxes,
                fontsize=20, fontfamily='Arial',
                verticalalignment='top', horizontalalignment='right')
    
    # ===== RIGHT SUBPLOT: Temperature (left y-axis) + 16m TWI (right y-axis) =====
    
    print(f"\n=== Plotting RIGHT subplot (Temperature + 16m TWI) ===")
    for i, date in enumerate(dates_right):
        try:
            time_slice = slice(f"{date}T{start_hour:02d}:00:00", f"{date}T{end_hour:02d}:59:59")
            selected_data = data.sel(time=time_slice)
            mean_temp = selected_data[temp_var].mean(dim='time')
            
            ax_right.plot(data.x.values, mean_temp.values, 
                         color=colors_right[i], 
                         marker='o',
                         markersize=7,
                         markeredgecolor='black',
                         markeredgewidth=1,
                         linewidth=0,
                         linestyle='-',
                         label=f'{date}')
            
            
        except Exception as e:
            pass
    
    # Add 16m TWI on right y-axis (right subplot)
    ax_right_twin = ax_right.twinx()
    
    plotted_methods_16m = set()
    if len(data_16m) > 0:
        for key, df in data_16m.items():
            if len(df) > 0:
                method = key.split('m_')[1]
                ax_right_twin.plot(df.index, df['twi_value'],
                                 linestyle='-',
                                 color=method_colors[method],
                                 linewidth=3.0,
                                 alpha=0.8)
                plotted_methods_16m.add(method)
    
    # Style right subplot
    ax_right.set_xlabel('X, Distance Along Stream (m)', fontsize=24, fontfamily='Arial')
    ax_right.set_ylabel('Nocturnal Mean $S_t$ (°C)', fontsize=24, fontfamily='Arial')
    ax_right.set_xlim(0, 2000)
    ax_right.set_xticks([0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000])
    ax_right.set_ylim(0, 15)
    ax_right.tick_params(axis='x', labelsize=24, length=6, width=1)
    ax_right.tick_params(axis='y', labelsize=24, length=6, width=1)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.grid(False)
    
    for label in ax_right.get_xticklabels():
        label.set_fontfamily('Arial')
        label.set_fontsize(24)
    for label in ax_right.get_yticklabels():
        label.set_fontfamily('Arial')
        label.set_fontsize(24)
    
    # Style TWI y-axis (right) for right subplot
    ax_right_twin.set_ylabel('TWI', fontsize=24, fontfamily='Arial', color='darkblue')
    ax_right_twin.set_ylim(10, 19)
    ax_right_twin.set_yticks([10, 12, 14, 16, 18])
    ax_right_twin.tick_params(axis='y', labelcolor='darkblue', labelsize=24, length=6, width=1)
    ax_right_twin.spines['top'].set_visible(False)
    ax_right_twin.spines['right'].set_color('darkblue')
    ax_right_twin.spines['right'].set_linewidth(2)
    
    for label in ax_right_twin.get_yticklabels():
        label.set_fontfamily('Arial')
        label.set_fontsize(24)
    
    # Add horizontal lines at TWI reference values on right plot (using TWI axis)
    for twi_value in [12, 14, 16]:
        ax_right_twin.axhline(y=twi_value, color='darkgray', linestyle='--', alpha=0.5, linewidth=1.5, zorder=1)
    
    # Add DEM label on right plot (right position)
    ax_right.text(0.95, 0.95, '16m DEM', transform=ax_right.transAxes,
                  fontsize=20, fontfamily='Arial',
                  verticalalignment='top', horizontalalignment='right')
    
    # ===== CREATE LEGENDS =====
    
    def format_method_name(method):
        method_names = {
            'dinf': 'D∞',
            'mdinf': 'MD∞', 
            'rho8': 'Rho8',
            'd8': 'D8',
            'fd8': 'FD8'
        }
        return method_names.get(method.lower(), method.upper())
    
    # Get date handles from both subplots
    ax_left_handles, ax_left_labels = ax_left.get_legend_handles_labels()
    ax_right_handles, ax_right_labels = ax_right.get_legend_handles_labels()
    
    # Create TWI method handles for left subplot (8m)
    method_handles_8m = []
    for method in sorted(plotted_methods_8m):
        method_handles_8m.append(Line2D([0], [0], color=method_colors[method], linewidth=4,
                                       label=format_method_name(method)))
    
    # Create TWI method handles for right subplot (16m)
    method_handles_16m = []
    for method in sorted(plotted_methods_16m):
        method_handles_16m.append(Line2D([0], [0], color=method_colors[method], linewidth=4,
                                        label=format_method_name(method)))
    
    # Calculate max legend rows for consistent positioning
    left_legend_items = len(dates_left) + len(method_handles_8m)
    right_legend_items = len(dates_right) + len(method_handles_16m)
    
    ncol = 3
    left_rows = (left_legend_items + ncol - 1) // ncol
    right_rows = (right_legend_items + ncol - 1) // ncol
    max_rows = max(left_rows, right_rows)
    
    plt.subplots_adjust(bottom=0.15 + (max_rows - 1) * 0.02)
    legend_y_pos = -0.12 - (max_rows - 1) * 0.03
    
    # Left subplot: Combine temperature dates and 8m TWI methods
    all_left_handles = ax_left_handles + method_handles_8m
    all_left_labels = ax_left_labels + [h.get_label() for h in method_handles_8m]
    
    ax_left.legend(all_left_handles, all_left_labels, frameon=False, loc='upper center',
                  bbox_to_anchor=(0.5, legend_y_pos), ncol=ncol, fontsize=20,
                  prop={'family': 'Arial', 'size': 20})
    
    # Right subplot: Combine temperature dates and 16m TWI methods
    all_right_handles = ax_right_handles + method_handles_16m
    all_right_labels = ax_right_labels + [h.get_label() for h in method_handles_16m]
    
    ax_right.legend(all_right_handles, all_right_labels, frameon=False, loc='upper center',
                   bbox_to_anchor=(0.5, legend_y_pos), ncol=ncol, fontsize=20,
                   prop={'family': 'Arial', 'size': 20})
    
    plt.tight_layout()
    plt.show()


def plot_temperature_dtw_and_twi_side_by_side(data, dates_left, dates_right, start_hour=0, end_hour=3):
    """
    Create side-by-side plots:
    - Left plot: Temperature (left y-axis) + DTW (right y-axis, visible)
    - Right plot: Temperature (left y-axis) + 16m TWI (right y-axis, visible)
    Both plots include horizontal reference lines at TWI values 12, 14, 16.

    """
    
    import numpy as np
    import xarray as xr
    
    # Determine temperature variable name
    if 'st' in data.variables:
        temp_var = 'st'
    elif 'temperature' in data.variables:
        temp_var = 'temperature'
    elif 'temp' in data.variables:
        temp_var = 'temp'
    else:
        temp_var = list(data.data_vars.keys())[0]
    
    # Load DTW data
    base_dir = r'C:\Users\pparvizi24\OneDrive - University of Oulu and Oamk\Parsa-PHD-OneDrive\DTS\Dataset - All\WBT_data\WBT_data'
    dtw_clipped_dir = os.path.join(base_dir, 'dtw_clipped')
    csv_file = os.path.join(dtw_clipped_dir, 'dtw_1d_1ha_clipped.csv')
    
    dtw_data = None
    try:
        dtw_df = pd.read_csv(csv_file, index_col='stream_length')
        dtw_data = dtw_df
    except Exception as e:
        pass
    
    # Load TWI data (8m for left plot, 16m for right plot)
    twi_data, method_colors, resolution_styles = load_twi_data_for_dual_axis()
    
    # Separate TWI data by resolution
    data_8m = {}
    data_16m = {}
    if twi_data:
        for key, df in twi_data.items():
            resolution = int(key.split('m_')[0])
            if resolution == 8:
                data_8m[key] = df
            elif resolution == 16:
                data_16m[key] = df
    
    # Set seaborn theme for professional appearance
    sns.set_theme(style="white")
    
    # Create the plot with two subplots side by side
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(20, 8), sharey=False)
    
    # Load burned features for background shading
    try:
        burned_features = pd.read_csv('dts_burned_features.csv')
        
        FEATURE_COLORS = {
            0: None,
            1: '#2F2F2F',    # Ditch inflow 1 (630m)
            2: '#2F2F2F',    # Ditch inflow 2 (700m)
            4: '#2F2F2F',    # Tributary entering area (1470m)
            8: '#2F2F2F',    # Upland Spring 2 (1860m)
            16: '#2F2F2F',   # Upstream spring (1900m)
            32: '#2F2F2F',   # Episodic ditch (450-470m)
            64: '#2F2F2F',   # Esker region (1000-1300m)
        }
        
        # Add background shading to both subplots
        for ax in [ax_left, ax_right]:
            for feature_value, color in FEATURE_COLORS.items():
                if feature_value == 0 or color is None:
                    continue
                feature_mask = burned_features['burned_features'] == feature_value
                if feature_mask.sum() > 0:
                    feature_x = burned_features.loc[feature_mask, 'x_dts'].values
                    min_x, max_x = feature_x.min(), feature_x.max()
                    ax.axvspan(min_x, max_x, color=color, alpha=0.12, zorder=0)
        
    except Exception as e:
        pass
    
    # Generate coolwarm colors for LEFT plot dates
    n_dates_left = len(dates_left)
    if n_dates_left > 0:
        colors_left = sns.color_palette("coolwarm", n_dates_left)
    
    # Generate coolwarm colors for RIGHT plot dates
    n_dates_right = len(dates_right)
    if n_dates_right > 0:
        colors_right = sns.color_palette("coolwarm", n_dates_right)
    
    # ===== LEFT SUBPLOT: Temperature (left y-axis) + 8m TWI (hidden) + DTW (right y-axis) =====
    
    for i, date in enumerate(dates_left):
        try:
            time_slice = slice(f"{date}T{start_hour:02d}:00:00", f"{date}T{end_hour:02d}:59:59")
            selected_data = data.sel(time=time_slice)
            mean_temp = selected_data[temp_var].mean(dim='time')
            
            ax_left.plot(data.x.values, mean_temp.values, 
                        color=colors_left[i], 
                        marker='o',
                        markersize=7,
                        markeredgecolor='black',
                        markeredgewidth=1,
                        linewidth=0,
                        linestyle='-',
                        label=f'{date}')
            
            
        except Exception as e:
            pass
    
    # Add 8m TWI on right y-axis (left subplot) - HIDDEN ticks
    ax_left_twin_twi = ax_left.twinx()
    
    plotted_methods_8m = set()
    if len(data_8m) > 0:
        for key, df in data_8m.items():
            if len(df) > 0:
                method = key.split('m_')[1]
                ax_left_twin_twi.plot(df.index, df['twi_value'],
                                linestyle='--',
                                color=method_colors[method],
                                linewidth=3.0,
                                alpha=0.8)
                plotted_methods_8m.add(method)
    
    # Style TWI y-axis (hidden)
    ax_left_twin_twi.set_ylim(10, 19)
    ax_left_twin_twi.set_yticks([10, 12, 14, 16, 18])
    ax_left_twin_twi.set_yticklabels([])  # Hide tick labels
    ax_left_twin_twi.tick_params(axis='y', length=0, labelsize=0, labelleft=False, labelright=False)
    ax_left_twin_twi.spines['top'].set_visible(False)
    ax_left_twin_twi.spines['right'].set_visible(False)
    
    # Add horizontal lines at TWI reference values on left plot (using TWI axis)
    for twi_value in [12, 14, 16]:
        ax_left_twin_twi.axhline(y=twi_value, color='darkgray', linestyle='--', alpha=0.5, linewidth=1.5, zorder=1)
    
    # Add DTW on second right y-axis (left subplot) - VISIBLE ticks
    ax_left_twin_dtw = ax_left.twinx()
    
    if dtw_data is not None:
        # Use smoothed DTW data for cleaner visualization (like your working code)
        dtw_column = 'dtw_smooth' if 'dtw_smooth' in dtw_data.columns else 'dtw_value'
        
        if dtw_column in dtw_data.columns:
            ax_left_twin_dtw.plot(dtw_data.index, dtw_data[dtw_column],
                            color='red',
                            linewidth=3.0,
                            linestyle='--',
                            alpha=0.8,
                            label='DTW')
    
    # Style left subplot
    ax_left.set_xlabel('X, Distance Along Stream (m)', fontsize=24, fontfamily='Arial')
    ax_left.set_ylabel('Nocturnal Mean $S_t$ (°C)', fontsize=24, fontfamily='Arial')
    ax_left.set_xlim(0, 2000)
    ax_left.set_xticks([0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000])
    ax_left.set_ylim(0, 15)
    ax_left.tick_params(axis='x', labelsize=24, length=6, width=1)
    ax_left.tick_params(axis='y', labelsize=24, length=6, width=1)
    ax_left.spines['top'].set_visible(False)
    ax_left.spines['right'].set_visible(False)
    ax_left.grid(False)
    
    for label in ax_left.get_xticklabels():
        label.set_fontfamily('Arial')
        label.set_fontsize(24)
    for label in ax_left.get_yticklabels():
        label.set_fontfamily('Arial')
        label.set_fontsize(24)
    
    # Style DTW y-axis (right) for left subplot
    ax_left_twin_dtw.set_ylabel('DTW (m)', fontsize=24, fontfamily='Arial', color='red')
    ax_left_twin_dtw.set_ylim(0, 0.3)
    ax_left_twin_dtw.tick_params(axis='y', labelcolor='red', labelsize=24, length=6, width=1)
    ax_left_twin_dtw.spines['top'].set_visible(False)
    ax_left_twin_dtw.spines['right'].set_color('red')
    ax_left_twin_dtw.spines['right'].set_linewidth(2)
    
    for label in ax_left_twin_dtw.get_yticklabels():
        label.set_fontfamily('Arial')
        label.set_fontsize(24)
    
    # Add DEM label on left plot
    ax_left.text(0.95, 0.95, '8m DEM', transform=ax_left.transAxes,
                fontsize=20, fontfamily='Arial',
                verticalalignment='top', horizontalalignment='right')
    
    # ===== RIGHT SUBPLOT: Temperature (left y-axis) + 16m TWI (right y-axis) =====
    
    for i, date in enumerate(dates_right):
        try:
            time_slice = slice(f"{date}T{start_hour:02d}:00:00", f"{date}T{end_hour:02d}:59:59")
            selected_data = data.sel(time=time_slice)
            mean_temp = selected_data[temp_var].mean(dim='time')
            
            ax_right.plot(data.x.values, mean_temp.values, 
                         color=colors_right[i], 
                         marker='o',
                         markersize=7,
                         markeredgecolor='black',
                         markeredgewidth=1,
                         linewidth=0,
                         linestyle='-',
                         label=f'{date}')
            
            
        except Exception as e:
            pass
    
    # Add 16m TWI on right y-axis (right subplot)
    ax_right_twin = ax_right.twinx()
    
    plotted_methods_16m = set()
    if len(data_16m) > 0:
        for key, df in data_16m.items():
            if len(df) > 0:
                method = key.split('m_')[1]
                ax_right_twin.plot(df.index, df['twi_value'],
                                 linestyle='-',
                                 color=method_colors[method],
                                 linewidth=3.0,
                                 alpha=0.8)
                plotted_methods_16m.add(method)
    
    # Style right subplot
    ax_right.set_xlabel('X, Distance Along Stream (m)', fontsize=24, fontfamily='Arial')
    ax_right.set_ylabel('Nocturnal Mean $S_t$ (°C)', fontsize=24, fontfamily='Arial')
    ax_right.set_xlim(0, 2000)
    ax_right.set_xticks([0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000])
    ax_right.set_ylim(0, 15)
    ax_right.tick_params(axis='x', labelsize=24, length=6, width=1)
    ax_right.tick_params(axis='y', labelsize=24, length=6, width=1)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.grid(False)
    
    for label in ax_right.get_xticklabels():
        label.set_fontfamily('Arial')
        label.set_fontsize(24)
    for label in ax_right.get_yticklabels():
        label.set_fontfamily('Arial')
        label.set_fontsize(24)
    
    # Style TWI y-axis (right) for right subplot
    ax_right_twin.set_ylabel('TWI', fontsize=24, fontfamily='Arial', color='darkblue')
    ax_right_twin.set_ylim(10, 19)
    ax_right_twin.set_yticks([10, 12, 14, 16, 18])
    ax_right_twin.tick_params(axis='y', labelcolor='darkblue', labelsize=24, length=6, width=1)
    ax_right_twin.spines['top'].set_visible(False)
    ax_right_twin.spines['right'].set_color('darkblue')
    ax_right_twin.spines['right'].set_linewidth(2)
    
    for label in ax_right_twin.get_yticklabels():
        label.set_fontfamily('Arial')
        label.set_fontsize(24)
    
    # Add horizontal lines at TWI reference values on right plot (using TWI axis)
    for twi_value in [12, 14, 16]:
        ax_right_twin.axhline(y=twi_value, color='darkgray', linestyle='--', alpha=0.5, linewidth=1.5, zorder=1)
    
    # Add DEM label on right plot
    ax_right.text(0.95, 0.95, '16m DEM', transform=ax_right.transAxes,
                  fontsize=20, fontfamily='Arial',
                  verticalalignment='top', horizontalalignment='right')
    
    # ===== CREATE LEGENDS =====
    
    def format_method_name(method):
        method_names = {
            'dinf': 'D∞',
            'mdinf': 'MD∞', 
            'rho8': 'Rho8',
            'd8': 'D8',
            'fd8': 'FD8'
        }
        return method_names.get(method.lower(), method.upper())
    
    # Get date handles from both subplots
    ax_left_handles, ax_left_labels = ax_left.get_legend_handles_labels()
    ax_right_handles, ax_right_labels = ax_right.get_legend_handles_labels()
    
    # Create TWI method handles for left subplot (8m)
    method_handles_8m = []
    for method in sorted(plotted_methods_8m):
        method_handles_8m.append(Line2D([0], [0], color=method_colors[method], linewidth=4,
                                       label=format_method_name(method)))
    
    # Create TWI method handles for right subplot (16m)
    method_handles_16m = []
    for method in sorted(plotted_methods_16m):
        method_handles_16m.append(Line2D([0], [0], color=method_colors[method], linewidth=4,
                                        label=format_method_name(method)))
    
    # Calculate max legend rows for consistent positioning
    left_legend_items = len(dates_left) + len(method_handles_8m)
    right_legend_items = len(dates_right) + len(method_handles_16m)
    
    ncol = 3
    left_rows = (left_legend_items + ncol - 1) // ncol
    right_rows = (right_legend_items + ncol - 1) // ncol
    max_rows = max(left_rows, right_rows)
    
    plt.subplots_adjust(bottom=0.15 + (max_rows - 1) * 0.02)
    legend_y_pos = -0.12 - (max_rows - 1) * 0.03
    
    # Left subplot: Combine temperature dates and 8m TWI methods (DTW shown on y-axis label)
    all_left_handles = ax_left_handles + method_handles_8m
    all_left_labels = ax_left_labels + [h.get_label() for h in method_handles_8m]
    
    ax_left.legend(all_left_handles, all_left_labels, frameon=False, loc='upper center',
                  bbox_to_anchor=(0.5, legend_y_pos), ncol=ncol, fontsize=20,
                  prop={'family': 'Arial', 'size': 20})
    
    # Right subplot: Combine temperature dates and TWI methods
    all_right_handles = ax_right_handles + method_handles_16m
    all_right_labels = ax_right_labels + [h.get_label() for h in method_handles_16m]
    
    ax_right.legend(all_right_handles, all_right_labels, frameon=False, loc='upper center',
                   bbox_to_anchor=(0.5, legend_y_pos), ncol=ncol, fontsize=20,
                   prop={'family': 'Arial', 'size': 20})
    
    plt.tight_layout()
    plt.show()