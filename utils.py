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
    
    print(f"DTS loaded: {len(dts_x)} points from {dts_x.min():.1f}m to {dts_x.max():.1f}m")
    print(f"Temperature variable: '{temp_var}'")
    
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
    
    if verbose:
        print("Burning POI features into vector...")
    
    # Burn POI features
    for feature_type, stream_distance in correct_poi_distances.items():
        if feature_type not in BURN_VALUES:
            continue
            
        # Find DTS points near this stream distance
        distances = np.abs(dts_x - stream_distance)
        nearby_indices = np.where(distances <= tolerance)[0]
        
        if len(nearby_indices) > 0:
            burned_vector.loc[nearby_indices, 'burned_features'] = BURN_VALUES[feature_type]
            if verbose:
                print(f"   Burned {feature_type} at {stream_distance}m: {len(nearby_indices)} points")
    
    # Burn esker regions if provided
    if esker_regions:
        if verbose:
            print("Burning esker regions...")
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
                    if verbose:
                        print(f"   Burned {esker_name} ({start}-{end}m): {len(available_indices)} points")
    
    # Summary
    if verbose:
        unique_values = burned_vector['burned_features'].unique()
        print(f"\nBurned Vector Summary:")
        print(f"   Unique values: {sorted(unique_values)}")
    
    # Save
    burned_vector.to_csv(output_file, index=False)
    if verbose:
        print(f"Saved: {output_file}\n")
    
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
        
        FEATURE_COLORS = {
            1: '#2F2F2F',   # Ditch inflow 1 (630m)
            2: '#2F2F2F',   # Ditch inflow 2 (700m)
            4: '#2F2F2F',   # Tributary entering area (1470m)
            8: '#2F2F2F',   # Upland Spring 2 (1860m)
            16: '#2F2F2F',  # Upstream spring (1900m)
            32: '#2F2F2F',  # Episodic ditch (380-450m)
            64: '#2F2F2F',  # Esker region (1000-1300m)
        }
        
        FEATURE_LABELS = {
            1: 'Ditch inflow 1 (630m)',
            2: 'Ditch inflow 2 (700m)',
            4: 'Tributary (1470m)',
            8: 'Upland Spring 2 (1860m)',
            16: 'Upstream spring (1890m)',
            32: 'Episodic ditch (380-450m)',
            64: 'Esker region (1000-1300m)'
        }
        
        print("Adding burned stream features as background...")
        for feature_value, color in FEATURE_COLORS.items():
            feature_mask = burned_features['burned_features'] == feature_value
            if feature_mask.sum() > 0:
                feature_x = burned_features.loc[feature_mask, 'x_dts'].values
                min_x, max_x = feature_x.min(), feature_x.max()
                patch = ax.axvspan(min_x, max_x, color=color, alpha=0.12, zorder=0)
                
                # Store for legend
                feature_patches.append(patch)
                feature_labels_list.append(FEATURE_LABELS.get(feature_value, f"Feature {feature_value}"))
    
        
    except FileNotFoundError:
        print(f"Warning: Burned features file '{burned_features_file}' not found. Skipping feature backgrounds.")
    except Exception as e:
        print(f"Warning: Error loading burned features: {e}")
    
    # Generate colors for dates
    n_dates = len(dates)
    if n_dates > 0:
        colors = sns.color_palette("coolwarm_r", n_dates)
    
    # Plot temperature for each date
    print(f"\nPlotting temperatures for {n_dates} dates...")
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
            
            print(f"   {date}: {mean_temp.min().values:.2f}°C to {mean_temp.max().values:.2f}°C")
            
        except Exception as e:
            print(f"   Warning: Error plotting {date}: {e}")
    
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
        print(f"\nFigure saved: {save_fp}")
    
    plt.show()
    









    

