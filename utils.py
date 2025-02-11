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
        aspect='auto',  # Stretch to fit
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
        aspect='auto',  # Stretch to fit
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










    

