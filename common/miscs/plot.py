import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_model(model, X, y):
    import matplotlib.pyplot as plt
    y_pred = model.predict(X)

    # Plot the data and the model's prediction
    plt.scatter(X, y, color='blue', label='Data Points')  # Original data
    plt.plot(X, y_pred, color='red', label='Fitted Line')  # Fitted line
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression Fit')
    plt.legend()
    plt.show()


def plot_dataframe_with_x_mark(df, x_col, y_col, x_mark):
    plt.plot(df[x_col], df[y_col], label=y_col)
    plt.axvline(x=x_mark, color='r', linestyle='--', label=f'x = {x_mark}')
    closest_x_index = np.abs(df[x_col] - x_mark).idxmin()
    closest_x = df.loc[closest_x_index, x_col]
    y_value_at_closest_x = df.loc[closest_x_index, y_col]
    plt.plot(closest_x, y_value_at_closest_x, 'ro', label=f'Closest point ({closest_x}, {y_value_at_closest_x})')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(y_col + ' with Marked Value')
    plt.legend()
    plt.show()
    print(f"The y-value at the closest x ({closest_x}) to x_mark ({x_mark}) is: {y_value_at_closest_x}")
    return y_value_at_closest_x
    
def plot_trends(result, plot_columns=None, figsize=(10, 6)):
    if plot_columns is None:
        plot_columns = [col for col in result.columns]
    plt.figure(figsize=figsize)
    for column in plot_columns:
        plt.plot(result[column], label=column)
    plt.title('Trends Across Partitions')
    plt.xlabel('Partition Index')
    plt.ylabel('y')
    plt.legend()
    plt.show()

def plot_multiple_lines(df, y_axis, title=None, t1=None, t2=None):
    if len(y_axis) > 4:
        raise Exception(f'Max 4 y_axis supported, given {len(y_axis)}')

    y1_label, y2_label, y3_label, y4_label = '', '', '', ''
    y1_cols, y2_cols, y3_cols, y4_cols = [], [], [], []

    for i, e in enumerate(list(y_axis.items())):
        label, cols = e[0], e[1]
        if i == 0:
            y1_label = label
            y1_cols = cols.copy()
        elif i == 1:
            y2_label = label
            y2_cols = cols.copy()
        elif i == 2:
            y3_label = label
            y3_cols = cols.copy()
        elif i == 3:
            y4_label = label
            y4_cols = cols.copy()

    if t1 or t2:
        df = df.between_time(t1, t2).copy()

    transparent = 0.5
    cmap = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, ax1 = plt.subplots(figsize=(18, 9))
    fig.subplots_adjust(right=0.8)

    lines = []

    for i, col in enumerate(y1_cols):
        line_i1 = ax1.plot(df[col], c=cmap[i], label=col, alpha=transparent)
        lines += line_i1

    ax1.set_xlabel("Time")
    ax1.set_ylabel(y1_label, fontsize=20)
    ax1.yaxis.label.set_color(lines[-1].get_color())
    ax1.tick_params(axis="x")
    ax1.tick_params(axis="y", colors=lines[-1].get_color())
    ax1.grid(True, axis="both", which="both")

    if y2_cols:
        ax2 = ax1.twinx()
        for i2, col in enumerate(y2_cols, len(y1_cols)):
            line_i2 = ax2.plot(df[col], c=cmap[i2], label=col, alpha=transparent)
            lines += line_i2
        ax2.set_ylabel(y2_label)
        ax2.yaxis.label.set_color(lines[-1].get_color())
        ax2.tick_params(axis="both", colors=lines[-1].get_color())

    if y3_cols:
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        for i3, col in enumerate(y3_cols, len(y1_cols) + len(y2_cols)):
            line_i3 = ax3.plot(df[col], c=cmap[i3], label=col, alpha=transparent)
            lines += line_i3
        ax3.set_ylabel(y3_label)
        ax3.yaxis.label.set_color(lines[-1].get_color())
        ax3.tick_params(axis="both", colors=lines[-1].get_color())

    if y4_cols:
        ax4 = ax1.twinx()
        ax4.spines['right'].set_position(('outward', 120))
        for i4, col in enumerate(y4_cols, len(y1_cols) + len(y2_cols) + len(y3_cols)):
            line_i4 = ax4.plot(df[col], '-', c=cmap[i4], label=col)
            lines += line_i4
        ax4.set_ylabel(y4_label)
        ax4.yaxis.label.set_color(lines[-1].get_color())
        ax4.tick_params(axis="both", colors=lines[-1].get_color())

    labs = [l.get_label() for l in lines]
    plt.legend(lines, labs, loc="center left", bbox_to_anchor=(0.0, 0.0), borderaxespad=0.0)
    #plt.close(fig)
    #return fig

def plot_partition_label_means(df, block_column, partition_label_column, target_column, selected_labels=None):
    if block_column not in df.columns:
        df = df.reset_index()
    grouped = df.groupby([block_column, partition_label_column])[target_column].mean().reset_index()
    if selected_labels is not None:
        grouped = grouped[grouped[partition_label_column].isin(selected_labels)]
    plt.figure(figsize=(10, 6))
    for partition_label in grouped[partition_label_column].unique():
        partition_data = grouped[grouped[partition_label_column] == partition_label]
        plt.plot(partition_data[block_column], partition_data[target_column], label=partition_label)
    plt.xlabel(f'Block Column: {block_column}')
    plt.ylabel(f'Average of {target_column}')
    plt.title(f'Average of {target_column} by {partition_label_column} and {block_column}')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_partition_outlier_boxplot(df, partition_label_column, target_column, selected_labels=None,  outlier_ratio=0.001):
    if selected_labels is not None:
        df = df[df[partition_label_column].isin(selected_labels)]
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    sns.boxplot(x=partition_label_column, y=target_column, data=df,width=0.5, 
                palette="coolwarm",
                linewidth=1.5,
                saturation=0.8,
                fliersize=0)
    df[partition_label_column] = df[partition_label_column].astype(str)
    for label in df[partition_label_column].unique():
        subset = df[df[partition_label_column] == label]
        q1 = subset[target_column].quantile(0.25)
        q3 = subset[target_column].quantile(0.75)
        iqr = q3 - q1
        outliers = subset[(subset[target_column] < (q1 - 1.5 * iqr)) | (subset[target_column] > (q3 + 1.5 * iqr))]
        sampled_outliers = outliers.sample(frac=outlier_ratio)
        for x in sampled_outliers[target_column]:
            plt.scatter(label, x, color='red')
    plt.xticks(rotation=45)
    plt.title(f'Distribution of {target_column} across {partition_label_column}')
    plt.ylabel(f'Values of {target_column}')
    plt.xlabel(partition_label_column)
    plt.tight_layout()
    plt.show()
    
def plot_time_series(data, colname, sampling_rate=None, aggregation_period=None):
    if sampling_rate:
        if not (0 < sampling_rate <= 1):
            raise ValueError("sampling_rate must be between 0 and 1")
        
        sample_size = int(len(data) * sampling_rate)

        sampled_indices = np.random.choice(data.index, size=sample_size, replace=False)

        sampled_data = data.loc[sampled_indices, colname].sort_index()

        plt.figure(figsize=(10, 6))
        plt.plot(sampled_data.index, sampled_data.values, marker='', linestyle='-')
        plt.title(f"Sampled Time Series Plot of '{colname}'")
        plt.xlabel("Index")
        plt.ylabel(colname)
        plt.grid(True)
        plt.show()
    if aggregation_period:
        if colname not in data.columns:
            raise ValueError(f"Column {colname} not found in the DataFrame.")
        aggregated_data = data[colname].resample(aggregation_period).mean()
        plt.figure(figsize=(10, 6))
        aggregated_data.plot(marker='o', linestyle='-')
        plt.title(f"Time Series Plot of '{colname}' Aggregated by {aggregation_period}")
        plt.xlabel("Date")
        plt.ylabel(f"Mean of {colname}")
        plt.grid(True)
        plt.show()