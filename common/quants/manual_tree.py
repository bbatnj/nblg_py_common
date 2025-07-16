import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def run_LR_under_win_bar(data, win_threshold):
    def extract_values(label):
        first_bracket, second_bracket = label.split('_')
        x = float(first_bracket.split(',')[0].strip().lstrip('('))
        y = float(second_bracket.split(',')[0].strip().lstrip('('))
        return pd.Series([x, y])
    
    df = data.copy()
    df[['x', 'y']] = df['label'].apply(extract_values)
    df['y_hat_win'] = df['y_hat_win'].round(2)
    filtered_data = df.query('y_hat_win > @win_threshold').copy()
    min_idx = filtered_data.groupby('x')['y_hat_win'].idxmin()
    result = filtered_data.loc[min_idx]

    X = result[['x']]
    y = result['y']
    
    lr = LinearRegression()
    lr.fit(X, y)
    
    coef = lr.coef_[0]
    intercept = lr.intercept_
    
    model = {
        'coef': coef,
        'intercept': intercept,
        'data': result,
        'win_bar': win_threshold,
        'formula': f'y = {coef:.2f} * x + {intercept:.2f}'
    }
    
    return model

def plot_regression(model):
    result = model['data']
    X = result[['x']]
    y = result['y']
    
    plt.figure(figsize=(10, 6))
    plt.scatter(result['x'], result['y'], color='blue', label='Data Points')
    plt.plot(result['x'], model['coef'] * result['x'] + model['intercept'], color='red', linewidth=2, label=f'Regression Line: {model["formula"]}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Regression of Filtered Data')
    plt.legend()
    plt.grid(True)
    plt.show()
