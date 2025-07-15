import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

class ModelAnalyzer:
    def __init__(self, data, target_column, model_input, test_size=0.04, data_test=None, **kwargs):
        """
        Initialize the AnalyzeFrame object.

        Args:
            data (DataFrame): The input data for analysis.
            target_column (str): The name of the target column in the data.
            model_input: The model for analysis.
            test_size (float): The proportion of the data to be used for testing.
        """
        self.data = data.dropna()
        self.x_cols = [c for c in self.data if c != target_column]
        self.y_col = target_column
        self.df_x_train = None

        self.y_test = None
        self.train_res = None
        self.test_residuals = None
        self.train_predictions = None
        self.test_predictions = None
        self.target_column = target_column
        self.scaler = StandardScaler()
        self.model = model_input
        
        self.divide_base_col = kwargs.get('divide_base_col', False)
        if not data_test:
            self.test_size = test_size
            self.__split_and_scale_data__()
        else:
            self.data_test = data_test
            self.__train_test_builder__()
    
    def __train_test_builder__(self):
        X_train, y_train = self.data.drop(self.target_column, axis=1), self.data[self.target_column]
        X_test, y_test = self.data_test.drop(self.target_column, axis=1), self.data_test[self.target_column]
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)
        self.y_train, self.y_test = y_train, y_test

    def __split_and_scale_data__(self):
        """
        Split the data into training and testing sets and scale the features.
        """
        num_rows = len(self.data)
        if self.divide_base_col:
            self.data = self.data.sort_values(by=self.divide_base_col)
            unique_values = self.data[self.divide_base_col].unique()
            num_rows = self.data[self.divide_base_col].nunique()
            
        split_point = int((1 - self.test_size) * num_rows)
        if self.divide_base_col:
            split_point = self.data.index.get_loc(self.data[self.data[self.divide_base_col] == unique_values[split_point]].index[0])
            self.data = self.data.drop(columns=self.divide_base_col)
            print('Split Point:', split_point)
        
        
        train_df, test_df = self.data.iloc[:split_point], self.data.iloc[split_point:]
        X_train, y_train = train_df.drop(self.target_column, axis=1), train_df[self.target_column]
        X_test, y_test = test_df.drop(self.target_column, axis=1), test_df[self.target_column]
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.df_X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = self.scaler.transform(X_test)
        self.df_X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        self.y_train, self.y_test = y_train, y_test

    def train(self):
        """
        Train the model using the training data.
        """

        if self.df_X_train_scaled is None or self.y_train is None:
            raise ValueError("Data has not been prepared for training.")

        self.model.fit(self.df_X_train_scaled.values, self.y_train.values)

        self.train_predictions = self.model.predict(self.df_X_train_scaled.values)
        self.test_predictions = self.model.predict(self.df_X_test_scaled.values)

        self.__calculate_residuals__()

    def __calculate_residuals__(self):
        """
        Calculate the residuals of the model predictions.
        """

        if self.train_res is None and self.train_predictions is not None:
            self.train_res = self.y_train - self.train_predictions.reshape(self.y_train.shape)
            self.train_res.name = 'train_res'

        if self.test_residuals is None and self.test_predictions is not None:
            self.test_residuals = self.y_test - self.test_predictions.reshape(self.y_test.shape)
            self.test_residuals.name = 'test_residuals'
    
    def calc_residual_cdf(self, x_s= None, train_or_test='train'):
        if self.train_res is None:
            self.train()

        if x_s is None:
            mean = self.train_res.mean()
            std_range = self.train_res.std() * 6
            x_s=np.arange(start=mean - std_range, stop=mean + std_range, step = std_range / 100)

        if train_or_test == 'train':
            return calc_cdf(pd.DataFrame(self.train_res), 'train_res', x_s)
        elif train_or_test == 'test':
            return calc_cdf(pd.DataFrame(self.test_residuals), 'test_residuals', x_s)

    def calc_residual_with_predict(self, predict_value, precision, x_s = None, train_or_test='train'):
        if self.train_res is None:
            self.train()

        if x_s is None:
            mean = self.train_res.mean()
            std_range = self.train_res.std() * 6
            x_s=np.arange(start=mean - std_range, stop=mean + std_range, step = std_range / 100)

        if train_or_test == 'train':
            #RH : use df.query
            train_in_range = self.train_res[(self.train_predictions >= predict_value - abs(precision)) & (self.train_predictions <= predict_value + abs(precision))]
            return calc_cdf(pd.DataFrame(train_in_range), 'train_res', x_s)
        elif train_or_test == 'test':
            # RH : use df.query
            test_in_range = self.test_residuals[(self.test_predictions >= predict_value - abs(precision)) & (self.test_predictions <= predict_value + abs(precision))]
            return calc_cdf(pd.DataFrame(test_in_range), 'test_residuals', x_s)

    def evaluate_model(self, is_train, num_partitions = 5, show_result = True, plot_stratify = False):
        """
        Evaluate the performance of the model.

        Args:
            is_train (bool): Whether to evaluate the model on the training data.
            num_partitions (int): The number of partitions for stratification.

        Returns:
            dict: A dict containing the mean squared error, R² score, mean absolute percentage error, and result of stratification.
        """
        if is_train == False:
            y_true = self.y_test
            predictions = self.test_predictions
            data_lable = 'Out Sample'
        elif is_train == True:
            y_true = self.y_train
            predictions = self.train_predictions
            data_lable = 'In Sample'
        else:
            raise ValueError("Invalid is_train parameter. Use True or False instead.")
        mse = mean_squared_error(y_true, predictions)
        r2 = r2_score(y_true, predictions)
        mae = abs(y_true - np.mean(y_true)).mean()
        mape = 1 - abs(y_true - predictions).mean() / mae

        df_results = pd.DataFrame({'Actual': y_true, 'Predicted': predictions})
        pearson_corr = df_results.corr().iloc[0, 1]
        spearman_corr = df_results.corr(method='spearman').iloc[0, 1]
        if show_result:
            print('==========================================================================')
            print(data_lable, "Analysis Result:")
            print("Mean Squared Error:", mse)
            print("R^2 Score:", r2)
            print('Mean Absolute Error:', mae)
            print("Mean Absolute Percentage Error:", mape)
            print("Results Correlation:\n", pearson_corr)
            print("Results Rank correlation:\n",spearman_corr)
            result = stratify_and_analyze(df_results, target='Actual', num_partitions=num_partitions)
        else:
            result = stratify_and_analyze(df_results, target='Actual', num_partitions=num_partitions,print_t_measure='No')
        if plot_stratify:
            plot_trends(result, plot_columns=['Predicted'], figsize=(10, 6))
        return {'mse': mse, 'r2': r2, 'mape': mape, 'stratify': result, 'pearson_corr': pearson_corr, 'spearman_corr': spearman_corr}

    def get_non_zero_xcols(self, threshold=1e-4):
        x_cols = self.x_cols
        coeffs = zip(x_cols, self.model.coef_)
        return [c[0] for c in coeffs if abs(c[1]) > threshold]

    @property
    def real_coef(self):
        features_mean = self.scaler.mean_
        features_std = self.scaler.scale_
        real_coef = self.model.coef_ / features_std
        return real_coef
    
    @property
    def features_mean(self):
        features_mean = self.scaler.mean_
        return features_mean

    def verify_params(self):
        #RH: let simplify var names
        manual_pred = (np.dot((self.df_x_test - self.features_mean), self.real_coef)
                              + self.model.intercept_)

        model_pred = self.test_predictions

        diff = np.abs(manual_pred - model_pred)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        print("Max diff:", max_diff)
        print("Mean diff:", mean_diff)
    
def stratify_and_analyze(df, target, num_partitions, print_t_measure='No'):
    df_splits = np.array_split(df, num_partitions)
    result = pd.DataFrame(columns=df.columns)
    for i, sub_df in enumerate(df_splits):
        corr = sub_df.corr().loc[target]
        result = pd.concat([result, pd.DataFrame(corr).T], ignore_index=True)
    for column in result.columns:
        if result[column].mean() < 0:
            result[column] *= -1
    result = result.drop(columns=[target])
    if print_t_measure == 'Yes':
       print(t_test_measure(result))  # Assuming t_test_measure is defined elsewhere
    return result
