import numpy as np
import pandas as pd


class CollaborativeFiltering(object):

    def __init__(self, k):
        """Class constructor for KMeans
        Arguments:
            k {int} -- number of similar items to consider
        """
        self.k = k

    def get_row_mean(self, data):
        """Returns the mean of each row in the DataFrame or the mean of the
        Series. If the parameter data is a DataFrame, the function will
        return a Series containing the mean of each row in the DataFrame. If
        the parameter data is a Series, the function will return a np.float64
        which is the mean of the Series. This function should not consider
        blank ratings represented as NaN.

        Arguments:
            data {DataFrame or Series} -- dataset
        Returns:
            Series or np.float64 -- row mean
        """
        # TODO: Implement this function based on the documentation.

        # TODO: Check if the parameter data is a Series or a DataFrame

        # TODO: return the mean of each row if the parameter data is a
        # DataFrame. Return the mean of the Series if the parameter data is a
        # Series.
        # Hint: Use pandas.DataFrame.mean() or pandas.Series.mean() functions.
        if isinstance(data, pd.DataFrame):
            return data.mean(axis=1, skipna=True)  # Compute mean row-wise
        elif isinstance(data, pd.Series):
            return data.mean(skipna=True)  # Compute mean for the Series
        else:
            raise TypeError("Input data must be a pandas DataFrame or Series")
            

    def normalize_data(self, data, row_mean):
        """Returns the data normalized by subtracting the row mean.

        For the arguments point1 and point2, you can only pass these
        combinations of data types:
        - DataFrame and Series -- returns DataFrame
        - Series and np.float64 -- returns Series

        For a DataFrame and a Series, if the shape of the DataFrame is
        (3, 2), the shape of the Series should be (3,) to enable broadcasting.
        This operation will result to a DataFrame of shape (3, 2)

        Arguments:
            data {DataFrame or Series} -- dataset
            row_mean {Series or np.float64} -- mean of each row
        Returns:
            DataFrame or Series -- normalized data
        """

        # TODO: Implement this function based on the documentation.

        # TODO: Check if the combination of parameters is correct
        # Normalize the parameter data by parameter row_mean.
        # HINT: Use pandas.DataFrame.subtract() or pandas.Series.subtract()
        # functions.
        normalized_data = data.subtract(row_mean, axis=0)
        return normalized_data

    def get_cosine_similarity(self, vector1, vector2):
        """Returns the cosine similarity between two vectors. These vectors can
        be represented as 2 Series objects. This function can also compute the
        cosine similarity between a list of vectors (represented as a
        DataFrame) and a single vector (represented as a Series), using
        broadcasting.

        For the arguments vector1 and vector2, you can only pass these
        combinations of data types:
        - Series and Series -- returns np.float64
        - DataFrame and Series -- returns pd.Series

        For a DataFrame and a Series, if the shape of the DataFrame is
        (3, 2), the shape of the Series should be (2,) to enable broadcasting.
        This operation will result to a Series of shape (3,)

        Arguments:
            vector1 {Series or DataFrame} - vector
            vector2 {Series or DataFrame} - vector
        Returns:
            np.float64 or pd.Series -- contains the cosine similarity between
            two vectors
        """

        # TODO: Implement this function based on the documentation.

        # TODO: Check if the parameter data is a Series or a DataFrame

        # TODO: Compute the cosine similarity between the two parameters.
        # HINT: Use np.sqrt() and pandas.DataFrame.sum() and/or
        # pandas.Series.sum() functions.

        if not (isinstance(vector1, (pd.Series, pd.DataFrame)) and 
                    isinstance(vector2, (pd.Series, pd.DataFrame))):
                raise TypeError("Inputs must be pandas Series or DataFrame.")
    
        # Compute dot product
        if isinstance(vector1, pd.DataFrame):
            dot_product = (vector1 * vector2).sum(axis=1)  # DataFrame case
        else:
            dot_product = (vector1 * vector2).sum()        # Series case
    
        # Compute magnitudes
        magnitude1 = np.sqrt((vector1 ** 2).sum(axis=1 if isinstance(vector1, pd.DataFrame) else None))
        magnitude2 = np.sqrt((vector2 ** 2).sum())
    
        # Avoid division by zero
        magnitude_product = magnitude1 * magnitude2
        if isinstance(magnitude_product, (pd.Series, pd.DataFrame)):
            magnitude_product = magnitude_product.replace(0, 1e-10)  # Prevent NaN for zero vectors
        elif magnitude_product == 0:
            magnitude_product = 1e-10
    
        cosine_similarity = dot_product / magnitude_product
        return cosine_similarity

    def get_k_similar(self, data, vector):

        data_row_mean = data.mean(axis=1)  # DataFrame → Series of row means
        vector_row_mean = vector.mean()     # Series → scalar mean
        
        # Normalize data and vector
        normalized_data = self.normalize_data(data, data_row_mean)
        normalized_vector = self.normalize_data(vector, vector_row_mean)
        
        # Compute cosine similarities
        similarities = self.get_cosine_similarity(normalized_data, normalized_vector)
        
        # Extract top k indices and their similarity scores
        top_k_indices = similarities.nlargest(self.k).index
        top_k_similarities = similarities.loc[top_k_indices]
        
        return top_k_indices, top_k_similarities

    def get_rating(self, data, index, column):
        """Returns the extrapolated rating for a missing value."""
        try:
            # Step 1: Exclude the target row and get its vector
            other_data = data.drop(index)  # Remove the target movie row
            target_vector = data.loc[index]  # Get the target movie's ratings
    
            # Step 2: Find top-k similar movies
            similar_indices, similarities = self.get_k_similar(other_data, target_vector)
            
            # Step 3: Get the specified user's ratings for similar movies
            similar_ratings = other_data.loc[similar_indices, column]
    
            # Step 4: Filter out NaN ratings (if user didn't rate some similar movies)
            valid_mask = similar_ratings.notna()
            similar_ratings = similar_ratings[valid_mask]
            similarities = similarities[valid_mask]
    
            # Step 5: Compute weighted average (handle division by zero)
            if len(similarities) == 0:
                return np.nan  # No valid similar ratings
            
            weighted_sum = (similarities * similar_ratings).sum()
            sum_similarities = similarities.abs().sum()
    
            return weighted_sum / sum_similarities if sum_similarities != 0 else np.nan
    
        except Exception as e:
            print(f"Error predicting rating: {e}")
            return np.nan

