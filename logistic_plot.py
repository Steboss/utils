import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from seaborn import lineplot
import seaborn as sns
from matplotlib import pyplot
from ..exceptions import TooManyLevelsError
sns.set()
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 3})
import time

class Emblem():
    """
        Creates an object for plotting useful diagnostic graphs for models and model data.

        Attributes:
            max_levels (int): maximum number of levels allowable for the grouping factor in an interaction plot (default 4).
            X_train (dataframe): model training data.
            y_train (series): training target values.
            X_val (dataframe): validation data for the model.
            y_val (series): validation target values.
            train (dataframe): training data with target values for plotting
            val (dataframe): validation data with target values for plotting
            num_splits (int): number of random splits to use for random split plot
            model (Model): the model to use in plots. If model is not provided, mean will be used
            preds_train (series): model predictions on the training data
            preds_val (series): model predictions on the validation data
            bands (int): the number of bands to use for aggregating numeric data for plotting

    """
    def __init__(self,X_train,y_train, X_val=None,y_val=None,model=None,num_splits=5,max_levels=4,bands=20):
        # initialise attributes
        self.bands = bands
        self.max_levels = max_levels
        self.X_train = X_train
        self.y_train = y_train
        self.train = self.X_train.copy()
        self.train['train_y'] = self.y_train
        self.X_val = X_val
        self.y_val = y_val
        self.num_splits = num_splits
        if self.X_val is not None and self.y_val is not None:
            self.val = self.X_val.copy()
            self.val['val_y'] = self.y_val
        else:
            self.val = None
        # Band all numeric factors with greater than bands to have bands unique values
        self.process_numeric_factors()
        self.model = model
        # Take the probability y=1 from the predict_proba array
        # This will fail if there are more columns in the data than in the model
        if self.model is not None:
            self.preds_train = self.model.predict_proba(self.X_train)[:,1]
            if self.X_val is not None:
                self.preds_val = self.model.predict_proba(self.X_val)[:,1]
            else:
                self.preds_val = None
        else:
            # Set as mean if no model provided
            self.preds_train = pd.Series([y_train.mean()]*len(y_train))
        self.train['preds_train'] = self.preds_train
        #  Curently do nothing with the validation predictions
        if self.preds_val is not None:
            self.val['preds_val'] = self.preds_val
        # Create the random_split factor with num_splits levels
        self.train['random_split'] = self.split()


    def process_numeric_factors(self):
        """
            Band all numeric factors with >bands levels to have bands levels.
        """
        a = pd.DataFrame(self.train.dtypes,columns=['dtype'])
        a['unique_vals'] = [len(self.train[e].unique()) for e in a.index]
        cols_to_band = list(a[(a['unique_vals'] > self.bands) & ((a['dtype'] == 'float64') | (a['dtype'] == 'int64'))].index)
        for fac in cols_to_band:
            banded,t_bins = self.band(self.train[fac],self.bands)
            self.train[fac] = banded
            if self.val is not None:
                self.val[fac]= pd.cut(self.val[fac],t_bins)

    def plot(self,fac,error_bars=True):
        """
            Create an Emblem-style plot.

            The plot shows the factor levels on the x-axis and proportion where target=1 on the first y-axis,
            proportion of volume on the second y-axis. The lines show the actuals in the training and validation
            data, error bars around the training actuals, and the model predictions. The bars at the bottom
            refer to the secondary axis and show volume by factor level. Note that the volumes will be
            approximately flat for banded factors, as this is a function of the banding.

            Parameters:
                fac (string): the factor to look at
                error_bars (boolean): whether to use bootstrap error bars in the plot around the training
                data y values
        """
        # Create the dataframe that is the basis of the plot and assign the different values
        emb = pd.DataFrame(self.train.groupby(fac)['train_y'].mean())
        emb['train_volume'] = (self.train.groupby(fac)['train_y'].count()/len(self.train))
        if self.preds_train is not None:
            emb['model_prediction'] = (self.train.groupby(fac)['preds_train'].mean())
        if self.X_val is not None and self.y_val is not None:
            emb['validation_y'] = (self.val.groupby(fac)['val_y'].mean())
        if error_bars:
            errors = self.bootstrap_error(fac)
        # Set the index to string to match the error bar index
        emb.rename(str, inplace=True)
        if error_bars:
            emb['lower_bound'] = errors[0].rename(str)
            emb['upper_bound'] = errors[1].rename(str)

        # Make the index a column for the melt
        emb.reset_index(inplace=True)
        # Melt into long format for seaborn
        df=pd.melt(emb,[fac])
        # Set figure size and various axis parameters
        fig, ax = pyplot.subplots(nrows=1,ncols=1,figsize=(20,10))
        ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
        ax.grid(False)
        # Plot the lines apart from the error bars
        sns.lineplot(ax=ax,x=fac, y='value', hue='variable', palette='deep',
             data=df[(df.variable != 'train_volume') & (df.variable != 'lower_bound') & (df.variable != 'upper_bound')],sort=False).set(ylabel='Proportion positive')
        # Plot the error bars to get the same hue and style (dashed)
        sns.lineplot(ax=ax,x=fac, y='value', hue='variable',style='variable', palette='deep',
             data=df[df.variable=='upper_bound'],
             sort=False,dashes=[(2,1)]).set(ylabel='Proportion positive')

        sns.lineplot(ax=ax,x=fac, y='value', hue='variable', style='variable', palette='deep',
             data=df[df.variable=='lower_bound'],
             sort=False,dashes=[(2,1)]).set(ylabel='Proportion positive')

        # Create the secondary axis
        ax2 = ax.twinx()
        ax2.set(ylim=(0,1))
        # Plot volume bars
        sns.barplot(ax=ax2,x = fac, y='value',data=df[df.variable =='train_volume'],color='y',alpha=0.5).set(ylabel='Volume')
        ax2.grid(False)
        ax2.set_title(fac)

    # deprecated
    def _plot(self,fac,use_model=True,show_validation=True,error_bars=True):
        """
            Deprecated plotting function, retained for continuity--Use plot().
        """
        emb = pd.DataFrame(self.train.groupby(fac)['train_y'].mean())#.rename(str)
        emb['train_volume'] = (self.train.groupby(fac)['train_y'].count()/len(self.train))#.rename(str)
        if self.preds_train is not None:
            emb['model_prediction'] = (self.train.groupby(fac)['preds_train'].mean())#.rename(str)
        if self.X_val is not None and self.y_val is not None:
            emb['validation_y'] = (self.val.groupby(fac)['val_y'].mean())#.rename(str)
        if error_bars:
            # Build in the params
            errors = self.bootstrap_error(self.train,fac)

        emb.rename(str, inplace=True)
        emb['train_volume'].plot(kind="bar", color='grey',legend=True,ylim=(0,1))
        emb['train_y'].plot(color="red", figsize=(20,10),legend=True,secondary_y=True)
        if self.preds_train is not None:
            emb['model_prediction'].plot(color="green",legend=True,secondary_y=True)
        if self.X_val is not None and self.y_val is not None:
            emb['validation_y'].plot(color="blue",legend=True,secondary_y=True)
        if error_bars:
            emb['lower_bound'] = errors[0].rename(str)
            emb['upper_bound'] = errors[1].rename(str)

            emb['lower_bound'].plot(ls="dashed",color="black",legend=True,secondary_y=True)
            emb['upper_bound'].plot(ls="dashed",color="black",legend=True,secondary_y=True)

    def random_split_plot(self,fac):
        """
            Create a random split plot with a given factor.

            The random split plot is similar to the Emblem plot created by plot(),
            but does not include validation data, just a mean line for the factor
            levels, a model line, and five random split lines to identify where factors are
            not consistent over random due to weak signal/high noise.

            Parameters:
                fac (string): the factor to examine
        """
        # Create the underlying data for the plot
        emb = pd.DataFrame(self.train.groupby(fac)['train_y'].mean())
        # Initialise labels for each split and add the split to the data cube
        labels = []
        for i in range(self.num_splits):
            label = 'train_y_{0}'.format(i)
            labels.append(label)
            emb[label] = self.train[self.train['random_split']==i].groupby(fac)['train_y'].mean()
        # Add train volume (overall)
        emb['train_volume'] = self.train.groupby(fac)['train_y'].count()/len(self.train)
        # Add model line if available
        if self.preds_train is not None:
            emb['model_prediction'] = (self.train.groupby(fac)['preds_train'].mean())
        # rename the index to str
        emb.rename(str, inplace=True)
        emb.reset_index(inplace=True)
        df=pd.melt(emb,[fac])
        fig, ax = pyplot.subplots(nrows=1,ncols=1,figsize=(20,10))
        ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
        sns.lineplot(ax=ax,x=fac, y='value', hue='variable',
             data=df[df.variable != 'train_volume'],sort=False).set(ylabel='Proportion positive')
        ax2 = ax.twinx()
        ax2.set(ylim=(0,1))
        sns.barplot(ax=ax2,x = fac, y='value',data=df[df.variable =='train_volume'],color='y',alpha=0.5).set(ylabel='Volume')
        ax2.grid(False)

    def interaction_grapher(self,base_factor,grouping_factor):
        """
            Plot showing potential interactions between two factors.

            This plot is similar to plot() output where fac = base_factor, but volume,
            training y values, and model predictions are all split by the grouping factor.
            An error will be raised if the grouping factor has too many levels (default
            4). You can change this, but it is hard to interpret the plot where there
            are too many groups.

            Parameters:
                base_factor (string): the factor that determines the values on the x-axis
                grouping_factor (string): determines the number of different lines and bars
        """
        if len(self.train[grouping_factor].unique()) > self.max_levels:
            raise TooManyLevelsError(
                "Grouping factor has too many levels."
                " Band factor into {} or fewer levels".format(self.max_levels)
            )
        # Generate the underlying data for the plot dynamically
        emb = None
        volume_cols = []
        for level in self.train[grouping_factor].unique():
            label_suffix = '_where_{}'.format(str(level))
            # For y values, initiate new emb if not already initiated
            if emb is None:
                emb = pd.DataFrame(self.train[self.train[grouping_factor]==level].groupby(base_factor)['train_y'].mean())
                emb.rename(columns={'train_y':'y'+label_suffix},inplace=True)
            else:
                emb['y'+label_suffix] = self.train[self.train[grouping_factor]==level].groupby(base_factor)['train_y'].mean()

            # For model scores
            emb['model'+label_suffix] = (self.train[self.train[grouping_factor]==level].groupby(base_factor)['preds_train'].mean())
            # For volume
            emb['volume'+label_suffix] = (self.train[self.train[grouping_factor]==level].groupby(base_factor)['train_y'].count()/len(self.train))
            volume_cols.append('volume'+label_suffix)

        # Rename index to string, reset index, and melt into long format for seaborn.
        emb.rename(str, inplace=True)
        emb.reset_index(inplace=True)

        df=pd.melt(emb,[base_factor])
        # Set up figure and axis
        fig, ax = pyplot.subplots(nrows=1,ncols=1,figsize=(20,10))
        ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
        ax.grid(False)
        # Plot lines, set up secondary axis, and plot bars
        sns.lineplot(ax=ax,x=base_factor, y='value', hue='variable', palette='deep',
             data=df.loc[~df['variable'].isin(volume_cols)],
                     sort=False).set(ylabel='Proportion positive')

        ax2 = ax.twinx()
        ax2.set(ylim=(0,1))
        sns.barplot(ax=ax2,x = base_factor, y='value', hue = 'variable',
                    data=df.loc[df['variable'].isin(volume_cols)],
                    alpha=0.5).set(ylabel='Volume')

        ax2.grid(False)
        ax2.set_title(base_factor+' split by '+grouping_factor)
        pyplot.legend(loc='upper center')

    def bootstrap_error(self,fac,target='train_y',width=0.95,iterations=100):
        """
            Bootstrap error bars for the training data y values.

            Creates iterations random samples at each level of the factor and calculates the proportion
            y=1. The values are ordered to obtain a parameterless confidence interval.

            Parameters:
                fac (string): the factor used in the plot
                width (float): the width of the interval. For 100 iterations, 0.95 gives you a 90%
                    confidence interval. A higher value will give a wider interval.
                iterations (int): The number of random bootstrap samples to draw. A higher value
                    will give a less variable estimate and allow more precision with the width.

        """
        results = {u:[] for u in self.train[fac].unique()}
        # For each iteration
        for i in range(iterations):
            # For each factor level
            for k in results.keys():
                # Create subset of rows which have the given factor level
                sub = self.train.loc[self.train[fac]==k]
                # Create a bootstrap sample of the same size
                samp = sub.sample(n=len(sub),replace=True)
                # Calculate mean proportion y in the bootstrap
                temp = samp[target].mean()
                # Store result
                results[k].append(temp)
        # Sort to get the upper and lower bounds for each level and convert to series
        lower = {k:sorted(results[k])[int(iterations*(1-width))] for k in results.keys()}
        upper = {k:sorted(results[k])[int(iterations*width)] for k in results.keys()}
        lower = pd.Series(lower)
        upper = pd.Series(upper)
        return (lower,upper)

    def band(self,factor_series,bands):
        """
            Band a numeric series to aggregate to fewer levels

            Parameters:
                factor_series (series): the factor to be banded
                bands (int): the desired number of bands

            Returns:
                banded_series (series): a new series banded to the desired number of bands
                t_bins (object): object to allow banding of validation data into the same buckets
                    as the training data.


        """
        banded_series,t_bins = pd.qcut(factor_series,bands,retbins=True,duplicates='drop')
        return banded_series,t_bins


    def split(self):
        """
            Create a random series of the same length as your data

            Returns:
                df (series): a random series with self.num_splits values of length
                    equal to self.train
        """
        df = pd.Series(range(self.num_splits)).sample(n=len(self.train),replace=True)
        df.index = range(len(df))
        return df

"""
This should probably be an object that stores the data and has a method to make the curve, rather than just a function returning a df
"""

def learning_curve(model,
                   features_train,
                   target_train,
                   features_test,
                   target_test,
                   target_col=None,
                   train_set_sizes=None,#[100,250,500,750,1000,2000,3000,4000,5000,6000],
                   iterations=None,#     [10, 10, 10, 10, 5,  5,  5,   5,   5,   5],
                  ):
    """
    Plot a learning curve for a given model. Use this for diagnostics on selected models. To speed up, reduce iterations
    or train_set_sizes. The test set should be the fixed dev set, rather than the actual one-use only test set.

    Each subset of the training data is selected randomly using stratified sampling--multiple iterations can be used at smaller sizes
    to iron out randomness.

    As training set size increases, training set performance will decrease and dev performance will increase.

    A large gap between train performance and dev performance indicates overfitting--remove noise or use a less flexible model
    Poor train performance indicates underfitting--add features and consider a more flexible model type
    The trajectory of the train performance gives you an upper-bound on the dev performance--if too low, drag the train up
    If the dev trajectory is steep, try to get more data


    Parameters:

        model: any model with standard sklearn-style fit() and predict_proba() methods
        features_train: features suitable for training the model
        target_train: suitable target values for the model training
        features_test: features suitable for scoring with the model
        target_test: suitable target values for the roc_auc_score
        train_set_sizes: list of size incremets to test
        iterations: list of iterations to perform at each size

    returns: dataframe of train and dev roc_auc_scores by training set size


    """
    n = features_train.shape[0]
    if train_set_sizes is None:
        train_set_proportions = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9])
        train_set_sizes = np.ceil(train_set_proportions*n).astype(int)
    if iterations is None:
        iterations = 5*(train_set_sizes < 1000) + 5

    if target_col is not None:
        target_train = target_train[target_col]
        target_test = target_test[target_col]

    n_tests = len(iterations)
    rows = []
    tocs = []
    for ti, (train_set_size, iteration) in enumerate(zip(train_set_sizes, iterations), start=1):
        print('Starting evaluation {} of {} with training set of size {}.'.format(ti, n_tests, train_set_size))
        toc_total = 0
        tocs = []
        train_aucs = []
        dev_aucs = []
        for i in range(iteration):

            X_train,_,y_train,_ = train_test_split(features_train, target_train, train_size=train_set_size,
                                                   test_size=n-train_set_size, stratify=target_train)
            tic = time.time()
            model.fit(X_train,y_train)
            toc = time.time() - tic
            probs_train = model.predict_proba(X_train)[:,1]
            probs_test = model.predict_proba(features_test)[:,1]
            roc_auc_train = roc_auc_score(y_train, probs_train)
            roc_auc_dev = roc_auc_score(target_test, probs_test)
            train_aucs.append(roc_auc_train)
            dev_aucs.append(roc_auc_dev)
            tocs.append(toc)
            toc_total += toc
        train_d = [{'VAL':a,'TYPE':'train','SIZE':train_set_size, 'FIT_TIME': t} for a, t in zip(train_aucs, tocs)]
        dev_d = [{'VAL':a,'TYPE':'dev','SIZE':train_set_size, 'FIT_TIME': t} for a, t in zip(dev_aucs, tocs)]
        rows += train_d
        rows += dev_d
        print('Complete in {:.0f} seconds.'.format(toc_total))
    df = pd.DataFrame(rows)
    lineplot(x='SIZE',y='VAL',hue='TYPE',style='TYPE', data = df)
    #lineplot(x='SIZE',y='VAL',hue='TYPE',style='TYPE', data = df)
    return plt.gca(), df
