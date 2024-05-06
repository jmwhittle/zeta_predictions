import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.graphics.gofplots import qqplot
from scipy.stats import norm



def dist_plots(df, cols_to_plot, nrows:int=10, ncols:int=3, figsize_x:int=10, figsize_y:int=25):
    fig,axes = plt.subplots(nrows = nrows, ncols = ncols, figsize = (figsize_x, figsize_y))
    axes = axes.flat
    naxes = nrows*ncols-1

    # enumerate interates through list and keeps count of iteration 
    # [(0, 'eat'), (1, 'sleep'), (2, 'repeat')] the first value returned is the count, the second is the list item
    for i, col in enumerate(cols_to_plot):
        sns.histplot(
            df, 
            x=col, 
            kde=True, # compute a kernel density estimate to smooth the distribution and show on the plot
            line_kws= {'linewidth':2.0}, # Parameters that control the KDE visualization
            stat = 'count', # show the number of observations in each bin
            color = (list(plt.rcParams['axes.prop_cycle'])*5)[i]["color"],
            ax = axes[i]
            )
            
        sns.rugplot(
            df, 
            x = col, 
            color = (list(plt.rcParams['axes.prop_cycle'])*5)[i]["color"],
            ax = axes[i]
            )
        
        axes[i].set_xlabel("")
        axes[i].set_title(f"{col}", fontsize = 8, fontweight = "bold", color = "darkblue")
        axes[i].tick_params(labelsize = 7)
            
            
    fig.delaxes(axes[naxes])
    fig.suptitle("Distribution of numerical variables", fontsize = 12, fontweight = "bold", color = "darkred")
    fig.tight_layout()
    fig.subplots_adjust(top = 0.9)

    return fig.show()


def cor_plot(df, cols_to_plot, method, figsize_x:int=25, figsize_y:int=15):
    sns.set_style("darkgrid")
    corr_matrix = df[cols_to_plot].corr(method =  method)
    mask = np.triu(np.ones_like(corr_matrix, dtype = bool))

    fig,ax = plt.subplots(figsize = (figsize_x, figsize_y))

    sns.heatmap(corr_matrix,
                cmap = "coolwarm",
                annot = True,
                annot_kws = {"fontsize":6, "fontweight":"bold"},
                square = True,
                mask = mask,
                linewidths = 1.0,
                linecolor = "white",
                ax = ax)

    ax.set_xticklabels(ax.get_xticklabels(), rotation = 0)
    ax.set_title('Correlation Matrix of numerical variables', fontsize = 10, fontweight = 'bold', color = 'darkblue')
    return fig.show()


def qq_plot(df, cols_to_plot, nrows:int=10, ncols:int=3, figsize=(9,25)):
    fig,axes = plt.subplots(nrows = nrows, ncols = ncols, figsize = figsize)
    axes = axes.flat

    for i,col in enumerate(cols_to_plot):
        qqplot(
            df[col],
            line = "s",
            ax = axes[i]
              )
        
        axes[i].set_title(f"{col}", 
                          fontsize = 8, 
                          fontweight = "bold", 
                          color = "darkblue")
        axes[i].tick_params(labelsize = 7)
        
        
    fig.delaxes(axes[29])
    fig.suptitle("QQ-plots", fontsize = 12, fontweight = "bold", color = "darkred")
    fig.tight_layout()

    return fig.show()


def normal_dist_plot(df, cols_to_plot, nrows:int=10, ncols:int=3, figsize=(9,20)):
    fig,axes = plt.subplots(nrows = nrows, ncols = ncols, figsize = figsize)
    axes = axes.flat

    for i,col in enumerate(cols_to_plot):
        sns.distplot(
        df[col],
        rug = True, 
        fit = norm, 
        color = (list(plt.rcParams['axes.prop_cycle'])*5)[i]["color"],
        ax = axes[i]
        )
        
        axes[i].set_xlabel("")
        axes[i].set_title(f"{col}", fontsize = 8, fontweight = "bold", color = "darkblue")
        axes[i].tick_params(labelsize = 7)
        
        
    fig.delaxes(axes[29])
    fig.suptitle("Distribution of numerical variables with respect to their normal distribution", fontsize = 12, fontweight = "bold", color = "darkred")
    fig.tight_layout()

    return fig.show()




# Confusion Matrix plot
def confusion_matrix_plot (cf_matrix_train,
                           cf_matrix_test,
                           classes:list)->None:

    """
    Function to plot the confusion matrices for the training and test set.

    Args:

      - cf_matrix_train(np.ndarray): confusion matrix of training set.
      - cf_matrix_test(np.ndarray): confusion matrix of testing set.
      - classes(list): list of containing the classes of the target variable.

    Return:
      - Confusion Matrix plots
    """

    # We calculate the confusion matrices for training and testing.
    confusion_train = cf_matrix_train
    confusion_test = cf_matrix_test

    # Calculate the percentages
    sumatoria_train = np.sum(confusion_train, axis = 1)
    porcentajes_train = confusion_train / sumatoria_train[:,np.newaxis]*100

    sumatoria_test = np.sum(confusion_test, axis = 1)
    porcentajes_test = confusion_test / sumatoria_test[:,np.newaxis]*100


    etiquetas_train = [['{} \n({:.1f}%)'.format(val, porc) for val,porc in zip(row,porc_row)] for row, porc_row in zip(confusion_train, porcentajes_train)]

    etiquetas_test = [['{} \n({:.1f}%)'.format(val, porc) for val,porc in zip(row,porc_row)] for row, porc_row in zip(confusion_test, porcentajes_test)]


    fig,axes = plt.subplots(1,2,figsize=(9,4))
    sns.heatmap(confusion_train,
                annot = np.array(etiquetas_train),
                fmt = '',
                cmap = 'Blues',
                cbar = False,
                square = True,
                linewidths = 0.7,
                linecolor = 'white',
                ax = axes[0])
    sns.heatmap(confusion_test,
                annot = np.array(etiquetas_test),
                fmt = '',
                cmap = 'Oranges',
                cbar = False,
                square = True,
                linewidths = 0.7,
                linecolor = 'white',
                ax = axes[1])
    # Add the texts TP, FN, FP, TN to the train matrix
    axes[0].text(0.5, 0.65, 'TN', ha='center', va='center', fontsize=9, fontweight='bold')
    axes[0].text(1.5, 0.65, 'FP', ha='center', va='center', fontsize=8, fontweight='bold')
    axes[0].text(0.5, 1.65, 'FN', ha='center', va='center', fontsize=8, fontweight='bold')
    axes[0].text(1.5, 1.65, 'TP', ha='center', va='center', fontsize=8, fontweight='bold')
    axes[0].set_title('Confusion Matrix Train',fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted', fontsize=10, fontweight='bold')
    axes[0].set_ylabel('Real', fontsize=10, fontweight='bold')
    axes[0].set_xticklabels(classes)
    axes[0].set_yticklabels(classes)
    axes[0].tick_params(rotation=0, size = 8)

    # Add the texts TP, FN, FP, TN to the test matrix
    axes[1].text(0.5, 0.65, 'TN', ha='center', va='center', fontsize=9, fontweight='bold')
    axes[1].text(1.5, 0.65, 'FP', ha='center', va='center', fontsize=8, fontweight='bold')
    axes[1].text(0.5, 1.65, 'FN', ha='center', va='center', fontsize=8, fontweight='bold')
    axes[1].text(1.5, 1.65, 'TP', ha='center', va='center', fontsize=8, fontweight='bold')
    axes[1].set_title('Confusion Matrix Test',fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicted', fontsize=10, fontweight='bold')
    axes[1].set_ylabel('Real', fontsize=10, fontweight='bold')
    axes[1].set_xticklabels(classes)
    axes[1].set_yticklabels(classes)
    axes[1].tick_params(rotation=0, size = 8)

    fig.subplots_adjust(top=0.9)
    fig.tight_layout()
    plt.show()