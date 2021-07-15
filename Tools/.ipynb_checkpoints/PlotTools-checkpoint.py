# Plotting Tools

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def prGreen(prt): print("\033[92m {}\033[00m" .format(prt))


def plot_mva(df, column, bins, logscale=False, ax=None, title=None, ls='dashed', alpha=0.5, sample='', cat="Matchlabel", Wt="Wt"):
    hcolor = ['#0f4c75', '#903749']
    histtype = "bar"
    if sample == 'test':
        histtype = "step"
    if ax is None:
        ax = plt.gca()
    for name, group in df.groupby(cat):
        if name == 0:
            label = "background"
        else:
            label = "signal"
        group[column].hist(bins=bins, histtype=histtype, alpha=alpha,
                           label=label+' '+sample, ax=ax, density=False, ls=ls, color=hcolor[name], weights=group[Wt]/group[Wt].sum(), linewidth=2)
    # ax.set_ylabel("density")
    ax.set_xlabel(column)
    ax.set_title(title)
    if logscale:
        ax.set_yscale("log", nonpositive='clip')
    ax.legend(loc='best')
    ax.tick_params(direction='in', which='both', bottom=True,
                   top=True, left=True, right=True)


def plot_roc_curve(df, score_column, tpr_threshold=0, ax=None, color=None, linestyle='-', label=None, cat="Matchlabel", Wt="Wt"):
    from sklearn import metrics
    if ax is None:
        ax = plt.gca()
    if label is None:
        label = score_column
    fpr, tpr, thresholds = metrics.roc_curve(
        df[cat], df[score_column], sample_weight=df[Wt])
    mask = tpr > tpr_threshold
    fpr, tpr = fpr[mask], tpr[mask]
    auc = metrics.auc(fpr, tpr)
    label = label+' auc='+str(round(auc*100, 1))+'%'
    ax.plot(tpr, fpr, label=label, color=color,
            linestyle=linestyle, linewidth=2, alpha=1.0)
    ax.set_yscale("log")
    ax.legend(loc='best')
    ax.tick_params(direction='in', which='both', bottom=True,
                   top=True, left=True, right=True)
    return auc


def plot_single_roc_point(df, var='Fall17isoV1wpLoose',
                          ax=None, marker='o',
                          markersize=6, color="red", label='', cat="Matchlabel", Wt="Wt"):
    backgroundpass = df.loc[(df[var] == 1) & (df[cat] == 0), Wt].sum()
    backgroundrej = df.loc[(df[var] == 0) & (df[cat] == 0), Wt].sum()
    signalpass = df.loc[(df[var] == 1) & (df[cat] == 1), Wt].sum()
    signalrej = df.loc[(df[var] == 0) & (df[cat] == 1), Wt].sum()
    backgroundrej = backgroundrej/(backgroundpass+backgroundrej)
    signaleff = signalpass/(signalpass+signalrej)
    ax.plot([signaleff], [1-backgroundrej], marker=marker,
            color=color, markersize=markersize, label=label)
    ax.set_yscale("log")
    ax.legend(loc='best')
    ax.tick_params(direction='in', which='both', bottom=True,
                   top=True, left=True, right=True)


def pngtopdf(ListPattern=[], Save="mydoc.pdf"):
    import glob
    import PIL.Image
    L = []
    for List in ListPattern:
        L += [PIL.Image.open(f) for f in glob.glob(List)]
    for i, Li in enumerate(L):
        rgb = PIL.Image.new('RGB', Li.size, (255, 255, 255))
        rgb.paste(Li, mask=Li.split()[3])
        L[i] = rgb
    L[0].save(Save, "PDF", resolution=100.0,
              save_all=True, append_images=L[1:])


def MakeFeaturePlots(df_final, features, feature_bins, Set="Train", MVA="XGB_1", OutputDirName='Output', cat="EleType", label=["Background", "Signal"], weight="NewWt"):
    fig, axes = plt.subplots(1, len(features), figsize=(len(features)*5, 5))
    prGreen("Making"+Set+" dataset feature plots")
    for m in range(len(features)):
        print('Feature: {}'.format(features[m]))
        for i, group_df in df_final[df_final['Dataset'] == Set].groupby(cat):
            group_df[features[m-1]].hist(histtype='step', bins=feature_bins[m-1], alpha=0.7, label=label[i],
                                         ax=axes[m-1], density=False, ls='-', weights=group_df[weight]/group_df[weight].sum(), linewidth=4)
            # df_new = pd.concat([group_df, df_new],ignore_index=True, sort=False)
        axes[m-1].legend(loc='upper right')
        axes[m-1].set_xlabel(features[m-1])
        axes[m-1].set_yscale("log")
        axes[m-1].set_title(features[m-1]+" ("+Set+" Dataset)")
    plt.savefig(OutputDirName+"/"+MVA+"/"+MVA+"_" +
                "featureplots_"+Set+".pdf", bbox_inches='tight')


def MakeFeaturePlots_sep(df_final, features, feature_bins, Set="Train", MVA="XGB_1", OutputDirName='Output', cat="EleType", label=["Background", "Signal"], weight="NewWt"):
    prGreen("Making"+Set+" dataset feature plots")
    for m in range(len(features)):
        print('Feature: {}'.format(features[m-1]))
        for i, group_df in df_final[df_final['Dataset'] == Set].groupby(cat):
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            group_df[features[m-1]].hist(histtype='step', bins=feature_bins[m-1], alpha=0.7, label=label[i],
                                         ax=ax, density=False, ls='-', weights=group_df[weight]/group_df[weight].sum(), linewidth=4)
            # df_new = pd.concat([group_df, df_new],ignore_index=True, sort=False)
        ax.legend(loc='best')
        ax.set_xlabel(features[m-1])
        ax.set_yscale("log")
        ax.set_title(features[m-1]+" ("+Set+" Dataset)")
        ax.tick_params(direction='in', which='both', bottom=True,
                       top=True, left=True, right=True)
        plt.savefig(OutputDirName+"/"+MVA+"/"+MVA+"_"+"featureplots_" +
                    features[m-1]+"_"+Set+".pdf", bbox_inches='tight')
        plt.close('all')


def MakeFeaturePlotsComb(df_final, features, feature_bins, MVA="XGB_1", OutputDirName='Output', cat="EleType", label=["Background", "Signal"], weight="NewWt"):
    fig, axes = plt.subplots(1, len(features), figsize=(len(features)*5, 5))
    prGreen("Making Combined"+" dataset feature plots")
    for m in range(len(features)):
        for i, group_df in df_final[df_final['Dataset'] == "Train"].groupby(cat):
            group_df[features[m-1]].hist(histtype='stepfilled', bins=feature_bins[m-1], alpha=0.5, label=label[i]+"_Train",
                                         ax=axes[m-1], density=False, ls='-', weights=group_df[weight]/group_df[weight].sum(), linewidth=4)
        for i, group_df in df_final[df_final['Dataset'] == "Test"].groupby(cat):
            group_df[features[m-1]].hist(histtype='step', bins=feature_bins[m-1], alpha=0.5, label=label[i]+"_Test",
                                         ax=axes[m-1], density=False, ls='--', weights=group_df[weight]/group_df[weight].sum(), linewidth=4)
            # df_new = pd.concat([group_df, df_new],ignore_index=True, sort=False)
        axes[m-1].legend(loc='upper right')
        axes[m-1].set_xlabel(features[m-1])
        axes[m-1].set_yscale("log")
        axes[m-1].set_title(features[m-1])
    plt.savefig(OutputDirName+"/"+MVA+"/"+MVA+"_" +
                "featureplots"+".pdf", bbox_inches='tight')


def MakeFeaturePlotsComb_sep(df_final, features, feature_bins, MVA="XGB_1", OutputDirName='Output', cat="EleType", label=["Background", "Signal"], weight="NewWt"):
    prGreen("Making Combined"+" dataset feature plots")
    hcolor = ['#0f4c75', '#903749']
    for m in range(len(features)):
        print('Feature: {}'.format(features[m-1]))
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        for i, group_df in df_final[df_final['Dataset'] == "Train"].groupby(cat):
            group_df[features[m-1]].hist(histtype='stepfilled', bins=feature_bins[m-1], alpha=0.4, label=label[i]+"_Train",
                                         ax=ax, density=False, ls='-', color=hcolor[i], weights=group_df[weight]/group_df[weight].sum(), linewidth=4)
        for i, group_df in df_final[df_final['Dataset'] == "Test"].groupby(cat):
            group_df[features[m-1]].hist(histtype='step', bins=feature_bins[m-1], alpha=0.9, label=label[i]+"_Test",
                                         ax=ax, density=False, color=hcolor[i], weights=group_df[weight]/group_df[weight].sum(), linewidth=1.5)
            # df_new = pd.concat([group_df, df_new],ignore_index=True, sort=False)
        ax.legend(loc='best')
        ax.set_xlabel(features[m-1])
        ax.set_yscale("log")
        ax.set_title(features[m-1])
        ax.tick_params(direction='in', which='both', bottom=True,
                       top=True, left=True, right=True)
        plt.savefig(OutputDirName+"/"+MVA+"/"+MVA+"_featureplots_comb_" +
                    features[m-1]+".pdf", bbox_inches='tight')
        plt.close('all')


def plot_featureimp(feature_importance, feature_names, MVA='XGB_1', OutputDirName='Output'):
    df_featimp = pd.DataFrame(
        feature_importance, index=feature_names, columns=['importance'])
    df_featimp.sort_values('importance', inplace=True, ascending=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    df_featimp['importance'].plot(kind='barh', ax=ax)
    ax.set_xlabel('Feature importance', fontsize=13)
    ax.set_ylabel('Feature', fontsize=13)
    ax.tick_params(direction='in', which='both', bottom=True,
                   top=False, left=True, right=True)
    ax.set_xlim(0, df_featimp['importance'].iloc[-1]*1.15)
    for i in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(i.get_width()+0.002, i.get_y(),
                '{:.3g}'.format(i.get_width()), fontsize=9)

    plt.savefig(OutputDirName+'/'+MVA+'/'+MVA +
                '_featureimportance.pdf', bbox_inches='tight')
    plt.close('all')


# Reference: 
# [1]: https://www.kaggle.com/drazen/heatmap-with-sized-markers
def heatmap(x, y, **kwargs):
    if 'color' in kwargs:
        color = kwargs['color']
    else:
        color = [1]*len(x)

    if 'palette' in kwargs:
        palette = kwargs['palette']
        n_colors = len(palette)
    else:
        n_colors = 256 # Use 256 colors for the diverging color palette
        palette = sns.color_palette("Blues", n_colors) 

    if 'color_range' in kwargs:
        color_min, color_max = kwargs['color_range']
    else:
        color_min, color_max = min(color), max(color) # Range of values that will be mapped to the palette, i.e. min and max possible correlation

    def value_to_color(val):
        if color_min == color_max:
            return palette[-1]
        else:
            val_position = float((val - color_min)) / (color_max - color_min) # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
            ind = int(val_position * (n_colors - 1)) # target index in the color palette
            return palette[ind]

    if 'size' in kwargs:
        size = kwargs['size']
    else:
        size = [1]*len(x)

    if 'size_range' in kwargs:
        size_min, size_max = kwargs['size_range'][0], kwargs['size_range'][1]
    else:
        size_min, size_max = min(size), max(size)

    size_scale = kwargs.get('size_scale', 250)

    def value_to_size(val):
        if size_min == size_max:
            return 1 * size_scale
        else:
            val_position = (val - size_min) * 0.99 / (size_max - size_min) + 0.01 # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
            return val_position * size_scale
    if 'x_order' in kwargs: 
        x_names = [t for t in kwargs['x_order']]
    else:
        x_names = [t for t in sorted(set([v for v in x]))]
    x_to_num = {p[1]:p[0] for p in enumerate(x_names)}

    if 'y_order' in kwargs: 
        y_names = [t for t in kwargs['y_order']]
    else:
        y_names = [t for t in sorted(set([v for v in y]))]
    y_to_num = {p[1]:p[0] for p in enumerate(y_names)}

    plot_grid = plt.GridSpec(1, 15, hspace=0.2, wspace=0.1) # Setup a 1x10 grid
    ax = plt.subplot(plot_grid[:,:-1]) # Use the left 14/15ths of the grid for the main plot

    marker = kwargs.get('marker', 's')

    kwargs_pass_on = {k:v for k,v in kwargs.items() if k not in [
         'color', 'palette', 'color_range', 'size', 'size_range', 'size_scale', 'marker', 'x_order', 'y_order'
    ]}

    ax.scatter(
        x=[x_to_num[v] for v in x],
        y=[y_to_num[v] for v in y],
        marker=marker,
        s=[value_to_size(v) for v in size], 
        c=[value_to_color(v) for v in color],
        **kwargs_pass_on
    )
    ax.set_xticks([v for k,v in x_to_num.items()])
    ax.set_xticklabels([k for k in x_to_num], rotation=90)
    ax.set_yticks([v for k,v in y_to_num.items()])
    ax.set_yticklabels([k for k in y_to_num])

    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)

    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
    ax.set_facecolor('#F1F1F1')
    
    # print ([x_to_num[v] for v in x])
    xpos = [x_to_num[v] for v in x]
    ypos = [y_to_num[v] for v in y]
    for i, cval in enumerate(color.tolist()):
        ax.text(xpos[i], ypos[i], '{0:.2f}'.format(cval), horizontalalignment='center', verticalalignment='center', size=4, color='black')

    # Add color legend on the right side of the plot
    if color_min < color_max:
        ax = plt.subplot(plot_grid[:,-1]) # Use the rightmost column of the plot

        col_x = [0]*len(palette) # Fixed x coordinate for the bars
        bar_y=np.linspace(color_min, color_max, n_colors) # y coordinates for each of the n_colors bars

        bar_height = bar_y[1] - bar_y[0]
        ax.barh(
            y=bar_y,
            width=[5]*len(palette), # Make bars 5 units wide
            left=col_x, # Make bars start at 0
            height=bar_height,
            color=palette,
            linewidth=0
        )
        ax.set_xlim(1, 2) # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
        ax.grid(False) # Hide grid
        ax.set_facecolor('white') # Make background white
        ax.set_xticks([]) # Remove horizontal ticks
        ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3)) # Show vertical ticks for min, middle and max
        ax.yaxis.tick_right() # Show vertical ticks on the right 


def corrplot(data, figsize=(8,8), size_scale=250, marker='s', plotname='plot'):
    import seaborn as sns
    sns.set() 

    corr = pd.melt(data.reset_index(), id_vars='index')
    corr.columns = ['x', 'y', 'value']
    plt.figure(figsize=figsize)
    heatmap(
        corr['x'], corr['y'],
        color=corr['value'], color_range=[-1, 1],
        palette=sns.diverging_palette(20, 220, n=2000),
        size=corr['value'].abs(), size_range=[0,1],
        marker=marker,
        x_order=data.columns,
        y_order=data.columns[::-1],
        size_scale=size_scale
    )
    plt.savefig('{}.png'.format(plotname), bbox_inches='tight')
    plt.savefig('{}.pdf'.format(plotname), bbox_inches='tight')
    plt.close('all')