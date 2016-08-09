
## GRADIENT DESCENT VIZ - GUI dashboard
#    Last update: August 2016


## INDEX
#    + Auxiliary geometry functions
#    + Plotting functions
#    + Loss calculation functions
#    + Data + Solution Area
##


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm


# Setting standard style
plt.style.use('bmh')


## AUXILIARY GEOMETRY FUNCTIONS -----------------------------------------------

def intersection_line_point(slope, offset, x_, y_):
    # intersection point
    int_x = (y_ + x_/slope - offset) / (slope + 1./slope)
    int_y = slope * int_x + offset
    
    return (int_x, int_y)


def dist_line2point(slope, offset, x_, y_):
    # obtain intersection point
    int_x, int_y = intersection_line_point(slope, offset, x_, y_)
    
    return( np.hypot(abs(int_x - x_), abs(int_y - y_)) )


def get_margin(slope, offset):
    # get distance from all points to line
    margins = []
    for i, x in df.iterrows():
        margins.append( dist_line2point(slope, offset, x.X1, x.X2) )
    
    # find point with smallest distance to line
    p_margin = np.argmin(margins)
    
    # output distance and point coordinates
    return (margins[p_margin], (df.loc[p_margin].X1, df.loc[p_margin].X2) )
    


## PLOTTING FUNCTIONS ---------------------------------------------------------

# Line factory
def line_factory(slope, offset):
    return np.vectorize(lambda x: slope * x + offset )



def plot_sep_line(slope, offset, loss_label, label='', past=None):
    
    # separation lines
    line = line_factory(slope, offset)
    line_y = line(X)
    
    # get margin
    margin, marg_p = get_margin(slope, offset)
    int_p = intersection_line_point(slope, offset, *marg_p)

    # select Loss values (Z_axis)
    if loss_label == 'Zero-Plus Loss':
        Z = z_zero
        actual_p = zero_plus_loss(slope, offset)
    elif loss_label == 'Hinge Loss':
        Z = z_hinge
        actual_p = hinge_loss(slope, offset)
    else:
        Z= z_log
        actual_p = log_loss(slope, offset)

    
    ## FIGURE
    fig = plt.figure(figsize=(17,10))

    ## 2D Plot
    # Plot data points
    ax_0 = fig.add_subplot(2, 2, 1)

    ax_0.scatter(df.X1.loc[df.Target > 0], df.X2.loc[df.Target > 0], marker='+', lw=3, s=300, color='green')  # plot +1's
    ax_0.scatter(df.X1.loc[df.Target < 0], df.X2.loc[df.Target < 0], marker='x', lw=3, s=300, color='red')  # plot -1's

    # Plot solution space
    ax_0.fill_between(X, linePP_y, linePPp_y, where=linePP_y >= linePPp_y, facecolor='b', alpha=.1, interpolate=True)
    ax_0.fill_between(X, linePPp_y, lineNP_y, where=linePPp_y >= lineNP_y, facecolor='b', alpha=.1, interpolate=True)
    ax_0.fill_between(X, linePPp_y, linePN_y, where=linePPp_y >= linePN_y, facecolor='b', alpha=.1, interpolate=True)

    # Plot separation line
    ax_0.plot(X, line_y, color='darkblue', ls='--', alpha=.7)
    
    # Plot margin
    ax_0.plot([int_p[0], marg_p[0]], [int_p[1], marg_p[1]], color='cyan', lw=4)

    ax_0.set_xlim(-4, 4)
    ax_0.set_ylim(-4, 4)
    ax_0.set_title('Solution Space - Margin: {:.2f}'.format(margin), size=18)


    ## 3D Plot
    ax_1 = fig.add_subplot(2, 2, 2, projection='3d')
    ax_1 = fig.gca(projection='3d')

    cset = ax_1.contourf(x_3d, y_3d, Z, zdir='z', offset=0, cmap=cm.coolwarm, alpha=.4)
    cset = ax_1.contourf(x_3d, y_3d, Z, zdir='x', offset=-4, cmap=cm.coolwarm, alpha=.4)
    cset = ax_1.contourf(x_3d, y_3d, Z, zdir='y', offset=4, cmap=cm.coolwarm, alpha=.4)
    ax_1.plot_surface(x_3d, y_3d, Z, rstride=8, cstride=8, alpha=0.2)

    # add actual line to 3D plot
    ax_1.scatter([slope], [offset], [actual_p], marker='o', s=300, c='cyan')
    ax_1.scatter([slope], [offset], [0], marker='+', s=200, lw=3, c='blue')
 
    ax_1.set_xlabel('slope')
    ax_1.set_xlim(-4, 4)
    ax_1.set_ylabel('offset')
    ax_1.set_ylim(-4, 4)
    ax_1.set_zlabel('Loss')
    # ax_1.set_zlim(-10, 10)

    if past is not None:
        # plot past data points
        for i, (_offset, _slope) in enumerate(past):
            ax_1.scatter([_slope], [_offset], [0], marker='+', s=150, lw=3, c='black')

    ax_1.set_title('{lbl} Loss Surface: {fun}'.format(lbl=label, fun=loss_label), size=18)
    
    plt.tight_layout()
    plt.show()


def grad_descent_plot(slope, offset, loss_label, LEARN_RATE=.5):
    print('INITIAL WEIGHTS - slope: {:.2f}   offset: {:.2f}'.format(slope, offset))
    plot_sep_line(slope, offset, loss_label, label='INITIAL CONFIGURATION')

    past = [ (offset, slope) ]

    for i,x in df.iterrows():
        print( '\n -> Using datapoint ({},{}) :'.format(x.X1, x.X2) )
        slope -= LEARN_RATE * (-1 * x.X2 * x.X1)
        offset -= LEARN_RATE * (-1 * x.X2)
        
        print('STEP #{} WEIGHTS - slope: {:.2f}   offset: {:.2f}'.format(i+1, slope, offset))
        plot_sep_line(slope, offset, loss_label, label='STEP #{}'.format(i+1), past=past)

        # Append to past
        past.append( (offset, slope) )




## LOSS FUNCTIONS -------------------------------------------------------------

def zero_plus_loss(slope, offset, regularization=None):
    prediction = np.sign( df.X2 - (slope * df.X1 + offset) )
    return (prediction != df.Target).sum()


def hinge_loss(slope, offset, regularization=.2):
    error = np.maximum(0, 1. - df.X2 * (slope * df.X1 + offset) )
    reg_cost = np.sqrt( slope**2 + offset**2)
    return error.sum() + regularization * reg_cost


def log_loss(slope, offset, regularization=.2):
    _p = 1 / (1 + np.exp(-1 * slope * df.X1 + offset))
    error = df.X2 * np.log(_p) + (1 - df.X2) * np.log(1 - _p)
    reg_cost = np.sqrt( slope**2 + offset**2)
    return error.sum() * -1 / error.count() + regularization * reg_cost



## DATA + SOLUTION AREA -------------------------------------------------------

# Get data
df = pd.read_csv('data/data.csv')

# Generate solution area
X = np.arange(-4,4, 1)

linePP = line_factory(-3./2, 0)
linePP_y = linePP(X)

linePPp = line_factory(-3./2, -5./2)
linePPp_y = linePPp(X)

lineNP = line_factory(-2./3, -5./3)
lineNP_y = lineNP(X)

linePN = line_factory(-4., -5.)
linePN_y = linePN(X)


x_3d = axes3d.get_test_data(0.05)[0]/10
y_3d = axes3d.get_test_data(0.05)[1]/10
# x_3d = np.tile(np.arange(-4,4.1,.1), (80,1))
# y_3d = np.tile(np.arange(-4,4.1,.1), (80,1))

z_zero = np.vectorize(zero_plus_loss)(x_3d, y_3d)
z_hinge = np.vectorize(hinge_loss)(x_3d, y_3d)
z_log = np.vectorize(log_loss)(x_3d, y_3d)

