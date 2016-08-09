
## GRADIENT DESCENT VIZ - GUI dashboard
#    Last update: August 2016


## INDEX
#    + Dashboard design
#    + Event handlers
##


from ipywidgets import widgets
from IPython.display import display, clear_output

from grad_desc_functionality import *


## DASHBOARD DESIGN -----------------------------------------------------------

dashboard = widgets.VBox()

# Title and spacers
header = widgets.HTML("<h3 style='color: darkblue; width: 900px; text-align: center;'>Separation Hyperplane</h3>")
vspace = widgets.HTML("<br>")
hspace = widgets.HTML("<h2 style='color: white;'>--</h2>")

# Style selection
slope_slider = widgets.FloatSlider(
    value=2,
    min=-3,
    max=3,
    step=0.1,
    description='Initial Slope:',
    slider_color='blue'
)

offset_slider = widgets.FloatSlider(
    value=0.5,
    min=-3,
    max=3,
    step=0.1,
    description='Initial Offset:',
    slider_color='blue'
)

learn_slider = widgets.FloatSlider(
    value=.5,
    min=0,
    max=1,
    step=0.1,
    description='Learn Rate:',
    slider_color='gray'
)

plot_button = widgets.Button(description="PLOT + Get Margin")

loss_select = widgets.Dropdown(options=['Hinge Loss', 'Zero-Plus Loss', 'Log Loss'])


# HTML
LOSS_TEMPLATE = '''<h4 style='color: darkblue; width: 900px; text-align: center;'>LOSS FUNCTIONS:</h4>
<ul>
<li><b>Zero-One Loss</b>: {zo}</li>
<li><b>Hinge Loss</b>: {hg:.2f}</li>
<li><b>Log Loss</b>: {log:.2f}</li>
</ul>
'''
loss_textarea = widgets.HTML(
    value='',
    )

# Final Layout
left_panel = widgets.VBox(children=[slope_slider, vspace, offset_slider, vspace, learn_slider]) 
right_panel = widgets.VBox(children=[vspace, plot_button, vspace, loss_select]) 

footer_panel = widgets.VBox(children=[hspace, loss_textarea])
body_panel = widgets.HBox(children=[left_panel, hspace, hspace, hspace, right_panel])

dashboard.children = [header, vspace, body_panel, footer_panel]


## EVENT HANDLERS -------------------------------------------------------------

def plot_on_demand(_):
    input_values = (slope_slider.value, offset_slider.value)
    loss_textarea.value = LOSS_TEMPLATE.format(zo=zero_plus_loss(*input_values),
                                               hg=hinge_loss(*input_values),
                                               log=log_loss(*input_values) )
    clear_output()
    grad_descent_plot(slope_slider.value, offset_slider.value, loss_select.value, learn_slider.value)

plot_button.on_click(plot_on_demand)

