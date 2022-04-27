#! /usr/bin/env python3
# coding:utf-8
"""
  Author:  Peter N. Saeta --<saeta@hmc.edu>
  Purpose: To package a number of common functions for solving differential equations
  Created: 03/26/20
"""

import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
from matplotlib import animation
from IPython.display import HTML


class P24ASolver:
    """
    Base class for solving a system of first-order differential
    equations, plotting results, and creating animations. To use
    this code, create a subclass as illustrated in "Project Template.ipynb"
    and define (at minimum) the following methods:

    __init__(self, time_range, **kwargs)
        Your function must call
            super().__init__(self, time_range, VARIABLE_NAMES)
        where VARIABLE_NAMES is a list of 2-tuples, each of which holds two
        strings. The first one is the variable name you will use to
        reference the variable and the second is the LaTeX representation
        of the variable's name. For example ('alpha', r'$\alpha$').
        Your __init__() method should also add fields to self that
        store the various parameter values for the simulation.

    derivatives(self, t, Y)
        Your function to compute the derivatives of the variables
        stored in Y at time t. Return a list of the derivative values.
        Remember to keep them in the same order as the
        variables are given in Y.

    Call

        self.solve(Y0) to solve the equations

    If you'd like to make animations, you need to provide two
    more functions:

    prepare_figure(self)
        make a call to matplotlib.pyplot to generate
        a figure and one or more axes. Adjust the axes as you wish,
        add any plots you like (typically with empty data lists) and
        store those in self.lines. Return (fig, axes).

    draw_frame(self, n)
        calculate data necessary to update the
        self.lines defined in prepare_figure to draw the nth frame
        out of a total of self.animate_kwargs['frames'] and return
        the list of lines that actually were updated. If you return
        nothing, we assume that all lines were updated.

    Then you can call

        animate(num_frames, **kwargs)
    """

    def __init__(
        self,
        variable_names:tuple,  # (("theta", r"$\theta$"), ("x", "$x$"))
        **kwargs):
        self.time_range = None
        self.variable_names = variable_names
        self.solution = None
        self.method = kwargs.get('method','RK45')
        self.rtol = kwargs.get('rtol', 1e-8)
        self.atol = kwargs.get('atol', 1e-8)
        self.events = []

    def __str__(self):
        return f"Solution for {self.time_range} has {'' if self.solution else 'not '} been computed"

    def __call__(self, t):
        """
        You can evaluate the variables at time t with a call to
        obj(t) for a P24ASolver object called obj.
        """
        if self.solution:
            if isinstance(t, (float, int)):
                return self.solution.sol(t)
            try:
                import numpy as np
                res = np.array([
                    self.solution.sol(x) for x in t
                ])
                return res
            except:
                pass

    @staticmethod
    def funcall(t, yy, self):
        der = self.derivatives(t, yy)
        return der

    def solve(self, Y0, time_range, **kwargs):
        """
        Solve the system of equations starting from initial conditions Y0.
        If you want to use solve_ivp's event mechanism, add one or more functions
        to self.events. Your function should have the form myfunction(t, Y) and
        be a member of your subclass of P24ASolver.
        """
        self.time_range = time_range
        
        # Process any keyword arguments
        for k,v in kwargs.items():
            setattr(self, k, v)
        
        self.solution = solve_ivp(
            self.funcall,
            self.time_range,
            Y0,
            method=self.method,
            dense_output=True,
            rtol = self.rtol,
            atol = self.atol,
            events=self.events,
            args = (self, )
        )
        return self

    def _names_to_indices(self, names):
        "Return a tuple of indices corresponding to the named variables"
        all_names = [x[0] for x in self.variable_names]
        return [all_names.index(x) for x in names]

    def derivatives(self, t, Y):
        "You need to override this function in your class"
        raise Exception(
            "Please define derivatives(self, t, Y) in your subclass")

    def plot(self, tvals, what_to_plot, **kwargs):
        """
        Add a plot of one or more variables to the passed axes.
        Prepare an np.arange(t_start, t_end, dt) or
        np.linspace(t_start, t_end, number_of_points) to cover
        the range of the solution you wish to plot.
        what_to_plot should be a list or tuple of variable
        names.

        Possible keyword arguments you can pass:

        axes
        title:str - a string to display
        logx:boolean - if True, display on a logarithmic time axis
        logy:boolean - if True, display variables on a logarithmic axis
        """
        axes = kwargs.get('axes')
        if axes == None:
            _, axes = plt.subplots(figsize=kwargs.get('figsize'))
        else:
            axes.clear()
        logx = kwargs.get('logx', False)
        logy = kwargs.get('logy', False)
        Y = self.solution.sol(tvals)
        if logx:
            axes.set_xscale('log')
        if logy:
            axes.set_yscale('log')

        for n in self._names_to_indices(what_to_plot):
            axes.plot(tvals, Y[n, :], label=self.variable_names[n][1])
        axes.set_xlabel(r"$t$")
        axes.legend()
        if 'title' in kwargs:
            axes.set_title(kwargs['title'])
        return self

    def frame(self, tvals, what_to_include):
        """
        Prepare a pandas DataFrame of values from the simulation
        """
        Y = self.solution.sol(tvals)
        df = pd.DataFrame(index=tvals)
        if what_to_include == 'all':
            what_to_include = [x[0] for x in self.variable_names]
        for n in self._names_to_indices(what_to_include):
            name = self.variable_names[n][0]
            df[name] = Y[n, :]
        return df

    def progress(self, fraction):
        """
        This doesn't seem to help during the computation of the animation.
        """
        try:
            import ipywidgets as widgets
            from IPython.display import display

            if not hasattr(self, 'progress_bar'):
                self.progress_bar = widgets.FloatProgress(
                    value=0,
                    min=0,
                    max=1.0,
                    step=0.01,
                    description='Progress:',
                    bar_style='info',
                    orientation='horizontal',
                    layout=widgets.Layout(width='500px')
                )
                display(self.progress_bar)

            if fraction < 0:
                del self.progress_bar
            else:
                self.progress_bar.value = fraction
        except:
            pass

    def prepare_figure(self):
        return False

    def draw_frame(self, t):
        return False

    def animate(self, frames:int, **kwargs):
        """
        Produce an animation. Optional keyword arguments are

            time_range (defaults to the full range of the solution)
            interval (in ms)
            blit (defaults to True)
        """
        try:
            time_range = kwargs.get('time_range', [0, self.solution.t[-1]])
            self.animate_kwargs = kwargs
            self.animate_kwargs['time_range'] = time_range
            self.animate_kwargs['dt'] = (
                time_range[1] - time_range[0]) / (frames - 1)
            self.animate_kwargs['frames'] = frames
            self._animation = None

            self._fig, self._ax = self.prepare_figure()  # create the figure to animate
            # self.init_animation()
            kw = self.animate_kwargs
            self._animation = animation.FuncAnimation(
                self._fig,
                lambda n: self.draw_frame(
                    kw['time_range'][0] + n * kw['dt']),
                frames=frames,  # should be the number of frames
                interval=kwargs.get('interval', 100),  # interval in ms
                blit=kwargs.get('blit', True)
            )

        except Exception as eeps:
            print(eeps)
            print("""
            To create an animation, your subclass must define two methods:
            prepare_figure(self)            return figure, axes
            draw_frame(self, frame_number)  return a tuple of the items updated
            """)
        return self._animation

    def save_animation(self, **kwargs):
        """
        Render the animation to an object that can be saved.
        """
        print("I'm going to start rendering now. This can take a long time...")
        mode = kwargs.get('renderer', 'javascript')
        if mode == 'ffmpeg':
            h = HTML(self._animation.to_html5_video())
        elif mode == 'javascript':
            h = HTML(self._animation.to_jshtml())
        self.progress(-1)  # reset the progress bar
        return h

