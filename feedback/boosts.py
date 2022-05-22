import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from p24asolver import P24ASolver
from math import sin, cos, sqrt
import multiprocessing as mp

AU = 1.495978707e11  # m
mSolar = 1.989e30  # kg


class Planet:
    def __init__(
        self,
        name: str,
        mass: float,
        semimajor: float,
        eccentricity: float,
        radius: float,
        theta0=0,
        periapsis=0,
    ):
        self.name = name
        self.mass = mass
        self.a = semimajor
        self.radius = radius
        self.period = None
        self.alpha = None
        self.e = None
        self.θ0 = theta0
        self.periapsis = periapsis
        self.eccentricity = eccentricity
        self.n = 4  # offset in Y to planet's θ

    def __str__(self):
        left = ['planet', 'mass (mMsolar)', 'a (AU)', 'radius (mAU)', 'period (y)', 'e']
        right = "%s;%.3g;%.3f;%.3g;%.2f;%.4g" % (
            self.name,
            self.mass * 1000,
            self.a,
            self.radius * 1000,
            self.period,
            self.eccentricity,
        )
        width = 2 + max([len(x) for x in left])
        fmt = "{0:>%ds}  {1}" % width
        return "\n".join([fmt.format(*x) for x in zip(left, right.split(';'))])

    def r(self, θ):
        return self.alpha / (1 + self.e * np.cos(θ - self.θ0))

    def xy(self, θ):
        r = self.r(θ)
        res = r * np.array([np.cos(θ), np.sin(θ)])
        return np.transpose(res)

    @property
    def eccentricity(self):
        return self.e

    @eccentricity.setter
    def eccentricity(self, e):
        self.e = e
        self.alpha = self.a * (1 - self.e**2)
        self.period = (self.a**1.5) / sqrt(1 + self.mass)

    def to_satellite(self, Y):
        """
        Return a unit vector in the satellite's (r, θ)
        basis that points from the satellite to the planet
        """
        twopi = np.pi * 2.0
        r, θ, θp = Y[0], Y[1], Y[self.n]
        rp = self.r(θp)
        dθ = (θp - θ) % twopi
        sign = 1 if dθ < np.pi else -1

        s2 = r**2 + rp**2 - 2 * r * rp * cos(dθ)
        s = sqrt(s2)
        cospsi = (r - rp * cos(dθ)) / s
        try:
            sinpsi = sqrt(1 - cospsi * cospsi)
        except:
            sinpsi = 0.0
        unit = np.array([-cospsi, sinpsi * sign])
        a = twopi * twopi * self.mass / s2
        return a, unit, s, rp, θp


Jupiter = Planet("Jupiter", 9.55e-4, 5.201, 0.0484, 7.1492e7 / AU)
Saturn = Planet("Saturn", 5.6834e26 / mSolar, 9.5826, 0.0565, 5.8232e7 / AU)


def rdot(t, Y, *args):
    "Event function looking for rdot passing through zero"
    return Y[2]


def crash(t, Y, *args):
    self = args[0]
    min_s = 1e10
    for p in self.planets:
        a, unit, s, rp, θp = p.to_satellite(Y)
        min_s = min(min_s, s - p.radius)
    return min_s


crash.terminal = True


class Satellite(P24ASolver):
    def __init__(self, planets, **kwargs):
        vars = [
            ('r', '$r$'),
            ('theta', r'$\theta$'),
            ('rdot', r'$\dot{r}$'),
            ('thetadot', r'$\dot{\theta}$'),
        ]
        for p in planets:
            vars.append(('theta_{p.name}', r'$\theta_{%s}$' % p.name))

        super().__init__(vars, **kwargs)
        try:
            self.planets = tuple(planets)
        except:
            self.planets = (planets,)
        self.n_planets = len(self.planets)

        # Let each planet know its position in Y
        for n in range(self.n_planets):
            self.planets[n].n = 4 + n

        self.events.append(rdot)
        self.events.append(crash)

    def derivatives(self, t, Y):
        r, θ, rdot, θdot = Y[:4]
        planets = Y[4:]
        twopi = 2 * np.pi
        fourpi2 = twopi * twopi
        ar = -fourpi2 / r**2  # Sun contribution
        aθ = 0.0
        dtheta_planets = []
        self.min_s = 1e10

        for n, ptheta in enumerate(planets):
            p = self.planets[n]
            a, unit, s, rp, θp = p.to_satellite(Y)

            ar += a * unit[0]
            aθ += a * unit[1]
            dtheta_planets.append(sqrt(p.alpha * fourpi2 * (1 + p.mass)) / rp**2)
            self.min_s = min(self.min_s, s - p.radius)
        res = [rdot, θdot, ar + r * θdot**2, (aθ - 2 * rdot * θdot) / r]
        res.extend(dtheta_planets)
        return res

    def frame(self, tvals):
        """
        blah
        """
        fourpi2 = 4 * np.pi * np.pi
        Y = self(tvals)
        columns = ['r', 'θ', 'rdot', 'θdot']
        columns.extend([f'θ{p.name}' for p in self.planets])
        df = pd.DataFrame(Y, columns=columns)
        df['x'] = Y[:, 0] * np.cos(Y[:, 1])
        df['y'] = Y[:, 0] * np.sin(Y[:, 1])
        # make place holders
        df['E'] = np.zeros(len(tvals))
        df['K'] = 0.5 * (df.rdot * df.rdot + np.power(df.r * df.θdot, 2))
        df['E'] = df['K']
        df['USun'] = -fourpi2 / df.r
        df['E'] += df['USun']

        for n, p in enumerate(self.planets):
            thetap = Y[:, n + 4]
            rp = p.r(thetap)
            xp, yp = rp * np.cos(thetap), rp * np.sin(thetap)
            name = p.name
            df[f'r{name}'] = rp
            df[f'x{name}'] = xp
            df[f'y{name}'] = yp
            s = np.linalg.norm(np.array([df['x'] - xp, df['y'] - yp]), axis=0)
            df[f's{name}'] = s
            df[f'U{name}'] = -fourpi2 * p.mass / s
            df['E'] += df[f'U{name}']
        df.index = tvals
        df.index.name = "t"
        return df.style.format(precision=3)

    def closest_approach(self, trange, planet=None):
        """
        Find the point of closest approach between the satellite and
        the given planet.
        """
        if planet is None:
            pname = f"s{self.planets[0].name}"
        else:
            pname = f"s{planet.name}"
        npts = 32
        f = self.frame(np.linspace(*trange, npts)).data
        while trange[1] - trange[0] > 1e-6:
            n = np.argmin(f[pname])
            t_min = f.index[n]
            if n == 0:
                t_max = f.index[1]
            elif n == npts - 1:
                t_max = t_min
                t_min = f.index[npts - 2]
            else:
                t_min, t_max = f.index[n - 1], f.index[n + 1]
            trange = (t_min, t_max)
            f = self.frame(np.linspace(*trange, npts)).data
        n = np.argmin(f[pname])
        return f.iloc[n]

    def plot_trajectory(self, trange=None, N: int = 300, **kwargs):
        tr = trange if trange else self.time_range
        tvals = np.linspace(*tr, N)
        Y = self(tvals)

        Sun = kwargs.get('Sun', True)
        pframe = None  # if we are working in the frame of a planet, this will get set
        frame = kwargs.get('frame', "")  # if frame is the name of a planet...
        if frame and isinstance(frame, str):
            frame = frame.lower()
            for n, p in enumerate(self.planets):
                if frame == p.name.lower():
                    pframe = p
                    break

        f, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal')

        x, y = Y[:, 0] * np.cos(Y[:, 1]), Y[:, 0] * np.sin(Y[:, 1])

        if pframe:
            thetap = Y[:, 4 + n]
            xy = pframe.xy(thetap)
            ax.add_patch(Circle((0, 0), radius=pframe.radius, color='r'))
            ax.plot(x - xy[:, 0], y - xy[:, 1], 'b.-')
        else:
            if Sun:
                ax.plot(
                    [
                        0,
                    ],
                    [
                        0,
                    ],
                    'yo',
                )  # the Sun!
            ax.plot(x, y, 'b', label='satellite')
            for n, p in enumerate(self.planets):
                thetap = Y[:, 4 + n]
                xy = p.xy(thetap)
                ax.plot(xy[:, 0], xy[:, 1], '.', label=p.name)
                if n == 0:
                    skipper = 10
                    xx, yy = [xy[0, 0],], [
                        xy[0, 1],
                    ]
                    for n in range(0, len(xy) - skipper, skipper):
                        if (n // skipper) % 2:
                            xx.append(xy[n, 0])
                            xx.append(xy[n + skipper, 0])
                            yy.append(xy[n, 1])
                            yy.append(xy[n + skipper, 1])
                        else:
                            xx.append(x[n])
                            xx.append(x[n + skipper])
                            yy.append(y[n])
                            yy.append(y[n + skipper])
            ax.plot(xx, yy, 'k', alpha=0.2, lw=0.25)

    def max_boost(self, Y0: list, t_final, vary: int, bounds=None):
        """Attempt to achieve maximum energy following encounter
        with one or more planets at end of integration time t_final.
        """
        p = Y0[vary]
        param = []
        t_range = (0, t_final)
        efinal = []

        # If bounds are specified, go with it
        # if not, we need to manufacture bounds
        if bounds is None:
            assert p != 0
            bounds = (p * 0.9, p * 1.1)

        for loop in range(2):
            pvals = np.linspace(*bounds, 11)
            for p in pvals:
                Y0[vary] = p
                self.solve(Y0, t_range)
                # Did we make it to the end of the integration?
                if self.solution.success:
                    f = self.frame([t_final])
                    E = float(f.data.E)
                    efinal.append(E)
                    param.append(p)
            # Now let's see who is best so far
            order = np.argsort(param)
            params = np.array(param)[order]
            energies = np.array(efinal)[order]
            order = np.argsort(energies)
            best_n = order[-1]
            best_E = energies[best_n]
            # If the best_n is either 0 or len(params)-1, we fail
            if best_n == 0 or best_n == len(params) - 1:
                raise Exception("Failed to find maximum in interval")
            # adjust the bounds
            bounds = (params[best_n - 1], params[best_n + 1])
        Y0[vary] = params[best_n]
        return Y0, best_E

    def plot_satellite_energy(self, trange=None, N: int = 150):
        if trange is None:
            trange = self.time_range
        tvals = np.linspace(*trange, N)
        df = self.frame(tvals)
        f, a = plt.subplots()
        a.plot(tvals, df.data.E)
        a.set_xlabel("$t$ (y)")
        a.set_ylabel("$E$")

    def geometry(self, Y):
        """Show the geometry of the Sun-Jupiter-satellite system"""

        # Use the derivatives function to compute everthing
        self.derivatives(0.0, Y)
        res = self.res

        x, y = res['r'] * cos(res['theta']), res['r'] * sin(res['theta'])
        xv, yv = [x,], [
            y,
        ]
        f, ax = plt.subplots()
        ax.plot(xv, yv, 'ko')  # satellite
        ax.set_aspect('equal')

        # ax.add_patch(Circle((0,0), radius=0.1, color='y')) # sun

        for n, p in enumerate(self.planets):
            xy = p.xy(Y[4 + n])
            xv.append(xy[0])
            yv.append(xy[1])
            vals = res['planets'][n]
            ax.plot([xy[0]], [xy[1]], 'ro')
            # ax.add_patch(Circle(xy, radius=0.1, color='r'))
            uv = -res['rhat'] * vals['cospsi'] + res['thetahat'] * vals['sinpsi']

            scale = np.linalg.norm(np.array([max(xv) - min(xv), max(yv) - min(yv)])) / 2
            ax.arrow(x=x, y=y, dx=uv[0] * scale, dy=uv[1] * scale, width=0.02 * scale)

    def prepare_figure(self):
        """
        I have added keyword arguments to enable customization of the animation.
        In particular, I'd like to be able to focus on the interaction with a planet
        """

        kwargs = self.animate_kwargs
        fig, ax = plt.subplots(figsize=(10, 8))
        t_range = kwargs['time_range']
        df = self.frame(np.linspace(*t_range, 100)).data
        xrange = [df.x.min(), df.x.max()]
        yrange = [df.y.min(), df.y.max()]
        dx = (xrange[1] - xrange[0]) / 5
        ax.set_xlim(xrange[0] - dx, xrange[1] + dx)
        dy = (yrange[1] - yrange[0]) / 5
        ax.set_ylim(yrange[0] - dy, yrange[1] + dy)

        self.shapes = []

        # Prepare ellipses for the planets
        θ = np.linspace(0, 2 * np.pi, 200)
        for p in self.planets:
            xy = p.xy(θ)
            ax.plot(xy[:, 0], xy[:, 1], 'k-', alpha=0.1)

        sun = Circle((0, 0), radius=0.05, color='y')
        ax.add_patch(sun)
        ax.set_aspect('equal')
        return fig, ax

    def draw_frame(self, t):
        Y = self(t)
        x, y = Y[0] * cos(Y[1]), Y[0] * sin(Y[1])
        sat = Circle((x, y), radius=0.02, color='k')
        for n in range(len(self.shapes) - 1, -1, -1):
            self.shapes[n].remove()
        self.shapes = [
            sat,
        ]
        for p in self.planets:
            xy = p.xy(Y[p.n])
            self.shapes.append(Circle(xy, radius=0.05))

        for s in self.shapes:
            self._ax.add_patch(s)
        return self.shapes


class CartesianSatellite(P24ASolver):
    def __init__(self, planets, **kwargs):
        vars = (
            ('thetaJ', r'$\theta_J$'),
            ('x', '$x$'),
            ('y', r'$y$'),
            ('xdot', r'$\dot{x}$'),
            ('ydot', r'$\dot{y}$'),
        )
        super().__init__(vars, **kwargs)
        try:
            self.planets = tuple(planets)
        except:
            self.planets = (planets,)
        self.n_planets = len(self.planets)

    def derivatives(self, t, Y):
        x, y, xdot, ydot = Y[:4]
        planets = Y[4:]
        twopi = 2 * np.pi
        fourpi2 = twopi * twopi

        r = np.linalg.norm((x, y))
        mag = -fourpi2 / r**3
        ax = mag * x
        ay = mag * y

        dtheta_planets = []

        for n, ptheta in enumerate(planets):
            p = self.planets[n]
            xp, yp = p.xy(ptheta)
            dx, dy = xp - x, yp - y
            s = np.linalg.norm((dx, dy))
            mag = fourpi2 / s**3 * p.mass
            ax += mag * dx
            ay += mag * dy
            dtheta_planets.append(
                sqrt(p.alpha * fourpi2 * (1 + p.mass)) / (xp**2 + yp**2)
            )

        res = [xdot, ydot, ax, ay]
        res.extend(dtheta_planets)
        return res


# Now to work on the planet clearing its orbit problem


def stray(t, Y, *args):
    """
    Event that triggers when the satellite wanders far enough away
    from a chosen radius. We assume that args[0] holds the Satellite
    object, and that two additional fields have been added to it:
    r0 holds the starting (reference) radius
    Δr holds the magnitude of departure that fires the event and terminates
       the integration
    """
    self = args[0]
    return abs(Y[0] - self.r0) - self.Δr


stray.terminal = True


def simsat(args, periods=1000, Δr=0.1):
    """
    Given a single planet, an initial circular orbit whose
    radius differs relatively from the planet's semimajor axis by
    δr, and which starts in a circular orbit with an angular remove
    of θ0 from the planet's position, simulate for the given
    number of periods of the planet's orbit and stop if the satellite's
    orbital radius deviates relatively by more than Δr.
    """
    n, planet, δr, θ0 = args
    planeta = planet.a  # the semimajor axis of the planet's orbit
    sat = Satellite([planet])

    sat.r0 = planeta * (1 + δr)
    sat.Δr = Δr * planeta
    sat.events = [stray, crash]
    ω = 2 * np.pi * sat.r0 ** (-1.5)
    ic = [sat.r0, θ0, 0.0, ω, 0.0]
    sat.solve(ic, (0, periods * planet.period))

    sat.strayed, sat.crashed = [len(sat.solution.t_events[n]) > 0 for n in range(2)]
    tfinal = sat.time_range[1]
    if sat.strayed:
        tfinal = sat.solution.t_events[0][0]
    elif sat.crashed:
        tfinal = sat.solution.t_events[1][0]
    sat.time_range = (0, tfinal)
    # Collect all the interesting statistics in a dictionary
    sat.results = dict(
        n=n,
        strayed=sat.strayed,
        crashed=sat.crashed,
        δr=ic[0] / planeta - 1.0,  # the relative initial deviation
        Δθ=ic[1],  # the angular displacement from Jupiter
        t_final=sat.time_range[1],
        δr_final=sat(sat.time_range[1])[0] / planeta - 1.0,
        m=planet.mass,
    )
    return sat


def simres(*args):
    s = simsat(args, 1000)
    res = s.results
    txt = f"{res['n']}\t"
    txt += "1\t" if res['strayed'] else "0\t"
    txt += "1\t" if res['crashed'] else "0\t"
    txt += "\t".join(
        [f"{res[x]:.6g}" for x in ('δr', 'Δθ', 't_final', 'δr_final', 'm')]
    )
    open('tmp.csv', 'a').write(txt + '\n')
    return res


pool_results = []


def collect_results(r):
    global pool_results
    pool_results.append(r)


def pool_ack(e):
    print('error')
    print(dir(e), "\n")
    print(f"-->{e.__cause__}<--")


def parallel_run(ics, planet):
    processors = mp.cpu_count() - 2
    pool = mp.Pool(processors, maxtasksperchild=10)
    # res = pool.map_async(simres, ics)
    for n, ic in enumerate(ics):
        pool.apply_async(
            simres,
            args=(n, planet, *ic),
            callback=collect_results,
            error_callback=pool_ack,
        )
    pool.close()
    pool.join()
    # return res


if __name__ == '__main__':
    from copy import deepcopy

    planet = deepcopy(Jupiter)
    planet.mass *= 10
    # planet.mass = planet.mass / 2
    ics = [
        (δr, θ0)
        for δr in np.linspace(-0.6, 0.6, 200)
        for θ0 in np.linspace(0.01, 2 * np.pi - 0.01, 50)
    ]
    open('tmp.csv', 'w').write('n\tstrayed\tcrashed\tδr\tΔθ\tt_final\tδr_final\tm\n')
    parallel_run(ics, planet)
