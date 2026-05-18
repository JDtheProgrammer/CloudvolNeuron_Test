"""
MatplotlibVis.py
================

Purpose
-------
Provide ``NeuronMatplotlibVisualizer``, a thin helper that traverses every
section returned by ``neuron.n.allsec()``, extracts its 3D point list
(``x3d``, ``y3d``, ``z3d``, ``diam3d``), and renders the resulting
morphology in a matplotlib 3D axis. The class can also pair this
matplotlib view with NEURON's native ``PlotShape`` window so both GUIs
appear simultaneously.

Inputs
------
A live NEURON model in the calling process. The class queries each
``Section`` for its 3D points; if a section has no explicit ``pt3d``
list, ``n.define_shape()`` (called before plotting) makes NEURON compute
one automatically from L / diam.

Optionally a ``neuron.h.PlotShape`` (or ``n.PlotShape``) instance can be
supplied to ``wrap_and_show`` / ``show_both`` so the NEURON shape window
is opened alongside the matplotlib figure.

Outputs
-------
* A ``matplotlib.figure.Figure`` returned from ``plot()``.
* If ``save_path`` is given, an image file (PNG by default) at 150 dpi.
* If ``show`` is True, an interactive matplotlib window.

How the module works
--------------------
1. ``_read_section_points`` extracts 3D coordinates for one section.
2. ``plot`` iterates over all sections, drawing each segment as a line
   whose width scales with the local diameter and adding a scatter
   marker at every 3D point.
3. ``wrap_show`` / ``wrap_and_show`` store a reference to a NEURON
   ``PlotShape`` so it can be displayed later via ``show_both``.
4. ``show_both`` opens the stored PlotShape and re-draws the matplotlib
   view, recovering gracefully if either GUI is unavailable.

Required modules
----------------
    neuron (imported lazily inside the plotting methods),
    matplotlib, mpl_toolkits.mplot3d, numpy,
    Python stdlib: typing.

Usage example (in BallStickDemo.py):

    from MatplotlibVis import NeuronMatplotlibVisualizer
    viz = NeuronMatplotlibVisualizer()
    ps = n.PlotShape(True)
    viz.wrap_and_show(ps)
    viz.show_both(ps, 0)   # show NEURON PlotShape and matplotlib 3D

Author / Project: CloudvolNeuron_Test
"""

from typing import Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np


class NeuronMatplotlibVisualizer:
    """Plot NEURON morphology in matplotlib 3D and integrate with NEURON's GUI.

    Parameters
    ----------
    scale_lw : float
        Multiplier converting the local section diameter (um) into the
        matplotlib line width used to draw that segment. Default 0.05
        keeps drawings legible without saturating the axes for thick
        somata.
    scatter_scale : float
        Multiplier converting diameter (um) into scatter marker size.
        The marker area is ``(d * scatter_scale) ** 2``.
    figsize : tuple[float, float]
        Inches passed to ``plt.figure(figsize=...)``.

    Attributes
    ----------
    _last_plotshape : neuron.PlotShape or None
        Most recently registered NEURON PlotShape, used by ``show_both``.
    _last_save_path : str or None
        Save path remembered for re-use by ``show_both``.
    _last_show_flag : bool
        Whether to call ``plt.show`` after the matplotlib build.
    """

    def __init__(self, scale_lw: float = 0.05, scatter_scale: float = 0.5, figsize=(7, 6)):
        self.scale_lw = scale_lw
        self.scatter_scale = scatter_scale
        self.figsize = figsize
        self._last_plotshape = None
        self._last_save_path = None
        self._last_show_flag = True

    # -------------------------------------------------------------------------
    # Internal helper to extract 3D morphology from a NEURON section
    # -------------------------------------------------------------------------
    def _read_section_points(self, sec):
        """Return ``(xs, ys, zs, ds)`` for one ``neuron.Section``.

        Prefers the section's explicit ``n3d()`` point list. If the section
        has none, falls back to ``psection()["morphology"]["pt3d"]``. If
        neither yields points, returns four empty lists so the caller can
        skip the section silently.

        All coordinates are returned in micrometres, matching NEURON's
        internal convention.
        """
        n3d = int(sec.n3d())
        if n3d > 0:
            xs = [sec.x3d(i) for i in range(n3d)]
            ys = [sec.y3d(i) for i in range(n3d)]
            zs = [sec.z3d(i) for i in range(n3d)]
            ds = [sec.diam3d(i) for i in range(n3d)]
            return xs, ys, zs, ds

        p = sec.psection()
        pts = p.get('morphology', {}).get('pt3d', [])
        if not pts:
            return [], [], [], []
        xs = [pt[0] for pt in pts]
        ys = [pt[1] for pt in pts]
        zs = [pt[2] for pt in pts]
        ds = [pt[3] for pt in pts]
        return xs, ys, zs, ds

    # -------------------------------------------------------------------------
    # Main plotting method
    # -------------------------------------------------------------------------
    def plot(self, save_path: Optional[str] = None, show: bool = True, ps=None):
        """Build and optionally show/save a matplotlib 3D plot of the model.

        If a PlotShape `ps` is provided, it will be linked so viz.show_both(ps, 0)
        displays both NEURON and matplotlib GUIs.
        """
        import neuron
        from neuron import n

        try:
            n.define_shape()
        except Exception:
            pass

        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')

        any_points = False
        for sec in n.allsec():
            xs, ys, zs, ds = self._read_section_points(sec)
            if not xs:
                continue
            any_points = True

            for i in range(len(xs) - 1):
                xseg = [xs[i], xs[i + 1]]
                yseg = [ys[i], ys[i + 1]]
                zseg = [zs[i], zs[i + 1]]
                lw = max(0.2, (ds[i] + ds[i + 1]) / 2.0) * self.scale_lw
                ax.plot(xseg, yseg, zseg, color='k', linewidth=lw)

            sizes = [(d * self.scatter_scale) ** 2 for d in ds]
            ax.scatter(xs, ys, zs, s=sizes, alpha=0.6)

        if not any_points:
            raise RuntimeError('No 3D morphology points found in any section.')

        ax.set_xlabel('X (µm)')
        ax.set_ylabel('Y (µm)')
        ax.set_zlabel('Z (µm)')
        ax.set_title('Neuron morphology (matplotlib 3D)')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
        if show:
            plt.show(block=False)

        if ps is not None:
            self.wrap_and_show(ps, save_path=save_path, show=show)
            self.show_both(ps, 0)

        return fig

    # -------------------------------------------------------------------------
    # Apply visualizer directly to a PlotShape
    # -------------------------------------------------------------------------
    def apply_to_plotshape(self, ps, save_path: Optional[str] = None, show: bool = True):
        """Apply visualizer using a NEURON PlotShape object."""
        try:
            from neuron import n
            n.define_shape()
        except Exception:
            pass
        return self.plot(save_path=save_path, show=show)

    # -------------------------------------------------------------------------
    # Safe: cannot attach attributes to hoc.PlotShape, so manage externally
    # -------------------------------------------------------------------------
    def wrap_show(self, ps, save_path: Optional[str] = None, show: bool = True):
        """Store PlotShape reference and prepare for viz.show_both(ps)."""
        self._last_plotshape = ps
        self._last_save_path = save_path
        self._last_show_flag = show
        print("✅ Stored PlotShape reference — use viz.show_both(ps) to display both GUIs.")

    def show_both(self, ps=None, *args, **kwargs):
        """Show both NEURON PlotShape and Matplotlib visualization."""
        if ps is None:
            ps = getattr(self, "_last_plotshape", None)
        if ps is None:
            raise ValueError("No PlotShape object available. Call wrap_show(ps) first.")

        result = None
        try:
            result = ps.show(*args, **kwargs)
        except Exception as e:
            print(f"[NeuronMatplotlibVisualizer] NEURON GUI error: {e}")

        try:
            self.apply_to_plotshape(
                ps,
                save_path=self._last_save_path,
                show=self._last_show_flag
            )
        except Exception as e:
            print(f"[NeuronMatplotlibVisualizer] Matplotlib visualization error: {e}")

        return result

    def wrap_and_show(self, ps, save_path: Optional[str] = None, show: bool = True):
        """Prepare PlotShape reference; call viz.show_both(ps) to display both GUIs."""
        self.wrap_show(ps, save_path=save_path, show=show)
        print("✅ Ready — call viz.show_both(ps, 0) to show both NEURON and Matplotlib GUIs.")
        return ps


# -------------------------------------------------------------------------
# Demo mode if run directly
# -------------------------------------------------------------------------
if __name__ == '__main__':
    try:
        import neuron
        from neuron import n
    except Exception:
        print('NEURON not available — import failed.')
    else:
        print('NEURON available, but __main__ demo requires a model to be defined.')
