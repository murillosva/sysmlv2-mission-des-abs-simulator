"""Closed-form collision-course intercept solver.

Given a constant-velocity target UAS at position :math:`\\mathbf{p}_u` with
velocity :math:`\\mathbf{v}_u`, and an interceptor at :math:`\\mathbf{p}_a` that
can choose its heading but flies at constant speed :math:`v_a`, the minimum
collision time :math:`t^*` is the smaller positive root of

.. math::

    (v_a^2 - \\|\\mathbf{v}_u\\|^2)\\,t^2 - 2\\,\\Delta\\mathbf{p}\\cdot\\mathbf{v}_u\\,t - \\|\\Delta\\mathbf{p}\\|^2 = 0

where :math:`\\Delta\\mathbf{p} = \\mathbf{p}_u - \\mathbf{p}_a`. The required
heading is

.. math::

    \\mathbf{p}^* &= \\mathbf{p}_u + \\mathbf{v}_u \\cdot t^*\\\\
    h_a &= \\mathrm{atan2}(p^*_y - p_{a,y},\\ p^*_x - p_{a,x})

This module computes :math:`t^*`, the intercept point, the interceptor
heading, and the time the interceptor first enters the weapon envelope (the
time at which :math:`\\|\\mathbf{p}_a(t) - \\mathbf{p}_u(t)\\| = d_{\\mathrm{env,max}}`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from cuas_sim.geometry.projection import XY


@dataclass(frozen=True, slots=True)
class InterceptGeometry:
    """Result of a collision-course intercept computation.

    All times are in seconds **from the moment of notification** (relative).
    All distances are in meters on the projected plane.

    Attributes
    ----------
    feasible
        ``True`` if a positive intercept time exists. ``False`` indicates the
        UAS is faster than the A-29 and runs away (geometrically there is no
        intercept).
    t_intercept
        Time to collision in seconds. NaN if not feasible.
    intercept_xy
        Intercept point in projected meters. (NaN, NaN) if not feasible.
    a29_heading_rad
        Interceptor heading in radians (atan2 convention, 0 = +x = East).
        NaN if not feasible.
    t_envelope_entry
        Time at which the A-29 first reaches distance ``d_envelope_max`` from
        the UAS. May be slightly less than ``t_intercept``. NaN if not
        feasible or the envelope is never reached.
    initial_distance_m
        Initial range from A-29 to UAS at the moment of notification.
    """

    feasible: bool
    t_intercept: float
    intercept_xy: XY
    a29_heading_rad: float
    t_envelope_entry: float
    initial_distance_m: float


def collision_course_intercept(
    a29_xy: XY,
    a29_speed_mps: float,
    uas_xy: XY,
    uas_heading_rad: float,
    uas_speed_mps: float,
    envelope_max_m: float,
) -> InterceptGeometry:
    """Solve for the collision-course intercept geometry.

    Parameters
    ----------
    a29_xy
        Current A-29 position on the projected plane (m).
    a29_speed_mps
        A-29 cruise speed (m/s).
    uas_xy
        Current UAS position on the projected plane (m).
    uas_heading_rad
        UAS heading in radians (atan2 convention).
    uas_speed_mps
        UAS speed (m/s).
    envelope_max_m
        Weapon envelope maximum range for the relevant configuration. The
        time at which the A-29 first enters this distance from the UAS
        defines :attr:`InterceptGeometry.t_envelope_entry`.

    Returns
    -------
    InterceptGeometry
        Closed-form solution; see field docs.

    .. deprecated:: 1.0.0
       For new code in PURSUING, use :func:`pure_pursuit_step`.
       This function is retained for animation replay, regression
       fixtures, and backwards compatibility.
    """
    dx = uas_xy.x - a29_xy.x
    dy = uas_xy.y - a29_xy.y
    d0 = math.hypot(dx, dy)
    vux = uas_speed_mps * math.cos(uas_heading_rad)
    vuy = uas_speed_mps * math.sin(uas_heading_rad)

    # (vA^2 - vU^2) t^2 - 2 Δp·vU t - |Δp|^2 = 0
    a = a29_speed_mps**2 - uas_speed_mps**2
    b = -2.0 * (dx * vux + dy * vuy)
    c = -(d0**2)

    t_int = _solve_intercept_time(a, b, c)
    if t_int is None or t_int <= 0:
        return _infeasible(d0)

    # Intercept point and required heading
    px = uas_xy.x + vux * t_int
    py = uas_xy.y + vuy * t_int
    head = math.atan2(py - a29_xy.y, px - a29_xy.x)

    # Time of envelope entry: first instant the closing separation equals
    # envelope_max_m. Delegates to the shared closed-form primitive so the
    # same root logic serves envelope entry and weapon release.
    t_env = time_to_separation(
        a29_xy, a29_speed_mps, uas_xy, uas_heading_rad, uas_speed_mps,
        a29_heading_rad=head, target_sep_m=envelope_max_m,
    )
    if t_env is None or t_env > t_int + 1e-6:
        # Either we never reach the envelope (e.g. envelope > initial dist
        # and recession), or the algebraic solution is past collision.
        # Clamp to t_int; the engagement starts at intercept.
        t_env = t_int if d0 > envelope_max_m else 0.0

    return InterceptGeometry(
        feasible=True,
        t_intercept=t_int,
        intercept_xy=XY(x=px, y=py),
        a29_heading_rad=head,
        t_envelope_entry=max(0.0, t_env),
        initial_distance_m=d0,
    )


def time_to_separation(
    a29_xy: XY,
    a29_speed_mps: float,
    uas_xy: XY,
    uas_heading_rad: float,
    uas_speed_mps: float,
    *,
    a29_heading_rad: float,
    target_sep_m: float,
) -> float | None:
    r"""First positive instant the closing A-29↔UAS separation equals ``target_sep_m``.

    With the A-29 flying a fixed heading ``a29_heading_rad`` at constant
    ``a29_speed_mps`` and the UAS on a constant-velocity course, the relative
    position is :math:`r(t) = \Delta p + w\,t` where
    :math:`\Delta p = p_u - p_a` and :math:`w = v_u - v_a\,\hat u`. Setting
    :math:`\lvert r(t)\rvert = d` yields the quadratic
    :math:`A t^2 + B t + C = 0` with :math:`A = \lVert w\rVert^2`,
    :math:`B = 2\,\Delta p\cdot w`, :math:`C = \lVert\Delta p\rVert^2 - d^2`.
    Returns the smallest strictly-positive root, or ``None`` if none exists.

    This is the shared primitive used both for weapon-envelope entry
    (``target_sep_m = envelope_max_m``) and for the weapon-release instant
    (``target_sep_m = d_fire``).
    """
    dx = uas_xy.x - a29_xy.x
    dy = uas_xy.y - a29_xy.y
    vux = uas_speed_mps * math.cos(uas_heading_rad)
    vuy = uas_speed_mps * math.sin(uas_heading_rad)
    vax = a29_speed_mps * math.cos(a29_heading_rad)
    vay = a29_speed_mps * math.sin(a29_heading_rad)
    wx = vux - vax
    wy = vuy - vay
    A = wx**2 + wy**2
    B = 2.0 * (dx * wx + dy * wy)
    C = dx**2 + dy**2 - target_sep_m**2
    return _earliest_positive_root(A, B, C)


def pure_pursuit_step(
    a29_xy: tuple[float, float],
    uas_xy: tuple[float, float],
    uas_vxy: tuple[float, float],
    v_a29: float,
    dt: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    r"""One discrete step of pure (lag) pursuit — A-29 aims at the current UAS position.

    The A-29 freezes its heading toward the UAS position at the start of the
    step and flies that heading at constant speed for ``dt`` seconds. The UAS
    holds its instantaneous velocity ``uas_vxy`` over the same interval. After
    the step the A-29 lands slightly off the new instantaneous bearing — the
    characteristic curve of pure pursuit::

              uas_{t}
                .
               / \
              /   . uas_{t+dt}
             /   /        ← A-29 aims here at the start of the step…
            /   /
           /  /
          / /
         a29_{t}  →→→  a29_{t+dt}    …and lands here, lagging the
                                     instantaneous bearing to uas_{t+dt}.

    Pure function — no RNG, no I/O. Used by ``EngagementController`` to drive
    the PURSUING phase one tracking update at a time.

    Parameters
    ----------
    a29_xy, uas_xy
        Current positions in projected meters (frame defined by ``LocalProjection``).
    uas_vxy
        UAS instantaneous velocity vector (m/s); held fixed across the step.
    v_a29
        A-29 cruise speed (m/s, > 0).
    dt
        Time step (s, > 0). Sourced from ``tracking_policy.next_update_interval``
        in production; arbitrary in unit tests.

    Returns
    -------
    tuple[tuple[float, float], tuple[float, float]]
        ``(a29_xy_new, uas_xy_new)`` after the step.
    """
    dx = uas_xy[0] - a29_xy[0]
    dy = uas_xy[1] - a29_xy[1]
    bearing = math.atan2(dy, dx)
    a29_new = (
        a29_xy[0] + v_a29 * math.cos(bearing) * dt,
        a29_xy[1] + v_a29 * math.sin(bearing) * dt,
    )
    uas_new = (
        uas_xy[0] + uas_vxy[0] * dt,
        uas_xy[1] + uas_vxy[1] * dt,
    )
    return a29_new, uas_new


def _infeasible(d0: float) -> InterceptGeometry:
    nan = float("nan")
    return InterceptGeometry(
        feasible=False,
        t_intercept=nan,
        intercept_xy=XY(x=nan, y=nan),
        a29_heading_rad=nan,
        t_envelope_entry=nan,
        initial_distance_m=d0,
    )


def _solve_intercept_time(a: float, b: float, c: float) -> float | None:
    """Return the smallest positive root of ``a t^2 + b t + c = 0``, or None.

    Handles the degenerate ``a == 0`` (equal speeds) case as a linear
    equation.
    """
    if abs(a) < 1e-12:
        # Linear: b*t + c = 0 → t = -c / b (only valid if b != 0)
        if abs(b) < 1e-12:
            return None
        t = -c / b
        return t if t > 0 else None
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return None
    sqrt_disc = math.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)
    positives = [t for t in (t1, t2) if t > 1e-9]
    if not positives:
        return None
    return min(positives)


def _earliest_positive_root(a: float, b: float, c: float) -> float | None:
    """Smallest positive root, or ``None`` if no positive real root."""
    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            return None
        t = -c / b
        return t if t > 0 else None
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return None
    sqrt_disc = math.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)
    positives = [t for t in (t1, t2) if t > 1e-9]
    if not positives:
        return None
    return min(positives)
