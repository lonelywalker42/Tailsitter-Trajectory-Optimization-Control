"""Trajectory optimization for tailsitter UAV transition flight using CasADi + IPOPT.

Refactored from matlab/trajectory/ (GPOPS-II) to Python using CasADi's direct
collocation with Radau points and the nlpsol interface.

State vector (11D):
    [u, w, theta, q, h, as, vs, xs, dt, de, us]
     ──plant 5D──  ──mass 3D─  ──augmented ctrl 3D─

Control vector (3D):
    [dtdot, dedot, usdot]  (rates of augmented controls)
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import Optional

import casadi as cs
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryOptConfig:
    """All parameters for the trajectory optimization problem."""

    # Time bounds
    t0: float = 0.0
    tf_min: float = 8.0
    tf_max: float = 20.0

    # State bounds  [u, w, theta, q, h, as, vs, xs, dt, de, us]
    x_min: np.ndarray = field(default_factory=lambda: np.array(
        [-10, -10, -2*np.pi/3, -10*np.pi, -100, -0.1, -0.03, -0.4, 0, -30, -0.01],
        dtype=np.float64))
    x_max: np.ndarray = field(default_factory=lambda: np.array(
        [30, 30, 2*np.pi/3, 10*np.pi, 100, 0.1, 0.03, 0, 1, 20, 0.01],
        dtype=np.float64))

    # Control bounds  [dtdot, dedot, usdot]
    u_min: np.ndarray = field(default_factory=lambda: np.array(
        [-0.1, -5, -0.1], dtype=np.float64))
    u_max: np.ndarray = field(default_factory=lambda: np.array(
        [0.1, 5, 0.1], dtype=np.float64))

    # Path constraints
    h_min: float = -20.0
    h_max: float = 50.0
    alpha_min: float = -60.0 * np.pi / 180  # rad
    alpha_max: float = 120.0 * np.pi / 180  # rad
    theta_max: float = 120.0 * np.pi / 180  # rad (allow full transition corridor)

    # Boundary conditions — forward flight
    x0_forward: np.ndarray = field(default_factory=lambda: np.array(
        [19.9264, 1.7147, 4.9184*np.pi/180, 0, 0, 0, 0, 0, 0.0572, 2.6443, 0],
        dtype=np.float64))
    # Boundary conditions — hover
    xf_hover: np.ndarray = field(default_factory=lambda: np.array(
        [0.01, 0, np.pi/2, 0, 0, 0, 0, 0, 0.6339, 0, 0],
        dtype=np.float64))

    # Initial guess final state (for forward2hover)
    xfg_forward2hover: np.ndarray = field(default_factory=lambda: np.array(
        [0.01, 0, np.pi/2, 5, 0, 0, 0, 0, 0.6339, 0, 0],
        dtype=np.float64))
    ug0_forward2hover: np.ndarray = field(default_factory=lambda: np.array(
        [0.1, 5, 0], dtype=np.float64))
    ugf_forward2hover: np.ndarray = field(default_factory=lambda: np.array(
        [0, 0, 0], dtype=np.float64))

    # Objective weights (normalized to O(1) for better conditioning)
    w_hdot: float = 0.001
    w_q: float = 0.01
    w_dtdot: float = 0.1
    w_dt: float = 0.1
    w_dus: float = 0.1

    # Boundary penalty weight (balanced with scaled collocation constraints)
    w_bc: float = 100.0

    # Two-phase solve: Phase 1 boundary relaxation (radians for angles, m/s for velocities)
    bc_relax_angle: float = 0.1   # ~5.7° relaxation for theta/alpha in Phase 1
    bc_relax_vel: float = 0.5     # m/s relaxation for u/w in Phase 1

    # Solver settings
    num_nodes: int = 200
    ipopt_max_iter: int = 5000
    ipopt_tol: float = 1e-6
    ipopt_print_level: int = 5

    # IPOPT algorithm tuning
    ipopt_mu_strategy: str = 'adaptive'       # 'adaptive' or 'monotone'
    ipopt_hessian_approximation: str = 'exact'  # 'exact' or 'limited-memory'
    ipopt_warm_start: bool = False

    # Collocation
    collocation_degree: int = 2  # 1=backward Euler, 2 or 3 for higher accuracy

    # Direction
    direction: str = "forward2hover"  # or "hover2forward"

    # Guess transition time
    tf_guess: float = 8.0

    # Trim solver
    auto_trim: bool = True  # compute trim points from optimizer dynamics
    trim_V_forward: float = 20.0  # forward flight trim velocity (m/s)
    trim_V_hover: float = 0.01   # hover trim velocity (m/s)
    trim_tol: float = 1e-12      # trim solver tolerance
    bc_tol: float = 0.5          # boundary condition tolerance (tight inequality)


# ---------------------------------------------------------------------------
# Aero / propulsion data (cfg1, matching matlab/trajectory/LonDyn.m)
# ---------------------------------------------------------------------------

_AERO_ALPHA_DEG = np.array([
    -180, -160, -140, -120, -100, -80, -70, -60, -50, -40,
    -30, -20, -10, -5, 0, 4, 8, 12, 14, 16,
    20, 30, 40, 50, 60, 70, 80, 90, 100, 120,
    140, 160, 180,
], dtype=np.float64)

_AERO_CL = np.array([
    0.000475662198815010, 0.709285645819618, 0.735086430546412,
    0.587723059907834, 0.164793197630020, -0.329666043449638,
    -0.597065838051350, -0.801544089532587, -0.981306754443713,
    -1.12697545227123, -1.08386713364055, -0.922612687294273,
    -0.721641774851876, -0.435033216589862, -0.0300427186306781,
    0.313337680052666, 0.651036776826860, 0.941906938775510,
    0.977917177090191, 0.972528516129032, 1.01325642659645,
    1.04839872547729, 1.08809458854510, 0.964799304805793,
    0.782445861751152, 0.579773724818960, 0.298734936142199,
    0.0366808921658986, -0.211116808426596, -0.678720021066491,
    -0.858023436471363, -0.806466759710336, 0.000214409690585912,
], dtype=np.float64)

_AERO_CD = np.array([
    0.0299568410796577, 0.311980429229756, 0.739374114549045,
    1.22118981435155, 1.40299297959184, 1.51735613429888,
    1.44803604476629, 1.28269286635945, 1.12187175773535,
    0.912802843976300, 0.639478751810402, 0.367034017116524,
    0.133343521790652, 0.0367489453587887, 0.0201523492824226,
    0.0261665495720869, 0.0482200710994075, 0.0919206298880843,
    0.142164034233048, 0.206602170901909, 0.348211486504279,
    0.603590635944700, 0.875435144173799, 1.10218118235681,
    1.27392229624753, 1.46380802106649, 1.50339844634628,
    1.48863279526004, 1.44759914680711, 1.29295816721527,
    0.794106564845293, 0.327687752468729, 0.0299682675444371,
], dtype=np.float64)

_AERO_CM = np.array([
    0.0214595741680647, 0.512943819575697, 0.496447995922615,
    0.438300538979380, 0.273583907919047, 0.0800134961455966,
    -0.142352525642931, -0.219460728407908, -0.236010156087409,
    -0.258126193378496, -0.134418864703009, -0.0435029775956168,
    0.0133756704113487, 0.0238700408163265, 0.0310153963770732,
    0.0285869967083608, 0.0238101297542951, 0.0246721291172036,
    0.0602906876765274, 0.0504827464588333, 0.0383370438106564,
    0.172037032852683, 0.282704359829259, 0.286848016309542,
    0.262249653209879, 0.177587108666568, -0.0321408872560471,
    -0.162238749601818, -0.244938603920236, -0.446472285247085,
    -0.525138443373187, -0.511236020301981, 0.0214607783558793,
], dtype=np.float64)

_ELEV_DEG = np.array(
    [-20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30], dtype=np.float64)

_ELEV_DCL = np.array([
    0.141503298953990, 0.103155701850633, 0.0766277229171238,
    0.0506382561626801, 0, -0.0964885085216885, -0.153361012361934,
    -0.216410679540633, -0.263103386731036, -0.337795801331285,
    -0.385546514519787,
], dtype=np.float64)

_ELEV_DCD = np.array([
    0.00915394630970665, 0.00309601345914707, -0.00174006290688322,
    -0.00604434203789046, 0, 0.00243037085802060, 0.0113742666959257,
    0.0202924146002486, 0.0305939287542974, 0.0430629507717065,
    0.0539121936946823,
], dtype=np.float64)

_ELEV_DCM = np.array([
    -0.0452428380301132, -0.0351559812270382, -0.0273545838730914,
    -0.0186967656989955, 0, 0.0179587589458260, 0.0278710526874642,
    0.0384488946463081, 0.0461874323090318, 0.0578716897789293,
    0.0643487863407590,
], dtype=np.float64)

_THROTTLE = np.array(
    [0, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0],
    dtype=np.float64)

_THRUST = np.array([
    0, 1.253, 1.527, 1.983, 2.469, 2.935, 3.400,
    3.933, 4.476, 5.042, 6.182, 7.053,
], dtype=np.float64)


# ---------------------------------------------------------------------------
# UAV physical parameters (matching LonDyn.m)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _UAVParams:
    S: float = 0.62
    c: float = 0.31
    m: float = 6.5
    ms: float = 0.2
    prop_R: float = 0.2032
    rho: float = 1.225
    g: float = 9.8
    Jy: float = 0.183

    @property
    def mu(self) -> float:
        return self.ms / self.m


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryResult:
    """Result of a trajectory optimization solve."""
    time: np.ndarray          # (N+1,)
    state: np.ndarray         # (N+1, 11)
    control: np.ndarray       # (N, 3)
    cost: float
    solve_time: float
    success: bool
    tf: float


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class TailsitterTrajectoryOptimizer:
    """CasADi direct-collocation trajectory optimizer for tailsitter UAV.

    Solves the same optimal control problem as matlab/trajectory/TraOptMain.m
    using CasADi nlpsol + IPOPT with Radau collocation.
    """

    def __init__(self, config: Optional[TrajectoryOptConfig] = None):
        self.cfg = config or TrajectoryOptConfig()
        self._params = _UAVParams()
        self._solver = None

        # Build CasADi interpolants for aero / propulsion tables
        self._build_interpolants()

    # ----- interpolants -----

    def _build_interpolants(self) -> None:
        alpha_rad = _AERO_ALPHA_DEG * np.pi / 180.0

        self._cL_interp = cs.interpolant(
            'cL', 'linear', [alpha_rad.tolist()], _AERO_CL.tolist())
        self._cD_interp = cs.interpolant(
            'cD', 'linear', [alpha_rad.tolist()], _AERO_CD.tolist())
        self._cm_interp = cs.interpolant(
            'cm', 'linear', [alpha_rad.tolist()], _AERO_CM.tolist())
        self._dcL_interp = cs.interpolant(
            'dcL', 'linear', [_ELEV_DEG.tolist()], _ELEV_DCL.tolist())
        self._dcD_interp = cs.interpolant(
            'dcD', 'linear', [_ELEV_DEG.tolist()], _ELEV_DCD.tolist())
        self._dcm_interp = cs.interpolant(
            'dcm', 'linear', [_ELEV_DEG.tolist()], _ELEV_DCM.tolist())
        self._thrust_interp = cs.interpolant(
            'thrust', 'linear', [_THROTTLE.tolist()], _THRUST.tolist())

    # ----- trim solver -----

    def _find_trim(self, V_target: float,
                   target_gamma: Optional[float] = None,
                   target_theta: Optional[float] = None) -> np.ndarray:
        """Find trim point at given airspeed using the optimizer's dynamics.

        Solves for [alpha, theta, dt, de] such that udot=wdot=qdot=0
        with V = sqrt(u^2 + w^2) = V_target, q=0, h=0, moving mass=0.

        Args:
            V_target: Target airspeed in m/s.
            target_gamma: If set, penalize deviation of flight path angle
                gamma = theta - alpha from this value (rad). Use 0 for level flight.
            target_theta: If set, penalize deviation of theta from this value (rad).
                Use pi/2 for hover.

        Returns:
            11D state vector at trim.
        """
        alpha_v = cs.MX.sym('alpha')
        theta_v = cs.MX.sym('theta')
        dt_v = cs.MX.sym('dt')
        de_v = cs.MX.sym('de')

        u_vel = V_target * cs.cos(alpha_v)
        w_vel = V_target * cs.sin(alpha_v)
        x_trim = cs.vertcat(u_vel, w_vel, theta_v, 0, 0, 0, 0, 0,
                            dt_v, de_v, 0)
        xdot_trim = self._ode(x_trim, cs.DM.zeros(3))

        # Cost: trim condition (udot=wdot=qdot=0) + flight condition penalty
        cost = xdot_trim[0]**2 + xdot_trim[1]**2 + xdot_trim[3]**2

        w_cond = 10.0  # moderate weight — push toward desired condition
        if target_gamma is not None:
            gamma_err = (theta_v - alpha_v) - target_gamma
            cost += w_cond * gamma_err**2
        if target_theta is not None:
            theta_err = theta_v - target_theta
            cost += w_cond * theta_err**2

        nlp = {
            'x': cs.vertcat(alpha_v, theta_v, dt_v, de_v),
            'f': cost,
        }
        solver = cs.nlpsol(
            'trim_solver', 'ipopt', nlp,
            {'ipopt.print_level': 0,
             'ipopt.tol': self.cfg.trim_tol,
             'ipopt.max_iter': 500})

        best_cost = 1e10
        best_x = None

        # Grid search over initial guesses — tailored to flight condition
        if target_theta is not None:
            # Hover: theta near target, alpha can vary
            alpha_guesses = [0, 5, -5, 10, -10, 15, -15]
            theta_guesses = [np.degrees(target_theta) + d
                             for d in [0, 5, -5, 10, -10]]
        elif target_gamma is not None:
            # Level flight: theta ≈ alpha + target_gamma
            alpha_guesses = [5, 10, -5, -10, 0, 15, 20]
            theta_guesses = [a + np.degrees(target_gamma)
                             for a in [5, 10, -5, -10, 0, 15, 20]]
        else:
            alpha_guesses = [5, 10, -5, -10, 0, 20, 30, 45, 60]
            theta_guesses = [5, 10, 30, 45, 60, 80, 90]

        dt_guesses = [0.05, 0.1, 0.3, 0.5]
        de_guesses = [0, 2, 5, -5]

        for a0 in alpha_guesses:
            for th0 in theta_guesses:
                for dt0 in dt_guesses:
                    for de0 in de_guesses:
                        try:
                            sol = solver(
                                x0=[a0 * np.pi / 180, th0 * np.pi / 180,
                                    dt0, de0],
                                lbx=[-60 * np.pi / 180, 0, 0, -30],
                                ubx=[90 * np.pi / 180, 120 * np.pi / 180,
                                     1, 20])
                            cost = float(sol['f'])
                            if cost < best_cost:
                                best_cost = cost
                                best_x = np.array(sol['x']).flatten()
                        except RuntimeError:
                            pass

        if best_x is None or best_cost > 10:
            raise RuntimeError(
                f"Trim solver failed at V={V_target} m/s "
                f"(best cost={best_cost:.2e}). "
                f"Check if a trim point exists in this dynamics model.")

        alpha_opt, theta_opt, dt_opt, de_opt = best_x
        u_opt = V_target * np.cos(alpha_opt)
        w_opt = V_target * np.sin(alpha_opt)

        return np.array([
            u_opt, w_opt, theta_opt, 0.0, 0.0,
            0.0, 0.0, 0.0,
            dt_opt, de_opt, 0.0
        ], dtype=np.float64)

    # ----- collocation coefficients -----

    @staticmethod
    def _collocation_coefficients(d: int):
        """Compute Radau collocation coefficient matrices C, D, B.

        Args:
            d: Degree of interpolating polynomial.

        Returns:
            C: (d+1, d+1) collocation equation coefficients.
            D: (d+1,) continuity equation coefficients.
            B: (d+1,) quadrature coefficients.
        """
        tau_root = [0] + list(cs.collocation_points(d, 'radau'))

        C = np.zeros((d + 1, d + 1))
        D = np.zeros(d + 1)
        B = np.zeros(d + 1)

        for j in range(d + 1):
            # Construct Lagrange polynomial
            p = np.poly1d([1])
            for r in range(d + 1):
                if r != j:
                    p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j] - tau_root[r])

            # Continuity: polynomial evaluated at tau=1
            D[j] = float(p(1.0))

            # Collocation: derivative at all collocation points
            pder = np.polyder(p)
            for r in range(d + 1):
                C[j, r] = float(pder(tau_root[r]))

            # Quadrature: integral from 0 to 1
            pint = np.polyint(p)
            B[j] = float(pint(1.0))

        return C, D, B

    # ----- dynamics -----

    def _ode(self, x: cs.MX, u_ctrl: cs.MX) -> cs.MX:
        """Longitudinal dynamics with augmented controls.

        Args:
            x: 11D state [u, w, theta, q, h, as, vs, xs, dt, de, us]
            u_ctrl: 3D control rates [dtdot, dedot, usdot]

        Returns:
            11D derivative
        """
        p = self._params

        # Unpack plant state
        u_vel = x[0]
        w_vel = x[1]
        theta = x[2]
        q = x[3]

        # Unpack augmented controls
        dt = x[8]
        de = x[9]
        us = x[10]

        # Airspeed (smooth)
        V0_sq = u_vel**2 + w_vel**2
        V0 = cs.sqrt(V0_sq + 1e-12)

        # Angle of attack
        alpha0 = cs.atan2(w_vel, u_vel + 1e-12)

        # Thrust
        Fprop = p.g * self._thrust_interp(dt)

        # Induced velocity
        v_ind = cs.sqrt(cs.fmax(Fprop, 0)
                        / (2 * p.rho * cs.pi * p.prop_R**2) + 1e-12)

        # Dynamic pressure
        q0 = 0.5 * p.rho * V0_sq
        q_ind = 0.5 * p.rho * (v_ind + V0 * cs.cos(alpha0))**2

        # Aero coefficients with elevator increments
        # Elevator sign inversion: MATLAB uses de=-u(2), so lookup at -de
        cL = self._cL_interp(alpha0) + self._dcL_interp(-de)
        cD = self._cD_interp(alpha0) + self._dcD_interp(-de)
        cm = self._cm_interp(alpha0)
        dcm = self._dcm_interp(-de)

        # Forces and moment
        L = q0 * p.S * cL
        D = q0 * p.S * cD
        my = q0 * p.S * p.c * cm + q_ind * p.S * p.c * dcm

        sin_a = cs.sin(alpha0)
        cos_a = cs.cos(alpha0)
        sin_t = cs.sin(theta)
        cos_t = cs.cos(theta)

        # Body-frame resultant forces
        fx = -sin_a * L + cos_a * D + 2 * Fprop - p.m * p.g * sin_t
        fz = cos_a * L + sin_a * D + p.m * p.g * cos_t

        # Equations of motion (simplified: no moving mass coupling)
        qdot = my / p.Jy
        udot = fx / p.m
        wdot = fz / p.m
        thetadot = q
        hdot = w_vel * cos_t - u_vel * sin_t

        # Moving mass states (zero dynamics - kept for state dimension)
        asdot = 0
        vsdot = 0
        xsdot = 0

        return cs.vertcat(udot, wdot, thetadot, qdot, hdot,
                          asdot, vsdot, xsdot,
                          u_ctrl[0], u_ctrl[1], u_ctrl[2])

    def _stage_cost(self, x: cs.MX, u_ctrl: cs.MX) -> cs.SX:
        """Stage cost L(x, u) for the integral objective."""
        cfg = self.cfg

        u_vel = x[0]
        w_vel = x[1]
        theta = x[2]
        q = x[3]
        dt = x[8]
        us = x[10]

        # Compute hdot
        cos_t = cs.cos(theta)
        sin_t = cs.sin(theta)
        hdot = w_vel * cos_t - u_vel * sin_t

        return (cfg.w_hdot * hdot**2
                + cfg.w_q * q**2
                + cfg.w_dtdot * u_ctrl[0]**2
                + cfg.w_dt * dt**2
                + cfg.w_dus * u_ctrl[2]**2)

    # ----- problem construction -----

    def build(self) -> None:
        """Build the CasADi NLP with Radau direct collocation."""
        cfg = self.cfg
        N = cfg.num_nodes
        d = cfg.collocation_degree
        nx = 11
        nu = 3

        # Collocation coefficients
        C, D, B = self._collocation_coefficients(d)

        # Decision variables
        w = []           # stacked variable vector
        lbw = []         # lower bounds
        ubw = []         # upper bounds
        w0 = []          # initial guess

        # Constraint expressions and bounds
        g = []
        lbg = []
        ubg = []

        # Time variable
        Tf = cs.MX.sym('Tf')
        w.append(Tf)
        lbw.append(cfg.tf_min)
        ubw.append(cfg.tf_max)
        w0.append(cfg.tf_guess)

        # State at mesh nodes: X[k] for k = 0..N
        X_syms = []
        for k in range(N + 1):
            Xk = cs.MX.sym(f'X_{k}', nx)
            X_syms.append(Xk)
            w.append(Xk)
            lbw.extend(cfg.x_min.tolist())
            ubw.extend(cfg.x_max.tolist())

        # Control at mesh intervals: U[k] for k = 0..N-1
        U_syms = []
        for k in range(N):
            Uk = cs.MX.sym(f'U_{k}', nu)
            U_syms.append(Uk)
            w.append(Uk)
            lbw.extend(cfg.u_min.tolist())
            ubw.extend(cfg.u_max.tolist())

        # Time step
        h = Tf / N

        # Objective (integral cost)
        J = 0

        # Boundary condition targets as fixed decision variables
        x0_target = cs.MX.sym('x0_target', nx)
        xf_target = cs.MX.sym('xf_target', nx)
        w.append(x0_target)
        lbw.extend([0.0] * nx)  # placeholder
        ubw.extend([0.0] * nx)  # placeholder
        w.append(xf_target)
        lbw.extend([0.0] * nx)  # placeholder
        ubw.extend([0.0] * nx)  # placeholder

        # Store stage cost before penalty (for continuation in solve())
        self._J_stage = J
        self._x0_target_sym = x0_target
        self._xf_target_sym = xf_target

        # Boundary penalty (soft constraints)
        w_bc = cfg.w_bc
        J += w_bc * cs.sumsqr(X_syms[0] - x0_target)
        J += w_bc * cs.sumsqr(X_syms[N] - xf_target)

        self._x0_target_idx = 1 + (N + 1) * nx + N * nu  # index in w vector
        self._xf_target_idx = self._x0_target_idx + nx

        # Path constraint on altitude at all mesh nodes
        for k in range(N + 1):
            g.append(X_syms[k][4])
            lbg.append(cfg.h_min)
            ubg.append(cfg.h_max)

        # Path constraint on alpha at all mesh nodes
        # alpha = atan2(w, u), computed from plant states
        for k in range(N + 1):
            u_k = X_syms[k][0]
            w_k = X_syms[k][1]
            alpha_k = cs.atan2(w_k, u_k + 1e-12)
            g.append(alpha_k)
            lbg.append(cfg.alpha_min)
            ubg.append(cfg.alpha_max)

        # Path constraint on theta upper bound
        for k in range(N + 1):
            g.append(X_syms[k][2])
            lbg.append(-np.pi)  # no lower bound beyond state bounds
            ubg.append(cfg.theta_max)

        # Collocation defect constraints
        # Scale by 1/h so constraints measure velocity error (~O(1))
        # instead of position error (~O(h)), improving NLP conditioning
        h_inv = N / Tf  # 1/h

        if d == 1:
            # Backward Euler: f evaluated at X[k+1]
            for k in range(N):
                Xk = X_syms[k]
                Uk = U_syms[k]
                Xk1 = X_syms[k + 1]

                fk1 = self._ode(Xk1, Uk)

                # Scaled: (Xk1 - Xk)/h - fk1 = 0  (velocity error form)
                g.append(h_inv * (Xk1 - Xk) - fk1)
                lbg.extend([0] * nx)
                ubg.extend([0] * nx)

                Lk = self._stage_cost(Xk, Uk)
                J += h * Lk
        else:
            # Higher-order Radau collocation (d >= 2)
            # Interior collocation states per interval
            Z_syms = []
            tau_root = [0] + list(cs.collocation_points(d, 'radau'))
            for k in range(N):
                Zk = []
                for j in range(1, d + 1):
                    Zjk = cs.MX.sym(f'Z_{k}_{j}', nx)
                    Zk.append(Zjk)
                    w.append(Zjk)
                    lbw.extend(cfg.x_min.tolist())
                    ubw.extend(cfg.x_max.tolist())
                Z_syms.append(Zk)
            # Store tau_root for initial guess computation below
            self._tau_root = tau_root

            for k in range(N):
                Xk = X_syms[k]
                Uk = U_syms[k]
                Xk1 = X_syms[k + 1]
                Zk = Z_syms[k]  # d interior states

                # Evaluate dynamics at all collocation points
                fk_all = [self._ode(Zk[j], Uk) for j in range(d)]

                # Collocation equations (scaled by 1/h):
                # (1/h) * [sum_j C[j,r] * Z[j] + D[r] * X[k+1]] - f(Z[r]) = 0
                for r in range(1, d + 1):
                    colloc_expr = C[0, r] * Xk
                    for j in range(1, d + 1):
                        colloc_expr += C[j, r] * Zk[j - 1]
                    colloc_expr += D[r] * Xk1
                    # Scale by 1/h for better conditioning
                    colloc_expr = h_inv * colloc_expr - fk_all[r - 1]

                    g.append(colloc_expr)
                    lbg.extend([0] * nx)
                    ubg.extend([0] * nx)

                # Continuity: X[k+1] = sum_j D[j] * [X[k], Z[1], ..., Z[d]]
                continuity = D[0] * Xk
                for j in range(1, d + 1):
                    continuity += D[j] * Zk[j - 1]
                g.append(Xk1 - continuity)
                lbg.extend([0] * nx)
                ubg.extend([0] * nx)

                # Quadrature for stage cost
                J += h * B[0] * self._stage_cost(Xk, Uk)
                for j in range(1, d + 1):
                    J += h * B[j] * self._stage_cost(Zk[j - 1], Uk)

        # Initial guess for states: S-curve interpolation
        if cfg.auto_trim:
            # Forward trim: no gamma constraint (induced-velocity model has no
            # level-flight trim at V=20; the only equilibria are climbing flight)
            x0_trim = self._find_trim(cfg.trim_V_forward)
            # Hover trim: enforce theta ≈ pi/2
            xf_trim = self._find_trim(cfg.trim_V_hover, target_theta=np.pi/2)
            if cfg.direction == "forward2hover":
                x0_guess = x0_trim.copy()
                xf_guess = xf_trim.copy()
            else:
                x0_guess = xf_trim.copy()
                xf_guess = x0_trim.copy()
            # Use trimmed control values for guess
            ug0 = np.array([0.0, 0.0, 0.0])
            ugf = np.array([0.0, 0.0, 0.0])
        elif cfg.direction == "forward2hover":
            x0_guess = cfg.x0_forward.copy()
            xf_guess = cfg.xfg_forward2hover.copy()
            ug0 = cfg.ug0_forward2hover
            ugf = cfg.ugf_forward2hover
        else:
            x0_guess = cfg.xf_hover.copy()
            xf_guess = cfg.x0_forward.copy()
            ug0 = cfg.ugf_forward2hover
            ugf = cfg.ug0_forward2hover

        tf_g = cfg.tf_guess
        for k in range(N + 1):
            s = k / N
            # Linear base for all states
            xk = x0_guess + s * (xf_guess - x0_guess)
            # S-curve (cubic Hermite) for theta: smooth pitch-over
            xk[2] = x0_guess[2] + (xf_guess[2] - x0_guess[2]) * (3*s**2 - 2*s**3)
            # Parabolic pitch rate: peaks at mid-transition
            xk[3] = 6 * (xf_guess[2] - x0_guess[2]) / tf_g * s * (1 - s)
            # Smooth throttle ramp (linear is fine, but clamp to [0,1])
            xk[8] = np.clip(x0_guess[8] + (xf_guess[8] - x0_guess[8]) * s, 0, 1)
            w0.extend(xk.tolist())

        for k in range(N):
            s = k / N
            w0.extend((ug0 + s * (ugf - ug0)).tolist())

        # Initial guess for boundary target variables
        w0.extend(x0_guess.tolist())
        w0.extend(xf_guess.tolist())

        # Initial guess for interior collocation states (d >= 2)
        if d >= 2:
            tau_root = self._tau_root
            for k in range(N):
                s_k = k / N
                s_k1 = (k + 1) / N
                for j in range(1, d + 1):
                    s_j = s_k + tau_root[j] * (s_k1 - s_k)
                    zj = x0_guess + s_j * (xf_guess - x0_guess)
                    zj[2] = x0_guess[2] + (xf_guess[2] - x0_guess[2]) * (3*s_j**2 - 2*s_j**3)
                    zj[3] = 6 * (xf_guess[2] - x0_guess[2]) / tf_g * s_j * (1 - s_j)
                    zj[8] = np.clip(x0_guess[8] + (xf_guess[8] - x0_guess[8]) * s_j, 0, 1)
                    w0.extend(zj.tolist())

        # Build NLP
        w_cat = cs.vertcat(*w)
        g_cat = cs.vertcat(*g)

        # Small time penalty to prevent tf from growing unbounded
        # Keep very small so it doesn't dominate the stage cost
        w_tf = 0.01
        nlp = {'x': w_cat, 'f': J + w_tf * Tf, 'g': g_cat}

        opts = {
            'ipopt.max_iter': cfg.ipopt_max_iter,
            'ipopt.tol': cfg.ipopt_tol,
            'ipopt.print_level': cfg.ipopt_print_level,
            'ipopt.mu_strategy': cfg.ipopt_mu_strategy,
            'ipopt.hessian_approximation': cfg.ipopt_hessian_approximation,
            'ipopt.linear_solver': 'mumps',
            'ipopt.nlp_scaling_method': 'gradient-based',
            'ipopt.nlp_scaling_max_gradient': 100,
            'ipopt.mu_init': 0.1,
            'ipopt.fixed_variable_treatment': 'make_parameter',
            'ipopt.acceptable_tol': 1e-3,
            'ipopt.acceptable_iter': 20,
            'ipopt.bound_push': 1e-3,
            'ipopt.bound_frac': 1e-3,
            # MUMPS pivoting for near-singular KKT systems
            'ipopt.mumps_pivtol': 1e-4,
            'ipopt.mumps_pivtolmax': 0.1,
            # Better multiplier initialization
            'ipopt.least_square_init_primal': 'yes',
        }

        if cfg.ipopt_warm_start:
            opts['ipopt.warm_start_init_point'] = 'yes'
            opts['ipopt.warm_start_bound_push'] = 1e-6
            opts['ipopt.warm_start_mult_bound_push'] = 1e-6

        self._solver = cs.nlpsol('solver', 'ipopt', nlp, opts)

        # Store for solve()
        self._w = w_cat
        self._g = g_cat
        self._J = J + w_tf * Tf
        self._X_syms = X_syms
        self._U_syms = U_syms
        self._Tf_sym = Tf
        self._lbw = np.array(lbw, dtype=np.float64)
        self._ubw = np.array(ubw, dtype=np.float64)
        self._lbg = np.array(lbg, dtype=np.float64)
        self._ubg = np.array(ubg, dtype=np.float64)
        self._w0 = np.array(w0, dtype=np.float64)

    # ----- solve -----

    def solve(self) -> TrajectoryResult:
        """Solve the trajectory optimization problem.

        Uses penalty-based boundary conditions with high weight to enforce
        boundary states without creating infeasibility.

        Must call build() first.
        """
        if self._solver is None:
            raise RuntimeError("Call build() before solve().")

        cfg = self.cfg
        N = cfg.num_nodes
        nx = 11
        nu = 3

        # Compute boundary targets
        if cfg.auto_trim:
            x0_val = self._find_trim(cfg.trim_V_forward)
            xf_val = self._find_trim(cfg.trim_V_hover, target_theta=np.pi/2)
            if cfg.direction == "hover2forward":
                x0_val, xf_val = xf_val, x0_val
        elif cfg.direction == "forward2hover":
            x0_val = cfg.x0_forward
            xf_val = cfg.xf_hover
        else:
            x0_val = cfg.xf_hover
            xf_val = cfg.x0_forward

        idx0 = self._x0_target_idx
        idxf = self._xf_target_idx

        # Fix boundary target variables (these are penalty targets, not state bounds)
        lbw = self._lbw.copy()
        ubw = self._ubw.copy()
        lbw[idx0:idx0+nx] = x0_val
        ubw[idx0:idx0+nx] = x0_val
        lbw[idxf:idxf+nx] = xf_val
        ubw[idxf:idxf+nx] = xf_val

        lbg = self._lbg.copy()
        ubg = self._ubg.copy()

        total_start = _time.time()

        # Continuation on w_bc: solve with increasing penalty weights
        w_bc_schedule = [100, 1000, 10000, 100000, 1e6]
        w_cur = self._w0.copy()

        for w_bc_val in w_bc_schedule:
            # Build NLP with this w_bc
            J_cont = self._J_stage + 0.01 * self._Tf_sym
            J_cont += w_bc_val * cs.sumsqr(self._X_syms[0] - self._x0_target_sym)
            J_cont += w_bc_val * cs.sumsqr(self._X_syms[N] - self._xf_target_sym)
            nlp_cont = {'x': self._w, 'f': J_cont, 'g': self._g}

            opts_cont = {
                'ipopt.max_iter': 1000,
                'ipopt.tol': cfg.ipopt_tol,
                'ipopt.print_level': 0,
                'ipopt.mu_strategy': cfg.ipopt_mu_strategy,
                'ipopt.hessian_approximation': cfg.ipopt_hessian_approximation,
                'ipopt.linear_solver': 'mumps',
                'ipopt.nlp_scaling_method': 'gradient-based',
                'ipopt.nlp_scaling_max_gradient': 100,
                'ipopt.mu_init': 0.1,
                'ipopt.fixed_variable_treatment': 'make_parameter',
                'ipopt.acceptable_tol': 0.01,
                'ipopt.acceptable_iter': 10,
                'ipopt.bound_push': 1e-6,
                'ipopt.bound_frac': 1e-6,
                'ipopt.mumps_pivtol': 1e-4,
                'ipopt.mumps_pivtolmax': 0.1,
                'ipopt.warm_start_init_point': 'yes',
                'ipopt.warm_start_bound_push': 1e-8,
                'ipopt.warm_start_mult_bound_push': 1e-8,
                'ipopt.least_square_init_primal': 'yes',
            }
            solver_cont = cs.nlpsol(f'solver_bc_{int(w_bc_val)}', 'ipopt', nlp_cont, opts_cont)
            sol_cont = solver_cont(x0=w_cur, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
            w_cur = np.array(sol_cont['x']).flatten()
            s_cont = solver_cont.stats()

            x0_err = np.degrees(np.abs(w_cur[1+2] - x0_val[2]))
            xf_err = np.degrees(np.abs(w_cur[1+N*nx+2] - xf_val[2]))
            print(f"  w_bc={w_bc_val:.0e}: {s_cont['return_status']}, "
                  f"iters={s_cont['iter_count']}, x0_err={x0_err:.1f}°, xf_err={xf_err:.1f}°")

            if s_cont['return_status'] not in ['Solve_Succeeded', 'Solved_To_Acceptable_Level']:
                break

        w_sol = w_cur
        s_final = s_cont
        solve_time = _time.time() - total_start

        tf_sol = float(w_sol[0])
        cost_sol = float(sol_cont['f'])

        # Parse state trajectory
        idx = 1  # skip Tf
        X_sol = np.zeros((N + 1, nx))
        for k in range(N + 1):
            X_sol[k, :] = w_sol[idx:idx + nx]
            idx += nx

        # Parse control trajectory
        U_sol = np.zeros((N, nu))
        for k in range(N):
            U_sol[k, :] = w_sol[idx:idx + nu]
            idx += nu

        time_vec = np.linspace(0, tf_sol, N + 1)

        # Check solver status
        success = s_final['return_status'] in ['Solve_Succeeded', 'Solved_To_Acceptable_Level']

        # Report boundary errors
        x0_err = X_sol[0] - x0_val
        xf_err = X_sol[N] - xf_val
        print(f"  x0 theta err: {np.degrees(x0_err[2]):.2f}°, "
              f"xf theta err: {np.degrees(xf_err[2]):.2f}°")

        return TrajectoryResult(
            time=time_vec,
            state=X_sol,
            control=U_sol,
            cost=cost_sol,
            solve_time=solve_time,
            success=success,
            tf=tf_sol,
        )

    def solve_from_guess(self, w0: np.ndarray) -> TrajectoryResult:
        """Solve using an external initial guess (e.g., from a coarse solve).

        Args:
            w0: Initial guess vector matching the decision variable layout.
                 Layout: [Tf, X_0(11), X_1(11), ..., X_N(11), U_0(3), ..., U_{N-1}(3)]

        Returns:
            TrajectoryResult from the fine solve.
        """
        if self._solver is None:
            raise RuntimeError("Call build() before solve_from_guess().")

        cfg = self.cfg
        N = cfg.num_nodes
        nx = 11
        nu = 3

        # Set boundary condition target values
        lbw = self._lbw.copy()
        ubw = self._ubw.copy()
        lbg = self._lbg.copy()
        ubg = self._ubg.copy()

        if cfg.auto_trim:
            x0_val = self._find_trim(cfg.trim_V_forward)
            xf_val = self._find_trim(cfg.trim_V_hover, target_theta=np.pi/2)
            if cfg.direction == "hover2forward":
                x0_val, xf_val = xf_val, x0_val
        elif cfg.direction == "forward2hover":
            x0_val = cfg.x0_forward
            xf_val = cfg.xf_hover
        else:
            x0_val = cfg.xf_hover
            xf_val = cfg.x0_forward

        # Fix boundary targets (lb=ub=value)
        idx0 = self._x0_target_idx
        idxf = self._xf_target_idx
        lbw[idx0:idx0+nx] = x0_val
        ubw[idx0:idx0+nx] = x0_val
        lbw[idxf:idxf+nx] = xf_val
        ubw[idxf:idxf+nx] = xf_val

        t_start = _time.time()
        sol = self._solver(x0=w0, lbx=lbw, ubx=ubw,
                           lbg=lbg, ubg=ubg)
        solve_time = _time.time() - t_start

        w_sol = np.array(sol['x']).flatten()
        tf_sol = float(w_sol[0])
        cost_sol = float(sol['f'])

        idx = 1
        X_sol = np.zeros((N + 1, nx))
        for k in range(N + 1):
            X_sol[k, :] = w_sol[idx:idx + nx]
            idx += nx

        U_sol = np.zeros((N, nu))
        for k in range(N):
            U_sol[k, :] = w_sol[idx:idx + nu]
            idx += nu

        time_vec = np.linspace(0, tf_sol, N + 1)

        status = self._solver.stats()
        success = status['return_status'] in ['Solve_Succeeded', 'Solved_To_Acceptable_Level']

        return TrajectoryResult(
            time=time_vec,
            state=X_sol,
            control=U_sol,
            cost=cost_sol,
            solve_time=solve_time,
            success=success,
            tf=tf_sol,
        )

    @staticmethod
    def resample_solution(
        result: TrajectoryResult, N_new: int,
        collocation_degree: int = 1,
    ) -> np.ndarray:
        """Resample a trajectory result onto a new mesh for warm-starting.

        Args:
            result: Coarse trajectory result.
            N_new: Number of nodes in the new (fine) mesh.
            collocation_degree: Radau degree (determines Z state count).

        Returns:
            w0: Initial guess vector for the fine mesh.
        """
        N_old = result.state.shape[0] - 1
        nx = result.state.shape[1]
        nu = result.control.shape[1]
        d = collocation_degree

        # Normalized time for old and new meshes
        t_old = np.linspace(0, 1, N_old + 1)
        t_new = np.linspace(0, 1, N_new + 1)

        w0 = [result.tf]  # Tf guess

        # Resample states at mesh nodes
        for j in range(nx):
            x_interp = np.interp(t_new, t_old, result.state[:, j])
            w0.extend(x_interp.tolist())

        # Resample controls
        t_ctrl_old = np.linspace(0, 1, N_old)
        t_ctrl_new = np.linspace(0, 1, N_new)
        for j in range(nu):
            u_interp = np.interp(t_ctrl_new, t_ctrl_old, result.control[:, j])
            w0.extend(u_interp.tolist())

        # Boundary target variables (use first and last states)
        w0.extend(result.state[0, :].tolist())
        w0.extend(result.state[-1, :].tolist())

        # Interior collocation states (d >= 2)
        if d >= 2:
            tau_root = [0] + list(cs.collocation_points(d, 'radau'))
            for k in range(N_new):
                s_k = k / N_new
                s_k1 = (k + 1) / N_new
                for j in range(1, d + 1):
                    s_j = s_k + tau_root[j] * (s_k1 - s_k)
                    zj = np.array([np.interp(s_j, t_old, result.state[:, i])
                                   for i in range(nx)])
                    w0.extend(zj.tolist())

        return np.array(w0, dtype=np.float64)
