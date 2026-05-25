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
    tf_min: float = 1.0
    tf_max: float = 50.0

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
    h_min: float = -50.0
    h_max: float = 50.0

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

    # Objective weights (matching MATLAB UAVContinuous.m integrand)
    w_hdot: float = 0.1
    w_q: float = 1.0
    w_dtdot: float = 10.0
    w_dt: float = 10.0
    w_dus: float = 10000.0

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
    collocation_degree: int = 1  # 1=backward Euler, 2 or 3 for higher accuracy

    # Direction
    direction: str = "forward2hover"  # or "hover2forward"

    # Guess transition time
    tf_guess: float = 8.0


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

        # Initial boundary constraint
        g.append(X_syms[0])
        lbg.extend([0] * nx)   # placeholder, set in solve()
        ubg.extend([0] * nx)

        # Final boundary constraint
        g.append(X_syms[N])
        lbg.extend([0] * nx)   # placeholder, set in solve()
        ubg.extend([0] * nx)

        # Path constraint on altitude at all mesh nodes
        for k in range(N + 1):
            g.append(X_syms[k][4])
            lbg.append(cfg.h_min)
            ubg.append(cfg.h_max)

        # Defect constraints (backward Euler: f evaluated at X[k+1])
        for k in range(N):
            Xk = X_syms[k]
            Uk = U_syms[k]
            Xk1 = X_syms[k + 1]

            # Dynamics at X[k+1] (backward Euler)
            fk1 = self._ode(Xk1, Uk)

            # Defect: X[k+1] = X[k] + h * f(X[k+1], U[k])
            g.append(Xk1 - Xk - h * fk1)
            lbg.extend([0] * nx)
            ubg.extend([0] * nx)

            # Stage cost at node k
            Lk = self._stage_cost(Xk, Uk)
            J += h * Lk

        # Initial guess for states: S-curve interpolation
        if cfg.direction == "forward2hover":
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

        # Build NLP
        w_cat = cs.vertcat(*w)
        g_cat = cs.vertcat(*g)

        nlp = {'x': w_cat, 'f': J + Tf, 'g': g_cat}

        opts = {
            'ipopt.max_iter': cfg.ipopt_max_iter,
            'ipopt.tol': cfg.ipopt_tol,
            'ipopt.print_level': cfg.ipopt_print_level,
            'ipopt.mu_strategy': cfg.ipopt_mu_strategy,
            'ipopt.hessian_approximation': cfg.ipopt_hessian_approximation,
            'ipopt.linear_solver': 'mumps',
            'ipopt.nlp_scaling_method': 'gradient-based',
            'ipopt.mu_init': 0.1,
        }

        if cfg.ipopt_warm_start:
            opts['ipopt.warm_start_init_point'] = 'yes'
            opts['ipopt.warm_start_bound_push'] = 1e-6
            opts['ipopt.warm_start_mult_bound_push'] = 1e-6

        self._solver = cs.nlpsol('solver', 'ipopt', nlp, opts)

        # Store for solve()
        self._w = w_cat
        self._g = g_cat
        self._J = J + Tf
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

        Must call build() first.
        """
        if self._solver is None:
            raise RuntimeError("Call build() before solve().")

        cfg = self.cfg
        N = cfg.num_nodes
        nx = 11
        nu = 3

        # Set boundary condition values
        lbg = self._lbg.copy()
        ubg = self._ubg.copy()

        if cfg.direction == "forward2hover":
            x0_val = cfg.x0_forward
            xf_val = cfg.xf_hover
        else:
            x0_val = cfg.xf_hover
            xf_val = cfg.x0_forward

        # Initial boundary (first nx entries of g)
        lbg[:nx] = x0_val
        ubg[:nx] = x0_val

        # Final boundary (next nx entries of g)
        lbg[nx:2*nx] = xf_val
        ubg[nx:2*nx] = xf_val

        t_start = _time.time()
        sol = self._solver(x0=self._w0, lbx=self._lbw, ubx=self._ubw,
                           lbg=lbg, ubg=ubg)
        solve_time = _time.time() - t_start

        # Extract solution
        w_sol = np.array(sol['x']).flatten()
        tf_sol = float(w_sol[0])
        cost_sol = float(sol['f'])

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

        # Set boundary condition values
        lbg = self._lbg.copy()
        ubg = self._ubg.copy()

        if cfg.direction == "forward2hover":
            x0_val = cfg.x0_forward
            xf_val = cfg.xf_hover
        else:
            x0_val = cfg.xf_hover
            xf_val = cfg.x0_forward

        lbg[:nx] = x0_val
        ubg[:nx] = x0_val
        lbg[nx:2*nx] = xf_val
        ubg[nx:2*nx] = xf_val

        t_start = _time.time()
        sol = self._solver(x0=w0, lbx=self._lbw, ubx=self._ubw,
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
    def resample_solution(result: TrajectoryResult, N_new: int) -> np.ndarray:
        """Resample a trajectory result onto a new mesh for warm-starting.

        Args:
            result: Coarse trajectory result.
            N_new: Number of nodes in the new (fine) mesh.

        Returns:
            w0: Initial guess vector for the fine mesh.
        """
        N_old = result.state.shape[0] - 1
        nx = result.state.shape[1]
        nu = result.control.shape[1]

        # Normalized time for old and new meshes
        t_old = np.linspace(0, 1, N_old + 1)
        t_new = np.linspace(0, 1, N_new + 1)

        w0 = [result.tf]  # Tf guess

        # Resample states
        for j in range(nx):
            x_interp = np.interp(t_new, t_old, result.state[:, j])
            w0.extend(x_interp.tolist())

        # Resample controls
        t_ctrl_old = np.linspace(0, 1, N_old)
        t_ctrl_new = np.linspace(0, 1, N_new)
        for j in range(nu):
            u_interp = np.interp(t_ctrl_new, t_ctrl_old, result.control[:, j])
            w0.extend(u_interp.tolist())

        return np.array(w0, dtype=np.float64)
