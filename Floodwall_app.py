"""
=============================================================================
 FLOODWALL DESIGN - STREAMLIT WEB APPLICATION
(Integrated Optimized Modules: Sizing + RCC + Scour + Reporting)
Codes: IS 456:2000 | IS 3370:2009 | NBC 105:2020 | IS 1893:2016 | IRC 6:2017
=============================================================================
"""

import math
import io
import warnings
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")


def compute_hydraulic_effects(gamma_water, flood_depth, protected_depth, seepage_relief_factor, B):
    """Preliminary hydraulic load model for flood-side water, uplift, and residual underseepage."""
    flood_depth = max(0.0, float(flood_depth))
    protected_depth = max(0.0, float(protected_depth))
    seepage_relief_factor = min(max(float(seepage_relief_factor), 0.0), 1.0)

    horizontal_force = 0.5 * gamma_water * (flood_depth ** 2 - protected_depth ** 2)
    if abs(flood_depth ** 2 - protected_depth ** 2) > 1e-9:
        horizontal_centroid = (flood_depth ** 3 - protected_depth ** 3) / (
            3 * (flood_depth ** 2 - protected_depth ** 2)
        )
    else:
        horizontal_centroid = 0.0

    net_head = max(flood_depth - protected_depth, 0.0)
    heel_head = min(flood_depth, protected_depth + (1.0 - seepage_relief_factor) * net_head)
    uplift_toe = gamma_water * flood_depth
    uplift_heel = gamma_water * heel_head
    uplift_force = 0.5 * (uplift_toe + uplift_heel) * B
    uplift_x = (B * (uplift_toe + 2 * uplift_heel) / (3 * (uplift_toe + uplift_heel))
                if uplift_toe + uplift_heel > 1e-9 else B / 2)

    return {
        'flood_depth': flood_depth,
        'protected_depth': protected_depth,
        'horizontal_force': horizontal_force,
        'horizontal_centroid': horizontal_centroid,
        'heel_head': heel_head,
        'uplift_toe': uplift_toe,
        'uplift_heel': uplift_heel,
        'uplift_force': uplift_force,
        'uplift_x': uplift_x,
    }

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Floodwall Design Tool",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a3a5c 0%, #2d6a9f 100%);
        padding: 1.5rem 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 { margin: 0; font-size: 1.9rem; font-weight: 700; }
    .main-header p  { margin: 4px 0 0; font-size: 0.85rem; opacity: 0.85; }
 
    .metric-card {
        background: white;
        border: 1px solid #e0e6ef;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.07);
        text-align: center;
    }
    .metric-card .label { font-size: 0.75rem; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-card .value { font-size: 1.6rem; font-weight: 700; color: #1a3a5c; margin: 4px 0; }
    .metric-card .unit  { font-size: 0.75rem; color: #888; }
 
    .pass-badge  { background:#d4edda; color:#155724; border-radius:4px; padding:2px 8px; font-size:0.82rem; font-weight:600; }
    .fail-badge  { background:#f8d7da; color:#721c24; border-radius:4px; padding:2px 8px; font-size:0.82rem; font-weight:600; }
    .warn-badge  { background:#fff3cd; color:#856404; border-radius:4px; padding:2px 8px; font-size:0.82rem; font-weight:600; }
    .info-badge  { background:#d1ecf1; color:#0c5460; border-radius:4px; padding:2px 8px; font-size:0.82rem; font-weight:600; }
 
    .section-title {
        font-size: 1.1rem; font-weight: 700; color: #1a3a5c;
        border-bottom: 2px solid #2d6a9f;
        padding-bottom: 6px; margin: 1rem 0 0.7rem;
    }
    .code-ref { font-size: 0.7rem; color: #777; font-style: italic; }
    div[data-testid="stExpander"] { border: 1px solid #dce3ef !important; border-radius: 8px !important; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# MODULE 1 – SIZING & OPTIMIZATION (Tightened economic version)
# =============================================================================

class ComprehensiveFloodwallDesign:
    def __init__(self):
        # --- Material Properties (IS 456:2000) ---
        self.fck = 25
        self.fy = 500
        self.gamma_concrete = 25
        self.gamma_water = 9.81
        # --- Soil Properties ---
        self.angle_of_friction = 35
        self.gamma_soil = 18
        self.gamma_sat = 20
        self.mu_friction = 0.50
        self.safe_bearing_capacity = 200
        # --- Geometry ---
        self.depth_of_foundation = 1.5
        self.freeboard = 0.5
        self.stem_top_min = 0.300
        self.protected_water_depth = 0.0
        self.seepage_relief_factor = 0.75
        # --- Surcharge (IRC 6:2017) ---
        self.surcharge_height = 1.2
        # --- Seismic Parameters (NBC 105:2020) ---
        self.seismic_zone = "II"
        self.Z = 0.16
        self.importance_factor = 1.5
        self.soil_type = "C"
        self.site_factor = 1.5
        self.response_factor = 1.5
        # --- Shear Key ---
        self.needs_shear_key = False
        # --- Safety Factor Requirements ---
        self.FOS_OT_static_req = 2.0
        self.FOS_OT_seismic_req = 1.5
        self.FOS_SL_static_req = 1.5
        self.FOS_SL_seismic_req = 1.25
        # --- Counterfort threshold ---
        self.counterfort_height_threshold = 7.0
        self._computed = False

    def _compute_coefficients(self):
        phi = math.radians(self.angle_of_friction)
        delta = math.radians(2/3 * self.angle_of_friction)
        self.Ka = (1 - math.sin(phi)) / (1 + math.sin(phi))
        self.Kp = (1 + math.sin(phi)) / (1 - math.sin(phi))
        self.K0 = 1 - math.sin(phi)
        self.qs = self.Ka * self.gamma_soil * self.surcharge_height
        self.Cd = (self.Z * self.importance_factor * self.site_factor) / self.response_factor
        self.Cd_h = self.Cd
        self.Cd_v = 0.67 * self.Cd_h
        theta = math.atan(self.Cd_h / (1 - self.Cd_v))
        try:
            num = math.cos(phi - theta) ** 2
            t1 = math.cos(theta) * math.cos(delta + theta)
            sn = math.sin(phi + delta) * math.sin(phi - theta)
            sd = math.cos(delta + theta)
            bk = (1 + math.sqrt(sn/sd))**2 if sd != 0 and sn/sd >= 0 else 1.0
            self.KAE = num / (t1 * bk)
        except Exception:
            self.KAE = self.Ka * 1.3
        self.delta_KAE = self.KAE - self.Ka
        self._computed = True

    def initial_assessment(self, hfl, ground_level):
        if not self._computed:
            self._compute_coefficients()
        water_height = hfl - ground_level + self.freeboard
        total_height = water_height + self.depth_of_foundation
        return {
            'water_height': round(water_height, 3),
            'total_height': round(total_height, 3),
            'design_choice': "COUNTERFORT" if total_height > self.counterfort_height_threshold else "CANTILEVER",
            'hfl': hfl,
            'ground_level': ground_level,
        }

    def _check_stability(self, H, Hs, B, toe, heel, tf, ts_base, ts_top,
                         use_shear_key=False, shear_key_depth=None, shear_key_width=None):
        if not self._computed:
            self._compute_coefficients()
        Ka = self.Ka; Kp = self.Kp; gs = self.gamma_soil; gc = self.gamma_concrete
        mu = self.mu_friction; Cd_h = self.Cd_h; Cd_v = self.Cd_v; Df = self.depth_of_foundation
        flood_depth = max(0.0, H - Df - self.freeboard)
        hyd = compute_hydraulic_effects(
            self.gamma_water, flood_depth, self.protected_water_depth,
            self.seepage_relief_factor, B
        )
        Pa_static = 0.5 * Ka * gs * H**2
        Ps_surcharge = Ka * gs * self.surcharge_height * H
        P_h = hyd['horizontal_force']
        Pa_total = Pa_static + Ps_surcharge + P_h
        Pa_dynamic = 0.5 * self.delta_KAE * gs * H**2
        sk_d = sk_w = W_sk = Fi_sk = 0.0
        if use_shear_key:
            sk_d = float(shear_key_depth if shear_key_depth is not None else max(0.5, H * 0.1))
            sk_w = float(shear_key_width if shear_key_width is not None else max(ts_base * 0.7, 0.3))
            W_sk = gc * sk_w * sk_d
            Fi_sk = Cd_h * W_sk
        uplift = hyd['uplift_force']
        uplift_x = hyd['uplift_x']
        W_stem = gc * ((ts_base + ts_top) / 2) * Hs
        W_base = gc * tf * B
        W_heel = gs * heel * Hs
        W_toe = gs * toe * Df
        W_sur = gs * self.surcharge_height * heel
        W_down = W_stem + W_base + W_heel + W_toe + W_sur + W_sk
        W_total = W_down - uplift
        csh = Hs * (2 * ts_top + ts_base) / (3 * (ts_top + ts_base))
        Fi_stem = Cd_h * W_stem; Fi_base = Cd_h * W_base; Fi_total = Fi_stem + Fi_base + Fi_sk
        xs = toe + ts_base / 2; xb = B / 2; xh = toe + ts_base + heel / 2; xt = toe / 2; xsu = xh; xsk = B / 2
        Mr = (W_stem * xs + W_base * xb + W_heel * xh + W_toe * xt +
              W_sur * xsu + W_sk * xsk - uplift * uplift_x)
        Mo_s = (Pa_static * (H / 3) + Ps_surcharge * (H / 2) +
                P_h * (tf + hyd['horizontal_centroid']))
        Mo_e = Pa_dynamic * (0.6 * H) + Fi_stem * (tf + csh) + Fi_base * (tf / 2) + Fi_sk * (tf / 2)
        FOS_OT_s = Mr / Mo_s if Mo_s > 0 else 999
        Wts = W_down * (1 - Cd_v) - uplift
        Mrs = (W_stem * (1 - Cd_v) * xs + W_base * (1 - Cd_v) * xb +
               W_heel * (1 - Cd_v) * xh + W_toe * (1 - Cd_v) * xt + W_sur * (1 - Cd_v) * xsu +
               W_sk * (1 - Cd_v) * xsk - uplift * uplift_x)
        Mo_se = Mo_s + Mo_e
        FOS_OT_e = Mrs / Mo_se if Mo_se > 0 else 999
        # Passive pressure (with optional shear key)
        if use_shear_key:
            Fp = 0.5 * Kp * gs * (tf + Df + sk_d) ** 2
        else:
            Fp = 0.5 * Kp * gs * (tf + Df) ** 2
        Fr_s = mu * max(W_total, 0.0) + Fp
        FOS_SL_s = Fr_s / Pa_total if Pa_total > 0 else 999
        Fr_e = mu * max(Wts, 0.0) + 0.5 * Fp
        Fd_e = Pa_total + Pa_dynamic + Fi_total
        FOS_SL_e = Fr_e / Fd_e if Fd_e > 0 else 999

        # Bearing pressure
        def bearing(W, Mn):
            if W <= 0:
                return 0, 0, 0
            xb2 = Mn / W
            e = max(0, B / 2 - xb2)
            if e <= B / 6:
                return (W / B) * (1 + 6 * e / B), (W / B) * (1 - 6 * e / B), e
            else:
                return 2 * W / (3 * max(B / 2 - e, 0.01)), 0.0, e

        qms, qns, es = bearing(W_total, Mr - Mo_s)
        qme, qne, ee = bearing(Wts, Mrs - Mo_se)
        q_max = max(qms, qme)
        q_min = min(qns, qne)
        mt_ok = (es <= B / 6) and (ee <= B / 4)
        ot_s = FOS_OT_s >= self.FOS_OT_static_req
        ot_e = FOS_OT_e >= self.FOS_OT_seismic_req
        sl_s = FOS_SL_s >= self.FOS_SL_static_req
        sl_e = FOS_SL_e >= self.FOS_SL_seismic_req
        br_ok = q_max <= self.safe_bearing_capacity
        return {
            'FOS_overturning': round(FOS_OT_s, 3),
            'FOS_overturning_seismic': round(FOS_OT_e, 3),
            'FOS_sliding': round(FOS_SL_s, 3),
            'FOS_sliding_seismic': round(FOS_SL_e, 3),
            'q_max': round(q_max, 2),
            'q_min': round(q_min, 2),
            'e_static': round(es, 4),
            'e_seismic': round(ee, 4),
            'middle_third_ok': mt_ok,
            'all_safe': ot_s and ot_e and sl_s and sl_e and br_ok and mt_ok,
            'ot_static_ok': ot_s,
            'ot_seismic_ok': ot_e,
            'sl_static_ok': sl_s,
            'sl_seismic_ok': sl_e,
            'bearing_ok': br_ok,
            'needs_shear_key': use_shear_key,
            'shear_key_depth': sk_d,
            'shear_key_width': sk_w,
            'hydrostatic_force': round(P_h, 2),
            'uplift_force': round(uplift, 2),
            'flood_depth': round(hyd['flood_depth'], 3),
            'protected_depth': round(hyd['protected_depth'], 3),
            'heel_uplift_head': round(hyd['heel_head'], 3),
            'uplift_toe_pressure': round(hyd['uplift_toe'], 2),
            'uplift_heel_pressure': round(hyd['uplift_heel'], 2),
            'W_total': round(W_total, 2),
            'W_down': round(W_down, 2),
            'Pa_total': round(Pa_total, 2),
            'Pa_dynamic': round(Pa_dynamic, 2),
            'Mr_total': round(Mr, 2),
            'Mo_static': round(Mo_s, 2),
            'Mo_seismic': round(Mo_e, 2),
        }

    def _float_range(self, start, stop, step):
        values = []
        current = start
        while current <= stop + 1e-9:
            values.append(round(current, 3))
            current += step
        return values or [round(start, 3)]

    def _estimate_section_cost(self, res):
        vol = ((res['stem_base'] + res['stem_top']) / 2 * res['stem_height'] +
               res['base_width'] * res['base_slab_thickness'])
        if res.get('needs_shear_key'):
            vol += res.get('shear_key_depth', 0) * res.get('shear_key_width', 0)
        steel_kg = vol * 100
        return vol * 9500 + steel_kg * 85

    def _shear_key_options(self, H, ts_base, use_shear_key):
        if not use_shear_key:
            return [(False, 0.0, 0.0)]
        base_depth = max(0.5, round(H * 0.10, 2))
        base_width = max(0.30, round(ts_base * 0.70, 2))
        depth_values = sorted({
            round(max(0.40, base_depth - 0.10), 2),
            round(base_depth, 2),
            round(min(1.20, base_depth + 0.10), 2),
        })
        width_values = sorted({
            round(max(0.30, base_width - 0.05), 2),
            round(base_width, 2),
            round(min(0.60, base_width + 0.05), 2),
        })
        return [(True, sk_depth, sk_width) for sk_depth in depth_values for sk_width in width_values]

    def _shear_key_flags(self, use_shear_key):
        return [use_shear_key] if use_shear_key is not None else [False, True]

    def _governing_check(self, res):
        checks = {
            'Overturning - Static': self.FOS_OT_static_req / max(res['FOS_overturning'], 1e-9),
            'Overturning - Seismic': self.FOS_OT_seismic_req / max(res['FOS_overturning_seismic'], 1e-9),
            'Sliding - Static': self.FOS_SL_static_req / max(res['FOS_sliding'], 1e-9),
            'Sliding - Seismic': self.FOS_SL_seismic_req / max(res['FOS_sliding_seismic'], 1e-9),
            'Bearing Pressure': res['q_max'] / max(self.safe_bearing_capacity, 1e-9),
        }
        return max(checks, key=checks.get)

    def _candidate_sort_key(self, res):
        static_excess = (
            max(0.0, res['FOS_overturning'] - self.FOS_OT_static_req) / self.FOS_OT_static_req +
            max(0.0, res['FOS_sliding'] - self.FOS_SL_static_req) / self.FOS_SL_static_req
        )
        seismic_excess = (
            max(0.0, res['FOS_overturning_seismic'] - self.FOS_OT_seismic_req) / self.FOS_OT_seismic_req +
            max(0.0, res['FOS_sliding_seismic'] - self.FOS_SL_seismic_req) / self.FOS_SL_seismic_req
        )
        bearing_slack = max(0.0, self.safe_bearing_capacity - res['q_max']) / max(self.safe_bearing_capacity, 1e-9)
        return (
            round(static_excess, 6),
            round(seismic_excess + 0.25 * bearing_slack, 6),
            round(self._estimate_section_cost(res), 2),
            round(res['base_width'], 3),
        )

    def _binary_search_B(self, H, Hs, tf, ts_base, ts_top, use_shear_key):
        """Binary search the minimum safe base for each toe ratio, then keep the closest safe design."""

        best_candidate = None
        toe_ratios = self._float_range(0.22, 0.45, 0.01)
        shear_key_options = self._shear_key_options(H, ts_base, use_shear_key)

        for use_shear_key_flag, sk_depth, sk_width in shear_key_options:
            for toe_ratio in toe_ratios:
                B_low = max(0.35 * H, (ts_base + 0.30) / max(1 - toe_ratio, 0.05))
                B_high = max(1.0 * H, B_low + 0.4)
                safe_found = False

                for _ in range(20):
                    toe = round(B_high * toe_ratio * 20) / 20
                    heel = B_high - toe - ts_base
                    if heel < 0.30:
                        B_high += 0.05
                        continue
                    res = self._check_stability(
                        H, Hs, B_high, toe, heel, tf, ts_base, ts_top,
                        use_shear_key=use_shear_key_flag,
                        shear_key_depth=sk_depth,
                        shear_key_width=sk_width,
                    )
                    if res['all_safe']:
                        safe_found = True
                        break
                    B_high += 0.05

                if not safe_found:
                    continue

                B_opt = None
                for _ in range(35):
                    B_mid = (B_low + B_high) / 2
                    toe = round(B_mid * toe_ratio * 20) / 20
                    heel = B_mid - toe - ts_base
                    if heel < 0.30:
                        B_low = B_mid
                        continue
                    res = self._check_stability(
                        H, Hs, B_mid, toe, heel, tf, ts_base, ts_top,
                        use_shear_key=use_shear_key_flag,
                        shear_key_depth=sk_depth,
                        shear_key_width=sk_width,
                    )
                    if res['all_safe']:
                        B_opt = B_mid
                        B_high = B_mid
                    else:
                        B_low = B_mid
                    if B_high - B_low < 0.005:
                        break

                if B_opt is None:
                    continue

                B_final = round(B_opt * 200) / 200
                toe_final = round(B_final * toe_ratio * 20) / 20
                heel_final = round(B_final - toe_final - ts_base, 3)
                if heel_final < 0.30:
                    continue
                result_final = self._check_stability(
                    H, Hs, B_final, toe_final, heel_final, tf, ts_base, ts_top,
                    use_shear_key=use_shear_key_flag,
                    shear_key_depth=sk_depth,
                    shear_key_width=sk_width,
                )
                if not result_final['all_safe']:
                    B_final = round((B_final + 0.05) * 200) / 200
                    toe_final = round(B_final * toe_ratio * 20) / 20
                    heel_final = round(B_final - toe_final - ts_base, 3)
                    if heel_final < 0.30:
                        continue
                    result_final = self._check_stability(
                        H, Hs, B_final, toe_final, heel_final, tf, ts_base, ts_top,
                        use_shear_key=use_shear_key_flag,
                        shear_key_depth=sk_depth,
                        shear_key_width=sk_width,
                    )
                    if not result_final['all_safe']:
                        continue

                result_final.update({
                    'base_width': round(B_final, 3),
                    'toe_length': round(toe_final, 3),
                    'heel_length': round(heel_final, 3),
                    'base_slab_thickness': round(tf, 3),
                    'stem_base': round(ts_base, 3),
                'stem_top': round(ts_top, 3),
                'stem_height': round(Hs, 3),
                'total_height': round(H, 3),
                'water_height': round(max(0.0, H - self.depth_of_foundation - self.freeboard), 3),
                'design_type': 'CANTILEVER',
                'governing_check': self._governing_check(result_final),
            })

                candidate = (self._candidate_sort_key(result_final), result_final)
                if best_candidate is None or candidate[0] < best_candidate[0]:
                    best_candidate = candidate

        return None if best_candidate is None else best_candidate[1]

    def design_cantilever_optimized(self, hfl, ground_level,
                                     use_shear_key=None,
                                     min_tf=None, min_ts_base=None):
        if not self._computed:
            self._compute_coefficients()
        water_height = hfl - ground_level + self.freeboard
        H = water_height + self.depth_of_foundation
        tf_nominal = max(round(H / 12 * 20) / 20, 0.30)
        tf_floor = max(0.30, min_tf) if min_tf is not None else 0.30
        ts_base_nominal = max(round(H / 12 * 20) / 20, 0.30)
        ts_base_floor = max(0.30, min_ts_base) if min_ts_base is not None else 0.30
        ts_top = self.stem_top_min
        tf_values = self._float_range(tf_floor, max(tf_floor, tf_nominal), 0.05)
        ts_base_values = self._float_range(ts_base_floor, max(ts_base_floor, ts_base_nominal), 0.05)
        shear_key_options = self._shear_key_flags(use_shear_key)

        best_candidate = None
        for tf in tf_values:
            Hs = H - tf
            if Hs <= ts_top:
                continue
            for ts_base in ts_base_values:
                for shear_key_flag in shear_key_options:
                    result = self._binary_search_B(H, Hs, tf, ts_base, ts_top, use_shear_key=shear_key_flag)
                    if result is None:
                        continue
                    candidate = (self._candidate_sort_key(result), result)
                    if best_candidate is None or candidate[0] < best_candidate[0]:
                        best_candidate = candidate

        return None if best_candidate is None else best_candidate[1]

    def design_counterfort_wall(self, hfl, ground_level, use_shear_key=None):
        if not self._computed:
            self._compute_coefficients()
        water_height = hfl - ground_level + self.freeboard
        H = water_height + self.depth_of_foundation
        tf = max(round(H / 14 * 20) / 20, 0.35)
        ts_base = max(round(H / 14 * 20) / 20, 0.30)
        ts_top = max(0.200, self.stem_top_min)
        Hs = H - tf
        cf_spacing = max(min(round(0.4 * H * 10) / 10, 4.0), 2.0)
        cf_thickness = max(0.3, ts_base)

        B_opt = None
        result_opt = None
        for shear_key_flag in self._shear_key_flags(use_shear_key):
            B_low = 0.5 * H
            B_high = 0.9 * H
            B_opt = None
            result_opt = None
            for _ in range(20):
                B_mid = (B_low + B_high) / 2
                toe = round(B_mid / 4 * 20) / 20
                heel = B_mid - toe - ts_base
                if heel < 0.5:
                    B_low = B_mid
                    continue
                res = self._check_stability(H, Hs, B_mid, toe, heel, tf, ts_base, ts_top,
                                            use_shear_key=shear_key_flag)
                if res['all_safe']:
                    B_high = B_mid
                    B_opt = B_mid
                    result_opt = res
                else:
                    B_low = B_mid
                if B_high - B_low < 0.01:
                    break
            if B_opt is not None:
                break
        if B_opt is None:
            return None
        B_opt = round(B_opt * 200) / 200
        toe = round(B_opt / 4 * 20) / 20
        heel = B_opt - toe - ts_base
        result_opt.update({
            'base_width': round(B_opt, 3),
            'toe_length': round(toe, 3),
            'heel_length': round(heel, 3),
            'base_slab_thickness': round(tf, 3),
            'stem_base': round(ts_base, 3),
            'stem_top': round(ts_top, 3),
            'stem_height': round(Hs, 3),
            'total_height': round(H, 3),
            'water_height': round(max(0.0, hfl - ground_level), 3),
            'design_type': 'COUNTERFORT',
            'counterfort_spacing': cf_spacing,
            'counterfort_thickness': cf_thickness,
        })
        return result_opt

    def get_design_parameters(self):
        if not self._computed:
            self._compute_coefficients()
        return {
            'fck': self.fck, 'fy': self.fy,
            'gamma_concrete': self.gamma_concrete, 'gamma_soil': self.gamma_soil,
            'angle_of_friction': self.angle_of_friction, 'mu_friction': self.mu_friction,
            'safe_bearing_capacity': self.safe_bearing_capacity,
            'depth_of_foundation': self.depth_of_foundation, 'freeboard': self.freeboard,
            'surcharge_height': self.surcharge_height,
            'protected_water_depth': self.protected_water_depth,
            'seepage_relief_factor': self.seepage_relief_factor,
            'Ka': round(self.Ka, 4), 'Kp': round(self.Kp, 4), 'K0': round(self.K0, 4),
            'KAE': round(self.KAE, 4), 'delta_KAE': round(self.delta_KAE, 4),
            'Cd_h': round(self.Cd_h, 4), 'Cd_v': round(self.Cd_v, 4),
            'seismic_zone': self.seismic_zone, 'Z': self.Z, 'importance_factor': self.importance_factor,
        }


# =============================================================================
# MODULE 2 – SCOUR ANALYSIS
# =============================================================================

def scour_analysis(discharge, river_width, silt_factor=1.76, D50=0.5,
                   mean_velocity=0.5, user_foundation_depth=1.5):
    q = discharge / river_width
    rd = 0.473 * ((q ** 2) / silt_factor) ** (1 / 3)
    gd = rd * 0.15
    sm = 0.3 if q < 1.0 else 0.5
    d1 = gd + sm
    ls = 1.35 * ((q ** 2) / silt_factor) ** (1 / 3)
    d2 = ls * 0.35 * 1.2
    if q < 1.0:
        usace_min = 1.0
        channel_class = "Small"
    elif q < 5.0:
        usace_min = 1.2
        channel_class = "Medium"
    else:
        usace_min = 1.5
        channel_class = "Large"
    d4 = 0.8
    req = round(max(d1, d2, usace_min, d4) * 10) / 10
    adopted = max(req, user_foundation_depth)
    if q < 1.0:
        prot = {
            'riprap_width': 2, 'riprap_depth': 0.3,
            'stone_size': '100-150mm', 'geotextile': 'recommended',
            'toe_extension': 0.5
        }
    elif q < 5.0:
        prot = {
            'riprap_width': 3, 'riprap_depth': 0.4,
            'stone_size': '150-250mm', 'geotextile': 'mandatory',
            'toe_extension': 1.0
        }
    else:
        prot = {
            'riprap_width': 4, 'riprap_depth': 0.5,
            'stone_size': '250-400mm', 'geotextile': 'mandatory',
            'toe_extension': 1.5
        }
    return {
        'q_unit': round(q, 3),
        'regime_depth': round(rd, 2),
        'required_depth': req,
        'adopted_depth': adopted,
        'channel_class': channel_class,
        'protection': prot,
        'methods': {
            'degradation': round(d1, 2),
            'lacey_modified': round(d2, 2),
            'usace': usace_min,
            'practical': d4,
        }
    }


# =============================================================================
# MODULE 3 – RCC DESIGN (with dynamic bar selection and shear key detailing)
# =============================================================================

class RCCDesignParameters:
    def __init__(self, fck=25, fy=500):
        self.fck = fck
        self.fy = fy
        self.gamma_c = 25
        self.Es = 200000
        self.Ec = 5000 * math.sqrt(fck)
        self.m_modular = self.Es / self.Ec
        self.cover_stem = 40
        self.cover_base = 50
        self.xu_max_d = 0.46
        self.Ru = 0.36 * fck * 0.46 * (1 - 0.42 * 0.46)
        self.w_cr_limit = 0.2
        self.gamma_f_DL = 1.5
        self.gamma_f_EL = 1.5
        self.gamma_f_LL = 1.5
        self.gamma_f_EQ = 1.2


def get_tau_c(pt_percent, fck):
    pt_table = [0.15, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50,
                1.75, 2.00, 2.25, 2.50, 2.75, 3.00]
    tc_m20 = [0.28, 0.36, 0.48, 0.56, 0.62, 0.67, 0.72,
              0.75, 0.79, 0.81, 0.82, 0.82, 0.82]
    tc_m25 = [0.29, 0.36, 0.49, 0.57, 0.64, 0.70, 0.74,
              0.78, 0.82, 0.84, 0.86, 0.86, 0.86]
    tc = tc_m25 if fck >= 25 else tc_m20
    pt = max(0.15, min(pt_percent, 3.00))
    for j in range(len(pt_table) - 1):
        if pt_table[j] <= pt <= pt_table[j + 1]:
            ratio = (pt - pt_table[j]) / (pt_table[j + 1] - pt_table[j])
            return tc[j] + ratio * (tc[j + 1] - tc[j])
    return tc[-1]


def calc_crack_width(M_service, b, d, D, Ast, cover, bar_dia, Es, Ec):
    if Ast <= 0 or M_service <= 0:
        return 0.0
    m = Es / Ec
    rho = Ast / (b * d)
    x = d * (math.sqrt(2 * m * rho + (m * rho) ** 2) - m * rho)
    if x <= 0 or x >= d:
        return 0.0
    z = d - x / 3
    fs = M_service * 1e6 / (Ast * z)
    e1 = fs / Es
    hx = D - x
    em = e1 * hx / (d - x)
    estiff = b * (D - x) ** 2 / (3 * Es * Ast * (d - x))
    em = max(em - estiff, 0.5 * e1)
    n_bars = max(Ast / (math.pi / 4 * bar_dia ** 2), 1)
    sp = min(1000 / n_bars, 300)
    acr = math.sqrt((sp / 2) ** 2 + (cover + bar_dia / 2) ** 2) - bar_dia / 2
    cmin = cover
    w_cr = 3 * acr * em / (1 + 2 * (acr - cmin) / (D - x))
    return max(w_cr, 0.0)


def calc_development_length(bar_dia, fy, fck, hook=False):
    if fck <= 20:
        tau_bd_plain = 1.2
    elif fck <= 25:
        tau_bd_plain = 1.4
    elif fck <= 30:
        tau_bd_plain = 1.5
    else:
        tau_bd_plain = 1.7
    tau_bd = tau_bd_plain * 1.6
    Ld = bar_dia * fy / (4 * tau_bd)
    if hook:
        Ld *= 0.7
    return Ld


def select_reinforcement(required_Ast, min_spacing=100, max_spacing=300,
                         min_dia=12, max_dia=25, cover=40, b=1000):
    best_dia = None
    best_spacing = None
    best_Ast = None
    for dia in range(min_dia, max_dia + 1, 2):
        max_Ast = math.pi / 4 * dia ** 2 * 1000 / min_spacing
        if max_Ast < required_Ast:
            continue
        spacing_needed = math.pi / 4 * dia ** 2 * 1000 / required_Ast
        spacing = math.floor(spacing_needed / 25) * 25
        spacing = max(spacing, min_spacing)
        spacing = min(spacing, max_spacing)
        Ast_prov = math.pi / 4 * dia ** 2 * 1000 / spacing
        if Ast_prov >= required_Ast:
            if best_dia is None or dia < best_dia:
                best_dia = dia
                best_spacing = spacing
                best_Ast = Ast_prov
    if best_dia is not None:
        return best_dia, best_spacing, best_Ast, False
    else:
        dia = max_dia
        spacing = min_spacing
        Ast_prov = math.pi / 4 * dia ** 2 * 1000 / spacing
        return dia, spacing, Ast_prov, True


def design_rcc_section(H, Hs, B, toe, heel, tf, ts_base, ts_top,
                       soil_params, seismic_params, rcc_params,
                       D_toe_soil=1.0, shear_key_depth=0.0, shear_key_width=0.0):
    p = rcc_params
    Ka = soil_params['Ka']; Kp = soil_params['Kp']
    dKAE = soil_params['delta_KAE']; gs = soil_params['gamma_s']
    gc = p.gamma_c; sh = soil_params['surcharge_height']
    Cd_h = seismic_params['Cd_h']; Cd_v = seismic_params['Cd_v']
    hyd = compute_hydraulic_effects(
        soil_params.get('gamma_w', 9.81),
        soil_params.get('flood_depth', max(H - D_toe_soil, 0.0)),
        soil_params.get('protected_depth', 0.0),
        soil_params.get('seepage_relief_factor', 0.75),
        B
    )
    P_h = hyd['horizontal_force']
    uplift = hyd['uplift_force']
    uplift_x = hyd['uplift_x']

    D_sb = ts_base * 1000; D_st = ts_top * 1000; tf_mm = tf * 1000
    bar_main = 16
    eff_stem = D_sb - p.cover_stem - bar_main / 2
    eff_base = tf_mm - p.cover_base - bar_main / 2

    # Forces (unfactored)
    Pa_static = 0.5 * Ka * gs * H ** 2
    Ps_surcharge = Ka * gs * sh * H
    Pa_total = Pa_static + Ps_surcharge + P_h
    Pa_dynamic = 0.5 * dKAE * gs * H ** 2
    sk_d = max(float(shear_key_depth or 0), 0.0)
    sk_w = max(float(shear_key_width or 0), 0.0)
    W_sk = gc * sk_d * sk_w
    W_stem = gc * ((ts_base + ts_top) / 2) * Hs
    W_base = gc * tf * B
    W_heel = gs * heel * Hs
    W_toe = gs * toe * D_toe_soil
    W_sur = gs * sh * heel
    W_down = W_stem + W_base + W_heel + W_toe + W_sur + W_sk
    W_total = W_down - uplift
    csh = Hs * (2 * ts_top + ts_base) / (3 * (ts_top + ts_base))
    Fi_stem = Cd_h * W_stem
    Fi_base = Cd_h * W_base
    Fi_sk = Cd_h * W_sk
    Fi_total = Fi_stem + Fi_base + Fi_sk
    xs = toe + ts_base / 2; xb = B / 2; xh = toe + ts_base + heel / 2; xt = toe / 2; xsu = xh; xsk = B / 2
    Mr = W_stem * xs + W_base * xb + W_heel * xh + W_toe * xt + W_sur * xsu + W_sk * xsk - uplift * uplift_x
    Mo_s = Pa_static * (H / 3) + Ps_surcharge * (H / 2) + P_h * (tf + hyd['horizontal_centroid'])
    Mo_e = Pa_dynamic * (0.6 * H) + Fi_stem * (tf + csh) + Fi_base * (tf / 2) + Fi_sk * (tf / 2)
    FOS_OT_s = Mr / Mo_s if Mo_s > 0 else 999
    Wts = W_down * (1 - Cd_v) - uplift
    Mrs = (W_stem * (1 - Cd_v) * xs + W_base * (1 - Cd_v) * xb +
           W_heel * (1 - Cd_v) * xh + W_toe * (1 - Cd_v) * xt + W_sur * (1 - Cd_v) * xsu +
           W_sk * (1 - Cd_v) * xsk - uplift * uplift_x)
    Mo_se = Mo_s + Mo_e
    FOS_OT_e = Mrs / Mo_se if Mo_se > 0 else 999
    Fp = 0.5 * Kp * gs * ((tf + D_toe_soil + sk_d) ** 2 if sk_d > 0 else (tf + D_toe_soil) ** 2)
    Fr_s = soil_params['mu'] * max(W_total, 0.0) + Fp
    FOS_SL_s = Fr_s / Pa_total if Pa_total > 0 else 999
    Fr_e = soil_params['mu'] * max(Wts, 0.0) + 0.5 * Fp
    Fd_e = Pa_total + Pa_dynamic + Fi_total
    FOS_SL_e = Fr_e / Fd_e if Fd_e > 0 else 999

    def bearing(W, Mn):
        if W <= 0:
            return 0, 0, 0
        xb2 = Mn / W
        e = max(0, B / 2 - xb2)
        if e <= B / 6:
            return (W / B) * (1 + 6 * e / B), (W / B) * (1 - 6 * e / B), e
        else:
            return 2 * W / (3 * max(B / 2 - e, 0.01)), 0.0, e

    qms, qns, es = bearing(W_total, Mr - Mo_s)
    qme, qne, ee = bearing(Wts, Mrs - Mo_se)
    q_max = max(qms, qme)
    q_min = min(qns, qne)
    BC_OK = "OK" if q_max <= soil_params['qa'] else "FAIL"

    # Stem design
    M_stem_s = ((Ka * gs * Hs ** 2 / 2) * (Hs / 3) +
                (Ka * gs * sh * Hs) * (Hs / 2) +
                P_h * hyd['horizontal_centroid'])
    M_stem_e = 0.5 * dKAE * gs * Hs ** 2 * (0.6 * Hs) + Fi_stem * csh
    V_stem_s = 0.5 * Ka * gs * Hs ** 2 + Ka * gs * sh * Hs + P_h
    V_stem_e = 0.5 * dKAE * gs * Hs ** 2 + Fi_stem
    Mu_stem = max(p.gamma_f_EL * M_stem_s,
                  p.gamma_f_EL * M_stem_s + p.gamma_f_EQ * M_stem_e,
                  p.gamma_f_EL * M_stem_s + 1.5 * M_stem_e)
    Vu_stem = max(p.gamma_f_EL * V_stem_s,
                  p.gamma_f_EL * V_stem_s + p.gamma_f_EQ * V_stem_e,
                  p.gamma_f_EL * V_stem_s + 1.5 * V_stem_e)
    M_svc = M_stem_s + M_stem_e

    d_eff = eff_stem
    disc = 1 - (4.6 * Mu_stem * 1e6) / (p.fck * 1000 * d_eff ** 2)
    if disc >= 0:
        Ast_req = (0.5 * p.fck / p.fy) * (1 - math.sqrt(disc)) * 1000 * d_eff
    else:
        Ast_req = Mu_stem * 1e6 / (0.87 * p.fy * 0.9 * d_eff)
    Ast_min = 0.12 * 1000 * D_sb / 100
    Ast_req = max(Ast_req, Ast_min)

    bar_dia_stem, spacing_stem, Ast_prov, _ = select_reinforcement(
        Ast_req, min_spacing=100, max_spacing=150, min_dia=12, max_dia=25, cover=p.cover_stem)

    # Distribution bars
    Ast_dist = 0.24 * 1000 * D_sb / 100
    bar_dia_dist = 12
    spacing_dist = math.pi / 4 * bar_dia_dist ** 2 * 1000 / Ast_dist
    spacing_dist = min(spacing_dist, 300)
    spacing_dist = min(spacing_dist, D_sb)
    spacing_dist = max(spacing_dist, 100)
    spacing_dist = math.floor(spacing_dist / 25) * 25
    if spacing_dist == 0:
        spacing_dist = 100

    # Curtailment
    H_mid = Hs / 2
    D_mid = D_st + (D_sb - D_st) * (H_mid / Hs)
    eff_mid = D_mid - p.cover_stem - bar_dia_stem / 2
    M_mid_s = (Ka * gs * H_mid ** 2 / 2) * (H_mid / 3) + (Ka * gs * sh * H_mid) * (H_mid / 2)
    W_mid = gc * ((D_mid / 1000 + ts_top) / 2) * H_mid
    Fi_mid = Cd_h * W_mid
    centroid_mid = (H_mid * (2 * ts_top + D_mid / 1000)) / (3 * (ts_top + D_mid / 1000))
    M_mid_e = 0.5 * dKAE * gs * H_mid ** 2 * (0.6 * H_mid) + Fi_mid * centroid_mid
    Mu_mid = max(p.gamma_f_EL * M_mid_s, p.gamma_f_EL * M_mid_s + p.gamma_f_EQ * M_mid_e)
    disc_mid = 1 - (4.6 * Mu_mid * 1e6) / (p.fck * 1000 * eff_mid ** 2)
    if disc_mid >= 0:
        Ast_mid_req = (0.5 * p.fck / p.fy) * (1 - math.sqrt(disc_mid)) * 1000 * eff_mid
    else:
        Ast_mid_req = Mu_mid * 1e6 / (0.87 * p.fy * 0.9 * eff_mid)
    Ast_mid_min = 0.12 * 1000 * D_mid / 100
    Ast_mid = max(Ast_mid_req, Ast_mid_min)
    spacing_mid = math.pi / 4 * bar_dia_stem ** 2 * 1000 / Ast_mid
    spacing_mid = min(spacing_mid, 300)
    spacing_mid = max(spacing_mid, 100)
    spacing_mid = math.floor(spacing_mid / 25) * 25
    if spacing_mid == 0:
        spacing_mid = 100

    # Base slab
    q_toe_edge = q_max
    q_heel_edge = q_min
    q_at_stem_heel = q_heel_edge + (q_toe_edge - q_heel_edge) * heel / B
    q_at_stem_toe = q_heel_edge + (q_toe_edge - q_heel_edge) * (B - toe) / B
    u_toe_edge = hyd['uplift_toe']
    u_heel_edge = hyd['uplift_heel']
    u_at_stem_heel = u_heel_edge + (u_toe_edge - u_heel_edge) * heel / B
    u_at_stem_toe = u_heel_edge + (u_toe_edge - u_heel_edge) * (B - toe) / B
    w_down_heel = gs * Hs + gc * tf + gs * sh - 0.5 * (u_heel_edge + u_at_stem_heel)
    M_heel_up = q_heel_edge * heel ** 2 / 2 + (q_at_stem_heel - q_heel_edge) * heel ** 2 / 3
    M_heel_down = w_down_heel * heel ** 2 / 2
    M_heel = p.gamma_f_EL * (M_heel_down - M_heel_up)
    w_down_toe = gc * tf + gs * D_toe_soil - 0.5 * (u_toe_edge + u_at_stem_toe)
    M_toe_up = q_at_stem_toe * toe ** 2 / 2 + (q_toe_edge - q_at_stem_toe) * toe ** 2 / 3
    M_toe_down = w_down_toe * toe ** 2 / 2
    M_toe = p.gamma_f_EL * (M_toe_up - M_toe_down)
    M_base = max(abs(M_heel), abs(M_toe))
    d_eff_base = eff_base
    disc_base = 1 - (4.6 * M_base * 1e6) / (p.fck * 1000 * d_eff_base ** 2)
    if disc_base >= 0:
        Ast_base_req = (0.5 * p.fck / p.fy) * (1 - math.sqrt(max(0, disc_base))) * 1000 * d_eff_base
    else:
        Ast_base_req = M_base * 1e6 / (0.87 * p.fy * 0.9 * d_eff_base)
    Ast_base_min = 0.12 * 1000 * tf_mm / 100
    Ast_base_req = max(Ast_base_req, Ast_base_min)
    bar_dia_base, spacing_base, Ast_base_prov, _ = select_reinforcement(
        Ast_base_req, min_spacing=100, max_spacing=150, min_dia=12, max_dia=25, cover=p.cover_base)

    # Shear check
    tau_v = Vu_stem * 1000 / (1000 * d_eff)
    pt_stem = 100 * Ast_prov / (1000 * d_eff)
    tau_c = get_tau_c(pt_stem, p.fck)
    UR_shear = tau_v / tau_c
    if tau_v <= tau_c:
        shear_OK = f"OK - UR:{UR_shear:.2f}"
        shear_pass = True
    else:
        d_req = math.ceil(Vu_stem * 1000 / (tau_c * 1000))
        shear_OK = f"FAIL - UR:{UR_shear:.2f} | d_req={d_req}mm (prov={d_eff:.0f}mm)"
        shear_pass = False

    # Crack width
    w_cr = calc_crack_width(M_svc, 1000, d_eff, D_sb, Ast_prov,
                            p.cover_stem, bar_dia_stem, p.Es, p.Ec)
    UR_crack = w_cr / p.w_cr_limit if p.w_cr_limit > 0 else 0
    if w_cr <= p.w_cr_limit:
        crack_OK = f"OK - UR:{UR_crack:.2f}"
        crack_pass = True
    else:
        recommendation = f"Reduce bar spacing (current {spacing_stem}mm > min 100mm)" if spacing_stem > 100 else "Increase section thickness"
        crack_OK = f"FAIL - UR:{UR_crack:.2f} | {recommendation}"
        crack_pass = False

    # Development length
    Ld_stem = calc_development_length(bar_dia_stem, p.fy, p.fck, hook=True)
    Ld_avail = tf_mm - p.cover_base
    if Ld_avail >= Ld_stem:
        Ld_OK = "OK"
        Ld_pass = True
    else:
        Ld_OK = f"FAIL - {Ld_stem:.0f}mm req, {Ld_avail:.0f}mm avail"
        Ld_pass = False

    # Shear key detailing
    shear_key_result = {}
    if shear_key_depth > 0 and shear_key_width > 0:
        d_sk = shear_key_depth
        w_sk = shear_key_width
        p_sk_top = Kp * gs * D_toe_soil
        p_sk_bot = Kp * gs * (D_toe_soil + d_sk)
        V_sk_uf = (p_sk_top + p_sk_bot) / 2 * d_sk
        M_sk_uf = p_sk_top * d_sk ** 2 / 2 + (p_sk_bot - p_sk_top) * d_sk ** 2 / 3
        Mu_sk = p.gamma_f_EL * M_sk_uf
        Vu_sk = p.gamma_f_EL * V_sk_uf
        D_sk_mm = w_sk * 1000
        d_sk_eff = D_sk_mm - p.cover_base - 6
        if d_sk_eff > 0:
            disc_sk = 1 - (4.6 * Mu_sk * 1e6) / (p.fck * 1000 * d_sk_eff ** 2)
            if disc_sk >= 0:
                Ast_sk_req = (0.5 * p.fck / p.fy) * (1 - math.sqrt(disc_sk)) * 1000 * d_sk_eff
            else:
                Ast_sk_req = Mu_sk * 1e6 / (0.87 * p.fy * 0.9 * d_sk_eff)
        else:
            Ast_sk_req = 0.0
        Ast_sk_min = 0.12 * 1000 * D_sk_mm / 100
        Ast_sk_req = max(Ast_sk_req, Ast_sk_min)
        bar_sk, spacing_sk, Ast_sk_prov, _ = select_reinforcement(
            Ast_sk_req, min_spacing=100, max_spacing=200, min_dia=10, max_dia=20, cover=p.cover_base)
        tau_v_sk = (Vu_sk * 1000 / (1000 * d_sk_eff)) if d_sk_eff > 0 else 0
        pt_sk = 100 * Ast_sk_prov / (1000 * d_sk_eff) if d_sk_eff > 0 else 0
        tau_c_sk = get_tau_c(pt_sk, p.fck)
        UR_shear_sk = tau_v_sk / tau_c_sk if tau_c_sk > 0 else 0
        if tau_v_sk <= tau_c_sk:
            sk_shear_check = f"OK - UR:{UR_shear_sk:.2f}"
        else:
            d_req_sk = math.ceil(Vu_sk * 1000 / (tau_c_sk * 1000))
            sk_shear_check = f"FAIL - UR:{UR_shear_sk:.2f} | d_req={d_req_sk}mm (prov={d_sk_eff:.0f}mm)"
        shear_key_result = {
            'SK_Depth_m': round(d_sk, 3),
            'SK_Width_m': round(w_sk, 3),
            'SK_p_top (kN/m2)': round(p_sk_top, 2),
            'SK_p_bot (kN/m2)': round(p_sk_bot, 2),
            'Mu_SK (kNm/m)': round(Mu_sk, 2),
            'Vu_SK (kN/m)': round(Vu_sk, 2),
            'Ast_SK_req (mm2/m)': round(Ast_sk_req, 0),
            'Ast_SK_prov (mm2/m)': round(Ast_sk_prov, 0),
            'SK Bar': f"T{bar_sk}@{int(spacing_sk)}",
            'SK Shear Check': sk_shear_check,
        }

    result = {
        "FOS_OT_static": round(FOS_OT_s, 2),
        "FOS_OT_seismic": round(FOS_OT_e, 2),
        "FOS_SL_static": round(FOS_SL_s, 2),
        "FOS_SL_seismic": round(FOS_SL_e, 2),
        "Hydrostatic Force (kN/m)": round(P_h, 2),
        "Uplift Force (kN/m)": round(uplift, 2),
        "Flood Depth (m)": round(hyd['flood_depth'], 3),
        "Protected Water Depth (m)": round(hyd['protected_depth'], 3),
        "Heel Uplift Head (m)": round(hyd['heel_head'], 3),
        "q_max (kN/m2)": round(q_max, 2),
        "q_min (kN/m2)": round(q_min, 2),
        "Bearing Check": BC_OK,
        "e_static (m)": round(es, 3),
        "e_seismic (m)": round(ee, 3),
        "Mu_stem (kNm/m)": round(Mu_stem, 2),
        "Vu_stem (kN/m)": round(Vu_stem, 2),
        "M_stem_service (kNm/m)": round(M_svc, 2),
        "Ast_stem_req (mm2/m)": round(Ast_req, 0),
        "Ast_stem_prov (mm2/m)": round(Ast_prov, 0),
        "Stem Main Bar": f"T{bar_dia_stem}@{int(spacing_stem)}",
        "Stem Dist Bar": f"T{bar_dia_dist}@{int(spacing_dist)}",
        "Curtailment": f"T{bar_dia_stem}@{int(spacing_mid)} above H/2",
        "M_heel (kNm/m)": round(M_heel, 2),
        "M_toe (kNm/m)": round(M_toe, 2),
        "Mu_base (kNm/m)": round(M_base, 2),
        "Ast_base_req (mm2/m)": round(Ast_base_req, 0),
        "Ast_base_prov (mm2/m)": round(Ast_base_prov, 0),
        "Base Main Bar": f"T{bar_dia_base}@{int(spacing_base)}",
        "tau_v (N/mm2)": round(tau_v, 3),
        "tau_c (N/mm2)": round(tau_c, 3),
        "UR_Shear": round(UR_shear, 3),
        "Shear Check": shear_OK,
        "Shear Pass": shear_pass,
        "Crack Width (mm)": round(w_cr, 3),
        "UR_Crack": round(UR_crack, 3),
        "Crack Check": crack_OK,
        "Crack Pass": crack_pass,
        "Ld (mm)": round(Ld_stem, 0),
        "Ld Available (mm)": round(Ld_avail, 0),
        "Ld Check": Ld_OK,
        "Ld Pass": Ld_pass,
    }
    result.update(shear_key_result)
    return result


def design_cantilever_optimized_with_rcc(engine, hfl, ground_level,
                                          soil_params, seismic_params,
                                          rcc_params=None,
                                          use_shear_key=None,
                                          D_toe_soil=1.0,
                                          max_iter=20):
    if rcc_params is None:
        rcc_params = RCCDesignParameters()
    min_tf = None
    min_ts_base = None
    rcc_result = None
    for iteration in range(max_iter):
        sizing = engine.design_cantilever_optimized(
            hfl, ground_level, use_shear_key=use_shear_key,
            min_tf=min_tf, min_ts_base=min_ts_base)
        if sizing is None:
            print(f"    [Iter {iteration + 1}] Geometry sizing failed – no stable configuration found.")
            return None
        H = sizing['total_height']; Hs = sizing['stem_height']
        B = sizing['base_width']; toe = sizing['toe_length']; heel = sizing['heel_length']
        tf = sizing['base_slab_thickness']; ts_base = sizing['stem_base']; ts_top = sizing['stem_top']
        sk_depth = float(sizing.get('shear_key_depth', 0) or 0)
        sk_width = float(sizing.get('shear_key_width', 0) or 0)
        rcc_result = design_rcc_section(
            H, Hs, B, toe, heel, tf, ts_base, ts_top,
            soil_params, seismic_params, rcc_params,
            D_toe_soil=D_toe_soil,
            shear_key_depth=sk_depth, shear_key_width=sk_width)
        shear_ok = rcc_result.get('Shear Pass', False)
        crack_ok = rcc_result.get('Crack Pass', False)
        ld_ok = rcc_result.get('Ld Pass', False)
        if shear_ok and crack_ok and ld_ok:
            sizing['rcc_result'] = rcc_result
            sizing['rcc_iterations'] = iteration + 1
            sizing['rcc_all_safe'] = True
            return sizing
        # Increment thickness on failure
        fail_parts = []
        if not shear_ok:
            fail_parts.append(f"Shear: {rcc_result['Shear Check']}")
        if not crack_ok:
            fail_parts.append(f"Crack: {rcc_result['Crack Check']}")
        if not ld_ok:
            fail_parts.append(f"Ld: {rcc_result['Ld Check']}")
        print(f"    [Iter {iteration + 1}] RCC FAIL – {' | '.join(fail_parts)}")
        print(f"      → Incrementing tf & ts_base by 50mm "
              f"(new floor: tf={((min_tf or tf) + 0.05) * 1000:.0f}mm, "
              f"ts_base={((min_ts_base or ts_base) + 0.05) * 1000:.0f}mm)")
        min_tf = (min_tf or tf) + 0.05
        min_ts_base = (min_ts_base or ts_base) + 0.05
    print(f"    WARNING: RCC checks still failing after {max_iter} iterations. "
          "Returning last result for review.")
    if sizing is not None:
        sizing['rcc_result'] = rcc_result
        sizing['rcc_iterations'] = max_iter
        sizing['rcc_all_safe'] = False
    return sizing


def configure_design_engine(hfl, ground_level, protected_wl, foundation_depth,
                            phi, gs, mu, sbc, fb, sur, seep_relief,
                            Z, IF, sitef, respf, fck, fy):
    engine = ComprehensiveFloodwallDesign()
    engine.safe_bearing_capacity = sbc
    engine.angle_of_friction = phi
    engine.depth_of_foundation = foundation_depth
    engine.freeboard = fb
    engine.protected_water_depth = max(0.0, protected_wl - ground_level)
    engine.seepage_relief_factor = seep_relief
    engine.gamma_soil = gs
    engine.mu_friction = mu
    engine.surcharge_height = sur
    engine.Z = Z
    engine.importance_factor = IF
    engine.site_factor = sitef
    engine.response_factor = respf
    engine.fck = fck
    engine.fy = fy
    engine._compute_coefficients()
    return engine


def build_design_context(engine, hfl, ground_level, protected_wl, gs, mu, sbc, sur, seep_relief, fck, fy):
    soil_p = {
        'Ka': engine.Ka, 'Kp': engine.Kp, 'KAE': engine.KAE,
        'delta_KAE': engine.delta_KAE, 'gamma_s': gs, 'mu': mu,
        'qa': sbc, 'surcharge_height': sur, 'gamma_w': engine.gamma_water,
        'flood_depth': max(0.0, hfl - ground_level),
        'protected_depth': max(0.0, protected_wl - ground_level),
        'seepage_relief_factor': seep_relief,
    }
    seis_p = {'Cd_h': engine.Cd_h, 'Cd_v': engine.Cd_v}
    rcc_p = RCCDesignParameters(fck=fck, fy=fy)
    return soil_p, seis_p, rcc_p


def build_safe_geometry_error_message(use_shear_key):
    if use_shear_key is True:
        return "Could not find a safe geometry with shear key enabled. Try increasing SBC or reducing loads."
    if use_shear_key is False:
        return ("Could not find a safe geometry with shear key disabled. "
                "Try increasing SBC, reducing loads, or switching Shear Key Mode to Auto or Enabled.")
    return ("Could not find a safe geometry after checking both without and with a shear key. "
            "Try increasing SBC or reducing loads.")


def run_design_case(engine, hfl, ground_level, soil_p, seis_p, rcc_p, foundation_depth,
                    use_shear_key=None):
    assessment = engine.initial_assessment(hfl, ground_level)
    dtype = assessment['design_choice']
    rcc_result = None
    if dtype == "CANTILEVER":
        sizing = design_cantilever_optimized_with_rcc(
            engine, hfl, ground_level, soil_p, seis_p, rcc_params=rcc_p,
            use_shear_key=use_shear_key, D_toe_soil=foundation_depth
        )
        if sizing:
            rcc_result = sizing.get('rcc_result')
    else:
        sizing = engine.design_counterfort_wall(hfl, ground_level, use_shear_key=use_shear_key)
        if sizing and sizing.get('all_safe'):
            rcc_result = design_rcc_section(
                sizing['total_height'], sizing['stem_height'],
                sizing['base_width'], sizing['toe_length'], sizing['heel_length'],
                sizing['base_slab_thickness'], sizing['stem_base'], sizing['stem_top'],
                soil_p, seis_p, rcc_p, D_toe_soil=foundation_depth,
                shear_key_depth=sizing.get('shear_key_depth', 0),
                shear_key_width=sizing.get('shear_key_width', 0)
            )
    return assessment, sizing, rcc_result


def summarize_batch_result(station, hfl, ground_level, engine, sizing):
    result = {
        'Station': station,
        'HFL': hfl,
        'Ground_Level': ground_level,
        'Water_Height': round(max(0.0, hfl - ground_level), 2),
    }
    if not sizing:
        result['Status'] = 'FAILED'
        return result

    rcc_safe = sizing.get('rcc_all_safe', True)
    if sizing.get('all_safe') and rcc_safe:
        status = 'SAFE'
    elif sizing.get('all_safe'):
        status = 'RCC_REVIEW'
    else:
        status = 'FAILED'

    result.update({
        'Status': status,
        'Design_Type': sizing.get('design_type', ''),
        'Total_Height_m': sizing.get('total_height'),
        'Base_Width_m': sizing.get('base_width'),
        'Stem_Base_mm': int(sizing['stem_base'] * 1000) if sizing.get('stem_base') is not None else None,
        'Stem_Top_mm': int(sizing['stem_top'] * 1000) if sizing.get('stem_top') is not None else None,
        'Base_Slab_mm': int(sizing['base_slab_thickness'] * 1000) if sizing.get('base_slab_thickness') is not None else None,
        'Shear_Key': 'YES' if sizing.get('needs_shear_key') else 'NO',
        'FOS_OT': sizing.get('FOS_overturning'),
        'FOS_SL': sizing.get('FOS_sliding'),
        'q_max_kN_m2': sizing.get('q_max'),
    })
    return result


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def draw_wall_section(sizing, ground_level, hfl):
    B = sizing['base_width']; toe = sizing['toe_length']; heel = sizing['heel_length']
    tf = sizing['base_slab_thickness']; Hs = sizing['stem_height']
    tsb = sizing['stem_base']; tst = sizing['stem_top']
    Df = sizing.get('total_height', 0) - Hs - tf
    has_sk = sizing.get('needs_shear_key', False)
    sk_d = sizing.get('shear_key_depth', 0)
    sk_w = sizing.get('shear_key_width', 0)

    fig = go.Figure()
    # Soil foundation
    fig.add_shape(type="rect", x0=-0.3, x1=B+0.3, y0=-tf-Df, y1=0,
                  fillcolor="rgba(210,180,140,0.3)", line_width=0)
    # HFL line
    hfl_rel = hfl - ground_level
    fig.add_shape(type="line", x0=-0.5, x1=B+0.5, y0=hfl_rel, y1=hfl_rel,
                  line=dict(color="#2196F3", dash="dash", width=2))
    fig.add_annotation(x=B+0.4, y=hfl_rel, text=f"HFL ({hfl:.2f}m)",
                       font=dict(color="#2196F3", size=11), showarrow=False, xanchor="left")

    # Base slab
    bx = [0, B, B, 0, 0]; by = [-tf, -tf, 0, 0, -tf]
    fig.add_trace(go.Scatter(x=bx, y=by, fill="toself",
                             fillcolor="rgba(96,125,139,0.7)", line_color="#455a64",
                             name="Base Slab"))
    # Shear key
    if has_sk and sk_d > 0:
        skx = [B/2 - sk_w/2, B/2 + sk_w/2, B/2 + sk_w/2, B/2 - sk_w/2, B/2 - sk_w/2]
        sky = [-tf, -tf, -tf - sk_d, -tf - sk_d, -tf]
        fig.add_trace(go.Scatter(x=skx, y=sky, fill="toself",
                                 fillcolor="rgba(96,125,139,0.9)", line_color="#37474f",
                                 name="Shear Key"))
    # Stem
    stem_x_left = toe
    sx = [stem_x_left, stem_x_left + tsb, stem_x_left + tst, stem_x_left, stem_x_left]
    sy = [0, 0, Hs, Hs, 0]
    fig.add_trace(go.Scatter(x=sx, y=sy, fill="toself",
                             fillcolor="rgba(70,90,110,0.85)", line_color="#263238",
                             name="Stem"))
    # Toe soil
    fig.add_shape(type="rect", x0=0, x1=toe, y0=0, y1=sizing.get('total_height',0)-Hs-tf,
                  fillcolor="rgba(210,180,100,0.4)", line_width=0)
    # Heel soil
    fig.add_shape(type="rect", x0=toe+tsb, x1=B, y0=0, y1=Hs,
                  fillcolor="rgba(180,140,80,0.35)", line_width=0)

    # Dimensions
    fig.add_annotation(x=B/2, y=-tf-0.25, text=f"B = {B:.2f} m",
                       font=dict(size=12, color="#333"), showarrow=False)
    fig.add_annotation(x=-0.25, y=Hs/2, text=f"H<sub>s</sub> = {Hs:.2f} m",
                       font=dict(size=11, color="#333"), showarrow=False, textangle=-90)

    fig.update_layout(
        title=dict(text="Floodwall Cross-Section", font=dict(size=14, color="#1a3a5c")),
        xaxis=dict(title="Width (m)", showgrid=True, gridcolor="#eee"),
        yaxis=dict(title="Height (m)", showgrid=True, gridcolor="#eee", scaleanchor="x"),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.15),
        height=520, margin=dict(l=60, r=80, t=50, b=80)
    )
    return fig


def gauge_fos(value, limit, title):
    color = "#28a745" if value >= limit else "#dc3545"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        title={"text": title, "font": {"size": 13}},
        number={"suffix": "", "font": {"size": 20}},
        gauge={
            "axis": {"range": [0, max(value * 1.5, limit * 2)], "tickfont": {"size": 10}},
            "bar": {"color": color},
            "steps": [
                {"range": [0, limit], "color": "#fdd"},
                {"range": [limit, max(value * 1.5, limit * 2)], "color": "#dfd"},
            ],
            "threshold": {"line": {"color": "#c00", "width": 3}, "value": limit}
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=10))
    return fig


# =============================================================================
# STREAMLIT UI
# =============================================================================

st.markdown("""
<div class="main-header">
  <h1>🏗️  Floodwall Design Tool</h1>
  <p>IS 456:2000 · IS 3370:2009 · NBC 105:2020 · IS 1893:2016 · IRC 6:2017 · NBC 000:1994</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Design Parameters")
    with st.expander("📐 Site & Hydraulic", expanded=True):
        hfl = st.number_input("High Flood Level – HFL (m)", value=426.00, step=0.05, format="%.2f")
        gl = st.number_input("Ground Level (m)", value=420.00, step=0.05, format="%.2f")
        protected_wl = st.number_input(
            "Protected Side Water Level (m)", value=420.00, step=0.05, format="%.2f",
            help="Use the landside standing water level if any. Set equal to ground level for dry protected side."
        )
        fb = st.number_input("Freeboard (m)", value=0.50, min_value=0.1, max_value=2.0, step=0.05)
        disch = st.number_input("Design Discharge Q (m³/s)", value=8.5, min_value=0.1, step=0.5)
        rw = st.number_input("River Width (m)", value=35.0, min_value=1.0, step=1.0)
        sf = st.number_input("Lacey Silt Factor", value=1.76, min_value=0.5, max_value=5.0, step=0.01)
        D50 = st.number_input("D50 Grain Size (mm)", value=0.5, min_value=0.1, step=0.1)
        seep_relief = st.slider(
            "Drain/Cutoff Relief Factor", 0.0, 1.0, 0.75, 0.05,
            help="0.0 keeps full residual seepage head at the heel; 1.0 assumes full uplift relief."
        )

    with st.expander("🪨 Soil Properties", expanded=True):
        phi = st.slider("Friction Angle φ (°)", 20, 45, 35)
        gs = st.number_input("γ_soil (kN/m³)", value=18.0, min_value=14.0, max_value=22.0, step=0.5)
        mu = st.number_input("Coeff. of Friction μ", value=0.50, min_value=0.2, max_value=0.8, step=0.01)
        sbc = st.number_input("Safe Bearing Capacity (kN/m²)", value=200.0, min_value=50.0, max_value=600.0, step=10.0)
        df = st.number_input("Foundation Depth (m)", value=1.5, min_value=0.5, max_value=5.0, step=0.1)
        shear_key_mode = st.selectbox(
            "Shear Key Mode",
            ["Auto", "Disabled", "Enabled"],
            index=0,
            help=("Auto lets the solver try the wall without and with a shear key. "
                  "Enabled forces a shear key. Disabled prevents one.")
        )
        shear_key_setting = {"Auto": None, "Disabled": False, "Enabled": True}[shear_key_mode]
        sur = st.number_input("Surcharge Height (m) [IRC 6:2017]", value=1.2, min_value=0.0, max_value=3.0, step=0.1)

    with st.expander("🏛️ Material (IS 456:2000)"):
        fck = st.selectbox("Concrete Grade (fck)", [20, 25, 30, 35], index=1)
        fy = st.selectbox("Steel Grade (fy)", [415, 500, 550], index=1)

    with st.expander("🌍 Seismic (NBC 105:2020)"):
        Z = st.number_input("PGA (Z)", value=0.16, min_value=0.05, max_value=0.50, step=0.01)
        IF = st.number_input("Importance Factor (I)", value=1.5, min_value=1.0, max_value=2.0, step=0.1)
        sitef = st.number_input("Site Factor", value=1.5, min_value=1.0, max_value=2.0, step=0.1)
        respf = st.number_input("Response Factor", value=1.5, min_value=1.0, max_value=2.0, step=0.1)

    st.markdown("---")
    run_btn = st.button("🚀  Run Design", use_container_width=True, type="primary")

# =============================================================================
# MAIN COMPUTATION
# =============================================================================

if run_btn:
    with st.spinner("Running analysis…"):
        # Scour analysis
        scour = scour_analysis(disch, rw, sf, D50, user_foundation_depth=df)
        df_adj = scour['adopted_depth']

        engine = configure_design_engine(
            hfl, gl, protected_wl, df_adj, phi, gs, mu, sbc, fb, sur,
            seep_relief, Z, IF, sitef, respf, fck, fy
        )
        soil_p, seis_p, rcc_p = build_design_context(
            engine, hfl, gl, protected_wl, gs, mu, sbc, sur, seep_relief, fck, fy
        )
        assessment, sizing, rcc_result = run_design_case(
            engine, hfl, gl, soil_p, seis_p, rcc_p, df_adj, use_shear_key=shear_key_setting
        )

    # Store results
    st.session_state['sizing'] = sizing
    st.session_state['rcc'] = rcc_result
    st.session_state['scour'] = scour
    st.session_state['engine'] = engine
    st.session_state['assessment'] = assessment
    st.session_state['hfl'] = hfl
    st.session_state['gl'] = gl
    st.session_state['df_adj'] = df_adj
    st.session_state['shear_key_mode'] = shear_key_mode
    st.session_state['shear_key_setting'] = shear_key_setting
    st.success("✅ Design complete!")

# =============================================================================
# DISPLAY RESULTS
# =============================================================================

if 'sizing' in st.session_state:
    sizing = st.session_state['sizing']
    rcc = st.session_state['rcc']
    scour = st.session_state['scour']
    engine = st.session_state['engine']
    assessment = st.session_state['assessment']
    hfl_r = st.session_state['hfl']
    gl_r = st.session_state['gl']
    df_adj = st.session_state['df_adj']
    shear_key_mode = st.session_state.get('shear_key_mode', 'Auto')
    shear_key_setting = st.session_state.get('shear_key_setting', None)

    if sizing is None or not sizing.get('all_safe'):
        st.error(build_safe_geometry_error_message(shear_key_setting))
        st.stop()
    if sizing.get('rcc_all_safe') is False:
        st.warning("Geometry satisfies stability checks, but RCC detailing is still under review after the optimization limit.")

    # Top summary cards
    cols = st.columns(6)
    cards = [
        ("Wall Type", sizing['design_type'], ""),
        ("Total Height", f"{sizing['total_height']:.2f}", "m"),
        ("Base Width", f"{sizing['base_width']:.2f}", "m"),
        ("FOS OT", f"{sizing['FOS_overturning']:.2f}", "≥2.0"),
        ("FOS Sliding", f"{sizing['FOS_sliding']:.2f}", "≥1.5"),
        ("Shear Key", "YES" if sizing.get('needs_shear_key') else "NO", ""),
    ]
    for col, (label, val, unit) in zip(cols, cards):
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="label">{label}</div>
              <div class="value">{val}</div>
              <div class="unit">{unit}</div>
            </div>""", unsafe_allow_html=True)

    st.caption(
        f"Shear key mode: {shear_key_mode}. Final geometry is "
        f"{'with' if sizing.get('needs_shear_key') else 'without'} a shear key."
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📐 Sizing & Cross-Section",
        "🛡️ Stability Checks",
        "🔩 RCC Design",
        "💧 Scour Analysis",
        "📋 Batch Processing",
        "📄 Report / Export",
    ])

    # Tab 1: Sizing & Cross‑Section
    with tab1:
        st.markdown('<div class="section-title">Wall Geometry</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1.4])
        with col1:
            params = [
                ("Design Type", sizing['design_type'], ""),
                ("Total Height (H)", f"{sizing['total_height']:.3f}", "m"),
                ("Water Height", f"{sizing['water_height']:.3f}", "m"),
                ("Stem Height (Hs)", f"{sizing['stem_height']:.3f}", "m"),
                ("Base Width (B)", f"{sizing['base_width']:.3f}", "m"),
                ("Toe Length", f"{sizing['toe_length']:.3f}", "m"),
                ("Heel Length", f"{sizing['heel_length']:.3f}", "m"),
                ("Base Slab Thick.", f"{sizing['base_slab_thickness']*1000:.0f}", "mm"),
                ("Stem Thick. (base)", f"{sizing['stem_base']*1000:.0f}", "mm"),
                ("Stem Thick. (top)", f"{sizing['stem_top']*1000:.0f}", "mm"),
                ("Shear Key Mode", shear_key_mode, ""),
                ("Shear Key", "Provided" if sizing.get('needs_shear_key') else "Not provided", ""),
            ]
            if sizing.get('needs_shear_key'):
                params += [
                    ("Shear Key Depth", f"{sizing.get('shear_key_depth',0):.2f}", "m"),
                    ("Shear Key Width", f"{sizing.get('shear_key_width',0):.2f}", "m"),
                ]
            if sizing['design_type'] == 'COUNTERFORT':
                params += [
                    ("Counterfort Spacing", f"{sizing.get('counterfort_spacing',0):.2f}", "m"),
                    ("Counterfort Thickness", f"{sizing.get('counterfort_thickness',0):.2f}", "m"),
                ]
            df_geom = pd.DataFrame(params, columns=["Parameter", "Value", "Unit"])
            st.dataframe(df_geom, hide_index=True, use_container_width=True)

            st.markdown('<div class="section-title">Earth Pressure Coefficients</div>', unsafe_allow_html=True)
            ep_data = [
                ("Ka (Active – Rankine)", f"{engine.Ka:.4f}"),
                ("Kp (Passive – Rankine)", f"{engine.Kp:.4f}"),
                ("K0 (At-rest)", f"{engine.K0:.4f}"),
                ("KAE (Seismic – M-O)", f"{engine.KAE:.4f}"),
                ("ΔKAE (Increment)", f"{engine.delta_KAE:.4f}"),
                ("Cd_h", f"{engine.Cd_h:.4f}"),
                ("Cd_v", f"{engine.Cd_v:.4f}"),
            ]
            st.dataframe(pd.DataFrame(ep_data, columns=["Coefficient", "Value"]),
                         hide_index=True, use_container_width=True)

            st.markdown('<div class="section-title">Hydraulic Load Case</div>', unsafe_allow_html=True)
            hyd_df = pd.DataFrame([
                ["Flood Depth", f"{sizing.get('flood_depth', 0):.3f}", "m"],
                ["Protected Water Depth", f"{sizing.get('protected_depth', 0):.3f}", "m"],
                ["Residual Heel Head", f"{sizing.get('heel_uplift_head', 0):.3f}", "m"],
                ["Hydrostatic Force", f"{sizing.get('hydrostatic_force', 0):.2f}", "kN/m"],
                ["Uplift Force", f"{sizing.get('uplift_force', 0):.2f}", "kN/m"],
            ], columns=["Parameter", "Value", "Unit"])
            st.dataframe(hyd_df, hide_index=True, use_container_width=True)

        with col2:
            fig_wall = draw_wall_section(sizing, gl_r, hfl_r)
            st.plotly_chart(fig_wall, use_container_width=True)

    # Tab 2: Stability Checks
    with tab2:
        st.markdown('<div class="section-title">Stability Factor of Safety</div>', unsafe_allow_html=True)
        g1, g2, g3, g4 = st.columns(4)
        with g1:
            st.plotly_chart(gauge_fos(sizing['FOS_overturning'], 2.0, "FOS Overturning (Static)"), use_container_width=True)
        with g2:
            st.plotly_chart(gauge_fos(sizing['FOS_overturning_seismic'], 1.5, "FOS Overturning (Seismic)"), use_container_width=True)
        with g3:
            st.plotly_chart(gauge_fos(sizing['FOS_sliding'], 1.5, "FOS Sliding (Static)"), use_container_width=True)
        with g4:
            st.plotly_chart(gauge_fos(sizing['FOS_sliding_seismic'], 1.25, "FOS Sliding (Seismic)"), use_container_width=True)

        governing = sizing.get('governing_check')
        if governing:
            if governing in ("Sliding - Seismic", "Overturning - Seismic", "Bearing Pressure"):
                st.info(f"Optimized geometry is governed by {governing}. Static FOS may remain above the minimum while seismic and bearing checks stay safe.")
            else:
                st.caption(f"Governing optimization check: {governing}")

        st.markdown('<div class="section-title">Detailed Stability Results</div>', unsafe_allow_html=True)
        def badge(ok):
            return '<span class="pass-badge">PASS</span>' if ok else '<span class="fail-badge">FAIL</span>'
        stab_rows = [
            ["Overturning – Static", f"{sizing['FOS_overturning']:.3f}", "≥ 2.0", badge(sizing['ot_static_ok']), "IS 3370:2009"],
            ["Overturning – Seismic", f"{sizing['FOS_overturning_seismic']:.3f}", "≥ 1.5", badge(sizing['ot_seismic_ok']), "IS 1893:2016"],
            ["Sliding – Static", f"{sizing['FOS_sliding']:.3f}", "≥ 1.5", badge(sizing['sl_static_ok']), "IS 3370:2009"],
            ["Sliding – Seismic", f"{sizing['FOS_sliding_seismic']:.3f}", "≥ 1.25", badge(sizing['sl_seismic_ok']), "IS 1893:2016"],
            ["Bearing Pressure", f"{sizing['q_max']} kN/m²", f"≤ {sbc} kN/m²", badge(sizing['bearing_ok']), "IS 1904:1986"],
            ["Eccentricity (static)", f"{sizing['e_static']} m", f"≤ B/6 = {sizing['base_width']/6:.3f} m",
             badge(sizing['e_static'] <= sizing['base_width'] / 6), "IS 3370:2009"],
            ["Middle Third Rule", "OK" if sizing['middle_third_ok'] else "Not Satisfied",
             "Within B/6", badge(sizing['middle_third_ok']), "IS 3370:2009"],
        ]
        df_stab = pd.DataFrame(stab_rows, columns=["Check", "Value", "Requirement", "Status", "Code"])
        st.write(df_stab.to_html(escape=False, index=False), unsafe_allow_html=True)

        st.markdown('<div class="section-title">Force Summary (kN/m)</div>', unsafe_allow_html=True)
        fc1, fc2 = st.columns(2)
        with fc1:
            force_data = pd.DataFrame({
                "Force": [
                    "Net Vertical Load (W-U)", "Downward Dead + Soil", "Hydrostatic Force",
                    "Uplift Force", "Active Earth + Hydro (Pa_total)", "Pa Dynamic (Seismic)"
                ],
                "Value (kN/m)": [
                    sizing['W_total'], sizing.get('W_down', sizing['W_total']),
                    sizing.get('hydrostatic_force', 0), sizing.get('uplift_force', 0),
                    sizing['Pa_total'], sizing['Pa_dynamic']
                ],
            })
            st.dataframe(force_data, hide_index=True, use_container_width=True)
        with fc2:
            moment_data = pd.DataFrame({
                "Moment": ["Resisting (Mr)", "Overturning Static (Mo)", "Overturning Seismic (Mo_e)"],
                "Value (kNm/m)": [sizing['Mr_total'], sizing['Mo_static'], sizing['Mo_seismic']],
            })
            st.dataframe(moment_data, hide_index=True, use_container_width=True)

    # Tab 3: RCC Design
    with tab3:
        if rcc is None:
            st.warning("RCC design not available (sizing failed).")
        else:
            st.markdown('<div class="section-title">Stem Reinforcement Design</div>', unsafe_allow_html=True)
            rc1, rc2 = st.columns(2)
            with rc1:
                st.markdown("**Design Moments & Shear**")
                stem_df = pd.DataFrame([
                    ["Mu_stem (Factored)", f"{rcc['Mu_stem (kNm/m)']:.2f}", "kNm/m"],
                    ["Vu_stem (Factored)", f"{rcc['Vu_stem (kN/m)']:.2f}", "kN/m"],
                    ["M_service (Unfactored)", f"{rcc['Mu_stem (kNm/m)']/1.5:.2f}", "kNm/m"],
                ], columns=["Item", "Value", "Unit"])
                st.dataframe(stem_df, hide_index=True, use_container_width=True)

                st.markdown("**Reinforcement**")
                reo_df = pd.DataFrame([
                    ["Ast Required", f"{rcc['Ast_stem_req (mm2/m)']:.0f}", "mm²/m"],
                    ["Ast Provided", f"{rcc['Ast_stem_prov (mm2/m)']:.0f}", "mm²/m"],
                    ["Main Bar (Stem)", rcc['Stem Main Bar'], ""],
                    ["Dist. Bar (Stem)", rcc['Stem Dist Bar'], ""],
                    ["Curtailment (H/2)", rcc['Curtailment'], ""],
                ], columns=["Parameter", "Value", "Unit"])
                st.dataframe(reo_df, hide_index=True, use_container_width=True)

            with rc2:
                st.markdown("**Section Checks**")
                chk_df = pd.DataFrame([
                    ["Shear Check", rcc['Shear Check'], f"τv={rcc['tau_v (N/mm2)']:.3f} ≤ τc={rcc['tau_c (N/mm2)']:.3f} N/mm²"],
                    ["Crack Width", rcc['Crack Check'], f"{rcc['Crack Width (mm)']:.3f} mm (limit 0.2mm)"],
                    ["Dev. Length", rcc['Ld Check'], f"Ld={rcc['Ld (mm)']:.0f}mm, avail={rcc['Ld Available (mm)']:.0f}mm"],
                    ["Bearing", rcc['Bearing Check'], f"qmax={rcc['q_max (kN/m2)']:.2f} kN/m²"],
                ], columns=["Check", "Result", "Details"])
                st.dataframe(chk_df, hide_index=True, use_container_width=True)

                st.markdown("**Eccentricities**")
                ecc_df = pd.DataFrame([
                    ["e_static", f"{rcc['e_static (m)']:.3f} m"],
                    ["e_seismic", f"{rcc['e_seismic (m)']:.3f} m"],
                    ["B/6 limit", f"{sizing['base_width']/6:.3f} m"],
                ], columns=["Parameter", "Value"])
                st.dataframe(ecc_df, hide_index=True, use_container_width=True)

            st.markdown('<div class="section-title">Base Slab Design</div>', unsafe_allow_html=True)
            bc1, bc2 = st.columns(2)
            with bc1:
                base_df = pd.DataFrame([
                    ["M_heel (Factored)", f"{rcc['M_heel (kNm/m)']:.2f}", "kNm/m"],
                    ["M_toe (Factored)", f"{rcc['M_toe (kNm/m)']:.2f}", "kNm/m"],
                    ["Mu_base", f"{rcc['Mu_base (kNm/m)']:.2f}", "kNm/m"],
                    ["Ast Required", f"{rcc['Ast_base_req (mm2/m)']:.0f}", "mm²/m"],
                    ["Ast Provided", f"{rcc['Ast_base_prov (mm2/m)']:.0f}", "mm²/m"],
                    ["Main Bar", rcc['Base Main Bar'], ""],
                ], columns=["Parameter", "Value", "Unit"])
                st.dataframe(base_df, hide_index=True, use_container_width=True)
            with bc2:
                st.markdown("**Bearing Pressure Diagram**")
                bear_fig = go.Figure()
                B = sizing['base_width']
                qmax = rcc['q_max (kN/m2)']; qmin = rcc['q_min (kN/m2)']
                bear_fig.add_trace(go.Scatter(x=[0, B], y=[qmin, qmax],
                    fill='tozeroy', fillcolor='rgba(33,150,243,0.2)',
                    line=dict(color='#1565C0', width=2), name="Bearing Pressure"))
                bear_fig.add_hline(y=sbc, line_dash="dash", line_color="red",
                                   annotation_text=f"SBC={sbc} kN/m²")
                bear_fig.update_layout(
                    xaxis_title="Base Width (m)", yaxis_title="Pressure (kN/m²)",
                    height=220, margin=dict(l=40, r=20, t=20, b=40), plot_bgcolor="white")
                st.plotly_chart(bear_fig, use_container_width=True)

            if 'SK Bar' in rcc:
                st.markdown('<div class="section-title">Shear Key Design</div>', unsafe_allow_html=True)
                sk_data = [
                    ("Depth", f"{rcc['SK_Depth_m']:.2f}", "m"),
                    ("Width", f"{rcc['SK_Width_m']:.2f}", "m"),
                    ("Mu (Factored)", f"{rcc['Mu_SK (kNm/m)']:.2f}", "kNm/m"),
                    ("Vu (Factored)", f"{rcc['Vu_SK (kN/m)']:.2f}", "kN/m"),
                    ("Ast Required", f"{rcc['Ast_SK_req (mm2/m)']:.0f}", "mm²/m"),
                    ("Ast Provided", f"{rcc['Ast_SK_prov (mm2/m)']:.0f}", "mm²/m"),
                    ("Reinforcement", rcc['SK Bar'], ""),
                    ("Shear Check", rcc['SK Shear Check'], ""),
                ]
                df_sk = pd.DataFrame(sk_data, columns=["Parameter", "Value", "Unit"])
                st.dataframe(df_sk, hide_index=True, use_container_width=True)

    # Tab 4: Scour Analysis
    with tab4:
        st.markdown('<div class="section-title">Scour Depth Analysis</div>', unsafe_allow_html=True)
        s1, s2 = st.columns(2)
        with s1:
            scour_info = [
                ("Design Discharge (Q)", f"{disch:.2f}", "m³/s"),
                ("Channel Width", f"{rw:.1f}", "m"),
                ("Unit Discharge (q)", f"{scour['q_unit']:.3f}", "m³/s/m"),
                ("Channel Classification", scour['channel_class'], ""),
                ("Lacey Regime Depth", f"{scour['regime_depth']:.2f}", "m"),
            ]
            si_df = pd.DataFrame(scour_info, columns=["Parameter", "Value", "Unit"])
            st.dataframe(si_df, hide_index=True, use_container_width=True)

            meth_df = pd.DataFrame([
                ["Method 1 – Degradation + Margin", f"{scour['methods']['degradation']:.2f}"],
                ["Method 2 – Modified Lacey", f"{scour['methods']['lacey_modified']:.2f}"],
                ["Method 3 – USACE Guidelines", f"{scour['methods']['usace']:.2f}"],
                ["Method 4 – Practical Minimum", f"{scour['methods']['practical']:.2f}"],
                ["➤ Required Depth (governing)", f"{scour['required_depth']:.1f}"],
                ["➤ Adopted Depth", f"{scour['adopted_depth']:.1f}"],
            ], columns=["Scour Method", "Depth (m)"])
            st.dataframe(meth_df, hide_index=True, use_container_width=True)

        with s2:
            methods_chart = go.Figure(go.Bar(
                x=["Method 1\nDegradation", "Method 2\nLacey Mod.", "Method 3\nUSACE", "Method 4\nPractical"],
                y=[scour['methods']['degradation'], scour['methods']['lacey_modified'],
                   scour['methods']['usace'], scour['methods']['practical']],
                marker_color=["#1565C0", "#1976D2", "#2196F3", "#64B5F6"],
                text=[f"{v:.2f} m" for v in [scour['methods']['degradation'],
                      scour['methods']['lacey_modified'], scour['methods']['usace'],
                      scour['methods']['practical']]],
                textposition="outside",
            ))
            methods_chart.add_hline(y=scour['required_depth'], line_dash="dash",
                                    line_color="red", annotation_text="Governing Depth")
            methods_chart.update_layout(title="Scour Depth by Method", yaxis_title="Depth (m)",
                height=280, margin=dict(l=40, r=20, t=40, b=60), plot_bgcolor="white")
            st.plotly_chart(methods_chart, use_container_width=True)

        st.markdown('<div class="section-title">Scour Protection Recommendations</div>', unsafe_allow_html=True)
        prot = scour['protection']
        p1, p2, p3, p4, p5 = st.columns(5)
        for col, (lbl, val) in zip([p1, p2, p3, p4, p5], [
            ("Riprap Width", f"{prot['riprap_width']} m"),
            ("Riprap Depth", f"{prot['riprap_depth']} m"),
            ("Stone Size", prot['stone_size']),
            ("Geotextile", prot['geotextile'].capitalize()),
            ("Toe Extension", f"{prot['toe_extension']} m"),
        ]):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="label">{lbl}</div>
                  <div class="value" style="font-size:1.1rem">{val}</div>
                </div>""", unsafe_allow_html=True)

    # Tab 5: Batch Processing
    with tab5:
        st.markdown('<div class="section-title">Batch Processing – Multiple Cross-Sections</div>', unsafe_allow_html=True)

        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.markdown("**Upload Excel/CSV** with columns: `Station`, `HFL`, `Ground_Level`")
            if st.button("📥 Download Sample Template"):
                sample = pd.DataFrame({
                    'Station':      ['0+000','0+050','0+100','0+150','0+200'],
                    'HFL':          [425.50, 425.75, 426.00, 426.25, 426.50],
                    'Ground_Level': [420.00, 420.30, 420.60, 420.90, 421.20],
                })
                buf = io.BytesIO()
                sample.to_excel(buf, index=False, engine='openpyxl')
                st.download_button("⬇️ Sample Excel", buf.getvalue(),
                                   "floodwall_template.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            uploaded = st.file_uploader("Upload Input File", type=["xlsx","csv"])

        with col_b:
            if uploaded:
                try:
                    df_in = (pd.read_csv(uploaded) if uploaded.name.endswith('.csv')
                             else pd.read_excel(uploaded))
                    required_cols = {'Station', 'HFL', 'Ground_Level'}
                    missing = required_cols - set(df_in.columns)
                    if missing:
                        st.error(f"Missing columns in file: {missing}. Required: Station, HFL, Ground_Level")
                    else:
                        st.info(f"Loaded {len(df_in)} sections")
                        st.dataframe(df_in.head(10), use_container_width=True)
                        if st.button("▶️  Run Batch Design", type="primary"):
                            # You may need to adapt the following lines to your app's variable names
                            batch_results = []
                            prog = st.progress(0)
                            scour_batch = scour_analysis(disch, rw, sf, D50, user_foundation_depth=df)
                            for i, row in df_in.iterrows():
                                hfl_b = float(row['HFL'])
                                gl_b  = float(row['Ground_Level'])
                                eng2 = configure_design_engine(
                                    hfl_b, gl_b, protected_wl, scour_batch['adopted_depth'],
                                    phi, gs, mu, sbc, fb, sur, seep_relief,
                                    Z, IF, sitef, respf, fck, fy
                                )
                                soil_p2, seis_p2, rcc_p2 = build_design_context(
                                    eng2, hfl_b, gl_b, protected_wl, gs, mu, sbc, sur,
                                    seep_relief, fck, fy
                                )
                                _, sz, _ = run_design_case(
                                    eng2, hfl_b, gl_b, soil_p2, seis_p2, rcc_p2,
                                    scour_batch['adopted_depth'], use_shear_key=shear_key_setting
                                )
                                r = summarize_batch_result(row['Station'], hfl_b, gl_b, eng2, sz)
                                batch_results.append(r)
                                prog.progress((i + 1) / len(df_in))

                            df_batch = pd.DataFrame(batch_results)
                            n_safe = len(df_batch[df_batch['Status'] == 'SAFE'])
                            st.success(f"✅ Batch complete: {n_safe} safe / {len(df_batch)} total sections")
                            st.dataframe(df_batch, use_container_width=True)

                            # Export to Excel
                            buf2 = io.BytesIO()
                            with pd.ExcelWriter(buf2, engine='openpyxl') as wr:
                                df_batch.to_excel(wr, sheet_name='Batch_Results', index=False)
                            st.download_button(
                                "⬇️ Download Batch Results (Excel)", buf2.getvalue(),
                                f"batch_floodwall_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True)

                            # Summary charts (optional, as in floodwallc-app.py)
                            df_s = df_batch[df_batch['Status'] == 'SAFE']
                            if len(df_s) > 0 and 'Base_Width_m' in df_s.columns:
                                fig_batch = make_subplots(rows=1, cols=2,
                                    subplot_titles=("Base Width vs Station", "FOS vs Station"))
                                fig_batch.add_trace(go.Bar(
                                    x=df_s['Station'].astype(str), y=df_s['Base_Width_m'],
                                    name="Base Width (m)", marker_color="#1565C0"), 1, 1)
                                fig_batch.add_trace(go.Scatter(
                                    x=df_s['Station'].astype(str), y=df_s['FOS_OT'],
                                    name="FOS OT", line=dict(color="#e65100", width=2),
                                    mode="lines+markers"), 1, 2)
                                fig_batch.add_trace(go.Scatter(
                                    x=df_s['Station'].astype(str), y=df_s['FOS_SL'],
                                    name="FOS SL", line=dict(color="#2e7d32", width=2),
                                    mode="lines+markers"), 1, 2)
                                fig_batch.add_hline(y=2.0, line_dash="dash", line_color="red",
                                    annotation_text="OT min=2.0", row=1, col=2)
                                fig_batch.add_hline(y=1.5, line_dash="dot", line_color="orange",
                                    annotation_text="SL min=1.5", row=1, col=2)
                                fig_batch.update_layout(height=380, plot_bgcolor="white")
                                st.plotly_chart(fig_batch, use_container_width=True)

                except Exception as e:
                    import traceback
                    st.error(f"Error processing file: {e}")

