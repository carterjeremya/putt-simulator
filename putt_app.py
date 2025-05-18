import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from putt_trajectory import simulate_putt, converge_on_aim_and_speed, compute_mu_from_stimp, plot_putt_trajectory

st.set_page_config(page_title="Putt Simulator", layout="centered")

st.title("Putt Simulator")

st.markdown("""
### ℹ️ How to Use This Golf Putt Simulator

**Inputs:**
- **Distance to Hole**: Straight-line distance from ball to cup, in feet.
- **Start Angle**: Direction around the hole (0° = directly right, 90° = directly above, -90° = directly below).
- **Stimp**: Green speed from the stimpmeter. Higher = faster greens (default is 10).
- **Slope**: Vertical slope of the green. Positive slope is uphill

**How to Use:**
1. Adjust sliders to set the starting position and green conditions.
2. The simulator computes the optimal aim angle and initial speed needed to hole the putt.
3. The plot shows the ball's path, your visual aim line, and the hole.

**Interpretation:**
- The **Visual Aim Cue** tells you how far **above/below** the hole to aim along the fall line
- **Entry Speed** shows how fast the ball would be moving as it enters the hole.
- **Equivalent Flat Putt Distance** how far the ball would roll on a flat green.

Putts are optimized to finish approximately two feet beyond the hole and enter on the high side of the cup.
""")


# --- Inputs ---
distance = st.slider("Distance to hole (ft)", 3, 30, 10, step=1)
angle = st.slider("Starting angle (deg)", -180, 180, 0, step=5)
stimp = st.slider("Green speed (Stimp)", 8, 13, 10, step=1)
slope_y = st.slider("Slope (rise/run)", 0.0, 0.04, 0.01, step=0.005, format="%.3f")

x0 = distance * np.cos(np.radians(angle))
y0 = distance * np.sin(np.radians(angle))
g = 32.174
mu = compute_mu_from_stimp(stimp, g)
aim_angle = (angle + 180) % 360
v0_init = np.sqrt(2 * g * (mu - np.sin(np.radians(angle)) * slope_y) * (distance + 2.0))

aim_angle, v0, entry_speed, sol = converge_on_aim_and_speed(
    x0, y0, stimp, slope_y, v0_init, aim_angle,
    entry_speed_target=1.5, entry_speed_tol=0.3, g=g
)

xf, yf = sol.y[0, -1], sol.y[1, -1]
d_equiv = 0.5 * v0**2 / (g * mu)
aim_distance = 0.5 * v0**2 / (g * mu)

aim_rad = np.radians(aim_angle)
aim_x = x0 + aim_distance * np.cos(aim_rad)
aim_y = y0 + aim_distance * np.sin(aim_rad)

# Compute slope (m) and intercept (b) of aim line: y = m*x + b
dx = aim_x - x0
dy = aim_y - y0

# Normalize angle to [0, 360)
normalized_angle = aim_angle % 360

if np.isclose(normalized_angle, 90, atol=0.1) or np.isclose(normalized_angle, 270, atol=0.1):
    st.write("**Visual Aim Cue:** Aim directly at the center of the hole.")
else:
    m = dy / dx
    b = y0 - m * x0  # y-intercept at x = 0

    offset_in = b * 12
    dir_str = "above" if offset_in > 0 else "below"
    
    if offset_in < 0.177 * 12:
        st.write(f"**Visual Aim Cue:** Aim inside the hole, {abs(offset_in):.1f} inches above center.")
    else:
        st.write(
            f"**Visual Aim Cue:** Aim so your line crosses the fall line "
            f"{abs(offset_in - 0.177 * 12 / 2):.1f} inches {dir_str} the hole."
        )


st.write(f"**Equivalent flat putt distance:** {d_equiv:.2f} ft")
st.write(f"**Initial Speed:** {v0:.2f} ft/s")
st.write(f"**Entry Speed at Hole:** {entry_speed:.2f} ft/s")
st.write(f"**Final Position**: ({xf:.2f}, {yf:.2f}) ft")

# --- Plot ---
fig = plot_putt_trajectory(sol, x0, y0, aim_angle, v0, stimp, g)
st.pyplot(fig)
