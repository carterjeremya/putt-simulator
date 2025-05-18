import streamlit as st
import numpy as np
from putt_trajectory import simulate_putt, converge_on_aim_and_speed, compute_mu_from_stimp, plot_putt_trajectory

_ = simulate_putt(10, 0, 6.0, 180, 10, 0.01)

st.set_page_config(page_title="Putt Simulator", layout="centered")

st.title("Putt Simulator")

st.markdown("""
### ℹ️ How to Use This Putt Simulator

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
- The **Visual Aim Cue** tells you how far **above** the hole to aim along the **fall line**
- **Entry Speed** shows how fast the ball would be moving as it enters the hole.
- **Equivalent Flat Putt Distance** how far the ball would roll on a flat green.
- **Final Position** shows approximately where the putt will finish.

**Putts are optimized to finish approximately 2-3 feet beyond the hole.**
""")

# --- Inputs ---
st.markdown("### Inputs")
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

b = 0
st.markdown("### ✅ Results")


if np.isclose(normalized_angle, 90, atol=0.1) or np.isclose(normalized_angle, 270, atol=0.1):
    st.write("**Visual Aim Cue:** Aim directly at the center of the hole.")
    
else:
    m = dy / dx
    b = y0 - m * x0  # y-intercept at x = 0

    offset_in = b * 12
    dir_str = "above" if offset_in > 0 else "below"
    
    if offset_in < 0.177 * 12:
        st.write(f"### Results\n**Visual Aim Cue:** Aim inside the hole, {abs(offset_in):.1f} inches above center.")
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
fig = plot_putt_trajectory(sol, x0, y0, aim_angle, v0, stimp, b)
st.pyplot(fig)

st.markdown("""
---

### 🏌️‍♂️ Suggested Practice Drills

#### 🎯 1. Dial In Distance Control on Flat Putts
- Find a **flat section** of the practice green.
- Mark putts at **5, 10, 15, and 20 feet**.
- Focus on consistent **stroke length and tempo**.
- Goal: Ball should finish **12–18 inches past the hole**.

#### 🧭 2. Train Your Eyes: Identify the Fall Line
- Find a hole with **visible slope**.
- Walk around the hole to estimate the **fall line** (true downhill).
- Putt from that point **straight at the hole** — does it go in?
- Adjust until you find the **true straight putt** line.

#### 🎯 3. Set a Visual Aim Point Above the Hole
- Choose a putt that breaks (e.g. left-to-right).
- Estimate a **visual target above the hole** along the fall line.
- Place a **ball marker or tee** at that point.
- Putt with pace to finish **~2 feet past the hole**, aiming at the marker.

#### 🔁 4. Combine Read + Pace Training
- Pick 3–4 putts from different angles.
- For each:
  - Read the break
  - Set your aim point
  - Match your **speed** to your **line**
- Track how often you finish inside a **6-inch “capture zone”** around the cup.

💡 *Pro tip: Use this simulator to plan your reads, then go test them on the green.*

---

#### ⚠️ Disclaimer

This tool is intended for **educational and entertainment purposes only**. It is **not approved for use during competition** or for **posting scores to an official USGA handicap index**.

Use of this tool during sanctioned play may violate **Rule 4.3a** of the USGA Rules of Golf.  

This software does **not constitute golf instruction or professional advice**.

Actual putting conditions may vary due to factors such as grain, moisture, imperfections, or course-specific slope characteristics.  

Always consult with a qualified instructor or rules official and follow all **Rules of Golf** and local tournament regulations.
""")

st.markdown("""
---
© 2024 Jeremy Carter. All rights reserved.
""")
