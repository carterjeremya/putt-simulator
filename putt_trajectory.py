import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def compute_mu_from_stimp(stimp, g=32.174):
    return 36 / (2 * g * stimp)

def putt_dynamics(t, state, g, mu, slope_y):
    x, y, vx, vy = state
    v_mag = np.hypot(vx, vy)
    if v_mag < 1e-2:
        return [0, 0, 0, 0]
    ax = -g * mu * vx / v_mag
    ay = -g * slope_y - g * mu * vy / v_mag
    return [vx, vy, ax, ay]

def stop_event(t, state, g, mu, slope_y):
    return np.hypot(state[2], state[3]) - 0.01
stop_event.terminal = True
stop_event.direction = -1

def simulate_putt(x0, y0, v0, aim_angle_deg, stimp, slope_y, g=32.174, t_max=10):
    mu = compute_mu_from_stimp(stimp, g)
    aim_rad = np.radians(aim_angle_deg)
    vx0, vy0 = v0 * np.cos(aim_rad), v0 * np.sin(aim_rad)
    y_init = [x0, y0, vx0, vy0]

    return solve_ivp(
        putt_dynamics, [0, t_max], y_init,
        args=(g, mu, slope_y),
        t_eval=np.linspace(0, t_max, 1000),
        events=stop_event,
        rtol=1e-3, atol=1e-5
    )

def is_high_side_entry(x, y, closest_idx, slope_y, ball_radius=0.07):
    slope_vec = np.array([0, slope_y])
    ball_vec = np.array([x[closest_idx], y[closest_idx]])
    cross = np.cross(slope_vec, ball_vec)
    offset = np.abs(cross / np.linalg.norm(slope_vec))
    return np.sign(cross) == np.sign(slope_y) and offset >= ball_radius

def converge_on_aim_and_speed(x0, y0, stimp, slope_y,
    v0_init, aim_angle_init,
    entry_speed_target=1.75, entry_speed_tol=0.25,
    hole_radius=0.177, max_iter=500,
    aim_step=0.20, speed_step=0.50,
    g=32.174):

    v0, aim_angle = v0_init, aim_angle_init
    best_sol = None
    entry_speed = None
    hole_distance = np.hypot(x0, y0)

    for _ in range(max_iter):
        sol = simulate_putt(x0, y0, v0, aim_angle, stimp, slope_y, g=g)
        x, y = sol.y[0], sol.y[1]
        vx, vy = sol.y[2], sol.y[3]
        rollout = np.sum(np.hypot(np.diff(x), np.diff(y)))

        if rollout <= hole_distance + 1.5:
            v0 += speed_step
            continue

        dists = np.hypot(x, y)
        closest_idx = np.argmin(dists)
        closest_dist = dists[closest_idx]
        entry_speed = np.hypot(vx[closest_idx], vy[closest_idx])

        if closest_dist <= hole_radius and y[closest_idx] > 0:
            if rollout > hole_distance + 1.5:
                v0 -= speed_step
        else:
            # Missed left or right — adjust aim
            cross = np.cross([-x0, -y0], [x[closest_idx], y[closest_idx]])
            aim_angle += -aim_step if cross > 0 else aim_step

        best_sol = sol

    return aim_angle, v0, entry_speed, best_sol  # ❌ Fallthrough

def plot_putt_trajectory(sol, x0, y0, aim_angle, v0, stimp, b=None, g=32.174, hole_radius=0.177):
    mu = compute_mu_from_stimp(stimp, g)
    aim_distance = 0.5 * v0**2 / (g * mu)
    aim_rad = np.radians(aim_angle)
    aim_x, aim_y = x0 + aim_distance * np.cos(aim_rad), y0 + aim_distance * np.sin(aim_rad)

    x, y = sol.y[0], sol.y[1]
    fig, ax = plt.subplots(figsize=(5, 5))  # Make sure this is returned
    ax.plot(x, y, lw=1, label='Ball path')
    ax.scatter([x[0]], [y[0]], s=10, c='black', label='Start')
    
    if b is not None:
        ax.scatter(0, b, s=10, c='black', marker='x', label='Visual Cue')
        
    ax.scatter([aim_x], [aim_y], s=10, c='blue', label='Aim Point')
    ax.plot([x0, aim_x], [y0, aim_y], 'b--', lw=1, label='Aim Line')
    ax.add_patch(plt.Circle((0, 0), hole_radius, color='red', alpha=0.3, label='Hole'))
    ax.axhline(0, color='gray', ls='--', lw=0.5)
    ax.axvline(0, color='gray', ls='--', lw=0.5)

    extent = max(np.max(np.abs(x)), np.max(np.abs(y)), 1) + 2
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect('equal')
    ax.set_title('Optimized Putt Trajectory')
    ax.set_xlabel('Left/Right (ft)')
    ax.set_ylabel('Up/Down (ft)')
    ax.grid(True)
    ax.legend()
    return fig

# --- Example run ---
if __name__ == "__main__":
    #### Inputs
    # d0:    distance from hole (ft)
    # a0:    inital angle around hole (+ is ccw, 0 is at 3 o'clock)
    # stimp: green speed
    # slope: green slope
    d0, a0, stimp, slope_y = 30, -90, 10, 0.01
    
    g = 32.174
    x0, y0 = d0 * np.cos(np.radians(a0)), d0 * np.sin(np.radians(a0))
    mu = compute_mu_from_stimp(stimp, g)

    aim_angle = (a0 + 180) % 360
    v0_init = np.sqrt(2 * g * (mu - np.sin(np.radians(a0)) * slope_y) * (d0 + 2.0))

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
        print("**Visual Aim Cue:** Aim directly at the center of the hole.")
        b = None
        
    else:
        m = dy / dx
        b = y0 - m * x0  # y-intercept at x = 0
    
        offset_in = b * 12
        dir_str = "above" if offset_in > 0 else "below"
        
        if offset_in < 0.177 * 12:
            print(f"**Visual Aim Cue:** Aim inside the hole, {abs(offset_in):.1f} inches above center.")
        else:
            print(
                f"**Visual Aim Cue:** Aim so your line crosses the fall line "
                f"{abs(offset_in - 0.177 * 12 / 2):.1f} inches {dir_str} the hole."
            )
    
    
    print(f"**Equivalent flat putt distance:** {d_equiv:.2f} ft")
    print(f"**Initial Speed:** {v0:.2f} ft/s")
    print(f"**Entry Speed at Hole:** {entry_speed:.2f} ft/s")
    print(f"**Final Position**: ({xf:.2f}, {yf:.2f}) ft")
    
    # --- Plot ---
    fig = plot_putt_trajectory(sol, x0, y0, aim_angle, v0, stimp, b)

    plt.show(fig)