# Biomechanics GUI — Mathematical Equations

This document describes every derived quantity computed by the application.

---

## 1. Input Data

| Source File | Contents |
|---|---|
| `*_px_person00.trc` | Marker positions in **pixel space** $(x_{px}, y_{px})$ per frame |
| `*_m_person00.trc` | Marker positions in **meter space** $(x, y)$ per frame |
| `*_angles_person00.mot` | Joint angles $\theta$ (degrees) per frame, computed by Sports2D |

All files share a common time array $t$ sampled at 250 fps. The constant time step is:

$$\Delta t = \text{mean}(\Delta t_i) = \frac{1}{250} = 0.004 \text{ s}$$

---

## 2. Linear Kinematics (per marker, meter space)

### 2.1 Velocity

Computed using NumPy's `np.gradient`, which uses **central differences** for interior points and **forward/backward differences** at boundaries:

$$v_x(t) = \frac{dx}{dt} \approx \frac{x(t + \Delta t) - x(t - \Delta t)}{2 \Delta t}$$

$$v_y(t) = \frac{dy}{dt} \approx \frac{y(t + \Delta t) - y(t - \Delta t)}{2 \Delta t}$$

### 2.2 Total Speed (magnitude of velocity vector)

$$\text{speed}(t) = \| \mathbf{v}(t) \| = \sqrt{v_x^2 + v_y^2}$$

### 2.3 Acceleration

Second derivative, computed as the gradient of velocity:

$$a_x(t) = \frac{dv_x}{dt} \approx \frac{v_x(t + \Delta t) - v_x(t - \Delta t)}{2 \Delta t}$$

$$a_y(t) = \frac{dv_y}{dt} \approx \frac{v_y(t + \Delta t) - v_y(t - \Delta t)}{2 \Delta t}$$

### 2.4 Total Acceleration (magnitude)

$$\text{accel}(t) = \| \mathbf{a}(t) \| = \sqrt{a_x^2 + a_y^2}$$

### 2.5 Relative Position (to Hip)

For any marker $M$ with position $(x_M, y_M)$ and the Hip marker at $(x_H, y_H)$:

$$x_{\text{rel}}(t) = x_M(t) - x_H(t)$$

$$y_{\text{rel}}(t) = y_M(t) - y_H(t)$$

This isolates limb motion from whole-body translation.

---

## 3. Angular Kinematics (per joint)

### 3.1 Joint Angle (from .mot file)

The raw angle $\theta$ is provided by Sports2D. For display as an **interior joint angle** $\theta_d \in [0°, 180°]$:

$$\theta_d = 180° - \theta$$

$$\theta_d = \theta_d \mod 360°$$

$$\theta_d = \begin{cases} 360° - \theta_d & \text{if } \theta_d > 180° \\ \theta_d & \text{otherwise} \end{cases}$$

### 3.2 Angular Velocity

Computed from the **raw** angle $\theta$ (not the display angle) to preserve sign (rotation direction):

$$\omega(t) = \frac{d\theta}{dt} \approx \frac{\theta(t + \Delta t) - \theta(t - \Delta t)}{2 \Delta t} \quad [\text{deg/s}]$$

### 3.3 Angular Acceleration

$$\alpha(t) = \frac{d\omega}{dt} \approx \frac{\omega(t + \Delta t) - \omega(t - \Delta t)}{2 \Delta t} \quad [\text{deg/s}^2]$$

> **Note on wrists:** Sports2D reports $\theta = 0$ for all frames at the wrist joints because there are no hand/finger markers to define a wrist angle. When all angular data is zero, the angular velocity and acceleration subplots are automatically hidden.

---

## 4. Visual Angle Arc (on-screen overlay)

When a joint is selected, the angle arc is computed **geometrically** from the current frame's marker positions, independent of the `.mot` file:

Given three markers — Parent $P$, Vertex $V$ (the joint), Child $C$:

$$\vec{u}_1 = P - V, \quad \vec{u}_2 = C - V$$

The angle of each limb vector from the positive x-axis (with y-axis flipped for screen coordinates):

$$\alpha_1 = \text{atan2}(-(P_y - V_y),\ P_x - V_x)$$

$$\alpha_2 = \text{atan2}(-(C_y - V_y),\ C_x - V_x)$$

The arc sweep is the angular span between the two limb directions:

$$\text{start} = \min(\alpha_1, \alpha_2), \quad \text{end} = \max(\alpha_1, \alpha_2)$$

If $\text{end} - \text{start} > 180°$, the arc is drawn the "short way around":

$$\text{start}, \text{end} = \text{end}, \text{start} + 360°$$

The displayed angle value is:

$$\theta_{\text{display}} = (\text{end} - \text{start}) \mod 360°$$

This ensures the number shown always matches the visual arc drawn on screen.

---

## 5. Trajectory Drawing

### 5.1 Absolute Trajectory

The full path of a selected marker from frame 0 to the current frame, drawn using the **pixel-space** positions $(x_{px}, y_{px})$ as a polyline via `cv2.polylines`. Points with $(x, y) = (0, 0)$ (missing detections) are filtered out.

### 5.2 Relative Trajectory

The marker's path relative to the Hip, visualized at the Hip's current position:

$$\text{draw}_x(t) = x_{H,\text{now}} + (x_M(t) - x_H(t))$$

$$\text{draw}_y(t) = y_{H,\text{now}} + (y_M(t) - y_H(t))$$

This shows the limb's motion pattern with whole-body translation removed.

---

## 6. Numerical Method: `np.gradient`

All derivatives use NumPy's `gradient` function with uniform spacing $\Delta t$:

| Location | Formula | Order |
|---|---|---|
| Interior points | $f'(t) = \frac{f(t+\Delta t) - f(t-\Delta t)}{2\Delta t}$ | 2nd order central |
| First point | $f'(t_0) = \frac{f(t_1) - f(t_0)}{\Delta t}$ | 1st order forward |
| Last point | $f'(t_n) = \frac{f(t_n) - f(t_{n-1})}{\Delta t}$ | 1st order backward |

This provides smooth derivatives without requiring explicit smoothing filters, though the 250 Hz sampling rate means high-frequency noise may be amplified in the acceleration (second derivative).

---

## 7. Circle Fitting (Kåsa Algebraic Method)

Used to find the best-fit circle through a set of 2D trajectory points, e.g. the wrist's relative position during a circular arm swing.

### 7.1 Problem Setup

Given $n$ points $(x_i, y_i)$, find the center $(c_x, c_y)$ and radius $r$ of the circle that minimizes the algebraic distance.

A point on a circle satisfies:

$$(x - c_x)^2 + (y - c_y)^2 = r^2$$

Expanding:

$$x^2 + y^2 = 2c_x x + 2c_y y + (r^2 - c_x^2 - c_y^2)$$

### 7.2 Least-Squares Formulation

Let $d = r^2 - c_x^2 - c_y^2$. This becomes a linear system $A\mathbf{p} = \mathbf{b}$:

$$\begin{bmatrix} 2x_1 & 2y_1 & 1 \\ 2x_2 & 2y_2 & 1 \\ \vdots & \vdots & \vdots \\ 2x_n & 2y_n & 1 \end{bmatrix} \begin{bmatrix} c_x \\ c_y \\ d \end{bmatrix} = \begin{bmatrix} x_1^2 + y_1^2 \\ x_2^2 + y_2^2 \\ \vdots \\ x_n^2 + y_n^2 \end{bmatrix}$$

Solved via `np.linalg.lstsq` (SVD-based least squares).

### 7.3 Recovering the Radius

$$r = \sqrt{d + c_x^2 + c_y^2}$$

### 7.4 Fit Quality (Residuals)

For each point, the radial residual is:

$$e_i = \sqrt{(x_i - c_x)^2 + (y_i - c_y)^2} - r$$

A positive $e_i$ means the point is outside the fitted circle; negative means inside.
