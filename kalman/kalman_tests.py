import numpy as np
import kalman
import matplotlib.pyplot as plt

R_est = np.zeros((4,4))
R_est[0,0] = 1.0
R_est[1,1] = 0.01
R_est[2,2] = 1.0
R_est[3,3] = 0.01
R_est = R_est * 1.0
R_actual = R_est * 1.0
dt = 0.05

# Stationary test with perfect data
def test1():
    print("Test1")
    L = 1001
    l_intercept = np.random.randint(20, 300)
    l_angle = -1.0 * np.random.random()
    r_intercept = np.random.randint(340, 620)
    r_angle = np.random.random()
    print("Initial = ", (l_intercept, l_angle, r_intercept, r_angle))
    measurement = np.array([l_intercept, l_angle, r_intercept, r_angle])
    kalman.kalman_init(measurement, R_est)
    t = 1
    kalman_estimates = [measurement]
    measurements = [measurement]
    actual_values = [measurement]
    for _ in range(L-1):
        new_results = kalman.kalman(measurement, dt)
        actual = measurement
        measurements.append(measurement)
        kalman_estimates.append(new_results)
        actual_values.append(actual)
        if t % 100 == 0:
            print("t = ", t, new_results)
        t += 1
    return np.array(kalman_estimates), np.array(measurements), np.array(actual_values)

# Initially wrong measurement, but all correct after
def test2():
    print("Test2")
    L = 101
    l_intercept = np.random.randint(20, 300)
    l_angle = -1.0 * np.random.random()
    r_intercept = np.random.randint(340, 620)
    r_angle = np.random.random()
    measurement = np.array([l_intercept, l_angle, r_intercept, r_angle])
    init_measurement = np.array([100, -3, 500, 2])
    print("Real = ", measurement)
    print("Initial = ", init_measurement)
    kalman.kalman_init(init_measurement, R_est)
    t = 1
    kalman_estimates = [init_measurement]
    measurements = [init_measurement]
    actual_values = [measurement]
    for _ in range(L-1):
        new_results = kalman.kalman(measurement, dt)
        actual = measurement
        measurements = [measurement]
        kalman_estimates.append(new_results)
        actual_values.append(actual)
        if t % 10 == 0:
            print("t = ", t, new_results)
        t += 1
    return np.array(kalman_estimates), np.array(measurements), np.array(actual_values)

# Random noise
def test3():
    def noisy(m, R):
        l_i, l_a, r_i, r_a = m
        l_i = np.random.normal(l_i, R[0,0])
        l_a = np.random.normal(l_a, R[1,1])
        r_i = np.random.normal(r_i, R[2,2])
        r_a = np.random.normal(r_a, R[3,3])
        return np.array([l_i, l_a, r_i, r_a])

    print("Test3")
    L = 101
    l_intercept = np.random.randint(20, 300)
    l_angle = -1.0 * np.random.random()
    r_intercept = np.random.randint(340, 620)
    r_angle = np.random.random()
    actual = np.array([l_intercept, l_angle, r_intercept, r_angle])
    print("Real = ", actual)
    noisy_start = noisy(actual, R_actual)
    kalman.kalman_init(noisy_start, R_est)
    t = 1
    kalman_estimates = [noisy_start]
    measurements = [noisy_start]
    actual_values = [actual]
    for _ in range(L-1):
        noisy_measurement = noisy(actual, R_actual)
        new_results = kalman.kalman(noisy_measurement, dt)
        kalman_estimates.append(new_results)
        actual_values.append(actual)
        measurements.append(noisy_measurement)
        if t % 10 == 0:
            print("t = ", t, new_results)
        t += 1
    return np.array(kalman_estimates), np.array(measurements), np.array(actual_values)

# Random noise, with occassionally really bad samples
def test4():
    def noisy(m, R):
        l_i, l_a, r_i, r_a = m
        l_i = np.random.normal(l_i, R[0,0])
        l_a = np.random.normal(l_a, R[1,1])
        r_i = np.random.normal(r_i, R[2,2])
        r_a = np.random.normal(r_a, R[3,3])
        return np.array([l_i, l_a, r_i, r_a])

    def bad_measurement():
        l_i = np.random.randint(20, 300)
        l_a = -1.0 * np.random.random()
        r_i = np.random.randint(340, 620)
        r_a = np.random.random()
        return np.array([l_i, l_a, r_i, r_a])

    print("Test4")
    L = 101
    l_intercept = np.random.randint(20, 300)
    l_angle = -1.0 * np.random.random()
    r_intercept = np.random.randint(340, 620)
    r_angle = np.random.random()
    actual = np.array([l_intercept, l_angle, r_intercept, r_angle])
    print("Real = ", actual)
    noisy_measurement = noisy(actual, R_actual)
    kalman.kalman_init(actual, R_est)
    t = 1
    kalman_estimates = [noisy_measurement]
    actual_values = [actual]
    measurements = [noisy_measurement]
    for _ in range(L-1):
        #kalman.kalman(measurement, dt)
        if np.random.random() < 0.05:
            noisy_measurement = bad_measurement()
        else:
            noisy_measurement = noisy(actual, R_actual)
        new_results = kalman.kalman(noisy_measurement, dt)
        kalman_estimates.append(new_results)
        actual_values.append(actual)
        measurements.append(noisy_measurement)
        if t % 10 == 0:
            print("t = ", t, new_results)
        t += 1
    return np.array(kalman_estimates), np.array(measurements), np.array(actual_values)

# Random noise along with quadratic drift in real value
def test5():
    def noisy(m, R):
        l_i, l_a, r_i, r_a = m
        l_i = np.random.normal(l_i, R[0,0])
        l_a = np.random.normal(l_a, R[1,1])
        r_i = np.random.normal(r_i, R[2,2])
        r_a = np.random.normal(r_a, R[3,3])
        return np.array([l_i, l_a, r_i, r_a])

    print("Test5")
    L = 101
    def l_intercept(t): # 20 - 50
        return 0.003 * t*t + 20
    def l_angle(t): # -0.5 - -0.8
        return -0.003 * t - 0.5
    def r_intercept(t): # 300 - 380
        return 0.008 * t*t + 300
    def r_angle(t): # 0.2 - -0.1
        return -0.003 * t + 0.2
    def real_measurement(t):
        return np.array([l_intercept(t), l_angle(t), r_intercept(t), r_angle(t)])
    def noisy_measurement(t, R):
        return noisy(real_measurement(t), R)
    real_start = real_measurement(0)
    real_end = real_measurement(L-1)
    print("Real Start = ", real_start)
    print("Real End = ", real_end)
    noisy_start = noisy_measurement(0,R_actual)
    kalman.kalman_init(noisy_start, R_est)
    t = 1
    kalman_estimates = [noisy_start]
    actual_values = [real_measurement(0)]
    measurements = [noisy_start]
    for t in range(1,L):
        #kalman.kalman(measurement, dt)
        measurement = noisy_measurement(t, R_actual)
        new_results = kalman.kalman(measurement, dt)
        actual = real_measurement(t)
        kalman_estimates.append(new_results)
        actual_values.append(actual)
        measurements.append(measurement)
        if t % 10 == 0:
            print("t = ", t, new_results)
        t += 1
    return np.array(kalman_estimates), np.array(measurements), np.array(actual_values)

def graph_test_results(kalman_estimates, measured, actual, attribute = "l_intercept"):
    fig, ax = plt.subplots()
    if attribute == "l_intercept":
        ax.plot(range(len(kalman_estimates)), kalman_estimates[:,0], label="Estimate")
        ax.plot(range(len(kalman_estimates)), measured[:,0], label="Measured")
        ax.plot(range(len(kalman_estimates)), actual[:,0], label="Actual")
    elif attribute == "l_angle":
        ax.plot(range(len(kalman_estimates)), kalman_estimates[:,1], label="Estimate")
        ax.plot(range(len(kalman_estimates)), measured[:,1], label="Measured")
        ax.plot(range(len(kalman_estimates)), actual[:,1], label="Actual")
    elif attribute == "r_intercept":
        ax.plot(range(len(kalman_estimates)), kalman_estimates[:,2], label="Estimate")
        ax.plot(range(len(kalman_estimates)), measured[:,2], label="Measured")
        ax.plot(range(len(kalman_estimates)), actual[:,2], label="Actual")
    elif attribute == "r_angle":
        ax.plot(range(len(kalman_estimates)), kalman_estimates[:,3], label="Estimate")
        ax.plot(range(len(kalman_estimates)), measured[:,3], label="Measured")
        ax.plot(range(len(kalman_estimates)), actual[:,3], label="Actual")
    else:
        return
    ax.legend()
    plt.ylabel("State Value")
    plt.xlabel("Time Step")
    plt.title("Kalman Estimate for " + attribute)
    plt.show()

if __name__ == "__main__":
    np.random.seed(2)
    kalman_estimates, measured, actual_values = test4()
    graph_test_results(kalman_estimates, measured, actual_values, attribute="r_angle")