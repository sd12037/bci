import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class InvertedPendulum(object):
    actions = [0, 1, 2]
    M = 8.
    m = 2.
    l = 0.5
    g = 9.8
    t = 0.1
    t_num = 1000

    def __init__(self, x, theta, noisy=True):
        self.x = x
        self.x_dot = 0.
        self.theta = theta
        self.theta_dot = 0.
        self.u = 0.
        self.noisy = noisy
        self.t_one = self.t / self.t_num

    def do_action(self, a):
        assert a in self.actions, str(a)+" is not in actions"

        if a == 0:
            self.u = -50.
        elif a == 1:
            self.u = 50.
        else:
            self.u = 0
        if self.noisy:
            self.u += np.random.uniform(-10, 10)

        self.update_state()
        return (self.theta, self.theta_dot), self.calc_reward()

    def update_state(self):
        for i in range(self.t_num):
            sintheta = np.sin(self.theta)
            costheta = np.cos(self.theta)
            ml = self.m*self.l
            total_mass = self.M + self.m

            x_acc = (4*self.u/3 + 4*ml*(self.theta_dot**2)*sintheta/3 - self.m*self.g*np.sin(2*self.theta)/2) / (4*total_mass - self.m*(costheta**2))
            thet_aacc = (total_mass*self.g*sintheta - ml*(self.theta_dot**2)*sintheta*costheta - self.u*costheta) / (4*total_mass*self.l/3 - ml*(costheta**2))

            self.x += self.x_dot*self.t_one + x_acc*(self.t_one**2)/2
            self.x_dot += x_acc * self.t_one
            self.theta += self.theta_dot*self.t_one + thet_aacc*(self.t_one**2)/2
            self.theta_dot += thet_aacc * self.t_one

    def calc_reward(self):
        if -np.pi/2 <= self.theta <= np.pi/2:
            return 0
        else:
            return 1

    def get_car_x(self):
        return self.x


def video(x_history, angle_history, l, t):
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
    ax.grid()
    line, = ax.plot([], [], 'o-', lw=2)
    time_text = ax.text(0.02, 0.95, 'aaaaaa', transform=ax.transAxes)

    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    def animate(i):
        line.set_data([x_history[i], x_history[i]+2*l*np.sin(angle_history[i])],
                      [0, 2*l*np.cos(angle_history[i])])
        time_text.set_text('time = {0:.1f}'.format(i*t))
        return line, time_text

    ani = animation.FuncAnimation(fig, animate, frames=range(len(x_history)),
                                  interval=1000*t, blit=False, init_func=init)
    plt.show()

if __name__ == '__main__':
    plant = InvertedPendulum(0, np.pi/12, False)
    angle_history = [np.pi/12]
    x_history = [0.]
    for i in range(200):
        next_s, reward = plant.do_action(2)
        print(next_s, next_s[0]*180/np.pi, plant.get_car_x())
        angle_history.append(next_s[0])
        x_history.append(plant.get_car_x())

    video(x_history, angle_history, plant.l, plant.t)
    plt.plot()
