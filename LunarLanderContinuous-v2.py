import gym
import numpy as np


def discretecontrol(x, target, tolerance, low, center, high):
    return (
        low if (x < (target - tolerance))
        else high if (x > (target + tolerance))
        else center
    )


env = gym.make("LunarLanderContinuous-v2")

for i in range(10):
    obs = env.reset()
    env.INITIAL_RANDOM = 8000  # Increase difficulty
    points = 0
    for t in range(250):
        env.render()

        x = obs[0]
        y = obs[1]
        dx = obs[2]
        dy = obs[3]
        a = obs[4]
        da = obs[5]
        contact_l = obs[6]
        contact_r = obs[7]

        total_contact = contact_l and contact_r

        x_ctrl = (x + dx) * 1
        a_ctrl = (a + da) * 1
        dy_ctrl = 2.5 * dy
        dy_target = -4 * y - 0.01 + 2 * abs(x_ctrl) * y

        side_act = discretecontrol(
            a_ctrl,
            x_ctrl,
            0,
            max(-1, a_ctrl - x_ctrl - 0.5),
            0,
            min(1, a_ctrl - x_ctrl + 0.5)
        )
        main_act = discretecontrol(
            dy_ctrl,
            dy_target,
            0,
            max(-1, min(1, 8 * (dy_target - dy_ctrl))),
            -1,
            -1,
        )

        action = np.array([main_act, side_act if not total_contact else 0])
        obs, reward, done, info = env.step(action)
        points += reward
        if done:
            print(f"Episode finished after {t + 1} timesteps.")
            break
