import typing
import numpy as np
class SpatialSearch():
    def __init__(self, loss_func: typing.Callable, radius = 10, center = (0, 0)) -> None:
        self.radius = radius
        self.radiuss = [radius]
        self.points = []
        self.losses = [1]
        self.min_losses = []
        self.center = center
        self.centers = [center]
        self.previous_center_index = 4
        self.loss_func = loss_func
        self.step_size = 0

    def compute_points(self):
        x, y = self.center
        self.points.clear()
        self.points.append((x - self.radius, y + self.radius))
        self.points.append((x, y + self.radius))
        self.points.append((x + self.radius, y + self.radius))
        self.points.append((x - self.radius, y))
        self.points.append((x, y))
        self.points.append((x + self.radius, y))
        self.points.append((x - self.radius, y - self.radius))
        self.points.append((x, y - self.radius))
        self.points.append((x + self.radius, y - self.radius))
        return

    def fit(self, learning_rate = 2, threshold = 10**(-5), limit_step = 1000):
        if learning_rate < 1:
            raise Exception('The learing rate must be not smaller than 1')
        self.compute_points()
        self.step_size = 0
        while(self.step_size <= limit_step):
            self.step_size += 1
            
            # Compute losses
            # This can be improve by only compute the loss
            # for new point, old points do not need. Hope you member late :)
            self.losses.clear()
            for x, y in self.points:
                self.losses.append(self.loss_func(x, y))
            a = np.array(self.losses)
            min_loss = a[np.isfinite(a)].min()
            # print(str(self.step_size) + ': ' + str(min_loss))

            self.min_losses.append(min_loss)
            indices_min = np.where(self.losses == min_loss)[0]
            # Update center and radius
            # == 4 min the center is not changed
            # Can we adapt the changed radius ratio (not fix at 2)?
            if 4 in indices_min:
                self.radius /= learning_rate
            else:
                self.center = self.points[indices_min[0]]
                self.radius *= learning_rate
            # Update neighboor points
            self.compute_points()
            self.centers.append(self.center)
            self.radiuss.append(self.radius)
            if (min_loss < threshold):
                return self.center
        return self.center
