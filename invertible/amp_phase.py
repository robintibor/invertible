import torch as th


class AmplitudePhase(th.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, fixed=None):
        n_chans = x.size()[1]
        assert n_chans % 2 == 0
        x1 = x[:, :n_chans // 2]
        x2 = x[:, n_chans // 2:]
        amps, phases = to_amp_phase(x1, x2)
        # https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant#Example_2:_polar-Cartesian_transformation
        # (here we have cartesian to polar!)
        logdet = -th.sum(th.log(amps), dim=tuple(range(1, len(amps.shape))))
        return th.cat((amps, phases), dim=1), logdet

    def invert(self, x, fixed=None):
        n_chans = x.size()[1]
        assert n_chans % 2 == 0
        amps = x[:, :n_chans // 2]
        phases = x[:, n_chans // 2:]
        x1, x2 = amp_phase_to_x_y(amps, phases)
        logdet = -th.sum(th.log(amps), dim=tuple(range(1, len(amps.shape))))
        return th.cat((x1, x2), dim=1), logdet


def to_amp_phase(x, y):
    amps = th.sqrt((x * x) + (y * y))
    phases = th.atan2(y, x)
    return amps, phases


def amp_phase_to_x_y(amps, phases):
    x, y = th.cos(phases), th.sin(phases)

    x = x * amps
    y = y * amps
    return x, y
