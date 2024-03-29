import torch as th
import numpy as np


# from https://github.com/y0ast/Glow-PyTorch
def squeeze2d(input, factor):
    if factor == 1:
        return input

    B, C, H, W = input.size()

    assert H % factor == 0 and W % factor == 0, "H or W modulo factor is not 0"

    x = input.view(B, C, H // factor, factor, W // factor, factor)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(B, C * factor * factor, H // factor, W // factor)

    return x

# from https://github.com/y0ast/Glow-PyTorch
def unsqueeze2d(input, factor):
    if factor == 1:
        return input

    factor2 = factor ** 2

    B, C, H, W = input.size()

    assert C % (factor2) == 0, "C module factor squared is not 0"

    x = input.view(B, C // factor2, factor, factor, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.view(B, C // (factor2), H * factor, W * factor)

    return x


class SubsampleSplitter(th.nn.Module):
    def __init__(self, stride, chunk_chans_first=True, checkerboard=False,
                 cat_at_end=True, via_reshape=False):
        super().__init__()
        if not hasattr(stride, '__len__'):
            stride = (stride, stride)
        elif len(stride) == 1: # HACK for now, just make it into 2d splitting
            assert (not checkerboard) and (not via_reshape)
            stride = (stride[0], 1)
        self.stride = stride
        self.chunk_chans_first = chunk_chans_first
        self.checkerboard = checkerboard
        self.cat_at_end = cat_at_end
        self.via_reshape = via_reshape
        if checkerboard:
            assert stride[0] == 2
            assert stride[1] == 2
        if self.via_reshape:
            assert stride[0] == stride[1]

    def forward(self, x, fixed=None):
        # Chunk chans first to ensure that each of the two streams in the
        # reversible network will see a subsampled version of the whole input
        # (in case the preceding blocks would not alter the input)
        # and not one half of the input
        was_1d = len(x.shape) == 3
        if was_1d:
            # HACK for now, just make it into 2d splitting
            x = x.unsqueeze(-1)
            self.stride = (self.stride[0], 1)

        if self.via_reshape:
            y = squeeze2d(x, self.stride[0])
            assert not was_1d
            return y, 0
        else:
            new_x = []
            if self.chunk_chans_first:
                xs = th.chunk(x, 2, dim=1)
            else:
                xs = [x]
            for one_x in xs:
                if not self.checkerboard:
                    for i_stride in range(self.stride[0]):
                        for j_stride in range(self.stride[1]):
                            new_x.append(
                                one_x[:, :, i_stride::self.stride[0],
                                j_stride::self.stride[1]])
                else:
                    new_x.append(one_x[:,:,0::2,0::2])
                    new_x.append(one_x[:,:,1::2,1::2])
                    new_x.append(one_x[:,:,0::2,1::2])
                    new_x.append(one_x[:,:,1::2,0::2])
            if was_1d:
                assert all([n.shape[-1] == 1 for n in new_x]), (
                    f"Shapes should be 1 at end {[n.shape for n in new_x]}")
                new_x = [n.squeeze(-1) for n in new_x]

            if self.cat_at_end:
                new_x = th.cat(new_x, dim=1)
            return new_x, 0


    def invert(self, features, fixed=None):
        if not self.cat_at_end:
            # just cat now then... then invert can be a single logic
            features = th.cat(features, dim=1)
        was_1d = len(features.shape) == 3
        if was_1d:
            # HACK for now, just make it into 2d splitting
            features = features.unsqueeze(-1)
            self.stride = (self.stride[0], 1)
        if self.via_reshape:
            assert not was_1d
            x = unsqueeze2d(features, self.stride[0])
            return x, 0
        else:
            # after splitting the input into two along channel dimension if possible
            # for i_stride in range(self.stride):
            #    for j_stride in range(self.stride):
            #        new_x.append(one_x[:,:,i_stride::self.stride, j_stride::self.stride])
            n_all_chans_before = features.size()[1] // (
                    self.stride[0] * self.stride[1])

            # if there was only one chan before, chunk had no effect
            if self.chunk_chans_first and (n_all_chans_before > 1):
                chan_features = th.chunk(features, 2, dim=1)
            else:
                chan_features = [features]
            all_previous_features = []
            for one_chan_features in chan_features:
                n_examples = one_chan_features.size()[0]
                n_chans = one_chan_features.size()[1] // (
                        self.stride[0] * self.stride[1])
                n_0 = one_chan_features.size()[2] * self.stride[0]
                n_1 = one_chan_features.size()[3] * self.stride[1]

                previous_features = th.zeros(
                    n_examples,
                    n_chans,
                    n_0,
                    n_1,
                    device=features[0].device)

                n_chans_before = previous_features.size()[1]
                cur_chan = 0
                if not self.checkerboard:
                    for i_stride in range(self.stride[0]):
                        for j_stride in range(self.stride[1]):
                            previous_features[:, :, i_stride::self.stride[0],
                                j_stride::self.stride[1]] = (
                                    one_chan_features[:,
                                    cur_chan * n_chans_before:
                                    cur_chan * n_chans_before + n_chans_before])
                            cur_chan += 1
                else:
                    # Manually go through 4 checkerboard positions
                    assert self.stride[0] == 2
                    assert self.stride[1] == 2
                    previous_features[:, :, 0::2, 0::2] = (
                        one_chan_features[:,
                        0 * n_chans_before:0 * n_chans_before + n_chans_before])
                    previous_features[:, :, 1::2, 1::2] = (
                        one_chan_features[:,
                        1 * n_chans_before:1 * n_chans_before + n_chans_before])
                    previous_features[:, :, 0::2, 1::2] = (
                        one_chan_features[:,
                        2 * n_chans_before:2 * n_chans_before + n_chans_before])
                    previous_features[:, :, 1::2, 0::2] = (
                        one_chan_features[:,
                        3 * n_chans_before:3 * n_chans_before + n_chans_before])
                all_previous_features.append(previous_features)
            features = th.cat(all_previous_features, dim=1)
            if was_1d:
                assert features.shape[-1] == 1
                features = features.squeeze(-1)
            return features, 0

    def __repr__(self):
        return ("SubsampleSplitter(stride={:s}, chunk_chans_first={:s}, "
               "checkerboard={:s})").format(str(self.stride),
                                           str(self.chunk_chans_first),
                                           str(self.checkerboard))
