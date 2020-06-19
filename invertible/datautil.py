class PreprocessedLoader(object):
    def __init__(self, dataloader, module, to_cuda):
        self.dataloader = dataloader
        self.module = module
        self.to_cuda = to_cuda

    def __iter__(self):
        for batch in self.dataloader:
            # First convert x, check if x is a tuple (e.g., multiple transforms
            # applied to same batch)
            x = batch[0]

            if hasattr(x, 'cuda'):
                if self.to_cuda:
                    x = x.cuda()
                x = self.module(x)
            else:
                preproced_xs = []
                for a_x in x:
                    if self.to_cuda:
                        a_x = a_x.cuda()
                    a_x = self.module(a_x)
                    preproced_xs.append(a_x)
                x = tuple(preproced_xs)
            remaining_batch = tuple(batch[1:])
            if self.to_cuda:
                remaining_batch = tuple([a.cuda() for a in remaining_batch])
            yield (x,) + remaining_batch

    def __len__(self):
        return len(self.dataloader)