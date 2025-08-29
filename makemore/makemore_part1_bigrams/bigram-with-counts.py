import torch

class Bigram():
    def __init__(self, file):
        self.file   = file
        self.N      = None
        self.data   = None
        self.stoi   = None
        self.itos   = None
        self.n      = None           # no. of chars in the dataset
        self.probs  = None

        self._intialize()
    
    def __call__(self, x):
        self._train()
        self._generate(x)

    def _intialize(self):
        self.data = open(self.file, 'r').read().splitlines()
        chars = sorted(list(set("".join(self.data))))

        self.stoi = {c: i+1 for i, c in enumerate(chars)}
        self.stoi["."] = 0
        self.itos = {i: c for c, i in self.stoi.items()}
        self.n = len(self.stoi)
        self.N = torch.zeros(self.n, self.n, dtype=torch.int32)

    def _train(self):
        for word in self.data:
            chs = ["."] + list(word) + ["."]
            # print(word)
            for ch1, ch2 in zip(chs, chs[1:]):
                stoi1 = self.stoi[ch1]
                stoi2 = self.stoi[ch2]
                self.N[stoi1][stoi2] += 1
        self.probs = self.N / self.N.sum(dim=1, keepdim=True)

    def _generate(self, x):
        for _ in range(x):
            ix = 0
            outs = []
            while True:
                ix = torch.multinomial(self.probs[ix], num_samples=1, replacement=True).item()
                outs.append(self.itos[ix])
                if ix == 0:
                    break
            print("".join(outs))


if __name__ == "__main__":
    # torch.manual_seed(2147483647)
    b = Bigram('names.txt')
    b(100)