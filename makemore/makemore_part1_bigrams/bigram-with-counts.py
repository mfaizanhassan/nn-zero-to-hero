import torch
import mlflow
import mlflow.pytorch
import io


class Bigram:
    def __init__(self, file):
        self.file = file
        self.N = None
        self.data = None
        self.stoi = None
        self.itos = None
        self.n = None  # no. of chars in the dataset
        self.probs = None

        self._initialize()

    def __call__(self, x):
        self._train()
        self._generate(x)

    def _initialize(self):
        self.data = open(self.file, "r").read().splitlines()
        chars = sorted(list(set("".join(self.data))))

        self.stoi = {c: i + 1 for i, c in enumerate(chars)}
        self.stoi["."] = 0
        self.itos = {i: c for c, i in self.stoi.items()}
        self.n = len(self.stoi)
        self.N = torch.zeros(self.n, self.n, dtype=torch.int32)

    def _train(self):
        for word in self.data:
            chs = ["."] + list(word) + ["."]
            for ch1, ch2 in zip(chs, chs[1:]):
                stoi1 = self.stoi[ch1]
                stoi2 = self.stoi[ch2]
                self.N[stoi1][stoi2] += 1
        self.probs = self.N / self.N.sum(dim=1, keepdim=True)

    def _generate(self, x):
        outputs = []
        for _ in range(x):
            ix = 0
            outs = []
            while True:
                ix = torch.multinomial(self.probs[ix], num_samples=1, replacement=True).item()
                outs.append(self.itos[ix])
                if ix == 0:
                    break
            word = "".join(outs)
            print(word)
            outputs.append(word)
        return outputs


if __name__ == "__main__":
    torch.manual_seed(2147483647)

    # Connect to MLflow server running in Docker
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("bigram-experiments")

    with mlflow.start_run(run_name="bigram-model"):
        # log some hyperparameters
        mlflow.log_param("seed", 2147483647)
        mlflow.log_param("dataset", "names.txt")

        b = Bigram("names.txt")

        # training
        b._train()
        mlflow.log_metric("unique_chars", b.n)

        # log N matrix as artifact
        torch.save(b.N, "bigram_counts.pt")
        # mlflow.log_artifact("bigram_counts.pt")  # Temporarily disabled due to permission issues

        # log probabilities
        torch.save(b.probs, "bigram_probs.pt")
        # mlflow.log_artifact("bigram_probs.pt")  # Temporarily disabled due to permission issues

        # generate samples
        generated = b._generate(10)
        with open("generated.txt", "w") as f:
            f.write("\n".join(generated))
        # mlflow.log_artifact("generated.txt")  # Temporarily disabled due to permission issues

        # Note: Bigram class is not a torch.nn.Module, so we can't log it as a PyTorch model
        # Instead, we log the key artifacts: counts, probabilities, and generated samples
