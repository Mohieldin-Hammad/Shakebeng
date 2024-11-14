import torch

class NeuralProbabilisticLanguageModel:
    def __init__(self, block_size, vocab_size, embedding_dim, hidden_dim, device, seed=42):
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.device = device

        g = torch.Generator().manual_seed(seed)
        
        self.C = torch.randn(vocab_size, embedding_dim, generator=g).to(device)

        self.W1 = torch.randn(block_size * embedding_dim, hidden_dim, generator=g).to(device)
        self.b1 = torch.randn(hidden_dim, generator=g).to(device)

        self.W2 = torch.randn(hidden_dim, vocab_size, generator=g).to(device)
        self.b2 = torch.randn(vocab_size, generator=g).to(device)

        self.params = [self.C, self.W1, self.b1, self.W2, self.b2]
        for param in self.params:
            param.requires_grad = True

    def forward(self, x):
        embeddings = self.C[x]
        hidden = torch.tanh(embeddings.view(-1, self.block_size * self.embedding_dim) @ self.W1 + self.b1)
        logits = hidden @ self.W2 + self.b2
        return logits

    def loss(self, logits, y):
        return torch.nn.functional.cross_entropy(logits, y)

    def get_batches(self, x, y, batch_size):
        n_batches = x.shape[0] // batch_size
        indices = torch.randperm(x.shape[0])
        for i in range(n_batches):
            batch_indices = indices[i * batch_size: (i + 1) * batch_size]
            yield x[batch_indices], y[batch_indices]

        if len(x) % batch_size != 0:
            batch_indices = indices[n_batches * batch_size:]
            yield x[batch_indices], y[batch_indices]
                



    def fit(self, x, y, num_epochs=200000, batch_size=128, learning_rate=0.1):
        for epoch in range(num_epochs):
            total_loss = 0
            n_batches = 0
            for x_batch, y_batch in self.get_batches(x, y, batch_size):            
                # ix = torch.randint(0, x.shape[0], (batch_size,))
                logits = self.forward(x_batch)
                loss  = self.loss(logits, y_batch)
                
                for param in self.params:
                    param.grad = None

                loss.backward()
                for param in self.params:
                    param.data += -learning_rate * param.grad

                total_loss += loss.item()
                n_batches += 1
                
            avg_loss = total_loss / n_batches
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')

    def evaluate(self, x, y, batch_size=128):
        total_loss = 0
        n_batches = 0
        with torch.no_grad():
            for x_batch, y_batch in self.get_batches(x, y, batch_size):
                logits = self.forward(x_batch)
                loss = self.loss(logits, y_batch)
                total_loss += loss.item()
                n_batches += 1

        avg_loss = total_loss / n_batches
        return avg_loss

    def generate(self, context_idx, n_words, word2idx, end_token='<END>'):
        if len(context_idx) != self.block_size:
            raise ValueError(f"Context should have {self.block_size} words")
        out = []
        for i in range(n_words):
            context_tensor = torch.tensor(context_idx, dtype=torch.long).to(self.device)
            logits = self.forward(context_tensor.unsqueeze(0))
            probs = torch.softmax(logits, dim=1)
            next_word_idx = torch.multinomial(probs, num_samples=1).item()
            context_idx = context_idx[1:] + [next_word_idx]
            out.append(next_word_idx)
            if next_word_idx == word2idx[end_token]:
                break
        return out