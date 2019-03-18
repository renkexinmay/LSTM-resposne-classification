SEED = 1234

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize = 'spacy')
LABEL = data.LabelField(dtype = torch.float)

from torchtext import datasets

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

print(f'Number of training examples: {len(train_data)}')
print(f'Number of test examples: {len(test_data)}')

print(vars(test_data.examples[0]))

import random

train_data, valid_data = train_data.split(random_state = random.seed(SEED))

print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')

TEXT.build_vocab(train_data, max_size = 25000, vectors = "glove.6B.100d")
LABEL.build_vocab(train_data)

print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}") # <pad> for blanks in short sentence; <unk> for less frequent words
print(f"Unique tokens in LABELS vocabulary: {len(LABEL.vocab)}")
print(TEXT.vocab.freqs.most_common(20))
print(TEXT.vocab.itos[:10])
print(LABEL.vocab.stoi)

BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE,
    device = device)

import torch.nn as nn

class RNN(nn.Module):
  def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
    super(RNN, self).__init__()
    # input dim = voc size = one-hot vec dim
    
    self.embedding = nn.Embedding(input_dim, embedding_dim)
    self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers = n_layers, bidirectional = bidirectional, dropout = dropout)
    self.fc = nn.Linear(hidden_dim*2, output_dim)
    self.dropout = nn.Dropout(dropout)
    
  def forward(self, x):
    
    #x = [sent len, batch size]
    embedded = self.dropout(self.embedding(x))
    
    #embedded = [sen len, batch size, emb dim]
    
    output, (hidden, cell) = self.rnn(embedded)
    
    #output = [sen len, batch size, hid dim]
    #hidden, cell = [num layers * num directions, batch size, hid dim]
    
    # concat the final forward (hidden[-2,:,:] and backward (hidden[-1, :,:])) hidden layers
    # and apply dropout
    
    hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]),dim = 1))
    
    # hidden = [batch size, hid dim * num directions]
       
    return self.fc(hidden.squeeze(0))

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100 # = pre-trained GloVe vectors loaded
HIDDEN_DIM = 256
OUTPUT_DIM = 1

N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5

model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)


pretrained_embeddings = TEXT.vocab.vectors

print(pretrained_embeddings.shape)

model.embedding.weight.data.copy_(pretrained_embeddings)

import torch.optim as optim
optimizer = optim.Adam(model.parameters())

criterion = nn.BCEWithLogitsLoss() #grad & loss

model = model.to(device)
criterion = criterion.to(device)
# place them on GPU


def binary_accuracy(preds, y):
  rounded_preds = torch.round(torch.sigmoid(preds))
  correct = (rounded_preds == y).float()
  acc = correct.sum()/len(correct)
  return acc

def train(model, iterator, optimizer, criterion):
  epoch_loss = 0
  epoch_acc = 0
  
  model.train()
  
  for batch in iterator:
    
    optimizer.zero_grad()
    
    predictions = model(batch.text).squeeze(1)
    
    loss = criterion(predictions, batch.label)
    
    acc = binary_accuracy(predictions, batch.label)
    
    loss.backward()
    
    optimizer.step()
    
    epoch_loss += loss.item()
    epoch_acc += acc.item()
    
    return epoch_loss/ len(iterator), epoch_acc/ len(iterator)

def evaluate(model, iterator, criterion):
  epoch_loss = 0
  epoch_acc = 0
  
  model.eval()
  
  with torch.no_grad():
    for batch in iterator:
      predictions = model(batch.text).squeeze(1)
      loss = criterion(predictions, batch.label)
      
      acc = binary_accuracy(predictions, batch.label)
      
      epoch_loss += loss.item()
      epoch_acc += acc.item()
      
  return epoch_loss/ len(iterator), epoch_acc/ len(iterator)

N_EPOCHS = 5

for epoch in range(N_EPOCHS):
  train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
  valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
  
  print(f'| Epoch: {epoch+1:02} | Train Loss:{train_loss:.3f}  | Train Accuracy: {train_acc*100:.2f}% | Val.Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.25}% |')

    
