import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):

    def __init__(self, embed_size, train_CNN=False) -> None:
        super().__init__()
        self.train_CNN = train_CNN
        self.inception = models.inception_v3(pretrained=True, aux_logits=True)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):

        features = self.inception(images)[0]
        for name, parameter in self.inception.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                parameter.requires_grad = True
            else:
                parameter.requires_grad = self.train_CNN

        return self.dropout(self.relu(features))
    

class DecoderLSTM(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab_size, num_layers) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs


class CNNtoRNN(nn.Module):

    def __init__(self, vocab_size, hidden_size, num_layers, embed_size) -> None:
        super().__init__()
        self.encoder = EncoderCNN(embed_size, train_CNN=False)
        self.decoder = DecoderLSTM(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []

        with torch.no_grad():
            x = self.encoder(image).unsqueeze(0)
            # print(x.shape)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoder.lstm(x, states)
                output = self.decoder.linear(hiddens.squeeze(0))
                # print(output.shape)
                predicted_word = output.unsqueeze(0).argmax(1)
                # print(predicted_word.shape)

                result_caption.append(predicted_word.item())
                x = self.decoder.embed(predicted_word)
                # print(x.shape)
                if vocabulary.itos[predicted_word.item()] == "<EOS>":
                    break
        
        return [vocabulary.itos[idx] for idx in result_caption]
