import torch
from torch import nn
import torch.nn.functional as FN


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.l1 = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),
        )
        self.l2 = nn.Sequential(
            nn.Conv2d(6, 16, 5, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(400, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.fc(x)

        return x


class CNN1(nn.Module):
    def __init__(self):
        super(CNN1, self).__init__()

        # 첫번째층
        # ImgIn shape=(?, 28, 28, 1)
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 두번째층
        # ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 전결합층 7x7x64 inputs -> 10 outputs
        self.fc = nn.Linear(7 * 7 * 64, 10, bias=True)

        # 전결합층 한정으로 가중치 초기화
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x.float())
        out = self.layer2(out)
        out = out.view(out.size(0), -1)  # 전결합층을 위해서 Flatten
        out = self.fc(out)

        return out


class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()

        self.keep_prob = 0.5

        # L1 ImgIn shape=(?, 28, 28, 1)
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # L2 ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # L3 ImgIn shape=(?, 7, 7, 64)
        #    Conv      ->(?, 7, 7, 128)
        #    Pool      ->(?, 4, 4, 128)
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )

        # L4 FC 4x4x128 inputs -> 625 outputs
        self.fc1 = nn.Linear(4 * 4 * 128, 625, bias=True)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.layer4 = nn.Sequential(self.fc1, nn.ReLU(), nn.Dropout(p=1 - self.keep_prob))

        # L5 Final FC 625 inputs -> 10 outputs
        self.fc2 = nn.Linear(625, 10, bias=True)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)  # Flatten them for FC
        out = self.layer4(out)
        out = self.fc2(out)

        return out


class CNN3(nn.Module):
    def __init__(self):
        super(CNN3, self).__init__()

        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = x.float()
        x = FN.relu(self.conv1(x))
        x = FN.max_pool2d(x, 2, 2)
        x = FN.relu(self.conv2(x))
        x = FN.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = FN.relu(self.fc1(x))
        x = self.fc2(x)

        return FN.log_softmax(x, dim=1)


class RNN(nn.Module):
    def __init__(self, device, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNN, self).__init__()

        self.device = device

        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # RNN 구축
        # batch_first=True 설정으로 input/output tensors to be of shape
        # (batch_dim, seq_dim, input_dim)
        # batch_dim = number of samples per batch
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity="relu")

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        device = self.device

        # hidden state 를 zeros로 초기화
        # (layer_dim, batch_size, hidden_dim)
        # h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(device)

        # 폭발/소실 현상을 방지하기 위해 hidden state를 분리해야 합니다
        # 이는 시간(BPTT)을 통해 잘린 역전파의 일부입니다
        out, hn = self.rnn(x, h0.detach())

        # 이전 time step의 hidden state 색인
        # out.size() --> 100, 28, 10
        # out[:, -1, :] --> 100, 10 --> 이전 time step의 hidden states만 필요!
        out = self.fc(out[:, -1, :])
        # out.size() --> 100, 10
        return out


class LSTM(nn.Module):
    def __init__(self, device, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()

        self.device = device

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        device = self.device

        # 초기 hidden 및 cell state 설정
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(device)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(device)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # 입력 및 hidden state를 모델에 전달하고 출력 획득
        out, hidden = self.lstm(x, (h0.detach(), c0.detach()))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # fully connected layer에 맞게 출력단 형상 변경
        out = self.fc(out[:, -1, :])

        return out
