import torch


class LSTMCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.i2h = torch.nn.Linear(input_size, 4 * hidden_size)
        self.h2h = torch.nn.Linear(hidden_size, 4 * hidden_size)

    def forward(self, x, hidden):
        hx, cx = hidden
        gates = self.i2h(x) + self.h2h(hx)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, cy


class RNNFunctionalizable(torch.nn.Module):
    def __init__(self, hidden_size, input_size=1, output_size=1, num_layers=1, input_batch=False):
        super(RNNFunctionalizable, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_cells = torch.nn.ModuleList([LSTMCell(input_size if i == 0 else hidden_size, hidden_size)
                                               for i in range(num_layers)])
        self.linear = torch.nn.Linear(hidden_size, output_size)
        self.sigmoid = torch.nn.Sigmoid()
        self.input_batch = input_batch

    def forward(self, input_seq, hidden_states=None):
        #print(input_seq)
        device = input_seq.device
        # init hidden states as zeros
        if hidden_states is None:
            h, c = self.get_initial_hidden_states(input_seq.shape[0], device)
        else:
            h, c = hidden_states

        if not self.input_batch:
            input_seq = input_seq.unsqueeze(0)
        output_seq = []
        for i in range(input_seq.shape[1]):
            for l in range(self.num_layers):
                h[l], c[l] = self.lstm_cells[l](h[l-1] if l > 0 else input_seq[:, i], (h[l], c[l]))
            #print(h)
            output_seq.append(h[-1])

        output_seq = torch.stack(output_seq, dim=1)

        output = self.linear(output_seq).squeeze(0)
        return output, (h, c)

    def get_initial_hidden_states(self, batch_size, device):
        h = [torch.zeros(batch_size, self.hidden_size).to(device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_size).to(device) for _ in range(self.num_layers)]
        return h, c


class RNNFast(torch.nn.Module):
    def __init__(self, hidden_size, input_size=1, output_size=1, num_layers=1, input_batch=False):
        super(RNNFast, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, output_size)
        self.sigmoid = torch.nn.Sigmoid()
        self.input_batch = input_batch

    def forward(self, input_seq, hidden_states=None):
        #print(input_seq)
        device = input_seq.device
        # init hidden states as zeros
        if hidden_states is None:
            h, c = self.get_initial_hidden_states(input_seq.shape[0], device)
        else:
            h, c = hidden_states

        # if h and c are lists, stack them
        if isinstance(h, list):
            h = torch.stack(h, dim=0)
            c = torch.stack(c, dim=0)

        if not self.input_batch:
            input_seq = input_seq.unsqueeze(0)

        output_seq, (h, c) = self.lstm(input_seq, (h, c))

        output = self.linear(output_seq)
        return output, (h, c)

    def get_initial_hidden_states(self, batch_size, device):
        h = [torch.zeros(batch_size, self.hidden_size).to(device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_size).to(device) for _ in range(self.num_layers)]
        return h, c


class ConstantSchedule:
    def __init__(self, value):
        self.value = value

    def __call__(self, t):
        return self.value


class PiecewiseLinearSchedule(ConstantSchedule):
    def __init__(self, timesteps: list, values: list):
        super().__init__(values[0])
        self.timesteps = timesteps
        self.values = values

    def __call__(self, t):
        for i, timestep in enumerate(self.timesteps):
            if t < timestep:
                if i == 0:
                    return self.values[0]
                else:
                    # interpolate between the two values
                    v1 = self.values[i - 1]
                    v2 = self.values[i]
                    t1 = self.timesteps[i - 1]
                    t2 = self.timesteps[i]
                    return v1 + (v2 - v1) * (t - t1) / (t2 - t1)
        return self.values[-1]