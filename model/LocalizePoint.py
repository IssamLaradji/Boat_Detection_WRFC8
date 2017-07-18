class LocalizePoint(BaseModel):
    def __init__(self, n_channels=1, n_outputs=1, loss_name="bce_localize", which_class=0):
        super(LocalizePoint, self).__init__(problem_type="classification", 
                                         loss_name=loss_name)
        self.which_class = which_class
        self.conv1 = nn.Conv2d(n_channels, 30, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(30, 60, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(60, 1, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(1, 1, kernel_size=5, stride=5, padding=2)
        
        self.pool = nn.MaxPool2d(kernel_size=2)
        #self.fc1 = nn.Linear(20, n_outputs, bias=False)

        self.n_outputs = n_outputs
        self.regression = False

        self.pools = 0
    # def forward(self, x):
    #     x = F.relu(F.max_pool2d(self.conv1(x), 2))
    #     x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    #     x = x.view(-1, 320)
    #     x = F.relu(self.fc1(x))
    #     #x = F.dropout(x, training=self.training)
    #     x = self.fc2(x)
    #     return x

    def _custom_base(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.sigmoid(self.pool(self.conv3(x)))

        return x

    def forward(self, x):
        nch = x.size()[1]

        for i in range(self.pools):
            x = self.pool(x)

        if self.regression:
            if nch == 6 or  nch==2:
                x1 = x[:,:nch/2]
                x2 = x[:,nch/2:]

                x1 = self._custom_base(x1)
                x2 = self._custom_base(x2)

                #x3 = self._custom_base(x1)
                #diff = torch.abs(x1 - x2)
                diff = F.relu(x1 - x2)  
                # return torch.max(0.0, torch.min(diff, 1.0))
                # diff = torch.abs(x1 - x2)
                #return (F.sigmoid(diff) - 0.5) * 2.0

                x = F.tanh(diff) 
            else:

                x = self._custom_base(x)
            
            x = self.conv4(x)

            x = x.view((x.size()[0], -1))            
            x = x.sum(1)

        else:
            if nch == 6 or  nch == 2:
                x1 = x[:, :nch/2]
                x2 = x[:, nch/2:]

                x1 = self._custom_base(x1)
                x2 = self._custom_base(x2)

                #x3 = self._custom_base(x1)
                #diff = torch.abs(x1 - x2)
                diff = F.relu(x1 - x2) 
                # return torch.max(0.0, torch.min(diff, 1.0))
                # diff = torch.abs(x1 - x2)
                #return (F.sigmoid(diff) - 0.5) * 2.0
                return F.tanh(diff) 
            else:
                x = self._custom_base(x)

        # x = F.max_pool2d(x, kernel_size=x.size()[2:])
        # x = x.view(-1, 1)


        return x

    def get_heatmap(self, x, output=1):
        n, _, n_rows, n_cols = x.shape

        x = tu.numpy2var(x)
        x = self.forward(x)
        x = tu.get_numpy(x)

        # images = np.zeros((n, n_rows, n_cols))
        # for i in range(x.shape[0]):
        #     #images[i] = sp.misc.imresize(x[i], size=(n_rows, n_cols), interp="bilinear")
        #     pass
        #     #images[i] = x[i]
        return x