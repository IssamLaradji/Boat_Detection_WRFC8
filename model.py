from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch 

def numpy2var(X):
    Xvar = Variable(torch.FloatTensor(X))
    return Xvar
    
def var2numpy(matrix):
    if torch.cuda.is_available():
        return matrix.data.cpu().numpy()
    else:
        return matrix.data.numpy()

class BoatDetector(nn.Module):
    def __init__(self, n_channels=1, n_outputs=1):
        super(BoatDetector, self).__init__()

        self.conv1 = nn.Conv2d(n_channels, 30, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(30, 30, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(30, 30, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(30, 1, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(1, 1, kernel_size=5, stride=5, padding=2)
        
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))

        x = F.relu(self.pool(self.conv3(x)))
        x = F.sigmoid(self.conv4(x))

        return x

    def predict(self, X):
        self.eval()

        Xvar = numpy2var(X)
        y_pred = self(Xvar)

        return var2numpy(y_pred) 