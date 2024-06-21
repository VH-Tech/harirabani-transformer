import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from hypercomplex.layers import PHMLinear
# Hyperparameters
input_size = 784  # 28x28
hidden_size = 128
output_size = 10  # Number of classes (0-9)
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = torchvision.datasets.MNIST(root='data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Simple PHM_MLP Model
class PHM_MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PHM_MLP, self).__init__()
        self.config = lambda: None
        self.config.reduction_factor = 32 #
        self.config.non_linearity = "gelu_new" #
        self.config.phm_c_init = "normal" #
        self.config.hypercomplex_division = 4 #
        self.config.learn_phm = True #
        self.config.hypercomplex_nonlinearity = "glorot-uniform" #
        self.config.shared_phm_rule = False #
        self.config.factorized_phm = False #
        self.config.shared_W_phm = False #*
        self.config.factorized_phm_rule = False #
        self.config.phm_rank = 1 #
        self.config.phm_init_range = 0.0001 #
        self.config.kronecker_prod = False #

        self.fc1 = PHMLinear(in_features=input_size,
                                      out_features=hidden_size,
                                      bias=True,
                                      c_init=self.config.phm_c_init,
                                      phm_dim=self.config.hypercomplex_division,
                                      learn_phm=self.config.learn_phm,
                                      w_init=self.config.hypercomplex_nonlinearity,
                                      shared_phm_rule=self.config.shared_phm_rule,
                                      factorized_phm=self.config.factorized_phm,
                                      shared_W_phm=self.config.shared_W_phm,
                                      factorized_phm_rule=self.config.factorized_phm_rule,
                                      phm_rank=self.config.phm_rank,
                                      phm_init_range=self.config.phm_init_range,
                                      kronecker_prod=self.config.kronecker_prod)
        self.relu = nn.ReLU()
        self.fc2 = PHMLinear(in_features=hidden_size,
                                      out_features=hidden_size,
                                      bias=True,
                                      c_init=self.config.phm_c_init,
                                      phm_dim=self.config.hypercomplex_division,
                                      learn_phm=self.config.learn_phm,
                                      w_init=self.config.hypercomplex_nonlinearity,
                                      shared_phm_rule=self.config.shared_phm_rule,
                                      factorized_phm=self.config.factorized_phm,
                                      shared_W_phm=self.config.shared_W_phm,
                                      factorized_phm_rule=self.config.factorized_phm_rule,
                                      phm_rank=self.config.phm_rank,
                                      phm_init_range=self.config.phm_init_range,
                                      kronecker_prod=self.config.kronecker_prod)
        self.fc3 = PHMLinear(in_features=hidden_size,
                                        out_features=hidden_size,
                                        bias=True,
                                        c_init=self.config.phm_c_init,
                                        phm_dim=self.config.hypercomplex_division,
                                        learn_phm=self.config.learn_phm,
                                        w_init=self.config.hypercomplex_nonlinearity,
                                        shared_phm_rule=self.config.shared_phm_rule,
                                        factorized_phm=self.config.factorized_phm,
                                        shared_W_phm=self.config.shared_W_phm,
                                        factorized_phm_rule=self.config.factorized_phm_rule,
                                        phm_rank=self.config.phm_rank,
                                        phm_init_range=self.config.phm_init_range,
                                        kronecker_prod=self.config.kronecker_prod)
        self.fc4 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out

#Simple MLP Model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out
    
model = MLP(input_size, hidden_size, output_size)
model_phm = PHM_MLP(input_size, hidden_size, output_size)

#print total parameters size of the model
print(sum(p.numel() for p in model.parameters()))
print(sum(p.numel() for p in model_phm.parameters()))
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, input_size)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, input_size)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the 10000 test images: {100 * correct / total:.2f}%')

model_phm = PHM_MLP(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_phm.parameters(), lr=learning_rate)

# Training the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, input_size)
        outputs = model_phm(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

# Test the model
model_phm.eval()

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, input_size)
        outputs = model_phm(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the 10000 test images: {100 * correct / total:.2f}%')
