import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt


class NET(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(200,200,1)

    def forward(self, x):
        return self.conv(x)


model = NET()
optimizer = optim.SGD(params = model.parameters(), lr=1e-4)

#在指定的epoch值，如[10,15，25，30]处对学习率进行衰减，lr = lr * gamma
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,15,25,30], gamma=0.1)

milestones = [30, 100, 120, 140]
warm_up_epochs = 5
gamma = 0.5
lr = 1e-4

# warm_up_with_multistep_lr
#warm_up_with_multistep_lr = lambda epoch: epoch / warm_up_epochs if epoch <= warm_up_epochs else gamma**len([m for m in milestones if m <= epoch])
warm_up_with_multistep_lr = lambda epoch: epoch / int(warm_up_epochs) if epoch <= int(warm_up_epochs) else gamma**len([m for m in milestones if m <= epoch])
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_multistep_lr)
# warm_up_with_multistep_lr = lambda epoch: 0.1 if epoch <= warm_up_epochs else gamma**len([m for m in milestones if m <= epoch])
# scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_multistep_lr)

# plt.figure()
x = list(range(150))
y = []

for epoch in range(150):
    scheduler.step()
    lr = scheduler.get_lr()
    print(epoch, scheduler.get_last_lr())
    y.append(scheduler.get_lr()[0])

plt.plot(x,y)
plt.show()
