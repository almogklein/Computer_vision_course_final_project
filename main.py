import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import torchattacks
from torchattacks import PGD, FGSM, DeepFool, AutoAttack, FFGSM, UPGD
from models import CNN

'''''
This code show the accuracy of a CNN model with/without adve_img train


'''''

print("Testing______________________________________________________________")
print("")

batch_size = 120

mnist_train = dsets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
mnist_test = dsets.MNIST(root='./data/', train=False, transform=transforms.ToTensor(), download=True)

train_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=mnist_test, batch_size=batch_size, shuffle=False)

print("mnist_data_loaded______________________________________________________________")

'''''
batch_size = 10
cifar10_train = dsets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=True)
cifar10_test = dsets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor(), download=True)

train_loader = torch.utils.data.DataLoader(cifar10_train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(cifar10_test, batch_size=batch_size, shuffle=False)
print("cifar10_data_loaded______________________________________________________________")

#OnePixel(model, pixels=10, inf_batch=120),
'''''


model = CNN().cpu()
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

atks = [
    FFGSM(model, eps=0.1, alpha=0.1),
    FGSM(model, eps=0.1),
    PGD(model, eps=0.1)
    ]


print("attacks_loaded______________________________________________________________")
print("")
print("")
print("")
print("____________________________start clean train_________________________________")
num_epochs = 1
for epoch in range(num_epochs):
    total_batch = len(mnist_train) // batch_size

    for i, (batch_images, batch_labels) in enumerate(train_loader):
        start_time1 = time.time()

        X = batch_images.cpu()
        Y = batch_labels.cpu()
        pre = model(X)
        cost = loss(pre, Y)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()


model.eval()
correct = 0
total = 0

for images, labels in test_loader:
    images = images.cpu()
    outputs = model(images)

    _, predicted = torch.max(outputs.data, 1)

    total += labels.size(0)
    correct += (predicted == labels.cpu()).sum()

print("")
print("______________________________CNN clean accuracy______________________________________________")
print('CNN accuracy clean learn: %.2f %%' % (100 * float(correct) / total))

print("")
print("______________________________CNN attacked accuracy_______________________")

for atk in atks:
    model.eval()
    correct = 0
    total = 0

    for images, labels in test_loader:
        images = atk(images, labels).cpu()
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels.cpu()).sum()
    print("attack img :" + type(atk).__name__)
    print('CNN accuracy clean learn : %.2f %%' % (100 * float(correct) / total))

print("")
print("")
print("____________________________start 33/67 train _____________________________")

for atk in atks:
    model = CNN().cpu()
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("")
    print("----------------------------new------------------------------------------")
    print("attack :" + type(atk).__name__)
    start_time3 = time.time()

    for epoch in range(num_epochs):
        start_time = time.time()
       #print("epoch %d_____________________________________________________________" % (epoch + 1))
        total_batch = len(mnist_train) // batch_size

        for i, (batch_images, batch_labels) in enumerate(train_loader):
            start_time1 = time.time()
            if i % 3 == 0:
                X = atk(batch_images, batch_labels).cpu()
                Y = batch_labels.cpu()
            else:
                X = batch_images.cpu()
                Y = batch_labels.cpu()

            pre = model(X)
            cost = loss(pre, Y)
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            if (i + 1) % 50 == 0:
               #print("iteration = %d" % (i + 1))
                end_time1 = time.time()
                elapsed_time1 = end_time1 - start_time1
               #print("Finished 50 (%.4f sec) iterations in: " % (float(elapsed_time1) * 500) + "%.4f" % (
               #      float(elapsed_time1) * 10) + "  sec per iteration")
               #print('Epoch [%d/%d], lter [%d/%d], Loss: %.4f' % (
               #    epoch + 1, num_epochs, i + 1, total_batch, cost.item()))

        end_time = time.time()
        elapsed_time = end_time - start_time
       #print("Finished epoch %d in: " % (epoch + 1) + str(float(elapsed_time) / 60) + " minuets")

    end_time3 = time.time()
    elapsed_time3 = end_time3 - start_time3
   #print("Finished  %d epochs in: " % num_epochs + str(float(elapsed_time3) / 60) + " minuets")

    model.eval()
    correct = 0
    total = 0

    for images, labels in test_loader:
        images = atk(images, labels).cpu()
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels.cpu()).sum()
    print("")
    print("______________________________CNN attacked accuracy train1/3_______________________________________")
    print("___________________________________" + type(atk).__name__ + "________________________________")
    print('CNN accuracy 1/3 learn: %.2f %%' % (100 * float(correct) / total))
    print("____________________________________________________________________________")
    print("")
    print("")
    print("______________________________Testing CNN1/3 with new attacks___________________________________")

    for atkk in atks:
        if atkk is not atk:
            model.eval()
            correct = 0
            total = 0

            for images, labels in test_loader:
                images = atkk(images, labels).cpu()
                outputs = model(images)

                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels.cpu()).sum()

            print('CNN accuracy 1/3 learn ', type(atkk).__name__, ' : %.2f %%' % (100 * float(correct) / total))

print("")
print("")
print("____________________________start 50/50 train______________________________")

for atk in atks:
    model = CNN().cpu()
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("")
    print("----------------------------new------------------------------------------")
    print("attack :" + type(atk).__name__)
    start_time3 = time.time()

    for epoch in range(num_epochs):
        start_time = time.time()
       #print("epoch %d_____________________________________________________________" % (epoch + 1))
        total_batch = len(mnist_train) // batch_size

        for i, (batch_images, batch_labels) in enumerate(train_loader):
            start_time1 = time.time()
            if i % 2 == 0:
                X = atk(batch_images, batch_labels).cpu()
                Y = batch_labels.cpu()
            else:
                X = batch_images.cpu()
                Y = batch_labels.cpu()

            pre = model(X)
            cost = loss(pre, Y)
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            if (i + 1) % 50 == 0:
               #print("iteration = %d" % (i + 1))
                end_time1 = time.time()
                elapsed_time1 = end_time1 - start_time1
               #print("Finished 50 (%.4f sec) iterations in: " % (float(elapsed_time1) * 500) + "%.4f" % (
               #      float(elapsed_time1) * 10) + "  sec per iteration")
               #print('Epoch [%d/%d], lter [%d/%d], Loss: %.4f' % (
               #    epoch + 1, num_epochs, i + 1, total_batch, cost.item()))

        end_time = time.time()
        elapsed_time = end_time - start_time
       #print("Finished epoch %d in: " % (epoch + 1) + str(float(elapsed_time) / 60) + " minuets")

    end_time3 = time.time()
    elapsed_time3 = end_time3 - start_time3
   #print("Finished  %d epochs in: " % num_epochs + str(float(elapsed_time3) / 60) + " minuets")

    model.eval()
    correct = 0
    total = 0

    for images, labels in test_loader:
        images = images.cpu()
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels.cpu()).sum()
    print("")
    print("______________________________CNN clean accuracy train5050____________________________________")
    print('CNN accuracy 5050 learn: %.2f %%' % (100 * float(correct) / total))

    model.eval()
    correct = 0
    total = 0

    for images, labels in test_loader:
        images = atk(images, labels).cpu()
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels.cpu()).sum()
    print("")
    print("______________________________CNN attacked accuracy train5050_______________________________________")
    print("___________________________________" + type(atk).__name__ + "________________________________")
    print('CNN accuracy 5050 learn: %.2f %%' % (100 * float(correct) / total))
    print("____________________________________________________________________________")
    print("")
    print("")
    print("______________________________Testing CNN5050 with new attacks___________________________________")

    for atkk in atks:
        if atkk is not atk:
            model.eval()
            correct = 0
            total = 0

            for images, labels in test_loader:
                images = atkk(images, labels).cpu()
                outputs = model(images)

                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels.cpu()).sum()

            print('CNN accuracy 5050 learn ', type(atkk).__name__, ' : %.2f %%' % (100 * float(correct) / total))

print("")
print("")
print("____________________________start full advatk train________________________")

for atk in atks:
    model = CNN().cpu()
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("")
    print("----------------------------new------------------------------------------")
    print("attack :" + type(atk).__name__)
    start_time3 = time.time()

    for epoch in range(num_epochs):
        start_time = time.time()
       #print("epoch %d_____________________________________________________________" % (epoch + 1))
        total_batch = len(mnist_train) // batch_size

        for i, (batch_images, batch_labels) in enumerate(train_loader):
            start_time1 = time.time()
            X = atk(batch_images, batch_labels).cpu()
            Y = batch_labels.cpu()

            pre = model(X)
            cost = loss(pre, Y)
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            if (i + 1) % 50 == 0:
               #print("iteration = %d" % (i + 1))
                end_time1 = time.time()
                elapsed_time1 = end_time1 - start_time1
               #print("Finished 50 (%.4f sec) iterations in: " % (float(elapsed_time1) * 500) + "%.4f" % (
               #      float(elapsed_time1) * 10) + "  sec per iteration")
               #print('Epoch [%d/%d], lter [%d/%d], Loss: %.4f' % (
               #    epoch + 1, num_epochs, i + 1, total_batch, cost.item()))

        end_time = time.time()
        elapsed_time = end_time - start_time
       #print("Finished epoch %d in: " % (epoch + 1) + str(float(elapsed_time) / 60) + " minuets")

    end_time3 = time.time()
    elapsed_time3 = end_time3 - start_time3
   #print("Finished  %d epochs in: " % num_epochs + str(float(elapsed_time3) / 60) + " minuets")

    model.eval()
    correct = 0
    total = 0

    for images, labels in test_loader:
        images = images.cpu()
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels.cpu()).sum()
    print("")
    print("______________________________CNN clean accuracy full advatk train______________________________")
    print('CNN accuracy full advatk learn: %.2f %%' % (100 * float(correct) / total))

    model.eval()
    correct = 0
    total = 0

    for images, labels in test_loader:
        images = atk(images, labels).cpu()
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels.cpu()).sum()
    print("")
    print("______________________________CNN attacked accuracy full advatk train______________________________________")
    print("___________________________________" + type(atk).__name__ + "________________________________")
    print('CNN accuracy full advatk learn: %.2f %%' % (100 * float(correct) / total))
    print("____________________________________________________________________________")
    print("")
    print("")
    print("______________________________Testing CNN full advatk with new attacks______________________________")
    print("")
    print("______________________________CNN new attacked accuracy full advatk train____________________________")

    for atkkk in atks:
        if atkkk is not atk:
            model.eval()
            correct = 0
            total = 0

            for images, labels in test_loader:
                images = atkkk(images, labels).cpu()
                outputs = model(images)

                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels.cpu()).sum()

            print('CNN accuracy full advatk learn ', type(atkkk).__name__,
                  ' : %.2f %%' % (100 * float(correct) / total))




